"""subagent 工具：主对话派出「分身」去独立干活的入口。

子代理（subagent）= 主对话临时派出去帮忙的独立 AI 实例，有自己的上下文和工具，
干完活只把结果带回来，不共享主对话的聊天记录。

这个文件在项目里的位置：注册在工具注册表（tools/registry.py）里，属于 core 工具集；
真正干活的子代理实例由 agent/__init__.py 的 AIAgent 类创建，本文件负责「怎么派、
怎么等结果、怎么中断」这套调度逻辑。

两种角色（role 参数）：
  - leaf（默认）：干活的执行者，不能再往下派人
    有 terminal/read_file 等工具
    调不了 subagent/clarify/memory/send_message（靠工具集隔离实现）

  - orchestrator：协调者，可以继续派自己的子代理
    能调用 subagent 往下派生
    但有层数上限 max_spawn_depth（默认 2 层，防止套娃无限派生）

三种运行模式：
  - 同步（默认）：主对话原地等子代理干完
  - 异步（background=True）：立刻返回一个任务 ID，结果稍后通过队列送回
  - 批量（tasks=[...]）：多个子代理真并行同时跑
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异步委托结果队列（一个线程安全的「收发件箱」）
# ---------------------------------------------------------------------------

class DelegationCompletionQueue:
    """后台子代理干完活后放结果的信箱。

    为什么需要：异步子代理在后台线程跑，主对话不能原地等它，得有个地方
    存放「XX 已完成，结果是……」的通知。后台线程往里放（push），主对话
    每轮循环开始时来取（drain）。用锁保证多线程放/取不打架。
    """

    def __init__(self):
        self._queue: List[dict] = []
        self._lock = threading.Lock()
        # idle wake（后台唤醒）：结果入队后要敲一下的回调（CLI 注册；None=未启用）
        self._wake_callback = None

    def set_wake_callback(self, fn) -> None:
        """注册 idle wake 回调：push 后被敲一下（不传参）。

        给 CLI 实现"主对话空闲时异步子代理完成 → 自动跑一轮处理结果"用。
        回调在子代理后台线程里执行，必须便宜、非阻塞（调用处已兜底）。

        参数：
        - fn：无参回调；传 None 等于注销。

        返回：无。
        """
        self._wake_callback = fn

    def push(self, result: dict) -> None:
        """子代理完成时调用：把结果（成功/失败 + 内容）放进信箱。"""
        with self._lock:
            self._queue.append(result)
        if self._wake_callback is not None:
            try:
                self._wake_callback()
            except Exception as e:
                logger.debug("delegation wake 回调失败（fail-open）: %s", e)

    def drain(self) -> List[dict]:
        """主对话取件：一次性拿走信箱里所有结果并清空，下次从空箱开始。"""
        with self._lock:
            results = list(self._queue)
            self._queue.clear()
            return results

    def has_pending(self) -> bool:
        # 信箱里还有没取的结果吗（给主对话判断要不要取件用）
        with self._lock:
            return len(self._queue) > 0


# 全局兜底信箱（只在没有 agent_ref 时才用它——
# 即直接调函数或跑测试的场景。生产环境的 dispatch 一定带 agent_ref，
# 结果会定向送进发起者自己的实例信箱，避免「agent A 的后台子代理结果
# 被 agent B 取走」这种串箱事故）
_delegation_queue = DelegationCompletionQueue()


# === async 子代理花名册（让 subagent_kill 工具能找到正在跑的任务）===
# key = delegation_id（_delegate_async 生成的 del_xxx），
# value = {"thread": 线程, "cancel_event": 取消信号}
# 注意：只在单个进程内有效，不跨进程；多个 AIAgent 实例共用这一份花名册
# （这样进程内所有真并发跑着的后台子代理都能被 kill 工具定位到）
_async_tasks: Dict[str, dict] = {}

# === 同步批量子代理登记表 ===
# Ctrl+C 的 KeyboardInterrupt 只投递给主线程，而批量任务跑在线程池线程里
# 收不到信号——_delegate_batch 内部的 KeyboardInterrupt 处理对这个场景
# 是死路。这里把每批的取消信号登记成全局表，让 CLI 的中断/退出路径能
# 按下所有还在跑的子代理的取消旗。
# value = {"cancel_events": [threading.Event, ...], "executor": 线程池}
_active_batches: list = []


def cancel_all_subagents(reason: str = "shutdown") -> int:
    """按下所有正在跑的子代理（同步批量 + 异步花名册）的取消旗。

    用在 CLI 的 Ctrl+C 中断和退出清理：子代理收到旗子后会在下一次调
    LLM 前优雅退出（带走部分结果），进程不必等它们把任务跑完。

    参数：
      - reason：取消原因（写日志用）

    返回：按下的取消旗数量。
    """
    count = 0
    # 同步批量子代理
    for batch in list(_active_batches):
        for ev in batch.get("cancel_events", []):
            try:
                ev.set()
                count += 1
            except Exception:
                pass
        ex = batch.get("executor")
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    # 异步子代理花名册（subagent_kill 管的那个）
    for info in _async_tasks.values():
        ev = info.get("cancel_event")
        try:
            if ev is not None and not ev.is_set():
                ev.set()
                count += 1
        except Exception:
            pass
    if count:
        logger.info("已取消 %d 个活跃子代理（%s）", count, reason)
    return count


def get_delegation_queue() -> DelegationCompletionQueue:
    """拿全局兜底信箱（测试或直接调用路径用）。"""
    return _delegation_queue


def _resolve_delegation_queue(kwargs) -> DelegationCompletionQueue:
    """决定结果送进哪个信箱。

    有 agent_ref（生产 dispatch 必有）→ 用该 agent 自己的实例信箱；
    没有 → 退回全局兜底信箱（只有直接调函数/测试会走这条路）。
    """
    return getattr(kwargs.get("agent_ref"), "_delegation_queue", None) \
        or _delegation_queue


def inline_mcp_spawn_allowed(agent_def, server_name: str, server_cfg: dict) -> bool:
    """spawn 子代理时，检查「项目来源」的内联 MCP 服务器有没有获得首次连接审批。

    子代理定义（.md 文件）里可以内嵌 MCP 外部工具服务器。来自项目目录
    （source="project"，即从 <cwd>/.omnimate/agents/ 扫出来的）的定义算不可信
    来源——必须用户批准过首次连接才允许连，否则跳过（fail-closed，出错宁可
    不放行）。用户级/CLI 注入的定义默认按 user 信任源处理，不拦。

    参数：
      - agent_def：子代理定义对象（agent/agent_defs.py 的 AgentDefinition）
      - server_name：要连的 MCP 服务器名字
      - server_cfg：该服务器的配置 dict（命令、参数等）

    返回：True=允许连；False=没获批/校验出错，调用方应跳过连接并告警。
    审批 key 的算法与 tools/mcp_tool.py 启动审批处同源：
    工作目录 resolve().lower() + agent-mcp::<服务器名> + 配置指纹。
    """

    if getattr(agent_def, "source", "user") != "project":
        return True
    try:
        from agent.settings import is_project_mcp_approved, mcp_approval_key
        from agent.workspace_context import get_workspace_cwd
        _pk = str(Path(get_workspace_cwd()).resolve()).lower()
        return is_project_mcp_approved(
            mcp_approval_key(_pk, f"agent-mcp::{server_name}", server_cfg))
    except Exception as e:
        logger.warning("内联 MCP 审批校验异常（fail-closed 拒绝）: %s", e)
        return False


# ---------------------------------------------------------------------------
# subagent 工具
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "subagent",
    "description": (
        "派生子代理（subagent）执行独立任务。"
        "子代理有独立的上下文和工具，只把结果带回主代理。\n\n"
        "主入口用 prompt 描述任务；subagent_type 选子代理类型。\n\n"
        "两种模式：\n"
        "- 同步（默认）：等待子代理完成后继续\n"
        "- 异步（background=True）：立即继续，结果稍后送达\n\n"
        "**并行规则（重要）**：需要同时派多个子代理（如并行探索多个模块）时，"
        "必须用 tasks=[...] 一次批量调用（内部真并行）。"
        "禁止发多个独立的 subagent 调用——独立调用是串行执行的，会逐个等待，浪费大量时间。\n\n"
        "角色：\n"
        "- leaf（默认）：执行者，不能再委托\n"
        "- orchestrator：可继续派生（受 max_spawn_depth 限制）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "子代理任务描述（必填之一）",
            },
            "subagent_type": {
                "type": "string",
                "default": "general-purpose",
                "description": (
                    "子代理类型：general-purpose=通用（minimal 工具集）；"
                    "custom=显式 enabled_toolsets；"
                    "或自定义子代理名（扫描 ~/.OmniMate/agents/*.md 和 ./.omnimate/agents/*.md 的 name 字段）。"
                    "自定义名时按定义的 model/tools/permissionMode/isolation/maxTurns 配置子代理。"
                    "或内置名 explore（只读研究）/ plan（只出计划）。"
                ),
            },
            "goal": {
                "type": "string",
                "description": "子代理的目标（兼容旧字段，等同 prompt）",
            },
            "context": {
                "type": "string",
                "description": "给子代理的额外上下文",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "批量任务列表（并行执行）。要并行多个任务就用这个字段，不要多次调用 subagent",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "是否后台运行（默认 false）。"
                    "**后台模式受限：(1) 工具白名单管控（禁 bg_start/team_spawn/"
                    "team_shutdown/task_complete/subagent/idle 等有全局副作用的工具）；"
                    "(2) 所有需审批的操作自动拒（用户不在场，async 子代理不能弹审批 UI）**"
                ),
                "default": False,
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "default": "leaf",
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "子代理启用的工具集（默认继承父代理）",
            },
            "summary_only": {
                "type": "boolean",
                "description": "是否只返回摘要（默认 True）。超长结果用 LLM 压缩成摘要，节省父代理 context。",
                "default": True,
            },
            "summary_len": {
                "type": "integer",
                "description": "摘要字数上限（100-2000，默认 300）。深度调研/长任务的子代理结果建议 800-1500，避免关键发现被压丢。",
                "default": 300,
            },
            "isolated_workspace": {
                "type": "boolean",
                "description": "是否在隔离工作区执行（默认 False）。True 时创建独立 worktree/临时目录，避免文件冲突。",
                "default": False,
            },
            "fork": {
                "type": ["boolean", "string"],
                "enum": [True, False, "full"],
                "description": (
                    "（高级）fork 模式：继承父 system prompt 字节 + 父对话前缀，"
                    "构造 cache-identical 前缀，prompt cache 命中省 token 50%+。"
                    "true=最近 N 个 assistant turn（默认，轻量）；"
                    "\"full\"=完整对话历史（user/assistant 流，适合复杂任务，截到 50 turn 上限）。"
                    "两种模式子代理都看不到父 tool_result 真实内容（只有占位符），"
                    "仅适合 read-only 探索/分析类任务。"
                    "destructive 操作（写文件/删文件/持久化任务）请用 fork=false。默认 false。"
                ),
                "default": False,
            },
        },
    },
}


def _handle_delegate_task(args: dict, **kwargs) -> str:
    """subagent 工具的总入口：解析 LLM 传来的参数，分发给对应的委托模式。

    把 LLM 传来的参数（任务描述、角色、是否后台等）整理清楚，
    再决定走批量、异步还是同步三条路。

    参数：
      - args：LLM 传来的工具参数（goal/prompt、tasks、background、role、
        subagent_type、isolated_workspace、fork 等）
      - **kwargs：运行时上下文（agent_ref 父代理引用、config、session_id 等，
        由 tools/registry.py 的 dispatch 注入）

    返回：JSON 字符串（各模式的结果或错误信息）。
    """
    # prompt 是主入口字段，goal 是旧名字，两者兼容
    goal = args.get("goal") or args.get("prompt", "")
    tasks = args.get("tasks")
    background = args.get("background", False)
    role = args.get("role", "leaf")
    subagent_type = args.get("subagent_type", "general-purpose")

    # 批量模式：传了任务列表就走并行批量
    if tasks:
        return _delegate_batch(tasks, background=background, **kwargs)

    if not goal:
        return json.dumps({"error": "goal 不能为空"}, ensure_ascii=False)

    # 检查角色权限：coordinator（协调者）受套娃层数限制
    # 深度优先读父代理自己的字段（每线程一份，线程安全），读不到再退回 kwargs 或按 0
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        current_depth = parent_agent.spawn_depth
    else:
        current_depth = int(kwargs.get("spawn_depth", 0))
    max_depth = kwargs.get("max_spawn_depth", 2)

    if role == "orchestrator" and current_depth >= max_depth:
        return json.dumps({
            "error": f"已达最大嵌套深度 {max_depth}",
            "current_depth": current_depth,
        }, ensure_ascii=False)

    # 把 subagent_type 塞进 kwargs 传给子代理创建逻辑（用于选工具集）
    kwargs["subagent_type"] = subagent_type
    # 把 isolated_workspace 也搬进 kwargs（LLM 传在 args 里，_run_child 只认 kwargs）
    # 这行不能漏——漏了 isolated_workspace=True 永远进不了 worktree 创建逻辑
    kwargs["isolated_workspace"] = args.get("isolated_workspace", False)
    # fork 同理必须搬进 kwargs——不搬的话 LLM 传 fork:true 是无效空参数
    kwargs["fork"] = args.get("fork", False)

    if background:
        return _delegate_async(goal, args.get("context", ""), role, **kwargs)
    else:
        return _delegate_sync(goal, args.get("context", ""), role, **kwargs)


def _delegate_sync(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """同步委托：主对话原地等子代理干完活，带超时（防止永远卡死）。

    同步模式下子代理在后台线程跑，主线程 join 等待。子代理卡住时主线程
    不能无限陪等，所以有两级退出机制（用「取消信号」协作式中断，
    而不是「直接扔下不管」）：
    1. 超时后主线程按下取消信号（cancel_event），子代理在每轮调 LLM 前会检查
       这个信号 → 优雅退出，并返回 _extract_partial_result() 保留已完成的部分；
    2. 如果子代理在 sync_cancel_timeout_seconds 秒内还没响应取消信号，
       主线程强制扔下它（abandon）——线程是 daemon，进程退出时自然消亡，
       宁可冒资源泄漏风险也不让主线程无限阻塞。

    参数：
      - goal：任务描述（子代理要干什么）
      - context：给子代理的补充背景信息
      - role：角色（leaf=执行者 / orchestrator=可继续派人的协调者）
      - **kwargs：运行时上下文（config、agent_ref、session_id、cancel_event 等）

    返回：JSON 字符串（成功带 result；被中断带 partial result；超时带错误说明）。
    """
    child_timeout = float(kwargs.get("child_timeout", 600))
    # 优雅退出窗口时长从 config.delegation.sync_cancel_timeout_seconds 读（默认 2 秒）
    _cfg = kwargs.get("config") or {}
    _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
    sync_cancel_timeout = float(_delegation_cfg.get(
        "sync_cancel_timeout_seconds", 2.0,
    ))

    # 创建取消信号，一路传给 _run_child → 子代理的对话主循环
    cancel_event = threading.Event()
    kwargs["cancel_event"] = cancel_event

    # 预先生成持久化 ID 传给 _run_child：超时强制扔下子代理时，
    # 能立刻把它的元数据标成 interrupted，而不是等下次启动时的 cleanup_stale_subagents
    # 来兜底。不这样做的话，残留的 running 状态会骗过 subagent_resume，
    # 让它以为这是个还能恢复的活任务
    _pid = None
    if ((kwargs.get("config") or {}).get("delegation", {})
            .get("subagent_persistence_enabled", True)):
        try:
            from agent.subagent_persistence import generate_agent_id as _gen_id
            _pid = _gen_id(parent_session_id=kwargs.get("session_id", ""))
            kwargs["subagent_agent_id"] = _pid
        except Exception:
            _pid = None

    # box 是线程间传结果的小盒子（普通 dict，主/子线程各写各的 key）
    box: dict = {}

    def _run():
        try:
            box["result"] = _run_child(goal, context, role, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=_run, daemon=True, name="delegate-sync")
    thread.start()
    thread.join(timeout=child_timeout)

    if thread.is_alive():
        # 超时了：按下取消信号，给子代理一个优雅退出的机会
        logger.info(
            "Task K: sync 子代理超时 %ss，set cancel_event 等优雅退出",
            child_timeout,
        )
        cancel_event.set()
        thread.join(timeout=sync_cancel_timeout)

        if thread.is_alive():
            # 子代理在优雅窗口内没响应取消信号 → 强制扔下不管
            # （daemon 线程，进程退出时自然结束；LLM 调用未完成可能漏资源，
            # 但这是保底方案——宁可漏也不能让主线程无限等）
            logger.warning(
                "Task K: 子代理在 %ss 内未响应 cancel，强制 abandon",
                sync_cancel_timeout,
            )
            # 立刻标 interrupted：僵尸线程如果之后真跑完了，
            # 会把状态覆盖成 completed/failed——那是对的；这里只是别让状态一直悬在 running
            if _pid:
                try:
                    from agent.subagent_persistence import mark_completed as _mark
                    _mark(_pid, "interrupted")
                except Exception:
                    logger.debug("abandon 标记 interrupted 失败（fail-open）")
            return json.dumps({
                "success": False,
                "error": (
                    f"子代理执行超时（{child_timeout}s），"
                    f"已 set cancel_event 并等待 {sync_cancel_timeout}s 仍未退出，"
                    f"强制 abandon"
                ),
                "mode": "sync",
            }, ensure_ascii=False)

        # 子代理响应了取消信号，优雅退出了
        # 此时 _run_child 返回的是部分结果（由子代理对话循环里
        # cancel_event 分支的 _extract_partial_result 提供）
        if "error" in box:
            logger.exception("Task K: 子代理 cancel 后异常退出")
            return json.dumps({
                "success": False,
                "error": str(box["error"]),
                "mode": "sync",
                "cancelled": True,
            }, ensure_ascii=False)
        partial = box.get("result", "")
        return json.dumps({
            "success": False,  # 被中断不算成功
            "result": partial,
            "mode": "sync",
            "cancelled": True,
            "message": (
                f"子代理被 cancel 中断（超时 {child_timeout}s），"
                f"返回 partial result"
            ),
        }, ensure_ascii=False)

    if "error" in box:
        logger.exception("子代理执行失败")
        return json.dumps({
            "success": False,
            "error": str(box["error"]),
            "mode": "sync",
        }, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "result": box["result"],
        "mode": "sync",
    }, ensure_ascii=False)


def _delegate_async(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """异步委托：立刻返回一个任务 ID，子代理在后台慢慢跑，结果稍后送回。

    后台子代理跑在后台线程里，用户不在旁边盯着，所以有两道安全限制：

    1. 工具白名单（ASYNC_AGENT_ALLOWED_TOOLS）：
       后台子代理没法让用户实时审批危险操作，所以只给它安全工具——
       用户传的工具集跟白名单取交集，再注入一份禁用工具黑名单兜底。
       config.delegation.async_tool_whitelist_enabled=False 可关（不推荐）。

    2. 默认拒审批（permission_mode='autoDeny'，宁可拒绝不可放行）：
       后台子代理弹不了审批界面（用户不在场），所有需要用户批准的破坏性命令一律直接
       返回 permission_denied。config.delegation.async_auto_deny_permission=False
       可关（不推荐）。自定义 .md 定义里的 permission_mode 优先级更高
       （经 kwargs.permission_mode 透传，_run_child 的 custom_def 分支会覆盖
       这里的注入）。

    参数：
      - goal：任务描述
      - context：补充背景
      - role：leaf / orchestrator
      - **kwargs：运行时上下文（agent_ref、config、session_id 等）

    返回：JSON 字符串，含 delegation_id（如 del_153045123456），
    可用于之后调 subagent_kill 中断。
    """
    delegation_id = f"del_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"

    # 结果定向送回发起者：有 agent_ref 就送它自己的实例信箱，
    # 取件方（AIAgent._drain_injected_messages）只读自家信箱；
    # 没有 agent_ref（直接调函数/测试）才落回全局兜底信箱
    _queue = _resolve_delegation_queue(kwargs)

    # 后台子代理也配一个取消信号并登记进花名册，
    # subagent_kill 工具按下这个信号就能让后台子代理优雅退出
    cancel_event = threading.Event()
    kwargs["cancel_event"] = cancel_event

    # === 工具白名单 ===
    # fail-open：白名单逻辑出错也不崩，退回未过滤的行为
    try:
        from toolsets import (
            ASYNC_AGENT_ALLOWED_TOOLSETS,
            ASYNC_AGENT_DISALLOWED_TOOLS,
        )
        _cfg = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
        _whitelist_enabled = _delegation_cfg.get("async_tool_whitelist_enabled", True)

        if _whitelist_enabled:
            # ① 工具集取交集（用户传的 ∩ 白名单，只留两边都允许的）
            _user_ts = kwargs.get("enabled_toolsets")
            if _user_ts:
                _filtered = [ts for ts in _user_ts if ts in ASYNC_AGENT_ALLOWED_TOOLSETS]
                kwargs["enabled_toolsets"] = _filtered
            else:
                # 用户没传 → 给一个安全默认集（而不是放开全部）
                kwargs["enabled_toolsets"] = list(ASYNC_AGENT_ALLOWED_TOOLSETS)

            # ② 把禁用工具黑名单塞进 config（内置黑名单 + 用户自己加的）
            _user_extra = _delegation_cfg.get("async_disallowed_tools", [])
            _disabled = list(ASYNC_AGENT_DISALLOWED_TOOLS) + list(_user_extra)
            # 已有的禁用清单不覆盖，取并集
            _parent_cfg = kwargs.get("config")
            _child_cfg = dict(_parent_cfg) if isinstance(_parent_cfg, dict) else {}
            _existing_disabled = _child_cfg.get("disabled_tools") or []
            _child_cfg["disabled_tools"] = list(dict.fromkeys(
                list(_existing_disabled) + _disabled
            ))
            kwargs["config"] = _child_cfg
    except Exception:
        logger.warning("async 工具白名单应用失败（fail-open）", exc_info=True)

    # === 后台子代理默认拒审批（permission_mode=autoDeny）===
    # 后台子代理弹不了审批界面（用户不在场），所有需要用户批准的命令一律直接拒绝
    # （fail-closed，宁可拒绝不可放行）。
    # 优先级：自定义 .md 的 permission_mode > config 显式覆盖 > 默认 autoDeny
    # （custom_def 分支在 _run_child 内部处理，这里只注入默认/配置值；
    #   _run_child 的 custom_def 分支会覆盖此处的 kwargs 注入）
    # fail-open：配置读取出错也不崩，退回默认行为（autoDeny 这个安全默认）
    try:
        _cfg_for_perm = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        _delegation_cfg_for_perm = (
            (_cfg_for_perm.get("delegation") or {}) if isinstance(_cfg_for_perm, dict) else {}
        )
        _auto_deny_enabled = _delegation_cfg_for_perm.get(
            "async_auto_deny_permission", True,
        )
        _config_perm_mode = _delegation_cfg_for_perm.get("async_permission_mode")

        if _auto_deny_enabled:
            # 默认 autoDeny；config 显式指定了别的模式就尊重它
            _async_perm_mode = _config_perm_mode if _config_perm_mode else "autoDeny"
        else:
            # 开关关了：不注入 autoDeny，用 config 指定的或普通 default 模式
            _async_perm_mode = _config_perm_mode if _config_perm_mode else "default"

        # 只在调用方没显式传 permission_mode 时才注入（别覆盖人家的明确指定）
        if "permission_mode" not in kwargs or kwargs.get("permission_mode") is None:
            kwargs["permission_mode"] = _async_perm_mode
    except Exception:
        logger.warning("async auto_deny 注入失败（fail-open）", exc_info=True)

    def _background():
        try:
            result = _run_child(goal, context, role, **kwargs)
            _queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": True,
                "result": result,
                # spawn 时的摘要长度意图跟着结果走（诊断用；_async_tasks
                # 条目在完成时已被 pop，花名册不是载体）
                "summary_len": kwargs.get("summary_len"),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": False,
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            # 干完就把自己从花名册里划掉（防止 _async_tasks 越积越多）
            _async_tasks.pop(delegation_id, None)

    # 后台线程里跑
    thread = threading.Thread(target=_background, daemon=True,
                              name=f"delegate-async-{delegation_id}")
    thread.start()
    # 登记进花名册，让 subagent_kill 工具能找到它
    # 额外记下 goal/开始时间：上下文被压缩后，
    # post_compact_recovery 能列出还在跑的子代理，防模型「失忆」忘了自己派过人
    _async_tasks[delegation_id] = {
        "thread": thread,
        "cancel_event": cancel_event,
        "goal": goal,
        "started_at": time.time(),
    }

    return json.dumps({
        "success": True,
        "mode": "async",
        "delegation_id": delegation_id,
        "message": (
            f"子代理已启动（ID: {delegation_id}），完成后会通知你"
            f"（若主对话已空闲，完成时会自动唤醒继续处理，无需轮询）"
            f"。如需中断，调用 subagent_kill(task_id=\"{delegation_id}\")"
        ),
    }, ensure_ascii=False)


def _start_progress_ticker(
    children_state: dict,
    stop_event,
    *,
    aux,
    session_id: str,
    interval: float = 30.0,
) -> "threading.Thread":
    """多个子代理并行跑的时候，每隔 interval 秒写一条进度播报。

    让并行等待时进度可见：有辅助小模型（aux）就让它把状态归纳成 1-2 句人话；
    没有就直接机械拼一行状态。写进 scratchpad 涂鸦区的 progress.md（涂鸦区
    7 天自动清理）+ 打一条 logger.info。出错全吞（fail-open）：进度播报绝不
    能反过来影响子代理本身。

    参数：
      - children_state：各子代理的状态字典（名字 → {status, goal...}）
      - stop_event：叫停信号，set 了 ticker 线程就退出
      - aux：辅助 LLM 路由（可空；空则用机械拼接）
      - session_id：当前会话 ID（定位涂鸦区目录）
      - interval：播报间隔秒数（默认 30）

    返回：启动好的 ticker 线程对象（调用方负责 set stop_event 停掉它）。
    """
    import threading

    def _tick():
        while not stop_event.wait(interval):
            try:
                lines = [
                    f"{name}: {info.get('status', '?')}（{info.get('goal', '')[:40]}）"
                    for name, info in children_state.items()
                ]
                if not lines:
                    continue
                text = "\n".join(lines)
                if aux is not None:
                    try:
                        import asyncio
                        resp = asyncio.run(aux.chat_completions([
                            {"role": "user", "content":
                             f"把以下子代理状态摘要成 1-2 句中文进度：\n{text}"},
                        ]))
                        summarized = resp.choices[0].message.content or ""
                        if summarized.strip():
                            text = summarized.strip()
                    except Exception:
                        pass  # 摘要失败就退回机械拼接的原文
                from agent.scratchpad import scratchpad_dir
                d = scratchpad_dir(session_id)
                d.mkdir(parents=True, exist_ok=True)
                p = d / "progress.md"
                # 只留最近 20 条（防长时间运行把文件撑大；涂鸦区不是知识库）
                try:
                    old = p.read_text(encoding="utf-8").splitlines()
                except OSError:
                    old = []
                stamp = time.strftime("%H:%M:%S")
                new = old[-19:] + [f"[{stamp}] {text}"]
                p.write_text("\n".join(new) + "\n", encoding="utf-8")
                logger.info("[子代理进度] %s", text)
            except Exception as e:
                logger.debug("progress ticker fail-open: %s", e)

    t = threading.Thread(target=_tick, daemon=True, name="delegate-progress")
    t.start()
    return t


def _delegate_batch(tasks: list, *, background: bool, **kwargs) -> str:
    """批量并行委托：一次派出多个子代理同时干活，等全干完一起收结果。

    LLM 传 tasks=[...] 时走这里，用线程池真并行。每个任务各配一个取消信号，
    Ctrl+C 时全部按下，让所有子代理在下次调 LLM 前退出。

    参数：
      - tasks：任务字典列表，每项含 goal/prompt、context、role
      - background：是否后台模式（透传给各子代理路径）
      - **kwargs：运行时上下文（config、agent_ref、session_id 等）

    返回：JSON 字符串，results 列表里每个任务一条成功/失败记录。
    """
    # 并发上限从 config.delegation.max_concurrent_children 读
    # （不能只看 kwargs——只看 kwargs 的话配置永远不生效，一直 fallback 到 3）
    _cfg = (kwargs.get("config") or {}) if isinstance(kwargs.get("config"), dict) else {}
    max_concurrent = int((_cfg.get("delegation") or {}).get("max_concurrent_children", 5))
    child_timeout = float(kwargs.get("child_timeout", 600))

    results = []
    # 不用 `with ThreadPoolExecutor` 写法的原因：它退出时会 shutdown(wait=True)
    # 死等所有子线程跑完，Ctrl+C 进来会卡在等待上 → 界面看起来"没反应"。
    # 所以手动管理：收到 Ctrl+C 时传播中断 + 不阻塞等待。
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    futures = {}
    # 每个任务一个取消信号（submit 时创建，传给 _run_child）
    batch_cancel_events = []
    # 登记到全局表：本批跑在线程池线程里收不到 KeyboardInterrupt，
    # Ctrl+C/退出清理靠 cancel_all_subagents() 顺藤摸瓜按下全部取消旗
    _batch_entry = {"cancel_events": batch_cancel_events, "executor": executor}
    _active_batches.append(_batch_entry)
    # 进度 ticker 的共享状态（tasks 没有名字字段，按序号起名；
    # 约定只改 value 不增删 key——ticker 线程遍历时就不会撞上字典变更）
    children_state = {}
    _progress_stop = threading.Event()
    try:
        for i, task in enumerate(tasks):
            goal = task.get("goal", "") or task.get("prompt", "")
            context = task.get("context", "")
            role = task.get("role", "leaf")

            # 每个任务配独立取消信号（并发任务之间不互相牵连）
            task_cancel = threading.Event()
            batch_cancel_events.append(task_cancel)
            task_kwargs = dict(kwargs)
            task_kwargs["cancel_event"] = task_cancel
            # 批量任务可各自指定摘要长度（不指定继承整个 subagent 调用的值）
            if task.get("summary_len") is not None:
                task_kwargs["summary_len"] = task["summary_len"]

            future = executor.submit(_run_child, goal, context, role, **task_kwargs)
            futures[future] = i
            name = f"子代理-{i + 1}"
            children_state[name] = {"status": "running", "goal": goal}

            # 子代理一完成立刻更新 ticker 状态（收集循环是按提交顺序阻塞等
            # result 的，done_callback 不受它影响——这样进度才是实时的；出错全吞）
            def _mark_child_done(f, _name=name):
                try:
                    if f.cancelled():
                        children_state[_name]["status"] = "cancelled"
                    elif f.exception() is None:
                        children_state[_name]["status"] = "done"
                        children_state[_name]["summary"] = str(f.result())[:80]
                    else:
                        children_state[_name]["status"] = "failed"
                        children_state[_name]["summary"] = str(f.exception())[:80]
                except Exception:
                    pass
            future.add_done_callback(_mark_child_done)

        # 2 个以上子代理并行时用户只能干等——起 30 秒一班的进度播报线程，
        # 让进度可见（aux LLM 摘要或机械拼接），写进涂鸦区 progress.md
        if len(children_state) >= 2:
            _agent_ref = kwargs.get("agent_ref")
            _start_progress_ticker(
                children_state, _progress_stop,
                aux=getattr(_agent_ref, "aux_llm_router", None),
                session_id=getattr(_agent_ref, "session_id", "") or "",
            )

        for future in futures:
            idx = futures[future]
            try:
                result = future.result(timeout=child_timeout)
                results.append({
                    "task_index": idx,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "task_index": idx,
                    "success": False,
                    "error": str(e),
                })
    except KeyboardInterrupt:
        # Ctrl+C：先把中断传给子代理（让它们下轮迭代退出）、取消还没启动的
        # 任务、shutdown(wait=False) 不阻塞，然后向上抛。
        # 另外：按下所有批量子代理的取消信号，让正在跑的
        # 在下次调 LLM 前优雅退出（不能只指望 parent.interrupt() 一条路）
        parent = kwargs.get("agent_ref")
        if parent is not None and hasattr(parent, "interrupt"):
            try:
                parent.interrupt()
            except Exception:
                pass
        for ev in batch_cancel_events:
            try:
                ev.set()
            except Exception:
                pass
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # 批量收场（含 Ctrl+C 中断）必须停掉进度播报线程（daemon 本身能兜底，
        # 但显式 set 让它立刻退，不在测试之间留下等待中的线程）
        _progress_stop.set()
        # 从全局登记表划掉本批（收场了就不再接受外部取消）
        try:
            _active_batches.remove(_batch_entry)
        except ValueError:
            pass

    executor.shutdown(wait=True)

    return json.dumps({
        "success": True,
        "mode": "batch",
        "results": results,
    }, ensure_ascii=False)


def _run_child(
    goal: str,
    context: str,
    role: str,
    **kwargs,
) -> str:
    """创建并跑起一个子代理——同步/异步/批量三条路最终都汇聚到这个函数。

    子代理是一个全新的 AIAgent 实例，跟主对话各过各的，自带：
    - 会话 ID
    - 迭代预算（最多循环多少轮，默认 50）
    - 工具集（leaf 角色受限）
    - 工作目录（可隔离成独立 worktree）

    铁律：子代理不继承父代理的对话历史（只通过 goal/context 拿必要信息）。

    其它职责：
    - 把子代理登记到父代理的 _children 名单，支持中断往下传（batch1-T4）；
    - 把子代理的执行轨迹（transcript）落盘，供事后查证/恢复（出错不挡主流程）。

    参数：
      - goal：任务描述
      - context：补充背景
      - role：leaf / orchestrator
      - **kwargs：运行时上下文（config、agent_ref、cancel_event、fork、
        subagent_type、isolated_workspace、permission_mode 等）

    返回：子代理产出的结果文本（可能已被摘要压缩）。
    """
    # 延迟导入，避免和 agent 包互相 import 死锁
    from agent import AIAgent

    # === 子代理轨迹落盘初始化（出错只 debug 记录，不影响主流程）===
    _persistence_enabled = (kwargs.get("config") or {}).get(
        "delegation", {},
    ).get("subagent_persistence_enabled", True)
    _child_agent_id = None
    if _persistence_enabled:
        try:
            from agent.subagent_persistence import (
                generate_agent_id, write_metadata as _sp_write_meta,
            )
            # 调用方预先生成的 id 优先用（_delegate_sync 强制扔下子代理时要靠它标状态）
            _child_agent_id = kwargs.pop("subagent_agent_id", None) \
                or generate_agent_id(
                    parent_session_id=kwargs.get("session_id", ""),
                )
            _sp_write_meta(_child_agent_id, {
                "agent_type": kwargs.get("subagent_type", "general-purpose"),
                "parent_session_id": kwargs.get("session_id", ""),
                "description": f"{goal[:100]}",
                "status": "running",
                "created_at": time.time(),
            })
            # === 用户指令先进轨迹文件（作为轨迹开头）===
            # 真被中断的子代理之后 resume 时，原始指令就是对话的起点。
            from agent.subagent_persistence import append_message as _sp_append
            _sp_append(_child_agent_id, {
                "role": "user",
                "content": f"{goal}\n上下文: {context}" if context else goal,
                "_ts": time.time(),
            })
            logger.debug("Task I: 子代理 transcript 持久化启用: %s", _child_agent_id)
        except Exception as e:
            logger.debug("Task I: transcript 持久化初始化失败（fail-open）: %s", e)
            _child_agent_id = None

    # LLM 连接配置：优先用 kwargs 里带的，缺了再从 config 补
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")
    auth_token = kwargs.get("auth_token")
    model = kwargs.get("model")
    model_format = kwargs.get("model_format")

    if not (api_key or auth_token) or not model:
        # 缺关键配置 → 从 config 补齐
        try:
            from config import load_config
            config = load_config()

            # 子代理优先用轻量便宜的小模型（default_haiku_model），省 token——
            # 相当于业界的「主对话用旗舰、跑腿的用小模型」模式
            haiku_name = config.get("default_haiku_model", "")
            # 新配置形态：config["haiku_model"]（llm 段注入的）
            haiku_cfg = config.get("haiku_model")
            if haiku_cfg:
                sub_cfg = haiku_cfg
            elif haiku_name and haiku_name in config.get("models", {}):
                # 旧式配置（仍兼容）：config["models"][小模型名]
                sub_cfg = config["models"][haiku_name]
            else:
                # 都没有 → 退回主模型配置段
                sub_cfg = config.get("model", {})

            if not base_url:
                base_url = sub_cfg.get("base_url")
            if not model:
                model = sub_cfg.get("model") or sub_cfg.get("name")
            if not api_key:
                api_key = sub_cfg.get("api_key") or ""
            if not auth_token:
                auth_token = sub_cfg.get("auth_token") or ""
            # 向后兼容：旧式 config.yaml 用 api_key_env 写环境变量名，去环境里取真值
            if not api_key and not auth_token:
                api_key_env = sub_cfg.get("api_key_env") or ""
                if api_key_env:
                    api_key = os.environ.get(api_key_env) or ""
            if not model_format:
                model_format = sub_cfg.get("format", "anthropic")
        except Exception as e:
            raise RuntimeError(f"子代理无法获取 LLM 配置: {e}")

    if not api_key and not auth_token:
        raise RuntimeError(
            "子代理无法获取 API key（settings.json 的 models 配置为空，"
            "且未设置环境变量）"
        )

    # 构造子代理的 system prompt（开场设定词）
    system_prompt = _build_child_system_prompt(goal, context, role)

    # 计算子代理的派生深度（第几层套娃）。线程安全的做法：靠参数传递，
    # 不写 os.environ（环境变量是进程级全局，并发线程会互相覆盖）
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        parent_depth = parent_agent.spawn_depth
    else:
        parent_depth = int(kwargs.get("spawn_depth", 0))
    child_spawn_depth = parent_depth + 1

    # 工具集选择
    # 提前解析类型 + 自定义定义：为了让 isolation=worktree 能赶在创建 worktree
    # 的分支之前生效（isolated 要在自定义定义写入 kwargs 之后再读，
    # 读早了 worktree 永远创建不出来）
    stype = kwargs.get("subagent_type", "general-purpose")
    custom_def = None
    if stype not in ("general-purpose", "custom"):
        # 传的是自定义子代理名：从 .md 定义文件加载
        from agent.agent_defs import get_agent_def
        custom_def = get_agent_def(stype)
        if custom_def is None:
            raise RuntimeError(
                f"未找到子代理定义: {stype}"
                f"（检查 ~/.OmniMate/agents/ 和 ./.omnimate/agents/）"
            )

    # 可选：隔离工作区（自定义 .md 定义 isolation=worktree 也会开启）
    # 千万别用 os.chdir 切目录——那是进程级全局操作，线程池里
    # 并发跑的多个子代理会互相踩对方的工作目录。所以用
    # workspace_cwd_context（contextvars.ContextVar，线程各一份互不干扰），
    # 子代理内的工具调 get_workspace_cwd() 拿到的就是自己的 worktree。
    isolated = kwargs.get("isolated_workspace", False) or (
        custom_def is not None and custom_def.isolation == "worktree")
    workspace_cleanup = None
    workspace_path = None
    if isolated:
        try:
            from tools.worktree import create_isolated_workspace
            workspace_path, workspace_cleanup = create_isolated_workspace(
                name=f"delegate-{goal[:20].replace(' ', '-')}",
            )
            logger.info("子代理在隔离工作区运行: %s", workspace_path)
        except Exception as e:
            logger.warning("创建隔离工作区失败，用当前目录: %s", e)

    # 工作目录上下文手动管理 token（不为这个把 200 行 try/finally 重新缩进一层）。
    # 用 contextvars 替代 os.chdir：并发子代理（线程池里）每线程各有一份，
    # 不会互相踩工作目录。token 在 finally 末尾 reset（出 try 块 = 出子代理的上下文）。
    from agent.workspace_context import _workspace_cwd
    _workspace_cwd_token = None
    if workspace_path is not None:
        _workspace_cwd_token = _workspace_cwd.set(str(workspace_path))

    # 拿父代理引用（用于中断传播）
    parent_agent = kwargs.get("agent_ref")

    child = None
    # 成功标志（SUBAGENT_START/STOP 钩子要用；默认 False，正常走到 try 末尾才设 True）
    _fork_success = False
    # 父代理的钩子注册表（用于 SUBAGENT_START/STOP 审计事件）
    _parent_hooks = getattr(parent_agent, "hooks_registry", None) if parent_agent else None
    # === 工作目录变化钩子：切进 worktree 时通知一声（出错全吞，不挡主流程）===
    if workspace_path is not None and _parent_hooks is not None:
        try:
            _parent_hooks.run_cwd_changed({
                "session_id": kwargs.get("session_id", ""),
                "old": str(Path.cwd()),  # 进程当前目录（近似值）
                "new": str(workspace_path),
            })
        except Exception:
            pass  # fail-open：钩子失败不影响子代理
    try:
        # 工具集选择：
        # - 自定义名：custom_def 已提前加载，按定义配置工具集/模型/权限/轮数上限
        # - custom：用显式传的 enabled_toolsets
        # - general-purpose：按角色给默认
        # permission_mode（权限模式）优先级：
        #   ① 自定义 .md 里显式写的 permission_mode（最高）
        #   ② kwargs 里的（_delegate_async 注入的 autoDeny，或调用方显式传的）
        #   ③ "default"（兜底）
        _injected_perm_mode = kwargs.get("permission_mode")
        if custom_def:
            # 按自定义定义配置
            child_toolsets = custom_def.tools or (
                ["core"] if role == "orchestrator" else ["minimal"])
            # 自定义 .md 的 disallowedTools 直接覆盖父 config（取替不是合并）
            disabled = custom_def.disallowed_tools or None
            child_model = custom_def.model or model
            # 自定义 .md 显式指定 > kwargs 注入（后台模式的 autoDeny）
            child_perm_mode = custom_def.permission_mode or _injected_perm_mode or "default"
            child_max_iter = custom_def.max_turns or kwargs.get("child_max_iterations", 50)
        elif stype == "custom":
            child_toolsets = kwargs.get("enabled_toolsets") or (
                ["core"] if role == "orchestrator" else ["minimal"])
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        elif role == "leaf":
            child_toolsets = kwargs.get("enabled_toolsets") or ["minimal"]
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)
        else:  # orchestrator
            child_toolsets = kwargs.get("enabled_toolsets") or ["core"]
            disabled = None
            child_model = model
            child_perm_mode = _injected_perm_mode or "default"
            child_max_iter = kwargs.get("child_max_iterations", 50)

        # 自定义子代理的 system_prompt 覆盖（用定义里的重写一份）
        if custom_def and custom_def.system_prompt:
            system_prompt = _build_child_system_prompt(
                goal, context, role, override=custom_def.system_prompt)

        # === 关键提醒（critical_reminder）拼到 system_prompt 末尾 ===
        # 拼在这里对缓存友好（system_prompt 只构建一次就走缓存，不必每轮重复注入）
        if custom_def and custom_def.critical_reminder:
            system_prompt += (
                f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
            )
            logger.debug(
                "Task N: critical_reminder 已拼到子代理 system_prompt (%d 字符)",
                len(custom_def.critical_reminder),
            )

        # === memory 字段：给子代理开独立记忆目录 ===
        # 默认继承父代理的记忆库（没传则子代理自己新建默认的）
        child_memory_store = kwargs.get("memory_store")
        if custom_def and custom_def.memory:
            from constants import get_omnimate_home
            agent_memory_home = get_omnimate_home() / ".agent-memory" / custom_def.name
            try:
                agent_memory_home.mkdir(parents=True, exist_ok=True)
                from agent.memory_store import MemoryStore
                child_memory_store = MemoryStore(omnimate_home=agent_memory_home)
                logger.info(
                    "子代理 %s 使用独立记忆目录: %s",
                    custom_def.name, agent_memory_home,
                )
            except Exception as e:
                logger.warning("创建子代理独立记忆目录失败（用父 store）: %s", e)

        # === skills 字段：把技能正文预装进 system_prompt ===
        # 注意时序：必须在 system_prompt 定稿之后、AIAgent 构造之前
        if custom_def and custom_def.skills:
            try:
                from agent.skill_commands import parse_frontmatter
                from constants import all_skills_dirs
                for skill_name in custom_def.skills:
                    found = False
                    for d in all_skills_dirs():
                        p = Path(d) / skill_name / "SKILL.md"
                        if p.exists():
                            _, body = parse_frontmatter(p.read_text(encoding="utf-8"))
                            system_prompt += (
                                f"\n\n## 预装技能：{skill_name}\n{body.strip()}\n"
                            )
                            found = True
                            break
                    if not found:
                        logger.warning("子代理预装技能未找到: %s", skill_name)
            except Exception as e:
                logger.warning("子代理预装技能失败（继续）: %s", e)

        # disabled_tools 传递方式：AIAgent.__init__ 没有这个参数，只能走 config 转交
        # （get_tool_definitions 运行时会从 self.config 读 disabled_tools）
        #
        # 两个来源的禁用清单要合并（取并集、保序、去重）：
        #   ① 自定义 .md 的 disallowed_tools（custom_def 路径，上面赋给了 `disabled`）
        #   ② _delegate_async 注入到 kwargs["config"]["disabled_tools"] 的后台黑名单兜底
        #      —— 非 custom_def 路径会把它弄丢，靠这里补上
        _injected_disabled = (
            (kwargs.get("config") or {}).get("disabled_tools")
            if isinstance(kwargs.get("config"), dict)
            else None
        )
        _all_disabled = []
        for _src in (disabled, _injected_disabled):
            if _src:
                for _tool in _src:
                    if _tool not in _all_disabled:
                        _all_disabled.append(_tool)

        child_config = None
        if _all_disabled:
            # 继承父 config（如果有的话），再补上 disabled_tools
            parent_cfg = kwargs.get("config")
            child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["disabled_tools"] = _all_disabled

        # === mcp_servers 字段：子代理只暴露定义里列出的 MCP 服务器 ===
        # 时序：必须在 child_config 构造之后、AIAgent 构造之前
        if custom_def and custom_def.mcp_servers:
            if child_config is None:
                parent_cfg = kwargs.get("config")
                child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["mcp_server_filter"] = custom_def.mcp_servers

        # === 内联 mcpServers：派生时临时连接，结束时断开（不留全局残留）===
        # （不进全局 .mcp.json 注册；连接进共享的 MCPManager，工具以
        #   mcp__<服务器名>__<工具名> 前缀动态注册；finally 里断开）
        _inline_mcp_connected: list = []  # 本次临时连上的服务器名
        if custom_def and getattr(custom_def, "inline_mcp_servers", None):
            try:
                from agent.mcp_client import get_mcp_manager
                from tools.mcp_tool import register_mcp_tools
                _mgr = get_mcp_manager()
                _app_cfg = kwargs.get("config")
                for _sname, _scfg in custom_def.inline_mcp_servers.items():
                    # 项目来源的内联 MCP 必须已获首连审批，否则跳过不连（宁可不放行）
                    if not inline_mcp_spawn_allowed(custom_def, _sname, _scfg):
                        logger.warning(
                            "内联 MCP server %s（agent %s，项目来源）未获首连审批，"
                            "跳过连接（启动时会请求审批，或手动在 settings.json "
                            "mcp.approved_project_servers 加 key）",
                            _sname, custom_def.name,
                        )
                        continue
                    _mgr.connect_one(_sname, _scfg, app_config=_app_cfg)
                    _inline_mcp_connected.append(_sname)
                if _inline_mcp_connected:
                    register_mcp_tools(_mgr, servers=_inline_mcp_connected)
                    # 让子代理能看到这些服务器的工具（跟 mcp_servers 过滤合并）
                    _filter = child_config.get("mcp_server_filter") if child_config else None
                    if child_config is None:
                        _pcfg = kwargs.get("config")
                        child_config = dict(_pcfg) if isinstance(_pcfg, dict) else {}
                    child_config["mcp_server_filter"] = (
                        (list(_filter) if _filter else []) + _inline_mcp_connected
                    )
            except Exception as e:
                logger.warning("内联 mcpServers 连接失败（fail-open 继续无 MCP）: %s", e)

        # === fork 子代理路径（前缀跟父代理一字不差，蹭 prompt cache 省 token）===
        # fork=True 时：子代理继承父代理的 system prompt 原字节 + 父对话前缀
        # （最近 N 轮 assistant 发言），构造出「缓存等价」的前缀，prompt cache
        # 一命中 token 省 50% 以上。构造失败就退回普通非 fork 路径（只打 warning）
        fork_mode = kwargs.get("fork", False)
        child_initial_messages = None
        if fork_mode:
            # 读 config 开关（默认 True）
            _cfg = kwargs.get("config") or {}
            _delegation_cfg = _cfg.get("delegation") if isinstance(_cfg, dict) else {}
            _fork_enabled = (_delegation_cfg or {}).get("fork_subagent_enabled", True)
            _max_turns = int((_delegation_cfg or {}).get("fork_max_parent_turns", 3))

            if _fork_enabled and parent_agent is not None:
                try:
                    from agent.fork_messages import (
                        build_forked_messages,
                        build_forked_system_prompt,
                    )
                    parent_messages = parent_agent.conversation_history or []
                    parent_sysprompt = parent_agent._get_system_prompt() or ""
                    # 把 system_prompt 换成 fork 版（父字节 + fork 标记）
                    system_prompt = build_forked_system_prompt(
                        parent_sysprompt, child_role=role,
                    )
                    # fork 覆盖了 system_prompt，得再补一次关键提醒
                    # （critical_reminder 常是安全提醒，fork 路径不能丢）
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )
                    # 构造初始消息（父前缀 + 给子代理的指令）
                    # fork="full" 走全量模式（完整 user/assistant 对话流，
                    # 截到 delegation.fork_full_history_max_turns 上限）
                    _fork_full = fork_mode == "full"
                    _full_max = int(
                        (_delegation_cfg or {}).get("fork_full_history_max_turns", 50)
                    )
                    child_initial_messages = build_forked_messages(
                        parent_messages=parent_messages,
                        parent_system_prompt=parent_sysprompt,
                        child_directive=f"{goal}\n上下文: {context}" if context else goal,
                        max_parent_turns=_max_turns,
                        full_history=_fork_full,
                        full_history_max_turns=_full_max,
                    )
                    logger.info(
                        "Task H: fork 子代理启用，继承 %d 条 messages",
                        len(child_initial_messages),
                    )
                except Exception as e:
                    # 出错兜底：fork 构造失败，退回普通非 fork 路径
                    logger.warning(
                        "Task H: fork 构造失败，fallback 到非 fork 路径: %s", e,
                    )
                    child_initial_messages = None
                    # system_prompt 本该保留前面 _build_child_system_prompt 的结果，
                    # 但如果上面已被 fork 版覆盖到一半又出错，得整个重建一遍
                    system_prompt = _build_child_system_prompt(goal, context, role)
                    if custom_def and custom_def.system_prompt:
                        system_prompt = _build_child_system_prompt(
                            goal, context, role, override=custom_def.system_prompt)
                    # 兜底路径同样要补关键提醒（别在 fallback 里丢了安全提醒）
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )

        # === 每轮把轨迹落盘（靠 POST_LLM_CALL 程序式钩子实现）===
        # 为什么自己建一套：子代理的钩子注册表不跟主代理共享（_run_child
        # 不传，子代理拿到的是 None）→ 新建一个空的独立 HookRegistry 注册
        # 程序式钩子，完全不碰主代理的注册表。
        # 落什么：轨迹 = 用户指令 + 每轮 assistant 正文；tool_calls 和工具结果
        # 不落盘（POST_LLM_CALL 只拿得到 LLM 响应；存了带 tool_calls 但没有
        # 配对结果的消息会造出「孤儿消息」→ API 直接报 400）。这样 resume 时
        # 初始消息就是纯 user/assistant 文本流，配对天然完整。
        # 已知耦合：走的是 AIAgent._run_post_llm_call_hook，受总开关
        # config["hooks"]["enabled"] 门控（默认开）；用户显式关掉 hooks 的话
        # 轮级记录会停（只剩 user 指令那一条）。
        _child_hooks = None
        if _child_agent_id:
            try:
                from agent.hooks import HookRegistry

                def _extract_turn_text(response):
                    """从 LLM 响应里抽出 assistant 的正文文本（None 安全，两种格式都兼容）。

                    参数：
                      - response：LLM 响应对象
                    返回：正文文本；抽不出来就返回 None。
                    """
                    try:
                        msg = response.choices[0].message
                    except Exception:
                        return None
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        # Anthropic 风格的 content 是块列表 → 只把 text 块拼起来
                        parts = [
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        content = "\n".join(p for p in parts if p)
                    return content or None

                def _on_llm_turn(response):
                    """每轮 LLM 响应完，把 assistant 正文追加进轨迹文件（出错全吞不挡路）。

                    参数：
                      - response：LLM 响应对象
                    返回：None（不改动响应本身）。
                    """
                    try:
                        text = _extract_turn_text(response)
                        if text:
                            from agent.subagent_persistence import append_message
                            append_message(_child_agent_id, {
                                "role": "assistant",
                                "content": text,
                                "_ts": time.time(),
                            })
                    except Exception:
                        pass  # fail-open：落盘失败不影响子代理
                    return None  # 不修改 response

                _child_hooks = HookRegistry()
                _child_hooks.register_post_llm_call(_on_llm_turn)
            except Exception as e:
                logger.debug(
                    "Task 3: 轮级 transcript hook 注册失败（fail-open）: %s", e)
                _child_hooks = None

        child = AIAgent(
            base_url=base_url,
            api_key=api_key or None,
            auth_token=auth_token or None,
            model=child_model,
            model_format=model_format or "anthropic",
            max_iterations=child_max_iter,
            enabled_toolsets=child_toolsets,
            system_prompt_override=system_prompt,
            spawn_depth=child_spawn_depth,
            permission_mode=child_perm_mode,
            effort_level=(custom_def.effort if custom_def else None) or getattr(parent_agent, "effort_level", None),
            config=child_config,
            memory_store=child_memory_store,
            initial_messages=child_initial_messages,
            # 轮级轨迹持久化：独立空 registry + POST_LLM_CALL
            # 程序式钩子每轮追加；最终响应的 append 已删掉，避免同一内容写两遍
            hooks_registry=_child_hooks,
            omit_project_memory=bool(custom_def.omit_claude_md) if custom_def else False,
        )

        # 登记到父代理的 _children 名单（中断时能顺着名单传下去）
        if parent_agent is not None:
            try:
                parent_agent._children.append(child)
            except Exception:
                pass

        # === 长任务进行中的进度播报（P1-10）===
        # 用辅助小模型周期性生成「正在做什么」的摘要，推给父代理的
        # stream_callback，让前端不至于干等。辅助模型不可用时降级成心跳。
        from agent.progress import ProgressReporter
        progress_stream = getattr(parent_agent, "_stream_callback", None) if parent_agent else None
        progress_aux = getattr(parent_agent, "aux_llm_router", None) if parent_agent else None
        progress_interval = (kwargs.get("config") or {}).get(
            "delegation", {},
        ).get("progress_interval", 30.0)

        with ProgressReporter(
            goal=goal,
            stream_callback=progress_stream,
            aux_llm_router=progress_aux,
            interval=progress_interval,
        ):
            # 子代理开始事件（在 child.chat 之前触发；出错全吞）
            if _parent_hooks is not None:
                try:
                    _parent_hooks.run_subagent_start({
                        "session_id": kwargs.get("session_id", ""),
                        "subagent": stype,
                        "goal": goal,
                        "spawn_depth": child_spawn_depth,
                    })
                except Exception:
                    pass  # fail-open

            # 正式跑子代理
            # AIAgent.chat 是 async。_run_child 在
            # 独立线程里跑（同步/异步两条路都起的 threading.Thread），线程里
            # 没有事件循环 → 用 asyncio.run 驱动它。
            # cancel_event 也传给子代理的对话主循环：它每轮开头检查信号，
            # 一旦被按下就退出并返回已完成的部分结果
            import asyncio
            _cancel_event = kwargs.get("cancel_event")
            # === initial_prompt 前置到第一条 user 消息 ===
            # 类似斜杠命令的预处理
            _child_first_msg = f"请执行任务: {goal}"
            if custom_def and custom_def.initial_prompt:
                _child_first_msg = (
                    f"{custom_def.initial_prompt}\n\n{_child_first_msg}"
                )
                logger.debug(
                    "Task N: initial_prompt 前置到子代理首 user (%d 字符)",
                    len(custom_def.initial_prompt),
                )
            if _cancel_event is not None:
                result = asyncio.run(
                    child.chat(_child_first_msg, cancel_event=_cancel_event)
                )
            else:
                result = asyncio.run(child.chat(_child_first_msg))

        # 幻觉检测（赶在摘要压缩之前做，这样警告能保留进摘要）
        # 注意：Path.cwd() 是进程级的（就是 os.getcwd），
        # 并发子代理会互相踩。优先用 kwargs 里的 cwd，没有就读线程局部的
        # get_workspace_cwd()。
        try:
            from agent.team.hallucination_check import verify_claims, append_warning
            from agent.workspace_context import get_workspace_cwd
            verification = verify_claims(
                result,
                task_store=kwargs.get("task_store"),
                fs_cwd=kwargs.get("cwd") or Path(get_workspace_cwd()),
            )
            result = append_warning(result, verification)
        except Exception as e:
            logger.warning("幻觉检测失败（fail-open）: %s", e)

        # === 放权模式下的交接复审 ===
        # bypass/auto 权限模式下，子代理产出会直接进父代理上下文——危险产出
        # （破坏命令证据/数据外发/凭证修改痕迹）由辅助 LLM 复审一遍，命中就在
        # 结果前面附警告（注意不拦截——怎么处理由父代理和用户自己决断）。
        # 由 feature flag delegation.handoff_review_enabled 门控（默认关）；出错全吞。
        try:
            _hr_cfg = (kwargs.get("config") or {}).get("delegation") or {}
            if _hr_cfg.get("handoff_review_enabled", False):
                result = _review_handoff(result, parent_agent)
        except Exception as e:
            logger.warning("交接复审失败（fail-open）: %s", e)

        # summary_only：结果太长就用 LLM 压成摘要，省父代理的上下文空间
        # summary_len：摘要长度（默认 300；长任务/深度调研可调大，见 schema 说明）
        summary_only = kwargs.get("summary_only", True)
        if summary_only and len(result) > 500:
            try:
                summary_len = int(kwargs.get("summary_len", 300))
            except (TypeError, ValueError):
                summary_len = 300
            summary_len = max(100, min(2000, summary_len))
            result = _summarize_child_result(
                result, child.llm_client, child.model, max_chars=summary_len,
            )

        # 走到这里说明成功了，把成功标志立起来（finally 里靠它触发结束事件）
        _fork_success = True

        # 把轨迹状态标成 completed
        if _child_agent_id:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "completed")
            except Exception:
                pass  # fail-open

        return result
    finally:
        # === 清杀子代理留下的运行状态 ===
        # 级联中断孙代理（免得后台线程往已死的父代理推结果）+ 停后台任务；
        # 可重复执行且出错不挡路，放在清理链最前（后面步骤不再依赖子代理活着）
        try:
            child.cleanup_runtime()
        except Exception:
            pass  # fail-open：这步失败不挡后面的清理

        # === 断开临时连的内联 MCP 服务器（别留全局残留连接）===
        if _inline_mcp_connected:
            try:
                from agent.mcp_client import get_mcp_manager
                _mgr = get_mcp_manager()
                for _sname in _inline_mcp_connected:
                    _mgr.disconnect_one(_sname)
            except Exception:
                pass  # fail-open：断不开也不挡清理链

        # === 出错时把轨迹标成 failed（_fork_success=False 说明没走到 return）===
        if _child_agent_id and not _fork_success:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "failed")
            except Exception:
                pass  # fail-open

        # 从父代理的 _children 名单里划掉自己
        if parent_agent is not None and child is not None:
            try:
                if child in parent_agent._children:
                    parent_agent._children.remove(child)
            except Exception:
                pass
        # 恢复工作目录上下文（替代 os.chdir 的回切）
        # ContextVar 的 token reset 只影响当前线程，踩不到别的并发子代理
        if _workspace_cwd_token is not None:
            try:
                _workspace_cwd.reset(_workspace_cwd_token)
            except Exception:
                pass
        if workspace_cleanup:
            # 智能清理 worktree：子代理有改动就保留现场，没改动才删
            # config.delegation.worktree_always_cleanup=True → 恢复旧行为（无脑总清理）
            _cfg = kwargs.get("config") or {}
            _delegation_cfg = _cfg.get("delegation") if isinstance(_cfg, dict) else {}
            _always_cleanup = (_delegation_cfg or {}).get("worktree_always_cleanup", False)
            try:
                if _always_cleanup:
                    workspace_cleanup(force=True)
                else:
                    cleaned = workspace_cleanup()
                    if cleaned is False and workspace_path is not None:
                        logger.warning(
                            "worktree 保留（子代理有改动）: %s", workspace_path,
                        )
            except Exception as e:
                logger.warning("worktree 智能清理异常（fail-open）: %s", e)

        # 子代理结束事件（成功失败都触发；出错全吞）
        if _parent_hooks is not None:
            try:
                _parent_hooks.run_subagent_stop({
                    "session_id": kwargs.get("session_id", ""),
                    "subagent": stype,
                    "goal": goal,
                    "success": _fork_success,
                })
            except Exception:
                pass  # fail-open


def _review_handoff(result: str, parent_agent) -> str:
    """交接复审：辅助 LLM 检查子代理要交回来的产出，发现危险内容就在前面贴警告。

    放权模式下子代理产出直接进父代理上下文，这一步让辅助 LLM 当安检员。
    判安全、辅助模型不可用或出错，都原样返回（fail-open，不拦路）。接口是
    同步的（_run_child 在线程里跑、没有事件循环，内部用 asyncio.run 搭桥）。

    参数：
      - result：子代理的产出文本
      - parent_agent：父代理（从它身上取 aux_llm_router 辅助模型）

    返回：可能带 `[⚠ 交接复审警告]` 前缀的产出文本。
    """
    import asyncio
    aux = getattr(parent_agent, "aux_llm_router", None)
    if aux is None or not result or not result.strip():
        return result
    if len(result) < 200:
        return result  # 短结果一眼能看完，省一次辅助模型调用

    prompt = (
        "复审以下子代理交给主代理的产出片段。判断它是否包含危险操作的证据：\n"
        "- 删除/覆盖大量文件或系统目录\n"
        "- 数据外发（上传/推送到外部、大段凭证或密钥输出）\n"
        "- 修改凭证/SSH/认证配置\n"
        "- 绕过或关闭安全机制\n"
        "只看产出本身，不推测意图。\n\n"
        f"产出：\n{result[:4000]}\n\n"
        '输出严格 JSON：{"dangerous": true/false, "warning": "<=40 字中文说明，仅 dangerous 时非空>"}'
    )
    try:
        resp = asyncio.run(aux.chat_completions(
            [{"role": "user", "content": prompt}],
        ))
        import json as _json
        text = resp.choices[0].message.content or ""
        parsed = _json.loads(text.strip().strip("`"))
        if isinstance(parsed, dict) and parsed.get("dangerous"):
            warning = str(parsed.get("warning", ""))[:80] or "子代理产出含危险操作证据"
            logger.warning("交接复审命中: %s", warning)
            return f"[⚠ 交接复审警告] {warning}\n\n{result}"
    except Exception as e:
        logger.debug("交接复审 aux 调用失败（放行原文）: %s", e)
    return result


def _summarize_child_result(
    result: str, client, model: str, max_chars: int = 300,
) -> str:
    """用 LLM 把子代理的结果压成指定字数以内的摘要（默认 300 字）。

    子代理动辄输出几千字，全文塞回主对话太费上下文——超长结果先摘要。
    摘要失败就返回原文（不能因为压缩失败把整个委托卡死）。

    参数：
      - result：子代理的原始结果文本
      - client：子代理的 LLM 客户端（child.llm_client）
      - model：模型名
      - max_chars：摘要字数上限（长任务的深度调研可调大，如 800/1500）

    返回：`[摘要] ...` 格式的压缩文本；失败时返回原文。

    接口说明：call_with_retry 是 async，本函数仍保持
    同步接口（调用方 _run_child 在独立线程里跑、没有事件循环），内部用
    asyncio.run() 驱动那个 async 函数。
    """
    # 输入材料随目标长度放宽（要写更长摘要就得多给原文），封顶 30000
    input_limit = min(30000, max(8000, max_chars * 10))
    prompt = (
        f"把以下子代理执行结果总结成 {max_chars} 字以内的摘要，保留：\n"
        "1. 核心结论\n"
        "2. 关键发现和数据\n"
        "3. 重要的文件路径、命令、错误信息\n"
        "4. 待办事项\n\n"
        f"子代理结果：\n{result[:input_limit]}"
    )
    try:
        import asyncio
        from agent.llm_retry import call_with_retry
        response = asyncio.run(call_with_retry(
            client,  # child.llm_client（LLM 客户端实例）
            [{"role": "user", "content": prompt}],
            background=True,  # 摘要属于后台活：遇 529 过载直接放弃不重试
        ))
        summary = response.choices[0].message.content
        return f"[摘要] {summary}\n\n[完整结果 {len(result)} 字符已省略]"
    except Exception as e:
        logger.debug("子代理结果摘要失败，返回原文: %s", e)
        return result


def _build_child_system_prompt(goal: str, context: str, role: str, override: str = None) -> str:
    """拼出子代理的 system prompt（开场设定词）。

    告诉子代理「你是谁、要干什么、守什么规矩」。

    参数：
      - goal：任务描述
      - context：父代理给的补充背景
      - role：leaf / orchestrator 角色
      - override：自定义子代理 .md 里写的 system_prompt；非空时以它为底，
        只在后面补上下文/约束/角色提示；None 时走默认模板

    返回：拼好的 system prompt 字符串。
    """
    if override:
        parts = [override]
    else:
        parts = [
            "你是一个子代理，由父代理派生执行独立任务。",
            f"\n你的角色: {role}",
            f"\n你的任务目标: {goal}",
        ]

    if context:
        parts.append(f"\n来自父代理的上下文:\n{context}")

    # 自定义 override 也要补一份通用约束（只是不再塞默认的 goal/role 那几句）
    if override:
        parts.append(
            "\n## 约束\n"
            "- 独立执行，不假设父代理历史\n"
            "- 结果要具体（文件路径、命令、数据）\n"
        )
    else:
        parts.append(
            "\n要求:\n"
            "- 专注完成任务，不要偏离目标\n"
            "- 完成后给出清晰的总结\n"
            "- 遇到不可解决的阻碍时，返回错误说明\n"
            "- 不要做任务范围外的事"
        )

    if role == "leaf":
        parts.append("\n你是 leaf 角色，不能再派生子代理。")

    return "\n".join(parts)


def _delegate_schema_overrides(schema: dict, runtime_ctx: dict) -> dict:
    """按运行时状态给 subagent 工具的说明文字追加「还剩几个坑位」。

    并发子代理有上限，把实时槽位数写进工具描述，让 LLM 提前知道，
    避免白派一次被拒、浪费一整轮。

    参数：
      - schema：原工具 schema
      - runtime_ctx：运行时上下文（从中取 agent 引用）

    返回：追加了运行时状态行的新 schema（原 schema 不动）。
    """
    agent = runtime_ctx.get("agent") if runtime_ctx else None
    if agent is None:
        return schema

    active = len(getattr(agent, "_children", []) or [])
    cfg = getattr(agent, "config", None) or {}
    max_children = (
        (cfg.get("delegation") or {}).get("max_concurrent_children", 5)
        if isinstance(cfg, dict) else 5
    )
    remaining = max(0, max_children - active)

    new_schema = dict(schema)
    desc = new_schema.get("description", "")
    status_line = (
        f"\n\n[运行时状态] 当前活跃子代理: {active}/{max_children}，"
        f"剩余可委派: {remaining}"
    )
    if remaining == 0:
        status_line += "\n⚠️ 已达并发上限，再委派会被拒绝。"
    new_schema["description"] = desc + status_line
    return new_schema


# ---------------------------------------------------------------------------
# subagent_kill 工具：中断后台跑的子代理
# ---------------------------------------------------------------------------

SUBAGENT_KILL_SCHEMA = {
    "name": "subagent_kill",
    "description": (
        "中断后台子代理（让子代理优雅退出 + 保留已完成部分）。"
        "适用场景：async 子代理跑偏、用户 ESC 想停、任务已完成想提前 kill。"
        "\n\n**注意**：\n"
        "- 只能 kill async 子代理（subagent(background=True) 返回的 delegation_id）\n"
        "- sync 子代理由父代理超时机制管理，不需要显式 kill\n"
        "- kill 是协作式的：set cancel_event，子代理在下次 LLM 调用前检查退出"
        "（不会真杀线程）\n"
        "- kill 后子代理仍有 sync_cancel_timeout_seconds 秒响应窗口"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "要中断的子代理 task_id（即 subagent(background=True) "
                    "返回的 delegation_id）"
                ),
            },
        },
        "required": ["task_id"],
    },
}


def _handle_subagent_kill(args: dict, **kwargs) -> str:
    """中断后台子代理。

    做法：查 _async_tasks 花名册找到目标，按下它的取消信号——子代理的
    对话主循环每轮开头都查这个信号，一发现被按下就退出，并返回
    _extract_partial_result() 保留已完成的部分。

    参数：
      - args：LLM 传的参数（task_id，即 subagent(background=True) 返回的 delegation_id）
      - **kwargs：运行时上下文（config 等）

    返回：JSON 字符串（已通知退出 / 各类错误）。查询或操作出错不影响主流程。
    """
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "task_id 不能为空",
            "error_type": "invalid_argument",
        }, ensure_ascii=False)

    # 查 config 开关（config.delegation.async_kill_enabled）
    _cfg = kwargs.get("config") or {}
    _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
    if not _delegation_cfg.get("async_kill_enabled", True):
        return json.dumps({
            "error": "subagent_kill 工具已被 config.delegation.async_kill_enabled=False 禁用",
            "error_type": "disabled",
        }, ensure_ascii=False)

    info = _async_tasks.get(task_id)
    if info is None:
        return json.dumps({
            "error": f"任务 {task_id} 不存在（可能已完成或 task_id 错误）",
            "error_type": "not_found",
        }, ensure_ascii=False)

    try:
        cancel_event = info.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

        # 给子代理留出优雅退出的时间（不等 thread.join() 完整跑完，只是软通知）
        # 注意：这里不阻塞主流程，线程什么时候真正退出由子代理主循环自己检查决定
        thread = info.get("thread")
        if thread is not None and thread.is_alive():
            # 不等 join——kill 工具本身得快速返回
            # 子代理在自己的线程里继续跑到取消信号检查生效为止
            logger.info(
                "Task K: subagent_kill task_id=%s，cancel_event 已 set",
                task_id,
            )

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "status": "killed",
            "message": (
                "已通知子代理退出（cancel_event 已 set），"
                "子代理将在下次 LLM 调用前退出并返回 partial result"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("subagent_kill 异常（fail-open）")
        return json.dumps({
            "error": f"kill 操作异常: {e}",
            "error_type": "internal_error",
            "task_id": task_id,
        }, ensure_ascii=False)


def _subagent_kill_check_fn() -> bool:
    """可见性开关：按 config.delegation.async_kill_enabled 决定 subagent_kill 对 LLM 显不显示。

    工具注册（登记）和暴露给 LLM（可见）是两步，这个 check_fn 就是那个
    动态开关——True（默认）→ 工具对 LLM 可见；False → 隐藏。
    注意：registry._check_fn_cached 调用 fn() 时不传任何参数，必须用
    无参签名（跟其他 check_fn 保持一致）。

    返回：bool，是否可见；读配置出错时返 True（fail-open）。
    """
    try:
        from agent.settings import load_settings
        cfg = load_settings() or {}
        delegation = cfg.get("delegation") or {}
        return bool(delegation.get("async_kill_enabled", True))
    except Exception:
        return True  # fail-open：读不到配置就默认可见


# 注册到 core 工具集（这样 resolve("core") 能找到）
registry.register(
    name="subagent",
    toolset="core",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
    isConcurrencySafe=False,  # 有副作用（派子代理耗资源 + 改子任务状态），必须串行
)
# 兼容别名：老会话/记忆里存的 delegate_task 调用依然能按名字命中
# toolset="_compat" 的用意：dispatch 只按名字命中（不看工具集），但
# resolve_toolset("core") 不会包含它——LLM 只看到 subagent 一个名字，
# delegate_task 纯粹给历史调用兜底
registry.register(
    name="delegate_task",
    toolset="_compat",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
    override=True,
    isConcurrencySafe=False,  # subagent 的别名，同样有副作用，必须串行
)
# subagent_kill 工具（中断后台子代理）
registry.register(
    name="subagent_kill",
    toolset="core",
    schema=SUBAGENT_KILL_SCHEMA,
    handler=_handle_subagent_kill,
    check_fn=_subagent_kill_check_fn,
    emoji="🛑",
    isConcurrencySafe=False,  # 有副作用（按取消信号 + 改花名册），串行
)
