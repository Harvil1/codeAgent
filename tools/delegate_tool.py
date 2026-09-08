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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List

from tools.registry import registry

# 结果后处理与进度播报拆到独立模块（纯搬迁）；带下划线原名 re-export——
# 外部函数内 import（verify 的 check_delegate_offload 等）与文件内调用点零改动
from tools.delegate_result import (  # noqa: F401
    _review_handoff,
    _offload_child_result,
    _attach_full_result_pointer,
    _summarize_child_result,
    _build_child_system_prompt,
    _start_progress_ticker,
)

# 委托配置纯函数三件拆到独立模块（纯搬迁）；同名 import——文件内调用点
# 零改动，且顺带构成 re-export（tools.delegate_tool.X 符号面保持可用）
from tools.delegate_setup import (  # noqa: F401
    inline_mcp_spawn_allowed,
    _validate_toolset_names,
    _delegate_schema_overrides,
)

# kill 工具簇三件拆到独立模块（纯搬迁）；同名 import——底部注册块引用
# 零改动，且顺带构成 re-export（tools.delegate_tool.X 符号面保持可用）。
# _async_tasks 花名册仍留本文件（spawn 侧写）；delegate_kill 侧在函数体内
# 运行时回读本模块拿花名册（本文件早已加载完，无循环）。
from tools.delegate_kill import (  # noqa: F401
    SUBAGENT_KILL_SCHEMA,
    _handle_subagent_kill,
    _subagent_kill_check_fn,
)

# _run_child 巨无霸拆到独立模块（纯搬迁）；re-export——三条链调用点
# （_delegate_sync/_delegate_async/_delegate_batch）、外部延迟 import
# （hook_exec/workflow_engine/plan_mode_tool）与 verify 的
# patch("tools.delegate_tool._run_child") 都解析主文件全局名，照旧命中
from tools.delegate_child import _run_child  # noqa: F401

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
                    "或自定义子代理名（扫描 ~/.codeAgent/agents/*.md 和 ./.codeAgent/agents/*.md 的 name 字段）。"
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

    # 套娃深度守卫不在这里做：入口只看得出 role，看不出子代理的套餐里
    # 带没带 subagent——只认 role=="orchestrator" 会让 role 默认 leaf 的
    # coordinator / 自定义 .md 派生绕过深度封顶（coordinator→coordinator→…
    # 无限链）。守卫统一放在 _run_child 里工具集成型之后，按
    # 「解析后的可见工具含 subagent 一族」判定（与角色名无关）。

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

    # live 面板进场（单个子代理：spinner 下挂 ⎿ 当前活动 + 工具计数）。
    # key 用 tool_call_id 派生（同回合多次委托不撞车）；纯展示 fail-open
    _ui_key = f"sync-{kwargs.get('tool_call_id') or id(box)}"
    kwargs["ui_child_key"] = _ui_key
    try:
        import cli_live
        cli_live.agent_begin(_ui_key, goal[:50])
    except Exception:
        pass

    def _run():
        try:
            box["result"] = _run_child(goal, context, role, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e
        finally:
            # 收场状态同步给 live 面板（abandon 时线程还活着到不了这里，
            # 黑板残留的 running 行由回合结束统一清）
            try:
                import cli_live
                cli_live.agent_finish(
                    _ui_key,
                    status="failed" if "error" in box else "done")
            except Exception:
                pass

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
    # live 面板进场（claude code 同款 Running N agents 树）：每个子代理
    # 一行，key 用「子代理-N」——_run_child 的 UI 钩子按同一把 key 上报
    # 工具活动（纯展示，cli_live 内部全吞异常，绝不挡委托本身）
    try:
        import cli_live
        cli_live.agents_begin([
            (f"子代理-{i + 1}",
             (t.get("goal") or t.get("prompt") or "")[:40] or f"task-{i}")
            for i, t in enumerate(tasks)
        ])
    except Exception:
        pass
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
            # live 面板的 key：子代理的工具活动按它归到自己的树杈上
            task_kwargs["ui_child_key"] = f"子代理-{i + 1}"
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
                        _st = "cancelled"
                    elif f.exception() is None:
                        children_state[_name]["status"] = "done"
                        children_state[_name]["summary"] = str(f.result())[:80]
                        _st = "done"
                    else:
                        children_state[_name]["status"] = "failed"
                        children_state[_name]["summary"] = str(f.exception())[:80]
                        _st = "failed"
                    try:
                        import cli_live
                        cli_live.agent_finish(_name, status=_st)
                    except Exception:
                        pass
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
