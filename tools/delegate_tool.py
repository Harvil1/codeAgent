"""subagent 工具：派生子代理执行独立任务（对齐 Claude Code Agent）。

两种角色：
  - leaf（默认）：执行者，不能再委托
    有 terminal/read_file 等工具
    不能调用 subagent/clarify/memory/send_message（通过工具集隔离）

  - orchestrator：协调者，可以继续委托
    可以调用 subagent 派生自己的子代理
    受 max_spawn_depth 限制（默认 2 层）

三种模式：
  - 同步（默认）：等待子代理完成
  - 异步（background=True）：立即返回，结果通过队列送达
  - 批量（tasks=[...]）：并行执行多个子代理
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
# 异步委托结果队列
# ---------------------------------------------------------------------------

class DelegationCompletionQueue:
    """异步委托完成结果的队列。

    后台子代理完成后，结果进入队列。
    父代理在每次循环开始时检查队列。
    """

    def __init__(self):
        self._queue: List[dict] = []
        self._lock = threading.Lock()

    def push(self, result: dict) -> None:
        """子代理完成时调用。"""
        with self._lock:
            self._queue.append(result)

    def drain(self) -> List[dict]:
        """父代理消费所有结果。"""
        with self._lock:
            results = list(self._queue)
            self._queue.clear()
            return results

    def has_pending(self) -> bool:
        with self._lock:
            return len(self._queue) > 0


# 全局单例（每个 agent 实例应该有自己的，这里简化）
_delegation_queue = DelegationCompletionQueue()


# === Task K: async 子代理注册表（让 subagent_kill 能找到正在跑的 task）===
# key = delegation_id（_delegate_async 生成的 del_xxx），
# value = {"thread": Thread, "cancel_event": threading.Event}
# 注意：进程内全局，不跨进程；多个 AIAgent 实例共享同一注册表
# （进程内真并发的 async 子代理才能被 kill 工具定位）
_async_tasks: Dict[str, dict] = {}


def get_delegation_queue() -> DelegationCompletionQueue:
    """获取全局委托完成队列。"""
    return _delegation_queue


# ---------------------------------------------------------------------------
# subagent 工具
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "subagent",
    "description": (
        "派生子代理（subagent）执行独立任务（对齐 Claude Code Agent 工具）。"
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
                "description": "子代理任务（Claude Code Agent 主入口，必填之一）",
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
                "description": "是否只返回摘要（默认 True）。超长结果用 LLM 压缩成 300 字摘要，节省父代理 context。",
                "default": True,
            },
            "isolated_workspace": {
                "type": "boolean",
                "description": "是否在隔离工作区执行（默认 False）。True 时创建独立 worktree/临时目录，避免文件冲突。",
                "default": False,
            },
            "fork": {
                "type": "boolean",
                "description": (
                    "（高级）fork 模式：继承父 system prompt 字节 + 父对话前缀（最近 N 个 assistant turn），"
                    "构造 cache-identical 前缀，prompt cache 命中省 token 50%+。"
                    "仅适合 read-only 探索/分析类任务（子代理看不到父 tool_result 真实内容，只有占位符）。"
                    "destructive 操作（写文件/删文件/持久化任务）请用 fork=False。默认 False。"
                ),
                "default": False,
            },
        },
    },
}


def _handle_delegate_task(args: dict, **kwargs) -> str:
    """处理委托请求（subagent）。"""
    # prompt 是 Claude Code Agent 主入口，兼容 goal
    goal = args.get("goal") or args.get("prompt", "")
    tasks = args.get("tasks")
    background = args.get("background", False)
    role = args.get("role", "leaf")
    subagent_type = args.get("subagent_type", "general-purpose")

    # 批量模式
    if tasks:
        return _delegate_batch(tasks, background=background, **kwargs)

    if not goal:
        return json.dumps({"error": "goal 不能为空"}, ensure_ascii=False)

    # 检查角色权限（orchestrator 受深度限制）
    # 深度优先从父 agent 字段读（线程安全的私份），fallback 到 kwargs 或 0
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

    # 透传 subagent_type 给子代理创建（工具集选择）
    kwargs["subagent_type"] = subagent_type
    # 透传 isolated_workspace（LLM 传的 args 字段，_run_child 从 kwargs 读）
    # 之前漏搬导致 isolated_workspace=True 永远进不去 worktree 创建逻辑
    kwargs["isolated_workspace"] = args.get("isolated_workspace", False)

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
    """同步委托：等待子代理完成。带超时（防止无限挂起）。

    Task K: 用 cancel_event 协作式中断替代 daemon=True + abandon。
    主线程超时后 set cancel_event，子代理在 LLM 调用前每轮检查 → 优雅退出，
    返回 _extract_partial_result() 保留已完成部分（借鉴 Claude Code extractPartialResult）。
    子代理在 sync_cancel_timeout_seconds 内不响应时，主线程强制 abandon
    （daemon=True 让进程退出时自然结束），避免主线程无限阻塞。
    """
    child_timeout = float(kwargs.get("child_timeout", 600))
    # Task K: 从 config.delegation.sync_cancel_timeout_seconds 读优雅退出窗口
    _cfg = kwargs.get("config") or {}
    _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
    sync_cancel_timeout = float(_delegation_cfg.get(
        "sync_cancel_timeout_seconds", 2.0,
    ))

    # Task K: 创建 cancel_event，透传给 _run_child → child AIAgent.run_conversation
    cancel_event = threading.Event()
    kwargs["cancel_event"] = cancel_event

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
        # Task K: 主线程超时 → set cancel_event 让子代理优雅退出
        logger.info(
            "Task K: sync 子代理超时 %ss，set cancel_event 等优雅退出",
            child_timeout,
        )
        cancel_event.set()
        thread.join(timeout=sync_cancel_timeout)

        if thread.is_alive():
            # 子代理未在 sync_cancel_timeout_seconds 内响应 cancel
            # → 强制 abandon（daemon=True 让进程退出时自然结束）
            # 资源仍可能泄漏（LLM 调用未完成），但这是 fail-safe，不应阻塞主线程
            logger.warning(
                "Task K: 子代理在 %ss 内未响应 cancel，强制 abandon",
                sync_cancel_timeout,
            )
            return json.dumps({
                "success": False,
                "error": (
                    f"子代理执行超时（{child_timeout}s），"
                    f"已 set cancel_event 并等待 {sync_cancel_timeout}s 仍未退出，"
                    f"强制 abandon"
                ),
                "mode": "sync",
            }, ensure_ascii=False)

        # 子代理在 cancel_event 触发后优雅退出
        # 此时 _run_child 应返回 partial result（由 child.run_conversation 内
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
    """异步委托：立即返回，后台执行。

    Task F: async 模式强制工具白名单（对齐 claude-code-main ASYNC_AGENT_ALLOWED_TOOLS）。
    后台子代理在 daemon 线程跑，用户无法实时审批 destructive 操作，
    因此过滤 enabled_toolsets（白名单交集）+ 注入 disabled_tools（黑名单兜底）。
    config.delegation.async_tool_whitelist_enabled=False 可关（不推荐）。

    Task J: async 模式默认注入 permission_mode='autoDeny'（fail-closed）。
    借鉴 Claude Code `shouldAvoidPermissionPrompts: true`。后台子代理不能弹审批
    UI（用户不在场），所有需 user approval 的破坏性命令直接 permission_denied。
    config.delegation.async_auto_deny_permission=False 可关（不推荐）。
    custom_def.permission_mode 优先（通过 kwargs.permission_mode 透传，
    _run_child 内 custom_def 分支会覆盖此处的注入）。
    """
    delegation_id = f"del_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"

    # === Task K: async 子代理也创建 cancel_event，注册到 _async_tasks ===
    # subagent_kill 工具可 set 此 event，让 async 子代理优雅退出
    cancel_event = threading.Event()
    kwargs["cancel_event"] = cancel_event

    # === Task F: async 工具白名单 ===
    # fail-open：白名单逻辑异常时不崩，退回原行为
    try:
        from toolsets import (
            ASYNC_AGENT_ALLOWED_TOOLSETS,
            ASYNC_AGENT_DISALLOWED_TOOLS,
        )
        _cfg = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
        _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
        _whitelist_enabled = _delegation_cfg.get("async_tool_whitelist_enabled", True)

        if _whitelist_enabled:
            # ① toolsets 取交集（用户传的跟白名单）
            _user_ts = kwargs.get("enabled_toolsets")
            if _user_ts:
                _filtered = [ts for ts in _user_ts if ts in ASYNC_AGENT_ALLOWED_TOOLSETS]
                kwargs["enabled_toolsets"] = _filtered
            else:
                # 用户没传 → 给一个安全默认（不是全部）
                kwargs["enabled_toolsets"] = list(ASYNC_AGENT_ALLOWED_TOOLSETS)

            # ② disabled_tools 注入 config（内置黑名单 + 用户扩展）
            _user_extra = _delegation_cfg.get("async_disallowed_tools", [])
            _disabled = list(ASYNC_AGENT_DISALLOWED_TOOLS) + list(_user_extra)
            # 不覆盖 custom_def 已设的 disabled_tools（取并集）
            _parent_cfg = kwargs.get("config")
            _child_cfg = dict(_parent_cfg) if isinstance(_parent_cfg, dict) else {}
            _existing_disabled = _child_cfg.get("disabled_tools") or []
            _child_cfg["disabled_tools"] = list(dict.fromkeys(
                list(_existing_disabled) + _disabled
            ))
            kwargs["config"] = _child_cfg
    except Exception:
        logger.warning("async 工具白名单应用失败（fail-open）", exc_info=True)

    # === Task J: async 子代理默认拒审批（permission_mode=autoDeny）===
    # 借鉴 Claude Code `shouldAvoidPermissionPrompts: true`：后台子代理不能弹
    # 审批 UI（用户不在场），所有需 user approval 的命令直接 fail-closed。
    # 优先级：custom_def.permission_mode > config 显式覆盖 > 默认 autoDeny
    # （custom_def 分支在 _run_child 内部处理，这里只注入默认/配置值；
    #   _run_child 的 custom_def 分支会覆盖此处的 kwargs 注入）
    # fail-open：配置异常时不崩，退回默认行为（autoDeny 安全默认）
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
            # 默认 autoDeny；config 显式指定其他 mode 时尊重
            _async_perm_mode = _config_perm_mode if _config_perm_mode else "autoDeny"
        else:
            # 开关关闭：不注入 autoDeny，用 config 指定或 default
            _async_perm_mode = _config_perm_mode if _config_perm_mode else "default"

        # 仅在调用方未显式传 permission_mode 时注入（避免覆盖显式调用）
        if "permission_mode" not in kwargs or kwargs.get("permission_mode") is None:
            kwargs["permission_mode"] = _async_perm_mode
    except Exception:
        logger.warning("async auto_deny 注入失败（fail-open）", exc_info=True)

    def _background():
        try:
            result = _run_child(goal, context, role, **kwargs)
            _delegation_queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": True,
                "result": result,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _delegation_queue.push({
                "delegation_id": delegation_id,
                "goal": goal,
                "success": False,
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            # Task K: 退出时从注册表清掉（防止 _async_tasks 无限增长）
            _async_tasks.pop(delegation_id, None)

    # 后台线程执行
    thread = threading.Thread(target=_background, daemon=True,
                              name=f"delegate-async-{delegation_id}")
    thread.start()
    # Task K: 注册到 _async_tasks，让 subagent_kill 工具能定位
    # T2（核心机制对齐第 2 项）：额外记录 goal/started_at，
    # post_compact_recovery 压缩后列出 running 子代理（防模型失忆）
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
            f"。如需中断，调用 subagent_kill(task_id=\"{delegation_id}\")"
        ),
    }, ensure_ascii=False)


def _delegate_batch(tasks: list, *, background: bool, **kwargs) -> str:
    """批量并行委托。

    Task K: 每个任务一个 cancel_event，跟踪到 batch_cancel_events。
    KeyboardInterrupt 时 set 所有 cancel_event（让所有子代理在 LLM 调用前退出）。
    """
    # 从 config 读并发上限（config.delegation.max_concurrent_children），
    # 不依赖 kwargs（之前永远 fallback 3，配置不生效）
    _cfg = (kwargs.get("config") or {}) if isinstance(kwargs.get("config"), dict) else {}
    max_concurrent = int((_cfg.get("delegation") or {}).get("max_concurrent_children", 5))
    child_timeout = float(kwargs.get("child_timeout", 600))

    results = []
    # 不用 `with ThreadPoolExecutor`：它退出时 shutdown(wait=True)，
    # Ctrl+C 的 KeyboardInterrupt 会卡在等子线程跑完 → 界面"没反应"。
    # 手动管理，KeyboardInterrupt 时传播中断 + 不阻塞等待。
    executor = ThreadPoolExecutor(max_workers=max_concurrent)
    futures = {}
    # Task K: 每任务的 cancel_event（submit 时创建，传给 _run_child）
    batch_cancel_events = []
    try:
        for i, task in enumerate(tasks):
            goal = task.get("goal", "") or task.get("prompt", "")
            context = task.get("context", "")
            role = task.get("role", "leaf")

            # Task K: 每任务独立 cancel_event（并发不互相干扰）
            task_cancel = threading.Event()
            batch_cancel_events.append(task_cancel)
            task_kwargs = dict(kwargs)
            task_kwargs["cancel_event"] = task_cancel

            future = executor.submit(_run_child, goal, context, role, **task_kwargs)
            futures[future] = i

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
        # Ctrl+C：先传播中断给子代理（让它们在下轮迭代退出），
        # 取消未启动任务，shutdown(wait=False) 不阻塞，再向上抛。
        # Task K: 新增——set 所有 batch 子代理的 cancel_event，
        # 让正在跑的子代理在 LLM 调用前优雅退出（不只是依赖 parent.interrupt()）
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
    """创建并运行一个子代理。

    子代理是一个独立的 AIAgent 实例，有自己的：
    - 会话 ID
    - 迭代预算（默认 50 次）
    - 工具集（leaf 模式受限）
    - 工作目录

    子代理不继承父代理的对话历史。

    batch1-T4: 子 agent 注册到父 agent._children，支持中断传播。
    Task I: sidechain transcript 持久化（fail-open，不影响主流程）。
    """
    # 延迟导入避免循环
    from agent import AIAgent

    # === Task I: sidechain transcript 持久化初始化（fail-open）===
    _persistence_enabled = (kwargs.get("config") or {}).get(
        "delegation", {},
    ).get("subagent_persistence_enabled", True)
    _child_agent_id = None
    if _persistence_enabled:
        try:
            from agent.subagent_persistence import (
                generate_agent_id, write_metadata as _sp_write_meta,
            )
            _child_agent_id = generate_agent_id(
                parent_session_id=kwargs.get("session_id", ""),
            )
            _sp_write_meta(_child_agent_id, {
                "agent_type": kwargs.get("subagent_type", "general-purpose"),
                "parent_session_id": kwargs.get("session_id", ""),
                "description": f"{goal[:100]}",
                "status": "running",
                "created_at": time.time(),
            })
            # === CCAR13 Task 3: user 指令先进 transcript（轨迹开头）===
            # 真正中断的子代理 resume 时，原始指令是对话起点。
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

    # 从 kwargs 或 config 获取 LLM 配置
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")
    auth_token = kwargs.get("auth_token")
    model = kwargs.get("model")
    model_format = kwargs.get("model_format")

    if not (api_key or auth_token) or not model:
        # 从 config 读取
        try:
            from config import load_config
            config = load_config()

            # 子代理优先用轻量模型(default_haiku_model),省 token
            # 类 业界 模式:主对话用 opus,子代理用 haiku
            haiku_name = config.get("default_haiku_model", "")
            # 新模式:config["haiku_model"](llm 段注入的)
            haiku_cfg = config.get("haiku_model")
            if haiku_cfg:
                sub_cfg = haiku_cfg
            elif haiku_name and haiku_name in config.get("models", {}):
                # 老模式:config["models"][haiku_name]
                sub_cfg = config["models"][haiku_name]
            else:
                # fallback 到主模型兼容段
                sub_cfg = config.get("model", {})

            if not base_url:
                base_url = sub_cfg.get("base_url")
            if not model:
                model = sub_cfg.get("model") or sub_cfg.get("name")
            if not api_key:
                api_key = sub_cfg.get("api_key") or ""
            if not auth_token:
                auth_token = sub_cfg.get("auth_token") or ""
            # 向后兼容:api_key_env 指向环境变量(老 config.yaml 格式)
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

    # 构造子代理的 system prompt
    system_prompt = _build_child_system_prompt(goal, context, role)

    # 计算子代理的 spawn 深度（线程安全：参数传递，不写 os.environ）
    parent_agent = kwargs.get("agent_ref")
    if parent_agent is not None and hasattr(parent_agent, "spawn_depth"):
        parent_depth = parent_agent.spawn_depth
    else:
        parent_depth = int(kwargs.get("spawn_depth", 0))
    child_spawn_depth = parent_depth + 1

    # 工具集选择（对齐 Claude Code Agent subagent_type）
    # 提前解析 stype + custom_def，让 isolation=worktree 能在 worktree 创建分支前生效
    # （曾有时序 bug：isolated 在 custom_def 写入 kwargs 前读取，worktree 永不创建）
    stype = kwargs.get("subagent_type", "general-purpose")
    custom_def = None
    if stype not in ("general-purpose", "custom"):
        # 自定义子代理名：从 .md 定义加载
        from agent.agent_defs import get_agent_def
        custom_def = get_agent_def(stype)
        if custom_def is None:
            raise RuntimeError(
                f"未找到子代理定义: {stype}"
                f"（检查 ~/.OmniMate/agents/ 和 ./.omnimate/agents/）"
            )

    # 可选：隔离工作区（自定义 .md 定义 isolation=worktree 也开启）
    # 注意：不再用 os.chdir（进程级全局，ThreadPoolExecutor 并发子代理会互相踩 cwd）
    # 改用 workspace_cwd_context（contextvars.ContextVar，线程隔离），
    # 在下方 try/finally 外层包一层 with，让子代理内的工具读 get_workspace_cwd() 拿到自己的 worktree。
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

    # workspace_cwd_guard: 手动管理 ContextVar token（不重新缩进 200 行 try/finally）
    # 用 contextvars 替代 os.chdir：并发子代理（ThreadPoolExecutor）每线程独立 ContextVar，
    # 不会互相踩 cwd。token 在 finally 末尾 reset（退出 try 块 = 退出子代理 context）。
    from agent.workspace_context import _workspace_cwd
    _workspace_cwd_token = None
    if workspace_path is not None:
        _workspace_cwd_token = _workspace_cwd.set(str(workspace_path))

    # batch1-T4: 获取父 agent 引用（用于中断传播）
    parent_agent = kwargs.get("agent_ref")

    child = None
    # round3 D2 NEW: SUBAGENT_START/STOP 用的标志（False 默认，try 末尾设 True）
    _fork_success = False
    # round3 D2 NEW: 父 agent 的 hooks_registry（用于 SUBAGENT_START/STOP 审计）
    _parent_hooks = getattr(parent_agent, "hooks_registry", None) if parent_agent else None
    # === Task N NEW: CWD_CHANGED hook — worktree 切换时触发（fail-open）===
    if workspace_path is not None and _parent_hooks is not None:
        try:
            _parent_hooks.run_cwd_changed({
                "session_id": kwargs.get("session_id", ""),
                "old": str(Path.cwd()),  # 进程 cwd（近似）
                "new": str(workspace_path),
            })
        except Exception:
            pass  # fail-open
    try:
        # 工具集选择（对齐 Claude Code Agent subagent_type）
        # - 自定义名：custom_def 已在 worktree 分支前加载，按定义配置 toolsets/model/perm/maxTurns
        # - custom：用显式 enabled_toolsets
        # - general-purpose：按角色默认
        # Task J: permission_mode 优先级：
        #   ① custom_def.permission_mode（最高，自定义 .md 显式指定）
        #   ② kwargs["permission_mode"]（_delegate_async 注入的 autoDeny 或调用方显式传）
        #   ③ "default"（兜底）
        _injected_perm_mode = kwargs.get("permission_mode")
        if custom_def:
            # 按定义配置
            child_toolsets = custom_def.tools or (
                ["core"] if role == "orchestrator" else ["minimal"])
            # 自定义 .md 的 disallowedTools 覆盖父 config（非 union）
            disabled = custom_def.disallowed_tools or None
            child_model = custom_def.model or model
            # custom_def 显式指定优先于 kwargs 注入（async autoDeny）
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

        # 自定义子代理 system_prompt 覆盖（重建 system_prompt）
        if custom_def and custom_def.system_prompt:
            system_prompt = _build_child_system_prompt(
                goal, context, role, override=custom_def.system_prompt)

        # === Task N NEW: critical_reminder 拼 system_prompt 末尾 ===
        # cache 友好（system_prompt 一次构建缓存，不在每轮注入）
        if custom_def and custom_def.critical_reminder:
            system_prompt += (
                f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
            )
            logger.debug(
                "Task N: critical_reminder 已拼到子代理 system_prompt (%d 字符)",
                len(custom_def.critical_reminder),
            )

        # === Task C1: memory 字段 — child 独立记忆目录 ===
        # 默认继承父 store（None 则 child 自己新建默认 store）
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

        # === Task C1: skills 字段 — 预装技能正文到 system_prompt ===
        # 注意：必须在 system_prompt 最终确定之后、AIAgent 构造之前
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

        # disabled_tools 透传：AIAgent.__init__ 无此参数，走 config 透传
        # （get_tool_definitions 运行时从 self.config 读 disabled_tools，见 D2）
        #
        # CCAR5 Important 1 修复：合并两个来源的 disabled_tools（取并集保序去重）：
        #   ① custom_def.disallowed_tools（custom_def 路径，上面赋给 `disabled`）
        #   ② Task F 在 _delegate_async 注入的 kwargs["config"]["disabled_tools"]
        #      （async 黑名单兜底）—— 非 custom_def 路径下曾丢失，这里补上
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
            # 继承父 config（如有）再加 disabled_tools
            parent_cfg = kwargs.get("config")
            child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["disabled_tools"] = _all_disabled

        # === Task C1: mcp_servers 字段 — child 只暴露列出的 MCP server ===
        # 必须在 child_config 构造之后、AIAgent 构造之前
        if custom_def and custom_def.mcp_servers:
            if child_config is None:
                parent_cfg = kwargs.get("config")
                child_config = dict(parent_cfg) if isinstance(parent_cfg, dict) else {}
            child_config["mcp_server_filter"] = custom_def.mcp_servers

        # === Task H: fork 子代理路径（cache-identical 省 token）===
        # fork=True 时：子代理继承父 system prompt 字节 + 父对话前缀（最近 N 个 assistant turn），
        # 构造 cache-identical 前缀，prompt cache 命中省 token 50%+。
        # fail-open：fork 构造失败 fallback 到非 fork 路径（仅 log warning）
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
                    # 覆盖 system_prompt 为 fork 版（父字节 + fork marker）
                    system_prompt = build_forked_system_prompt(
                        parent_sysprompt, child_role=role,
                    )
                    # Task N Important fix: fork 覆盖 system_prompt 后再拼一次 critical_reminder
                    # （critical_reminder 常是安全提醒，fork 路径不能丢）
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )
                    # 构造初始 messages（父前缀 + directive）
                    child_initial_messages = build_forked_messages(
                        parent_messages=parent_messages,
                        parent_system_prompt=parent_sysprompt,
                        child_directive=f"{goal}\n上下文: {context}" if context else goal,
                        max_parent_turns=_max_turns,
                    )
                    logger.info(
                        "Task H: fork 子代理启用，继承 %d 条 messages",
                        len(child_initial_messages),
                    )
                except Exception as e:
                    # fail-open：fork 构造失败，回退到非 fork 路径
                    logger.warning(
                        "Task H: fork 构造失败，fallback 到非 fork 路径: %s", e,
                    )
                    child_initial_messages = None
                    # system_prompt 保留前面 _build_child_system_prompt 的结果
                    # 但如果上面 build_forked_system_prompt 已覆盖又出错，需要重建
                    system_prompt = _build_child_system_prompt(goal, context, role)
                    if custom_def and custom_def.system_prompt:
                        system_prompt = _build_child_system_prompt(
                            goal, context, role, override=custom_def.system_prompt)
                    # Task N Important fix: fallback 也要补 critical_reminder
                    if custom_def and custom_def.critical_reminder:
                        system_prompt += (
                            f"\n\n## CRITICAL REMINDER\n{custom_def.critical_reminder}"
                        )

        # === CCAR13 Task 3: 每轮 transcript 落盘（POST_LLM_CALL 程序式 hook）===
        # 机制选择（grep 结论）：子代理的 hooks_registry 不共享主 agent
        # （此前 _run_child 未传，child 拿 None）→ 新建独立空 HookRegistry
        # 注册程序式 hook，零污染主 agent 的 registry。
        # 语义：轨迹 = user 指令 + 每轮 assistant 文本；tool_calls / tool result
        # 不落盘（POST_LLM_CALL 只拿得到 LLM 响应；带 tool_calls 无配对 result
        # 会造孤儿消息 → API 400）。resume 时 initial_messages 是纯 user/assistant
        # 文本流，配对天然完整。
        # 已知耦合：走 AIAgent._run_post_llm_call_hook，受 config["hooks"]["enabled"]
        # 门控（默认 True）；用户显式关 hooks 会停轮级记录（只留 user 指令一条）。
        _child_hooks = None
        if _child_agent_id:
            try:
                from agent.hooks import HookRegistry

                def _extract_turn_text(response):
                    """从 LLM 响应提取 assistant 文本（None 安全，格式兼容）。"""
                    try:
                        msg = response.choices[0].message
                    except Exception:
                        return None
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        # Anthropic 风格 content blocks → 只拼 text 块
                        parts = [
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        content = "\n".join(p for p in parts if p)
                    return content or None

                def _on_llm_turn(response):
                    """每轮 LLM 响应后 append assistant 文本（fail-open）。"""
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
            # CCAR13 Task 3: 轮级 transcript 持久化（独立空 registry + POST_LLM_CALL
            # 程序式 hook，每轮 append；on_response 最终响应 append 已删避免双写）
            hooks_registry=_child_hooks,
            omit_project_memory=bool(custom_def.omit_claude_md) if custom_def else False,
        )

        # batch1-T4: 注册到父 agent._children（中断传播）
        if parent_agent is not None:
            try:
                parent_agent._children.append(child)
            except Exception:
                pass

        # === P1-10: pendingToolUseSummary ===
        # 长任务执行期间用 aux_llm 周期生成进度摘要，推到父 agent 的 stream_callback
        # 让前端知道"还在做什么"。aux_llm 不可用时降级为心跳。
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
            # round3 D2 NEW: SUBAGENT_START（child.chat 前触发，fail-open）
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

            # 运行子代理
            # Task D4 fix: AIAgent.chat 已改 async。_run_child 在独立线程里跑
            # （_delegate_sync / _delegate_async 均起 threading.Thread），无事件循环 → asyncio.run 驱动。
            # Task K: cancel_event 从 kwargs 透传到 child AIAgent.run_conversation，
            # 子代理主循环每轮检查 cancel_event.is_set() → 退出并返回 partial result
            import asyncio
            _cancel_event = kwargs.get("cancel_event")
            # === Task N NEW: initial_prompt 前置到首 user turn ===
            # slash 风格预处理（对齐 Claude Code Agent.initialPrompt 字段）
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

        # 06 NEW: 幻觉检测（在 summary_only 压缩前做，保留警告进摘要）
        # Round 1 fix: Path.cwd() 是进程级（=os.getcwd），并发子代理会踩。
        # 优先用 kwargs cwd，否则走线程局部的 get_workspace_cwd()。
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

        # summary_only：超长结果用 LLM 生成摘要，节省父代理 context
        summary_only = kwargs.get("summary_only", True)
        if summary_only and len(result) > 500:
            result = _summarize_child_result(result, child.llm_client, child.model)

        # round3 D2 NEW: 子代理成功标志（finally 里据此触发 SUBAGENT_STOP）
        _fork_success = True

        # === Task I: 标记子代理完成 ===
        if _child_agent_id:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "completed")
            except Exception:
                pass  # fail-open

        return result
    finally:
        # === Task I: 异常时标记 failed（_fork_success=False 表示没到 return）===
        if _child_agent_id and not _fork_success:
            try:
                from agent.subagent_persistence import mark_completed
                mark_completed(_child_agent_id, "failed")
            except Exception:
                pass  # fail-open

        # batch1-T4: 从父 agent._children 移除
        if parent_agent is not None and child is not None:
            try:
                if child in parent_agent._children:
                    parent_agent._children.remove(child)
            except Exception:
                pass
        # 恢复 workspace cwd context（替代 os.chdir）
        # ContextVar token reset 只影响当前线程，不会踩到其他并发子代理
        if _workspace_cwd_token is not None:
            try:
                _workspace_cwd.reset(_workspace_cwd_token)
            except Exception:
                pass
        if workspace_cleanup:
            # Task G: 智能清理 worktree（有改动保留，无改动清理）
            # config.delegation.worktree_always_cleanup=True → 旧行为（总是清理）
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

        # round3 D2 NEW: SUBAGENT_STOP（无论成功失败都触发，fail-open）
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


def _summarize_child_result(result: str, client, model: str) -> str:
    """用 LLM 把子代理结果总结成 300 字以内的摘要。

    摘要失败时返回原文（不阻塞委托流程）。

    改造说明（T_D1 fix）：call_with_retry 改 async 后，本函数保持同步接口
    （调用方 _run_child 在独立线程里跑，无事件循环），内部用 asyncio.run()
    驱动 async call_with_retry。
    """
    prompt = (
        "把以下子代理执行结果总结成 300 字以内的摘要，保留：\n"
        "1. 核心结论\n"
        "2. 关键发现和数据\n"
        "3. 重要的文件路径、命令、错误信息\n"
        "4. 待办事项\n\n"
        f"子代理结果：\n{result[:8000]}"
    )
    try:
        import asyncio
        from agent.llm_retry import call_with_retry
        response = asyncio.run(call_with_retry(
            client,  # child.llm_client（LLMClient 实例）
            [{"role": "user", "content": prompt}],
        ))
        summary = response.choices[0].message.content
        return f"[摘要] {summary}\n\n[完整结果 {len(result)} 字符已省略]"
    except Exception as e:
        logger.debug("子代理结果摘要失败，返回原文: %s", e)
        return result


def _build_child_system_prompt(goal: str, context: str, role: str, override: str = None) -> str:
    """构建子代理的 system prompt。

    override 非空时，base 用 override（自定义子代理 .md 的 system_prompt），
    仅追加上下文/约束/角色提示。None 时走默认构建（与历史行为一致）。
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

    # 自定义 override 也补一份通用约束（不含默认 goal/role 文本）
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
    """根据运行时状态改 subagent schema description（03）。

    让 LLM 看到当前剩余并发槽位，避免"试 spawn 被拒"浪费一轮。
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
# Task K: subagent_kill 工具（中断 async 子代理）
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
    """Task K: 中断 async 子代理。

    查 _async_tasks 注册表，set cancel_event 让子代理在 LLM 调用前退出。
    子代理主循环（AIAgent.run_conversation）每轮开头检查 cancel_event，
    触发就返回 _extract_partial_result()。

    fail-open：注册表查询/线程 join 异常不影响主流程（返错误 JSON）。
    """
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "task_id 不能为空",
            "error_type": "invalid_argument",
        }, ensure_ascii=False)

    # 检查 config 开关（config.delegation.async_kill_enabled）
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

        # 给子代理最多 2s 优雅退出（不等 thread.join() 完整跑完，只是软通知）
        # 注意：不在这里阻塞主流程，thread 的退出由 run_conversation 内部检查决定
        thread = info.get("thread")
        if thread is not None and thread.is_alive():
            # 不等 join——kill 工具本身应快速返回
            # 子代理在自己的线程里继续跑直到 cancel_event 检查生效
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
    """check_fn：config.delegation.async_kill_enabled 控制可见性。

    True（默认）→ 工具对 LLM 可见；False → 隐藏（check_fn 返 False）

    ⚠️ registry._check_fn_cached 调 fn() 不传参——必须用无参签名（跟其他 check_fn 一致）。
    """
    try:
        from agent.settings import load_settings
        cfg = load_settings() or {}
        delegation = cfg.get("delegation") or {}
        return bool(delegation.get("async_kill_enabled", True))
    except Exception:
        return True  # fail-open


# 注册到 core 工具集（让 resolve("core") 能找到）
registry.register(
    name="subagent",
    toolset="core",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
    isConcurrencySafe=False,  # 副作用：spawn 子 agent（重资源 + 改子任务状态），必须串行
)
# 兼容 alias：历史会话/记忆里的 delegate_task 调用仍可 dispatch 命中
# toolset="_compat"：dispatch 按名字命中（不依赖 toolset），但 resolve_toolset("core")
# 不会包含它——LLM 只看到 subagent，delegate_task 仅给老调用兜底
registry.register(
    name="delegate_task",
    toolset="_compat",
    schema=DELEGATE_TASK_SCHEMA,
    handler=_handle_delegate_task,
    schema_overrides_fn=_delegate_schema_overrides,
    emoji="🤝",
    override=True,
    isConcurrencySafe=False,  # alias of subagent，同样有副作用，必须串行
)
# Task K: subagent_kill 工具（中断 async 子代理）
registry.register(
    name="subagent_kill",
    toolset="core",
    schema=SUBAGENT_KILL_SCHEMA,
    handler=_handle_subagent_kill,
    check_fn=_subagent_kill_check_fn,
    emoji="🛑",
    isConcurrencySafe=False,  # 副作用：set cancel_event + 改注册表，串行
)
