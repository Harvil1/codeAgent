"""自动心跳桥：runtime 活动每 60s 自动 bump task.last_heartbeat_at。

防 spawned worker 跑长任务时 dispatcher watchdog 误回收（HarvilAgent 暂时
没有 watchdog，但字段值得维护，未来 dispatcher 接入即可用）。

逻辑：
  POST_TOOL_USE hook 触发（每次工具调用结束）→
    读 HARVIL_KANBAN_TASK env →
      未设（主 agent / legacy）→ no-op
      已设 → rate-limit（60s/进程）→ TaskStore.heartbeat(tid)

所有失败静默（log debug），不能影响 agent 主循环。

Hook 签名约定：fn(tool_name, args, result) -> Optional[str]
本 hook 是 side-effect only，永远返回 None（不替换 result）。
"""
import logging
import time

logger = logging.getLogger(__name__)


_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_last_attempt: float = 0.0


def maybe_heartbeat() -> bool:
    """Best-effort 自动心跳。返回 True 表示真的写了 heartbeat，False 表示跳过/失败。

    不会抛异常——调用方（hook）不用 try/except。
    使用 TaskStore 全局单例（spawned worker 的 agent_home 与主 agent 一致）。
    """
    global _last_attempt
    try:
        from agent.team.task_binding import get_bound_task_id
        tid = get_bound_task_id()
        if not tid:
            return False
        now = time.monotonic()
        if (now - _last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return False
        _last_attempt = now
        from agent.task_store import get_task_store
        store = get_task_store()
        result = store.heartbeat(tid)
        return result is not None
    except Exception:
        logger.debug("auto-heartbeat failed", exc_info=True)
        return False


def reset_for_test() -> None:
    """测试用：重置 rate-limit 计时。"""
    global _last_attempt
    _last_attempt = 0.0


def _post_tool_use_hook(tool_name: str, args: dict, result: str):
    """POST_TOOL_USE hook 签名：fn(tool_name, args, result) -> Optional[str]。

    本 hook 是 side-effect only，永远返回 None（不替换 result）。
    异常被 HookRegistry 吞掉（视为 None），maybe_heartbeat 内部已 try/except 是双保险。
    """
    maybe_heartbeat()
    return None


def register(hooks_registry) -> None:
    """注册到 HookRegistry。

    hooks_registry: agent/hooks.py 的 HookRegistry 实例。
    传入 None 时 no-op（hooks 系统未启用）。
    """
    if hooks_registry is None:
        return
    hooks_registry.register_post_tool_use(
        _post_tool_use_hook, name="auto_heartbeat",
    )
