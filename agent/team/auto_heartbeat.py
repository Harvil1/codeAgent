"""自动心跳：工人只要还在干活，就定期给绑定的任务「打卡报平安」。

打比方：快递员每送一单就给调度台发个定位，调度台看他一直有动静就知道
他没出事。这里的「打卡」是更新 task.last_heartbeat_at 字段。

为什么需要：工人（spawned worker）跑长任务时，调度方的看门狗
（watchdog）靠这个字段判断「这工人还活着吗」。OmniMate 目前还没有
watchdog，但字段先维护起来，将来接上就能直接用。

流程（每次工具调用结束时被 POST_TOOL_USE hook 触发）：
  读环境变量 OMNIMATE_KANBAN_TASK（工牌）→
    没设（主 agent / 老式调用）→ 什么都不做
    已设 → 节流（每个进程至少隔 60 秒才打一次卡）→ TaskStore.heartbeat(tid)

失败一律静默（只记 debug 日志），绝不能因为打卡失败拖垮 agent 主循环。

Hook 签名约定：fn(tool_name, args, result) -> Optional[str]
本 hook 只干副作用的活，永远返回 None（不会替换工具结果）。
"""
import logging
import time

logger = logging.getLogger(__name__)


_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_last_attempt: float = 0.0


def maybe_heartbeat() -> bool:
    """尽力打一次卡。成功写入心跳返回 True，跳过或失败返回 False。

    为什么「尽力」：这只是个报平安的辅助动作，任何失败都不值得让
    主循环中断，所以内部全兜住、绝不抛异常——调用方（hook）不用包 try/except。

    直接用 TaskStore 全局单例（spawned worker 和主 agent 共用同一个
    agent_home，单例拿到的就是对的库）。

    返回：True=真的写了心跳；False=没戴工牌 / 节流窗口内 / 出错。
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
    """测试专用：把节流计时器归零。

    背景：模块级变量记录着上次打卡时间，不重置的话连续测试会被 60 秒
    节流挡住。无返回值。
    """
    global _last_attempt
    _last_attempt = 0.0


def _post_tool_use_hook(tool_name: str, args: dict, result: str):
    """POST_TOOL_USE hook：每次工具调用结束后被叫一次，触发打卡。

    参数：
        tool_name：刚执行完的工具名（本 hook 不用它）
        args：那次工具调用的参数（不用）
        result：工具返回结果（不用）

    返回：永远 None——本 hook 只干副作用的活（打卡），不会替换工具结果。
    双保险说明：异常本会被 HookRegistry 吞掉当 None 处理，而
    maybe_heartbeat 内部自己也全 try/except 了，两层兜底。
    """
    maybe_heartbeat()
    return None


def register(hooks_registry) -> None:
    """把上面的打卡 hook 挂到 hook 注册表上。

    参数：
        hooks_registry：agent/hooks.py 的 HookRegistry 实例；
        传 None 表示 hooks 系统没启用，直接什么都不做。

    无返回值。
    """
    if hooks_registry is None:
        return
    hooks_registry.register_post_tool_use(
        _post_tool_use_hook, name="auto_heartbeat",
    )
