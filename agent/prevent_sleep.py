"""Windows 防休眠（CCAR15 Task 5，ctypes 零依赖）。

对齐 CCB preventSleep：goal 循环 / 后台任务运行中，系统不能进休眠
（长任务跑一半机器睡了，醒来时网络断、任务僵死）。

机制：kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
—— 线程级电源状态覆盖，进程退出或恢复 ES_CONTINUOUS 后失效。

设计原则：
- 零第三方依赖（ctypes 直调 kernel32）
- fail-open（任何异常返回 False 不抛——防休眠失败绝不影响主对话）
- 非 Windows no-op（返回 False，调用方每轮调也不会刷日志/报错）
- 引用计数按 reason 记录（dict，便于调试谁占着唤醒态）
- atexit 兜底释放（正常退出路径之外的最后保险）
- CCAR12 教训：ctypes restype 必须显式声明（默认 c_int 会截断返回值；
  SetThreadExecutionState 返回 DWORD，返回 0 表示失败）

使用：
    from agent import prevent_sleep
    prevent_sleep.acquire("busy")    # +1
    prevent_sleep.release("busy")    # -1，全部归零时恢复系统默认

主循环接线（agent/__init__.py:run_conversation 每轮开头）：
    goal active 或 bg running 时 acquire("busy")，否则 release("busy")
"""
import atexit
import logging
import sys

logger = logging.getLogger(__name__)

# Win32 电源状态常量（winbase.h）
_ES_CONTINUOUS = 0x80000000        # 状态持续生效直到下次调用
_ES_SYSTEM_REQUIRED = 0x00000001   # 系统保持唤醒（阻止休眠/待机）

# 平台判断（对齐 notifier.py 的 sys.platform 写法）
_IS_WIN = sys.platform == "win32"

# 引用计数：reason -> 持有次数（调试可见"谁占着唤醒态"）
_reasons: dict = {}
# 当前是否已把线程电源状态置为 CONTINUOUS|SYSTEM_REQUIRED
_state_dirty = False

# kernel32 句柄（Windows 下初始化；失败置 None → 全部 no-op）
_kernel32 = None
if _IS_WIN:
    try:
        import ctypes

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # CCAR12 教训：restype 必须显式（返回 DWORD；0 = 失败）
        _kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        _kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    except Exception as e:  # pragma: no cover - Windows 上 ctypes 加载失败极罕见
        logger.warning("kernel32 加载失败，防休眠 no-op: %s", e)
        _kernel32 = None


def acquire(reason: str) -> bool:
    """引用计数 +1；首次（0→正）时置线程为持续唤醒。

    Args:
        reason: 持有原因标识（如 "busy"），同 reason 多次 acquire 累加计数。

    Returns:
        bool: True 表示当前处于唤醒态；False 表示非 Windows / ctypes 不可用 /
            调用失败（fail-open 不抛）。
    """
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return False
    try:
        _reasons[reason] = _reasons.get(reason, 0) + 1
        if _state_dirty:
            # 已处于唤醒态（还有别的 reason 占着），无需重复调系统 API
            return True
        rc = _kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        if rc == 0:
            logger.debug("SetThreadExecutionState(唤醒) 失败")
            return False
        _state_dirty = True
        return True
    except Exception as e:
        logger.debug("prevent_sleep.acquire fail-open: %s", e)
        return False


def release(reason: str) -> bool:
    """引用计数 -1；全部归零时恢复 ES_CONTINUOUS（系统默认电源策略）。

    重复 release（计数已为 0）是 no-op 返回 False，不下穿到负数（防御）。
    非 Windows 返回 False 但不算错误（调用方每轮调，不该刷日志）。
    """
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return False
    try:
        cnt = _reasons.get(reason, 0)
        if cnt <= 0:
            # 防御：没持有就 release（重复/越界），静默拒绝
            return False
        if cnt > 1:
            _reasons[reason] = cnt - 1
            return True
        # 该 reason 归零
        del _reasons[reason]
        if _reasons:
            # 还有别的 reason 占着，保持唤醒
            return True
        if not _state_dirty:
            return True
        rc = _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        if rc == 0:
            logger.debug("SetThreadExecutionState(恢复) 失败")
            return False
        _state_dirty = False
        return True
    except Exception as e:
        logger.debug("prevent_sleep.release fail-open: %s", e)
        return False


def is_active() -> bool:
    """当前是否有任何 reason 持有唤醒态。"""
    return bool(_reasons)


def _release_all() -> None:
    """atexit 兜底：进程退出前恢复系统默认电源状态（fail-open）。"""
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return
    try:
        if _state_dirty:
            _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            _state_dirty = False
        _reasons.clear()
    except Exception as e:
        logger.debug("prevent_sleep._release_all fail-open: %s", e)


# 正常退出路径之外的兜底（异常退出/解释器关闭）
atexit.register(_release_all)


def _reset_for_test() -> None:
    """测试专用：清空引用计数与唤醒态标记。生产代码勿调。"""
    global _state_dirty
    _reasons.clear()
    _state_dirty = False
