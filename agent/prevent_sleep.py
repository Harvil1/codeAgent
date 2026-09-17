"""Windows 防休眠：agent 忙的时候不许电脑睡觉。

给谁用：主循环（agent/__init__.py）每轮开头调用——goal 驱动的
长任务或后台任务跑一半，机器突然休眠，醒来时网络断了、任务卡死，前功尽弃。
所以干活期间要按住"保持唤醒"的开关。

原理（大白话）：Windows 提供一个系统函数 SetThreadExecutionState，可以
让当前线程告诉系统"我需要机器醒着"（ES_SYSTEM_REQUIRED），并且持续生效
（ES_CONTINUOUS）。进程退出或显式恢复后，这个"按住不放"的状态自动失效。
本项目用 ctypes（Python 自带的调系统 DLL 的工具）直接调 kernel32 里的
这个函数，不装任何第三方包。

设计原则：
- 出任何问题都只返回 False、不抛异常——防休眠失败绝不能连累正常对话
  （fail-open：宁可机器睡了，也不能程序崩了）
- 不是 Windows 就什么都不做返回 False（调用方每轮都调，也不会刷日志报错）
- 用"引用计数"管理：谁需要醒着就 acquire 加一票，忙完 release 减一票，
  按原因（reason）记账，调试时能看清是"谁"还占着唤醒态
- atexit 兜底：程序退出前无论如何把电源状态还回去（防泄漏）
- ctypes 调系统函数必须显式声明返回值类型
  （restype），默认会当 32 位整数处理、把 64 位返回值截断；
  SetThreadExecutionState 返回 DWORD，返回 0 才表示失败——不显式声明
  就分不清"成功"和"被截断的失败"

用法：
    from agent import prevent_sleep
    prevent_sleep.acquire("busy")    # 加一票"我要醒着"
    prevent_sleep.release("busy")    # 减一票，全部归零时恢复系统默认

主循环的接法：每轮开头看状态——goal 激活或后台任务在跑就 acquire("busy")，
否则 release("busy")。
"""
import atexit
import logging
import sys

logger = logging.getLogger(__name__)

# Win32 电源状态常量（系统头文件 winbase.h 里定义的魔法数字）
_ES_CONTINUOUS = 0x80000000        # "持续生效"——按住这个状态不撒手，直到下次调用改它
_ES_SYSTEM_REQUIRED = 0x00000001   # "系统要醒着"——阻止休眠/待机

# 是不是 Windows（写法与 notifier.py 保持一致）
_IS_WIN = sys.platform == "win32"

# 引用计数账本：原因 → 该原因当前持有几票（调试时能看清是谁还占着唤醒态）
_reasons: dict = {}
# 当前是否已经把线程电源状态设成"持续保持唤醒"了（避免重复调系统函数）
_state_dirty = False

# kernel32 动态库的句柄（Windows 下加载；加载失败置 None，之后所有操作自动变 no-op）
_kernel32 = None
if _IS_WIN:
    try:
        import ctypes

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 返回值类型必须显式声明成 32 位无符号整数
        # （DWORD）；不声明的话 ctypes 默认按带符号 int 处理，返回值会变味，
        # 而"返回 0 = 失败"的判断就不可靠了
        _kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        _kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    except Exception as e:  # pragma: no cover - Windows 上 ctypes 加载失败极罕见
        logger.warning("kernel32 加载失败，防休眠 no-op: %s", e)
        _kernel32 = None


def acquire(reason: str) -> bool:
    """投一票"机器要醒着"：计数加一；从没人要到有人要的那一刻才真正调系统函数。

    多处可能同时需要机器醒着（比如 goal 在跑 + 后台任务在跑），用投票
    避免反复设置/取消电源状态互相打架——只在票数从 0 变正时设置一次。

    参数：
        reason: 谁在要（标识字符串，如 "busy"）；同一个 reason 多次 acquire
            就累加计数，对应地要 release 同样多次才能抵消

    返回：True = 当前处于唤醒态；False = 不是 Windows / 系统库不可用 /
        调用失败（不抛异常，静默降级）。
    """
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return False
    try:
        _reasons[reason] = _reasons.get(reason, 0) + 1
        if _state_dirty:
            # 已经有人按着唤醒态了，不用再调一次系统函数
            return True
        rc = _kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        if rc == 0:
            logger.warning("SetThreadExecutionState(唤醒) 失败")
            return False
        _state_dirty = True
        return True
    except Exception as e:
        logger.warning("prevent_sleep.acquire fail-open: %s", e)
        return False


def release(reason: str) -> bool:
    """撤回一票"机器要醒着"：计数减一；所有票都撤光时才恢复系统默认电源策略。

    与 acquire 配对的"还票"操作。只有账本彻底清空（没人再需要醒着）
    才把电源状态还回去——中途还有人占着就继续按住。

    参数：
        reason: 之前 acquire 时用的同一个标识（如 "busy"）

    返回：True = 操作完成（含"还有别人占着，继续保持唤醒"的情况）；
        False = 没持有过就 release（防御性拒绝，计数绝不减成负数）、
        不是 Windows、或调用失败。非 Windows 返回 False 只是"没干这件事"，
        不算错误——调用方每轮都调，不应该被日志刷屏。
    """
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return False
    try:
        cnt = _reasons.get(reason, 0)
        if cnt <= 0:
            # 防御：根本没持有还来还票（重复调/乱调），静默拒绝，不让计数变负
            return False
        if cnt > 1:
            _reasons[reason] = cnt - 1
            return True
        # 这个 reason 的票还清了，从账本里删掉
        del _reasons[reason]
        if _reasons:
            # 账上还有别人的票，继续按住唤醒态
            return True
        if not _state_dirty:
            return True
        rc = _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        if rc == 0:
            logger.warning("SetThreadExecutionState(恢复) 失败")
            return False
        _state_dirty = False
        return True
    except Exception as e:
        logger.warning("prevent_sleep.release fail-open: %s", e)
        return False


def is_active() -> bool:
    """看一眼现在还有没有人要机器醒着（账本里还有任何记账就返回 True）。"""
    return bool(_reasons)


def _release_all() -> None:
    """最后的保险：进程退出前无论如何把电源状态恢复成系统默认。

    正常流程靠 release 归零恢复；调用方忘了还票、或程序异常退出时，
    "按住唤醒"会一直生效（机器再也睡不着）。所以注册到 atexit——
    解释器关闭时自动清账。失败也只记日志，不影响退出。
    """
    global _state_dirty
    if not _IS_WIN or _kernel32 is None:
        return
    try:
        if _state_dirty:
            _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            _state_dirty = False
        _reasons.clear()
    except Exception as e:
        logger.warning("prevent_sleep._release_all fail-open: %s", e)


# 兜底注册：正常流程之外的退出路径（异常/解释器关闭）也走一次恢复
atexit.register(_release_all)


def _reset_for_test() -> None:
    """测试专用：清空投票账本和"已设置"标记。生产代码不要调。"""
    global _state_dirty
    _reasons.clear()
    _state_dirty = False
