"""preventSleep（Windows ctypes 防休眠）单元 + 接线测试。

覆盖：
1. 非 Windows no-op（_IS_WIN=False 时 acquire/release 返回 False）
2. Windows mock kernel32 调用序列（acquire → 唤醒态 → release → 恢复）
3. 引用计数两个 reason 独立
4. atexit 注册 + restype 显式声明
5. 主循环接线（goal active → acquire("busy")；闲 → release）
6. 重复 release 不下穿到负数
"""
import ctypes
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import prevent_sleep

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_AWAKE = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED


@pytest.fixture(autouse=True)
def _clean_state():
    """每个测试前后清空模块状态，防止真实系统调用/跨测试泄漏。"""
    prevent_sleep._reset_for_test()
    saved = (prevent_sleep._IS_WIN, prevent_sleep._kernel32)
    yield
    prevent_sleep._reset_for_test()
    prevent_sleep._IS_WIN, prevent_sleep._kernel32 = saved


def _mock_kernel32():
    """mock kernel32：SetThreadExecutionState 返回非 0（成功）。"""
    k = MagicMock()
    k.SetThreadExecutionState.return_value = 1  # DWORD 非 0 = 成功
    return k


# ---------------------------------------------------------------------------
# 1. 非 Windows no-op
# ---------------------------------------------------------------------------

def test_non_windows_noop():
    """非 Windows：acquire/release 返回 False，is_active 恒 False，无系统调用。"""
    prevent_sleep._IS_WIN = False
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k
    assert prevent_sleep.acquire("busy") is False
    assert prevent_sleep.release("busy") is False
    assert prevent_sleep.is_active() is False
    k.SetThreadExecutionState.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Windows mock kernel32：调用序列
# ---------------------------------------------------------------------------

def test_acquire_release_call_sequence():
    """acquire → ES_CONTINUOUS|ES_SYSTEM_REQUIRED；release 归零 → ES_CONTINUOUS。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k

    assert prevent_sleep.acquire("busy") is True
    assert prevent_sleep.is_active() is True
    k.SetThreadExecutionState.assert_called_once_with(_AWAKE)

    assert prevent_sleep.release("busy") is True
    assert prevent_sleep.is_active() is False
    # 归零恢复：第二次调用应是 ES_CONTINUOUS
    assert k.SetThreadExecutionState.call_count == 2
    assert k.SetThreadExecutionState.call_args_list[1] == ((_ES_CONTINUOUS,),)


def test_repeat_acquire_same_reason_no_extra_syscall():
    """同 reason 二次 acquire（计数 2）不重复调系统 API。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k

    prevent_sleep.acquire("busy")
    prevent_sleep.acquire("busy")
    k.SetThreadExecutionState.assert_called_once_with(_AWAKE)
    # 第一次 release 只减计数，不恢复
    prevent_sleep.release("busy")
    assert k.SetThreadExecutionState.call_count == 1
    assert prevent_sleep.is_active() is True
    # 第二次 release 才归零恢复
    prevent_sleep.release("busy")
    assert k.SetThreadExecutionState.call_count == 2
    assert prevent_sleep.is_active() is False


def test_syscall_failure_fail_open():
    """SetThreadExecutionState 返回 0（失败）或抛异常：fail-open 返回 False 不抛。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    k.SetThreadExecutionState.return_value = 0  # DWORD 0 = 失败
    prevent_sleep._kernel32 = k
    assert prevent_sleep.acquire("busy") is False

    # syscall 抛异常：acquire 路径 fail-open 不抛
    k2 = _mock_kernel32()
    k2.SetThreadExecutionState.side_effect = OSError("boom")
    prevent_sleep._kernel32 = k2
    assert prevent_sleep.acquire("x") is False

    # release 路径异常同样 fail-open：先构造 dirty 态（syscall 成功），
    # 再换成会抛异常的 kernel32，release 触发恢复调用时炸但不外抛
    prevent_sleep._reset_for_test()
    k3 = _mock_kernel32()
    prevent_sleep._kernel32 = k3
    assert prevent_sleep.acquire("a") is True  # dirty=True
    k4 = _mock_kernel32()
    k4.SetThreadExecutionState.side_effect = OSError("boom")
    prevent_sleep._kernel32 = k4
    assert prevent_sleep.release("a") is False


def test_release_all_restores():
    """atexit 兜底 _release_all：有持有态时恢复 ES_CONTINUOUS 并清计数。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k
    prevent_sleep.acquire("a")
    prevent_sleep.acquire("b")
    prevent_sleep._release_all()
    assert prevent_sleep.is_active() is False
    assert k.SetThreadExecutionState.call_args_list[-1] == ((_ES_CONTINUOUS,),)


# ---------------------------------------------------------------------------
# 3. 引用计数两个 reason 独立
# ---------------------------------------------------------------------------

def test_two_reasons_independent():
    """a acquire + b acquire + a release → 仍 active；b release → 归零恢复。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k

    prevent_sleep.acquire("a")
    prevent_sleep.acquire("b")
    assert prevent_sleep.is_active() is True
    assert prevent_sleep._reasons == {"a": 1, "b": 1}

    prevent_sleep.release("a")
    assert prevent_sleep.is_active() is True, "b 还占着，不应恢复"
    assert k.SetThreadExecutionState.call_count == 1, "未归零不应恢复"

    prevent_sleep.release("b")
    assert prevent_sleep.is_active() is False
    assert k.SetThreadExecutionState.call_count == 2
    assert k.SetThreadExecutionState.call_args_list[1] == ((_ES_CONTINUOUS,),)


# ---------------------------------------------------------------------------
# 4. atexit 注册 + restype 显式声明
# ---------------------------------------------------------------------------

def test_atexit_registered_on_module_load():
    """模块加载时 atexit.register 被调（兜底释放）。"""
    with patch("atexit.register") as m_reg:
        importlib.reload(prevent_sleep)
        m_reg.assert_called_once_with(prevent_sleep._release_all)
    # 恢复：真实 reload 重新注册（防止后续退出路径测试失真）
    importlib.reload(prevent_sleep)


def test_restype_explicitly_set():
    """SetThreadExecutionState.restype 必须显式 c_uint32（不设会被当 int 截断指针）。"""
    with patch("ctypes.WinDLL") as m_win:
        m_kernel = MagicMock()
        m_win.return_value = m_kernel
        importlib.reload(prevent_sleep)
        assert (m_kernel.SetThreadExecutionState.restype
                is ctypes.c_uint32), "restype 未显式声明为 c_uint32"
        assert list(m_kernel.SetThreadExecutionState.argtypes) == [
            ctypes.c_uint32
        ], "argtypes 未显式声明"
    importlib.reload(prevent_sleep)


# ---------------------------------------------------------------------------
# 5. 重复 release 防御（不下穿负数）
# ---------------------------------------------------------------------------

def test_repeated_release_never_negative():
    """未持有就 release：返回 False，计数不出现负数。"""
    prevent_sleep._IS_WIN = True
    k = _mock_kernel32()
    prevent_sleep._kernel32 = k

    assert prevent_sleep.release("busy") is False  # 从未 acquire
    assert prevent_sleep._reasons == {}
    prevent_sleep.acquire("busy")
    prevent_sleep.release("busy")
    assert prevent_sleep.release("busy") is False  # 已归零再 release
    assert prevent_sleep._reasons == {}
    assert prevent_sleep.is_active() is False


# ---------------------------------------------------------------------------
# 6. 主循环接线（goal active → acquire("busy")；闲 → 不调）
# ---------------------------------------------------------------------------

def _make_agent():
    from agent import AIAgent
    return AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home="/tmp/fake",
    )


def _mock_llm(text: str):
    m = MagicMock()
    resp = MagicMock()
    resp.choices = [
        MagicMock(message=MagicMock(content=text, tool_calls=None),
                  finish_reason="stop")
    ]
    m.chat_completions = AsyncMock(return_value=resp)
    return m


async def test_wiring_goal_active_acquires():
    """goal active 时主循环调 prevent_sleep.acquire("busy")。"""
    agent = _make_agent()
    agent.llm_client = _mock_llm("ok")
    goal = MagicMock()
    goal.status = "active"
    agent.set_goal_state(goal)

    with patch.object(prevent_sleep, "acquire", MagicMock()) as m_acq, \
            patch.object(prevent_sleep, "release", MagicMock()) as m_rel:
        await agent.run_conversation("hello")
        m_acq.assert_called_once_with("busy")
        m_rel.assert_not_called()
    assert agent._prevent_sleep_held is True


async def test_wiring_idle_never_acquires():
    """无 goal 无 bg：acquire/release 都不调。"""
    agent = _make_agent()
    agent.llm_client = _mock_llm("ok")

    with patch.object(prevent_sleep, "acquire", MagicMock()) as m_acq, \
            patch.object(prevent_sleep, "release", MagicMock()) as m_rel:
        await agent.run_conversation("hello")
        m_acq.assert_not_called()
        m_rel.assert_not_called()


async def test_wiring_busy_to_idle_releases():
    """先忙（acquire）后闲（goal paused）：release 被调，标志回落。"""
    agent = _make_agent()
    agent.llm_client = _mock_llm("ok")

    # 第一轮：goal active → acquire
    goal = MagicMock()
    goal.status = "active"
    agent.set_goal_state(goal)
    with patch.object(prevent_sleep, "acquire", MagicMock()) as m_acq:
        await agent.run_conversation("hello")
        m_acq.assert_called_once_with("busy")
    assert agent._prevent_sleep_held is True

    # 第二轮：goal paused → release
    goal.status = "paused"
    with patch.object(prevent_sleep, "release", MagicMock()) as m_rel:
        await agent.run_conversation("hello")
        m_rel.assert_called_once_with("busy")
    assert agent._prevent_sleep_held is False


async def test_wiring_config_disabled():
    """config security.prevent_sleep=False：goal active 也不 acquire。"""
    agent = _make_agent()
    agent.llm_client = _mock_llm("ok")
    agent.config["security"] = {"prevent_sleep": False}
    goal = MagicMock()
    goal.status = "active"
    agent.set_goal_state(goal)

    with patch.object(prevent_sleep, "acquire", MagicMock()) as m_acq:
        await agent.run_conversation("hello")
        m_acq.assert_not_called()


async def test_wiring_bg_running_acquires():
    """bg 任务 running 时也 acquire（无需 goal）。"""
    agent = _make_agent()
    agent.llm_client = _mock_llm("ok")

    bg = MagicMock()
    running_task = MagicMock()
    running_task.status = "running"
    bg.list_tasks.return_value = [running_task]
    agent.bg_manager = bg

    with patch.object(prevent_sleep, "acquire", MagicMock()) as m_acq:
        await agent.run_conversation("hello")
        m_acq.assert_called_once_with("busy")
    assert agent._prevent_sleep_held is True
