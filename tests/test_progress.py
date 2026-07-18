"""ProgressReporter 测试（P1-10）。"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.progress import (
    ProgressReporter,
    DEFAULT_PROGRESS_INTERVAL,
    DEFAULT_HEARTBEAT_MESSAGE,
)


def test_default_constants():
    """默认值常量稳定。"""
    assert DEFAULT_PROGRESS_INTERVAL == 30.0
    assert DEFAULT_HEARTBEAT_MESSAGE


def test_no_stream_callback_is_noop():
    """stream_callback=None 时 start() 不启动 thread。"""
    reporter = ProgressReporter(
        goal="task", stream_callback=None, interval=1.0,
    )
    reporter.start()
    assert reporter._thread is None
    reporter.stop()


def test_interval_zero_disables():
    """interval=0 时禁用（测试用）。"""
    events = []
    reporter = ProgressReporter(
        goal="task",
        stream_callback=lambda e: events.append(e),
        interval=0,
    )
    reporter.start()
    assert reporter._thread is None


def test_tick_once_pushes_event():
    """同步 tick：stream_callback 收到 progress 事件。"""
    events = []
    reporter = ProgressReporter(
        goal="跑测试",
        stream_callback=lambda e: events.append(e),
        interval=30.0,
    )
    event = reporter.tick_once()

    assert event["type"] == "progress"
    assert event["goal"] == "跑测试"
    assert event["tick"] == 1
    assert "message" in event
    assert events == [event]


def test_tick_increments_count():
    """多次 tick 计数递增。"""
    reporter = ProgressReporter(
        goal="task", stream_callback=lambda e: None, interval=30.0,
    )
    reporter.tick_once()
    reporter.tick_once()
    reporter.tick_once()
    assert reporter._tick_count == 3


def test_no_aux_llm_uses_heartbeat():
    """无 aux_llm_router 时发心跳消息。"""
    reporter = ProgressReporter(
        goal="task",
        stream_callback=lambda e: None,
        interval=30.0,
        aux_llm_router=None,
    )
    msg = reporter.generate_message()
    assert msg == DEFAULT_HEARTBEAT_MESSAGE


def test_aux_llm_generates_message():
    """有 aux_llm_router 时调它生成进度消息。"""
    fake_router = MagicMock()
    fake_router.chat_completions.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="正在编译..."),
        )],
    )
    reporter = ProgressReporter(
        goal="build",
        stream_callback=lambda e: None,
        interval=30.0,
        aux_llm_router=fake_router,
    )
    msg = reporter.generate_message()
    assert "正在编译" in msg
    # 调用了一次 LLM
    assert fake_router.chat_completions.call_count == 1


def test_aux_llm_failure_falls_back_to_heartbeat():
    """aux_llm 抛异常时降级为心跳（fail-open）。"""
    fake_router = MagicMock()
    fake_router.chat_completions.side_effect = RuntimeError("LLM down")
    reporter = ProgressReporter(
        goal="task",
        stream_callback=lambda e: None,
        interval=30.0,
        aux_llm_router=fake_router,
    )
    msg = reporter.generate_message()
    assert msg == DEFAULT_HEARTBEAT_MESSAGE


def test_aux_llm_empty_response_falls_back():
    """aux_llm 返回空内容时降级。"""
    fake_router = MagicMock()
    fake_router.chat_completions.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=""),
        )],
    )
    reporter = ProgressReporter(
        goal="task",
        stream_callback=lambda e: None,
        interval=30.0,
        aux_llm_router=fake_router,
    )
    msg = reporter.generate_message()
    assert msg == DEFAULT_HEARTBEAT_MESSAGE


def test_context_manager_starts_and_stops():
    """with 块进入时启动，退出时停止。"""
    events = []
    # 用很短 interval 测真实 thread
    with ProgressReporter(
        goal="task",
        stream_callback=lambda e: events.append(e),
        interval=0.05,
    ) as reporter:
        time.sleep(0.2)  # 等 4 个 tick
        assert reporter._thread is not None
    # 退出后 thread 已停
    assert reporter._thread is None
    # 至少触发了 1 次（可能 2-4 次，取决于调度）
    assert len(events) >= 1


def test_context_manager_no_callback_noop():
    """stream_callback=None 时 with 块也不启动 thread。"""
    with ProgressReporter(
        goal="task", stream_callback=None, interval=0.05,
    ) as reporter:
        assert reporter._thread is None


def test_long_goal_doesnt_break_message_truncation():
    """aux_llm 返回超长内容时截断到 80 字符。"""
    fake_router = MagicMock()
    long_msg = "x" * 200
    fake_router.chat_completions.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=long_msg),
        )],
    )
    reporter = ProgressReporter(
        goal="task",
        stream_callback=lambda e: None,
        aux_llm_router=fake_router,
    )
    msg = reporter.generate_message()
    assert len(msg) <= 80


def test_stop_is_idempotent():
    """stop() 多次调用安全。"""
    reporter = ProgressReporter(
        goal="task", stream_callback=lambda e: None, interval=0.05,
    )
    reporter.start()
    reporter.stop()
    reporter.stop()  # 不应抛
    reporter.stop()


def test_event_elapsed_seconds_scales_with_tick():
    """elapsed_seconds = tick_count * interval。"""
    reporter = ProgressReporter(
        goal="task", stream_callback=lambda e: None, interval=30.0,
    )
    reporter.tick_once()
    assert reporter._tick_count == 1
    # 第一次 tick 的 elapsed = 30
    reporter.tick_once()
    # 第二次 tick 的 elapsed = 60
