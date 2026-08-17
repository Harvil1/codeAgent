"""AutonomousLifecycle 测试。"""
import time
from unittest.mock import MagicMock

import pytest

from agent.team.lifecycle import (
    AutonomousLifecycle, STATE_WORK, STATE_IDLE, STATE_SHUTDOWN,
)


def test_lifecycle_work_then_shutdown_no_messages():
    """无 inbox 消息 + 无任务 → IDLE 超时 → SHUTDOWN。"""
    work_fn = MagicMock(return_value="done")
    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.2,  # 短超时
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="hello")
    work_fn.assert_called_once_with("hello")
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_idle_picks_up_inbox_message():
    """IDLE 中 inbox 来消息 → 回 WORK。"""
    from agent.team.bus import TeamMessage
    calls = {"count": 0, "tasks": []}

    def work_fn(task):
        calls["tasks"].append(task)
        calls["count"] += 1
        return "ok"

    msg = TeamMessage(
        id="m1", from_="main", to="worker",
        type="request", content="next task",
        ts="2026-07-12T15:00:00", request_id=None,
    )
    inbox_states = [[], [msg], []]  # 第 1 次（初始）空，第 2 次有消息，第 3 次空
    state_idx = {"i": 0}

    def poll_inbox():
        i = state_idx["i"]
        state_idx["i"] += 1
        return inbox_states[i] if i < len(inbox_states) else []

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=poll_inbox,
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.2,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="initial")
    # 应该跑了 2 次（initial + next task）
    assert calls["count"] == 2
    assert "next task" in calls["tasks"]
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_work_exception_continues_to_idle():
    """work_fn 抛异常 → 进 IDLE 不崩。"""
    call_count = {"n": 0}

    def work_fn(task):
        call_count["n"] += 1
        raise RuntimeError("boom")

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.1,
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="x")
    assert call_count["n"] == 1  # 异常后进 IDLE，超时后 SHUTDOWN，不再调 work
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_multiple_work_cycles():
    """多次 WORK→IDLE→WORK 循环（inbox 给 2 条消息）。"""
    from agent.team.bus import TeamMessage
    work_count = {"n": 0}

    def work_fn(task):
        work_count["n"] += 1
        return "ok"

    msgs = [
        TeamMessage(id="m1", from_="main", to="w",
                    type="message", content="t1", ts="x", request_id=None),
        TeamMessage(id="m2", from_="main", to="w",
                    type="message", content="t2", ts="x", request_id=None),
    ]
    # inbox 序列：空 → m1 → m2 → 空 → 空...
    seq = [[], [msgs[0]], [msgs[1]], []]
    idx = {"i": 0}

    def poll_inbox():
        i = idx["i"]
        idx["i"] += 1
        return seq[i] if i < len(seq) else []

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=poll_inbox,
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.15,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="initial")
    # initial + t1 + t2 = 3 次工作
    assert work_count["n"] == 3


def test_lifecycle_on_shutdown_called():
    """SHUTDOWN 时调 on_shutdown_fn。"""
    called = {"x": False}
    def on_shutdown():
        called["x"] = True
    lifecycle = AutonomousLifecycle(
        work_fn=lambda t: None,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        on_shutdown_fn=on_shutdown,
        idle_timeout=0.05,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="x")
    assert called["x"] is True
    assert lifecycle.state == STATE_SHUTDOWN
