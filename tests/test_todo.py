"""TodoWrite 测试。"""

import json

import pytest

from agent.todo import TodoManager, get_todo_manager
from tools.registry import registry
from model_tools import ensure_tools_discovered

ensure_tools_discovered()


# ---------------------------------------------------------------------------
# TodoManager.write
# ---------------------------------------------------------------------------

def test_write_replaces_list():
    mgr = TodoManager()
    mgr.write([
        {"text": "A", "status": "pending"},
        {"text": "B", "status": "in_progress"},
    ])
    assert len(mgr.todos) == 2
    assert mgr.todos[0].text == "A"
    assert mgr.todos[1].status == "in_progress"


def test_write_resets_round_counter():
    mgr = TodoManager()
    mgr.write([{"text": "x", "status": "pending"}])
    for _ in range(5):
        mgr.increment_round()
    assert mgr.rounds_since_update == 5

    mgr.write([{"text": "y", "status": "pending"}])
    assert mgr.rounds_since_update == 0


def test_write_only_one_in_progress():
    mgr = TodoManager()
    result = mgr.write([
        {"text": "A", "status": "in_progress"},
        {"text": "B", "status": "in_progress"},
    ])
    assert result["success"] is False
    assert "1 个 in_progress" in result["error"]
    assert len(mgr.todos) == 0  # 失败时不更新


def test_write_invalid_status():
    mgr = TodoManager()
    result = mgr.write([
        {"text": "A", "status": "done"},  # 非法
    ])
    assert result["success"] is False
    assert "非法 status" in result["error"]


def test_write_returns_counts():
    mgr = TodoManager()
    result = mgr.write([
        {"text": "A", "status": "completed"},
        {"text": "B", "status": "in_progress"},
        {"text": "C", "status": "pending"},
    ])
    assert result["success"] is True
    assert result["count"] == 3
    assert result["completed"] == 1
    assert result["remaining"] == 2


# ---------------------------------------------------------------------------
# reminder 机制
# ---------------------------------------------------------------------------

def test_should_remind_after_3_rounds():
    mgr = TodoManager()
    mgr.write([{"text": "任务", "status": "in_progress"}])

    assert not mgr.should_remind()  # 0 轮
    mgr.increment_round()
    mgr.increment_round()
    assert not mgr.should_remind()  # 2 轮
    mgr.increment_round()
    assert mgr.should_remind()  # 3 轮


def test_no_remind_if_all_completed():
    mgr = TodoManager()
    mgr.write([{"text": "x", "status": "completed"}])
    for _ in range(5):
        mgr.increment_round()
    assert not mgr.should_remind()


def test_format_reminder():
    mgr = TodoManager()
    mgr.write([
        {"text": "做 A", "status": "completed"},
        {"text": "做 B", "status": "in_progress"},
        {"text": "做 C", "status": "pending"},
    ])
    for _ in range(3):
        mgr.increment_round()
    reminder = mgr.format_for_reminder()
    assert "todo_reminder" in reminder
    assert "做 A" in reminder
    assert "做 B" in reminder
    assert "3 轮未更新" in reminder


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    mgr = TodoManager()
    mgr.write([{"text": "x", "status": "pending"}])
    mgr.increment_round()
    mgr.reset()
    assert mgr.todos == []
    assert mgr.rounds_since_update == 0


# ---------------------------------------------------------------------------
# todo_write 工具集成
# ---------------------------------------------------------------------------

def test_todo_write_tool_basic():
    result = registry.dispatch(
        "todo_write",
        {"items": [
            {"text": "步骤 1", "status": "in_progress"},
            {"text": "步骤 2", "status": "pending"},
        ]},
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["count"] == 2


def test_todo_write_tool_rejects_multiple_in_progress():
    result = registry.dispatch(
        "todo_write",
        {"items": [
            {"text": "A", "status": "in_progress"},
            {"text": "B", "status": "in_progress"},
        ]},
    )
    data = json.loads(result)
    assert data["success"] is False


def test_todo_write_tool_replaces_list():
    mgr = get_todo_manager()
    mgr.reset()

    registry.dispatch("todo_write", {"items": [
        {"text": "旧任务 1", "status": "pending"},
        {"text": "旧任务 2", "status": "pending"},
    ]})
    assert len(mgr.todos) == 2

    registry.dispatch("todo_write", {"items": [
        {"text": "新任务", "status": "in_progress"},
    ]})
    assert len(mgr.todos) == 1  # 替换，不是追加
    assert mgr.todos[0].text == "新任务"


# ---------------------------------------------------------------------------
# AIAgent 集成（todo_manager 属性存在）
# ---------------------------------------------------------------------------

def test_agent_has_todo_manager():
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[])
    assert agent.todo_manager is not None
