# tests/test_goal.py
"""GoalState 状态机 + 持久化的单元测试（CCAR8 Task 10）。

仅覆盖状态机本身，主循环集成在 Task 11。
"""
import json
import logging
from pathlib import Path

from agent.goal import GoalState


def test_goal_state_initial():
    g = GoalState(objective="测试目标")
    assert g.status == "active"
    assert g.iteration_count == 0
    assert g.token_budget == 0
    assert g.pause_reason is None
    assert g.goal_id  # 非空


def test_goal_state_save_and_load(tmp_path: Path):
    g = GoalState(objective="完成 X", token_budget_limit=100_000)
    g.iteration_count = 3
    g.token_budget = 5000
    g.task_ids = ["task_1", "task_2"]
    g.save(tmp_path / ".goal" / "current.json")

    loaded = GoalState.load(tmp_path / ".goal" / "current.json")
    assert loaded is not None
    assert loaded.objective == "完成 X"
    assert loaded.iteration_count == 3
    assert loaded.token_budget == 5000
    assert loaded.token_budget_limit == 100_000
    assert loaded.task_ids == ["task_1", "task_2"]
    assert loaded.status == "active"


def test_goal_load_missing_file_returns_none(tmp_path: Path):
    """文件不存在时返回 None（调用方决定是否新建）。"""
    loaded = GoalState.load(tmp_path / "nonexistent.json")
    assert loaded is None


def test_goal_load_corrupted_returns_none(tmp_path: Path, caplog):
    """损坏文件返回 None + log warning。"""
    path = tmp_path / "bad.json"
    path.write_text("not json {{{", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = GoalState.load(path)
    assert loaded is None


def test_goal_save_atomic(tmp_path: Path):
    """save 用 tmp + rename 原子写。"""
    g = GoalState(objective="x")
    path = tmp_path / ".goal" / "current.json"
    g.save(path)
    assert path.exists()
    # 不应该有残留 tmp 文件
    # （注意：Path 对象 bool() 恒为 True，必须用 .exists()）
    assert not (tmp_path / ".goal" / "current.json.tmp").exists()


def test_goal_pause_and_resume():
    g = GoalState(objective="x")
    g.pause(reason="manual")
    assert g.status == "paused"
    assert g.pause_reason == "manual"
    g.resume()
    assert g.status == "active"
    assert g.pause_reason is None


def test_goal_complete():
    g = GoalState(objective="x")
    g.complete()
    assert g.status == "completed"


def test_goal_cancel():
    g = GoalState(objective="x")
    g.cancel()
    assert g.status == "cancelled"


def test_goal_evaluate_continue_no_conditions():
    """无 budget 限制 + 无完成条件 → continue。"""
    g = GoalState(objective="x")
    decision = g.evaluate_after_turn(tokens_used=100)
    assert decision == "continue"
    assert g.iteration_count == 1
    assert g.token_budget == 100


def test_goal_evaluate_pause_on_budget():
    """超 budget → pause。"""
    g = GoalState(objective="x", token_budget_limit=1000)
    decision = g.evaluate_after_turn(tokens_used=1500)
    assert decision == "pause"
    assert g.pause_reason == "budget_exceeded"


def test_goal_evaluate_complete_when_all_tasks_done():
    """所有关联任务完成 → complete（需要注入完成状态）。"""
    g = GoalState(objective="x")
    decision = g.evaluate_after_turn(tokens_used=100, all_tasks_done=True)
    assert decision == "complete"


def test_goal_pause_on_network():
    """网络异常 pause。"""
    g = GoalState(objective="x")
    g.pause(reason="network")
    assert g.status == "paused"
    assert g.pause_reason == "network"


def test_goal_notes_appended():
    """pause/resume 等事件追加到 notes。"""
    g = GoalState(objective="x")
    g.pause(reason="manual")
    assert any("manual" in n for n in g.notes)
    g.resume()
    assert any("resume" in n.lower() for n in g.notes)
