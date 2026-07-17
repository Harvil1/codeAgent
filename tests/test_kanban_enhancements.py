"""Kanban heartbeat / comment / artifacts 测试。"""
import json
from pathlib import Path

import pytest

from agent.task_store import TaskStore


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return TaskStore(harvil_home=tmp_path)


# ---------------------------------------------------------------------------
# Task 1: TaskStore 新字段 + 4 方法
# ---------------------------------------------------------------------------

def test_create_includes_new_fields(store):
    """create 出来的 task 含 last_heartbeat_at / comments / artifacts。"""
    task = store.create(subject="X")
    assert task["last_heartbeat_at"] is None
    assert task["comments"] == []
    assert task["artifacts"] == []


def test_heartbeat_sets_timestamp(store):
    import time
    task = store.create(subject="X")
    before = task["updated_at"]
    time.sleep(0.01)  # 确保 timestamp 不同
    updated = store.heartbeat(task["id"])
    assert updated is not None
    assert updated["last_heartbeat_at"] is not None
    assert updated["last_heartbeat_at"] > before


def test_add_comment_appends(store):
    task = store.create(subject="X")
    updated = store.add_comment(task["id"], author="w1", content="hi")
    assert len(updated["comments"]) == 1
    c = updated["comments"][0]
    assert c["author"] == "w1"
    assert c["content"] == "hi"
    assert "created_at" in c
    # 再加一条
    updated2 = store.add_comment(task["id"], author="w2", content="yo")
    assert len(updated2["comments"]) == 2


def test_add_artifacts_dedups(store):
    task = store.create(subject="X")
    updated = store.add_artifacts(task["id"], ["/a/b.txt", "/c/d.txt"])
    assert updated["artifacts"] == ["/a/b.txt", "/c/d.txt"]
    # 重复加同样的 → 去重
    updated2 = store.add_artifacts(task["id"], ["/a/b.txt", "/e/f.txt"])
    assert updated2["artifacts"] == ["/a/b.txt", "/c/d.txt", "/e/f.txt"]


def test_remove_artifacts(store):
    task = store.create(subject="X")
    store.add_artifacts(task["id"], ["/a", "/b", "/c"])
    updated = store.remove_artifacts(task["id"], ["/b"])
    assert updated["artifacts"] == ["/a", "/c"]


def test_legacy_task_compat(store, tmp_path):
    """旧 task JSON 没新字段时，setdefault 兜底。"""
    # 手动写一个不带新字段的 task JSON
    legacy = {
        "id": "task_legacy",
        "subject": "old",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    f = store._task_file("task_legacy")
    f.write_text(json.dumps(legacy), encoding="utf-8")
    # 用新方法不应崩
    t = store.add_comment("task_legacy", author="x", content="y")
    assert t is not None
    assert len(t["comments"]) == 1
    t2 = store.add_artifacts("task_legacy", ["/p"])
    assert t2["artifacts"] == ["/p"]


# ---------------------------------------------------------------------------
# Task 2: task_heartbeat handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_heartbeat, _infer_author


def test_heartbeat_handler_updates_task(store, monkeypatch):
    """handler 调用后 task.last_heartbeat_at 非 None。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_heartbeat(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["last_heartbeat_at"] is not None


def test_heartbeat_handler_with_note_adds_comment(store, monkeypatch):
    """带 note 的 heartbeat 同时加一条 comment，author=main（无 team_name）。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_heartbeat(
        {"id": task["id"], "note": "epoch 50/100"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    refreshed = store.get(task["id"])
    assert len(refreshed["comments"]) == 1
    assert refreshed["comments"][0]["content"] == "epoch 50/100"
    assert refreshed["comments"][0]["author"] == "main"


def test_heartbeat_handler_blocks_foreign_id(store, monkeypatch):
    """env 绑 task_A 时 heartbeat(task_B) → permission_denied。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_heartbeat(
        {"id": task_b["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_infer_author():
    """_infer_author: 有 team_name → team_name；无 → 'main'。"""
    assert _infer_author({"team_name": "w1"}) == "w1"
    assert _infer_author({}) == "main"
    assert _infer_author({"team_name": None}) == "main"


# ---------------------------------------------------------------------------
# Task 3: task_comment handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_comment


def test_comment_handler_appends(store, monkeypatch):
    """handler 后 comments 多一条。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_comment(
        {"id": task["id"], "content": "first comment"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    refreshed = store.get(task["id"])
    assert len(refreshed["comments"]) == 1
    assert refreshed["comments"][0]["content"] == "first comment"


def test_comment_handler_rejects_empty_content(store, monkeypatch):
    """content 为空 → 错误。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_comment(
        {"id": task["id"], "content": "   "},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "content" in data["error"]


def test_comment_handler_blocks_foreign_id(store, monkeypatch):
    """跨任务 comment 被拒。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_comment(
        {"id": task_b["id"], "content": "hi"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"
