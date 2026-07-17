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
