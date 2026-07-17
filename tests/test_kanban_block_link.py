"""Kanban block/unblock/link + DAG 测试。"""
import json

import pytest

from agent.task_store import TaskStore, VALID_STATUSES


@pytest.fixture
def store(tmp_path):
    return TaskStore(harvil_home=tmp_path)


# ---------------------------------------------------------------------------
# Task 1: TaskStore has_path + add_dependency
# ---------------------------------------------------------------------------

def test_valid_statuses_includes_blocked():
    assert "blocked" in VALID_STATUSES


def test_create_includes_block_fields(store):
    task = store.create(subject="X")
    assert task["block_reason"] is None
    assert task["block_kind"] is None


def test_has_path_direct_dependency(store):
    """A.blocked_by=[B] → has_path(A, B) == True。"""
    b = store.create(subject="B")
    a = store.create(subject="A", blocked_by=[b["id"]])
    assert store.has_path(a["id"], b["id"]) is True


def test_has_path_indirect(store):
    """A→B→C（A 依赖 B，B 依赖 C）→ has_path(A, C) == True。"""
    c = store.create(subject="C")
    b = store.create(subject="B", blocked_by=[c["id"]])
    a = store.create(subject="A", blocked_by=[b["id"]])
    assert store.has_path(a["id"], c["id"]) is True


def test_has_path_no_path(store):
    """无关任务 → False。"""
    a = store.create(subject="A")
    z = store.create(subject="Z")
    assert store.has_path(a["id"], z["id"]) is False


def test_add_dependency_basic(store):
    """add_dependency(child, parent) → child.blocked_by 含 parent。"""
    parent = store.create(subject="P")
    child = store.create(subject="C")
    updated = store.add_dependency(child["id"], parent["id"])
    assert updated is not None
    assert parent["id"] in updated["blocked_by"]


def test_add_dependency_self_link_raises(store):
    """add_dependency(X, X) → ValueError。"""
    x = store.create(subject="X")
    with pytest.raises(ValueError, match="self-link"):
        store.add_dependency(x["id"], x["id"])


def test_add_dependency_cycle_rejected(store):
    """已有 A→B，再加 B→A → ValueError。"""
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # 现在 B 依赖 A。如果加 A 依赖 B（add_dependency(A, B)），成环
    with pytest.raises(ValueError, match="cycle"):
        store.add_dependency(a["id"], b["id"])


def test_add_dependency_idempotent(store):
    """加同样的边两次 → blocked_by 只一条。"""
    parent = store.create(subject="P")
    child = store.create(subject="C")
    store.add_dependency(child["id"], parent["id"])
    updated = store.add_dependency(child["id"], parent["id"])
    assert updated["blocked_by"].count(parent["id"]) == 1


def test_add_dependency_no_validate_bypasses_cycle(store):
    """validate=False 跳过 cycle 检测（紧急逃生口）。"""
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # 加 A→B 不验证（明知会成环）
    updated = store.add_dependency(a["id"], b["id"], validate=False)
    assert updated is not None
    assert b["id"] in updated["blocked_by"]


# ---------------------------------------------------------------------------
# Task 2: task_block + task_unblock handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_block, _handle_task_unblock


def test_block_sets_status_and_reason(store, monkeypatch):
    """task_block → status=blocked, block_reason=reason, block_kind=kind。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "waiting for API", "kind": "needs_input"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "blocked"
    assert data["task"]["block_reason"] == "waiting for API"
    assert data["task"]["block_kind"] == "needs_input"


def test_block_rejects_empty_reason(store, monkeypatch):
    """reason="" → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "   "},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "reason" in data["error"]


def test_block_rejects_invalid_kind(store, monkeypatch):
    """kind="random" → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "X", "kind": "random"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "kind" in data["error"]


def test_block_blocks_foreign_id(store, monkeypatch):
    """跨任务 block 被 permission_denied。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_block(
        {"id": task_b["id"], "reason": "X"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_unblock_resets_to_pending(store, monkeypatch):
    """task_unblock 默认回 pending，清空 block_reason/kind。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    store.update(task["id"], status="blocked",
                 block_reason="X", block_kind="needs_input")
    result = _handle_task_unblock(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "pending"
    assert data["task"]["block_reason"] is None
    assert data["task"]["block_kind"] is None


def test_unblock_custom_status(store, monkeypatch):
    """new_status='in_progress' → status=in_progress。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    store.update(task["id"], status="blocked", block_reason="X")
    result = _handle_task_unblock(
        {"id": task["id"], "new_status": "in_progress"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "in_progress"


def test_unblock_non_blocked_rejected(store, monkeypatch):
    """对 pending 任务调 unblock → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_unblock(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "blocked" in data["error"]
