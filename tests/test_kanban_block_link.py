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
