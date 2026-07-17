"""task_binding 模块单元测试。"""
import pytest

from agent.team.task_binding import (
    ENV_VAR,
    TaskOwnershipError,
    get_bound_task_id,
    assert_owned,
)


# ---------------------------------------------------------------------------
# get_bound_task_id
# ---------------------------------------------------------------------------

def test_env_var_constant():
    assert ENV_VAR == "HARVIL_KANBAN_TASK"


def test_get_bound_task_id_none_when_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert get_bound_task_id() is None


def test_get_bound_task_id_returns_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    assert get_bound_task_id() == "task_abc"


def test_get_bound_task_id_returns_none_for_empty(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "")
    assert get_bound_task_id() is None


# ---------------------------------------------------------------------------
# assert_owned
# ---------------------------------------------------------------------------

def test_assert_owned_passes_when_unset(monkeypatch):
    """主 agent / legacy 调用（env 未设）不受限。"""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert_owned("any_task_id")  # 不抛


def test_assert_owned_passes_when_match(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    assert_owned("task_abc")  # 不抛


def test_assert_owned_raises_on_mismatch(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    with pytest.raises(TaskOwnershipError) as exc_info:
        assert_owned("task_xyz")
    msg = str(exc_info.value)
    assert "task_abc" in msg
    assert "task_xyz" in msg


def test_task_ownership_error_is_permission_error():
    """TaskOwnershipError 是 PermissionError 子类（handler 用 except PermissionError 精准捕获）。"""
    assert issubclass(TaskOwnershipError, PermissionError)


# ---------------------------------------------------------------------------
# task_tools handler 集成
# ---------------------------------------------------------------------------

import json

from agent.task_store import get_task_store
from tools.task_tools import _handle_task_update, _handle_task_complete


def test_task_complete_blocks_foreign_id(monkeypatch, tmp_path):
    """env 绑定 task_A → task_complete(task_B) 返回 permission_denied。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    result = _handle_task_complete(
        {"id": "task_B"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("error_type") == "permission_denied"
    assert "task_A" in data.get("error", "")
    assert "task_B" in data.get("error", "")


def test_task_update_blocks_foreign_id(monkeypatch, tmp_path):
    """env 绑定 task_A → task_update(task_B) 返回 permission_denied。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    result = _handle_task_update(
        {"id": "task_B", "status": "completed"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("error_type") == "permission_denied"


def test_task_complete_allows_matching_id(monkeypatch, tmp_path):
    """env 绑定 → task_complete(同 id) 正常执行。"""
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])

    result = _handle_task_complete(
        {"id": task["id"]},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True
    refreshed = store.get(task["id"])
    assert refreshed["status"] == "completed"


def test_task_update_allows_matching_id(monkeypatch, tmp_path):
    """env 绑定 → task_update(同 id) 正常执行。"""
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])

    result = _handle_task_update(
        {"id": task["id"], "status": "in_progress"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_complete_allows_when_unset(monkeypatch, tmp_path):
    """主 agent / legacy 调用（env 未设）正常执行。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")

    result = _handle_task_complete(
        {"id": task["id"]},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_create_not_gated(monkeypatch, tmp_path):
    """task_create 不受门控——worker 可创建新任务（不算跨任务操纵）。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    from tools.task_tools import _handle_task_create
    result = _handle_task_create(
        {"subject": "新任务"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_list_not_gated(monkeypatch, tmp_path):
    """task_list 不受门控——读取不敏感。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    from tools.task_tools import _handle_task_list
    result = _handle_task_list({}, harvil_home=str(tmp_path))
    data = json.loads(result)
    # 不报 permission_denied 即可
    assert data.get("error_type") != "permission_denied"
