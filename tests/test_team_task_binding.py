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
