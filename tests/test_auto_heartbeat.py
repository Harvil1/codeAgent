"""auto_heartbeat 模块单元测试。"""
import pytest

from agent.team.auto_heartbeat import (
    maybe_heartbeat,
    reset_for_test,
    register,
    _post_tool_use_hook,
)
from agent.task_store import TaskStore


@pytest.fixture(autouse=True)
def _reset_state():
    """每个测试前重置 rate-limit 计时。"""
    reset_for_test()
    yield
    reset_for_test()


def test_maybe_heartbeat_noop_without_env(monkeypatch, tmp_path):
    """env 未设 → return False，不调 store。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    # env 没设，task 存在也不应触发
    result = maybe_heartbeat()
    assert result is False
    refreshed = store.get(task["id"])
    assert refreshed["last_heartbeat_at"] is None


def test_maybe_heartbeat_writes_with_env(monkeypatch, tmp_path):
    """env 设了 + task 存在 → 写入 last_heartbeat_at。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])
    # 注意：maybe_heartbeat 用全局 task_store 单例，需要把 task 加到单例能找到的地方
    # 测试里直接用 monkeypatch 替换 get_task_store
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", store)
    result = maybe_heartbeat()
    assert result is True
    refreshed = store.get(task["id"])
    assert refreshed["last_heartbeat_at"] is not None


def test_maybe_heartbeat_rate_limit(monkeypatch, tmp_path):
    """连续调两次 < 60s → 第二次 no-op。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", store)

    first = maybe_heartbeat()
    assert first is True
    first_ts = store.get(task["id"])["last_heartbeat_at"]
    # 立即再调
    second = maybe_heartbeat()
    assert second is False
    second_ts = store.get(task["id"])["last_heartbeat_at"]
    assert first_ts == second_ts  # 没更新


def test_maybe_heartbeat_silent_failure(monkeypatch, tmp_path):
    """store.heartbeat 抛异常 → maybe_heartbeat 返 False 不上抛。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_nonexistent")
    # 全局单例为 None → get_task_store() 走默认路径，找不到 task 返 None
    # maybe_heartbeat 应静默
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", None)
    result = maybe_heartbeat()
    assert result is False  # 不抛


def test_post_tool_use_hook_returns_none(monkeypatch, tmp_path):
    """_post_tool_use_hook 永远返回 None（不替换 result）。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    ret = _post_tool_use_hook("some_tool", {"x": 1}, "result string")
    assert ret is None


def test_register_none_noop():
    """register(None) 不抛。"""
    register(None)  # 不应抛


def test_register_real_registry():
    """register 真实 HookRegistry → 内部 register_post_tool_use 被调用。"""
    from agent.hooks import HookRegistry
    reg = HookRegistry()
    register(reg)
    # 验证：注册后 POST_TOOL_USE hook 列表非空
    from agent.hooks import HookEvent
    assert len(reg._hooks[HookEvent.POST_TOOL_USE]) >= 1
