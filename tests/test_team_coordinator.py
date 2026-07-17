"""TeamCoordinator 测试。"""
import json
from pathlib import Path

import pytest

from agent.team.coordinator import TeamCoordinator, TeamMember


def test_register_lead(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    m = coord.register(name="main", role="lead")
    assert m.name == "main"
    assert m.role == "lead"
    assert m.status == "running"


def test_spawn_returns_member_with_pid(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    # 用最简单的 subprocess：python -c "pass"
    m = coord.spawn(name="w1", role="worker",
                     task="dummy", command=[_python(), "-c", "pass"])
    assert m.pid is not None and m.pid > 0
    # 等子进程退出
    import time
    time.sleep(0.5)
    members = coord.list_members()
    assert any(mm.name == "w1" for mm in members)


def test_spawn_duplicate_name_raises(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    with pytest.raises(ValueError, match="exists"):
        coord.register(name="main", role="lead")


def test_list_members(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    coord.register(name="w1", role="worker")
    members = coord.list_members()
    assert len(members) == 2
    names = {m.name for m in members}
    assert names == {"main", "w1"}


def test_update_status(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    m = coord.register(name="w1", role="worker")
    coord.update_status("w1", "completed")
    members = coord.list_members()
    assert members[0].status == "completed"


def test_spawn_max_members_raises(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 1}})
    coord.register(name="main", role="lead")
    with pytest.raises(RuntimeError, match="max"):
        coord.register(name="w1", role="worker")


def _python():
    import sys
    return sys.executable


class _FakePopen:
    """假 Popen，捕获 env 用于断言。"""
    def __init__(self, cmd, **kwargs):
        self.captured_env = dict(kwargs.get("env") or {})
        self.pid = 12345
        self._poll = 0

    def poll(self):
        return self._poll


def test_spawn_without_task_id_no_env(tmp_path, monkeypatch):
    """不传 task_id → 子进程 env 不含 HARVIL_KANBAN_TASK。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    coord.spawn(name="w1", role="worker", task="...", depth=1)
    assert "HARVIL_KANBAN_TASK" not in captured["env"]


def test_spawn_with_task_id_claims_and_sets_env(tmp_path, monkeypatch):
    """传 task_id → TaskStore.claim + env 注入。"""
    from agent.task_store import get_task_store
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="测试任务")

    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    coord.spawn(
        name="w1", role="worker", task="...",
        depth=1, task_id=task["id"],
    )

    # env 注入
    assert captured["env"]["HARVIL_KANBAN_TASK"] == task["id"]
    # claim 持久化
    refreshed = store.get(task["id"])
    assert refreshed["owner"] == "w1"
    assert refreshed["status"] == "in_progress"


def test_spawn_with_invalid_task_id_raises(tmp_path, monkeypatch):
    """task_id 不存在 → ValueError，且不调 Popen。"""
    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    popen_called = []
    def fake_popen(cmd, **kwargs):
        popen_called.append(cmd)
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    import pytest
    with pytest.raises(ValueError, match="不存在"):
        coord.spawn(
            name="w1", role="worker", task="...",
            task_id="task_nonexistent",
        )
    assert popen_called == []  # 没启动子进程
    # registry 里状态应为 failed
    members = coord.list_members()
    assert any(m.name == "w1" and m.status == "failed" for m in members)
