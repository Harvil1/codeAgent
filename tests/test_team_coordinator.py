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
