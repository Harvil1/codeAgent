"""team_tool 工具测试。"""
import json
import sys
from pathlib import Path

import tools.team_tool  # 触发注册
from agent.team.bus import MessageBus
from agent.team.coordinator import TeamCoordinator
from tools.registry import registry


def _setup(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    coord = TeamCoordinator(team_dir=tmp_path, omnimate_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    return bus, coord


async def test_team_send(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = await registry.dispatch(
        "team_send",
        {"to": "w1", "content": "hello"},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert "id" in parsed


async def test_team_inbox_consumptive(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    bus.send(from_="w1", to="main", type_="message", content="hi")
    result_str = await registry.dispatch(
        "team_inbox", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["messages"][0]["content"] == "hi"
    # 再读应为空
    result_str2 = await registry.dispatch(
        "team_inbox", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    assert json.loads(result_str2)["count"] == 0


async def test_team_members(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = await registry.dispatch(
        "team_members", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["members"][0]["name"] == "main"


async def test_team_spawn_simple(tmp_path: Path):
    """spawn 一个最简子进程（不真的起 worker，用 echo 等价命令）。"""
    bus, coord = _setup(tmp_path)
    # 用真实 command 覆盖 worker 入口（测试中不真跑 agent）
    # 这里只测 registry.dispatch 调 coordinator.spawn 的 wiring
    # 真实 spawn 测试在 T8 integration
    # 这里测 handler 返回正确 JSON 格式
    # 直接 mock coord.spawn
    from unittest.mock import patch
    from agent.team.coordinator import TeamMember
    fake_member = TeamMember(
        name="w1", role="worker", pid=12345, status="running",
        created_at="2026-07-12T00:00:00", task="dummy",
    )
    with patch.object(coord, "spawn", return_value=fake_member):
        result_str = await registry.dispatch(
            "team_spawn",
            {"name": "w1", "task": "dummy"},
            team_bus=bus, team_coordinator=coord, team_name="main",
        )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["name"] == "w1"
    assert parsed["pid"] == 12345


async def test_team_shutdown_unknown(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = await registry.dispatch(
        "team_shutdown", {"name": "nonexistent"},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is False
    assert "error" in parsed


def test_team_spawn_with_task_id_forwards(monkeypatch, tmp_path):
    """team_spawn(args={..., 'task_id': X}) → coord.spawn(task_id=X)。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    result = _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": "task_abc"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    import json
    data = json.loads(result)
    assert data["success"] is True
    assert captured["task_id"] == "task_abc"
    assert captured["name"] == "w1"


def test_team_spawn_without_task_id_passes_none(monkeypatch):
    """不传 task_id → coord.spawn 不收 task_id 参数（None）。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    _handle_team_spawn(
        args={"name": "w1", "task": "do X"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    assert captured.get("task_id") is None


def test_team_spawn_empty_task_id_treated_as_absent(monkeypatch):
    """空字符串 task_id → 归一化为 None。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": ""},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    assert captured.get("task_id") is None


def test_team_spawn_invalid_task_id_returns_error(monkeypatch):
    """coord.spawn 抛 ValueError → 返回 invalid_task_id。"""
    from tools.team_tool import _handle_team_spawn

    class _FakeCoord:
        def spawn(self, **kwargs):
            raise ValueError("task_id task_xxx 不存在")

    result = _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": "task_xxx"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    import json
    data = json.loads(result)
    assert data["success"] is False
    assert data["error_type"] == "invalid_task_id"
