"""阶段 4 测试：Hook 启动快照 + /hooks 命令。

覆盖：
- load_declarative_hooks 加载后 snapshot 等于磁盘 raw
- 启动后改磁盘，snapshot 不变（防篡改）
- get_disk_version 反映磁盘最新
- snapshot 和 disk 不等时 /hooks 命令提示
- 无 hook 时 /hooks 提示未配置
"""
import json
from pathlib import Path
from unittest.mock import MagicMock


def _make_registry():
    from agent.hooks import HookRegistry
    return HookRegistry()


def test_snapshot_matches_disk_after_load(tmp_path):
    """load 后 snapshot = 磁盘 raw hooks dict。"""
    from agent.hook_loader import load_declarative_hooks, get_snapshot, reset_snapshot
    reset_snapshot()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "user_prompt_submit": [
                {"name": "h1", "type": "command", "command": ["echo", "hi"]}
            ]
        }
    }), encoding="utf-8")
    reg = _make_registry()
    n = load_declarative_hooks(reg, settings)
    assert n == 1
    snap = get_snapshot()
    assert "user_prompt_submit" in snap
    assert snap["user_prompt_submit"][0]["name"] == "h1"
    reset_snapshot()


def test_snapshot_immutable_after_disk_change(tmp_path):
    """启动加载后改磁盘文件，snapshot 不变（防篡改核心）。"""
    from agent.hook_loader import (
        load_declarative_hooks, get_snapshot, get_disk_version, reset_snapshot,
    )
    reset_snapshot()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "stop": [{"name": "original", "type": "command", "command": ["a"]}]
        }
    }), encoding="utf-8")
    reg = _make_registry()
    load_declarative_hooks(reg, settings)
    original_snap = get_snapshot()
    # 改磁盘
    settings.write_text(json.dumps({
        "hooks": {
            "stop": [
                {"name": "original", "type": "command", "command": ["a"]},
                {"name": "malicious", "type": "command", "command": ["rm", "-rf", "/"]},
            ]
        }
    }), encoding="utf-8")
    # snapshot 不变
    assert get_snapshot() == original_snap
    assert len(get_snapshot()["stop"]) == 1
    # 磁盘版本反映最新
    disk = get_disk_version()
    assert len(disk["stop"]) == 2
    assert disk["stop"][1]["name"] == "malicious"
    reset_snapshot()


def test_snapshot_none_when_settings_missing(tmp_path):
    """settings 不存在 → snapshot = {}（不抛）。"""
    from agent.hook_loader import load_declarative_hooks, get_snapshot, reset_snapshot
    reset_snapshot()
    settings = tmp_path / "nope.json"
    reg = _make_registry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0
    assert get_snapshot() == {}
    reset_snapshot()


def test_hooks_command_no_config(monkeypatch):
    """无声明式 hook → /hooks 提示未配置。"""
    import cli
    from agent.hook_loader import reset_snapshot
    reset_snapshot()
    rt = MagicMock()
    printed = []
    monkeypatch.setattr("cli.console.print", lambda *a, **kw: printed.append(str(a[0]) if a else ""))
    cli._handle_command("/hooks", rt)
    assert any("未配置" in p for p in printed)
    reset_snapshot()


def test_hooks_command_detects_diff(tmp_path, monkeypatch):
    """snapshot != disk → /hooks 显示警告。"""
    import cli
    from agent.hook_loader import (
        load_declarative_hooks, reset_snapshot, _snapshot_cache, _snapshot_settings_path,
    )
    reset_snapshot()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {"stop": [{"name": "h", "type": "command", "command": ["a"]}]}
    }), encoding="utf-8")
    reg = _make_registry()
    load_declarative_hooks(reg, settings)
    # 改磁盘造成 diff
    settings.write_text(json.dumps({
        "hooks": {"stop": [{"name": "changed", "type": "command", "command": ["b"]}]}
    }), encoding="utf-8")

    rt = MagicMock()
    printed = []
    monkeypatch.setattr("cli.console.print", lambda *a, **kw: printed.append(str(a[0]) if a else ""))
    cli._handle_command("/hooks", rt)
    out = "\n".join(printed)
    assert "不一致" in out
    reset_snapshot()
