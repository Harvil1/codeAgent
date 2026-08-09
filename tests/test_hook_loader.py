"""hook_loader 测试：settings.json 解析 + 校验。"""
import json
from pathlib import Path

import pytest

from agent.hooks import HookEvent, HookRegistry
from agent.hook_loader import load_declarative_hooks


def _write(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_load_returns_zero_when_file_missing(tmp_path):
    """文件不存在 = 静默返回 0。"""
    reg = HookRegistry()
    n = load_declarative_hooks(reg, tmp_path / "nonexistent.json")
    assert n == 0


def test_load_valid_file(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "pre_tool_use": [
                {"name": "h1", "command": ["./h1.sh"], "timeout": 5.0},
            ],
            "post_tool_use": [
                {"name": "h2", "command": ["./h2.sh"]},
            ],
        }
    })
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 2
    assert len(reg._hooks[HookEvent.PRE_TOOL_USE]) == 1
    assert len(reg._hooks[HookEvent.POST_TOOL_USE]) == 1


def test_load_fail_closed_field(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "pre_tool_use": [
                {"name": "h", "command": ["./h.sh"], "fail_closed": True},
            ],
        }
    })
    reg = HookRegistry()
    load_declarative_hooks(reg, settings)
    hook = reg._hooks[HookEvent.PRE_TOOL_USE][0]
    assert hook.fail_closed is True


def test_load_env_field(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {
        "hooks": {
            "post_tool_use": [
                {"name": "h", "command": ["./h.sh"], "env": {"K": "v"}},
            ],
        }
    })
    reg = HookRegistry()
    load_declarative_hooks(reg, settings)
    hook = reg._hooks[HookEvent.POST_TOOL_USE][0]
    assert hook.script.env == {"K": "v"}


def test_load_invalid_event_name_raises(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"unknown_event": []}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="未知 event"):
        load_declarative_hooks(reg, settings)


def test_load_missing_hooks_field_raises(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"wrong_field": {}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="缺少.*hooks"):
        load_declarative_hooks(reg, settings)


def test_load_event_value_not_list_raises(tmp_path):
    """event 的值不是 list（如 string / null）时 raise ValueError。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": "not_a_list"}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="必须是 list"):
        load_declarative_hooks(reg, settings)


def test_load_event_value_null_raises(tmp_path):
    """event 的值是 null 时 raise ValueError。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": None}})
    reg = HookRegistry()
    with pytest.raises(ValueError, match="必须是 list"):
        load_declarative_hooks(reg, settings)


def test_load_missing_malformed_json_raises(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("not a json {{{", encoding="utf-8")
    reg = HookRegistry()
    with pytest.raises(json.JSONDecodeError):
        load_declarative_hooks(reg, settings)


def test_load_skips_hook_missing_name(tmp_path, caplog):
    """缺 name 字段的 hook 跳过 + warning。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"command": ["./h.sh"]}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0
    assert any("missing name" in r.message.lower() or "name" in r.message.lower()
               for r in caplog.records)


def test_load_skips_hook_missing_command(tmp_path, caplog):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"name": "h"}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0


def test_load_skips_hook_empty_command(tmp_path):
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {"pre_tool_use": [{"name": "h", "command": []}]}})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 0


def test_load_partial_failure_continues(tmp_path, caplog):
    """一个坏 hook 不影响其他好 hook。"""
    settings = tmp_path / "settings.json"
    _write(settings, {"hooks": {
        "pre_tool_use": [
            {"name": "good", "command": ["./g.sh"]},
            {"command": ["./no-name.sh"]},  # 缺 name，跳过
            {"name": "also_good", "command": ["./ag.sh"]},
        ]
    }})
    reg = HookRegistry()
    n = load_declarative_hooks(reg, settings)
    assert n == 2  # 两个 good 加载成功
