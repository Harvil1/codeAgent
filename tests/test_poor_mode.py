# tests/test_poor_mode.py
import copy
from agent.poor_mode import apply_poor_preset, POOR_PRESET, _set_dotted


def test_poor_preset_contains_all_flags():
    """POOR_PRESET 必须覆盖 7 个开关。"""
    expected_keys = {
        "reflection.enabled",
        "context.summarize_9section",
        "memory.auto_extract",
        "cache_monitor.enabled",
        "verification_agent.enabled",
        "curator.enabled",
        "context.reactive_compact_enabled",
    }
    assert set(POOR_PRESET.keys()) == expected_keys
    # 全是 False（全关）
    assert all(v is False for v in POOR_PRESET.values())


def test_set_dotted_creates_nested():
    d = {}
    _set_dotted(d, "a.b.c", True)
    assert d == {"a": {"b": {"c": True}}}


def test_set_dotted_overrides_existing():
    d = {"a": {"b": True}}
    _set_dotted(d, "a.b", False)
    assert d == {"a": {"b": False}}


def test_apply_poor_preset_flips_all_flags():
    """应用 preset 后所有目标 flag 变 False。"""
    config = {
        "reflection": {"enabled": True},
        "context": {"summarize_9section": True, "reactive_compact_enabled": True},
        "memory": {"auto_extract": True},
        "cache_monitor": {"enabled": True},
        "verification_agent": {"enabled": True},
        "curator": {"enabled": True},
    }
    new_config = apply_poor_preset(config, on=True)
    # 7 个 flag 全 False
    assert new_config["reflection"]["enabled"] is False
    assert new_config["context"]["summarize_9section"] is False
    assert new_config["context"]["reactive_compact_enabled"] is False
    assert new_config["memory"]["auto_extract"] is False
    assert new_config["cache_monitor"]["enabled"] is False
    assert new_config["verification_agent"]["enabled"] is False
    assert new_config["curator"]["enabled"] is False


def test_apply_poor_preset_does_not_modify_input():
    """不能修改入参（避免副作用）。"""
    config = {"reflection": {"enabled": True}}
    snapshot = copy.deepcopy(config)
    apply_poor_preset(config, on=True)
    assert config == snapshot


def test_apply_poor_preset_off_returns_untouched():
    """on=False 不动 config。"""
    config = {"reflection": {"enabled": True}}
    result = apply_poor_preset(config, on=False)
    assert result is config  # 同一对象


def test_apply_poor_preset_preserves_unrelated_keys():
    """不影响其他 config 字段。"""
    config = {
        "reflection": {"enabled": True, "interval_hours": 24},
        "security": {"sandbox_enabled": True},
    }
    new_config = apply_poor_preset(config, on=True)
    assert new_config["reflection"]["interval_hours"] == 24
    assert new_config["security"]["sandbox_enabled"] is True
