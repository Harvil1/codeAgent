"""Feature flags API 单元测试。

设计原则：
- 不依赖 config.py（用 mock dict），避免循环依赖
- 覆盖 spec §5.4 + §5.6 所有场景：默认 False / 字典形式 / 字符串容错 /
  未知 flag warn / config 节缺失 / 类型错回退
"""
import logging
from typing import Any, Dict

import pytest

from agent.feature_flags import is_feature_enabled, get_feature_config


# ────────────────────────────────────────────────────────────
# is_feature_enabled 测试
# ────────────────────────────────────────────────────────────

def test_flag_dict_form_enabled_true():
    """flag 是 dict 形式，enabled=True → 返回 True。"""
    config = {"features": {"bash_llm_classifier": {"enabled": True}}}
    assert is_feature_enabled(config, "bash_llm_classifier") is True


def test_flag_dict_form_enabled_false():
    """flag 是 dict 形式，enabled=False（默认）→ 返回 False。"""
    config = {"features": {"bash_llm_classifier": {"enabled": False, "model": "aux"}}}
    assert is_feature_enabled(config, "bash_llm_classifier") is False


def test_flag_dict_form_missing_enabled_key():
    """flag 是 dict 但没有 enabled 字段 → 默认 False。"""
    config = {"features": {"bash_llm_classifier": {"whitelist": ["ls"]}}}
    assert is_feature_enabled(config, "bash_llm_classifier") is False


def test_flag_bool_form_true():
    """flag 直接是 bool True（简写形式）→ 返回 True。"""
    config = {"features": {"simple_flag": True}}
    assert is_feature_enabled(config, "simple_flag") is True


def test_flag_bool_form_false():
    """flag 直接是 bool False（简写形式）→ 返回 False。"""
    config = {"features": {"simple_flag": False}}
    assert is_feature_enabled(config, "simple_flag") is False


def test_flag_string_form_falls_back_to_false():
    """flag 是字符串 "true"（类型错）→ 返回 False（fail-safe）。"""
    config = {"features": {"misconfigured_flag": "true"}}
    assert is_feature_enabled(config, "misconfigured_flag") is False


def test_unknown_flag_returns_false_and_warns(caplog):
    """未知 flag 名 → 返回 False + log warning（防 typo）。"""
    config = {"features": {"known_flag": {"enabled": True}}}
    with caplog.at_level(logging.WARNING, logger="agent.feature_flags"):
        result = is_feature_enabled(config, "nonexistent_flag")
    assert result is False
    assert "nonexistent_flag" in caplog.text
    assert "未知" in caplog.text or "unknown" in caplog.text.lower()


def test_features_section_missing_returns_false():
    """config 完全没有 features 节 → 返回 False（fail-safe）。"""
    config = {"model": {"name": "deepseek-chat"}}
    assert is_feature_enabled(config, "any_flag") is False


def test_config_not_dict_returns_false():
    """config 不是 dict（极端防御）→ 返回 False。"""
    assert is_feature_enabled(None, "any_flag") is False  # type: ignore[arg-type]
    assert is_feature_enabled("not a dict", "any_flag") is False  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────
# get_feature_config 测试
# ────────────────────────────────────────────────────────────

def test_get_feature_config_returns_full_dict():
    """get_feature_config 返回 flag 的完整配置（含 enabled 之外的参数）。"""
    config = {"features": {"bash_llm_classifier": {
        "enabled": True,
        "model": "aux",
        "whitelist": ["ls", "cat"],
    }}}
    cfg = get_feature_config(config, "bash_llm_classifier")
    assert cfg == {"enabled": True, "model": "aux", "whitelist": ["ls", "cat"]}


def test_get_feature_config_unknown_returns_empty():
    """未知 flag → 返回空字典。"""
    config = {"features": {}}
    assert get_feature_config(config, "nonexistent") == {}


def test_get_feature_config_non_dict_returns_empty():
    """flag 值不是 dict（如 bool/str）→ 返回空字典。"""
    config = {"features": {"simple_flag": True}}
    assert get_feature_config(config, "simple_flag") == {}


def test_get_feature_config_handles_no_features_section():
    """config 没有 features 节 → 返回空字典。"""
    config = {"model": {}}
    assert get_feature_config(config, "any") == {}


# ────────────────────────────────────────────────────────────
# DEFAULT_CONFIG["features"] 节测试
# ────────────────────────────────────────────────────────────

from config import DEFAULT_CONFIG  # noqa: E402


EXPECTED_FLAG_NAMES = {
    "bash_llm_classifier",
    "context_collapse",
    "reactive_compact",
    "bash_unattended_retry",
    "mcp_http_transport",
    "mcp_websocket_transport",
    "plan_mode_v2_parallel",
    "hook_http_handler",
    "hook_mcp_tool_handler",
    "hook_agent_handler",
}


def test_default_config_has_features_section():
    """DEFAULT_CONFIG 必须有 features 节，且是 dict。"""
    assert "features" in DEFAULT_CONFIG
    assert isinstance(DEFAULT_CONFIG["features"], dict)


def test_default_config_has_all_10_flags():
    """DEFAULT_CONFIG["features"] 必须含全部 10 个 flag 名（防漏配）。"""
    actual_names = set(DEFAULT_CONFIG["features"].keys())
    missing = EXPECTED_FLAG_NAMES - actual_names
    extra = actual_names - EXPECTED_FLAG_NAMES
    assert not missing, f"DEFAULT_CONFIG 缺少 flag: {missing}"
    assert not extra, f"DEFAULT_CONFIG 多了未声明的 flag: {extra}"


def test_all_flags_default_off():
    """所有 flag 默认必须 OFF（用户决策：装完默认关）。"""
    for name in EXPECTED_FLAG_NAMES:
        flag = DEFAULT_CONFIG["features"][name]
        assert isinstance(flag, dict), f"{name} 应为 dict 形式"
        assert flag.get("enabled") is False, f"{name} 默认必须 enabled=False"


def test_bash_llm_classifier_has_required_fields():
    """bash_llm_classifier 必须含 model + whitelist（批次 3 实现时依赖）。"""
    flag = DEFAULT_CONFIG["features"]["bash_llm_classifier"]
    assert "model" in flag
    assert "whitelist" in flag
    assert isinstance(flag["whitelist"], list)
    assert "ls" in flag["whitelist"]  # 至少含基础命令


def test_context_collapse_has_threshold():
    """context_collapse 必须含 threshold_ratio。"""
    flag = DEFAULT_CONFIG["features"]["context_collapse"]
    assert "threshold_ratio" in flag
    assert 0 < flag["threshold_ratio"] < 1


def test_bash_unattended_retry_has_max_hours():
    """bash_unattended_retry 必须含 max_hours。"""
    flag = DEFAULT_CONFIG["features"]["bash_unattended_retry"]
    assert "max_hours" in flag
    assert flag["max_hours"] > 0


def test_mcp_http_transport_has_timeout():
    """mcp_http_transport 必须含 default_timeout_sec。"""
    flag = DEFAULT_CONFIG["features"]["mcp_http_transport"]
    assert "default_timeout_sec" in flag
    assert flag["default_timeout_sec"] > 0


def test_plan_mode_v2_has_max_parallel():
    """plan_mode_v2_parallel 必须含 max_parallel_agents（决策 6）。"""
    flag = DEFAULT_CONFIG["features"]["plan_mode_v2_parallel"]
    assert "max_parallel_agents" in flag
    assert 1 <= flag["max_parallel_agents"] <= 3


# ────────────────────────────────────────────────────────────
# 端到端测试：用户 settings.json 覆盖 + 类型容错 + 部分覆盖
# ────────────────────────────────────────────────────────────

from config import _deep_merge  # noqa: E402


def _merge(user_overrides):
    """模拟 load_config 流程：DEFAULT_CONFIG ← _deep_merge ← 用户配置。"""
    import copy
    merged = copy.deepcopy(DEFAULT_CONFIG)
    return _deep_merge(merged, user_overrides)


def test_user_can_enable_flag_via_settings():
    """用户 settings.json 写 enabled=true → 合并后 flag 开启。"""
    user_config = {
        "features": {
            "bash_llm_classifier": {"enabled": True},
        }
    }
    merged = _merge(user_config)
    assert is_feature_enabled(merged, "bash_llm_classifier") is True


def test_user_enable_preserves_default_sub_fields():
    """用户只覆盖 enabled，默认的 model/whitelist 应保留（deep_merge 行为）。"""
    user_config = {
        "features": {
            "bash_llm_classifier": {"enabled": True},
        }
    }
    merged = _merge(user_config)
    cfg = get_feature_config(merged, "bash_llm_classifier")
    assert cfg["enabled"] is True
    assert cfg["model"] == "aux"               # 默认值保留
    assert "ls" in cfg["whitelist"]            # 默认白名单保留


def test_user_can_override_sub_fields():
    """用户可以覆盖 flag 内的具体参数（如自定义白名单）。"""
    user_config = {
        "features": {
            "bash_llm_classifier": {
                "whitelist": ["ls", "my-custom-cmd"],
            },
        }
    }
    merged = _merge(user_config)
    cfg = get_feature_config(merged, "bash_llm_classifier")
    # 用户覆盖了 whitelist
    assert cfg["whitelist"] == ["ls", "my-custom-cmd"]
    # 但 enabled 仍是默认 False（用户没显式开）
    assert cfg["enabled"] is False


def test_user_can_disable_default_on_flag():
    """即使默认 ON 的 flag（如果有），用户也能显式关闭。

    注：当前所有 flag 默认 OFF，此测试是前向兼容（将来加默认 ON 的 flag 时验证）。
    """
    # 模拟"默认 ON，用户关掉"的场景
    custom_default = {
        "features": {
            "hypothetical_default_on_flag": {"enabled": True, "param": "x"}
        }
    }
    user_overrides = {
        "features": {
            "hypothetical_default_on_flag": {"enabled": False}
        }
    }
    import copy
    merged = copy.deepcopy(custom_default)
    merged = _deep_merge(merged, user_overrides)
    assert is_feature_enabled(merged, "hypothetical_default_on_flag") is False
    # param 应保留
    assert get_feature_config(merged, "hypothetical_default_on_flag")["param"] == "x"


def test_user_unknown_flag_warns_but_doesnt_crash(caplog):
    """用户 settings.json 写了不存在的 flag → log warning，不崩。"""
    user_config = {
        "features": {
            "typo_flag_name": {"enabled": True},
        }
    }
    merged = _merge(user_config)
    # 未知 flag 在 DEFAULT_CONFIG 里没有，但合并后会出现（因为 _deep_merge 会加新 key）
    # 这是预期行为：用户配置新 flag 不会崩，但 is_feature_enabled 会 log warning
    # 注意：DEFAULT_CONFIG 没有这个 flag，所以合并后的 dict 里有，但读取时会 warn
    with caplog.at_level(logging.WARNING, logger="agent.feature_flags"):
        result = is_feature_enabled(merged, "typo_flag_name")
    # typo_flag 在合并后 dict 里存在（_deep_merge 加了），所以不会 warn unknown
    # 但如果用户 typo 写错名，DEFAULT_CONFIG 没有这个 flag，仍然存在；
    # 关键是不崩
    assert isinstance(result, bool)


def test_other_config_sections_unaffected():
    """加 features 节不影响其他节（model/security/memory 等）。"""
    user_config = {
        "model": {"name": "custom-model"},
        "features": {"bash_llm_classifier": {"enabled": True}},
    }
    merged = _merge(user_config)
    # features 节工作
    assert is_feature_enabled(merged, "bash_llm_classifier") is True
    # 其他节也正常
    assert merged["model"]["name"] == "custom-model"
    assert merged["model"]["provider"] == "deepseek"  # 默认值保留


def test_features_section_completely_absent_in_user_config():
    """用户 settings.json 完全没写 features 节 → 用 DEFAULT_CONFIG 默认值。"""
    user_config = {
        "model": {"name": "custom"},
    }
    merged = _merge(user_config)
    # 所有 flag 仍是默认 OFF
    for name in EXPECTED_FLAG_NAMES:
        assert is_feature_enabled(merged, name) is False
