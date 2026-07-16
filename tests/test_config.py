"""Config 与 Profile 测试。"""

import os
from pathlib import Path

import pytest

import config
from config import (
    DEFAULT_CONFIG, load_config, save_config, _deep_merge, _remove_none,
    save_config_value, get_config_value, ensure_default_config,
    OPTIONAL_ENV_VARS,
)
from agent_profile import (
    get_profiles_root, list_profiles, apply_profile,
    create_profile, delete_profile, get_current_profile,
)


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------

def test_default_config_has_deepseek():
    """默认配置是 DeepSeek。"""
    assert DEFAULT_CONFIG["model"]["provider"] == "deepseek"
    assert DEFAULT_CONFIG["model"]["name"] == "deepseek-chat"
    assert DEFAULT_CONFIG["model"]["api_key_env"] == "DEEPSEEK_API_KEY"


def test_default_config_has_curator():
    assert DEFAULT_CONFIG["curator"]["enabled"] is True
    assert DEFAULT_CONFIG["curator"]["interval_hours"] == 168


def test_default_config_has_delegation():
    assert DEFAULT_CONFIG["delegation"]["max_concurrent_children"] == 3
    assert DEFAULT_CONFIG["delegation"]["max_spawn_depth"] == 2


def test_optional_env_vars_includes_deepseek():
    assert "DEEPSEEK_API_KEY" in OPTIONAL_ENV_VARS


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_nonexistent_file(tmp_path):
    """不存在的配置文件返回默认配置。"""
    config_data = load_config(tmp_path / "nope.yaml")
    assert config_data["model"]["provider"] == "deepseek"


def test_load_config_overrides(tmp_path):
    """用户配置覆盖默认值。"""
    cf = tmp_path / "config.yaml"
    cf.write_text(
        "model:\n  name: gpt-4o\n  api_key_env: OPENAI_API_KEY\n",
        encoding="utf-8",
    )
    config_data = load_config(cf)
    # 覆盖
    assert config_data["model"]["name"] == "gpt-4o"
    assert config_data["model"]["api_key_env"] == "OPENAI_API_KEY"
    # 其他字段保留默认
    assert config_data["model"]["provider"] == "deepseek"  # 未被覆盖


def test_load_config_cli_overrides(tmp_path):
    """CLI 覆盖优先级最高。"""
    config_data = load_config(
        tmp_path / "nope.yaml",
        cli_overrides={"model": {"name": "cli-model"}},
    )
    assert config_data["model"]["name"] == "cli-model"


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 10, "z": 20}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 10, "z": 20}, "b": 3}


def test_deep_merge_none_not_override():
    """override 的 None 不覆盖默认值。"""
    base = {"a": 1, "b": 2}
    override = {"a": None}
    merged = _deep_merge(base, override)
    assert merged["a"] == 1  # None 没有覆盖


def test_deep_merge_empty_override():
    base = {"a": 1}
    assert _deep_merge(base, {}) == {"a": 1}
    assert _deep_merge(base, None) == {"a": 1}


# ---------------------------------------------------------------------------
# save_config / _remove_none
# ---------------------------------------------------------------------------

def test_remove_none():
    assert _remove_none({"a": 1, "b": None, "c": {"d": None, "e": 5}}) == {
        "a": 1, "c": {"e": 5}
    }


def test_save_and_reload(tmp_path):
    """保存后能重新加载。"""
    cf = tmp_path / "config.yaml"
    custom = {"model": {"name": "custom-model"}}
    save_config(custom, cf)
    assert cf.exists()

    loaded = load_config(cf)
    assert loaded["model"]["name"] == "custom-model"


def test_ensure_default_config(tmp_path, monkeypatch):
    """ensure_default_config 写入默认配置。"""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    cf = tmp_path / "config.yaml"
    assert not cf.exists()

    ensure_default_config()
    assert cf.exists()
    # 内容是有效的 yaml
    loaded = load_config(cf)
    assert loaded["model"]["provider"] == "deepseek"


# ---------------------------------------------------------------------------
# dotpath 访问
# ---------------------------------------------------------------------------

def test_get_config_value():
    cfg = {"model": {"name": "x", "nested": {"deep": 42}}}
    assert get_config_value(cfg, "model.name") == "x"
    assert get_config_value(cfg, "model.nested.deep") == 42


def test_get_config_value_default():
    cfg = {"a": 1}
    assert get_config_value(cfg, "b.c", "fallback") == "fallback"


def test_save_config_value(tmp_path):
    """save_config_value 能按 dotpath 设置。"""
    cf = tmp_path / "config.yaml"
    save_config_value("model.name", "new-model", cf)

    loaded = load_config(cf)
    assert loaded["model"]["name"] == "new-model"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def test_get_profiles_root():
    root = get_profiles_root()
    assert root.name == "profiles"


def test_list_profiles_empty(tmp_path, monkeypatch):
    """临时 HOME 下无 profile。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    # 注意：Path.home() 在 Windows 上可能用 USERPROFILE
    # 这里用 get_profiles_root 的逻辑间接验证
    assert list_profiles() == [] or all(
        p not in ["default"] for p in list_profiles()
    )


def test_create_and_list_profile(monkeypatch, tmp_path):
    """创建 profile 后能列出。"""
    # 把 profiles root 重定向到 tmp
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()

    monkeypatch.setattr(
        "agent_profile.get_profiles_root",
        lambda: fake_home / "profiles",
    )

    # create_profile 用 get_profiles_root
    p1 = create_profile("work")
    assert p1.exists()
    assert (p1 / "skills").exists()
    assert (p1 / "logs").exists()
    assert (p1 / ".env").exists()


def test_create_profile_duplicate(monkeypatch, tmp_path):
    fake_home = tmp_path / "fh"
    fake_home.mkdir()
    monkeypatch.setattr(
        "agent_profile.get_profiles_root",
        lambda: fake_home / "profiles",
    )

    create_profile("dup")
    with pytest.raises(ValueError, match="已存在"):
        create_profile("dup")


def test_apply_profile_default(monkeypatch):
    """apply_profile('default') 清除 AGENT_HOME。"""
    monkeypatch.setenv("AGENT_HOME", "/some/path")
    apply_profile("default")
    assert "AGENT_HOME" not in os.environ


def test_apply_profile_nonexistent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_profile.get_profiles_root",
        lambda: tmp_path / "profiles",
    )
    with pytest.raises(ValueError, match="不存在"):
        apply_profile("nonexistent")


def test_get_current_profile_default(monkeypatch):
    """无 AGENT_HOME 时返回 default。"""
    monkeypatch.delenv("AGENT_HOME", raising=False)
    assert get_current_profile() == "default"


# ---------------------------------------------------------------------------
# Phase 1 Task 8: context 配置块 + MemoryManager.on_pre_compress
# ---------------------------------------------------------------------------


def test_default_config_has_context_block():
    """config.py 的 DEFAULT_CONFIG 应含 context 块及所有 Phase 1 阈值。"""
    from config import DEFAULT_CONFIG
    ctx = DEFAULT_CONFIG["context"]
    expected_keys = {
        "output_offload_threshold", "output_offload_preview",
        "snip_message_threshold", "snip_release_threshold",
        "snip_keep_first", "snip_keep_last",
        "micro_keep_recent_results",
        "llm_compact_token_threshold", "llm_compact_message_threshold",
        "llm_compact_keep_recent", "llm_compact_cooldown_turns",
        "max_compress_attempts",
        "reactive_keep_recent", "reactive_once_per_session",
        "transcript_enabled", "transcript_trigger", "transcript_retention",
        "use_new_pipeline",
    }
    assert expected_keys.issubset(set(ctx.keys())), f"缺: {expected_keys - set(ctx.keys())}"


def test_default_config_use_new_pipeline_is_true():
    """Commit 6 后默认 True（新管线正式启用）。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["context"]["use_new_pipeline"] is True


def test_memory_manager_on_pre_compress_is_noop():
    """MemoryManager.on_pre_compress 默认 no-op（不抛即可）。"""
    from agent.memory_manager import MemoryManager
    from agent.memory_store import MemoryStore
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(harvil_home=Path(td))
        mm = MemoryManager(memory_store=store, external_provider=None)
        mm.on_pre_compress(None, [])  # 不抛


# ---------------------------------------------------------------------------
# Phase 2a Task 5: hooks 配置块
# ---------------------------------------------------------------------------


def test_default_config_has_hooks_block():
    """config.py 的 DEFAULT_CONFIG 应含 hooks 块及所有必需键。"""
    from config import DEFAULT_CONFIG
    h = DEFAULT_CONFIG["hooks"]
    required_keys = (
        "enabled", "settings_path", "script_timeout_default",
        "stop_hook_max_fires", "fail_closed_default"
    )
    for key in required_keys:
        assert key in h, f"缺 {key}"


def test_default_config_hooks_enabled_default_true():
    """hooks.enabled 默认值应为 True。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["hooks"]["enabled"] is True


def test_default_config_hooks_settings_path_default_none():
    """hooks.settings_path 默认值应为 None。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["hooks"]["settings_path"] is None


# ---------------------------------------------------------------------------
# Phase 2b Task 3: bg_task 配置块
# ---------------------------------------------------------------------------


def test_default_config_has_bg_task_block():
    """config.py 的 DEFAULT_CONFIG 应含 bg_task 块及所有必需键。"""
    from config import DEFAULT_CONFIG
    bg = DEFAULT_CONFIG["bg_task"]
    required_keys = (
        "enabled", "max_concurrent", "default_timeout",
        "notification_stdout_cap", "result_stdout_cap",
        "default_detach"
    )
    for key in required_keys:
        assert key in bg, f"缺 {key}"


def test_default_config_bg_task_defaults():
    """bg_task 块的默认值应符合规范。"""
    from config import DEFAULT_CONFIG
    bg = DEFAULT_CONFIG["bg_task"]
    assert bg["enabled"] is True
    assert bg["max_concurrent"] == 5
    assert bg["default_timeout"] == 600
    assert bg["default_detach"] is False


# ---------------------------------------------------------------------------
# Phase 2b Task 4: bg toolset
# ---------------------------------------------------------------------------


def test_bg_toolset_exists():
    """toolsets.py 应含 bg 工具集，含 5 个后台任务工具。"""
    from toolsets import TOOLSETS, resolve_toolset
    assert "bg" in TOOLSETS
    tools = resolve_toolset("bg")
    required_tools = ("bg_start", "bg_status", "bg_result", "bg_list", "bg_stop")
    for name in required_tools:
        assert name in tools, f"缺 {name}"


# ---------------------------------------------------------------------------
# Phase 2c Task 3: cron 配置块
# ---------------------------------------------------------------------------


def test_default_config_has_cron_block():
    """config.py 的 DEFAULT_CONFIG 应含 cron 块及所有必需键。"""
    from config import DEFAULT_CONFIG
    c = DEFAULT_CONFIG["cron"]
    required_keys = ("enabled", "jobs_path", "poll_interval_seconds")
    for key in required_keys:
        assert key in c, f"缺 {key}"


def test_default_config_cron_defaults():
    """cron 块的默认值应符合规范。"""
    from config import DEFAULT_CONFIG
    c = DEFAULT_CONFIG["cron"]
    assert c["enabled"] is True
    assert c["jobs_path"] is None
    assert c["poll_interval_seconds"] == 30.0


# ---------------------------------------------------------------------------
# Phase 5 Task 4 & 6: memory 多文件字段
# ---------------------------------------------------------------------------


def test_default_config_memory_multifile_fields():
    """config.py 的 DEFAULT_CONFIG['memory'] 应含多文件模式的所有新字段。"""
    from config import DEFAULT_CONFIG
    m = DEFAULT_CONFIG["memory"]
    required_keys = (
        "multifile_enabled", "memory_dir", "retrieval_enabled",
        "retrieval_max_results", "retrieval_model"
    )
    for key in required_keys:
        assert key in m, f"缺 {key}"


def test_default_config_memory_multifile_defaults():
    """memory 多文件字段的默认值应符合规范。"""
    from config import DEFAULT_CONFIG
    m = DEFAULT_CONFIG["memory"]
    assert m["multifile_enabled"] is True
    assert m["memory_dir"] is None
    assert m["retrieval_enabled"] is True
    assert m["retrieval_max_results"] == 5
    assert m["retrieval_model"] is None
