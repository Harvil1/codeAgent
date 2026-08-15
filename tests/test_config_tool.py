"""config_tool 测试（CCAR12 Task 7）：LLM 安全改配置（白名单 7 键精确匹配）。

覆盖五层：
1. config_get：读值（runtime 优先）/ 白名单外拒
2. config_set：白名单外拒 / round-trip 落盘 / runtime 生效 / 类型强转 / hook 触发
3. handler dispatch 契约（args, **kwargs，防 silent-dead-code）
4. schema 键契约（OpenAI "parameters"，CCAR11 第 5 例教训）
5. 白名单本身（frozenset 精确匹配，7 键）
"""
import inspect
import json
from unittest.mock import MagicMock

import pytest

import tools.config_tool  # noqa: F401 触发注册
from tools.config_tool import (
    _CONFIG_WHITELIST,
    _handle_config_get,
    _handle_config_set,
    CONFIG_GET_SCHEMA,
    CONFIG_SET_SCHEMA,
)
from tools.registry import registry
from agent.settings import load_settings, save_settings


@pytest.fixture
def settings_home(tmp_path, monkeypatch):
    """把 OMNIMATE_HOME 指到 tmp（settings.json 读写全在沙箱里）。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    return tmp_path


def _make_agent_ref(config=None):
    """构造带 config dict 的假 agent_ref。"""
    ref = MagicMock()
    ref.config = config if config is not None else {}
    return ref


# ---------------------------------------------------------------------------
# 1. 白名单本身
# ---------------------------------------------------------------------------

class TestWhitelist:
    def test_whitelist_exact_seven_keys(self):
        """白名单恰好 7 键（brief 指定），frozenset。"""
        assert isinstance(_CONFIG_WHITELIST, frozenset)
        assert len(_CONFIG_WHITELIST) == 7

    def test_whitelist_exact_match_not_prefix(self):
        """精确匹配：前缀相同但更长的键不在白名单里。"""
        assert "notifications.enabled" in _CONFIG_WHITELIST
        assert "notifications" not in _CONFIG_WHITELIST
        assert "notifications.enabled.extra" not in _CONFIG_WHITELIST
        assert "security.command_approval" not in _CONFIG_WHITELIST  # 白名单外
        assert "llm.base_url" not in _CONFIG_WHITELIST  # 敏感键必须拒


# ---------------------------------------------------------------------------
# 2. config_get
# ---------------------------------------------------------------------------

class TestConfigGet:
    def test_get_from_settings(self, settings_home):
        """settings.json 里有值 → 读回。"""
        settings = load_settings()
        settings["notifications"] = {"enabled": False}
        save_settings(settings)

        result = _handle_config_get({"key": "notifications.enabled"})
        data = json.loads(result)
        assert data == {"key": "notifications.enabled", "value": False}

    def test_get_prefers_runtime_config(self, settings_home):
        """agent_ref.config 优先（反映实际生效值，而非磁盘旧值）。"""
        settings = load_settings()
        settings["trace"] = {"enabled": True}
        save_settings(settings)

        ref = _make_agent_ref({"trace": {"enabled": False}})
        result = _handle_config_get({"key": "trace.enabled"}, agent_ref=ref)
        data = json.loads(result)
        assert data["value"] is False

    def test_get_missing_key_returns_none(self, settings_home):
        """键存在路径但从未设置 → value=None（不报错）。"""
        result = _handle_config_get({"key": "trace.enabled"})
        data = json.loads(result)
        assert data["key"] == "trace.enabled"
        assert data["value"] is None

    def test_get_denied_outside_whitelist(self, settings_home):
        """白名单外 → permission_denied。"""
        result = _handle_config_get({"key": "llm.auth_token"})
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_get_denied_prefix_lookalike(self, settings_home):
        """前缀相似但不在白名单 → 拒（防绕过）。"""
        result = _handle_config_get({"key": "notifications.enabledx"})
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_get_missing_args(self, settings_home):
        """缺 key → invalid_args。"""
        result = _handle_config_get({})
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"


# ---------------------------------------------------------------------------
# 3. config_set
# ---------------------------------------------------------------------------

class TestConfigSet:
    def test_set_roundtrip_persisted(self, settings_home):
        """set → settings.json 落盘（重新 load_settings 验证 round-trip）。"""
        result = _handle_config_set(
            {"key": "trace.retention_days", "value": 5}, agent_ref=_make_agent_ref()
        )
        data = json.loads(result)
        assert data["key"] == "trace.retention_days"
        assert data["value"] == 5

        # 磁盘 round-trip
        settings = load_settings()
        assert settings["trace"]["retention_days"] == 5

    def test_set_runtime_applied(self, settings_home):
        """set → agent_ref.config 同步（嵌套路径 set，runtime 立即生效）。"""
        ref = _make_agent_ref(
            {"memory": {"curator": {"enabled": True, "interval_hours": 168}}}
        )
        result = _handle_config_set(
            {"key": "memory.curator.enabled", "value": False}, agent_ref=ref
        )
        data = json.loads(result)
        assert data["runtime_applied"] is True
        assert ref.config["memory"]["curator"]["enabled"] is False
        # 其他键不受影响
        assert ref.config["memory"]["curator"]["interval_hours"] == 168

    def test_set_creates_nested_dicts(self, settings_home):
        """runtime config 缺中间层 → setdefault 逐层创建。"""
        ref = _make_agent_ref({})
        _handle_config_set(
            {"key": "statusline.enabled", "value": False}, agent_ref=ref
        )
        assert ref.config["statusline"]["enabled"] is False

    def test_set_no_agent_ref_still_persists(self, settings_home):
        """无 agent_ref（如直接 dispatch 测试）→ 落盘照常，runtime_applied=False。"""
        result = _handle_config_set({"key": "trace.enabled", "value": True})
        data = json.loads(result)
        assert data["runtime_applied"] is False
        assert load_settings()["trace"]["enabled"] is True

    def test_set_denied_outside_whitelist(self, settings_home):
        """白名单外 → permission_denied，不落盘。"""
        result = _handle_config_set(
            {"key": "llm.auth_token", "value": "stolen"}, agent_ref=_make_agent_ref()
        )
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"
        # 默认 settings 本来就有 llm 段（auth_token=""）——验证没被写入脏值
        assert load_settings()["llm"].get("auth_token") in ("", None)

    def test_set_denied_prefix_lookalike(self, settings_home):
        """前缀相似（更长/更短）→ 拒。"""
        for key in ("notifications", "notifications.enabled.more"):
            result = _handle_config_set({"key": key, "value": True})
            data = json.loads(result)
            assert data["error_type"] == "permission_denied"

    def test_set_coerce_bool(self, settings_home):
        """现有 bool → bool(value) 强转（传 1 → True）。"""
        settings = load_settings()
        settings["notifications"] = {"enabled": True}
        save_settings(settings)

        result = _handle_config_set(
            {"key": "notifications.enabled", "value": 1},
            agent_ref=_make_agent_ref(),
        )
        data = json.loads(result)
        assert data["value"] is True
        assert load_settings()["notifications"]["enabled"] is True

    def test_set_coerce_int_from_string(self, settings_home):
        """现有 int → int(value) 强转（LLM 传字符串数字也能落）。"""
        settings = load_settings()
        settings["trace"] = {"retention_days": 7}
        save_settings(settings)

        result = _handle_config_set(
            {"key": "trace.retention_days", "value": "14"},
            agent_ref=_make_agent_ref(),
        )
        data = json.loads(result)
        assert data["value"] == 14
        assert load_settings()["trace"]["retention_days"] == 14

    def test_set_coerce_int_garbage_rejected(self, settings_home):
        """int 键传不可转值 → invalid_args（不落盘）。"""
        settings = load_settings()
        settings["trace"] = {"retention_days": 7}
        save_settings(settings)

        result = _handle_config_set(
            {"key": "trace.retention_days", "value": "abc"},
            agent_ref=_make_agent_ref(),
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"
        assert load_settings()["trace"]["retention_days"] == 7

    def test_set_old_value_recorded(self, settings_home):
        """成功响应里带 old_value（审计友好）。"""
        settings = load_settings()
        settings["trace"] = {"enabled": True}
        save_settings(settings)

        result = _handle_config_set(
            {"key": "trace.enabled", "value": False}, agent_ref=_make_agent_ref()
        )
        data = json.loads(result)
        assert data["old_value"] is True

    def test_set_missing_value(self, settings_home):
        """缺 value → invalid_args。"""
        result = _handle_config_set(
            {"key": "trace.enabled"}, agent_ref=_make_agent_ref()
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"

    def test_set_missing_key(self, settings_home):
        """缺 key → invalid_args。"""
        result = _handle_config_set({"value": True}, agent_ref=_make_agent_ref())
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"


# ---------------------------------------------------------------------------
# 4. CONFIG_CHANGE hook
# ---------------------------------------------------------------------------

class TestConfigChangeHook:
    def test_hook_fired_with_changed_keys(self, settings_home):
        """set 成功 → run_config_change 一次，payload 带 changed_keys。"""
        hooks = MagicMock()
        ref = _make_agent_ref()
        ref.hooks_registry = hooks
        result = _handle_config_set(
            {"key": "trace.enabled", "value": False},
            agent_ref=ref,
            hooks_registry=hooks,
            session_id="sess_1",
        )
        data = json.loads(result)
        assert "error" not in data

        hooks.run_config_change.assert_called_once()
        payload = hooks.run_config_change.call_args[0][0]
        assert payload["changed_keys"] == ["trace.enabled"]
        assert payload["session_id"] == "sess_1"

    def test_hook_exception_fail_open(self, settings_home):
        """hook 抛异常 → set 仍算成功（fail-open，hook 不阻塞配置写入）。"""
        hooks = MagicMock()
        hooks.run_config_change.side_effect = RuntimeError("hook boom")
        result = _handle_config_set(
            {"key": "trace.enabled", "value": False},
            agent_ref=_make_agent_ref(),
            hooks_registry=hooks,
        )
        data = json.loads(result)
        assert "error" not in data
        assert load_settings()["trace"]["enabled"] is False

    def test_no_hook_registry_ok(self, settings_home):
        """hooks_registry=None → 跳过 hook，不报错。"""
        result = _handle_config_set(
            {"key": "trace.enabled", "value": True},
            agent_ref=_make_agent_ref(),
            hooks_registry=None,
        )
        data = json.loads(result)
        assert "error" not in data


# ---------------------------------------------------------------------------
# 5. handler 契约 + schema 契约 + 注册
# ---------------------------------------------------------------------------

class TestContract:
    @pytest.mark.parametrize("handler", [_handle_config_get, _handle_config_set])
    def test_handler_signature_matches_dispatch_contract(self, handler):
        """handler 必须 (args, **kwargs)——registry.dispatch 调
        handler(args, **dispatch_kwargs)，缺 VAR_KEYWORD 会变
        silent-dead-code（CCAR8 教训）。"""
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        assert len(params) >= 1
        assert params[0].name == "args"
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    @pytest.mark.parametrize("schema", [CONFIG_GET_SCHEMA, CONFIG_SET_SCHEMA])
    def test_schema_uses_openai_parameters_key(self, schema):
        """schema 参数键必须是 "parameters"（OpenAI），不是 inputSchema。"""
        assert "parameters" in schema
        assert "inputSchema" not in schema
        assert isinstance(schema["parameters"].get("properties"), dict)

    def test_registered_in_core_toolset(self):
        """两个工具已注册且属于 core toolset。"""
        for name in ("config_get", "config_set"):
            entry = registry.get(name)
            assert entry is not None, f"{name} 未注册"
            assert entry.toolset == "core"

    def test_concurrency_classification(self):
        """config_get 只读（SAFE）config_set 写盘（UNSAFE）。"""
        assert registry.get("config_get").isConcurrencySafe is True
        assert registry.get("config_set").isConcurrencySafe is False

    def test_core_tools_list_updated(self):
        """_CORE_TOOLS 含两个新工具（LLM 可见性）。"""
        from toolsets import _CORE_TOOLS
        assert "config_get" in _CORE_TOOLS
        assert "config_set" in _CORE_TOOLS
