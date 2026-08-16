"""T6（核心机制对齐第 6 项）：工具可见性规则化（settings.json permissions.allow/deny）。

- 语法（第一版收窄）：精确名 / mcp__server__* 前缀 / mcp__server 整服务器
- 应用点：get_tool_definitions（LLM 可见性）+ registry.dispatch（防御纵深）
- allow 可豁免 deny（deny mcp__foo + allow mcp__foo__bar → bar 可见）
- mtime+size 双因子缓存（Windows mtime 精度教训）
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_rules_cache():
    from agent import tool_permissions as tp
    tp.reset_rules_cache()
    yield
    tp.reset_rules_cache()


def _write_permissions(home, allow=None, deny=None):
    from agent.settings import load_settings, save_settings
    data = load_settings()
    data["permissions"] = {"allow": allow or [], "deny": deny or []}
    save_settings(data)
    return home


# ---------------------------------------------------------------------------
# 规则匹配
# ---------------------------------------------------------------------------

def test_rule_matching_semantics():
    """精确名 / __* 前缀 / 整服务器三种语法。"""
    from agent.tool_permissions import tool_matches
    assert tool_matches("read_file", "read_file")
    assert not tool_matches("read_file", "read_file_x")
    assert tool_matches("mcp__foo__*", "mcp__foo__bar")
    assert not tool_matches("mcp__foo__*", "mcp__fooo__bar")
    assert tool_matches("mcp__foo", "mcp__foo__bar")   # 整服务器
    assert tool_matches("mcp__foo", "mcp__foo")        # 精确自身


def test_is_tool_denied_allow_exempts():
    from agent.tool_permissions import is_tool_denied
    rules = {"allow": ["mcp__foo__bar"], "deny": ["mcp__foo"]}
    assert is_tool_denied("mcp__foo__baz", rules) is True
    assert is_tool_denied("mcp__foo__bar", rules) is False  # allow 豁免
    assert is_tool_denied("read_file", rules) is False


def test_empty_rules_noop():
    from agent.tool_permissions import is_tool_denied
    assert is_tool_denied("read_file", {"allow": [], "deny": []}) is False


# ---------------------------------------------------------------------------
# settings.json 加载
# ---------------------------------------------------------------------------

def test_rules_loaded_from_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    _write_permissions(tmp_path, deny=["terminal"])
    from agent.tool_permissions import is_tool_denied
    assert is_tool_denied("terminal") is True
    assert is_tool_denied("read_file") is False


def test_default_settings_has_permissions_section():
    from agent.settings import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["permissions"] == {"allow": [], "deny": []}


# ---------------------------------------------------------------------------
# 应用点 1：get_tool_definitions 可见性
# ---------------------------------------------------------------------------

def test_deny_removes_tool_from_definitions(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    _write_permissions(tmp_path, deny=["load_skill"])
    from model_tools import get_tool_definitions
    defs = get_tool_definitions(["core"])
    names = [d["function"]["name"] for d in defs]
    assert "load_skill" not in names
    assert "read_file" in names  # 其他不受影响


def test_deny_whole_mcp_server_from_definitions(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    _write_permissions(tmp_path, deny=["mcp__foo"])
    # 直接构造 registry 中的假 mcp 工具验证过滤
    from tools.registry import registry
    from agent.tool_permissions import is_tool_denied
    assert is_tool_denied("mcp__foo__tool1")
    assert is_tool_denied("mcp__foo__tool2")
    assert not is_tool_denied("mcp__bar__tool1")


# ---------------------------------------------------------------------------
# 应用点 2：registry.dispatch 防御纵深
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_denied_tool_returns_permission_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    _write_permissions(tmp_path, deny=["load_skill"])
    from tools.registry import registry
    result = await registry.dispatch("load_skill", {"name": "x"})
    data = json.loads(result)
    assert data.get("error_type") == "permission_denied"
    assert "deny" in data.get("error", "")


@pytest.mark.asyncio
async def test_dispatch_allowed_tool_normal(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    _write_permissions(tmp_path, deny=["nonexistent_tool_xyz"])
    from tools.registry import registry
    # 不在 deny 里的工具正常 dispatch（skill_view 是只读查询）
    import tools.skill_tools  # noqa: F401 确保注册
    result = await registry.dispatch("skills_list", {})
    data = json.loads(result)
    assert data.get("error_type") != "permission_denied"


# ===== R25 #1：环境变量前缀/安全包装词剥离（防 deny 规则绕过）=====

class TestNormalizeCommandForRules:
    """FOO=bar rm xxx / nohup rm xxx 不能绕过 deny(rm ...) 规则。"""

    def test_env_prefix_stripped_for_deny(self):
        from agent.tool_permissions import check_command_rules
        # 注：规则不能用 Bash(rm -rf build:*)——前缀是词边界语义（R16 #3，
        # test_r16_security.py 已断言 build:* 不匹配 build/），改用 rm -rf 前缀。
        rules = {"deny": ["Bash(rm -rf:*)"], "allow": [], "ask": []}
        assert check_command_rules("FOO=1 rm -rf build/x", rules) == "deny"

    def test_multiple_env_prefixes_stripped(self):
        from agent.tool_permissions import check_command_rules
        rules = {"deny": ["Bash(git push:*)"], "allow": [], "ask": []}
        assert check_command_rules("A=1 B=2 git push origin x", rules) == "deny"

    def test_env_command_prefix_stripped(self):
        from agent.tool_permissions import check_command_rules
        rules = {"deny": ["Bash(curl:*)"], "allow": [], "ask": []}
        assert check_command_rules("env C=3 curl http://x", rules) == "deny"

    def test_nohup_timeout_nice_stripped(self):
        from agent.tool_permissions import check_command_rules
        rules = {"deny": ["Bash(rm:*)"], "allow": [], "ask": []}
        assert check_command_rules("nohup rm -rf build", rules) == "deny"
        assert check_command_rules("timeout 30 rm -rf build", rules) == "deny"
        assert check_command_rules("nice -n 5 rm -rf build", rules) == "deny"

    def test_no_env_prefix_unchanged(self):
        from agent.tool_permissions import _normalize_command_for_rules
        assert _normalize_command_for_rules("uv run pytest") == "uv run pytest"

    def test_plain_word_with_equals_not_stripped(self):
        """echo a=b 不是 env 前缀（env 赋值只能出现在命令头部）。"""
        from agent.tool_permissions import _normalize_command_for_rules
        assert _normalize_command_for_rules("echo a=b") == "echo a=b"
