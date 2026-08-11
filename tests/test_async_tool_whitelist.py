"""Task F: async 子代理工具白名单（ASYNC_AGENT_ALLOWED_TOOLS）测试。

测试要点：
1. 白名单常量存在且内容正确
2. _delegate_async 把白名单应用到子代理（child agent 的 toolsets + disabled_tools）
3. sync 模式不受影响（回归保护）
4. config 开关（async_tool_whitelist_enabled / async_disallowed_tools）
5. fail-open：白名单配置错误时不崩
6. 端到端：通过 _run_child 验证 async 子代理确实带白名单
"""

import json
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from toolsets import (
    ASYNC_AGENT_ALLOWED_TOOLSETS,
    ASYNC_AGENT_DISALLOWED_TOOLS,
    resolve_toolset,
)
from tools.delegate_tool import (
    _delegate_async,
    _handle_delegate_task,
    get_delegation_queue,
)


# ---------------------------------------------------------------------------
# 1. 白名单常量
# ---------------------------------------------------------------------------

class TestAsyncWhitelistConstants:
    """常量定义正确性。"""

    def test_allowed_toolsets_is_set(self):
        assert isinstance(ASYNC_AGENT_ALLOWED_TOOLSETS, (set, frozenset))

    def test_allowed_toolsets_contains_core(self):
        """core 工具集应在白名单里（只读+受限写工具）。"""
        assert "core" in ASYNC_AGENT_ALLOWED_TOOLSETS

    def test_allowed_toolsets_excludes_team(self):
        """team 工具集不在白名单里（影响其他 agent 进程）。"""
        assert "team" not in ASYNC_AGENT_ALLOWED_TOOLSETS

    def test_allowed_toolsets_excludes_bg(self):
        """bg 工具集不在白名单里（后台任务嵌套）。"""
        assert "bg" not in ASYNC_AGENT_ALLOWED_TOOLSETS

    def test_disallowed_tools_is_set(self):
        assert isinstance(ASYNC_AGENT_DISALLOWED_TOOLS, (set, frozenset))

    def test_disallowed_tools_includes_bg_start(self):
        """bg_start 有全局副作用（启动子进程），应禁。"""
        assert "bg_start" in ASYNC_AGENT_DISALLOWED_TOOLS

    def test_disallowed_tools_includes_team_shutdown(self):
        """team_shutdown 杀其他 agent，应禁。"""
        assert "team_shutdown" in ASYNC_AGENT_DISALLOWED_TOOLS

    def test_disallowed_tools_includes_team_spawn(self):
        """team_spawn 嵌套派生，应禁。"""
        assert "team_spawn" in ASYNC_AGENT_DISALLOWED_TOOLS

    def test_disallowed_tools_includes_task_complete(self):
        """task_complete 推进全局任务状态机，应禁。"""
        assert "task_complete" in ASYNC_AGENT_DISALLOWED_TOOLS

    def test_disallowed_does_not_block_read_file(self):
        """白名单不能太严：只读工具不应被禁。"""
        assert "read_file" not in ASYNC_AGENT_DISALLOWED_TOOLS

    def test_disallowed_does_not_block_terminal(self):
        """terminal 走权限闸门已经够安全，不应在 async 里二次禁。"""
        assert "terminal" not in ASYNC_AGENT_DISALLOWED_TOOLS


# ---------------------------------------------------------------------------
# 2. _delegate_async 应用白名单
# ---------------------------------------------------------------------------

class TestDelegateAsyncAppliesWhitelist:
    """_delegate_async 把白名单应用到 _run_child 的 kwargs。"""

    def test_async_passes_restricted_toolsets_to_run_child(self):
        """_delegate_async 调 _run_child 时传的 kwargs 应含受限制的 toolsets。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                enabled_toolsets=["core", "team", "bg"],
            )
            # 等后台线程跑完
            time.sleep(0.1)

        # toolsets 应该被过滤：只保留白名单内的
        passed_toolsets = captured["kwargs"].get("enabled_toolsets") or []
        # 即使调用方传了 team/bg，async 模式应过滤掉
        for ts in passed_toolsets:
            assert ts in ASYNC_AGENT_ALLOWED_TOOLSETS, (
                f"async 子代理不应暴露 toolset={ts}（不在白名单）"
            )

    def test_async_adds_disallowed_to_config(self):
        """_delegate_async 传给 _run_child 的 config 应含 disabled_tools。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                config={},
            )
            time.sleep(0.1)

        cfg = captured["kwargs"].get("config") or {}
        disabled = cfg.get("disabled_tools") or []
        for tool_name in ASYNC_AGENT_DISALLOWED_TOOLS:
            assert tool_name in disabled, (
                f"async 子代理的 config.disabled_tools 应含 {tool_name}"
            )


# ---------------------------------------------------------------------------
# 3. sync 模式不受影响（回归保护）
# ---------------------------------------------------------------------------

class TestSyncModeUnaffected:
    """sync 模式（background=False）不应应用白名单。"""

    def test_sync_does_not_filter_toolsets(self):
        """sync 模式下 enabled_toolsets 原样传给 _run_child。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        from tools.delegate_tool import _delegate_sync
        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_sync(
                goal="test", context="", role="leaf",
                enabled_toolsets=["core", "team", "bg"],
            )

        passed = captured["kwargs"].get("enabled_toolsets") or []
        # sync 模式应保留调用方传的全部 toolsets
        assert "team" in passed, "sync 模式不应过滤 toolsets"
        assert "bg" in passed

    def test_sync_does_not_add_disabled_tools(self):
        """sync 模式不应在 config 里塞 disabled_tools。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        from tools.delegate_tool import _delegate_sync
        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_sync(
                goal="test", context="", role="leaf",
                config={"existing": True},
            )

        cfg = captured["kwargs"].get("config") or {}
        # sync 模式不应注入 disabled_tools
        assert "disabled_tools" not in cfg or cfg["disabled_tools"] is None


# ---------------------------------------------------------------------------
# 4. config 开关
# ---------------------------------------------------------------------------

class TestConfigSwitches:
    """config.delegation 的两个 flag 生效。"""

    def test_whitelist_disabled_via_config(self):
        """config.delegation.async_tool_whitelist_enabled=False 时不过滤。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cfg = {
            "delegation": {
                "async_tool_whitelist_enabled": False,
            }
        }

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                enabled_toolsets=["core", "team"],
                config=cfg,
            )
            time.sleep(0.1)

        # 白名单关闭后 toolsets 不应被过滤
        passed = captured["kwargs"].get("enabled_toolsets") or []
        assert "team" in passed, "白名单关闭时 team 不应被过滤"

    def test_user_extended_disallowed_via_config(self):
        """config.delegation.async_disallowed_tools 扩展禁用列表。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cfg = {
            "delegation": {
                "async_disallowed_tools": ["memory", "web_search"],
            }
        }

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                config=cfg,
            )
            time.sleep(0.1)

        child_cfg = captured["kwargs"].get("config") or {}
        disabled = child_cfg.get("disabled_tools") or []
        # 用户扩展的应被加入
        assert "memory" in disabled
        assert "web_search" in disabled
        # 内置的也在
        assert "bg_start" in disabled


# ---------------------------------------------------------------------------
# 5. fail-open
# ---------------------------------------------------------------------------

class TestFailOpen:
    """白名单机制异常不崩主流程。"""

    def test_no_config_does_not_crash(self):
        """config 为空时 _delegate_async 正常执行。"""
        with patch("tools.delegate_tool._run_child", return_value="ok"):
            result = _delegate_async(
                goal="test", context="", role="leaf",
            )
            time.sleep(0.1)
        data = json.loads(result)
        assert data["success"] is True

    def test_none_config_does_not_crash(self):
        """config=None 时 _delegate_async 正常执行。"""
        with patch("tools.delegate_tool._run_child", return_value="ok"):
            result = _delegate_async(
                goal="test", context="", role="leaf",
                config=None,
            )
            time.sleep(0.1)
        data = json.loads(result)
        assert data["success"] is True


# ---------------------------------------------------------------------------
# 6. 端到端：通过 _handle_delegate_task 起子代理验证白名单
# ---------------------------------------------------------------------------

class TestEndToEndThroughHandler:
    """通过工具入口（_handle_delegate_task）验证 async 白名单生效。"""

    @pytest.mark.asyncio
    async def test_background_true_triggers_whitelist(self):
        """background=True 走 _delegate_async 路径，白名单生效。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            result = await _handle_delegate_task.__wrapped__ if hasattr(_handle_delegate_task, '__wrapped__') else None
            # 直接调同步入口
            _handle_delegate_task(
                {"goal": "test async", "background": True},
            )
            time.sleep(0.15)

        # 验证白名单已应用
        assert "kwargs" in captured, "_run_child 应被调用"
        passed = captured["kwargs"].get("enabled_toolsets") or []
        for ts in passed:
            assert ts in ASYNC_AGENT_ALLOWED_TOOLSETS

    @pytest.mark.asyncio
    async def test_background_false_no_whitelist(self):
        """background=False 走 _delegate_sync 路径，白名单不生效。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _handle_delegate_task(
                {"goal": "test sync", "background": False},
                enabled_toolsets=["core", "team"],
            )

        # sync 模式不过滤
        passed = captured["kwargs"].get("enabled_toolsets") or []
        assert "team" in passed


# ---------------------------------------------------------------------------
# 7. config.py 默认值
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """config.py 的 delegation 节应有新 flag。"""

    def test_default_config_has_async_whitelist_enabled(self):
        from config import DEFAULT_CONFIG
        delegation = DEFAULT_CONFIG.get("delegation", {})
        assert "async_tool_whitelist_enabled" in delegation
        # 默认 True（安全默认 > 事后补救）
        assert delegation["async_tool_whitelist_enabled"] is True

    def test_default_config_has_async_disallowed_tools(self):
        from config import DEFAULT_CONFIG
        delegation = DEFAULT_CONFIG.get("delegation", {})
        assert "async_disallowed_tools" in delegation
        # 默认空列表（用户可扩展）
        assert delegation["async_disallowed_tools"] == []
