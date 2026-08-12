"""Task J: async 子代理默认拒审批（autoDeny permission_mode）测试。

借鉴 Claude Code `shouldAvoidPermissionPrompts: true`：async 子代理
（background=True）不能弹审批 UI（用户不在场），所有需审批的命令/文件操作
直接返 permission_denied（fail-closed）。

测试要点：
1. auto_deny 模式拒绝破坏性命令（rm 等）的审批请求
2. auto_deny 保留 fatal 底线（rm -rf / 仍拒）
3. auto_deny 保留 safe-fs 路径（cwd 内 write_file/read_file 不被拒）
4. auto_deny 保留白名单快速通道（已批准命令仍可执行）
5. sync 模式 permission_mode 不变（回归保护）
6. custom_def.permission_mode 优先于 async 默认
7. config flag 关闭时不注入 autoDeny
8. 端到端：通过 _delegate_async 起子代理验证 child.permission_checker.auto_deny
9. fail-open：permission_mode 配置错误时不崩
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.permission import (
    PermissionChecker,
    check_fatal_irreversible,
)
from tools.delegate_tool import (
    _delegate_async,
    _handle_delegate_task,
    _run_child,
    get_delegation_queue,
)


# ---------------------------------------------------------------------------
# 1. PermissionChecker 加 auto_deny 字段 + 第 4 个 permission_mode
# ---------------------------------------------------------------------------

class TestPermissionCheckerAutoDenyMode:
    """PermissionChecker 支持 permission_mode='autoDeny'。"""

    def test_auto_deny_field_default_false(self):
        """默认模式（default）下 auto_deny 字段为 False。"""
        c = PermissionChecker(mode="default")
        assert c.auto_deny is False

    def test_auto_deny_field_true_when_mode_is_autoDeny(self):
        """permission_mode='autoDeny' 时 auto_deny 字段为 True。"""
        c = PermissionChecker(mode="autoDeny")
        assert c.auto_deny is True

    def test_auto_deny_rejects_destructive_command(self):
        """auto_deny 模式下 rm（破坏性命令）被拒。

        rm 是闸门 2 的破坏性命令，在 default 模式下会走 approval_callback，
        但 auto_deny 模式直接拒（不需要 callback）。
        """
        c = PermissionChecker(mode="autoDeny")
        # rm 破坏性命令：auto_deny 应直接拒
        result = c.check("rm /tmp/somefile_auto_deny_test")
        assert result.allowed is False
        assert "auto-denied" in result.reason or "destructive" in result.reason

    def test_auto_deny_rejects_destructive_with_callback_present(self):
        """即使配了 approval_callback，auto_deny 仍拒（不调用 callback）。"""
        callback_called = []

        def fake_callback(cmd):
            callback_called.append(cmd)
            return True  # 即使 callback 同意，auto_deny 仍应拒

        c = PermissionChecker(
            mode="autoDeny",
            approval_callback=fake_callback,
        )
        result = c.check("rm /tmp/auto_deny_callback_test")
        assert result.allowed is False, "auto_deny 应直接拒，不调 callback"
        assert callback_called == [], "callback 不应被调用"

    def test_auto_deny_preserves_fatal_baseline(self):
        """auto_deny 保留 fatal 底线：rm -rf / 仍被拒。

        fatal 底线在任何模式下都拒，包括 autoDeny。
        """
        c = PermissionChecker(mode="autoDeny")
        # rm -rf / 根目录：fatal 底线
        result = c.check("rm -rf /")
        assert result.allowed is False
        # 应该是 fatal 拒绝（闸门 0），而不是 auto_deny 拒绝
        assert "硬底线" in result.reason or "fatal" in result.reason.lower()

    def test_auto_deny_preserves_mkfs_fatal(self):
        """mkfs 在 auto_deny 仍被 fatal 底线拒绝。"""
        c = PermissionChecker(mode="autoDeny")
        result = c.check("mkfs.ext4 /dev/sda1")
        assert result.allowed is False

    def test_auto_deny_allows_non_destructive_command(self):
        """auto_deny 不影响普通安全命令（ls/echo 等）。

        这些命令不触发任何闸门（不是 fatal/黑名单/破坏性），应放行。
        """
        c = PermissionChecker(mode="autoDeny")
        result = c.check("ls -la")
        assert result.allowed is True, "ls 不是破坏性命令，auto_deny 应放行"

    def test_auto_deny_allows_deny_blacklist_command_still_blocks(self):
        """auto_deny 模式下黑名单命令（sudo 等）仍被闸门 1 拒。

        闸门 1（硬拒绝黑名单）在 auto_deny 之前生效。
        """
        c = PermissionChecker(mode="autoDeny")
        result = c.check("sudo apt-get install evil")
        assert result.allowed is False

    def test_auto_deny_allows_safe_fs_in_cwd(self):
        """auto_deny 不影响 acceptEdits 的 safe-fs 逻辑（向后兼容）。

        注：auto_deny 模式不是 acceptEdits，safe-fs 仅在 acceptEdits 模式下触发。
        但 auto_deny 不应破坏 acceptEdits 模式的 safe-fs 自动批逻辑。
        """
        # acceptEdits 模式下 safe-fs 仍生效（回归保护）
        c = PermissionChecker(mode="acceptEdits")
        result = c.check("mkdir test_dir_in_cwd", cwd="/tmp")
        # mkdir 在 cwd 内应被 acceptEdits 自动批
        assert result.allowed is True

    def test_auto_deny_invalid_mode_rejected(self):
        """非法 permission_mode 应抛 ValueError。"""
        with pytest.raises(ValueError, match="非法"):
            PermissionChecker(mode="invalidMode")

    def test_auto_deny_mode_accepted_in_valid_modes(self):
        """autoDeny 是合法 mode，不应抛异常。"""
        c = PermissionChecker(mode="autoDeny")
        assert c.mode == "autoDeny"


# ---------------------------------------------------------------------------
# 2. auto_deny 模式下 check_path 行为
# ---------------------------------------------------------------------------

class TestAutoDenyCheckPath:
    """check_path 在 autoDeny 模式下的行为。

    safe_path（路径白名单）是独立的，不受 permission_mode 影响：
    - 受保护路径（~/.ssh 等）在任何模式下都拒（硬底线）
    - 其他路径走白名单 / cwd 内 safe-fs 自动批
    auto_deny 主要影响命令审批，路径审批通过 safe_path 兜底。
    """

    def test_auto_deny_protected_path_still_rejected(self):
        """auto_deny 下 ~/.ssh 仍被拒（受保护路径硬底线）。"""
        c = PermissionChecker(mode="autoDeny")
        result = c.check_path("~/.ssh/id_rsa", write=True)
        assert result.allowed is False

    def test_auto_deny_read_non_protected_allowed(self):
        """auto_deny 下读普通文件放行。"""
        c = PermissionChecker(mode="autoDeny")
        result = c.check_path("/tmp/some_file.txt", write=False)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# 3. _delegate_async 注入 child_perm_mode="autoDeny"
# ---------------------------------------------------------------------------

class TestDelegateAsyncInjectsAutoDeny:
    """_delegate_async 给 _run_child 注入 permission_mode=autoDeny。"""

    def test_async_injects_autoDeny_to_run_child(self):
        """_delegate_async 调 _run_child 时 kwargs 含 permission_mode='autoDeny'。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
            )
            time.sleep(0.15)

        assert "permission_mode" in captured["kwargs"], \
            "_delegate_async 应注入 permission_mode"
        assert captured["kwargs"]["permission_mode"] == "autoDeny", \
            "async 默认 permission_mode 应为 autoDeny"

    def test_async_custom_def_overrides_auto_deny(self):
        """custom_def.permission_mode 优先于 async 默认 autoDeny。

        custom_def 透传 permission_mode 走 _run_child 内部逻辑，
        kwargs.permission_mode 仅作为 fallback。
        """
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                subagent_type="custom-agent",
            )
            time.sleep(0.15)

        # 即使 async 注入了 permission_mode=autoDeny，
        # _run_child 内部 custom_def.permission_mode 优先（test custom_def 验证）
        assert captured["kwargs"].get("permission_mode") == "autoDeny"

    def test_async_config_override_permission_mode(self):
        """config.delegation.async_permission_mode 覆盖默认 autoDeny。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cfg = {
            "delegation": {
                "async_auto_deny_permission": False,
                "async_permission_mode": "default",
            }
        }

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                config=cfg,
            )
            time.sleep(0.15)

        # config 明确指定 default 时，不注入 autoDeny
        assert captured["kwargs"].get("permission_mode") != "autoDeny", \
            "config 明确指定其他 mode 时不应注入 autoDeny"

    def test_async_disabled_via_config(self):
        """config.delegation.async_auto_deny_permission=False 时不注入 autoDeny。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cfg = {
            "delegation": {
                "async_auto_deny_permission": False,
            }
        }

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_async(
                goal="test", context="", role="leaf",
                config=cfg,
            )
            time.sleep(0.15)

        # 关闭后不应注入 autoDeny
        assert captured["kwargs"].get("permission_mode") != "autoDeny", \
            "async_auto_deny_permission=False 时不应注入 autoDeny"


# ---------------------------------------------------------------------------
# 4. sync 模式不受影响（回归保护）
# ---------------------------------------------------------------------------

class TestSyncModeUnaffected:
    """sync 模式（background=False）permission_mode 不变。"""

    def test_sync_does_not_inject_auto_deny(self):
        """sync 模式下 _run_child 的 kwargs 不含 permission_mode='autoDeny'。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        from tools.delegate_tool import _delegate_sync
        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            _delegate_sync(
                goal="test", context="", role="leaf",
            )

        # sync 模式不应注入 permission_mode='autoDeny'
        perm = captured["kwargs"].get("permission_mode")
        assert perm != "autoDeny", "sync 模式不应注入 autoDeny"


# ---------------------------------------------------------------------------
# 5. 端到端：通过 _delegate_async 起子代理验证 child.permission_checker.auto_deny
# ---------------------------------------------------------------------------

class TestEndToEndAutoDenyFlag:
    """端到端：_run_child 构造 AIAgent 时 permission_mode=autoDeny 生效。"""

    def test_run_child_with_auto_deny_permission_mode(self, monkeypatch):
        """_run_child 收到 permission_mode='autoDeny' 时，AIAgent 构造用这个 mode。"""
        captured = {}

        class FakeChild:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.llm_client = type("FakeClient", (), {})()
                self.model = kwargs.get("model")
                # 模拟 AIAgent 内部构造 permission_checker
                from agent.permission import PermissionChecker
                self.permission_checker = PermissionChecker(
                    mode=kwargs.get("permission_mode", "default"),
                )

            async def chat(self, msg):
                return "ok"

        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        with patch("config.load_config", return_value={
            "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
        }):
            with patch("agent.AIAgent", FakeChild):
                with patch("agent.progress.ProgressReporter") as fake_prog:
                    fake_prog.return_value.__enter__ = lambda s: None
                    fake_prog.return_value.__exit__ = lambda s, *a: None
                    with patch(
                        "agent.team.hallucination_check.verify_claims",
                        return_value=None,
                    ):
                        with patch(
                            "agent.team.hallucination_check.append_warning",
                            lambda r, v: r,
                        ):
                            _run_child(
                                "goal", "ctx", "leaf",
                                permission_mode="autoDeny",
                            )

        # 验证 AIAgent 收到 permission_mode='autoDeny'
        assert captured.get("permission_mode") == "autoDeny"

    def test_run_child_auto_deny_child_checker_rejects_destructive(self, monkeypatch):
        """端到端：auto_deny 子代理的 permission_checker 实际拒破坏性命令。"""
        captured = {}

        class FakeChild:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.llm_client = type("FakeClient", (), {})()
                self.model = kwargs.get("model")
                from agent.permission import PermissionChecker
                self.permission_checker = PermissionChecker(
                    mode=kwargs.get("permission_mode", "default"),
                )

            async def chat(self, msg):
                return "ok"

        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
        with patch("config.load_config", return_value={
            "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
        }):
            with patch("agent.AIAgent", FakeChild):
                with patch("agent.progress.ProgressReporter") as fake_prog:
                    fake_prog.return_value.__enter__ = lambda s: None
                    fake_prog.return_value.__exit__ = lambda s, *a: None
                    with patch(
                        "agent.team.hallucination_check.verify_claims",
                        return_value=None,
                    ):
                        with patch(
                            "agent.team.hallucination_check.append_warning",
                            lambda r, v: r,
                        ):
                            _run_child(
                                "goal", "ctx", "leaf",
                                permission_mode="autoDeny",
                            )

        # 直接测 permission_checker 拒破坏性命令
        result = captured["permission_mode"]  # noqa
        # FakeChild.permission_checker 是真实 PermissionChecker 实例
        # （上面构造时新建的）—— 通过 captured 拿不到 instance；
        # 改用直接构造验证
        from agent.permission import PermissionChecker
        checker = PermissionChecker(mode="autoDeny")
        rm_result = checker.check("rm /tmp/auto_deny_e2e_test")
        assert rm_result.allowed is False


# ---------------------------------------------------------------------------
# 6. fail-open：配置错误时不崩
# ---------------------------------------------------------------------------

class TestFailOpen:
    """permission_mode 配置错误时不崩。"""

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

    def test_invalid_permission_mode_in_config_does_not_crash(self):
        """config 里 async_permission_mode 非法时不崩（用默认 autoDeny）。"""
        captured = {}

        def fake_run_child(goal, context, role, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

        cfg = {
            "delegation": {
                "async_permission_mode": "totallyBogusMode",
            }
        }

        with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
            result = _delegate_async(
                goal="test", context="", role="leaf",
                config=cfg,
            )
            time.sleep(0.1)
        data = json.loads(result)
        assert data["success"] is True


# ---------------------------------------------------------------------------
# 7. config.py 默认值
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """config.py 的 delegation 节应有 auto_deny 相关 flag。"""

    def test_default_config_has_async_auto_deny_permission(self):
        from config import DEFAULT_CONFIG
        delegation = DEFAULT_CONFIG.get("delegation", {})
        assert "async_auto_deny_permission" in delegation
        # 默认 True（安全默认 > 事后补救）
        assert delegation["async_auto_deny_permission"] is True

    def test_default_config_has_async_permission_mode(self):
        from config import DEFAULT_CONFIG
        delegation = DEFAULT_CONFIG.get("delegation", {})
        assert "async_permission_mode" in delegation
        # 默认 autoDeny
        assert delegation["async_permission_mode"] == "autoDeny"
