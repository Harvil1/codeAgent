"""sandbox_runner 单元测试。"""
import shutil
import sys
from unittest.mock import patch

import pytest


def test_sandbox_unavailable_error_is_runtime_error():
    """SandboxUnavailableError 必须是 RuntimeError 子类（调用方好捕获）。"""
    from agent.sandbox_runner import SandboxUnavailableError
    assert issubclass(SandboxUnavailableError, RuntimeError)


def test_is_available_linux_with_bwrap():
    """Linux 上 bwrap 存在 → True。"""
    from agent.sandbox_runner import is_available
    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value="/usr/bin/bwrap"):
        # 清缓存
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is True


def test_is_available_linux_without_bwrap():
    """Linux 上 bwrap 不存在 → False，reason 含'未安装'。"""
    from agent.sandbox_runner import is_available, availability_reason
    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value=None):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is False
        reason = availability_reason()
        assert "未安装" in reason or "bwrap" in reason


def test_is_available_macos_with_sandbox_exec():
    """macOS 上 sandbox-exec 存在 → True。"""
    from agent.sandbox_runner import is_available
    with patch("sys.platform", "darwin"), \
         patch("shutil.which", return_value="/usr/bin/sandbox-exec"):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is True


def test_is_available_windows_unsupported():
    """Windows 平台 → False，reason 明确说不支持。"""
    from agent.sandbox_runner import is_available, availability_reason
    with patch("sys.platform", "win32"):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is False
        reason = availability_reason()
        assert "Windows" in reason or "不支持" in reason


def test_is_available_caches_result(monkeypatch):
    """is_available 第二次调用不重新 which（缓存生效）。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None
    call_count = {"n": 0}

    def fake_which(name):
        call_count["n"] += 1
        return "/usr/bin/bwrap"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", fake_which)
    mod.is_available()
    mod.is_available()
    mod.is_available()
    assert call_count["n"] == 1  # 只调用 1 次
