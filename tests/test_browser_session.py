"""BrowserSession + URL safety 测试。"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# URL safety（不依赖 playwright）
# ---------------------------------------------------------------------------

from tools.browser_tool import _is_safe_url


def test_safe_url_https():
    safe, _ = _is_safe_url("https://example.com/path?q=1")
    assert safe is True


def test_safe_url_http():
    safe, _ = _is_safe_url("http://localhost:8000")
    assert safe is True


def test_unsafe_url_file_scheme():
    safe, reason = _is_safe_url("file:///etc/passwd")
    assert safe is False
    assert "file" in reason


def test_unsafe_url_data_scheme():
    safe, reason = _is_safe_url("data:text/plain,blob")
    assert safe is False
    assert "data" in reason


def test_unsafe_url_javascript_scheme():
    safe, reason = _is_safe_url("javascript:alert(1)")
    assert safe is False
    assert "javascript" in reason


def test_unsafe_url_empty():
    safe, reason = _is_safe_url("")
    assert safe is False


def test_unsafe_url_no_scheme():
    safe, reason = _is_safe_url("example.com")
    # urlparse 不带 scheme → scheme="" → 不在 _ALLOWED_SCHEMES
    assert safe is False


def test_unsafe_url_missing_netloc():
    safe, reason = _is_safe_url("http://")
    assert safe is False
    assert "netloc" in reason


# ---------------------------------------------------------------------------
# check_fn（mock playwright import）
# ---------------------------------------------------------------------------

from tools.browser_tool import _check_browser_available


def test_check_browser_available_installed(monkeypatch):
    """playwright 已装 → True。"""
    # 默认应装好（Task 1.1 uv add 过），所以 True
    assert _check_browser_available() is True


def test_check_browser_available_not_installed(monkeypatch):
    """mock ImportError → False。"""
    import sys
    monkeypatch.setitem(sys.modules, "playwright", None)
    assert _check_browser_available() is False


# ---------------------------------------------------------------------------
# BrowserSession（mock playwright.sync_api）
# ---------------------------------------------------------------------------

from agent.browser_session import BrowserSession


def test_session_not_started_on_construct():
    """实例化后 _started=False。"""
    s = BrowserSession()
    assert s._started is False


def test_session_cleanup_when_not_started_noop():
    """未启动就 cleanup，no-op 不抛。"""
    s = BrowserSession()
    s.cleanup()  # 不应抛
    assert s._started is False


def test_session_cleanup_idempotent():
    """多次 cleanup 不抛。"""
    s = BrowserSession()
    s.cleanup()
    s.cleanup()
    s.cleanup()
    assert s._started is False


def test_session_lazy_init_calls_playwright(monkeypatch):
    """get_page 触发 sync_playwright().start() + chromium.launch + new_page。"""
    fake_pw = MagicMock()
    fake_browser = MagicMock()
    fake_page = MagicMock()
    fake_pw.chromium.launch.return_value = fake_browser
    fake_browser.new_page.return_value = fake_page

    fake_pw_ctx = MagicMock()
    fake_pw_ctx.start.return_value = fake_pw

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright",
        lambda: fake_pw_ctx,
    )

    s = BrowserSession(headless=True)
    assert s._started is False
    page = s.get_page()
    assert page is fake_page
    assert s._started is True
    fake_pw.chromium.launch.assert_called_once_with(headless=True)
    fake_browser.new_page.assert_called_once()


def test_session_cleanup_closes_all(monkeypatch):
    """cleanup 关 page/browser/playwright 三件。"""
    fake_pw = MagicMock()
    fake_browser = MagicMock()
    fake_page = MagicMock()
    fake_pw.chromium.launch.return_value = fake_browser
    fake_browser.new_page.return_value = fake_page
    fake_pw_ctx = MagicMock()
    fake_pw_ctx.start.return_value = fake_pw
    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright",
        lambda: fake_pw_ctx,
    )

    s = BrowserSession()
    s.get_page()
    s.cleanup()
    fake_page.close.assert_called_once()
    fake_browser.close.assert_called_once()
    fake_pw.stop.assert_called_once()
    assert s._started is False
