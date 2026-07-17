"""13 个 browser_* handler 行为测试。"""
import json
from unittest.mock import MagicMock

import pytest

from tools.browser_tool import (
    _handle_browser_navigate,
    _handle_browser_close,
)


# ---------------------------------------------------------------------------
# _get_session helper
# ---------------------------------------------------------------------------

def _make_agent_with_session(session=None):
    """构造 mock agent_ref，可选挂 session。"""
    agent = MagicMock()
    agent.browser_session = session
    return agent


# ---------------------------------------------------------------------------
# browser_navigate
# ---------------------------------------------------------------------------

def test_navigate_blocks_file_scheme():
    result = _handle_browser_navigate(
        {"url": "file:///etc/passwd"},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert data["error_type"] == "unsafe_url"


def test_navigate_blocks_data_scheme():
    result = _handle_browser_navigate(
        {"url": "data:text/plain,blob"},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert data["error_type"] == "unsafe_url"


def test_navigate_blocks_javascript_scheme():
    result = _handle_browser_navigate(
        {"url": "javascript:alert(1)"},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert data["error_type"] == "unsafe_url"


def test_navigate_blocks_empty_url():
    result = _handle_browser_navigate(
        {"url": ""},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert "error" in data


def test_navigate_browser_unavailable():
    """agent_ref=None → browser_unavailable。"""
    result = _handle_browser_navigate(
        {"url": "https://example.com"},
        agent_ref=None,
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


def test_navigate_session_not_initialized():
    """agent.browser_session=None → browser_unavailable。"""
    result = _handle_browser_navigate(
        {"url": "https://example.com"},
        agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


def test_navigate_success():
    """mock session.get_page().goto 返 response → 返回 success JSON。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_response = MagicMock()
    fake_response.status = 200
    fake_page.goto.return_value = fake_response
    fake_page.url = "https://example.com/"
    fake_page.title.return_value = "Example"
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_navigate(
        {"url": "https://example.com"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["url"] == "https://example.com/"
    assert data["title"] == "Example"
    assert data["status"] == 200


def test_navigate_handles_timeout():
    """page.goto 抛异常 → navigation_error。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.goto.side_effect = TimeoutError("navigation timeout")
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_navigate(
        {"url": "https://example.com"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["error_type"] == "navigation_error"


# ---------------------------------------------------------------------------
# browser_close
# ---------------------------------------------------------------------------

def test_close_calls_cleanup():
    """browser_close → session.cleanup()。"""
    fake_session = MagicMock()
    result = _handle_browser_close(
        {},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_session.cleanup.assert_called_once()


def test_close_browser_unavailable():
    """agent.browser_session=None → browser_unavailable。"""
    result = _handle_browser_close(
        {},
        agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"
