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


# ---------------------------------------------------------------------------
# Task 3: browser_snapshot + browser_click + browser_type
# ---------------------------------------------------------------------------

from tools.browser_tool import (  # noqa: E402
    _handle_browser_snapshot,
    _handle_browser_click,
    _handle_browser_type,
)


def test_snapshot_success():
    """mock session.snapshot 返树 → 返回。"""
    fake_session = MagicMock()
    fake_session.snapshot.return_value = {
        "tree": {"role": "WebArea", "name": "Test"},
        "truncated": False,
        "chars": 50,
    }
    result = _handle_browser_snapshot(
        {},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["chars"] == 50
    assert data["truncated"] is False


def test_snapshot_custom_max_chars():
    """max_chars 参数透传到 session.snapshot。"""
    fake_session = MagicMock()
    fake_session.snapshot.return_value = {
        "tree": {}, "truncated": True, "chars": 100,
    }
    _handle_browser_snapshot(
        {"max_chars": 100},
        agent_ref=_make_agent_with_session(fake_session),
    )
    fake_session.snapshot.assert_called_once_with(100)


def test_snapshot_browser_unavailable():
    result = _handle_browser_snapshot(
        {}, agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


def test_click_empty_ref():
    result = _handle_browser_click(
        {"ref": ""},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert "error" in data


def test_click_stale_ref():
    """session.resolve_ref 返 None → stale_ref。"""
    fake_session = MagicMock()
    fake_session.resolve_ref.return_value = None
    result = _handle_browser_click(
        {"ref": "a99"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["error_type"] == "stale_ref"


def test_click_success():
    """mock resolve_ref + page.click → success。"""
    fake_session = MagicMock()
    fake_session.resolve_ref.return_value = 'button:has-text("Submit")'
    fake_page = MagicMock()
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_click(
        {"ref": "a1"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.click.assert_called_once_with('button:has-text("Submit")', timeout=10000)


def test_click_browser_unavailable():
    result = _handle_browser_click(
        {"ref": "a1"}, agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


def test_type_success():
    """mock session + page.fill → success。"""
    fake_session = MagicMock()
    fake_session.resolve_ref.return_value = 'input[aria-label="Search"]'
    fake_page = MagicMock()
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_type(
        {"ref": "a1", "text": "hello"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.fill.assert_called_once_with('input[aria-label="Search"]', "hello")
    fake_page.press.assert_not_called()  # submit=False


def test_type_with_submit():
    """submit=True → page.fill + page.press('Enter')。"""
    fake_session = MagicMock()
    fake_session.resolve_ref.return_value = 'input[aria-label="Q"]'
    fake_page = MagicMock()
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_type(
        {"ref": "a1", "text": "q", "submit": True},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.press.assert_called_once_with('input[aria-label="Q"]', "Enter")
