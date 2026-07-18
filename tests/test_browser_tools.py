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


# ---------------------------------------------------------------------------
# Task 4: browser_scroll + browser_press_key + browser_back + browser_forward
# ---------------------------------------------------------------------------

from tools.browser_tool import (  # noqa: E402
    _handle_browser_scroll,
    _handle_browser_press_key,
    _handle_browser_back,
    _handle_browser_forward,
)


def test_scroll_down_main_page():
    """ref 不传 → 滚主页面（page.mouse.wheel）。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_session.get_page.return_value = fake_page
    fake_session.resolve_ref.return_value = None  # 不 resolve

    result = _handle_browser_scroll(
        {"direction": "down", "amount": 2},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    # page.mouse.wheel(dy=...) 被调用
    assert fake_page.mouse.wheel.called


def test_press_key_enter():
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_press_key(
        {"key": "Enter"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.press.assert_called_once_with("body", "Enter")


def test_press_key_empty():
    result = _handle_browser_press_key(
        {"key": ""},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert "error" in data


def test_back_calls_go_back():
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.url = "https://example.com/back"
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_back(
        {}, agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.go_back.assert_called_once()


def test_forward_calls_go_forward():
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.url = "https://example.com/forward"
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_forward(
        {}, agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.go_forward.assert_called_once()


# ---------------------------------------------------------------------------
# Task 5: browser_get_images + browser_console + browser_vision
# ---------------------------------------------------------------------------

from tools.browser_tool import (  # noqa: E402
    _handle_browser_get_images,
    _handle_browser_console,
    _handle_browser_vision,
)


def test_get_images_extracts_img_urls():
    """mock page.eval_on_selector_all 返 URL list。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.eval_on_selector_all.return_value = [
        {"src": "https://example.com/a.png", "width": 200, "height": 100},
        {"src": "https://example.com/b.png", "width": 50, "height": 50},
    ]
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_get_images(
        {"min_width": 100, "min_height": 100},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    # width=50 < min_width=100 → 过滤
    assert len(data["images"]) == 1
    assert data["images"][0]["src"] == "https://example.com/a.png"


def test_console_returns_logs():
    """mock page.no_more_dialogs / 用 getattr 模拟 console 消息。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    # Playwright 真实 API 没直接读取历史 console 的方法
    # 我们的 handler 应该订阅 console event + 缓存到 page._harvil_console_logs
    # 测试里 mock 一个缓存属性
    fake_page._harvil_console_logs = [
        {"type": "log", "text": "hello"},
        {"type": "error", "text": "oops"},
        {"type": "warning", "text": "careful"},
    ]
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_console(
        {"level": "warning"},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    # level=warning 应包含 warning + error，不包含 log
    types = [entry["type"] for entry in data["logs"]]
    assert "warning" in types
    assert "error" in types
    assert "log" not in types


def test_vision_success():
    """mock session.get_page().screenshot + LLM client → success。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.screenshot.return_value = b"fake-png-bytes"
    fake_session.get_page.return_value = fake_page

    fake_agent = _make_agent_with_session(fake_session)
    # Task 3：显式置 _vision_client=None，确保走 _browser_vision_client 路径（兼容旧测试意图）
    fake_agent._vision_client = None
    # mock LLM client（agent 持有的 openai client）
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="A simple page"))]
    fake_client.chat.completions.create.return_value = fake_response
    fake_agent._browser_vision_client = fake_client  # 注入
    fake_agent.llm_client = None  # 防止 MagicMock 自动生成

    result = _handle_browser_vision(
        {"query": "describe"},
        agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert "A simple page" in data["description"]


def test_vision_no_api_key():
    """没 LLM client → vision_unavailable。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_page.screenshot.return_value = b"x"
    fake_session.get_page.return_value = fake_page

    fake_agent = _make_agent_with_session(fake_session)
    # Task 3：三层 client 全部 None
    fake_agent._vision_client = None
    fake_agent._browser_vision_client = None
    fake_agent.llm_client = None

    result = _handle_browser_vision(
        {"query": "describe"}, agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "vision_unavailable"


def test_get_images_browser_unavailable():
    result = _handle_browser_get_images(
        {}, agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


# ---------------------------------------------------------------------------
# Task 6: browser_cdp
# ---------------------------------------------------------------------------

from tools.browser_tool import _handle_browser_cdp  # noqa: E402


def test_cdp_empty_command():
    result = _handle_browser_cdp(
        {"command": ""},
        agent_ref=_make_agent_with_session(MagicMock()),
    )
    data = json.loads(result)
    assert "error" in data


def test_cdp_success():
    """mock page.context.new_cdp_session.send → success。"""
    fake_session = MagicMock()
    fake_page = MagicMock()
    fake_cdp_client = MagicMock()
    fake_cdp_client.send.return_value = {"value": 42}
    fake_page.context.new_cdp_session.return_value = fake_cdp_client
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_cdp(
        {"command": "Runtime.evaluate", "args": {"expression": "6*7"}},
        agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["result"] == {"value": 42}
    fake_cdp_client.send.assert_called_once_with(
        "Runtime.evaluate", {"expression": "6*7"},
    )


def test_cdp_browser_unavailable():
    result = _handle_browser_cdp(
        {"command": "Page.reload"},
        agent_ref=_make_agent_with_session(None),
    )
    data = json.loads(result)
    assert data["error_type"] == "browser_unavailable"


# ---------------------------------------------------------------------------
# Task 3 (image feature): browser_vision 三层回退
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_page():
    """mock Playwright page，screenshot 返 bytes。"""
    page = MagicMock()
    page.screenshot.return_value = b"fake-png-bytes"
    return page


def test_browser_vision_falls_back_to_vision_client(fake_page, monkeypatch):
    """browser_vision 优先用 _vision_client（Task 3 三层回退）。"""
    import json
    from unittest.mock import MagicMock
    from tools.browser_tool import _handle_browser_vision

    # 构造 mock agent：_vision_client 有，_browser_vision_client 无，llm_client 有
    vision_client = MagicMock()
    msg = MagicMock()
    msg.message.content = "from _vision_client"
    resp = MagicMock()
    resp.choices = [msg]
    vision_client.chat.completions.create.return_value = resp

    main_client = MagicMock()
    main_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="from llm_client"))]
    )

    fake_agent = MagicMock()
    fake_agent._vision_client = vision_client  # 优先
    fake_agent._browser_vision_client = "should-not-be-used"  # MagicMock 自动有，显式标记
    fake_agent.llm_client = main_client  # 不应被调用
    fake_agent.config = {"model": {"name": "test"}}

    # mock BrowserSession：_get_session 从 agent_ref.browser_session 取
    fake_session = MagicMock()
    fake_session.get_page.return_value = fake_page
    fake_agent.browser_session = fake_session

    result = _handle_browser_vision(
        {"query": "describe"},
        agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data.get("description") == "from _vision_client"
    # 主 client 不应被调用
    main_client.chat.completions.create.assert_not_called()


def test_browser_vision_falls_back_to_llm_client(fake_page):
    """三层都不存在 _vision_client 和 _browser_vision_client 时用 llm_client。"""
    import json
    from unittest.mock import MagicMock
    from tools.browser_tool import _handle_browser_vision

    main_client = MagicMock()
    msg = MagicMock()
    msg.message.content = "from llm_client"
    resp = MagicMock()
    resp.choices = [msg]
    main_client.chat.completions.create.return_value = resp

    fake_agent = MagicMock()
    fake_agent._vision_client = None
    # 让 getattr(fake_agent, "_browser_vision_client", None) 返回 None
    fake_agent._browser_vision_client = None
    fake_agent.llm_client = main_client
    fake_agent.config = {"model": {"name": "test-model"}}
    fake_session = MagicMock()
    fake_session.get_page.return_value = fake_page
    fake_agent.browser_session = fake_session

    result = _handle_browser_vision(
        {"query": "describe"},
        agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data.get("description") == "from llm_client"
