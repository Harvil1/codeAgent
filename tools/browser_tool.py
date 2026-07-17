"""浏览器自动化工具集（13 个）。

基于 Playwright sync API。所有工具通过 agent_ref.browser_session 操作浏览器。
playwright 未装时 check_fn 返 False，工具自动隐藏。
"""
import json
import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 安全 helper
# ---------------------------------------------------------------------------

_BLOCKED_SCHEMES = {"file", "data", "javascript", "vbscript", "about"}
_ALLOWED_SCHEMES = {"http", "https"}


def _is_safe_url(url: str) -> Tuple[bool, str]:
    """返回 (safe, reason)。safe=False 时 reason 解释为什么。"""
    if not url or not url.strip():
        return False, "URL 为空"
    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        return False, f"URL 解析失败: {e}"
    scheme = (parsed.scheme or "").lower()
    if scheme in _BLOCKED_SCHEMES:
        return False, f"禁止的 scheme: {scheme}"
    if scheme not in _ALLOWED_SCHEMES:
        return False, f"非允许 scheme: {scheme}（只允许 http/https）"
    if not parsed.netloc:
        return False, "URL 缺少 netloc"
    return True, ""


def _check_browser_available() -> bool:
    """check_fn：playwright 是否可用。"""
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# session 取用 + 错误构造 helper
# ---------------------------------------------------------------------------

def _get_session(kwargs: dict):
    """从 kwargs 取 agent 持有的 session。

    返回 BrowserSession 实例或 None（agent_ref 未提供 / 无 browser_session 属性 /
    browser_session 为 None）。
    """
    agent = kwargs.get("agent_ref")
    if agent is None:
        return None
    return getattr(agent, "browser_session", None)


def _err(msg: str, error_type: Optional[str] = None) -> str:
    """构造 JSON 错误。error_type=None 时省略 error_type 字段。"""
    d = {"error": msg}
    if error_type:
        d["error_type"] = error_type
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

BROWSER_NAVIGATE_SCHEMA = {
    "name": "browser_navigate",
    "description": "导航到 URL。返回最终 URL + 标题 + HTTP status。会 lazy 启动 Chromium。",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "目标 URL（http/https）"},
        },
        "required": ["url"],
    },
}

BROWSER_CLOSE_SCHEMA = {
    "name": "browser_close",
    "description": "关闭浏览器（释放 Chromium 进程）。下次 navigate 会重启。",
    "parameters": {"type": "object", "properties": {}},
}

BROWSER_SNAPSHOT_SCHEMA = {
    "name": "browser_snapshot",
    "description": (
        "获取页面的 accessibility tree（无障碍树）。每个可交互节点分配 ref。"
        "LLM 通过 ref 调 click/type/scroll。默认截断到 8000 字符。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "max_chars": {
                "type": "integer", "default": 8000,
                "description": "返回的最大字符数",
            },
        },
    },
}

BROWSER_CLICK_SCHEMA = {
    "name": "browser_click",
    "description": "点击 snapshot 里 ref 指向的元素。",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "snapshot 返回的 ref（如 'a12'）"},
        },
        "required": ["ref"],
    },
}

BROWSER_TYPE_SCHEMA = {
    "name": "browser_type",
    "description": "在 snapshot 里 ref 指向的输入框输入文本。可选 submit=True 输完按 Enter。",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
            "text": {"type": "string"},
            "submit": {"type": "boolean", "default": False, "description": "输入后按 Enter"},
        },
        "required": ["ref", "text"],
    },
}

BROWSER_SCROLL_SCHEMA = {
    "name": "browser_scroll",
    "description": "滚动页面或某个元素。不传 ref 则滚主页面。",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "可选，滚某个元素；不传则滚主页面"},
            "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
            "amount": {"type": "integer", "default": 1, "description": "滚动步数（每次约一屏）"},
        },
    },
}

BROWSER_PRESS_KEY_SCHEMA = {
    "name": "browser_press_key",
    "description": "按键盘键（Enter、Tab、Escape、ArrowDown 等）。参见 KeyboardEvent.key。",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "键名（如 'Enter', 'Escape', 'ArrowDown'）"},
        },
        "required": ["key"],
    },
}

BROWSER_BACK_SCHEMA = {
    "name": "browser_back",
    "description": "浏览器后退。",
    "parameters": {"type": "object", "properties": {}},
}

BROWSER_FORWARD_SCHEMA = {
    "name": "browser_forward",
    "description": "浏览器前进。",
    "parameters": {"type": "object", "properties": {}},
}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def _handle_browser_navigate(args: dict, **kwargs) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return _err("url 不能为空")
    safe, reason = _is_safe_url(url)
    if not safe:
        return _err(f"URL 不安全: {reason}", "unsafe_url")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        page = session.get_page()
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        final_url = page.url
        title = page.title()
        status = response.status if response else None
        return json.dumps({
            "success": True,
            "url": final_url,
            "title": title,
            "status": status,
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"导航失败: {e}", "navigation_error")


def _handle_browser_close(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        session.cleanup()
        return json.dumps({"success": True}, ensure_ascii=False)
    except Exception as e:
        return _err(f"关闭失败: {e}", "close_error")


def _handle_browser_snapshot(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    max_chars = args.get("max_chars", 8000)
    try:
        result = session.snapshot(max_chars)
        return json.dumps({
            "success": True,
            **result,
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"snapshot 失败: {e}", "snapshot_error")


def _handle_browser_click(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    if not ref:
        return _err("ref 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        selector = session.resolve_ref(ref)
        if selector is None:
            return _err(
                f"无效 ref: {ref}（可能 snapshot 过期，请重新调 browser_snapshot）",
                "stale_ref",
            )
        page = session.get_page()
        page.click(selector, timeout=10000)
        return json.dumps({"success": True, "clicked": ref}, ensure_ascii=False)
    except Exception as e:
        return _err(f"点击失败: {e}", "click_error")


def _handle_browser_type(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    text = args.get("text", "")
    if not ref:
        return _err("ref 不能为空")
    if text is None:
        return _err("text 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        selector = session.resolve_ref(ref)
        if selector is None:
            return _err(
                f"无效 ref: {ref}（可能 snapshot 过期）",
                "stale_ref",
            )
        page = session.get_page()
        page.fill(selector, text)
        if args.get("submit"):
            page.press(selector, "Enter")
        return json.dumps(
            {"success": True, "typed": ref, "chars": len(text)},
            ensure_ascii=False,
        )
    except Exception as e:
        return _err(f"输入失败: {e}", "type_error")


def _handle_browser_scroll(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    direction = args.get("direction", "down")
    amount = args.get("amount", 1)
    ref = args.get("ref")
    try:
        page = session.get_page()
        dy = amount * 600 if direction == "down" else -amount * 600
        if ref:
            selector = session.resolve_ref(ref)
            if selector is None:
                return _err(f"无效 ref: {ref}", "stale_ref")
            el = page.query_selector(selector)
            if el:
                el.scroll_into_view_if_needed()
        else:
            page.mouse.wheel(0, dy)
        return json.dumps({
            "success": True, "direction": direction, "amount": amount,
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"滚动失败: {e}", "scroll_error")


def _handle_browser_press_key(args: dict, **kwargs) -> str:
    key = (args.get("key") or "").strip()
    if not key:
        return _err("key 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        page = session.get_page()
        page.press("body", key)
        return json.dumps({"success": True, "key": key}, ensure_ascii=False)
    except Exception as e:
        return _err(f"按键失败: {e}", "press_key_error")


def _handle_browser_back(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        page = session.get_page()
        page.go_back(wait_until="domcontentloaded", timeout=30000)
        return json.dumps({"success": True, "url": page.url}, ensure_ascii=False)
    except Exception as e:
        return _err(f"后退失败: {e}", "navigation_error")


def _handle_browser_forward(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        page = session.get_page()
        page.go_forward(wait_until="domcontentloaded", timeout=30000)
        return json.dumps({"success": True, "url": page.url}, ensure_ascii=False)
    except Exception as e:
        return _err(f"前进失败: {e}", "navigation_error")


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="browser_navigate", toolset="browser",
    schema=BROWSER_NAVIGATE_SCHEMA, handler=_handle_browser_navigate,
    check_fn=_check_browser_available, emoji="🌐",
)
registry.register(
    name="browser_close", toolset="browser",
    schema=BROWSER_CLOSE_SCHEMA, handler=_handle_browser_close,
    check_fn=_check_browser_available, emoji="✖",
)
registry.register(
    name="browser_snapshot", toolset="browser",
    schema=BROWSER_SNAPSHOT_SCHEMA, handler=_handle_browser_snapshot,
    check_fn=_check_browser_available, emoji="📸",
)
registry.register(
    name="browser_click", toolset="browser",
    schema=BROWSER_CLICK_SCHEMA, handler=_handle_browser_click,
    check_fn=_check_browser_available, emoji="👆",
)
registry.register(
    name="browser_type", toolset="browser",
    schema=BROWSER_TYPE_SCHEMA, handler=_handle_browser_type,
    check_fn=_check_browser_available, emoji="⌨️",
)
registry.register(
    name="browser_scroll", toolset="browser",
    schema=BROWSER_SCROLL_SCHEMA, handler=_handle_browser_scroll,
    check_fn=_check_browser_available, emoji="📜",
)
registry.register(
    name="browser_press_key", toolset="browser",
    schema=BROWSER_PRESS_KEY_SCHEMA, handler=_handle_browser_press_key,
    check_fn=_check_browser_available, emoji="⌨",
)
registry.register(
    name="browser_back", toolset="browser",
    schema=BROWSER_BACK_SCHEMA, handler=_handle_browser_back,
    check_fn=_check_browser_available, emoji="⬅",
)
registry.register(
    name="browser_forward", toolset="browser",
    schema=BROWSER_FORWARD_SCHEMA, handler=_handle_browser_forward,
    check_fn=_check_browser_available, emoji="➡",
)
