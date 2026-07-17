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
