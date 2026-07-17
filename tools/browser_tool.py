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
