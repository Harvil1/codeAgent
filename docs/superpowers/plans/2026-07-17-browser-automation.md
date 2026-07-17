# 浏览器自动化工具集 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 引入 Playwright Python + 13 个 browser_* 工具 + BrowserSession 类，让 agent 能操作网页。

**Architecture:** 新建 `agent/browser_session.py` 管 Chromium 生命周期；`tools/browser_tool.py` 加 13 schema/handler；新 `"browser"` toolset 默认 off，需 config 启用；check_fn 在 playwright 未装时隐藏工具。

**Tech Stack:** Python 3.11+、Playwright sync API、pytest、uv

**Spec:** `docs/superpowers/specs/2026-07-17-browser-automation-design.md`

## Global Constraints

- **语言约定**：注释/文档/commit 用中文；代码标识符用英文（CLAUDE.md）
- **文件 I/O**：必须 `encoding="utf-8"`
- **工具结果契约**：handler 返回 JSON 字符串；错误用 `{"error": "...", "error_type": "..."}`
- **依赖管理**：用 `uv add`，不要手改 pyproject.toml
- **测试基线**：当前 740 测试通过；本计划完成后预期 740 + 22 = 762
- **Playwright 安装**：`uv add playwright` + 用户手动 `uv run playwright install chromium`
- **check_fn 不可见机制**：playwright 未装时工具自动隐藏，不报错

---

## 文件结构

| 文件 | 状态 | 职责 |
|---|---|---|
| `pyproject.toml` | ♻️ 改 | 加 `playwright>=1.40` 依赖 |
| `agent/browser_session.py` | 🆕 新增 | `BrowserSession` 类（lazy init/cleanup/snapshot + ref 分配） |
| `tools/browser_tool.py` | 🆕 新增 | 13 schema + 13 handler + `_check_browser_available` + `_is_safe_url` |
| `toolsets.py` | ♻️ 改 | 加 `"browser"` toolset |
| `agent/__init__.py` | ♻️ 改 | AIAgent 加 `browser_session` 属性 + shutdown cleanup |
| `tests/test_browser_session.py` | 🆕 新增 | BrowserSession + URL safety 测试（mock playwright） |
| `tests/test_browser_tools.py` | 🆕 新增 | 13 个 handler 行为测试（mock BrowserSession） |

---

## Task 1: Setup + BrowserSession + URL safety

**Files:**
- Modify: `pyproject.toml`（`uv add playwright`）
- Create: `agent/browser_session.py`
- Test: `tests/test_browser_session.py`（新建）

**Interfaces:**
- Consumes: 无（foundation task）
- Produces:
  - `BrowserSession(headless=True)` — lazy Chromium init via `get_page()`
  - `BrowserSession.cleanup()` — idempotent
  - `BrowserSession.snapshot(max_chars=8000) -> dict` — accessibility tree + ref assignment
  - `BrowserSession.resolve_ref(ref) -> Optional[str]` — returns CSS selector (used in Task 3)
  - `tools/browser_tool.py:_is_safe_url(url) -> tuple[bool, str]` — URL scheme 白名单
  - `tools/browser_tool.py:_check_browser_available() -> bool` — check_fn

- [ ] **Step 1.1: 加 playwright 依赖**

Run:
```bash
uv add "playwright>=1.40"
```

Expected: `pyproject.toml` 增加 `playwright>=1.40` 依赖；`uv.lock` 更新。

**不要** 跑 `uv run playwright install chromium`（每个用户/CI 自己跑）。

- [ ] **Step 1.2: 写失败测试**

Create `tests/test_browser_session.py`:

```python
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
```

- [ ] **Step 1.3: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_session.py -v`

Expected: 多个 FAIL — `ModuleNotFoundError: No module named 'tools.browser_tool'` / `'agent.browser_session'`

- [ ] **Step 1.4: 创建 tools/browser_tool.py 的 helper**

Create `tools/browser_tool.py`:

```python
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
```

- [ ] **Step 1.5: 创建 agent/browser_session.py**

Create `agent/browser_session.py`:

```python
"""Chromium 会话管理：lazy 启动 + accessibility snapshot + cleanup。"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# 只给这些 role 分配 ref（可交互节点）
_INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab",
    "combobox", "option", "slider", "spinbutton", "switch",
}


class BrowserSession:
    """单 AIAgent 共享的浏览器会话。

    生命周期：
      - AIAgent.__init__ 创建（不启动浏览器）
      - 首次 get_page() 启动 Chromium + 开 page
      - 后续 navigate/click/type 复用同一 page
      - AIAgent 退出调 cleanup() 关浏览器
    """

    def __init__(self, *, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._page = None
        self._started = False
        # ref -> CSS selector 映射（每次 snapshot 重置）
        self._ref_map: dict = {}

    def _ensure_started(self) -> None:
        """Lazy 启动。首次调用时启 Chromium。"""
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "playwright 未安装。请运行："
                "uv add playwright && uv run playwright install chromium"
            ) from e
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()
        self._started = True
        logger.info("BrowserSession 已启动 (headless=%s)", self._headless)

    def get_page(self):
        """获取当前 page（lazy 启动）。"""
        self._ensure_started()
        return self._page

    def cleanup(self) -> None:
        """关闭浏览器。多次调用安全。"""
        if not self._started:
            return
        try:
            if self._page:
                self._page.close()
        except Exception as e:
            logger.warning("关闭 page 失败: %s", e)
        try:
            if self._browser:
                self._browser.close()
        except Exception as e:
            logger.warning("关闭 browser 失败: %s", e)
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.warning("停止 playwright 失败: %s", e)
        self._started = False
        self._page = None
        self._browser = None
        self._playwright = None
        self._ref_map = {}

    def snapshot(self, max_chars: int = 8000) -> dict:
        """获取 accessibility snapshot + 分配 ref。

        返回 {"tree": <node>, "truncated": bool, "chars": int}。
        同时填充 self._ref_map: ref -> CSS selector。
        """
        page = self.get_page()
        raw = page.accessibility.snapshot()
        if raw is None:
            return {"tree": None, "truncated": False, "chars": 0}

        self._ref_map = {}  # 重置
        ref_counter = [0]

        def _build_selector(node: dict) -> str:
            """根据 role+name 构造 CSS selector（简化版）。"""
            role = node.get("role", "")
            name = node.get("name", "").replace('"', '\\"')
            if role == "button":
                return f'button:has-text("{name}")'
            if role == "link":
                return f'a:has-text("{name}")'
            if role == "textbox":
                return f'input[aria-label="{name}"], textarea[aria-label="{name}"]'
            if role == "combobox":
                return f'select[aria-label="{name}"]'
            if role == "checkbox":
                return f'input[type="checkbox"][aria-label="{name}"]'
            return ""

        def _assign_refs(node: dict) -> dict:
            if not isinstance(node, dict):
                return node
            role = node.get("role", "")
            if role in _INTERACTIVE_ROLES:
                ref_counter[0] += 1
                ref = f"a{ref_counter[0]}"
                node["ref"] = ref
                sel = _build_selector(node)
                if sel:
                    self._ref_map[ref] = sel
            children = node.get("children", [])
            new_children = [_assign_refs(c) for c in children]
            if new_children:
                node["children"] = new_children
            return node

        tree = _assign_refs(raw)

        text = json.dumps(tree, ensure_ascii=False)
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "...[truncated]"
            try:
                tree = json.loads(text)
            except Exception:
                tree = {"truncated_text": text[:max_chars]}
        return {
            "tree": tree,
            "truncated": truncated,
            "chars": len(text),
        }

    def resolve_ref(self, ref: str) -> Optional[str]:
        """ref → CSS selector。snapshot 过期或无效返 None。"""
        return self._ref_map.get(ref)
```

- [ ] **Step 1.6: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_session.py -v`

Expected: 15 passed

- [ ] **Step 1.7: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `740 + 15 = 755 passed`

- [ ] **Step 1.8: Commit**

```bash
git add pyproject.toml uv.lock agent/browser_session.py tools/browser_tool.py tests/test_browser_session.py
git commit -m "feat(browser): BrowserSession 类 + URL safety helper + playwright 依赖

新建 agent/browser_session.py：lazy Chromium 启动、accessibility snapshot
+ ref 分配、idempotent cleanup。新建 tools/browser_tool.py 顶部 helper：
_is_safe_url（http/https 白名单）+ _check_browser_available（check_fn）。
uv add playwright（用户需自行 uv run playwright install chromium）。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: browser_navigate + browser_close + toolset 注册 + AIAgent 集成

**Files:**
- Modify: `tools/browser_tool.py`（加 2 schema + 2 handler + 注册）
- Modify: `toolsets.py`（加 `"browser"` toolset）
- Modify: `agent/__init__.py`（AIAgent 加 `browser_session` 属性 + shutdown cleanup）
- Test: `tests/test_browser_tools.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `BrowserSession.get_page()` + `_is_safe_url` + `_check_browser_available`
- Produces:
  - `browser_navigate` / `browser_close` 工具注册
  - AIAgent 持有 `browser_session` 属性（lazy）
  - `_get_session(kwargs)` helper 从 `agent_ref.browser_session` 取 session

- [ ] **Step 2.1: 写失败测试**

Create `tests/test_browser_tools.py`:

```python
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
```

- [ ] **Step 2.2: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v`

Expected: ImportError: `cannot import name '_handle_browser_navigate'`

- [ ] **Step 2.3: 加 helper + schema + handler 到 browser_tool.py**

In `tools/browser_tool.py`, after the existing helpers (`_is_safe_url`, `_check_browser_available`), add:

```python
from agent.browser_session import BrowserSession


def _get_session(kwargs: dict) -> Optional[BrowserSession]:
    """从 kwargs 取 agent 持有的 session。"""
    agent = kwargs.get("agent_ref")
    if agent is None:
        return None
    return getattr(agent, "browser_session", None)


def _err(msg: str, error_type: Optional[str] = None) -> str:
    """构造 JSON 错误。error_type=None 时省略。"""
    d = {"error": msg}
    if error_type:
        d["error_type"] = error_type
    return json.dumps(d, ensure_ascii=False)
```

Then add the 2 schemas:

```python
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
```

Then add the 2 handlers:

```python
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
```

Finally add the 2 registrations:

```python
from tools.registry import registry

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
```

- [ ] **Step 2.4: 加 browser toolset 到 toolsets.py**

In `toolsets.py`, add new entry to `TOOLSETS` dict (after `"team"`):

```python
"browser": {
    "description": "浏览器自动化（13 个工具，基于 Playwright）",
    "tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_press_key",
        "browser_back", "browser_forward", "browser_close",
        "browser_get_images", "browser_vision", "browser_console",
        "browser_cdp",
    ],
    "includes": [],
},
```

- [ ] **Step 2.5: AIAgent 集成 BrowserSession**

In `agent/__init__.py`, find `AIAgent.__init__`. After the existing `self.hooks_registry = hooks_registry` line and the auto_heartbeat registration (from ⑪b), add:

```python
        # === ⑪c NEW: 浏览器会话 ===
        # 仅在 browser toolset 启用时创建（lazy init，不立即启 Chromium）
        self.browser_session = None
        if "browser" in (enabled_toolsets or []):
            try:
                from agent.browser_session import BrowserSession
                self.browser_session = BrowserSession(headless=True)
                logger.info("BrowserSession 已创建（lazy，未启动）")
            except Exception as e:
                logger.warning(
                    "BrowserSession 初始化失败（browser 工具将不可用）: %s", e,
                )
                self.browser_session = None
```

Then find where AIAgent shutdown/cleanup is called (search for `def shutdown` or similar, or look at `__del__`). Add at the end:

```python
        # 关闭浏览器
        if self.browser_session:
            try:
                self.browser_session.cleanup()
            except Exception as e:
                logger.warning("关闭 browser_session 失败: %s", e)
```

If no explicit shutdown method exists, use `__del__` or add a `cleanup()` method called from cli.py on exit. (Implementer: check existing patterns — `agent/__init__.py` may have `_cleanup_resources` or similar.)

- [ ] **Step 2.6: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_tools.py -v`

Expected: 10 passed

- [ ] **Step 2.7: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `755 + 10 = 765 passed`（部分测试可能因 `browser_session` 属性注入到 AIAgent 而需要其他测试适配，确认零回归即可）

- [ ] **Step 2.8: Commit**

```bash
git add tools/browser_tool.py toolsets.py agent/__init__.py tests/test_browser_tools.py
git commit -m "feat(browser): browser_navigate + browser_close + toolset 注册 + AIAgent 集成

新增 2 个 handler（含 URL safety 检查、agent_ref.browser_session 取 session）。
toolsets.py 加 'browser' toolset（默认 off）。AIAgent.__init__ 在 browser
toolset 启用时 lazy 创建 BrowserSession；shutdown 时 cleanup。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: browser_snapshot + browser_click + browser_type

**Files:**
- Modify: `tools/browser_tool.py`（加 3 schema + 3 handler + 注册）
- Test: `tests/test_browser_tools.py`（追加 6 个测试）

**Interfaces:**
- Consumes: Task 1 的 `BrowserSession.snapshot()` / `BrowserSession.resolve_ref()` / `BrowserSession.get_page()`
- Produces: `browser_snapshot` / `browser_click` / `browser_type` 工具注册

- [ ] **Step 3.1: 写失败测试（追加到 test_browser_tools.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 3: browser_snapshot + browser_click + browser_type
# ---------------------------------------------------------------------------

from tools.browser_tool import (
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
```

- [ ] **Step 3.2: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v -k "snapshot or click or type"`

Expected: ImportError

- [ ] **Step 3.3: 加 3 schema + 3 handler + 注册**

In `tools/browser_tool.py`, after `BROWSER_CLOSE_SCHEMA`, add schemas:

```python
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
```

After `_handle_browser_close`, add handlers:

```python
def _handle_browser_snapshot(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    max_chars = args.get("max_chars", 8000)
    try:
        result = session.snapshot(max_chars=max_chars)
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
```

Finally, add 3 registrations:

```python
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
```

- [ ] **Step 3.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_tools.py -v -k "snapshot or click or type"`

Expected: 10 passed（含 Task 3 新增的 10 个）

- [ ] **Step 3.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `765 + 9 = 774 passed`（snapshot 3 + click 5 + type 2 = 10 新增，但 test_click_empty_ref/test_click_browser_unavailable 与 navigate 共用模式可能少 1-2 个）

- [ ] **Step 3.6: Commit**

```bash
git add tools/browser_tool.py tests/test_browser_tools.py
git commit -m "feat(browser): browser_snapshot + browser_click + browser_type

snapshot 返回 accessibility tree + ref 分配。click/type 用 ref →
session.resolve_ref(ref) → CSS selector → page.click/fill。stale_ref
错误提示 LLM 重 snapshot。type 支持 submit=True 触发 Enter。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: browser_scroll + browser_press_key + browser_back + browser_forward

**Files:**
- Modify: `tools/browser_tool.py`（加 4 schema + 4 handler + 注册）
- Test: `tests/test_browser_tools.py`（追加 ~5 个测试）

**Interfaces:**
- Consumes: `BrowserSession.get_page()`
- Produces: 4 个简单导航工具

- [ ] **Step 4.1: 写失败测试（追加）**

Append:

```python
from tools.browser_tool import (
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
    fake_session.get_page.return_value = fake_page

    result = _handle_browser_forward(
        {}, agent_ref=_make_agent_with_session(fake_session),
    )
    data = json.loads(result)
    assert data["success"] is True
    fake_page.go_forward.assert_called_once()
```

- [ ] **Step 4.2: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v -k "scroll or press_key or back or forward"`

Expected: ImportError

- [ ] **Step 4.3: 加 4 schema + 4 handler + 注册**

After Task 3 schemas, add:

```python
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
```

After Task 3 handlers, add:

```python
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
```

Add 4 registrations:

```python
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
```

- [ ] **Step 4.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_tools.py -v -k "scroll or press_key or back or forward"`

Expected: 5 passed

- [ ] **Step 4.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `774 + 5 = 779 passed`

- [ ] **Step 4.6: Commit**

```bash
git add tools/browser_tool.py tests/test_browser_tools.py
git commit -m "feat(browser): browser_scroll + browser_press_key + browser_back + browser_forward

scroll 支持 ref（元素）或主页面；press_key 在 body 上按键；
back/forward 用 Playwright page.go_back/go_forward。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: browser_get_images + browser_console + browser_vision

**Files:**
- Modify: `tools/browser_tool.py`（加 3 schema + 3 handler + 注册）
- Test: `tests/test_browser_tools.py`（追加 ~6 个测试）

**Interfaces:**
- Consumes: `BrowserSession.get_page()`；agent.config（取 model/api_key 做 vision LLM 调用）
- Produces: 3 个数据提取工具

- [ ] **Step 5.1: 写失败测试（追加）**

Append:

```python
from tools.browser_tool import (
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
    # mock LLM client（agent 持有的 openai client）
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="A simple page"))]
    fake_client.chat.completions.create.return_value = fake_response
    fake_agent._browser_vision_client = fake_client  # 注入

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
    fake_agent._browser_vision_client = None

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
```

- [ ] **Step 5.2: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v -k "get_images or console or vision"`

Expected: ImportError

- [ ] **Step 5.3: 加 3 schema + 3 handler + 注册**

After Task 4 schemas, add:

```python
BROWSER_GET_IMAGES_SCHEMA = {
    "name": "browser_get_images",
    "description": "提取页面上所有 img 的 URL 列表（按 min_width/min_height 过滤小图）。",
    "parameters": {
        "type": "object",
        "properties": {
            "min_width": {"type": "integer", "default": 100, "description": "过滤掉太小的图"},
            "min_height": {"type": "integer", "default": 100},
        },
    },
}

BROWSER_CONSOLE_SCHEMA = {
    "name": "browser_console",
    "description": "读取浏览器 console 日志（自上次 navigate 起）。",
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["log", "info", "warning", "error"],
                "default": "log",
                "description": "最低级别过滤（error > warning > info > log）",
            },
        },
    },
}

BROWSER_VISION_SCHEMA = {
    "name": "browser_vision",
    "description": (
        "截图 + 用 LLM 视觉模型分析（用 agent 配置的默认 model）。"
        "适合页面布局/视觉问题。具体 DOM 内容用 browser_snapshot。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "问 LLM 什么（如 '描述这个页面的布局'）"},
        },
        "required": ["query"],
    },
}
```

After Task 4 handlers, add:

```python
def _handle_browser_get_images(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    min_w = args.get("min_width", 100)
    min_h = args.get("min_height", 100)
    try:
        page = session.get_page()
        # Playwright：在浏览器里跑 JS 提取 img 信息
        raw_imgs = page.eval_on_selector_all(
            "img",
            """(imgs) => imgs.map(img => ({
                src: img.src || img.currentSrc || "",
                width: img.naturalWidth || img.width || 0,
                height: img.naturalHeight || img.height || 0,
                alt: img.alt || "",
            }))""",
        )
        filtered = [
            img for img in raw_imgs
            if img.get("src")
            and img.get("width", 0) >= min_w
            and img.get("height", 0) >= min_h
        ]
        return json.dumps({
            "success": True,
            "images": filtered,
            "count": len(filtered),
            "total_found": len(raw_imgs),
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"提取图片失败: {e}", "get_images_error")


# console 日志级别优先级
_CONSOLE_LEVELS = {"log": 0, "info": 1, "warning": 2, "error": 3}


def _handle_browser_console(args: dict, **kwargs) -> str:
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    level = args.get("level", "log")
    try:
        page = session.get_page()
        logs = getattr(page, "_harvil_console_logs", [])
        threshold = _CONSOLE_LEVELS.get(level, 0)
        filtered = [
            entry for entry in logs
            if _CONSOLE_LEVELS.get(entry.get("type", "log"), 0) >= threshold
        ]
        return json.dumps({
            "success": True,
            "logs": filtered,
            "count": len(filtered),
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"读取 console 失败: {e}", "console_error")


def _handle_browser_vision(args: dict, **kwargs) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    agent = kwargs.get("agent_ref")
    client = getattr(agent, "_browser_vision_client", None) if agent else None
    if client is None:
        return _err(
            "vision LLM client 未配置（agent._browser_vision_client 为 None）",
            "vision_unavailable",
        )
    try:
        page = session.get_page()
        png_bytes = page.screenshot()
        import base64
        b64 = base64.b64encode(png_bytes).decode("ascii")
        # OpenAI 兼容 vision API
        response = client.chat.completions.create(
            model=kwargs.get("config", {}).get("model", {}).get("name", "deepseek-chat"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )
        description = response.choices[0].message.content
        return json.dumps({
            "success": True,
            "description": description,
            "query": query,
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"vision 调用失败: {e}", "vision_error")
```

**Note for implementer**: console 日志缓存机制需要在 `BrowserSession._ensure_started` 里订阅 page console event 并缓存。Add to BrowserSession:

```python
def _ensure_started(self):
    # ... existing code ...
    self._page = self._browser.new_page()
    # 订阅 console 事件，缓存到 page._harvil_console_logs
    self._page._harvil_console_logs = []
    def _on_console(msg):
        try:
            self._page._harvil_console_logs.append({
                "type": msg.type,
                "text": msg.text,
            })
            # 限长防溢出
            if len(self._page._harvil_console_logs) > 200:
                self._page._harvil_console_logs = \
                    self._page._harvil_console_logs[-200:]
        except Exception:
            pass
    self._page.on("console", _on_console)
    # 清空 console 在每次 navigate 时
    self._page.on("framenavigated", lambda *_: setattr(
        self._page, "_harvil_console_logs", []
    ))
    self._started = True
```

Add 3 registrations:

```python
registry.register(
    name="browser_get_images", toolset="browser",
    schema=BROWSER_GET_IMAGES_SCHEMA, handler=_handle_browser_get_images,
    check_fn=_check_browser_available, emoji="🖼",
)
registry.register(
    name="browser_console", toolset="browser",
    schema=BROWSER_CONSOLE_SCHEMA, handler=_handle_browser_console,
    check_fn=_check_browser_available, emoji="📊",
)
registry.register(
    name="browser_vision", toolset="browser",
    schema=BROWSER_VISION_SCHEMA, handler=_handle_browser_vision,
    check_fn=_check_browser_available, emoji="👁",
)
```

- [ ] **Step 5.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_tools.py -v -k "get_images or console or vision"`

Expected: 5 passed

- [ ] **Step 5.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `779 + 5 = 784 passed`

- [ ] **Step 5.6: Commit**

```bash
git add tools/browser_tool.py tests/test_browser_tools.py
git commit -m "feat(browser): browser_get_images + browser_console + browser_vision

get_images 用 Playwright eval_on_selector_all 提取 + 尺寸过滤。
console 缓存到 page._harvil_console_logs（订阅 console 事件，navigate 时清空）。
vision 截图 + base64 + OpenAI 兼容 vision API。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: browser_cdp + 全测回归 + push

**Files:**
- Modify: `tools/browser_tool.py`（加 1 schema + 1 handler + 注册）
- Test: `tests/test_browser_tools.py`（追加 2 个测试）

**Interfaces:**
- Consumes: `page.context.new_cdp_session(page)`
- Produces: `browser_cdp` 工具注册

- [ ] **Step 6.1: 写失败测试（追加）**

Append:

```python
from tools.browser_tool import _handle_browser_cdp


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
```

- [ ] **Step 6.2: 跑测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v -k cdp`

Expected: ImportError

- [ ] **Step 6.3: 加 schema + handler + 注册**

After Task 5 schemas, add:

```python
BROWSER_CDP_SCHEMA = {
    "name": "browser_cdp",
    "description": (
        "直接发 Chrome DevTools Protocol 命令（逃生舱）。"
        "只在其他 browser_* 工具都搞不定时用。"
        "常用命令：Page.reload、Runtime.evaluate、Network.getX 等。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "CDP 方法名（如 'Page.reload'）"},
            "args": {"type": "object", "description": "CDP 参数"},
        },
        "required": ["command"],
    },
}
```

After Task 5 handlers, add:

```python
def _handle_browser_cdp(args: dict, **kwargs) -> str:
    command = (args.get("command") or "").strip()
    if not command:
        return _err("command 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    try:
        cdp_args = args.get("args") or {}
        page = session.get_page()
        client = page.context.new_cdp_session(page)
        result = client.send(command, cdp_args)
        return json.dumps({"success": True, "result": result}, ensure_ascii=False)
    except Exception as e:
        return _err(f"CDP 命令失败: {e}", "cdp_error")
```

Add registration:

```python
registry.register(
    name="browser_cdp", toolset="browser",
    schema=BROWSER_CDP_SCHEMA, handler=_handle_browser_cdp,
    check_fn=_check_browser_available, emoji="🔌",
)
```

- [ ] **Step 6.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_browser_tools.py -v -k cdp`

Expected: 3 passed

- [ ] **Step 6.5: 跑完整测试套件**

Run: `uv run pytest tests/ -q`

Expected: `784 + 3 = 787 passed`

- [ ] **Step 6.6: 跑复刻检查清单**

Run: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python scripts/verify.py 2>&1 | tail -3`

Expected: `总计 22：19 通过，3 失败`（3 pre-existing MemoryStore 失败）

- [ ] **Step 6.7: 验证 13 个工具都注册了**

Run:
```bash
uv run python -c "
from tools.registry import registry
browser_tools = sorted([n for n in registry.list_all() if n.startswith('browser_')])
print(f'{len(browser_tools)} browser tools:')
for t in browser_tools:
    print(f'  - {t}')
"
```

Expected: `13 browser tools:` + list with all 13 names.

- [ ] **Step 6.8: Commit**

```bash
git add tools/browser_tool.py tests/test_browser_tools.py
git commit -m "feat(browser): browser_cdp CDP 逃生舱 + 13 工具完成

直接发 Chrome DevTools Protocol 命令。其他工具搞不定时用。
browser toolset 全部 13 个工具就位。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 6.9: Push 到 origin**

```bash
git push origin master
```

Expected: 6 个新 commit（Task 1-6）+ spec + plan commit 都推送成功

- [ ] **Step 6.10: 更新交接文档**

修改 `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`：
- 「当前状态」从 740 改为 787
- 在「第 6 批改进」后加：

```markdown
### 第 7 批改进（740 → 787）

| # | 能力 | 说明 |
|---|---|---|
| ③ | 浏览器自动化工具集 | 新 `browser` toolset（13 工具，基于 Playwright sync API）：navigate/snapshot/click/type/scroll/press_key/back/forward/close/get_images/vision/console/cdp。URL scheme 白名单；accessibility tree + ref 系统；check_fn 在 playwright 未装时隐藏工具 |
```

- 从「剩余待做」删除 ③ 行

---

## 完成标准

- [ ] 6 个新 commit 已 push 到 origin/master
- [ ] `uv run pytest tests/ -q` 显示 787 passed
- [ ] verify.py 与起点一致
- [ ] `python -c "from tools.registry import registry; print(sorted(n for n in registry.list_all() if n.startswith('browser_')))"` 输出 13 个工具
- [ ] 交接文档已更新（740 → 787，③ 移入完成）
