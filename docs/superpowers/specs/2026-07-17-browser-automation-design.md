# 浏览器自动化工具集 设计

- **日期**：2026-07-17
- **状态**：用户授权直接推进
- **范围**：③ 浏览器自动化工具集。13 个工具让 agent 操作网页（navigate/click/type/snapshot 等）
- **依赖**：现有 toolsets.py + tools/registry.py（已就绪）
- **参考**：`D:\project\hermes-agent-main\项目文档\05-执行环境后端.md` §二

---

## 摘要

引入 Playwright Python（sync API）作为浏览器后端，新建 `tools/browser_tool.py` 加 13 个工具，新建 `agent/browser_session.py` 管理 Chromium 进程生命周期。工具注册到新 `"browser"` toolset（不在 core 默认集，需显式 `enabled_toolsets=["core", "browser"]` 启用）。`check_fn` 在 Playwright 未安装时自动隐藏工具。

**不在范围**：headless 模式切换、多 tab 管理、cookie 持久化、HTTP 鉴权、文件上传、视觉分析 LLM 模型选择（先硬编码 deepseek-chat，未来可配）。

---

## §1 问题与目标

### 问题

当前 agent 只能通过 `terminal + curl` 或 `execute_code + requests` 访问网络，无法：
1. 处理 JS 渲染的页面（SPA）
2. 与表单/按钮交互（登录、提交、翻页）
3. 提取按视觉布局组织的内容（"侧栏第 3 个链接"）
4. 处理需要点击才能展开的内容（accordion、modal）

### 目标

1. 13 个工具覆盖典型浏览器操作（导航、交互、读取、调试）
2. 基于 accessibility tree（无障碍树），LLM 不需要看截图就能定位元素
3. URL 安全闸门（拒绝 `file://`、`javascript:` 等危险 scheme）
4. 按需启用（`browser` toolset 默认 off，需在 config 开启）
5. 工具不可见时静默隐藏（playwright 未装 → check_fn 返 False）

### 非目标

- 不做 multi-tab 管理（先单 tab）
- 不做 cookie/session 持久化（agent 退出 → 浏览器清空）
- 不做 HTTP 鉴权自动填充
- 不做 file upload（结构复杂，后续单开）
- 不集成 vision LLM 的多模型路由（先用 config 里默认 model）

---

## §2 架构

```
LLM 调用 browser_* 工具
    │
    ▼
tools/browser_tool.py handler
    │
    ├─ _check_browser_available()  ← check_fn：playwright 装了吗？
    │
    ├─ URL 安全校验（navigate/click 链接）
    │   ├─ 拒绝 file://, data:, javascript:
    │   └─ 只允许 http/https
    │
    ▼
agent/browser_session.py BrowserSession
    │
    ├─ 单例：每个 AIAgent 一个 session
    ├─ lazy 启动：首次调用 browser_navigate 才启 Chromium
    ├─ 持久：navigate 之间保持页面状态（cookie、localStorage）
    └─ shutdown：AIAgent 退出时调 cleanup() 关 Chromium
        │
        ▼
    Playwright sync API
        │
        ▼
    Chromium (headless by default)
```

### 关键不变量

1. **Lazy init**：BrowserSession 实例化时不启 Chromium，首次 `get_page()` 才启
2. **单例 per agent**：一个 AIAgent 共享一个 BrowserSession（同 Chromium、同 page）
3. **Toolset 默认 off**：`browser` 不在 core，必须 config 显式启用
4. **check_fn 安全网**：playwright 未安装时工具不可见，不报错
5. **URL scheme 白名单**：navigate 只允许 http/https，其他直接拒绝
6. **Snapshot 有上限**：8000 字符默认，避免吃光 context

### Ref 系统（accessibility tree）

```
LLM 调 browser_snapshot()
    │
    ▼
BrowserSession.snapshot()
    │
    ▼  Playwright page.accessibility.snapshot()
    │
    ▼  递归遍历 + 给每个可交互节点分配 ref
    │
    ▼
返回树形 JSON：
{
  "role": "WebArea", "name": "Page Title",
  "children": [
    {"role": "heading", "name": "Welcome", "level": 1},
    {"role": "textbox", "name": "Search", "ref": "a12"},
    {"role": "button", "name": "Submit", "ref": "a13"},
    ...
  ]
}

LLM 用 ref="a12" 调 browser_type(ref="a12", text="hello")
                  或 browser_click(ref="a13")
```

**Ref 分配规则**：
- 只给可交互节点分配（button、link、textbox、checkbox、menuitem 等）
- 静态节点（heading、paragraph、list）不分配
- Ref 格式：`a` + 数字递增（`a1`, `a2`, ..., `a100`）
- 每次 snapshot 重置计数（ref 跨 snapshot 不稳定）

---

## §3 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/browser_session.py` | 🆕 新增 | `BrowserSession` 类：lazy Chromium 启动、`get_page()`、`cleanup()`、accessibility snapshot + ref 分配 |
| `tools/browser_tool.py` | 🆕 新增 | 13 schema + 13 handler + `check_fn` + URL 安全 helper |
| `toolsets.py` | ♻️ 改 | 加 `"browser"` toolset（13 个工具名） |
| `agent/__init__.py` | ♻️ 改 | AIAgent 加 `browser_session` 属性 + shutdown 时 `cleanup()` |
| `pyproject.toml` | ♻️ 改 | 加 `playwright>=1.40` 依赖 |
| `tests/test_browser_session.py` | 🆕 新增 | BrowserSession + URL 安全测试（mock playwright） |
| `tests/test_browser_tools.py` | 🆕 新增 | 13 个 handler 行为测试（mock BrowserSession） |

---

## §4 组件设计

### §4.1 `agent/browser_session.py`

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

    def _ensure_started(self) -> None:
        """Lazy 启动。首次调用时启 Chromium。"""
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "playwright 未安装。请运行：uv add playwright && uv run playwright install chromium"
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

    def snapshot(self, max_chars: int = 8000) -> dict:
        """获取 accessibility snapshot + 分配 ref。

        返回 {"tree": <node>, "truncated": bool, "chars": int}。
        """
        page = self.get_page()
        raw = page.accessibility.snapshot()
        if raw is None:
            return {"tree": None, "truncated": False, "chars": 0}

        # 给可交互节点分配 ref
        ref_counter = [0]  # 闭包可变

        def _assign_refs(node: dict, depth: int = 0) -> dict:
            if not isinstance(node, dict):
                return node
            role = node.get("role", "")
            if role in _INTERACTIVE_ROLES:
                ref_counter[0] += 1
                node["ref"] = f"a{ref_counter[0]}"
            children = node.get("children", [])
            new_children = []
            for c in children:
                new_children.append(_assign_refs(c, depth + 1))
            if new_children:
                node["children"] = new_children
            return node

        tree = _assign_refs(raw)

        # 序列化 + 截断
        import json
        text = json.dumps(tree, ensure_ascii=False)
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "...[truncated]"
            try:
                tree = json.loads(text)  # 截断后可能 invalid JSON
            except Exception:
                tree = {"truncated_text": text[:max_chars]}
        return {
            "tree": tree,
            "truncated": truncated,
            "chars": len(text),
        }


# 进程级单例（AIAgent 直接持有，无需全局单例）
```

### §4.2 `tools/browser_tool.py`

#### URL 安全 helper

```python
import re
from urllib.parse import urlparse

_BLOCKED_SCHEMES = {"file", "data", "javascript", "vbscript", "about"}
_ALLOWED_SCHEMES = {"http", "https"}


def _is_safe_url(url: str) -> tuple[bool, str]:
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
```

#### check_fn

```python
def _check_browser_available() -> bool:
    """check_fn：playwright 是否可用。"""
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False
```

#### 13 个 schema（节选关键的）

```python
BROWSER_NAVIGATE_SCHEMA = {
    "name": "browser_navigate",
    "description": "导航到 URL。返回最终 URL + 标题。会 lazy 启动 Chromium。",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "目标 URL（http/https）"},
        },
        "required": ["url"],
    },
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
    "description": "在 snapshot 里 ref 指向的输入框输入文本。",
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
    "description": "滚动页面或某个元素。",
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
    "description": "按键盘键（Enter、Tab、Escape、ArrowDown 等）。",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "键名（如 'Enter', 'Escape'）"},
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

BROWSER_CLOSE_SCHEMA = {
    "name": "browser_close",
    "description": "关闭浏览器（释放 Chromium 进程）。下次 navigate 会重启。",
    "parameters": {"type": "object", "properties": {}},
}

BROWSER_GET_IMAGES_SCHEMA = {
    "name": "browser_get_images",
    "description": "提取页面上所有 img 的 URL 列表。",
    "parameters": {
        "type": "object",
        "properties": {
            "min_width": {"type": "integer", "default": 100, "description": "过滤掉太小的图"},
            "min_height": {"type": "integer", "default": 100},
        },
    },
}

BROWSER_VISION_SCHEMA = {
    "name": "browser_vision",
    "description": "截图 + 用 LLM 视觉模型分析（用 agent 配置的默认 model）。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "问 LLM 什么（如 '描述这个页面的布局'）"},
        },
        "required": ["query"],
    },
}

BROWSER_CONSOLE_SCHEMA = {
    "name": "browser_console",
    "description": "读取浏览器 console 日志。",
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["log", "info", "warning", "error"],
                "default": "log",
                "description": "最低级别过滤",
            },
        },
    },
}

BROWSER_CDP_SCHEMA = {
    "name": "browser_cdp",
    "description": (
        "直接发 Chrome DevTools Protocol 命令（逃生舱）。"
        "只在其他工具都搞不定时用。"
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

#### Handler（节选 navigate + click + cdp）

```python
def _get_session(kwargs: dict) -> Optional[BrowserSession]:
    """从 kwargs 取 agent 持有的 session。"""
    agent = kwargs.get("agent_ref")
    if agent is None:
        return None
    return getattr(agent, "browser_session", None)


def _handle_browser_navigate(args: dict, **kwargs) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return json.dumps({"error": "url 不能为空"}, ensure_ascii=False)
    safe, reason = _is_safe_url(url)
    if not safe:
        return json.dumps(
            {"error": f"URL 不安全: {reason}", "error_type": "unsafe_url"},
            ensure_ascii=False,
        )
    session = _get_session(kwargs)
    if session is None:
        return json.dumps(
            {"error": "browser_session 未初始化", "error_type": "browser_unavailable"},
            ensure_ascii=False,
        )
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
        return json.dumps(
            {"error": f"导航失败: {e}", "error_type": "navigation_error"},
            ensure_ascii=False,
        )


def _handle_browser_click(args: dict, **kwargs) -> str:
    ref = (args.get("ref") or "").strip()
    if not ref:
        return json.dumps({"error": "ref 不能为空"}, ensure_ascii=False)
    session = _get_session(kwargs)
    if session is None:
        return json.dumps(
            {"error": "browser_session 未初始化", "error_type": "browser_unavailable"},
            ensure_ascii=False,
        )
    # 通过 ref 找 locator：ref 是 snapshot 分配的，需要 page locator
    # 简化：用 page.locator + ref 数据属性。实际上 Playwright 没原生 ref，
    # 我们需要在 snapshot 时记下节点的 backend DOM node 或 selector。
    # 见 §4.3 ref 解析策略
    try:
        selector = session.resolve_ref(ref)
        if selector is None:
            return json.dumps(
                {"error": f"无效 ref: {ref}（可能 snapshot 过期）",
                 "error_type": "stale_ref"},
                ensure_ascii=False,
            )
        page = session.get_page()
        page.click(selector, timeout=10000)
        return json.dumps({"success": True, "clicked": ref}, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {"error": f"点击失败: {e}", "error_type": "click_error"},
            ensure_ascii=False,
        )


def _handle_browser_cdp(args: dict, **kwargs) -> str:
    command = (args.get("command") or "").strip()
    if not command:
        return json.dumps({"error": "command 不能为空"}, ensure_ascii=False)
    session = _get_session(kwargs)
    if session is None:
        return json.dumps(
            {"error": "browser_session 未初始化", "error_type": "browser_unavailable"},
            ensure_ascii=False,
        )
    try:
        cdp_args = args.get("args") or {}
        page = session.get_page()
        # Playwright sync API: page.query_selector + ... 没有 direct CDP send
        # 但有 page.evaluate + 一些 CDP session 接口
        # 实际用 CDPSession：client = page.context.new_cdp_session(page)
        client = page.context.new_cdp_session(page)
        result = client.send(command, cdp_args)
        return json.dumps({"success": True, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {"error": f"CDP 命令失败: {e}", "error_type": "cdp_error"},
            ensure_ascii=False,
        )
```

### §4.3 Ref 解析策略

Playwright accessibility.snapshot() 返回的节点没有 DOM selector，需要自己建 ref → selector 映射。**简化方案**：

**Snapshot 时**：递归遍历，给每个可交互节点分配 ref，同时记录该节点的 CSS selector（基于 role + name + index）：
```python
# 在 BrowserSession.snapshot() 里
ref_map = {}  # ref -> CSS selector

def _build_selector(node, parent_selector="", index=0):
    role = node.get("role", "")
    name = node.get("name", "")
    # 简化：用 role tag + 文本
    if role == "button":
        sel = f'button:has-text("{name}")'
    elif role == "link":
        sel = f'a:has-text("{name}")'
    elif role == "textbox":
        # textbox 的 name 通常是 label 或 placeholder
        sel = f'input[aria-label="{name}"], textarea[aria-label="{name}"]'
    elif role == "combobox":
        sel = f'select[aria-label="{name}"]'
    else:
        sel = parent_selector  # fallback
    return sel
```

**限制**：名字重复时 click 可能点错。生产级方案需要 Playwright 的 `locator` + `filter`，但本批先做基础版。

**另一个方案（更可靠）**：snapshot 时给每个节点注入 `data-harvil-ref` 属性，click 直接 `[data-harvil-ref="a12"]`。但这要改 DOM，副作用大。先用简化 selector 方案。

### §4.4 AIAgent 集成

```python
# agent/__init__.py AIAgent.__init__ 末尾（auto_heartbeat 注册之后）
self.browser_session = None  # lazy：仅在 browser toolset 启用时创建
if "browser" in (enabled_toolsets or []):
    try:
        from agent.browser_session import BrowserSession
        self.browser_session = BrowserSession(headless=True)
    except Exception as e:
        logger.warning("BrowserSession 初始化失败（browser 工具将不可用）: %s", e)
        self.browser_session = None

# AIAgent.shutdown / __del__ 加：
if self.browser_session:
    try:
        self.browser_session.cleanup()
    except Exception:
        pass
```

### §4.5 toolsets.py 加 browser

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

---

## §5 错误处理

| 场景 | error_type | 说明 |
|---|---|---|
| playwright 未装 | （工具不可见） | check_fn 返 False，工具不暴露给 LLM |
| browser_session 未初始化 | `browser_unavailable` | agent 没 enabled browser toolset |
| navigate URL 空 | （无 error_type） | "url 不能为空" |
| navigate URL 危险 scheme | `unsafe_url` | "URL 不安全: 禁止的 scheme: file" |
| navigate 超时 | `navigation_error` | "导航失败: TimeoutError" |
| click 无效 ref | `stale_ref` | "无效 ref: a99（snapshot 过期）" |
| click 元素未找到 | `click_error` | Playwright 抛异常 |
| cdp 命令失败 | `cdp_error` | Playwright 抛异常 |
| cleanup 异常 | （无返回，只 log） | 不影响 agent 退出 |

---

## §6 测试设计

### `tests/test_browser_session.py`（~6 个测试，mock playwright）

- `test_session_lazy_init_not_started_on_construct` — 实例化后 `_started is False`
- `test_session_ensure_started_calls_playwright` — monkeypatch playwright.sync_api 验证调用
- `test_session_cleanup_idempotent` — 多次 cleanup 不抛
- `test_session_cleanup_when_not_started` — 未启动就 cleanup，no-op
- `test_snapshot_assigns_refs_to_interactive` — mock snapshot 输入含 button，验证返回 `ref: "a1"`
- `test_snapshot_truncates_over_max_chars` — max_chars=100 + 巨大树，验证 `truncated=True`

### `tests/test_browser_tools.py`（~14 个测试，mock BrowserSession）

- `test_navigate_blocks_file_scheme` — URL="file:///x" → `unsafe_url`
- `test_navigate_blocks_data_scheme` — URL="data:text/plain,x" → `unsafe_url`
- `test_navigate_blocks_javascript_scheme` — URL="javascript:alert(1)" → `unsafe_url`
- `test_navigate_blocks_empty` — URL="" → error
- `test_navigate_blocks_missing_netloc` — URL="http://" → error
- `test_navigate_success` — mock page.goto 返 response，验证返回 JSON
- `test_navigate_browser_unavailable` — agent_ref=None → `browser_unavailable`
- `test_click_empty_ref` — ref="" → error
- `test_click_stale_ref` — session.resolve_ref 返 None → `stale_ref`
- `test_click_success` — mock resolve_ref + page.click，验证返回
- `test_cdp_empty_command` — command="" → error
- `test_cdp_success` — mock new_cdp_session + send，验证返回
- `test_check_browser_available_with_playwright` — 安装时返 True
- `test_check_browser_available_without_playwright` — monkeypatch ImportError，返 False

### `tests/test_browser_integration.py`（可选，~2 个测试）

需要真 playwright，CI 跑：
- `test_integration_navigate_snapshot_click` — 真访问 `https://example.com` + snapshot + 点 link
- `test_integration_close_releases_process` — close 后 cleanup 不漏进程

**新增测试约 22 个**。预期 740 + 22 = **762 测试通过**。

---

## §7 实现顺序（6 个 task）

| Task | 做什么 | 新增测试数 | 累计 |
|---|---|---|---|
| 1 | pyproject 加 playwright + `BrowserSession` 类（lazy init/cleanup/snapshot）+ URL safety helper | ~6 | 746 |
| 2 | browser_navigate + browser_close handler + check_fn + toolset 注册 + AIAgent 集成 | ~6 | 752 |
| 3 | browser_snapshot + browser_click + browser_type（含 ref 解析） | ~6 | 758 |
| 4 | browser_scroll + browser_press_key + browser_back + browser_forward | ~2 | 760 |
| 5 | browser_get_images + browser_console + browser_vision | ~3 | 763（+1 vision mock 复杂度） |
| 6 | browser_cdp + 全测回归 + push | ~1 | 764 |

每 task 一个 commit。最后 push origin/master。

---

## §8 风险与权衡

1. **Playwright 安装体积**：Chromium ~200MB。`uv add playwright` 只装 Python 包；用户需手动 `uv run playwright install chromium`。`check_fn` 处理未装情况。
2. **Ref 系统 limitation**：简化 selector 在重复 name 时会点错。未来可加 `data-harvil-ref` 注入或用 Playwright Inspector API。
3. **测试 mock 复杂度**：Playwright sync API 不易 mock。考虑用 `pytest-mock` + 自定义 FakePage 类。
4. **Sync vs Async**：选 sync Playwright 与 HarvilAgent 主循环一致；缺点是不能并发多 page 操作。
5. **Vision LLM 调用**：`browser_vision` 截图 base64 + 走 agent 配置的 model（OpenAI 兼容）。如果 model 不支持 vision（如 deepseek-chat 文本模型），handler 返错误提示用户换 model。
6. **Headless 默认**：agent 不需要 GUI，默认 headless=True。未来加截图 debug 时可临时关。

---

## §9 未来扩展

- multi-tab：browser_new_tab / browser_switch_tab
- file upload：browser_upload(ref, file_path)
- cookie 持久化：browser_save_state / browser_load_state
- HTTP 鉴权：browser_set_auth(url, user, pass)
- iframe 支持：browser_enter_frame / browser_exit_frame
- vision LLM 多模型路由（aux_llm router 复用）
- CDP 自动化更高级用法（拦截网络请求、改 UA 等）
