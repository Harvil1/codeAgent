"""Chromium 会话管理：lazy 启动 + accessibility snapshot + cleanup。"""
import json
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
        # 订阅 console 事件，缓存到 page._omnimate_console_logs
        self._page._omnimate_console_logs = []

        def _on_console(msg):
            try:
                self._page._omnimate_console_logs.append({
                    "type": msg.type,
                    "text": msg.text,
                })
                # 限长防溢出
                if len(self._page._omnimate_console_logs) > 200:
                    self._page._omnimate_console_logs = \
                        self._page._omnimate_console_logs[-200:]
            except Exception:
                pass

        self._page.on("console", _on_console)
        # 清空 console 在每次 navigate 时
        self._page.on("framenavigated", lambda *_: setattr(
            self._page, "_omnimate_console_logs", []
        ))
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
