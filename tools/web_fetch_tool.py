"""WebFetch 工具：抓取 URL 内容并转纯文本（对齐 Claude Code WebFetch）。

用 httpx（已装）抓取，标准库 html.parser 转纯文本。
- 超时 15s、follow redirects、User-Agent
- 大小上限 500KB（超了截断）
- 二进制/非文本 Content-Type 拒绝
- 失败 fail-open：返回 error JSON，不阻塞 agent
"""

import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_BYTES = 500_000       # 500KB
_MAX_TEXT_CHARS = 20000    # 返回文本上限
_TIMEOUT = 15.0


def _html_to_text(html: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """HTML → 纯文本（标准库 html.parser，跳过 script/style）。"""
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = max(0, self.skip - 1)

        def handle_data(self, data):
            if self.skip == 0 and data.strip():
                self.parts.append(data.strip())

    extractor = _TextExtractor()
    extractor.feed(html)
    return "\n".join(extractor.parts)[:max_chars]


def _handle_web_fetch(args: dict, **kwargs) -> str:
    url = (args.get("url") or "").strip()
    prompt = (args.get("prompt") or "").strip()

    if not url:
        return json.dumps(
            {"error": "url 不能为空", "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    if not url.startswith(("http://", "https://")):
        return json.dumps({
            "error": f"不支持的 URL: {url}（需 http/https）",
            "error_type": "invalid_url",
        }, ensure_ascii=False)

    try:
        import httpx
        resp = httpx.get(
            url, timeout=_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": "OmniMate/0.1",
                     "Accept": "text/html,text/plain,application/json,*/*"},
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        return json.dumps({
            "error": f"抓取超时（{_TIMEOUT}s）", "error_type": "timeout", "url": url,
        }, ensure_ascii=False)
    except httpx.HTTPError as e:
        return json.dumps({
            "error": f"抓取失败: {e}", "error_type": "http_error", "url": url,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("web_fetch 抓取异常: %s", e)
        return json.dumps({
            "error": f"抓取异常: {e}", "error_type": "fetch_error", "url": url,
        }, ensure_ascii=False)

    content_type = resp.headers.get("content-type", "").lower()
    body = resp.content
    truncated = False
    if len(body) > _MAX_BYTES:
        body = body[:_MAX_BYTES]
        truncated = True

    # 非文本内容拒绝（二进制/图片/视频等）
    if not any(k in content_type for k in ("text/", "html", "json", "xml")) and content_type:
        return json.dumps({
            "error": f"非文本内容类型: {content_type}",
            "error_type": "binary_content", "url": url,
        }, ensure_ascii=False)

    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = body.decode("latin-1", errors="replace")

    # HTML → 纯文本
    if "html" in content_type or text.lstrip().startswith(("<", "<!doctype")):
        content = _html_to_text(text)
    else:
        content = text.strip()

    if len(content) > _MAX_TEXT_CHARS:
        content = content[:_MAX_TEXT_CHARS]
        truncated = True

    return json.dumps({
        "url": url,
        "content": content,
        "content_type": content_type,
        "truncated": truncated,
        "bytes": len(body),
        "prompt": prompt or None,
    }, ensure_ascii=False)


WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": (
        "抓取网页/URL 内容并转为纯文本（对齐 Claude Code WebFetch）。"
        "适合查文档、读页面、看 API 说明。返回正文（最多 2 万字符）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 URL（http/https）"},
            "prompt": {
                "type": "string",
                "description": "可选：告诉模型关注什么（当前返回全文，prompt 供后续处理）",
            },
        },
        "required": ["url"],
    },
}


registry.register(
    name="web_fetch",
    toolset="core",
    schema=WEB_FETCH_SCHEMA,
    handler=_handle_web_fetch,
    emoji="🌐",
)
