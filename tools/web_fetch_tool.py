"""网页抓取工具：把一个网址（URL）的内容下载下来，去掉 HTML 标签只留文字。

网络请求用 httpx 库（项目已装），HTML 转纯文本用 Python 自带的 html.parser。
几条保命规则：
- 15 秒超时；网站跳转（重定向）自动跟过去；带上 User-Agent 表明身份
- 下载内容最多收 500KB，超了就砍掉
- 图片、视频这类二进制内容直接拒绝（模型看不了，也防止误下载大文件）
- 任何失败都不炸整个对话，只返回一段 error JSON 让模型自己想办法
"""

import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_BYTES = 500_000       # 下载内容最多收 50 万字节（约 500KB）
_MAX_TEXT_CHARS = 20000    # 最终返回给模型的文字最多 2 万字符
_TIMEOUT = 15.0


def _html_to_text(html: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """把 HTML 网页代码变成干净的纯文字——扔掉所有标签，跳过 script/style 里的代码和样式。

    模型只需要正文文字；脚本和样式代码是纯噪音，整段跳过。用 Python 标准库
    html.parser 实现，不引第三方依赖。

    参数：
        html：网页的 HTML 源码字符串。
        max_chars：最多保留多少字符（默认 2 万）。

    返回：一行一段拼接的纯文本（超长部分砍掉）。
    """
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


async def _handle_web_fetch(args: dict, **kwargs) -> str:
    """下载指定网页并转成纯文本返回；带了"关注点"时还会让小模型先提炼一遍。

    干什么：校验网址 → 用 httpx 下载 → 拒绝非文本内容 → HTML 转纯文本 →
    （可选）小模型提炼 → 返回 JSON。

    参数：
        args：工具参数字典，来自模型——url（要抓的网址，必须是 http/https）、
            prompt（关注点：抓回来想重点看什么，比如"这个库怎么安装"）。
        **kwargs：分发器注入的运行上下文——重点用 agent_ref（主对话对象），
            从它身上取 aux_llm_router（小模型路由器）来做提炼。

    返回：JSON 字符串，成功时含 content（正文）、truncated（有没有被截断）、
        refined（是不是经过提炼）等字段；失败时是 {"error": ..., "error_type": ...}。
    """
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

    # 二进制内容（图片/视频/压缩包等）直接拒绝——模型看不了，也防误收大文件
    if not any(k in content_type for k in ("text/", "html", "json", "xml")) and content_type:
        return json.dumps({
            "error": f"非文本内容类型: {content_type}",
            "error_type": "binary_content", "url": url,
        }, ensure_ascii=False)

    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = body.decode("latin-1", errors="replace")

    # 是 HTML 网页（或长得像 HTML）就转成纯文本；本来就是纯文本就直接用
    if "html" in content_type or text.lstrip().startswith(("<", "<!doctype")):
        content = _html_to_text(text)
    else:
        content = text.strip()

    if len(content) > _MAX_TEXT_CHARS:
        content = content[:_MAX_TEXT_CHARS]
        truncated = True

    # === 小模型提炼（按关注点
    # 先提炼再交回主模型，省主模型的上下文额度）。带了 prompt 且主对话身上有
    # 小模型路由器（aux_llm_router）时走提炼；小模型失败或没配置就降级返回
    # 全文（fail-open：宁可多花点上下文也不报错卡住）。===
    refined = False
    if prompt:
        aux = getattr(kwargs.get("agent_ref"), "aux_llm_router", None)
        if aux is not None:
            try:
                refined_text = await _refine_with_aux(aux, url, prompt, content)
                if refined_text:
                    content = refined_text
                    refined = True
                    truncated = False  # 提炼过的内容是完整产物，"截断"标记不再有意义
            except Exception as e:
                logger.warning("web_fetch aux 提炼失败（降级全文）: %s", e)

    return json.dumps({
        "url": url,
        "content": content,
        "content_type": content_type,
        "truncated": truncated,
        "refined": refined,
        "bytes": len(body),
        "prompt": prompt or None,
    }, ensure_ascii=False)


async def _refine_with_aux(aux_router, url: str, prompt: str, content: str) -> str:
    """让便宜的小模型按"关注点"从网页正文里挑出相关内容，压缩后返回。

    网页全文动辄上万字，直接塞给主模型太费上下文额度；让便宜的小模型先筛
    一遍，只回跟关注点相关的部分。

    参数：
        aux_router：小模型的路由器（有 chat_completions 方法可发对话请求）。
        url：网页地址（写进提示词里，让小模型知道内容来源）。
        prompt：关注点（模型想从这个网页里了解什么）。
        content：网页正文的纯文本。

    返回：提炼结果字符串；小模型输出异常时返回空串（表示放弃提炼，
        调用方收到空串就退回用全文）。
    """
    refine_prompt = (
        f"根据以下关注点，从网页内容中提炼与它相关的信息：\n"
        f"关注点：{prompt}\n\n"
        f"网页内容（来自 {url}）：\n{content[:15000]}\n\n"
        f"输出：直接给出提炼结果（中文，尽量保留原文关键细节如数字/路径/版本号，"
        f"不超过 2000 字）。内容与关注点无关时只输出一行：无相关内容。"
    )
    resp = await aux_router.chat_completions(
        [{"role": "user", "content": refine_prompt}],
    )
    try:
        text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""
    text = str(text).strip()
    return text[:2000]


WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": (
        "抓取网页/URL 内容并转为纯文本。"
        "适合查文档、读页面、看 API 说明。传 prompt 时由小模型按关注点提炼"
        "（省上下文，返回 refined=true）；不传返回全文（最多 2 万字符）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 URL（http/https）"},
            "prompt": {
                "type": "string",
                "description": "关注点：抓取后由小模型按此提炼内容（推荐总是提供）",
            },
        },
        "required": ["url"],
    },
}


# 模块级注册：这个文件一被 import 就自动登记进中央注册表
registry.register(
    name="web_fetch",
    toolset="core",
    schema=WEB_FETCH_SCHEMA,
    handler=_handle_web_fetch,
    emoji="🌐",
    isConcurrencySafe=False,  # 要访问外部网络（费带宽、耗时长、外部网站可能有副作用），串行更稳
)
