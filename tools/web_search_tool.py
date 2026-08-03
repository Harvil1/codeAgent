"""web_search 工具：网络搜索（Tavily API）。

对齐 Claude Code WebSearch：返回结构化结果（标题/URL/摘要）。
未配置 TAVILY_API_KEY 时工具自动隐藏（check_fn 返回 False）。
"""

import json
import logging
import os

import requests  # 顶层导入：测试需 patch tools.web_search_tool.requests.post

from tools.registry import registry

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def _check_tavily_configured() -> bool:
    """check_fn：有 TAVILY_API_KEY 才暴露工具。"""
    return bool(os.environ.get("TAVILY_API_KEY"))


WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "网络搜索，返回结构化结果（标题/URL/摘要）。"
        "适合查最新信息、API 文档、库版本。比 web_fetch 更适合'查'而非'读某页'。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "default": 5, "description": "返回结果数（1-10）"},
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "basic",
                "description": "basic 快/浅，advanced 慢/深",
            },
        },
        "required": ["query"],
    },
}


def _handle_web_search(args: dict, **kwargs) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return json.dumps(
            {"error": "未配置 TAVILY_API_KEY", "error_type": "not_configured"},
            ensure_ascii=False,
        )

    max_results = max(1, min(10, int(args.get("max_results", 5))))
    search_depth = args.get("search_depth", "basic")

    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
            },
            timeout=20,
        )
    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Tavily 请求失败: {e}", "error_type": "request_error"},
            ensure_ascii=False,
        )

    if resp.status_code != 200:
        return json.dumps(
            {
                "error": f"Tavily API {resp.status_code}: {resp.text[:200]}",
                "error_type": "tavily_api_error",
            },
            ensure_ascii=False,
        )

    data = resp.json()
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return json.dumps(
        {"query": query, "results": results, "answer": data.get("answer", "")},
        ensure_ascii=False,
    )


registry.register(
    name="web_search",
    toolset="core",
    schema=WEB_SEARCH_SCHEMA,
    handler=_handle_web_search,
    check_fn=_check_tavily_configured,
    emoji="🔍",
)
