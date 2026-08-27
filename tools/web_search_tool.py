"""网络搜索工具：替模型上网查资料（背后调 Tavily 搜索服务的接口）。

返回结构化结果——每条含标题、网址、摘要。
没配置 TAVILY_API_KEY 环境变量时，这个工具会自动"隐身"（check_fn 返回 False，
模型根本看不到它，也就不会白调用然后报错）。
"""

import json
import logging
import os

# requests 必须在文件顶部导入而不能在函数里：测试要用
# patch tools.web_search_tool.requests.post 来替换它，函数内导入会 patch 不中
import requests

from tools.registry import registry

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def _check_tavily_configured() -> bool:
    """开关函数（check_fn）：配置了 TAVILY_API_KEY 才把这个工具亮给模型看。

    注册表每次暴露工具列表前调此函数决定显隐；没配 key 时搜索必然失败，直接隐藏。

    返回：True 表示已配置（显示工具），False 表示没配置（隐藏）。
    """
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
    """上网搜索关键词，返回一组结构化结果（标题/网址/摘要）。

    流程：校验关键词和 API key → 把搜索条数夹在 1~10 → 调 Tavily 接口 →
    把结果挑拣成统一格式返回。

    参数：
        args：工具参数字典，来自模型——query（搜索关键词）、
            max_results（要几条结果）、search_depth（basic 快而浅 /
            advanced 慢而深）。
        **kwargs：分发器注入的运行上下文（本函数未用到，签名保持
            工具统一契约）。

    返回：JSON 字符串，成功含 query / results（每条有 title、url、content）/
        answer（Tavily 顺带给的总答案）；失败是 {"error": ..., "error_type": ...}。
    """
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


# 模块级注册：这个文件一被 import 就自动登记进中央注册表
registry.register(
    name="web_search",
    toolset="core",
    schema=WEB_SEARCH_SCHEMA,
    handler=_handle_web_search,
    check_fn=_check_tavily_configured,
    emoji="🔍",
    isConcurrencySafe=False,  # 要调外部 Tavily 接口（费调用配额、耗时长），串行更稳
)
