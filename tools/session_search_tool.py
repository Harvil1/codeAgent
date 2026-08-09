"""session_search 工具：让 agent 搜索过去的对话。

通过 kwargs 接收 session_store（由 agent 在 dispatch 时注入）。
"""

import json
from typing import Optional

from tools.registry import registry


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "搜索过去的对话历史。当用户提到之前聊过的内容，"
        "或你怀疑有相关的跨会话上下文时使用。\n"
        "避免让用户重复说过的话。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数（默认 10）",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


def _handle_session_search(args: dict, **kwargs) -> str:
    query = (args.get("query") or "").strip()
    limit = args.get("limit", 10)

    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    session_store = kwargs.get("session_store")
    if session_store is None:
        return json.dumps({"error": "会话存储未初始化"}, ensure_ascii=False)

    results = session_store.search(query, limit=limit)

    if not results:
        return json.dumps({
            "results": [],
            "message": "未找到匹配的对话",
        }, ensure_ascii=False)

    # 格式化结果给 LLM
    formatted = []
    for r in results:
        formatted.append({
            "session_title": r.get("title") or "(无标题)",
            "role": r["role"],
            "timestamp": r["timestamp"],
            "snippet": r.get("snippet", ""),
            "session_id": r["session_id"],
        })

    return json.dumps({
        "results": formatted,
        "total": len(formatted),
    }, ensure_ascii=False)


registry.register(
    name="session_search",
    toolset="core",
    schema=SESSION_SEARCH_SCHEMA,
    handler=_handle_session_search,
    emoji="🔍",
    isConcurrencySafe=True,  # 只读：搜历史对话（FTS 查询），无副作用，可并发
)
