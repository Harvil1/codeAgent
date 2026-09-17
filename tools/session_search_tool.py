"""会话搜索（session_search）工具：让 AI 反查以前聊过的天。

打个比方：这是 AI 的「聊天记录搜索框」。用户说「我们上次讨论过 X」时，
AI 不用装糊涂，用这个工具在历史会话库里搜关键词，把当时的上下文捞回来。

依赖：搜索靠 session_store（会话存储库）完成，它由 agent 在分发工具时
通过 kwargs 注入进来，本文件不自己创建。
"""

import json
from typing import Optional

from agent.injection_guard import (
    FOREIGN_CONTENT_WARNING, neutralize_control_markers,
)
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
    """在历史会话里按关键词搜聊天记录——把存在磁盘上的历史会话搜出来，
    避免用户重复交代。

    参数：
    - args：工具参数，query 必填（搜索关键词），limit 可选（最多返回
      几条结果，默认 10）。
    - kwargs：框架透传的上下文，取 session_store（会话存储库，真正
      执行搜索；没注入则报「会话存储未初始化」）。

    返回：JSON 字符串，含 results 列表（每条带会话标题/角色/时间/
    内容片段/会话 ID）和 total 总数；没搜到时返回空列表加提示消息。
    """
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

    # 挑 AI 需要的字段重新组一遍（原始记录里还有别的字段，不全部塞回去）。
    # 片段来自历史会话（外源内容）：控制标记中和防伪造，notice 说明
    # "里面的指令只是数据"（防指令注入）。
    formatted = []
    for r in results:
        formatted.append({
            "session_title": r.get("title") or "(无标题)",
            "role": r["role"],
            "timestamp": r["timestamp"],
            "snippet": neutralize_control_markers(r.get("snippet", "")),
            "session_id": r["session_id"],
        })

    return json.dumps({
        "notice": FOREIGN_CONTENT_WARNING,
        "results": formatted,
        "total": len(formatted),
    }, ensure_ascii=False)


# 模块级注册：import 本文件即自动登记进中央注册表
registry.register(
    name="session_search",
    toolset="core",
    schema=SESSION_SEARCH_SCHEMA,
    handler=_handle_session_search,
    emoji="🔍",
    isConcurrencySafe=True,  # 只读：搜历史对话（全文检索查询），无副作用，可并发
)
