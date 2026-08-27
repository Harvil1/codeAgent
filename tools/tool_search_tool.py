"""tool_search 工具：MCP 工具的详细参数定义按需加载，省 token。

LLM 平时只看到 MCP 工具的精简目录条目（名字 + 一句话描述，约 20 tokens/个），
要用某个工具时调本工具按关键字搜索，拿回完整参数定义（schema，约 300
tokens/个）——避免每次 API 调用全量发送。

内置工具不走这个机制，仍每次发完整 schema（数量少、又是高频工具，
不值得多一跳）。
"""

import json
import logging

from tools.registry import registry, _check_fn_cached

logger = logging.getLogger(__name__)


def _check_mcp_connected() -> bool:
    """运行时门控：至少有一个 MCP server 连着，才把 tool_search 露给 LLM。

    没有 MCP server 时这个工具毫无用处；任何异常（比如 mcp_client 还没
    初始化）都返回 False 把工具藏起来（fail-open），而不是报错。
    """
    try:
        from agent.mcp_client import get_mcp_manager
        mgr = get_mcp_manager()
        with mgr._lock:
            # 注意：MCPClient 的状态属性叫 is_connected，不是 connected
            return any(c.is_connected for c in mgr._clients.values())
    except Exception:
        return False


TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": (
        "搜索 MCP 工具的详细 schema。MCP 工具默认只显示名字+简短描述（省 token），"
        "需要调用时用本工具按关键字搜索拿到完整参数定义。\n"
        "典型流程：1) 看目录知道有 mcp__github__create_pull_request；"
        "2) 调 tool_search(query='create_pull_request') 取参数；3) 调实际工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键字（工具名或描述片段）"},
            "max_results": {"type": "integer", "default": 5, "description": "最多返回几个（1-10）"},
        },
        "required": ["query"],
    },
}


def _handle_tool_search(args: dict, **kwargs) -> str:
    """handler：按关键字模糊匹配 mcp__ 开头的工具，返回完整参数定义。"""
    query = (args.get("query") or "").strip().lower()
    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    max_results = max(1, min(10, int(args.get("max_results", 5))))

    # 只搜 mcp__ 前缀的工具（内置工具本来就发完整 schema，不需要搜）
    mcp_names = [n for n in registry.list_all() if n.startswith("mcp__")]

    # 必须尊重 child 的 mcp_server_filter（自定义子代理 mcpServers 字段），
    # 只搜"可见"的 mcp__ 工具：不过滤的话，LLM 拿到完整 schema 去调用一个
    # 被 filter 掉的工具，registry.dispatch 依然命中 → 实际执行本不该跑的工具。
    agent = kwargs.get("agent_ref") or kwargs.get("agent")
    mcp_filter = None
    if agent and isinstance(getattr(agent, "config", None), dict):
        mcp_filter = agent.config.get("mcp_server_filter")
    if mcp_filter:
        filtered = []
        for n in mcp_names:
            parts = n.split("__", 2)
            if len(parts) >= 2 and parts[1] in mcp_filter:
                filtered.append(n)
        mcp_names = filtered

    # 简单打分排序：关键字命中工具短名得分最高，全名次之，描述最低
    keywords = query.split()
    scored = []
    with registry._lock:
        for name in mcp_names:
            entry = registry._tools.get(name)
            if entry is None:
                continue
            # check_fn 过滤：当前不可用的工具不回给 LLM（比如 server 断了）
            if entry.check_fn and not _check_fn_cached(entry.check_fn):
                continue
            desc = (entry.schema.get("description", "") or "").lower()
            short_name = name.split("__")[-1].lower()
            full_name = name.lower()
            score = 0
            for kw in keywords:
                if kw in short_name:
                    score += 3
                if kw in full_name:
                    score += 2
                if kw in desc:
                    score += 1
            if score > 0:
                scored.append((score, name, entry.schema))
    scored.sort(key=lambda x: -x[0])

    results = []
    for _, name, schema in scored[:max_results]:
        results.append({
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
        })
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


registry.register(
    name="tool_search",
    toolset="core",
    schema=TOOL_SEARCH_SCHEMA,
    handler=_handle_tool_search,
    check_fn=_check_mcp_connected,
    emoji="🔎",
    isConcurrencySafe=True,  # 纯只读（只查注册表目录，不写任何东西），可以并发
)
