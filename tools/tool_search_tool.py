"""tool_search 工具：MCP 工具 schema 按需加载。

对齐 Claude Code ToolSearch：LLM 看到 MCP 工具的精简目录条目（name + 描述），
需要详细参数时调本工具按关键字搜索，返回完整 schema。

省 token 原理：N 个 MCP 工具每次 API 调用都全量发 schema 会吃掉很多 token
（每个工具 ~300 tokens × N）。ToolSearch 把它们压缩成短目录条目
（每个 ~20 tokens × N），只在 LLM 真要用时才回完整 schema。

built-in 工具不走这个机制，仍每次发完整 schema（数量少，且是高频工具）。
"""

import json
import logging

from tools.registry import registry, _check_fn_cached

logger = logging.getLogger(__name__)


def _check_mcp_connected() -> bool:
    """check_fn：至少有一个 MCP server 连接才暴露 tool_search。

    fail-open：任何异常（如 mcp_client 未初始化）都返回 False，
    让 tool_search 自动隐藏（避免暴露无用的工具给 LLM）。
    """
    try:
        from agent.mcp_client import get_mcp_manager
        mgr = get_mcp_manager()
        with mgr._lock:
            # MCPClient 的状态属性是 is_connected（不是 connected）
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
    """tool_search handler：按关键字模糊匹配 mcp__ 工具，返回完整 schema。"""
    query = (args.get("query") or "").strip().lower()
    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    max_results = max(1, min(10, int(args.get("max_results", 5))))

    # 取所有 mcp__ 工具（只对这些生效，built-in 仍走完整 schema）
    mcp_names = [n for n in registry.list_all() if n.startswith("mcp__")]

    # 尊重 child 的 mcp_server_filter（自定义子代理 mcpServers 字段）
    # spec 第 286 行承诺：ToolSearch 只搜可见 mcp__ 工具（filter 后的子集）。
    # 否则 LLM 拿到完整 schema 后调用，registry.dispatch 仍命中 → 实际执行被 filter 掉的工具。
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

    # 打分（关键字在 name 或 description 里）
    keywords = query.split()
    scored = []
    with registry._lock:
        for name in mcp_names:
            entry = registry._tools.get(name)
            if entry is None:
                continue
            # check_fn 过滤（不可用的不回给 LLM）
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
    isConcurrencySafe=True,  # 只读：查 registry 目录（无写），可并发
)
