"""把 MCP server 的工具注册到 OmniMate 的 registry。

启动时调用 register_mcp_tools()，把所有连接的 MCP server 的工具
以 mcp__<server>__<tool> 前缀注册到 registry，让 LLM 能调用。

check_fn：只在 MCP server 连接时才暴露工具（动态门控）。
"""

import json
import logging
from typing import Dict

from agent.mcp_client import get_mcp_manager, MCPManager
from tools.registry import registry

logger = logging.getLogger(__name__)


def register_mcp_tools(manager: MCPManager = None) -> int:
    """把所有 MCP server 的工具注册到 registry。

    返回注册的工具数量。
    """
    if manager is None:
        manager = get_mcp_manager()

    tools = manager.get_all_tools()
    count = 0

    for tool in tools:
        full_name = tool["full_name"]
        server_name = tool["server"]
        original_name = tool["original_name"]

        schema = {
            "name": full_name,
            "description": (
                f"{tool.get('description', '')} "
                f"[MCP server: {server_name}]"
            ).strip(),
            "parameters": tool.get("inputSchema") or {
                "type": "object",
                "properties": {},
            },
        }

        # 闭包捕获 manager 和 full_name
        def make_handler(mgr, fname):
            def handler(args: dict, **kwargs) -> str:
                result = mgr.call(fname, args or {})
                return json.dumps(result, ensure_ascii=False)
            return handler

        # check_fn：对应 server 连接时才暴露
        def make_check(mgr, sname):
            def check():
                with mgr._lock:
                    client = mgr._clients.get(sname)
                return client is not None and client.connected
            return check

        try:
            registry.register(
                name=full_name,
                toolset="mcp",
                schema=schema,
                handler=make_handler(manager, full_name),
                check_fn=make_check(manager, server_name),
                emoji="🔌",
            )
            count += 1
        except Exception as e:
            logger.warning("注册 MCP 工具 %s 失败: %s", full_name, e)

    if count:
        logger.info("已注册 %d 个 MCP 工具", count)
    return count


def initialize_mcp() -> int:
    """启动时调用：加载配置、连接 server、注册工具。

    返回注册的工具数量。
    """
    manager = get_mcp_manager()
    try:
        manager.connect_all()
        return register_mcp_tools(manager)
    except Exception as e:
        logger.warning("MCP 初始化失败: %s", e)
        return 0
