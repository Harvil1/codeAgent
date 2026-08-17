"""把 MCP server 的工具注册到 OmniMate 的 registry。

启动时调用 register_mcp_tools()，把所有连接的 MCP server 的工具
以 mcp__<server>__<tool> 前缀注册到 registry，让 LLM 能调用。

check_fn：只在 MCP server 连接时才暴露工具（动态门控）。
"""

import json
import logging
from pathlib import Path
from typing import Dict

from agent.mcp_client import get_mcp_manager, MCPManager
from tools.registry import registry

logger = logging.getLogger(__name__)


def _make_server_check(mgr: MCPManager, sname: str):
    """构造 per-server 连接门控 check_fn（闭包捕获 manager + server 名）。

    对应 server 的 client 存在且 connected 时才暴露工具（server 断开自动隐藏）。
    """
    def check():
        with mgr._lock:
            client = mgr._clients.get(sname)
        return client is not None and client.connected
    return check


def register_mcp_tools(manager: MCPManager = None, servers: list = None) -> int:
    """把 MCP server 的工具注册到 registry。

    servers=None 注册全部；否则只注册指定 server 列表（R24 #38 内联临时
    server 用——只暴露 agent 声明的那些）。返回注册的工具数量。
    """
    if manager is None:
        manager = get_mcp_manager()

    tools = manager.get_all_tools()
    if servers is not None:
        wanted = set(servers)
        tools = [t for t in tools if t.get("server") in wanted]
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
        try:
            registry.register(
                name=full_name,
                toolset="mcp",
                schema=schema,
                handler=make_handler(manager, full_name),
                check_fn=_make_server_check(manager, server_name),
                emoji="🔌",
                # MCP 工具保守标 False：不知道具体副作用（可能是写文件/发请求），
                # 安全默认 > 事后补救，让它们走串行路径
                isConcurrencySafe=False,
            )
            count += 1
        except Exception as e:
            logger.warning("注册 MCP 工具 %s 失败: %s", full_name, e)

    # CCAR12 Task 5：每个连接中的 server 额外注册 resources 协议工具
    # （命名对齐 mcp__<server>__<tool> 动态模式，check_fn 同款 per-server 门控）
    count += _register_resource_tools(manager)

    if count:
        logger.info("已注册 %d 个 MCP 工具", count)
    return count


def _register_resource_tools(manager: MCPManager) -> int:
    """为每个连接中的 MCP server 注册 list/read resources 两工具。

    注册名：mcp__<server>__list_resources / mcp__<server>__read_resource，
    走 registry 的 mcp__ 动态命名空间（model_tools 自动发现 + catalog 精简条目
    + mcp_server_filter 过滤都天然生效，不发明新机制）。

    server 不支持 resources 协议时工具仍注册（调用时返回友好错误，
    而不是启动时探测一次就永久隐藏——能力探测留 follow-up）。
    """
    with manager._lock:
        clients = {
            name: client for name, client in manager._clients.items()
            if client.connected
        }

    registered = 0
    for server_name in sorted(clients):
        # ---- mcp__<server>__list_resources（无参数）----
        list_schema = {
            "name": f"mcp__{server_name}__list_resources",
            "description": (
                f"列出 MCP server {server_name} 的 resources"
                f"（uri/name/mimeType/description）"
                f" [MCP server: {server_name}]"
            ),
            "parameters": {"type": "object", "properties": {}},
        }

        def make_list_handler(mgr, sname):
            def handler(args: dict, **kwargs) -> str:
                result = mgr.list_resources(sname)
                return json.dumps(result, ensure_ascii=False)
            return handler

        # ---- mcp__<server>__read_resource（uri 参数）----
        read_schema = {
            "name": f"mcp__{server_name}__read_resource",
            "description": (
                f"读取 MCP server {server_name} 的单个 resource 内容"
                f"（按 uri） [MCP server: {server_name}]"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "resource 的 URI（来自 list_resources）",
                    },
                },
                "required": ["uri"],
            },
        }

        def make_read_handler(mgr, sname):
            def handler(args: dict, **kwargs) -> str:
                uri = (args or {}).get("uri", "")
                result = mgr.read_resource(sname, uri)
                return json.dumps(result, ensure_ascii=False)
            return handler

        # check_fn 复用现有 per-server 门控
        for name, schema, handler in (
            (list_schema["name"], list_schema, make_list_handler(manager, server_name)),
            (read_schema["name"], read_schema, make_read_handler(manager, server_name)),
        ):
            try:
                registry.register(
                    name=name,
                    toolset="mcp",
                    schema=schema,
                    handler=handler,
                    check_fn=_make_server_check(manager, server_name),
                    emoji="🔌",
                    # resources 读取理论上只读，但走外部进程/网络，
                    # 与其他 MCP 工具一致保守标 False（串行）
                    isConcurrencySafe=False,
                )
                registered += 1
            except Exception as e:
                logger.warning("注册 MCP resources 工具 %s 失败: %s", name, e)

    return registered


def initialize_mcp(approval_callback=None) -> int:
    """启动时调用：加载配置、连接 server、注册工具。

    approval_callback: fn(name, desc) -> bool。项目级 .mcp.json 的 server
    首次连接前必须获批（R25 #3，对齐 CC mcpServerApproval）；未批准/无
    callback → fail-closed 跳过。用户级 ~/.OmniMate/.mcp.json 是用户直接
    编辑的，不需要审批。

    返回注册的工具数量。
    """
    manager = get_mcp_manager()
    try:
        # 用户级（受用户直接控制）
        manager.connect_all()

        # R25 #3：项目级 .mcp.json 首连审批
        from agent.mcp_client import load_mcp_config, load_project_mcp_config
        from agent.settings import is_project_mcp_approved, persist_project_mcp_approval

        proj_path, proj_servers = load_project_mcp_config()
        user_cfg = load_mcp_config()
        if proj_path and proj_servers:
            # 路径小写归一（Windows 盘符大小写不敏感）
            proj_key = str(proj_path.parent.resolve()).lower()
            approved_now: Dict[str, dict] = {}
            for name, cfg in proj_servers.items():
                if name in user_cfg:
                    logger.warning(
                        "项目 MCP server %s 与用户级同名，跳过项目级（用户级优先）", name,
                    )
                    continue
                from agent.settings import mcp_approval_key
                key = mcp_approval_key(proj_key, name, cfg)
                if not is_project_mcp_approved(key):
                    desc = json.dumps(
                        {k: cfg.get(k) for k in ("command", "url", "transport", "args")},
                        ensure_ascii=False,
                    )
                    ok = False
                    if approval_callback is not None:
                        try:
                            ok = bool(approval_callback(name, desc))
                        except Exception as e:
                            logger.warning("MCP 审批 callback 异常（视为拒绝）: %s", e)
                    if not ok:
                        logger.warning(
                            "项目 MCP server %s 未获批准，跳过（fail-closed）", name,
                        )
                        continue
                    persist_project_mcp_approval(key)
                approved_now[name] = cfg
            if approved_now:
                manager.connect_all(approved_now)

        # R29 #2：项目级 agent .md 的内联 MCP server 同款首连审批
        # （威胁模型与项目 .mcp.json 相同：clone 陌生 repo 带入）
        try:
            from agent.agent_defs import project_inline_mcp_servers
            inline_servers = project_inline_mcp_servers()
        except Exception:
            inline_servers = {}
        if inline_servers:
            from agent.settings import mcp_approval_key
            # proj_key 统一用 workspace cwd resolve lower（项目可能只有
            # agent 内联没有 .mcp.json——与 delegate spawn 校验处必须同源）
            from agent.workspace_context import get_workspace_cwd
            try:
                inline_proj_key = str(Path(get_workspace_cwd()).resolve()).lower()
            except Exception:
                inline_proj_key = ""
            inline_approved: Dict[str, dict] = {}
            for name, cfg in inline_servers.items():
                if name in user_cfg:
                    continue  # 用户级同名优先，不重复处理
                key = mcp_approval_key(inline_proj_key, f"agent-mcp::{name}", cfg)
                if not is_project_mcp_approved(key):
                    desc = json.dumps(
                        {k: cfg.get(k) for k in ("command", "url", "transport", "args")},
                        ensure_ascii=False,
                    )
                    ok = False
                    if approval_callback is not None:
                        try:
                            ok = bool(approval_callback(
                                f"(agent 内联) {name}", desc))
                        except Exception as e:
                            logger.warning("内联 MCP 审批 callback 异常（视为拒绝）: %s", e)
                    if not ok:
                        logger.warning(
                            "项目 agent 内联 MCP server %s 未获批准，"
                            "spawn 时将跳过（fail-closed）", name,
                        )
                        continue
                    persist_project_mcp_approval(key)
                inline_approved[name] = cfg
            # 内联 server 不在此连接（spawn 时临时连）——审批只管放行名单
            if inline_approved:
                logger.info("项目 agent 内联 MCP 已批准: %s", sorted(inline_approved))
        return register_mcp_tools(manager)
    except Exception as e:
        logger.warning("MCP 初始化失败: %s", e)
        return 0
