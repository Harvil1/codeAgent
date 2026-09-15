"""把 MCP server（外挂工具服务）的工具登记进 CodeAgent 的工具注册表。

MCP 是接外部工具的标准协议。程序启动时会调 register_mcp_tools()，
把所有已连接的 MCP server 提供的工具按 mcp__<server>__<tool> 的命名
登记到 registry（中央工具注册表），LLM 就能像调内置工具一样调用它们。

check_fn（运行时门控函数）负责动态可见性：只有对应 server 还连着，
工具才出现在 LLM 面前；server 一断开就自动隐藏。
"""

import json
import logging
from pathlib import Path
from typing import Dict

from agent.mcp_client import get_mcp_manager, MCPManager
from tools.registry import registry

logger = logging.getLogger(__name__)


def _make_server_check(mgr: MCPManager, sname: str):
    """造一个"这个 server 还连着吗"的检查函数（闭包记住 manager 和 server 名）。

    registry 的 check_fn 机制用它决定工具显不显——对应 server 的 client
    存在且已连接才返回 True（server 断开时工具自动隐藏）。
    """
    def check():
        with mgr._lock:
            client = mgr._clients.get(sname)
        return client is not None and client.connected
    return check


def register_mcp_tools(manager: MCPManager = None, servers: list = None) -> int:
    """把 MCP server 的工具批量登记进 registry。

    启动时全量登记；内联临时 server 只需暴露 agent 声明的那几个，
    所以有 servers 过滤参数。

    参数：
        manager: MCP 管理器（None 时自动取全局单例）
        servers: 只登记这些名字的 server（None = 全部登记）

    返回：
        成功登记的工具数量
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

        # 闭包包住 manager 和 full_name（防循环变量晚绑定串号）
        def make_handler(mgr, fname):
            def handler(args: dict, **kwargs) -> str:
                result = mgr.call(fname, args or {})
                return json.dumps(result, ensure_ascii=False)
            return handler

        # check_fn 门控：对应 server 连着才暴露
        try:
            registry.register(
                name=full_name,
                toolset="mcp",
                schema=schema,
                handler=make_handler(manager, full_name),
                check_fn=_make_server_check(manager, server_name),
                emoji="🔌",
                # MCP 工具一律保守标 False：外部工具的副作用我们看不见
                # （可能写文件、发网络请求），安全默认 > 事后补救，
                # 让它们排队串行执行
                isConcurrencySafe=False,
            )
            count += 1
        except Exception as e:
            logger.warning("注册 MCP 工具 %s 失败: %s", full_name, e)

    # 给每个连着的 server 追加 resources（资源清单）协议工具，
    # 命名对齐 mcp__<server>__<tool> 动态模式，check_fn 用同款 per-server 门控。
    # servers 过滤同样要带上——内联 spawn 场景只该暴露声明的子集，
    # 这里给所有 server 无条件注册会让过滤形同虚设
    count += _register_resource_tools(manager, servers=servers)

    if count:
        logger.info("已注册 %d 个 MCP 工具", count)
    return count


def _register_resource_tools(manager: MCPManager, servers: list = None) -> int:
    """给每个连着的 MCP server 配上"列资源/读资源"两个工具。

    MCP server 除了工具还能提供 resources（静态资源，比如一份文档）。
    注册名是 mcp__<server>__list_resources / mcp__<server>__read_resource，
    走 registry 现成的 mcp__ 动态命名空间——model_tools 自动发现、
    catalog 精简条目、mcp_server_filter 过滤全都天然生效。

    注意：server 不支持 resources 协议时工具照样注册（调用时会返回
    友好错误），而不是启动时探测一次就永久藏起来——能力探测留作 follow-up。

    参数：
        manager: MCP 管理器
        servers: 只登记这些名字的 server（None = 全部；与
                register_mcp_tools 的过滤同源）

    返回：
        成功登记的工具数量
    """
    wanted = set(servers) if servers is not None else None
    with manager._lock:
        clients = {
            name: client for name, client in manager._clients.items()
            if client.connected
            and (wanted is None or name in wanted)
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

        # ---- mcp__<server>__read_resource（按 uri 读）----
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

        # check_fn 直接复用现有的 per-server 门控
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
                    # 读资源理论上只读，但毕竟走外部进程/网络，
                    # 跟其他 MCP 工具一致保守标 False（串行）
                    isConcurrencySafe=False,
                )
                registered += 1
            except Exception as e:
                logger.warning("注册 MCP resources 工具 %s 失败: %s", name, e)

    return registered


def initialize_mcp(approval_callback=None) -> int:
    """启动时的总入口：加载配置 → 连接 server → 登记工具。

    审批只针对"项目级"配置——用户级 ~/.codeAgent/.mcp.json 是用户自己手写的，
    天然可信；但项目里的 .mcp.json 可能是 clone 陌生仓库带进来的，所以每个
    server 第一次连接前必须先过审批。没批准、或者压根没有
    approval_callback（非交互场景）→ fail-closed 直接跳过不连。

    参数：
        approval_callback: 审批函数 fn(name, desc) -> bool，
            返回 True 表示用户同意连接这个 server；None 表示没人可问

    返回：
        登记的工具数量（初始化失败返回 0）
    """
    manager = get_mcp_manager()
    # 应用配置（feature flag 用）：sse/websocket transport 的开关在
    # settings.json features 节——不传 app_config 的话 is_feature_enabled
    # 拿到 None 按 fail-safe 恒 False，用户配置开了也连不上（内联 spawn
    # 路径 delegate_child 就传了，这里是漏传不是设计）
    try:
        from agent.settings import load_settings
        _app_cfg = load_settings()
    except Exception:
        _app_cfg = None
    try:
        # 用户级配置（用户直接控制，直接连）
        manager.connect_all(app_config=_app_cfg)

        # 项目级 .mcp.json 首连审批
        from agent.mcp_client import load_mcp_config, load_project_mcp_config
        from agent.settings import is_project_mcp_approved, persist_project_mcp_approval

        proj_path, proj_servers = load_project_mcp_config()
        user_cfg = load_mcp_config()
        if proj_path and proj_servers:
            # 路径转小写归一（Windows 盘符大小写不敏感，避免同路径判成两个项目）
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
                manager.connect_all(approved_now, app_config=_app_cfg)

        # 项目级 agent .md 里内联声明的 MCP server 走同款首连审批
        # （威胁模型跟项目 .mcp.json 一样：clone 陌生 repo 可能带进恶意配置）
        try:
            from agent.agent_defs import project_inline_mcp_servers
            inline_servers = project_inline_mcp_servers()
        except Exception:
            inline_servers = {}
        if inline_servers:
            from agent.settings import mcp_approval_key
            # proj_key 统一取 workspace cwd resolve 后转小写——项目可能只有
            # agent 内联而没有 .mcp.json，这里必须跟 delegate spawn 校验处
            # 同源，否则两边算出不同的 key 会对不上
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
            # 内联 server 不在这里连接（等 spawn 时临时连）——审批只管放行名单
            if inline_approved:
                logger.info("项目 agent 内联 MCP 已批准: %s", sorted(inline_approved))

        # 插件自带的 MCP server（.mcp.json）：用户主动 /plugin install
        # 装的 = 明确授权，直接连，不再走审批。与用户级同名的跳过
        # （用户手写的配置优先级最高，跟项目级同款规则）。
        try:
            plugin_servers = collect_plugin_mcp_servers()
        except Exception as e:
            logger.warning("插件 MCP 扫描失败（跳过）: %s", e)
            plugin_servers = {}
        plugin_wanted = {
            conn: info["cfg"] for conn, info in plugin_servers.items()
            if conn not in user_cfg
        }
        if plugin_wanted:
            connect_servers_and_register(plugin_wanted)

        return register_mcp_tools(manager)
    except Exception as e:
        logger.warning("MCP 初始化失败: %s", e)
        return 0


def collect_plugin_mcp_servers() -> Dict[str, dict]:
    """扫所有已启用插件根下的 .mcp.json，汇总插件要连的 MCP server。

    只看「已启用」的插件（清单 enabled 字段，判法和
    constants.all_skills_dirs 一致）；官方 claude code 布局
    （.claude-plugin/plugin.json）也认。

    官方插件配置里常用 ${CLAUDE_PLUGIN_ROOT} 指插件自己目录——这里
    替换成实际路径（只支持这一个变量，别的原样保留）。

    返回 {连接名: {"plugin": 插件名, "cfg": server配置, "dir": 插件目录}}。
    连接名默认用 server 本名；两个插件撞名时后扫的改成
    <插件名>_<server>（扫描按名字排序，结果确定）。
    """
    from constants import plugins_dir
    import json as _json

    root = plugins_dir()
    out: Dict[str, dict] = {}
    if not root.exists():
        return out
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir() or pdir.name == "marketplaces":
            continue
        mf = pdir / "plugin.json"
        if not mf.exists():
            mf = pdir / ".claude-plugin" / "plugin.json"
        if not mf.exists():
            continue
        try:
            data = _json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("enabled", True):
            continue
        mcp_file = pdir / ".mcp.json"
        if not mcp_file.exists():
            continue
        try:
            servers = _json.loads(
                mcp_file.read_text(encoding="utf-8")
            ).get("mcpServers", {})
        except Exception as e:
            logger.warning("插件 %s 的 .mcp.json 读坏（跳过）: %s", pdir.name, e)
            continue
        if not isinstance(servers, dict):
            continue
        for sname, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            # ${CLAUDE_PLUGIN_ROOT} → 插件实际路径（command/args 里都换）
            cfg = _json.loads(_json.dumps(cfg))  # 深拷贝，别改插件原文件的内容
            token = "${CLAUDE_PLUGIN_ROOT}"
            for key in ("command",):
                if isinstance(cfg.get(key), str):
                    cfg[key] = cfg[key].replace(token, str(pdir))
            if isinstance(cfg.get("args"), list):
                cfg["args"] = [
                    a.replace(token, str(pdir)) if isinstance(a, str) else a
                    for a in cfg["args"]
                ]
            conn = sname if sname not in out else f"{pdir.name}_{sname}"
            out[conn] = {"plugin": pdir.name, "cfg": cfg, "dir": pdir}
    return out


def connect_servers_and_register(servers: Dict[str, dict]) -> int:
    """连接一批 MCP server 并登记它们的工具（插件接线用，逐个 fail-open）。

    参数：
        servers：{连接名: server配置}（collect_plugin_mcp_servers 产出的
                 cfg 那份形状）

    返回：
        成功登记的工具数。单个 server 连不上只记日志跳过，不连累别的。
    """
    manager = get_mcp_manager()
    # app_config 同样要带上（sse/ws flag）——与 initialize_mcp 主路径一致
    try:
        from agent.settings import load_settings
        _app_cfg = load_settings()
    except Exception:
        _app_cfg = None
    connected: list = []
    for name, cfg in servers.items():
        try:
            manager.connect_one(name, cfg, app_config=_app_cfg)
            connected.append(name)
        except Exception as e:
            logger.warning("MCP server %s 连接失败（跳过）: %s", name, e)
    if not connected:
        return 0
    return register_mcp_tools(manager, servers=connected)
