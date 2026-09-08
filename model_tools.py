"""工具分发层——LLM 和工具之间的总调度台。

把主对话（agent）和中央工具登记处（registry，一个登记所有工具的大账本）连起来：
- 主对话开场时调 get_tool_definitions()，拿到要发给 LLM 的"工具说明书"列表
  （schema，告诉 LLM 有哪些工具可用、每个工具怎么传参数）
- LLM 决定调用某个工具后，主对话调 handle_function_call() 去真正执行
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, discover_builtin_tools
from toolsets import resolve_toolset

logger = logging.getLogger(__name__)

# 模块级暂存：记下最近一次解析出哪些工具名（调试或 UI 展示用）
_last_resolved_tool_names: List[str] = []

# 工具名清单解析缓存：键 -> tuple(tool_names)。每轮重建 toolset 展开 +
# mcp list_all 扫描 + deny 过滤在工具多时是重复劳动；registry.generation
# 在任何登记/注销时 +1，天然当失效信号。只缓存名字（definitions 构建不缓存
# ——schema_overrides_fn 带实时槽位信息，缓存会出陈旧值）。
_tool_names_cache: dict = {}

# 标记工具发现这件事是否已经做过（做过就不重复做）
_tools_discovered = False


def ensure_tools_discovered():
    """确保所有工具模块已被加载。

    工具模块在被 import 的那一刻会自动把自己登记进 registry（"自注册"
    机制），但 import 不会自动发生，需要有人推一把。
    幂等：重复调用没有副作用，第一次之后的调用直接跳过。
    """
    global _tools_discovered
    if _tools_discovered:
        return
    discover_builtin_tools()
    _tools_discovered = True


def get_tool_definitions(
    enabled_toolsets: List[str],
    *,
    disabled_tools: List[str] = None,
    agent=None,
) -> List[dict]:
    """拿到本轮要发给 LLM 的"工具说明书"列表（每份说明书都花 token，
    按场景筛选出真正该可见的那批，不全量发）。

    流程（大白话）：
    1. 先确保工具都登记好了（推一把 import）
    2. 把启用（enable）的工具集展开成一份工具名清单
    3. 划掉明确禁用的工具
    4. 去登记处取每份说明书；check_fn（工具自带的"我现在适不适合出场"检查）
       不通过的工具会被自动略过——这就实现了"登记了但当前不可见"

    名字清单这套解析（第 2、3 步的展开 + mcp 扫描 + 过滤）带缓存：
    键里含 registry.generation（每次登记/注销自动 +1），名单没变就直接
    复用上一轮的结果；但"说明书"本身绝不缓存——schema_overrides_fn
    带实时槽位信息，缓存会出陈旧值。

    参数：
        enabled_toolsets: 启用哪些工具集（如 ["core", "mcp"]）。
        disabled_tools: 要明确划掉的工具名清单。
        agent: 当前 AIAgent 实例。传了它，工具的"动态改写说明书"回调
            （schema_overrides_fn）就能拿到它——比如让说明书里的
            并发剩余槽位数反映当下真实情况。

    返回：schema 字典列表，直接可以塞进 LLM 请求。
    """
    ensure_tools_discovered()

    # ===== 先把所有"影响名单的开关"取出来，压成缓存键 =====
    # 子代理可以通过 config["mcp_server_filter"]
    # 圈定"只看得见哪几个 MCP 服务器"，防止它乱碰别的服务器
    mcp_filter = (agent.config.get("mcp_server_filter")
                  if agent and isinstance(getattr(agent, "config", None), dict)
                  else None)
    # 技能作用域同样影响名单，一并取出来进缓存键
    scope = getattr(agent, "_skill_tool_scope", None) if agent else None
    scope_key = (
        (tuple(scope[0] or ()), tuple(scope[1] or ()))
        if scope else None
    )
    # settings.json 里 permissions.deny 配的禁用规则，要在 LLM 看到之前就把
    # 工具整类拿掉（支持精确名 / mcp__server__* 通配 / mcp__server 整服务器）。
    # rules 只在这里加载一次再传给 is_tool_denied——旧版逐工具各自调
    # is_tool_denied(n)，每次都要 exists+stat 一遍 settings.json，40+ 个
    # 工具就是 40+ 次系统调用/轮（Windows 上 stat 不便宜）。
    # allow/deny 两份清单都压进缓存键（任一规则改了缓存跟着失效）。
    # 仍选 fail-open（配置坏了就全拒会把 agent 砖死），但必须大声报 ERROR。
    _deny_key: tuple = ()
    _rules = None
    try:
        from agent.tool_permissions import (
            is_tool_denied, load_tool_permission_rules,
        )
        _rules = load_tool_permission_rules()
        # 键要同时含 allow 和 deny：is_tool_denied 先查 allow 豁免再查 deny，
        # 只压 deny 的话改 allow 不会失效缓存（豁免不生效直到 generation 变化）
        _deny_key = (
            tuple(_rules.get("allow") or []),
            tuple(_rules.get("deny") or []),
        )
    except Exception as e:
        logger.error("deny 规则加载失败，工具可见性过滤本调用失效（fail-open）: %s", e)

    _cache_key = (
        tuple(enabled_toolsets),
        tuple(disabled_tools or ()),
        scope_key,
        tuple(mcp_filter or []),
        registry.generation,
        _deny_key,
    )
    _hit = _tool_names_cache.get(_cache_key)
    if _hit is not None:
        # 命中缓存：整段名字解析（展开/扫描/去重/过滤）全省了
        tool_names = list(_hit)
    else:
        # 把每个启用的工具集展开成具体工具名
        tool_names: List[str] = []
        for ts in enabled_toolsets:
            tool_names.extend(resolve_toolset(ts))

        # mcp 工具集特殊：它的工具是运行时连上外部服务器才动态登记的，
        # 名字都以 mcp__ 开头，这里现场扫一遍登记处把它们捞出来
        # （mcp_filter 已在缓存键构造处取好，这里直接复用）
        if "mcp" in enabled_toolsets:
            for name in registry.list_all():
                if name.startswith("mcp__") and name not in tool_names:
                    # 名字格式是 mcp__<服务器名>__<工具名>；有 filter 时只留圈定的服务器
                    if mcp_filter:
                        parts = name.split("__", 2)
                        if len(parts) >= 2 and parts[1] not in mcp_filter:
                            continue
                    tool_names.append(name)

        # 去重（保持原有先后顺序不变；dict.fromkeys 是"按首次出现去重"的惯用写法）
        tool_names = list(dict.fromkeys(tool_names))

        # 划掉明确禁用的
        if disabled_tools:
            disabled_set = set(disabled_tools)
            tool_names = [n for n in tool_names if n not in disabled_set]

        # 技能可以在触发时改工具可见范围（声明"只许用这些 / 不许用那些"），
        # 这里应用这个作用域。注意：这个作用域一旦设置，整个会话都生效
        # （scope 已在缓存键构造处取好，这里直接复用）
        if scope:
            allowed_tools, disallowed_tools_scope = scope
            if allowed_tools:
                allow_set = set(allowed_tools)
                tool_names = [n for n in tool_names if n in allow_set]
            if disallowed_tools_scope:
                dis_scope = set(disabled_tools or []) | set(disallowed_tools_scope)
                tool_names = [n for n in tool_names if n not in dis_scope]

        # deny 过滤：_rules 已在缓存键构造处加载好，这里传参复用；
        # _rules 为 None 说明那边加载失败——跳过过滤保持 fail-open
        # （ERROR 已经在键构造处大声报过了）
        if _rules is not None:
            tool_names = [n for n in tool_names if not is_tool_denied(n, rules=_rules)]

        # 容量保护：键里含 generation，老键永远不会再命中，攒超过 32 个就整个清空
        if len(_tool_names_cache) > 32:
            _tool_names_cache.clear()
        _tool_names_cache[_cache_key] = tuple(tool_names)

    global _last_resolved_tool_names
    _last_resolved_tool_names = tool_names

    # 为什么要分两拨下发：
    # - 自带工具：数量少、用得勤，直接发完整说明书
    # - mcp__ 工具：可能一大堆，全发完整说明书太费 token，所以只发
    #   "目录条目"（名字 + 一句话简介）；LLM 真想用时调 tool_search
    #   按需取详细参数——这就是"目录先行、详情按需"的省 token 设计
    # 注意：上面的 mcp_server_filter 已经筛过 tool_names 里的 mcp__ 名字，
    # 这里只是让筛剩下的 mcp__ 走目录路径，两步不冲突。
    builtin_names = [n for n in tool_names if not n.startswith("mcp__")]
    mcp_names = [n for n in tool_names if n.startswith("mcp__")]

    runtime_ctx = {"agent": agent} if agent is not None else None
    definitions = registry.get_definitions(builtin_names, quiet=True, runtime_ctx=runtime_ctx)

    # mcp__ 用精简目录条目（只有名字 + 一句话描述 + 提示）
    for name in mcp_names:
        cat = registry.get_catalog_entry(name)
        if cat is not None:
            definitions.append({"type": "function", "function": cat})

    return definitions


async def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    memory_store=None,
    session_store=None,
    codeagent_home=None,
    tool_call_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    hooks_registry=None,
    bg_manager=None,
    team_bus=None,
    team_coordinator=None,
    team_name=None,
    agent_ref=None,
) -> str:
    """执行 LLM 发来的工具调用，返回 JSON 字符串形式的结果。

    这是主对话调工具的唯一入口。做成 async（异步）是因为工具执行期间
    不该卡住整个程序；各种"环境信息"（记忆库、会话库、hook 登记处等）
    会原样透传给工具本体，工具想要哪个自己取。

    时机上还夹了三层"钩子"（hook，工具执行前后插入的自定义逻辑）：
    - PRE_TOOL_USE（执行前）：可以喊"停"（结果直接短路返回 hook_deny），
      也可以偷偷替换传给工具的参数
    - POST_TOOL_USE（执行后）：可以改写工具返回的结果字符串
    - hooks_registry 传 None、或配置里 hooks.enabled=False 时，这些钩子
      全部跳过（老用法完全不受影响）

    参数：
        function_name: LLM 要调的工具名。
        function_args: LLM 传的工具参数（字典）。
        task_id: 关联的持久化任务 ID（如果有）。
        session_id: 当前会话 ID（hook 里区分"这是哪个会话的事"用）。
        memory_store: 记忆库对象（工具按需取用）。
        session_store: 会话库对象（工具按需取用）。
        codeagent_home: agent 的数据根目录（默认 ~/.codeAgent）。
        tool_call_id: 本次工具调用的编号（对应消息历史里那条 tool 消息）。
        config: 运行时配置字典。
        hooks_registry: 钩子登记处（None = 不跑任何钩子）。
        bg_manager: 后台任务管理器。
        team_bus: 团队同步通信总线。
        team_coordinator: 团队协调者对象。
        team_name: 团队名。
        agent_ref: 主对话 agent 的引用（让工具能反过头来找主对话办事）。

    返回：JSON 字符串。这是项目铁律——工具结果必须是 JSON 字符串，
    错误统一是 {"error": ..., "error_type": ...}。
    """
    ensure_tools_discovered()

    # 执行前的钩子
    hooks_enabled = (config or {}).get("hooks", {}).get("enabled", True)
    if hooks_registry and hooks_enabled:
        # 钩子链可能要等子进程跑完，
        # 直接在事件循环线程里等会把整个循环冻住（流式输出、并发工具全停），
        # 所以丢到旁边的工作线程去执行
        deny_reason, modified_args = await asyncio.to_thread(
            hooks_registry.run_pre_tool_use,
            function_name, function_args,
            session_id=session_id or "",
        )
        if deny_reason is not None:
            return json.dumps({
                "error": f"hook denied: {deny_reason}",
                "error_type": "hook_deny",
            }, ensure_ascii=False)
        if modified_args is not None:
            function_args = modified_args

    # 交给中央登记处去找到对应工具并执行（环境信息一并捎过去）
    # dispatch 本身是 async 的，所以要 await
    result = await registry.dispatch(
        function_name,
        function_args,
        task_id=task_id,
        session_id=session_id,
        memory_store=memory_store,
        session_store=session_store,
        codeagent_home=codeagent_home,
        tool_call_id=tool_call_id,
        config=config,
        bg_manager=bg_manager,
        team_bus=team_bus,
        team_coordinator=team_coordinator,
        team_name=team_name,
        agent_ref=agent_ref,
        hooks_registry=hooks_registry,    # 接着往下传给 task_tools 等下游工具用
    )

    # 执行后的钩子
    if hooks_registry and hooks_enabled:
        # 同执行前钩子：移出事件循环线程（钩子链可能一个个等子进程）
        result = await asyncio.to_thread(
            hooks_registry.run_post_tool_use,
            function_name, function_args, result,
            session_id=session_id or "",
        )

    # 工具失败的钩子
    # 执行后钩子跑完，再看结果里带没带 error/error_type——带了就触发
    # "失败审计"钩子。原则：审计钩子自己出事绝不影响主流程（fail-open）。
    if hooks_registry and hooks_enabled:
        try:
            parsed = json.loads(result) if isinstance(result, str) else None
            if isinstance(parsed, dict) and ("error" in parsed or "error_type" in parsed):
                hooks_registry.run_post_tool_use_failure({
                    "session_id": session_id or "",
                    "tool": function_name,
                    "error": parsed.get("error", ""),
                    "error_type": parsed.get("error_type", ""),
                })  # 这种审计钩子很轻、不等子进程，直接调就行
        except Exception:
            pass  # fail-open：审计失败不声张、不挡路

    # 空结果保护
    # 为什么要多此一举：工具啥都没返回时（空串/纯空白/空 dict {}），
    # 部分 API 会把空工具结果当成"流出了异常"，模型也可能误判这轮没跑完。
    # 所以换成一句显式的话"该工具跑完了但没有输出"。
    # {"success": true} 这种非空返回不受影响。
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped or stripped == "{}":
            result = json.dumps({
                "content": f"({function_name} completed with no output)",
                "empty_output": True,
            }, ensure_ascii=False)

    # === 输出统一封顶：单条工具结果超阈值就落盘留预览+指针 ===
    # 各工具自觉调 finalize_tool_output 的老路子漏了一大片（search_files/
    # glob/memory_recall/memory list/MCP 全没接），这里在总出口收口——
    # 任何工具（含未来新工具）的大输出都自动治理，类别化堵死「一条结果
    # 撑爆上下文」。幂等：结果已是 offload 占位（含 full_at+truncated）
    # 就跳过，不会双重落盘。fail-open：封顶失败不挡结果返回。
    if isinstance(result, str) and tool_call_id and codeagent_home:
        try:
            # 幂等判定走真解析：结果确实是 offload 占位（truncated 为真 +
            # 带 full_at）才跳过。旧版子串匹配会被「正文里恰好含这两个
            # 字面量」的大结果（如 read_file 读会话 JSONL）误跳过——
            # 那等于回到无封顶的直通行为。
            _already = False
            try:
                _parsed = json.loads(result)
                _already = (isinstance(_parsed, dict)
                            and _parsed.get("truncated") is True
                            and "full_at" in _parsed)
            except ValueError:  # JSONDecodeError 本就是 ValueError 子类
                _already = False
            if not _already:
                from agent.output_offload import finalize_tool_output
                result = finalize_tool_output(
                    result, tool_call_id, codeagent_home, config,
                )
        except Exception as e:
            logger.warning("工具输出统一封顶失败（fail-open）: %s", e)

    return result


def get_last_resolved_tool_names() -> List[str]:
    """看一眼最近一次发给 LLM 的工具名清单（get_tool_definitions 的筛选结果
    暂存在模块变量里，这里取出来给调试或 UI 展示用）。

    返回：工具名列表的副本（改它不影响内部状态）。
    """
    return list(_last_resolved_tool_names)
