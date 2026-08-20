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

# 标记工具发现这件事是否已经做过（做过就不重复做）
_tools_discovered = False


def ensure_tools_discovered():
    """确保所有工具模块已被加载。

    背景：每个工具文件在被 import 的那一刻会自动把自己登记进 registry
    （"自注册"机制），但 import 不会自动发生，需要有人推一把。
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
    """拿到本轮要发给 LLM 的"工具说明书"列表。

    背景：不是把所有工具一股脑全发给 LLM——每份说明书都要花 token，
    所以要按场景筛选出真正该可见的那批。

    流程（大白话）：
    1. 先确保工具都登记好了（推一把 import）
    2. 把启用（enable）的工具集展开成一份工具名清单
    3. 划掉明确禁用的工具
    4. 去登记处取每份说明书；check_fn（工具自带的"我现在适不适合出场"检查）
       不通过的工具会被自动略过——这就实现了"登记了但当前不可见"

    参数：
        enabled_toolsets: 启用哪些工具集（如 ["core", "mcp"]）。
        disabled_tools: 要明确划掉的工具名清单。
        agent: 当前 AIAgent 实例。传了它，工具的"动态改写说明书"回调
            （schema_overrides_fn）就能拿到它——比如让说明书里的
            并发剩余槽位数反映当下真实情况。

    返回：schema 字典列表，直接可以塞进 LLM 请求。
    """
    ensure_tools_discovered()

    # 把每个启用的工具集展开成具体工具名
    tool_names: List[str] = []
    for ts in enabled_toolsets:
        tool_names.extend(resolve_toolset(ts))

    # mcp 工具集特殊：它的工具是运行时连上外部服务器才动态登记的，
    # 名字都以 mcp__ 开头，这里现场扫一遍登记处把它们捞出来
    if "mcp" in enabled_toolsets:
        # 历史功能（Task C1）：子代理可以通过 config["mcp_server_filter"]
        # 圈定"只看得见哪几个 MCP 服务器"，防止它乱碰别的服务器
        mcp_filter = (agent.config.get("mcp_server_filter")
                      if agent and isinstance(getattr(agent, "config", None), dict)
                      else None)
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
    scope = getattr(agent, "_skill_tool_scope", None) if agent else None
    if scope:
        allowed_tools, disallowed_tools_scope = scope
        if allowed_tools:
            allow_set = set(allowed_tools)
            tool_names = [n for n in tool_names if n in allow_set]
        if disallowed_tools_scope:
            dis_scope = set(disabled_tools or []) | set(disallowed_tools_scope)
            tool_names = [n for n in tool_names if n not in dis_scope]

    # 核心机制对齐第 6 项（T6）：settings.json 里 permissions.deny 配的禁用规则，
    # 要在 LLM 看到之前就把工具整类拿掉（支持精确名 / mcp__server__* 通配 /
    # mcp__server 整个服务器）。
    # 历史踩坑（R30c-B7 修复）：这里选择"坏了也放行"（fail-open）——如果配置
    # 文件损坏就直接全拒，agent 会整个被砖死；但以前出错时静默吞掉不打日志，
    # 这道安全防线等于无声消失，所以现在必须显式打 ERROR。
    try:
        from agent.tool_permissions import is_tool_denied
        tool_names = [n for n in tool_names if not is_tool_denied(n)]
    except Exception as e:
        logger.error("deny 规则加载失败，工具可见性过滤本调用失效（fail-open）: %s", e)

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
    omnimate_home=None,
    tool_call_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    hooks_registry=None,  # P2-T7 新增
    bg_manager=None,      # P2b-T7 新增
    team_bus=None,               # P4a-T6 新增
    team_coordinator=None,       # P4a-T6 新增
    team_name=None,              # P4a-T6 新增
    agent_ref=None,              # P4b-T2 新增
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
        omnimate_home: agent 的数据根目录（默认 ~/.OmniMate）。
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

    # 先修一遍参数类型（LLM 偶尔会把整数传成字符串之类的低级错误）
    function_args = _coerce_tool_args(function_name, function_args)

    # 执行前的钩子（P2-T7 引入）
    hooks_enabled = (config or {}).get("hooks", {}).get("enabled", True)
    if hooks_registry and hooks_enabled:
        # 历史踩坑（R30 审计 Medium-6 修复）：钩子链可能要等子进程跑完，
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
        omnimate_home=omnimate_home,
        tool_call_id=tool_call_id,
        config=config,
        bg_manager=bg_manager,  # P2b-T7 新增
        team_bus=team_bus,               # P4a-T6 新增
        team_coordinator=team_coordinator,   # P4a-T6 新增
        team_name=team_name,             # P4a-T6 新增
        agent_ref=agent_ref,             # P4b-T2 新增
        hooks_registry=hooks_registry,    # round3 D2 新增：接着往下传给 task_tools 等下游工具用
    )

    # 执行后的钩子（P2-T7 引入）
    if hooks_registry and hooks_enabled:
        # 同执行前钩子：移出事件循环线程（钩子链可能一个个等子进程）
        result = await asyncio.to_thread(
            hooks_registry.run_post_tool_use,
            function_name, function_args, result,
            session_id=session_id or "",
        )

    # 工具失败的钩子（round3 D2 引入）
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
                })  # 这种审计钩子很轻、不等子进程，直接调就行（Medium-6 的例外）
        except Exception:
            pass  # fail-open：审计失败不声张、不挡路

    # 空结果保护（R20 第 33 项，对齐 Claude Code 的 toolResultStorage）
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

    return result


def _coerce_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """修正 LLM 传参数时的常见类型错误。

    背景：LLM 偶尔会把整数写成字符串、把列表写成单个值，这类小错
    在这里统一矫正。设计上每个工具可以登记自己的矫正规则，但当前
    是简化版。

    参数：
        name: 工具名（用来查该工具自己的矫正规则）。
        args: LLM 传来的原始参数。

    返回：矫正后的参数字典（当前版本原样返回）。
    """
    # 简化版：暂不做任何矫正
    return args


def get_last_resolved_tool_names() -> List[str]:
    """看一眼最近一次发给 LLM 的工具名清单。

    背景：get_tool_definitions 会把筛选结果暂存在模块变量里，
    这里取出来给调试或 UI 展示用。

    返回：工具名列表的副本（改它不影响内部状态）。
    """
    return list(_last_resolved_tool_names)
