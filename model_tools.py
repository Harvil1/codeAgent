"""工具分发层。

连接 agent 和 registry：
- agent 调用 get_tool_definitions() 获取要发给 LLM 的 schema
- agent 调用 handle_function_call() 执行 LLM 返回的工具调用
"""

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, discover_builtin_tools
from toolsets import resolve_toolset

logger = logging.getLogger(__name__)

# 模块级状态：记录最近解析的工具名（用于调试）
_last_resolved_tool_names: List[str] = []

# 标记是否已触发工具发现
_tools_discovered = False


def ensure_tools_discovered():
    """确保工具模块已被 import（触发自注册）。幂等。"""
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
    """获取要发给 LLM 的工具 schema 列表。

    流程：
    1. 确保工具已发现（import tools/*.py）
    2. 解析启用的工具集，得到工具名列表
    3. 减去显式禁用的工具
    4. 从 registry 获取定义（自动过滤 check_fn 不通过的）

    参数：
        agent: 当前 AIAgent 实例。传入时 schema_overrides_fn 会拿到它
            （让工具 schema 反映运行时状态，如剩余并发槽位）。
    """
    ensure_tools_discovered()

    # 解析启用的工具
    tool_names: List[str] = []
    for ts in enabled_toolsets:
        tool_names.extend(resolve_toolset(ts))

    # 如果启用 mcp toolset，动态发现所有 mcp__ 前缀工具
    if "mcp" in enabled_toolsets:
        for name in registry.list_all():
            if name.startswith("mcp__") and name not in tool_names:
                tool_names.append(name)

    # 去重（保序）
    tool_names = list(dict.fromkeys(tool_names))

    # 减去禁用的
    if disabled_tools:
        disabled_set = set(disabled_tools)
        tool_names = [n for n in tool_names if n not in disabled_set]

    global _last_resolved_tool_names
    _last_resolved_tool_names = tool_names

    # 从 registry 获取（check_fn 过滤）
    runtime_ctx = {"agent": agent} if agent is not None else None
    return registry.get_definitions(tool_names, quiet=True, runtime_ctx=runtime_ctx)


def handle_function_call(
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
    hooks_registry=None,  # === P2-T7 NEW ===
    bg_manager=None,      # === P2b-T7 NEW ===
    team_bus=None,               # === P4a-T6 NEW ===
    team_coordinator=None,       # === P4a-T6 NEW ===
    team_name=None,              # === P4a-T6 NEW ===
    agent_ref=None,              # === P4b-T2 NEW ===
) -> str:
    """分发工具调用，返回 JSON 字符串结果。

    这是 agent 调用工具的入口。
    context 参数会被透传给工具 handler（按需取用）。

    PRE_TOOL_USE hook 在 dispatch 前执行：deny 短路返回 hook_deny，
    modify_args 替换入参。
    POST_TOOL_USE hook 在 dispatch 后执行：可改写 result 字符串。
    hooks_registry=None 或 config.hooks.enabled=False 时跳过所有 hook
    （完全向后兼容）。
    """
    ensure_tools_discovered()

    # 参数类型强制转换（LLM 有时会传错类型）
    function_args = _coerce_tool_args(function_name, function_args)

    # === P2-T7 NEW: PRE_TOOL_USE hook ===
    hooks_enabled = (config or {}).get("hooks", {}).get("enabled", True)
    if hooks_registry and hooks_enabled:
        deny_reason, modified_args = hooks_registry.run_pre_tool_use(
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

    # 分发到 registry（传递上下文给 handler）
    result = registry.dispatch(
        function_name,
        function_args,
        task_id=task_id,
        session_id=session_id,
        memory_store=memory_store,
        session_store=session_store,
        omnimate_home=omnimate_home,
        tool_call_id=tool_call_id,
        config=config,
        bg_manager=bg_manager,  # === P2b-T7 NEW ===
        team_bus=team_bus,               # === P4a-T6 NEW ===
        team_coordinator=team_coordinator,   # === P4a-T6 NEW ===
        team_name=team_name,             # === P4a-T6 NEW ===
        agent_ref=agent_ref,             # === P4b-T2 NEW ===
    )

    # === P2-T7 NEW: POST_TOOL_USE hook ===
    if hooks_registry and hooks_enabled:
        result = hooks_registry.run_post_tool_use(
            function_name, function_args, result,
            session_id=session_id or "",
        )

    return result


def _coerce_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """修正 LLM 传参的常见类型错误。

    例如：LLM 把整数传成字符串、把 list 传成单值等。
    完整实现中每个工具可注册自己的 coerce 函数。
    """
    # 简化版：直接返回
    return args


def get_last_resolved_tool_names() -> List[str]:
    """返回最近一次解析的工具名列表（用于调试/UI）。"""
    return list(_last_resolved_tool_names)
