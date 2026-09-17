"""委托配置纯函数三件套：从 delegate_tool 拆出（纯搬迁）。

本模块收拢三个零状态纯函数，全部从 tools/delegate_tool.py 逐字节平移：

  - inline_mcp_spawn_allowed   spawn 时的项目来源内联 MCP 首连审批门（fail-closed）
  - _validate_toolset_names    套餐名清单校验（spawn 时把关 enabled_toolsets）
  - _delegate_schema_overrides 按运行时并发槽位给 subagent 工具说明追加余量提示

搬迁铁律（委托链拆分约定）：
  - 行为零变化——函数体与 delegate_tool 原文逐字节一致；
  - 函数内延迟 import（agent.settings / agent.workspace_context /
    toolsets）原样保留，不提升到模块顶层；
  - 模块顶层不调 registry.register()——这里是函数库，不是注册工具；
  - delegate_tool 侧同名模块级 import——文件内调用点解析零改动，
    同时顺带构成 re-export（tools.delegate_tool.X 符号面保持可用）。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def inline_mcp_spawn_allowed(agent_def, server_name: str, server_cfg: dict) -> bool:
    """spawn 子代理时，检查「项目来源」的内联 MCP 服务器有没有获得首次连接审批。

    子代理定义（.md 文件）里可以内嵌 MCP 外部工具服务器。来自项目目录
    （source="project"，即从 <cwd>/.codeAgent/agents/ 扫出来的）的定义算不可信
    来源——必须用户批准过首次连接才允许连，否则跳过（fail-closed，出错宁可
    不放行）。用户级/CLI 注入的定义默认按 user 信任源处理，不拦。

    参数：
      - agent_def：子代理定义对象（agent/agent_defs.py 的 AgentDefinition）
      - server_name：要连的 MCP 服务器名字
      - server_cfg：该服务器的配置 dict（命令、参数等）

    返回：True=允许连；False=没获批/校验出错，调用方应跳过连接并告警。
    审批 key 的算法与 tools/mcp_tool.py 启动审批处同源：
    工作目录 resolve().lower() + agent-mcp::<服务器名> + 配置指纹。
    """

    if getattr(agent_def, "source", "user") != "project":
        return True
    try:
        from agent.settings import is_project_mcp_approved, mcp_approval_key
        from agent.workspace_context import get_workspace_cwd
        _pk = str(Path(get_workspace_cwd()).resolve()).lower()
        return is_project_mcp_approved(
            mcp_approval_key(_pk, f"agent-mcp::{server_name}", server_cfg))
    except Exception as e:
        logger.warning("内联 MCP 审批校验异常（fail-closed 拒绝）: %s", e)
        return False


def _validate_toolset_names(names):
    """校验套餐名清单：有未知名就返回错误消息（含合法值清单），全合法返回 None。

    为什么在 spawn 时再校验一遍：agent .md 在扫描时校验过，但 kwargs
    传进来的 enabled_toolsets（用户/技能注入）没有别的把关点。
    """
    from toolsets import TOOLSETS
    bad = [t for t in (names or []) if t not in TOOLSETS]
    if not bad:
        return None
    return (
        f"未知工具集名: {bad}（合法值: {sorted(TOOLSETS)}）；"
        "请检查 tools/enabled_toolsets 拼写"
    )


def _delegate_schema_overrides(schema: dict, runtime_ctx: dict) -> dict:
    """按运行时状态给 subagent 工具的说明文字追加「还剩几个坑位」。

    并发子代理有上限，把实时槽位数写进工具描述，让 LLM 提前知道，
    避免白派一次被拒、浪费一整轮。

    参数：
      - schema：原工具 schema
      - runtime_ctx：运行时上下文（从中取 agent 引用）

    返回：追加了运行时状态行的新 schema（原 schema 不动）。
    """
    agent = runtime_ctx.get("agent") if runtime_ctx else None
    if agent is None:
        return schema

    active = len(getattr(agent, "_children", []) or [])
    cfg = getattr(agent, "config", None) or {}
    max_children = (
        (cfg.get("delegation") or {}).get("max_concurrent_children", 5)
        if isinstance(cfg, dict) else 5
    )
    remaining = max(0, max_children - active)

    new_schema = dict(schema)
    desc = new_schema.get("description", "")
    status_line = (
        f"\n\n[运行时状态] 当前活跃子代理: {active}/{max_children}，"
        f"剩余可委派: {remaining}"
    )
    if remaining == 0:
        status_line += "\n⚠️ 已达并发上限，再委派会被拒绝。"
    new_schema["description"] = desc + status_line
    return new_schema
