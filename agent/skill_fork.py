"""技能隔离执行（context: fork）。

frontmatter context: fork 的技能在独立子代理上下文跑，不污染主 agent。
同步等待：主 agent 等子代理跑完，结果作为新 user 消息注入主循环。
"""

import logging

from agent import AIAgent

logger = logging.getLogger(__name__)


def run_skill_in_fork(
    skill_name: str,
    skill_body: str,
    user_query: str,
    agent_ref,
    **kwargs,
) -> str:
    """在隔离子代理里跑一个技能（context: fork）。

    构造一个独立 AIAgent 实例，system_prompt = 技能正文，goal = 用户 query。
    同步等子代理完成，返回子代理最终回复（str）。

    fail-open：子代理构造/运行失败时返回错误消息字符串（不抛）。
    """
    try:
        # 继承父 agent 的 LLM 配置 + 工具集（leaf 角色，minimal 工具集）
        _hooks = getattr(agent_ref, "hooks_registry", None)
        child = AIAgent(
            base_url=agent_ref.base_url,
            api_key=agent_ref.api_key,
            auth_token=agent_ref.auth_token,
            model=agent_ref.model,
            model_format=agent_ref.model_format,
            max_iterations=30,
            enabled_toolsets=["minimal"],
            system_prompt_override=(
                f"# 技能：{skill_name}\n\n"
                f"{skill_body}\n\n"
                "你是被主 agent 派来执行上述技能的子代理。按技能指引完成任务，"
                "把最终结果作为回复返回。"
            ),
            spawn_depth=getattr(agent_ref, "spawn_depth", 0) + 1,
            omnimate_home=agent_ref.omnimate_home,
            effort_level=getattr(agent_ref, "effort_level", None),
            hooks_registry=_hooks,
        )
        # round4 NEW: fork 子代理也进 SUBAGENT 审计（不走 _run_child，需手动触发）
        _fork_success = False
        if _hooks is not None:
            try:
                _hooks.run_subagent_start({
                    "subagent": f"skill:{skill_name}", "goal": user_query,
                    "spawn_depth": getattr(agent_ref, "spawn_depth", 0) + 1,
                })
            except Exception:
                pass
        try:
            result = child.chat(user_query)
            _fork_success = True
            return result or "(子代理无输出)"
        finally:
            if _hooks is not None:
                try:
                    _hooks.run_subagent_stop({
                        "subagent": f"skill:{skill_name}", "goal": user_query,
                        "success": _fork_success,
                    })
                except Exception:
                    pass
    except Exception as e:
        logger.warning("run_skill_in_fork 失败: %s", e)
        return f"(技能 {skill_name} 隔离执行失败: {e})"
