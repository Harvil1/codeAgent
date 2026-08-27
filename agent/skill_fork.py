"""技能隔离执行（frontmatter 里写 context: fork 的技能走这条路）。

这类技能不在主对话里跑，而是临时开一个「分身」（独立子代理实例）去干，
干完把结果带回。好处是技能跑得再乱也不污染主 agent 的对话历史和状态。

主 agent 同步等：期间什么都不干，等子代理跑完，把结果作为一条新的
user 消息注入主循环继续对话。
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
    """在隔离子代理里跑一个 context: fork 技能。

    做法：现造一个独立 AIAgent 实例——技能正文当它的 system prompt
    （给它的角色说明书），用户问题当它的任务。主 agent 停下来等它干完。

    参数：
        skill_name：技能名（用在提示词和审计日志里）。
        skill_body：技能正文文本。
        user_query：用户的原始问题。
        agent_ref：主 agent 实例（fork 它的 LLM 配置等，见下）。
        **kwargs：兼容调用方多传的参数，本函数不用。

    返回：子代理的最终回复文本（字符串）。
    fail-open：子代理构造或运行失败时返回错误提示字符串，不抛异常。
    """
    try:
        # 子代理直接抄主 agent 的 LLM 连接配置；工具只给最小集（leaf 角色，
        # 不再派生下级，防递归开分身）
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
        # fork 子代理不走 delegate_tool 的 _run_child 通道，
        # 那边会自动发的 subagent 审计事件这里得手动补发，否则 hook 看不见这次派生
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
            # AIAgent.chat 是 async，而本函数是
            # 同步的、从 cli.py 直接调（那里没有事件循环）——必须用
            # asyncio.run 驱动，否则直接拿到协程对象而不是结果，等于白跑。
            import asyncio
            result = asyncio.run(child.chat(user_query))
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
