"""goal 工具（CCAR12 Task 4）：LLM 可自主启动/查看/暂停/恢复/取消 goal。

5 个工具：
- goal_start: 启动新 goal（走 agent/goal.start_goal_agent 共享函数，与 CLI /goal 同源）
- goal_status: 只读查状态（无 goal 时 status=none）
- goal_pause / goal_resume / goal_clear: 操作 agent._goal_state

接线：agent 从 dispatch_kwargs["agent_ref"] 拿（CCAR8 mailbox 模式）。
goal 激活后主循环的 goal-continue 分支自动多轮推进（无需额外接线）。

并发分类：
- goal_status: isConcurrencySafe=True（只读）
- 其余 4 个: isConcurrencySafe=False（改 goal 状态机 + 落盘，串行）
"""
import json
import logging
from pathlib import Path

from agent.goal import goal_persist_path, start_goal_agent
from tools.registry import registry

logger = logging.getLogger(__name__)


GOAL_START_SCHEMA = {
    "name": "goal_start",
    "description": (
        "启动一个目标驱动的长任务：goal 激活后主循环自动多轮推进，"
        "直到目标完成、token 预算耗尽或被 pause/clear。"
        "如已有 active goal 会被自动 pause（superseded_by_new_goal）。"
        "objective 要具体、可判断完成（如'让全部测试通过'而不是'搞好代码'）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "目标描述（具体、可判断完成）",
            },
            "token_budget": {
                "type": "integer",
                "default": 200000,
                "description": "token 预算上限（默认 20 万，超过自动 pause）",
            },
        },
        "required": ["objective"],
    },
}

GOAL_STATUS_SCHEMA = {
    "name": "goal_status",
    "description": (
        "查看当前 goal 状态（objective/status/iteration_count/"
        "token 预算/pause_reason）。无 goal 时 status=none。"
    ),
    "parameters": {"type": "object", "properties": {}},
}

GOAL_PAUSE_SCHEMA = {
    "name": "goal_pause",
    "description": "暂停当前 goal（可带 reason，默认 manual）。",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "default": "manual",
                "description": "暂停原因（如 manual / 等用户确认）",
            },
        },
    },
}

GOAL_RESUME_SCHEMA = {
    "name": "goal_resume",
    "description": "恢复已暂停的 goal（清除 pause_reason，回到 active）。",
    "parameters": {"type": "object", "properties": {}},
}

GOAL_CLEAR_SCHEMA = {
    "name": "goal_clear",
    "description": "取消当前 goal（标记 cancelled + 删持久化文件 + agent 摘挂）。",
    "parameters": {"type": "object", "properties": {}},
}


def _get_agent(dispatch_kwargs: dict):
    """从 dispatch 上下文取 agent_ref。拿不到返回 None（不抛）。"""
    return dispatch_kwargs.get("agent_ref")


def _no_agent(tool_name: str) -> str:
    return json.dumps(
        {
            "error": "dispatch 上下文无 agent_ref",
            "error_type": "not_configured",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _no_goal(tool_name: str) -> str:
    return json.dumps(
        {
            "error": "无 active goal（先 goal_start）",
            "error_type": "no_active_goal",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _handle_goal_start(args: dict, **dispatch_kwargs) -> str:
    """启动新 goal（共享函数与 CLI /goal 同源）。

    注意：不追加 [goal_start] user 消息到 conversation_history——
    工具路径在 tool result 回填前插 user 消息会破坏消息历史严格交替；
    goal 激活后由主循环 goal-continue 分支自然驱动。
    """
    objective = (args.get("objective") or "").strip()
    if not objective:
        return json.dumps(
            {"error": "objective 必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    token_budget = args.get("token_budget", 200_000)
    if not isinstance(token_budget, int) or token_budget <= 0:
        return json.dumps(
            {
                "error": "token_budget 必须是正整数",
                "error_type": "invalid_args",
            },
            ensure_ascii=False,
        )

    agent = _get_agent(dispatch_kwargs)
    if agent is None:
        return _no_agent("goal_start")

    try:
        gs = start_goal_agent(agent, objective, token_budget=token_budget)
        return json.dumps(
            {
                "started": True,
                "goal_id": gs.goal_id,
                "objective": gs.objective,
                "status": gs.status,
                "token_budget_limit": gs.token_budget_limit,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("goal_start 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_goal_status(args: dict, **dispatch_kwargs) -> str:
    """只读查 goal 状态。"""
    agent = _get_agent(dispatch_kwargs)
    if agent is None:
        return _no_agent("goal_status")

    gs = getattr(agent, "_goal_state", None)
    if gs is None:
        return json.dumps({"status": "none"}, ensure_ascii=False)
    return json.dumps(
        {
            "objective": gs.objective,
            "goal_id": gs.goal_id,
            "status": gs.status,
            "iteration_count": gs.iteration_count,
            "token_budget": gs.token_budget,
            "token_budget_limit": gs.token_budget_limit,
            "pause_reason": gs.pause_reason,
            "task_count": len(gs.task_ids),
        },
        ensure_ascii=False,
    )


def _handle_goal_pause(args: dict, **dispatch_kwargs) -> str:
    """暂停当前 goal（默认 reason=manual）。"""
    agent = _get_agent(dispatch_kwargs)
    if agent is None:
        return _no_agent("goal_pause")

    gs = getattr(agent, "_goal_state", None)
    if gs is None:
        return _no_goal("goal_pause")

    try:
        reason = (args.get("reason") or "manual").strip() or "manual"
        gs.pause(reason=reason)
        gs.save(goal_persist_path(agent))
        return json.dumps(
            {"status": gs.status, "pause_reason": gs.pause_reason},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("goal_pause 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_goal_resume(args: dict, **dispatch_kwargs) -> str:
    """恢复已暂停的 goal。"""
    agent = _get_agent(dispatch_kwargs)
    if agent is None:
        return _no_agent("goal_resume")

    gs = getattr(agent, "_goal_state", None)
    if gs is None:
        return _no_goal("goal_resume")

    try:
        gs.resume()
        gs.save(goal_persist_path(agent))
        return json.dumps(
            {"status": gs.status, "pause_reason": gs.pause_reason},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("goal_resume 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_goal_clear(args: dict, **dispatch_kwargs) -> str:
    """取消当前 goal（cancelled + 删持久化文件 + agent 摘挂）。"""
    agent = _get_agent(dispatch_kwargs)
    if agent is None:
        return _no_agent("goal_clear")

    gs = getattr(agent, "_goal_state", None)
    if gs is None:
        return _no_goal("goal_clear")

    try:
        gs.cancel()
        gs.save(goal_persist_path(agent))
        setter = getattr(agent, "set_goal_state", None)
        if callable(setter):
            setter(None)
        else:
            agent._goal_state = None
        # 删持久化文件（对齐 CLI /goal clear）
        try:
            path = Path(goal_persist_path(agent))
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning("goal 持久化文件删除失败（忽略）: %s", e)
        return json.dumps(
            {"cleared": True, "goal_id": gs.goal_id}, ensure_ascii=False
        )
    except Exception as e:
        logger.warning("goal_clear 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动执行）
registry.register(
    name="goal_start",
    toolset="core",
    schema=GOAL_START_SCHEMA,
    handler=_handle_goal_start,
    emoji="🎯",
    isConcurrencySafe=False,  # 改 goal 状态机 + 落盘
)
registry.register(
    name="goal_status",
    toolset="core",
    schema=GOAL_STATUS_SCHEMA,
    handler=_handle_goal_status,
    emoji="🎯",
    isConcurrencySafe=True,  # 只读
)
registry.register(
    name="goal_pause",
    toolset="core",
    schema=GOAL_PAUSE_SCHEMA,
    handler=_handle_goal_pause,
    emoji="🎯",
    isConcurrencySafe=False,  # 改 goal 状态机 + 落盘
)
registry.register(
    name="goal_resume",
    toolset="core",
    schema=GOAL_RESUME_SCHEMA,
    handler=_handle_goal_resume,
    emoji="🎯",
    isConcurrencySafe=False,  # 改 goal 状态机 + 落盘
)
registry.register(
    name="goal_clear",
    toolset="core",
    schema=GOAL_CLEAR_SCHEMA,
    handler=_handle_goal_clear,
    emoji="🎯",
    isConcurrencySafe=False,  # 改状态 + 删持久化文件 + agent 摘挂
)
