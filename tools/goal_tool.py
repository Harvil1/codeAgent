"""goal（目标驱动）工具：让 AI 自己启动/查看/暂停/恢复/取消一个长期目标。

打个比方：goal 就像给 AI 立一个「项目目标」，立完之后主对话循环会自动
一轮一轮往前推（干一步、看结果、再干一步），直到目标完成、token 预算
花光、或被人叫停——不用用户每步都催。本文件提供 5 个工具：

- goal_start：立新目标（内部走 agent/goal.py 的 start_goal_agent 共享
  函数，和 CLI 的 /goal 命令是同一份代码，两边行为永远一致）
- goal_status：只读查当前目标状态（没目标时返回 status=none）
- goal_pause / goal_resume / goal_clear：暂停/恢复/取消，操作的是
  agent._goal_state 里那个目标状态对象

怎么拿到 agent：从 dispatch_kwargs["agent_ref"] 取（沿用
mailbox 的接线模式）。目标激活后主循环的 goal-continue 分支会自动推进，
这里不用额外接线。

并发分类：goal_status 标 isConcurrencySafe=True（只读，可并发）；
其余 4 个都标 False（要改目标状态机 + 写盘，必须排队串行）。
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
    """从工具调用的上下文里把 agent 实例取出来。

    参数：
    - dispatch_kwargs：框架透传的上下文字典，里面有 "agent_ref"。

    返回：AIAgent 实例；取不到返回 None（不抛异常，交给调用方处理）。
    """
    return dispatch_kwargs.get("agent_ref")


def _no_agent(tool_name: str) -> str:
    """拼一个「上下文里没有 agent」的标准错误 JSON。

    参数：
    - tool_name：当前工具名，放进错误里方便排查。

    返回：JSON 字符串（error_type=not_configured）。
    """
    return json.dumps(
        {
            "error": "dispatch 上下文无 agent_ref",
            "error_type": "not_configured",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _no_goal(tool_name: str) -> str:
    """拼一个「当前没有活动目标」的标准错误 JSON。

    参数：
    - tool_name：当前工具名，放进错误里方便排查。

    返回：JSON 字符串（error_type=no_active_goal，提示先 goal_start）。
    """
    return json.dumps(
        {
            "error": "无 active goal（先 goal_start）",
            "error_type": "no_active_goal",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _handle_goal_start(args: dict, **dispatch_kwargs) -> str:
    """立一个新的目标（启动 goal）。

    内部调用共享的 start_goal_agent 函数，与 CLI 的 /goal 命令
    同一份代码，保证两条入口行为一致。

    注意：这里故意不往对话历史里追加 [goal_start] 的 user
    消息——工具路径在工具结果还没回填时插 user 消息，会破坏消息历史
    「user/assistant 严格交替」的规矩（API 会报错）。目标激活后主循环
    的 goal-continue 分支自然会接手推进，不需要靠这条消息。

    参数：
    - args：工具参数，objective 必填（目标描述），token_budget 可选
      （token 预算上限，默认 20 万，花超自动暂停）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent 实例。

    返回：JSON 字符串，成功时含 goal_id/objective/status 等；失败时是
    {"error": ..., "error_type": ...} 格式的错误。
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
    """只读查看当前目标的状态（目标/进度/预算/暂停原因等）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent 实例。

    返回：JSON 字符串；没有活动目标时返回 {"status": "none"}，有则返回
    objective/status/iteration_count/token 预算等信息。
    """
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
    """暂停当前目标（把自动推进的脚步停下）。

    参数：
    - args：工具参数，reason 可选（暂停原因，默认 manual=人工叫停）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent 实例。

    返回：JSON 字符串，含暂停后的 status 和 pause_reason；无活动目标时
    返回 no_active_goal 错误。
    """
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
    """恢复已暂停的目标（清掉暂停原因，回到活动状态继续推进）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent 实例。

    返回：JSON 字符串，含恢复后的 status 和 pause_reason（此时为空）；
    无活动目标时返回 no_active_goal 错误。
    """
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
    """彻底取消当前目标（三件事：标记 cancelled + 删持久化文件 + 从
    agent 身上摘掉状态对象）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent 实例。

    返回：JSON 字符串，成功含 cleared=True 和 goal_id；无活动目标时
    返回 no_active_goal 错误。
    """
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
        # 删持久化文件（和 CLI /goal clear 的行为保持一致）
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


# 模块级注册：本文件被 import 的那一刻就把五个工具登记进中央注册表
registry.register(
    name="goal_start",
    toolset="core",
    schema=GOAL_START_SCHEMA,
    handler=_handle_goal_start,
    emoji="🎯",
    isConcurrencySafe=False,  # 要改 goal 状态机 + 写盘，不能并发
)
registry.register(
    name="goal_status",
    toolset="core",
    schema=GOAL_STATUS_SCHEMA,
    handler=_handle_goal_status,
    emoji="🎯",
    isConcurrencySafe=True,  # 只读，可并发
)
registry.register(
    name="goal_pause",
    toolset="core",
    schema=GOAL_PAUSE_SCHEMA,
    handler=_handle_goal_pause,
    emoji="🎯",
    isConcurrencySafe=False,  # 要改 goal 状态机 + 写盘，不能并发
)
registry.register(
    name="goal_resume",
    toolset="core",
    schema=GOAL_RESUME_SCHEMA,
    handler=_handle_goal_resume,
    emoji="🎯",
    isConcurrencySafe=False,  # 要改 goal 状态机 + 写盘，不能并发
)
registry.register(
    name="goal_clear",
    toolset="core",
    schema=GOAL_CLEAR_SCHEMA,
    handler=_handle_goal_clear,
    emoji="🎯",
    isConcurrencySafe=False,  # 要改状态 + 删持久化文件 + 摘挂 agent 字段，不能并发
)
