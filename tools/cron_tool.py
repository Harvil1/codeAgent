"""定时任务（cron）工具：让 AI 能自己创建/查看/删除定时任务。

打个比方：这是给 AI 一个「闹钟遥控器」。闹钟本体（CronScheduler，在
agent/cron.py 里，靠后台线程每秒看一眼到没到点，到点就把提醒塞进主对话）；
本文件给 LLM（AI 模型）提供工具调用方式的管理入口，
不必手工编辑 jobs.json。

怎么拿到闹钟本体：工具被调用时，框架会把 agent 实例放在
dispatch_kwargs["agent_ref"] 里，从它的 cron_scheduler 字段取调度器
（agent/__init__.py 构造时已挂好，cli.py 注入，这里不用再接线）。
如果取出来是 None（配置里 cron.enabled=False，或者调度器启动失败），
工具统一返回 not_configured 错误。

并发分类（决定工具能不能同时跑）：三个工具都标了
isConcurrencySafe=False（要写 jobs.json 文件 + 会改变调度行为，
同时跑会互相踩，必须排队串行）。另外 cron_create/cron_delete 列入了
ASYNC_AGENT_DISALLOWED_TOOLS（后台子代理——主对话派出去的分身——
不该偷偷注册或删定时任务）。
"""
import json
import logging
from datetime import datetime

from agent.cron_parser import cron_match
from tools.registry import registry

logger = logging.getLogger(__name__)


CRON_CREATE_SCHEMA = {
    "name": "cron_create",
    "description": (
        "创建定时任务（5 字段 cron 表达式：分 时 日 月 周）。"
        "到点后消息会作为通知注入对话。示例：'*/5 * * * *' 每 5 分钟、"
        "'0 9 * * 1-5' 工作日 9 点、'30 14 28 2 *' 每年 2 月 28 日 14:30。"
        "注意：任务默认 7 天后自动过期（max_age_days）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cron": {
                "type": "string",
                "description": "5 字段 cron 表达式，如 '*/5 * * * *'",
            },
            "message": {
                "type": "string",
                "description": "到期时要提醒/执行的内容（会注入对话）",
            },
            "catch_up": {
                "type": "boolean",
                "default": False,
                "description": (
                    "True 时启动会补跑错过的触发——每次启动每个任务只补 1 次"
                    "（停机错过多次也只补最近一次，通知标 catch_up=True 与 missed_between 窗口）"
                ),
            },
            "recurring": {
                "type": "boolean",
                "description": "是否循环任务（默认 true）。false=一次性，触发后自动 disable。",
            },
            "template": {
                "type": "string",
                "description": (
                    "任务模板名（~/.codeAgent/templates/*.md 或项目 .codeAgent/templates/*.md）。"
                    "给定后 cron/message/catch_up 用模板值；显式传的 cron/message/catch_up/recurring 参数优先。"
                ),
            },
        },
        # 用模板时 cron/message 可以不传（从模板取值），所以不写死在 required 里；
        # 两个都没给时由 handler 里再兜底校验（返回 invalid_args 错误）。
        "required": [],
    },
}

CRON_LIST_SCHEMA = {
    "name": "cron_list",
    "description": "列出所有定时任务（id、cron 表达式、消息、启用状态）。",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

CRON_DELETE_SCHEMA = {
    "name": "cron_delete",
    "description": "按 job_id 删除定时任务。",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "cron_list 返回的任务 id",
            },
        },
        "required": ["job_id"],
    },
}


def _get_scheduler(dispatch_kwargs: dict):
    """从工具调用的上下文里把 cron 调度器（闹钟本体）取出来。

    handler 被调用时框架传来的 dispatch_kwargs 里带着 agent 实例，
    调度器就挂在它的 cron_scheduler 字段上。

    参数：
    - dispatch_kwargs：框架透传的上下文字典，里面有 "agent_ref"。

    返回：CronScheduler 实例；取不到就返回 None（不抛异常，让调用方
    自己决定怎么提示用户）。
    """
    agent_ref = dispatch_kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    return getattr(agent_ref, "cron_scheduler", None)


def _not_configured(tool_name: str) -> str:
    """拼一个「调度器没配置」的标准错误 JSON。

    参数：
    - tool_name：当前工具名（如 cron_create），放进错误里方便排查。

    返回：JSON 字符串（error_type=not_configured）。
    """
    return json.dumps(
        {
            "error": "cron 调度器未配置（config cron.enabled=False 或启动失败）",
            "error_type": "not_configured",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _handle_cron_create(args: dict, **dispatch_kwargs) -> str:
    """创建一个定时任务（帮 AI 设闹钟）。

    流程：先看有没有指定模板（模板里带默认的 cron/消息等值）→
    校验 cron 表达式合法性（拿 cron_match 试跑一次，不抛异常才算合法）→
    交给 scheduler.add_job 落盘 → 把 job_id 返回给 AI。

    参数：
    - args：工具参数，可含 cron（5 字段 cron 表达式）、message（到点要
      提醒的内容）、catch_up（是否补跑停机期间错过的触发）、recurring
      （是否循环，false 表示一次性）、template（任务模板名，给了就用
      模板里的默认值）。
    - dispatch_kwargs：框架透传的上下文，用来取 agent_ref 再拿调度器。

    返回：JSON 字符串，成功时含 job_id/cron/message；失败时是
    {"error": ..., "error_type": ...} 格式的错误。
    """
    template_name = str(args.get("template") or "").strip()
    if template_name:
        from agent.templates import load_task_templates
        tpl = load_task_templates().get(template_name)
        if tpl is None:
            return json.dumps({
                "error": f"模板不存在: {template_name}",
                "error_type": "invalid_template",
                "available": sorted(load_task_templates().keys()),
            }, ensure_ascii=False)
        # AI 明确传了的参数压过模板默认值（就高不就低）
        cron = str(args.get("cron") or "").strip() or tpl["cron"]
        message = str(args.get("message") or "").strip() or tpl["message"]
        catch_up = bool(args.get("catch_up", tpl["catch_up"]))
        # recurring 要吃模板默认值（显式参数可覆盖）——漏了模板值的
        # 话，一次性任务会被当成循环任务。
        recurring = bool(args.get("recurring", tpl["recurring"]))
    else:
        cron = (args.get("cron") or "").strip()
        message = (args.get("message") or "").strip()
        catch_up = bool(args.get("catch_up", False))
        recurring = bool(args.get("recurring", True))
    if not cron or not message:
        return json.dumps(
            {"error": "cron 和 message 都是必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )

    # 先验合法性再入库：cron_match 遇到非法表达式（字段数不对/超出范围/
    # 语法错误）会抛 ValueError，这里提前拦住
    try:
        cron_match(cron, datetime.now())
    except ValueError as e:
        return json.dumps(
            {
                "error": f"cron 表达式非法: {e}",
                "error_type": "invalid_cron_expr",
                "cron": cron,
            },
            ensure_ascii=False,
        )

    scheduler = _get_scheduler(dispatch_kwargs)
    if scheduler is None:
        return _not_configured("cron_create")

    try:
        job = scheduler.add_job(cron, message, catch_up=catch_up, recurring=recurring)
        return json.dumps(
            {"job_id": job.id, "cron": job.cron, "message": job.message},
            ensure_ascii=False,
        )
    except ValueError as e:
        # add_job 自己抛的 ValueError（如 job_id 撞号；当前工具层不传 id，
        # 纯属防御性兜底，正常走不到这）
        return json.dumps(
            {"error": str(e), "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("cron_create 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_cron_list(args: dict, **dispatch_kwargs) -> str:
    """列出当前所有的定时任务（把闹钟清单念给 AI 听）。

    参数：
    - args：工具参数（本工具不需要参数）。
    - dispatch_kwargs：框架透传的上下文，用来取调度器。

    返回：JSON 字符串，含 jobs 列表和 count 总数；出错时是错误 JSON。
    """
    scheduler = _get_scheduler(dispatch_kwargs)
    if scheduler is None:
        return _not_configured("cron_list")
    try:
        jobs = scheduler.list_jobs()
        return json.dumps(
            {"jobs": jobs, "count": len(jobs)}, ensure_ascii=False
        )
    except Exception as e:
        logger.warning("cron_list 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


def _handle_cron_delete(args: dict, **dispatch_kwargs) -> str:
    """按任务 ID 删除一个定时任务（取消闹钟）。

    参数：
    - args：工具参数，job_id 必填（cron_list 返回里的那个 id）。
    - dispatch_kwargs：框架透传的上下文，用来取调度器。

    返回：JSON 字符串，成功含 deleted=True；任务不存在时返回
    not_found 错误。
    """
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return json.dumps(
            {"error": "job_id 必填", "error_type": "invalid_args"},
            ensure_ascii=False,
        )

    scheduler = _get_scheduler(dispatch_kwargs)
    if scheduler is None:
        return _not_configured("cron_delete")

    try:
        deleted = scheduler.remove_job(job_id)
        if deleted:
            return json.dumps({"deleted": True, "job_id": job_id}, ensure_ascii=False)
        return json.dumps(
            {
                "error": f"job 不存在: {job_id}",
                "error_type": "not_found",
                "deleted": False,
                "job_id": job_id,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("cron_delete 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册：本文件被 import 的那一刻就把三个工具登记进中央注册表
registry.register(
    name="cron_create",
    toolset="core",
    schema=CRON_CREATE_SCHEMA,
    handler=_handle_cron_create,
    emoji="⏰",
    isConcurrencySafe=False,  # 要写 jobs.json 还会改调度行为，不能并发
)
registry.register(
    name="cron_list",
    toolset="core",
    schema=CRON_LIST_SCHEMA,
    handler=_handle_cron_list,
    emoji="⏰",
    isConcurrencySafe=False,  # 按需求归为不安全（跟 create/delete 一组管理）
)
registry.register(
    name="cron_delete",
    toolset="core",
    schema=CRON_DELETE_SCHEMA,
    handler=_handle_cron_delete,
    emoji="⏰",
    isConcurrencySafe=False,  # 要写 jobs.json 还会改调度行为，不能并发
)
