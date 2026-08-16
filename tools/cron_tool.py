"""cron 工具（CCAR12 Task 3）：LLM 可自主创建/列出/删除定时任务。

包装既有 CronScheduler（agent/cron.py），调度本身（后台线程 tick + drain_due
注入主循环）不变——这里只补上此前缺失的 LLM 操作入口（之前 jobs.json 只能手编）。

接线：scheduler 从 dispatch_kwargs["agent_ref"].cron_scheduler 拿。
agent/__init__.py 构造时已挂 self.cron_scheduler（cli.py 注入），无需新接线；
为 None（config cron.enabled=False 或启动失败）时返回 not_configured。

三工具全 isConcurrencySafe=False（写 jobs.json + 影响调度行为，串行），
且 cron_create/cron_delete 进 ASYNC_AGENT_DISALLOWED_TOOLS（后台子代理不应
注册/删定时任务——对齐该集合既有注释意图）。
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
                "description": "True 时错过的一次触发会在启动时补跑（默认 False）",
            },
            "recurring": {
                "type": "boolean",
                "description": "是否循环任务（默认 true）。false=一次性，触发后自动 disable。",
            },
            "template": {
                "type": "string",
                "description": (
                    "任务模板名（~/.OmniMate/templates/*.md 或项目 .omnimate/templates/*.md）。"
                    "给定后 cron/message/catch_up 用模板值；显式传的 cron/message 参数优先。"
                ),
            },
        },
        # template 模式下 cron/message 可省略（取模板值），不再硬性 required；
        # 全缺时由 handler 兜底校验（invalid_args）。
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
    """从 dispatch 上下文取 cron_scheduler。拿不到返回 None（不抛）。"""
    agent_ref = dispatch_kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    return getattr(agent_ref, "cron_scheduler", None)


def _not_configured(tool_name: str) -> str:
    return json.dumps(
        {
            "error": "cron 调度器未配置（config cron.enabled=False 或启动失败）",
            "error_type": "not_configured",
            "tool": tool_name,
        },
        ensure_ascii=False,
    )


def _handle_cron_create(args: dict, **dispatch_kwargs) -> str:
    """创建定时任务。

    流程：参数校验 → cron 表达式校验（cron_match 不抛才算合法）→
    scheduler.add_job → 返回 job_id。
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
        # 显式参数优先模板值
        cron = str(args.get("cron") or "").strip() or tpl["cron"]
        message = str(args.get("message") or "").strip() or tpl["message"]
        catch_up = bool(args.get("catch_up", tpl["catch_up"]))
        # R26 #18 review：模板 recurring 透传（false=一次性），显式参数覆盖模板值
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

    # 表达式校验：cron_match 非法时抛 ValueError（5 字段/范围/语法）
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
        # add_job 的 job_id 冲突等（当前工具层不传 id，防御性兜底）
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
    """列出所有定时任务。"""
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
    """按 job_id 删除定时任务。"""
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


# 模块级注册（import 时自动执行）
registry.register(
    name="cron_create",
    toolset="core",
    schema=CRON_CREATE_SCHEMA,
    handler=_handle_cron_create,
    emoji="⏰",
    isConcurrencySafe=False,  # 写 jobs.json + 影响调度行为
)
registry.register(
    name="cron_list",
    toolset="core",
    schema=CRON_LIST_SCHEMA,
    handler=_handle_cron_list,
    emoji="⏰",
    isConcurrencySafe=False,  # 按 brief 归 UNSAFE（与 create/delete 同组管理）
)
registry.register(
    name="cron_delete",
    toolset="core",
    schema=CRON_DELETE_SCHEMA,
    handler=_handle_cron_delete,
    emoji="⏰",
    isConcurrencySafe=False,  # 写 jobs.json + 影响调度行为
)
