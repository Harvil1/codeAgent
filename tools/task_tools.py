"""Task System 工具集：持久化任务管理。

和 todo_write 的区别：
  - todo_write：内存清单，单会话，扁平
  - task_create/update/complete：持久化到 .tasks/，跨会话，DAG 依赖

工具：
  task_create(subject, description, blocked_by)  创建任务
  task_update(id, status, owner, ...)            更新任务
  task_complete(id)                              完成任务
  task_list(status)                              列出任务
"""

import json
from typing import Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry


def _ownership_denied(msg: str) -> str:
    """把 TaskOwnershipError 消息包成 permission_denied JSON 错误。

    msg 来自 assert_owned 抛出的异常，已包含 bound task_id 和 attempted
    task_id（如 "worker bound to task 'task_A', cannot operate on 'task_B'"）。
    """
    return json.dumps({
        "error": msg,
        "error_type": "permission_denied",
    }, ensure_ascii=False)


def _get_owned_task(args: dict, kwargs: dict):
    """过 ownership + 取任务。失败返回 (None, error_json)；成功返回 (task, None)。

    所有 task_* 写工具共用此 helper，避免 empty-id + ownership 检查重复。
    """
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return None, json.dumps(
            {"error": "id 不能为空"}, ensure_ascii=False,
        )
    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return None, _ownership_denied(str(e))
    store = _get_store(kwargs)
    task = store.get(task_id)
    if task is None:
        return None, json.dumps(
            {"error": f"任务不存在: {task_id}"}, ensure_ascii=False,
        )
    return task, None


def _infer_author(kwargs: dict) -> str:
    """从上下文推断 comment author。"""
    team_name = kwargs.get("team_name")
    if team_name:
        return team_name
    return "main"


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

TASK_CREATE_SCHEMA = {
    "name": "task_create",
    "description": (
        "创建持久化任务（跨会话保留）。用于多步项目追踪。"
        "支持依赖（blocked_by），任务 B 可声明依赖任务 A，A 完成前 B 不能开始。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "任务标题（简短）"},
            "description": {"type": "string", "description": "任务详细描述"},
            "blocked_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "依赖的任务 ID 列表（这些任务完成后本任务才能开始）",
            },
            "owner": {"type": "string", "description": "所有者（agent 名或用户）"},
        },
        "required": ["subject"],
    },
}

TASK_UPDATE_SCHEMA = {
    "name": "task_update",
    "description": "更新持久化任务的状态、所有者等字段。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "新状态",
            },
            "owner": {"type": "string", "description": "认领者"},
            "description": {"type": "string", "description": "更新描述"},
        },
        "required": ["id"],
    },
}

TASK_COMPLETE_SCHEMA = {
    "name": "task_complete",
    "description": "标记任务为完成。会自动解锁依赖本任务的其他任务。",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
        },
        "required": ["id"],
    },
}

TASK_LIST_SCHEMA = {
    "name": "task_list",
    "description": "列出持久化任务。可选按状态过滤。",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "按状态过滤（默认全部）",
            },
        },
    },
}

TASK_HEARTBEAT_SCHEMA = {
    "name": "task_heartbeat",
    "description": (
        "报告当前任务仍在进行（更新 last_heartbeat_at）。"
        "长任务（训练/编码/爬虫）每几分钟调一次。"
        "可选 note 会作为 comment 追加。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "note": {"type": "string", "description": "可选，附加为 comment"},
        },
        "required": ["id"],
    },
}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def _get_store(kwargs: dict):
    home = kwargs.get("harvil_home")
    return get_task_store(home)


def _handle_task_create(args: dict, **kwargs) -> str:
    subject = (args.get("subject") or "").strip()
    if not subject:
        return json.dumps({"error": "subject 不能为空"}, ensure_ascii=False)

    store = _get_store(kwargs)
    task = store.create(
        subject=subject,
        description=args.get("description", ""),
        blocked_by=args.get("blocked_by"),
        owner=args.get("owner"),
    )
    return json.dumps({"success": True, "task": task}, ensure_ascii=False)


def _handle_task_update(args: dict, **kwargs) -> str:
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return json.dumps({"error": "id 不能为空"}, ensure_ascii=False)

    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))

    store = _get_store(kwargs)
    fields = {}
    for key in ("status", "owner", "description", "subject"):
        if key in args and args[key] is not None:
            if key == "status" and args[key] not in VALID_STATUSES:
                return json.dumps(
                    {"error": f"非法 status: {args[key]}"}, ensure_ascii=False,
                )
            fields[key] = args[key]

    task = store.update(task_id, **fields)
    if task is None:
        return json.dumps({"error": f"任务不存在: {task_id}"}, ensure_ascii=False)
    return json.dumps({"success": True, "task": task}, ensure_ascii=False)


def _handle_task_complete(args: dict, **kwargs) -> str:
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return json.dumps({"error": "id 不能为空"}, ensure_ascii=False)

    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))

    store = _get_store(kwargs)
    task = store.complete(task_id)
    if task is None:
        return json.dumps({"error": f"任务不存在: {task_id}"}, ensure_ascii=False)

    # 检查解锁了哪些任务（含 id/subject/status，方便 LLM 判断下一步）
    ready = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in store.find_ready()
    ]
    return json.dumps({
        "success": True,
        "task": task,
        "unblocked": ready,
    }, ensure_ascii=False)


def _handle_task_list(args: dict, **kwargs) -> str:
    status = args.get("status")
    store = _get_store(kwargs)
    tasks = store.list_all(status=status)
    return json.dumps({
        "tasks": tasks,
        "count": len(tasks),
        "ready": [t["id"] for t in store.find_ready()],
    }, ensure_ascii=False)


def _handle_task_heartbeat(args: dict, **kwargs) -> str:
    """更新 last_heartbeat_at；note 非空时附加为 comment。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    note = args.get("note")
    store = _get_store(kwargs)
    if note:
        author = _infer_author(kwargs)
        store.add_comment(task["id"], author=author, content=note)
    updated = store.heartbeat(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

registry.register(
    name="task_create", toolset="core",
    schema=TASK_CREATE_SCHEMA, handler=_handle_task_create, emoji="📝",
)
registry.register(
    name="task_update", toolset="core",
    schema=TASK_UPDATE_SCHEMA, handler=_handle_task_update, emoji="✏️",
)
registry.register(
    name="task_complete", toolset="core",
    schema=TASK_COMPLETE_SCHEMA, handler=_handle_task_complete, emoji="✅",
)
registry.register(
    name="task_list", toolset="core",
    schema=TASK_LIST_SCHEMA, handler=_handle_task_list, emoji="📋",
)
registry.register(
    name="task_heartbeat", toolset="core",
    schema=TASK_HEARTBEAT_SCHEMA, handler=_handle_task_heartbeat, emoji="💓",
)
