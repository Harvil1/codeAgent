"""后台任务工具：5 个 handler 注册到 registry。

工具：
- bg_start: 启动后台任务
- bg_status: 查询任务状态
- bg_result: 查询完整输出
- bg_list: 列出所有任务
- bg_stop: 终止任务

handler 通过 kwargs 接收 bg_manager（由 agent 透传）。
"""
import json
import logging
from pathlib import Path

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

BG_START_SCHEMA = {
    "name": "bg_start",
    "description": "启动后台任务（异步执行长命令，立即返回 task_id）",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "命令及参数（list 形式，不经 shell）",
            },
            "cwd": {"type": "string", "description": "工作目录（可选）"},
            "detach": {
                "type": "boolean",
                "description": "是否脱离 agent 进程组（默认 false）",
            },
            "timeout": {"type": "number", "description": "超时秒数（默认 600）"},
        },
        "required": ["command"],
    },
}

BG_STATUS_SCHEMA = {
    "name": "bg_status",
    "description": "查询后台任务状态",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}

BG_RESULT_SCHEMA = {
    "name": "bg_result",
    "description": "查询后台任务的完整输出（stdout/stderr 各 cap 5000）",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}

BG_LIST_SCHEMA = {
    "name": "bg_list",
    "description": "列出所有后台任务",
    "parameters": {"type": "object", "properties": {}},
}

BG_STOP_SCHEMA = {
    "name": "bg_stop",
    "description": "终止后台任务",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def _handle_bg_start(args: dict, **kwargs) -> str:
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    command = args.get("command")
    if not command or not isinstance(command, list):
        return json.dumps({
            "error": "bg_start requires 'command' as non-empty list",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    cwd_raw = args.get("cwd")
    cwd = Path(cwd_raw) if cwd_raw else None
    detach = bool(args.get("detach", False))
    timeout = args.get("timeout")
    try:
        task_id = bg_manager.start(
            command, cwd=cwd, detach=detach,
            timeout=timeout if timeout is not None else None,
        )
    except RuntimeError as e:
        return json.dumps({
            "error": str(e),
            "error_type": "bg_task_full",
        }, ensure_ascii=False)
    task = bg_manager.status(task_id)
    return json.dumps({
        "task_id": task_id,
        "status": task.status,
        "pid": task.pid,
    }, ensure_ascii=False)


def _handle_bg_status(args: dict, **kwargs) -> str:
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "bg_status requires 'task_id'",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    task = bg_manager.status(task_id)
    if task is None:
        return json.dumps({
            "error": f"task not found: {task_id}",
            "error_type": "bg_task_not_found",
        }, ensure_ascii=False)
    runtime = None
    if task.started_at:
        from datetime import datetime
        end = task.ended_at or datetime.now()
        runtime = (end - task.started_at).total_seconds()
    return json.dumps({
        "task_id": task.task_id,
        "status": task.status,
        "pid": task.pid,
        "runtime_seconds": runtime,
    }, ensure_ascii=False)


def _handle_bg_result(args: dict, **kwargs) -> str:
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "bg_result requires 'task_id'",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    task = bg_manager.result(task_id)
    if task is None:
        return json.dumps({
            "error": f"task not found: {task_id}",
            "error_type": "bg_task_not_found",
        }, ensure_ascii=False)
    return json.dumps({
        "task_id": task.task_id,
        "status": task.status,
        "exit_code": task.exit_code,
        "stdout": task.stdout,
        "stderr": task.stderr,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "ended_at": task.ended_at.isoformat() if task.ended_at else None,
    }, ensure_ascii=False)


def _handle_bg_list(args: dict, **kwargs) -> str:
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    tasks = bg_manager.list_tasks()
    summaries = [
        {
            "task_id": t.task_id,
            "status": t.status,
            "command": t.command,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "ended_at": t.ended_at.isoformat() if t.ended_at else None,
        }
        for t in tasks
    ]
    return json.dumps({"tasks": summaries}, ensure_ascii=False)


def _handle_bg_stop(args: dict, **kwargs) -> str:
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "bg_stop requires 'task_id'",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    ok = bg_manager.stop(task_id)
    if not ok:
        return json.dumps({
            "error": f"task not found: {task_id}",
            "error_type": "bg_task_not_found",
        }, ensure_ascii=False)
    task = bg_manager.status(task_id)
    return json.dumps({
        "task_id": task_id,
        "status": task.status if task else "stopped",
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

registry.register(
    name="bg_start", toolset="core",
    schema=BG_START_SCHEMA, handler=_handle_bg_start, emoji="🚀",
)
registry.register(
    name="bg_status", toolset="core",
    schema=BG_STATUS_SCHEMA, handler=_handle_bg_status, emoji="📊",
)
registry.register(
    name="bg_result", toolset="core",
    schema=BG_RESULT_SCHEMA, handler=_handle_bg_result, emoji="📄",
)
registry.register(
    name="bg_list", toolset="core",
    schema=BG_LIST_SCHEMA, handler=_handle_bg_list, emoji="📋",
)
registry.register(
    name="bg_stop", toolset="core",
    schema=BG_STOP_SCHEMA, handler=_handle_bg_stop, emoji="🛑",
)
