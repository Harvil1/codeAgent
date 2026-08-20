"""后台任务工具：让 LLM 能把命令扔到后台去跑，不用干等结果。

好比你在灶上炖着汤（后台任务）继续炒菜（主对话），随时过来看一眼火候。
共注册 5 个工具：
- bg_start: 启动一个后台任务（长命令异步跑，马上返回任务号）
- bg_status: 查某个任务的状态
- bg_result: 取某个任务的完整输出
- bg_list: 列出所有后台任务
- bg_stop: 停掉某个任务

在项目里的位置：tools 层工具，真正管进程的是 agent/background.py 的
BackgroundManager（经 kwargs 里的 bg_manager 透传进来）。
"""
import json
import logging
import os
from pathlib import Path

from config import load_config
from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# check_fn（可见性开关）：用 config.bg_task.enabled 决定这批 bg_* 工具
# 对 LLM 是否可见——关掉开关，工具直接从 LLM 眼里消失
# ---------------------------------------------------------------------------

def _check_bg_enabled() -> bool:
    """读配置里的 bg_task.enabled，判断这组后台工具是否开放。

    返回：True 表示开放。默认 True；连配置都读失败时也按开放处理（宁可放行）。
    """
    try:
        return bool(load_config().get("bg_task", {}).get("enabled", True))
    except Exception:
        return True  # 配置读取失败时默认可用


# ---------------------------------------------------------------------------
# schema：工具的"说明书"，发给 LLM 看的参数定义
# ---------------------------------------------------------------------------

BG_START_SCHEMA = {
    "name": "bg_start",
    "description": (
        "启动后台任务（异步执行长命令，立即返回 task_id）。"
        "monitor=true 走流式监视语义（tail -f/watch/轮询命令）：豁免停滞看门狗"
        "（安静是常态）、stdout 增量落盘 output_file（read_file 随时查）、"
        "timeout 默认 24h、进程退出时通知。一次性命令不要用 monitor。"
    ),
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
            "timeout": {"type": "number", "description": "超时秒数（默认 600；monitor 默认 86400）"},
            "monitor": {
                "type": "boolean",
                "description": "流式监视模式（tail -f/watch/轮询），默认 false",
            },
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
# handler：真正干活的处理函数
# ---------------------------------------------------------------------------

_BG_UNAVAILABLE_ERR = json.dumps({
    "error": "background manager not available",
    "error_type": "bg_unavailable",
}, ensure_ascii=False)


def _require_bg_manager(kwargs):
    """从 kwargs 里取后台任务管理器 bg_manager。

    背景：所有 bg_* 工具都依赖它，缺了就干不了活，所以统一在这里检查。

    参数：
        kwargs: dispatch 透传的命名上下文
    返回：(管理器, None) 或不可用时 (None, 错误 JSON 字符串)。
    """
    bg_manager = kwargs.get("bg_manager")
    if bg_manager is None:
        return None, _BG_UNAVAILABLE_ERR
    return bg_manager, None


def _require_task_id(args, tool_name):
    """从 args 里取 task_id（任务编号）并检查非空。

    参数：
        args: LLM 传来的工具参数
        tool_name: 工具名（拼进错误消息，让 LLM 知道是谁在要这个参数）
    返回：(task_id, None) 或参数为空时 (None, 错误 JSON 字符串)。
    """
    task_id = args.get("task_id")
    if not task_id:
        return None, json.dumps({
            "error": f"{tool_name} requires 'task_id'",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    return task_id, None


def _task_not_found_err(task_id):
    """构造"任务不存在"的错误 JSON。

    参数：
        task_id: 查不到的任务编号
    返回：JSON 字符串，error_type 为 bg_task_not_found。
    """
    return json.dumps({
        "error": f"task not found: {task_id}",
        "error_type": "bg_task_not_found",
    }, ensure_ascii=False)


def _handle_bg_start(args: dict, **kwargs) -> str:
    """bg_start 的处理函数：先过权限审查，再把命令交给后台管理器去跑。

    背景：后台任务容易被人当成"绕过安检的后门"——历史上它确实漏检过，
    所以现在启动前必须完整走权限闸门（见下方 S1 fix 注释）。

    参数：
        args: LLM 传来的参数——command（命令及参数的列表）、cwd（工作目录，
              可选）、detach（是否脱离主进程独立存活，默认否）、timeout
              （超时秒数，可选）、monitor（流式监视模式，默认否）
        kwargs: 命名上下文——bg_manager（后台管理器）、permission_checker
                （权限检查器，可选）、以及权限模式相关字段
    返回：JSON 字符串。成功带 task_id/status/pid；权限被拒或参数错误
    带相应 error。
    """
    bg_manager, err = _require_bg_manager(kwargs)
    if err:
        return err
    command = args.get("command")
    if not command or not isinstance(command, list):
        return json.dumps({
            "error": "bg_start requires 'command' as non-empty list",
            "error_type": "invalid_args",
        }, ensure_ascii=False)

    # 历史踩坑（S1 修复）：bg_start 必须过权限闸门，包括 fatal 硬底线。
    # 早期版本完全跳过 PermissionChecker，等于开了个后门——
    # 借 bg_start 就能跑 rm -rf / 这种任何模式都该拦的命令。
    import shlex
    from agent.permission import (
        get_default_checker, check_fatal_irreversible,
    )
    from tools._common import get_mode_override_from_kwargs

    command_str = " ".join(shlex.quote(str(c)) for c in command)
    cwd_raw = args.get("cwd")
    if not cwd_raw:
        # 并发子代理的工作目录：优先读 ContextVar（线程隔离的变量，各线程互不串），
        # 读不到再退回 os.getcwd()——避免并发子代理互相踩工作目录
        from agent.workspace_context import get_workspace_cwd
        cwd_raw = get_workspace_cwd()
    cwd_for_check = cwd_raw

    # 闸门 0b：fatal 不可逆命令（rm -rf / 这类），任何权限模式都挡
    fatal = check_fatal_irreversible(command_str)
    if fatal:
        return json.dumps({
            "error": f"权限拒绝: 硬底线: {fatal}",
            "error_type": "permission_denied",
            "gate": "deny",
            "command": command_str,
        }, ensure_ascii=False)

    # 再过常规三道闸门（黑名单 → 规则 → 审批）
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = get_mode_override_from_kwargs(kwargs)
    perm = checker.check(command_str, cwd=cwd_for_check, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: {perm.reason}",
            "error_type": "permission_denied",
            "gate": perm.gate,
            "command": command_str,
        }, ensure_ascii=False)

    cwd = Path(cwd_raw) if cwd_raw else None
    detach = bool(args.get("detach", False))
    timeout = args.get("timeout")
    monitor = bool(args.get("monitor", False))  # 流式监视模式（R22 轮 #32 引入）：给 tail -f 这类持续输出命令用的
    try:
        task_id = bg_manager.start(
            command, cwd=cwd, detach=detach,
            timeout=timeout if timeout is not None else None,
            monitor=monitor,
        )
    except RuntimeError as e:
        return json.dumps({
            "error": str(e),
            "error_type": "bg_task_full",
        }, ensure_ascii=False)
    task = bg_manager.status(task_id)
    resp = {
        "task_id": task_id,
        "status": task.status,
        "pid": task.pid,
    }
    if monitor:
        resp["monitor"] = True
        if task.output_file:
            resp["output_file"] = task.output_file
        resp["hint"] = "监视器已启动：用 read_file(output_file) 查增量输出；进程退出时会收到通知"
    return json.dumps(resp, ensure_ascii=False)


def _handle_bg_status(args: dict, **kwargs) -> str:
    """bg_status 的处理函数：查一个后台任务现在的状态和跑了多久。

    参数：
        args: LLM 传来的参数，只用到 task_id（任务编号）
        kwargs: 命名上下文，用 bg_manager（后台管理器）
    返回：JSON 字符串，含 task_id/status/pid/runtime_seconds（运行秒数）。
    任务还在跑时，运行时长按"到现在为止"实时算。
    """
    bg_manager, err = _require_bg_manager(kwargs)
    if err:
        return err
    task_id, err = _require_task_id(args, "bg_status")
    if err:
        return err
    task = bg_manager.status(task_id)
    if task is None:
        return _task_not_found_err(task_id)
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
    """bg_result 的处理函数：取一个后台任务的完整输出（stdout/stderr/退出码）。

    参数：
        args: LLM 传来的参数，只用到 task_id（任务编号）
        kwargs: 命名上下文，用 bg_manager（后台管理器）
    返回：JSON 字符串，含任务的 stdout、stderr、exit_code 和起止时间
    （stdout/stderr 各自最多给 5000 字符，超了会截断）。
    """
    bg_manager, err = _require_bg_manager(kwargs)
    if err:
        return err
    task_id, err = _require_task_id(args, "bg_result")
    if err:
        return err
    task = bg_manager.result(task_id)
    if task is None:
        return _task_not_found_err(task_id)
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
    """bg_list 的处理函数：列出当前所有后台任务的概要。

    参数：
        args: 无实际参数（schema 里 properties 为空）
        kwargs: 命名上下文，用 bg_manager（后台管理器）
    返回：JSON 字符串，tasks 数组里每项含 task_id/status/command/起止时间。
    """
    bg_manager, err = _require_bg_manager(kwargs)
    if err:
        return err
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
    """bg_stop 的处理函数：停掉一个后台任务（杀掉子进程）。

    参数：
        args: LLM 传来的参数，只用到 task_id（任务编号）
        kwargs: 命名上下文，用 bg_manager（后台管理器）
    返回：JSON 字符串，含 task_id 和停止后的 status；任务不存在时报错。
    """
    bg_manager, err = _require_bg_manager(kwargs)
    if err:
        return err
    task_id, err = _require_task_id(args, "bg_stop")
    if err:
        return err
    ok = bg_manager.stop(task_id)
    if not ok:
        return _task_not_found_err(task_id)
    task = bg_manager.status(task_id)
    return json.dumps({
        "task_id": task_id,
        "status": task.status if task else "stopped",
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册：本文件一被 import 就把 5 个工具登记进 registry
# （check_fn 挂上可见性开关，config 关了这组工具就对 LLM 隐身）
# ---------------------------------------------------------------------------

registry.register(
    name="bg_start", toolset="bg",
    schema=BG_START_SCHEMA, handler=_handle_bg_start, emoji="🚀",
    check_fn=_check_bg_enabled,
    isConcurrencySafe=False,  # 有副作用：会启动子进程，必须串行
)
registry.register(
    name="bg_status", toolset="bg",
    schema=BG_STATUS_SCHEMA, handler=_handle_bg_status, emoji="📊",
    check_fn=_check_bg_enabled,
    isConcurrencySafe=True,  # 只读：查状态，无副作用，可并发
)
registry.register(
    name="bg_result", toolset="bg",
    schema=BG_RESULT_SCHEMA, handler=_handle_bg_result, emoji="📄",
    check_fn=_check_bg_enabled,
    isConcurrencySafe=True,  # 只读：取输出，无副作用，可并发
)
registry.register(
    name="bg_list", toolset="bg",
    schema=BG_LIST_SCHEMA, handler=_handle_bg_list, emoji="📋",
    check_fn=_check_bg_enabled,
    isConcurrencySafe=True,  # 只读：列任务，无副作用，可并发
)
registry.register(
    name="bg_stop", toolset="bg",
    schema=BG_STOP_SCHEMA, handler=_handle_bg_stop, emoji="🛑",
    check_fn=_check_bg_enabled,
    isConcurrencySafe=False,  # 有副作用：会杀子进程，必须串行
)
