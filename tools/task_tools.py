"""Task System 工具集：持久化任务管理。

和已废弃的 TodoWrite（todo_write 工具，已删除）的区别：
  - TodoWrite（已废弃）：曾是内存清单工具，单会话，扁平结构，已被删除
  - task_create/update/complete：持久化到 .tasks/，跨会话，DAG 依赖

工具：
  task_create(subject, description, blocked_by)  创建任务
  task_update(id, status, owner, ...)            更新任务
  task_complete(id)                              完成任务
  task_list(status)                              列出任务
"""

import functools
import json
import os
from pathlib import Path
from typing import List, Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry


MAX_ARTIFACT_SIZE = 100 * 1024 * 1024  # 100MB


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


def with_owned_task(fn):
    """装饰器：自动过 ownership 门控并注入已校验的 task 对象。

    7 个 task_* 写工具（update/complete/heartbeat/comment/artifacts/block/unblock）
    都要先做 empty-id + ownership + existence 三步检查，逻辑完全一致。
    抽出来用装饰器消除 ~21 行重复代码。

    handler 签名变为 ``fn(args, task, **kwargs)``，task 已保证存在且通过 ownership。
    """
    @functools.wraps(fn)
    def wrapped(args, **kwargs):
        task, err = _get_owned_task(args, kwargs)
        if err:
            return err
        return fn(args, task, **kwargs)
    return wrapped


def _infer_author(kwargs: dict) -> str:
    """从上下文推断 comment author。"""
    team_name = kwargs.get("team_name")
    if team_name:
        return team_name
    return "main"


def _validate_artifact_path(path: str) -> Optional[str]:
    """验证单个附件路径。返回 None=OK，否则返回错误描述。"""
    p = Path(path)
    if not p.exists():
        return f"路径不存在: {path}"
    if not os.path.isfile(path):
        return f"不是文件（拒绝目录）: {path}"
    if not os.access(path, os.R_OK):
        return f"不可读: {path}"
    size = p.stat().st_size
    if size > MAX_ARTIFACT_SIZE:
        return f"文件过大: {size} bytes（上限 {MAX_ARTIFACT_SIZE}）"
    return None


def _artifact_error_type(msg: str) -> str:
    """根据验证错误消息选 error_type。"""
    if "过大" in msg:
        return "artifact_too_large"
    return "invalid_artifact_path"


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
    "description": (
        "更新持久化任务的状态、所有者等字段。\n\n"
        "阻塞场景：设 status='blocked' 时必填 block_kind 和 block_reason。\n"
        "同 kind 阻塞 3 次会自动升级到 triage（让 orchestrator 介入），避免死循环。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked", "triage"],
                "description": "新状态",
            },
            "owner": {"type": "string", "description": "认领者"},
            "description": {"type": "string", "description": "更新描述"},
            "block_kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": (
                    "阻塞时必填。dependency=等任务、needs_input=等输入、"
                    "capability=能力不足、transient=临时"
                ),
            },
            "block_reason": {
                "type": "string",
                "description": "阻塞原因（自由文本）",
            },
        },
        "required": ["id"],
    },
}

TASK_COMPLETE_SCHEMA = {
    "name": "task_complete",
    "description": (
        "标记任务完成。会自动解锁依赖本任务的其他任务。"
        "可选 artifacts：完成时一并加入交付物路径（与 task_artifacts 同款验证）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选。完成时一并加入的交付物路径",
            },
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

TASK_COMMENT_SCHEMA = {
    "name": "task_comment",
    "description": (
        "给任务追加一条持久化留言（写进任务本，跨会话保留）。"
        "用于：给下一个 worker 留问题、记录部分发现、记设计决策。"
        "临时推理不要写这里，放普通回复里。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "content": {"type": "string", "description": "留言内容"},
        },
        "required": ["id", "content"],
    },
}

TASK_ARTIFACTS_SCHEMA = {
    "name": "task_artifacts",
    "description": (
        "管理任务的交付物文件路径列表（add / remove）。"
        "路径必须存在、可读、单个文件 ≤100MB。"
        "用于让下游（gateway notifier 等）知道交付物在哪。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要加入的文件绝对路径列表",
            },
            "remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要移除的文件路径列表",
            },
        },
        "required": ["id"],
    },
}

TASK_BLOCK_SCHEMA = {
    "name": "task_block",
    "description": (
        "把任务标记为 blocked（卡住）。必须填 reason 解释为什么卡住。"
        "可选 kind：'dependency'（等其他任务）/ 'needs_input'（等人决策）/ "
        "'capability'（缺权限/凭证）/ 'transient'（偶发失败可能恢复）。"
        "kind 仅作人类可读标签，不影响自动化行为。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "reason": {"type": "string", "description": "为什么卡住（必填）"},
            "kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": "可选，block 类型标签",
            },
        },
        "required": ["id", "reason"],
    },
}

TASK_UNBLOCK_SCHEMA = {
    "name": "task_unblock",
    "description": (
        "解除 task 的 blocked 状态。默认回到 pending；可选 new_status='in_progress'。"
        "清空 block_reason / block_kind。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "new_status": {
                "type": "string",
                "enum": ["pending", "in_progress"],
                "default": "pending",
            },
        },
        "required": ["id"],
    },
}

TASK_LINK_SCHEMA = {
    "name": "task_link",
    "description": (
        "post-creation 加依赖边：让 child 依赖 parent（parent 完成前 child 不能开始）。"
        "含 cycle 检测和 self-link 拒绝。"
        "parent_id 和 child_id 都过 ownership 门控。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "string", "description": "被依赖的任务 ID"},
            "child_id": {"type": "string", "description": "加依赖的任务 ID"},
        },
        "required": ["parent_id", "child_id"],
    },
}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def _get_store(kwargs: dict):
    home = kwargs.get("omnimate_home")
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

    # round3 D2 NEW: TASK_CREATED hook（fail-open）
    _hooks = kwargs.get("hooks_registry")
    if _hooks is None:
        _agent = kwargs.get("agent_ref")
        _hooks = getattr(_agent, "hooks_registry", None) if _agent else None
    if _hooks is not None:
        try:
            _hooks.run_task_created({
                "session_id": kwargs.get("session_id", ""),
                "task_id": task["id"],
                "subject": task.get("subject", ""),
                "owner": task.get("owner"),
            })
        except Exception:
            pass  # fail-open

    return json.dumps({"success": True, "task": task}, ensure_ascii=False)


@with_owned_task
def _handle_task_update(args: dict, task, **kwargs) -> str:
    """更新 task 字段（status / owner / description / subject）。

    06 NEW: status='blocked' 时走 mark_blocked（记录 block_history + 升级 triage）。
    """
    store = _get_store(kwargs)

    # 阻塞特殊路径：走 mark_blocked（自动升级）
    if args.get("status") == "blocked":
        try:
            result = store.mark_blocked(
                task["id"],
                kind=args.get("block_kind", "transient"),
                reason=args.get("block_reason", ""),
            )
        except KeyError as e:
            return json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            )
        return json.dumps({
            "success": True,
            "action": "block",
            "id": task["id"],
            "new_status": result["status"],
            "block_count": result["block_count"],
            "block_kind": result["kind"],
            "upgraded_to_triage": result["status"] == "triage",
        }, ensure_ascii=False)

    # 普通更新路径
    fields = {}
    for key in ("status", "owner", "description", "subject"):
        if key in args and args[key] is not None:
            if key == "status" and args[key] not in VALID_STATUSES:
                return json.dumps(
                    {"error": f"非法 status: {args[key]}"}, ensure_ascii=False,
                )
            fields[key] = args[key]

    updated = store.update(task["id"], **fields)
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_complete(args: dict, task, **kwargs) -> str:
    """标记完成。可选 artifacts 一次性提交（先验证，原子性）。"""
    # 新增：先验证 artifacts，再 complete（任一失败 → 整个调用不变）
    artifacts: List[str] = args.get("artifacts") or []
    store = _get_store(kwargs)
    if artifacts:
        for p in artifacts:
            verr = _validate_artifact_path(p)
            if verr:
                return json.dumps(
                    {"error": verr, "error_type": _artifact_error_type(verr)},
                    ensure_ascii=False,
                )
        store.add_artifacts(task["id"], artifacts)

    completed = store.complete(task["id"])
    # 检查解锁了哪些任务（含 id/subject/status，方便 LLM 判断下一步）
    ready = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in store.find_ready()
    ]

    # round3 D2 NEW: TASK_COMPLETED hook（fail-open）
    _hooks = kwargs.get("hooks_registry")
    if _hooks is None:
        _agent = kwargs.get("agent_ref")
        _hooks = getattr(_agent, "hooks_registry", None) if _agent else None
    if _hooks is not None:
        try:
            _hooks.run_task_completed({
                "session_id": kwargs.get("session_id", ""),
                "task_id": task["id"],
                "subject": task.get("subject", ""),
                "unblocked": ready,
            })
        except Exception:
            pass  # fail-open

    # === R21 #48：任务全清 + verification nudge（对齐 CC TodoWrite allDone/nudge）===
    extra = _all_done_cleanup(store)
    nudge = _verification_nudge(store)
    if nudge:
        extra["reminder"] = nudge

    return json.dumps({
        "success": True,
        "task": completed,
        "unblocked": ready,
        **extra,
    }, ensure_ascii=False)


# 验证关键词（任务描述命中则不提醒——用户已规划了验证步骤）
_VERIF_KEYWORDS = ("verif", "验证", "test", "测试", "检查")


def _all_done_cleanup(store) -> dict:
    """R21 #48：全部任务 completed → 自动清空列表（软删 status=deleted，可恢复）。

    对齐 CC allDone→[]：列表清空让下个任务集从干净状态开始；
    软删保留文件（完全可逆铁律）。
    """
    try:
        all_tasks = store.list_all()
        active = [
            t for t in all_tasks
            if t.get("status") not in ("completed", "deleted")
        ]
        if all_tasks and not active:
            cleared = 0
            for t in all_tasks:
                if t.get("status") == "completed":
                    try:
                        store.update(t["id"], status="deleted")
                        cleared += 1
                    except Exception:
                        pass
            if cleared:
                return {"all_tasks_cleared": True, "cleared_count": cleared}
        return {}
    except Exception:
        return {}


def _verification_nudge(store) -> "str | None":
    """R21 #48：≥3 条活跃任务且描述无验证字样 → 返回提醒文本（附在 complete 结果里）。

    提醒走 tool result（零主循环改动，LLM 直接看到）。
    """
    try:
        active = [
            t for t in store.list_all()
            if t.get("status") in ("pending", "in_progress")
        ]
        if len(active) < 3:
            return None
        desc_text = " ".join(
            f"{t.get('subject', '')} {t.get('description', '')}"
            for t in active
        ).lower()
        if any(k in desc_text for k in _VERIF_KEYWORDS):
            return None
        return (
            f"提醒：当前有 {len(active)} 条活跃任务，描述中都没有验证步骤——"
            f"完成前别忘了验证（跑测试 / 复核结果）。"
        )
    except Exception:
        return None


def _handle_task_list(args: dict, **kwargs) -> str:
    status = args.get("status")
    store = _get_store(kwargs)
    tasks = store.list_all(status=status)
    return json.dumps({
        "tasks": tasks,
        "count": len(tasks),
        "ready": [t["id"] for t in store.find_ready()],
    }, ensure_ascii=False)


@with_owned_task
def _handle_task_heartbeat(args: dict, task, **kwargs) -> str:
    """更新 last_heartbeat_at；note 非空时附加为 comment。"""
    note = args.get("note")
    store = _get_store(kwargs)
    if note:
        author = _infer_author(kwargs)
        store.add_comment(task["id"], author=author, content=note)
    updated = store.heartbeat(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_comment(args: dict, task, **kwargs) -> str:
    """追加 comment 到 task.comments。"""
    content = (args.get("content") or "").strip()
    if not content:
        return json.dumps(
            {"error": "content 不能为空"}, ensure_ascii=False,
        )
    author = _infer_author(kwargs)
    store = _get_store(kwargs)
    updated = store.add_comment(task["id"], author=author, content=content)
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_artifacts(args: dict, task, **kwargs) -> str:
    """管理 task.artifacts 列表（add / remove）。原子性：批量 add 任一失败整批拒绝。"""
    add_paths: List[str] = args.get("add") or []
    remove_paths: List[str] = args.get("remove") or []
    store = _get_store(kwargs)

    # 原子性：先全部验证 add，任一失败 → 整批拒绝
    for p in add_paths:
        verr = _validate_artifact_path(p)
        if verr:
            return json.dumps(
                {"error": verr, "error_type": _artifact_error_type(verr)},
                ensure_ascii=False,
            )

    if add_paths:
        store.add_artifacts(task["id"], add_paths)
    if remove_paths:
        store.remove_artifacts(task["id"], remove_paths)
    updated = store.get(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_block(args: dict, task, **kwargs) -> str:
    """task_block: status=blocked + block_reason + block_kind。"""
    reason = (args.get("reason") or "").strip()
    if not reason:
        return json.dumps(
            {"error": "reason 不能为空"}, ensure_ascii=False,
        )
    kind = args.get("kind")
    if kind is not None and kind not in (
        "dependency", "needs_input", "capability", "transient",
    ):
        return json.dumps(
            {"error": f"非法 kind: {kind}"}, ensure_ascii=False,
        )
    store = _get_store(kwargs)
    updated = store.update(
        task["id"],
        status="blocked",
        block_reason=reason,
        block_kind=kind,
    )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_unblock(args: dict, task, **kwargs) -> str:
    """task_unblock: 解除 blocked，清空 block_reason/block_kind。"""
    new_status = args.get("new_status") or "pending"
    if new_status not in ("pending", "in_progress"):
        return json.dumps(
            {"error": f"非法 new_status: {new_status}（只允许 pending 或 in_progress）"},
            ensure_ascii=False,
        )
    if task.get("status") != "blocked":
        return json.dumps(
            {"error": f"任务不是 blocked 状态（当前: {task.get('status')}）"},
            ensure_ascii=False,
        )
    store = _get_store(kwargs)
    updated = store.update(
        task["id"],
        status=new_status,
        block_reason=None,
        block_kind=None,
    )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


def _assert_dual_ownership(parent_id: str, child_id: str) -> Optional[str]:
    """双 id 的 ownership 门控(task_link 等用)。

    parent_id 和 child_id 都要过 assert_owned。
    返回 None=通过,返回 str=给客户端的 JSON 错误(permission_denied)。
    """
    try:
        assert_owned(parent_id)
        assert_owned(child_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))
    return None


def _handle_task_link(args: dict, **kwargs) -> str:
    """task_link: 加 child 依赖 parent 的边（含 cycle+self-link 检测，双重 ownership 门控）。"""
    parent_id = (args.get("parent_id") or "").strip()
    child_id = (args.get("child_id") or "").strip()
    if not parent_id or not child_id:
        return json.dumps(
            {"error": "parent_id 和 child_id 必需"}, ensure_ascii=False,
        )
    # 双重 ownership 门控:parent 和 child 都要过
    deny = _assert_dual_ownership(parent_id, child_id)
    if deny:
        return deny
    store = _get_store(kwargs)
    try:
        updated = store.add_dependency(child_id, parent_id, validate=True)
    except ValueError as e:
        msg = str(e)
        error_type = (
            "cycle_detected" if "cycle" in msg or "self-link" in msg
            else "invalid_args"
        )
        return json.dumps(
            {"error": msg, "error_type": error_type},
            ensure_ascii=False,
        )
    if updated is None:
        return json.dumps(
            {"error": f"任务不存在: {child_id}"}, ensure_ascii=False,
        )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

registry.register(
    name="task_create", toolset="core",
    schema=TASK_CREATE_SCHEMA, handler=_handle_task_create, emoji="📝",
    isConcurrencySafe=False,  # 副作用：写任务文件，必须串行
)
registry.register(
    name="task_update", toolset="core",
    schema=TASK_UPDATE_SCHEMA, handler=_handle_task_update, emoji="✏️",
    isConcurrencySafe=False,  # 副作用：改任务字段，必须串行
)
registry.register(
    name="task_complete", toolset="core",
    schema=TASK_COMPLETE_SCHEMA, handler=_handle_task_complete, emoji="✅",
    isConcurrencySafe=False,  # 副作用：状态机推进 + 写 artifacts，必须串行
)
registry.register(
    name="task_list", toolset="core",
    schema=TASK_LIST_SCHEMA, handler=_handle_task_list, emoji="📋",
    isConcurrencySafe=True,  # 只读：列任务，无副作用，可并发
)
registry.register(
    name="task_heartbeat", toolset="core",
    schema=TASK_HEARTBEAT_SCHEMA, handler=_handle_task_heartbeat, emoji="💓",
    isConcurrencySafe=False,  # 副作用：更新 last_active_at，必须串行
)
registry.register(
    name="task_comment", toolset="core",
    schema=TASK_COMMENT_SCHEMA, handler=_handle_task_comment, emoji="💬",
    isConcurrencySafe=False,  # 副作用：追加评论，必须串行
)
registry.register(
    name="task_artifacts", toolset="core",
    schema=TASK_ARTIFACTS_SCHEMA, handler=_handle_task_artifacts, emoji="📎",
    isConcurrencySafe=False,  # 副作用：加/改 artifacts 列表，必须串行
)
registry.register(
    name="task_block", toolset="core",
    schema=TASK_BLOCK_SCHEMA, handler=_handle_task_block, emoji="⏸",
    isConcurrencySafe=False,  # 副作用：改 blocked_by DAG，必须串行
)
registry.register(
    name="task_unblock", toolset="core",
    schema=TASK_UNBLOCK_SCHEMA, handler=_handle_task_unblock, emoji="▶",
    isConcurrencySafe=False,  # 副作用：改 blocked_by DAG，必须串行
)
registry.register(
    name="task_link", toolset="core",
    schema=TASK_LINK_SCHEMA, handler=_handle_task_link, emoji="🔗",
    isConcurrencySafe=False,  # 副作用：改任务依赖关系，必须串行
)
