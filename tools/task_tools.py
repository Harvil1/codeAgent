"""Task System 的 LLM 工具集：管理「存得住」的任务清单。

这个文件是干嘛的：给 LLM 提供一套任务管理工具（建任务、改状态、标完成、列清单、
留言、记交付物、标记卡住/解除卡住、加依赖）。底层存储在 agent/task_store.py
（每个任务一个 JSON 文件，放在 ~/.OmniMate/.tasks/ 下），关掉会话再开，
任务还在——这就是「持久化」。

历史背景（为什么不直接用清单）：早期有个内存版清单工具 TodoWrite，只在单个
会话里活着、结构是平的，已经被删掉了。现在这套的区别：
  - TodoWrite（已废弃）：存内存里，关会话就没了，平铺结构
  - task_create/update/complete：存文件，跨会话保留，支持 DAG 依赖
    （DAG = 有向无环图，说白了就是「任务之间的先后顺序图」：任务 B 可以
    声明「我要等任务 A 做完才能开始」）

提供的工具一览：
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


MAX_ARTIFACT_SIZE = 100 * 1024 * 1024  # 单个交付物文件的上限：100MB（再大塞进任务记录没意义还占空间）


def _ownership_denied(msg: str) -> str:
    """把「越权操作任务」的报错包成工具协议要求的 JSON 错误串。

    背景：团队协作里一个工人（worker）可能被绑定到某个任务上，只准操作自己
    那个任务。越权时 assert_owned 会抛 TaskOwnershipError，本函数把异常文字
    转成统一格式（error_type=permission_denied）返回给 LLM。

    参数：
        msg：异常的文字，里面已经写清了绑定的任务和想碰的任务
            （如 "worker bound to task 'task_A', cannot operate on 'task_B'"）。

    返回：JSON 字符串，表示权限被拒。
    """
    return json.dumps({
        "error": msg,
        "error_type": "permission_denied",
    }, ensure_ascii=False)


def _get_owned_task(args: dict, kwargs: dict):
    """三合一检查：id 非空 → 没越权 → 任务真的存在，然后取出任务对象。

    背景：所有「写任务」的工具开头都要做这三步一样的检查，抽成一个公共函数
    免得每个工具抄一遍。

    参数：
        args：LLM 传来的工具参数（从里面取 id）。
        kwargs：系统传来的上下文（从里面取 omnimate_home 用来定位任务仓库）。

    返回：两种情况二选一——
        (task, None)：检查全过，task 是任务对象；
        (None, error_json)：哪步挂了就返回哪步的 JSON 错误串。
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
    """装饰器（套在函数外面的公共包装）：先自动做三合一检查，再把任务对象递给函数。

    背景：7 个写任务的工具（update/complete/heartbeat/comment/artifacts/block/unblock）
    开头的三步检查（id 非空、没越权、任务存在）一模一样，抽成装饰器省掉
    约 21 行重复代码。

    参数：
        fn：被包装的原工具 handler。

    被包装后的函数签名变成 ``fn(args, task, **kwargs)``——task 已经保证
    存在且通过了越权检查，函数体里可以直接用。

    返回：包装后的新函数。
    """
    @functools.wraps(fn)
    def wrapped(args, **kwargs):
        task, err = _get_owned_task(args, kwargs)
        if err:
            return err
        return fn(args, task, **kwargs)
    return wrapped


def _infer_author(kwargs: dict) -> str:
    """猜这条留言是谁写的。

    参数：
        kwargs：系统上下文。里面有 team_name（团队协作时的工人名）就署它的名，
        没有就说明是主对话自己写的，署 "main"。

    返回：署名字符串。
    """
    team_name = kwargs.get("team_name")
    if team_name:
        return team_name
    return "main"


def _validate_artifact_path(path: str) -> Optional[str]:
    """检查一个交付物文件路径能不能登记进任务。

    背景：交付物（artifacts）就是「这个任务做出来的文件在哪」，登记前要确认
    文件真实存在、能读、不是目录、没超大小上限，防止登记一堆死链。

    参数：
        path：待检查的文件路径。

    返回：None 表示没问题；否则返回一句人话错误描述（给 LLM 看）。
    """
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
    """根据验证错误文字挑一个错误类型标签。

    参数：
        msg：_validate_artifact_path 返回的错误描述。

    返回：错误里带「过大」就标 artifact_too_large（文件超上限），
    其他情况标 invalid_artifact_path（路径本身不对）。
    """
    if "过大" in msg:
        return "artifact_too_large"
    return "invalid_artifact_path"


# ---------------------------------------------------------------------------
# schema（工具说明书：LLM 看着这些描述来决定怎么调工具）
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
# handler（真正干活的函数，上面 schema 只是说明书）
# ---------------------------------------------------------------------------

def _get_store(kwargs: dict):
    """从上下文拿任务仓库实例。

    参数：
        kwargs：系统上下文。omnimate_home 是 agent 数据目录
        （默认 ~/.OmniMate），不传就用默认。

    返回：TaskStore 实例（任务仓库，带缓存，同一个目录只会建一份）。
    """
    home = kwargs.get("omnimate_home")
    return get_task_store(home)


def _handle_task_create(args: dict, **kwargs) -> str:
    """task_create 的实现：新建一个任务。

    参数：
        args：LLM 传的参数——subject（标题，必填）、description（详细描述）、
        blocked_by（要等哪些任务做完才能开始，任务 ID 列表）、owner（认领人）。
        kwargs：系统上下文（omnimate_home、hooks_registry、agent_ref、session_id）。

    返回：JSON 字符串，带新建的任务对象；标题为空则返回错误。
    """
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

    # 历史出处（round3 D2 新增）：建完任务后触发 TASK_CREATED 钩子
    # （钩子 = 用户配置的附加动作，比如发通知）。fail-open：钩子挂了不影响建任务。
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
    """task_update 的实现：改任务的字段（状态 / 认领人 / 描述 / 标题）。

    特殊分支：状态改成 blocked（卡住）时不能直接改，要走仓库的 mark_blocked
    专用通道——它会记下卡住的历史，而且同一种卡法连续 3 次会自动升级成
    triage（转给调度者处理），避免死循环空转（历史出处：06 轮新增）。

    参数：
        args：LLM 传的参数——id（任务 ID）、status、owner、description、
        subject、block_kind / block_reason（标卡住时用）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，普通更新带回任务对象；标卡住带回
    block 结果（含卡住次数、是否已升级 triage）。
    """
    store = _get_store(kwargs)

    # 标卡住是特殊路径：必须走 mark_blocked（附带记录历史 + 自动升级逻辑）
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

    # 普通更新路径：只挑 LLM 实际传了的字段去改
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
    """task_complete 的实现：把任务标记为完成。

    顺手做两件事：一是允许一次性带上交付物文件路径（先逐个验证再提交，
    讲究原子性——任何一个文件不合格，整个操作都不生效，不会改一半）；
    二是查一下完成后哪些等着它的任务解锁了，告诉 LLM 下一步能干嘛。

    参数：
        args：LLM 传的参数——id（任务 ID）、artifacts（可选，完成时
        一并登记的交付物文件路径列表）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（omnimate_home、hooks_registry、agent_ref、session_id）。

    返回：JSON 字符串，带完成的任务、解锁清单，以及全清/提醒等附加信息。
    """
    # 原子性关键：先把交付物全部验完再动手，任何一个不合格就整单退回，任务状态不动
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
    # 带上 id/标题/状态一并返回，LLM 不用再查一次就能判断接下来做哪个
    ready = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in store.find_ready()
    ]

    # 历史出处（round3 D2 新增）：任务完成时触发 TASK_COMPLETED 钩子（fail-open，挂了不影响主流程）
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

    # 历史出处（R21 #48，对齐 Claude Code TodoWrite 的 allDone/nudge 行为）：
    # 检查是否全部任务都完成了 + 该不该提醒「记得验证」
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


# 这些词出现在任务描述里就不提醒验证——说明用户已经规划了验证步骤
_VERIF_KEYWORDS = ("verif", "验证", "test", "测试", "检查")


def _all_done_cleanup(store) -> dict:
    """检查是不是所有任务都做完了，做完了就在结果里附一个 all_done 标志。

    设计取舍（历史出处 R21 #48，对齐 Claude Code TodoWrite 的 allDone 语义）：
    「清空清单」只是显示层的概念，这里只加一个 all_done=true 的标志来提示
    LLM「全部做完了，别再列任务」。**不真去删任务**——completed 状态留着
    随时可查；真删的话会破坏 test_task_complete_allows_matching_id 等
    既有行为语义（当时的裁决：不删）。

    参数：
        store：任务仓库。

    返回：全部完成时返回 {"all_done": True, "total": 总数}，否则空 dict。
    任何异常都返回空 dict（这个锦上添花的检查不能拖垮主流程）。
    """
    try:
        all_tasks = store.list_all()
        active = [
            t for t in all_tasks
            if t.get("status") not in ("completed", "deleted")
        ]
        if all_tasks and not active:
            return {"all_done": True, "total": len(all_tasks)}
        return {}
    except Exception:
        return {}


def _verification_nudge(store) -> "str | None":
    """活够 3 条任务且都没提验证字样时，返回一句「别忘了验证」的提醒。

    历史出处 R21 #48。提醒文本直接附在 complete 的工具结果里——LLM 天然
    看得见，一行主循环代码都不用改。

    参数：
        store：任务仓库。

    返回：需要提醒时返回提醒文本；活任务不足 3 条、或描述里已含验证
    字样、或出任何异常时，都返回 None（不提醒）。
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
    """task_list 的实现：列出任务清单。

    参数：
        args：LLM 传的参数——status（可选，只看某个状态的任务）。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，含任务列表、总数、以及 ready 字段（现在就能开工的
    任务 ID——依赖都满足了的那种，LLM 挑活直接用它）。
    """
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
    """task_heartbeat 的实现：报平安——「我这活还活着，别当我挂了」。

    背景：长任务（训练模型、跑编码、爬数据）一跑几十分钟，外面需要有办法
    区分「还在干活」和「已经死了」。定期调一次这个工具，就是刷新任务的
    最后心跳时间戳（last_heartbeat_at）。

    参数：
        args：LLM 传的参数——id（任务 ID）、note（可选，附带的说明文字，
        会作为一条留言存进任务）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home、team_name 署名用）。

    返回：JSON 字符串，带刷新后的任务对象。
    """
    note = args.get("note")
    store = _get_store(kwargs)
    if note:
        author = _infer_author(kwargs)
        store.add_comment(task["id"], author=author, content=note)
    updated = store.heartbeat(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


@with_owned_task
def _handle_task_comment(args: dict, task, **kwargs) -> str:
    """task_comment 的实现：往任务的留言本里追加一条。

    留言是持久化的，专门写给后面接手的人（或下一个会话的自己）看：
    记录部分发现、设计决策、留下的疑问。临时想法别写这里。

    参数：
        args：LLM 传的参数——id（任务 ID）、content（留言内容，必填非空）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home、team_name 署名用）。

    返回：JSON 字符串，带追加留言后的任务对象。
    """
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
    """task_artifacts 的实现：登记/移除任务的交付物文件路径。

    原子性保证：批量添加时先把所有路径验完，任何一个不合格就整批退回，
    不会出现「加了一半」的脏状态。

    参数：
        args：LLM 传的参数——id（任务 ID）、add（要登记的文件路径列表）、
        remove（要移除的路径列表）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，带更新后的任务对象。
    """
    add_paths: List[str] = args.get("add") or []
    remove_paths: List[str] = args.get("remove") or []
    store = _get_store(kwargs)

    # 原子性：先全部验完 add 列表，任何一个不合格 → 整批拒绝，一个都不登记
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
    """task_block 的实现：把任务标记为卡住（blocked），必须写明原因。

    参数：
        args：LLM 传的参数——id（任务 ID）、reason（为什么卡住，必填）、
        kind（可选的卡住类型标签：等任务 / 等人拍板 / 缺权限 / 偶发故障）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，带更新后的任务对象；原因或类型不合法返回错误。
    """
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
    """task_unblock 的实现：解除卡住状态，顺便清掉卡住原因和类型。

    参数：
        args：LLM 传的参数——id（任务 ID）、new_status（解除后落到哪个状态，
        只能是 pending 待办或 in_progress 进行中，默认 pending）。
        task：装饰器已经校验过的任务对象。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，带更新后的任务对象；任务本来就不卡或状态值非法
    则返回错误。
    """
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
    """涉及两个任务 ID 时的越权检查（task_link 这种「甲依赖乙」的操作用）。

    背景：普通操作只查一个任务有没有越权，但加依赖这种操作牵扯两边，
    两个 ID 都得过检查——只查一边会留漏洞。

    参数：
        parent_id：被依赖的任务 ID（甲方）。
        child_id：加依赖的任务 ID（乙方）。

    返回：None 表示两边都通过；否则返回权限被拒的 JSON 错误串。
    """
    try:
        assert_owned(parent_id)
        assert_owned(child_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))
    return None


def _handle_task_link(args: dict, **kwargs) -> str:
    """task_link 的实现：给两个任务牵线——「child 要等 parent 做完才能开工」。

    画成图就是加一条依赖边（甲 → 乙）。安全检查一个不少：不许自己依赖
    自己、不许连出循环依赖（A 等 B、B 又等 A，谁也动不了）、两个任务的
    越权检查都要过。

    参数：
        args：LLM 传的参数——parent_id（被等的任务）、child_id（要等的任务）。
        kwargs：系统上下文（取 omnimate_home）。

    返回：JSON 字符串，带更新后的任务对象；缺参/越权/成环各自返回
    对应的错误（error_type 会标 cycle_detected 或 invalid_args）。
    """
    parent_id = (args.get("parent_id") or "").strip()
    child_id = (args.get("child_id") or "").strip()
    if not parent_id or not child_id:
        return json.dumps(
            {"error": "parent_id 和 child_id 必需"}, ensure_ascii=False,
        )
    # 双重越权检查：parent 和 child 两边都得过，只查一边会留漏洞
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
# 注册（import 这个模块时把 10 个工具登记进中央注册表，LLM 才看得见它们）
# ---------------------------------------------------------------------------

registry.register(
    name="task_create", toolset="core",
    schema=TASK_CREATE_SCHEMA, handler=_handle_task_create, emoji="📝",
    isConcurrencySafe=False,  # 有副作用（写任务文件），并发会互相覆盖，必须排队执行
)
registry.register(
    name="task_update", toolset="core",
    schema=TASK_UPDATE_SCHEMA, handler=_handle_task_update, emoji="✏️",
    isConcurrencySafe=False,  # 有副作用（改任务字段），必须排队执行
)
registry.register(
    name="task_complete", toolset="core",
    schema=TASK_COMPLETE_SCHEMA, handler=_handle_task_complete, emoji="✅",
    isConcurrencySafe=False,  # 有副作用（推进状态 + 写交付物），必须排队执行
)
registry.register(
    name="task_list", toolset="core",
    schema=TASK_LIST_SCHEMA, handler=_handle_task_list, emoji="📋",
    isConcurrencySafe=True,  # 只读不改动任何东西，多个一起跑也安全
)
registry.register(
    name="task_heartbeat", toolset="core",
    schema=TASK_HEARTBEAT_SCHEMA, handler=_handle_task_heartbeat, emoji="💓",
    isConcurrencySafe=False,  # 有副作用（刷新心跳时间戳），必须排队执行
)
registry.register(
    name="task_comment", toolset="core",
    schema=TASK_COMMENT_SCHEMA, handler=_handle_task_comment, emoji="💬",
    isConcurrencySafe=False,  # 有副作用（追加留言），必须排队执行
)
registry.register(
    name="task_artifacts", toolset="core",
    schema=TASK_ARTIFACTS_SCHEMA, handler=_handle_task_artifacts, emoji="📎",
    isConcurrencySafe=False,  # 有副作用（改交付物列表），必须排队执行
)
registry.register(
    name="task_block", toolset="core",
    schema=TASK_BLOCK_SCHEMA, handler=_handle_task_block, emoji="⏸",
    isConcurrencySafe=False,  # 有副作用（改依赖图相关状态），必须排队执行
)
registry.register(
    name="task_unblock", toolset="core",
    schema=TASK_UNBLOCK_SCHEMA, handler=_handle_task_unblock, emoji="▶",
    isConcurrencySafe=False,  # 有副作用（改依赖图相关状态），必须排队执行
)
registry.register(
    name="task_link", toolset="core",
    schema=TASK_LINK_SCHEMA, handler=_handle_task_link, emoji="🔗",
    isConcurrencySafe=False,  # 有副作用（改任务依赖关系），必须排队执行
)
