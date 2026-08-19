"""workflow 工具：确定性编排的 LLM 入口（R28 W4，蓝图 §2）。

五 action：run（name=registry 名 或 script=内联）/ resume（只信 run 目录快照）/
list / status / kill。DSL 写法见 skills/workflow-dsl/SKILL.md（窄腰：description
一句话，文档按需 load_skill）。

C3（CCB 借鉴 detached launch）：run/resume 支持 wait=false——后台线程 + 自有
事件循环跑，立即返回 run_id 不阻塞主循环；完成经 delegation 队列以后台通知
送达；同会话可 kill（取消通道用 threading.Event，跨线程可用）。run 目录
KEEP_MAX_RUNS=50 LRU 清理。
"""
import asyncio
import json
import logging
import shutil
import threading
import time
import uuid

from tools.registry import registry

logger = logging.getLogger(__name__)

# run_id -> threading.Event（活跃 run 的取消通道；进程内）。
# C3：threading.Event——detached run 跑在后台线程的事件循环里，主循环线程
# 要能跨线程 set（asyncio.Event 不跨线程/循环）
_ACTIVE_RUNS: dict = {}

# C3（CCB KEEP_MAX_RUNS）：run 目录上限（LRU by mtime 清理最旧，跳过活跃）
_KEEP_MAX_RUNS = 50


def _cleanup_old_runs(base) -> int:
    """run 目录超上限时按 LRU 清理最旧（fail-open；跳过活跃 run）。"""
    try:
        dirs = [d for d in base.iterdir() if d.is_dir()]
        if len(dirs) <= _KEEP_MAX_RUNS:
            return 0
        dirs.sort(key=lambda p: p.stat().st_mtime)
        removed = 0
        for d in dirs[:len(dirs) - _KEEP_MAX_RUNS]:
            if d.name in _ACTIVE_RUNS:
                continue
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        if removed:
            logger.info("workflow run 目录 LRU 清理 %d 个（KEEP=%d）",
                        removed, _KEEP_MAX_RUNS)
        return removed
    except Exception as e:
        logger.debug("run 目录清理失败（fail-open）: %s", e)
        return 0


def _notify_completion(run_id: str, args: dict, kwargs: dict, out_json: str):
    """detached run 完成 → delegation 队列推后台通知（模型下轮看到）。fail-open。"""
    try:
        result = json.loads(out_json)
        from tools.delegate_tool import _resolve_delegation_queue
        _resolve_delegation_queue(kwargs).push({
            "delegation_id": run_id,
            "goal": f"workflow {args.get('name') or run_id}",
            "success": bool(result.get("ok")),
            "result": str(result.get("return") or result.get("error") or "")[:2000],
            "completed_at": time.time(),
        })
    except Exception as e:
        logger.debug("workflow 完成通知失败（fail-open）: %s", e)


def _launch_detached(run_id, run_dir, source, journal, args, kwargs, *,
                     budget_total=None, resume=False, declared_total=None) -> str:
    """C3：后台线程跑 _execute，立即返回（不阻塞主循环）。"""
    from agent import _spawn_detached

    # 先注册取消通道再起线程——否则 kill 可能赶在 _execute 注册前到达
    ev = threading.Event()
    _ACTIVE_RUNS[run_id] = ev

    async def _bg():
        try:
            out_json = await _execute(
                run_id, run_dir, source, journal, args, kwargs,
                budget_total=budget_total, resume=resume,
                declared_total=declared_total, cancel_event=ev,
            )
            _notify_completion(run_id, args, kwargs, out_json)
        except Exception as e:
            logger.warning("detached workflow %s 异常: %s", run_id, e)
            try:
                journal.save_meta({
                    "status": "failed", "error": str(e),
                    "finished_at": time.time(),
                })
            except Exception:
                pass
            _notify_completion(run_id, args, kwargs,
                               json.dumps({"ok": False, "error": str(e)}))

    _spawn_detached(_bg(), f"workflow-{run_id}")
    return json.dumps({
        "ok": True,
        "run_id": run_id,
        "detached": True,
        "status": "running",
        "hint": "已后台运行：workflow status 查进度 / workflow kill 取消；"
                "完成后以后台通知送达",
    }, ensure_ascii=False)


async def _handle_workflow(args: dict, **kwargs) -> str:
    from agent.workflow_engine import run_workflow, make_agent_runner
    from agent.workflow_journal import WorkflowJournal
    from agent.workflow_registry import load_workflow_scripts
    from constants import get_omnimate_home

    action = str(args.get("action", ""))
    config = kwargs.get("config") or {}

    if action == "run":
        source = str(args.get("script") or "")
        name = str(args.get("name") or "")
        if name:
            source = load_workflow_scripts().get(name, "")
            if not source:
                return json.dumps({
                    "error": f"workflow 不存在: {name}",
                    "error_type": "invalid_name",
                    "available": sorted(load_workflow_scripts().keys()),
                }, ensure_ascii=False)
        if not source:
            return json.dumps({"error": "script 或 name 必填其一",
                               "error_type": "invalid_args"}, ensure_ascii=False)
        run_id = f"wf_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = get_omnimate_home() / ".workflows" / run_id
        journal = WorkflowJournal.create(run_dir, source)
        _cleanup_old_runs(run_dir.parent)  # C3：KEEP_MAX_RUNS LRU 清理
        if not bool(args.get("wait", True)):  # C3：detached launch
            return _launch_detached(run_id, run_dir, source, journal, args, kwargs)
        return await _execute(run_id, run_dir, source, journal, args, kwargs)

    if action == "resume":
        run_id = str(args.get("run_id") or "")
        run_dir = get_omnimate_home() / ".workflows" / run_id
        if not (run_dir / "script.py").exists():
            return json.dumps({"error": f"run 不存在: {run_id}",
                               "error_type": "invalid_run_id"}, ensure_ascii=False)
        # 只信快照（防缓存投毒）；hash 失配在 load 内截断
        source = (run_dir / "script.py").read_text(encoding="utf-8")
        journal = WorkflowJournal.load(run_dir)
        meta = journal.load_meta()
        # 剩余额度制：总额 - 累计已花，耗尽直接拒（防多次 resume 无限烧钱）
        total = int(meta.get("budget_total") or 0)
        cumulative = int(meta.get("cumulative_spent") or 0)
        remaining = total - cumulative
        if remaining <= 0:
            return json.dumps({
                "ok": False,
                "error": f"累计预算已耗尽（{cumulative}/{total}），resume 拒绝",
                "error_type": "budget_exceeded",
                "run_id": run_id,
            }, ensure_ascii=False)
        if not bool(args.get("wait", True)):  # C3：detached resume
            return _launch_detached(run_id, run_dir, source, journal, args, kwargs,
                                    budget_total=remaining, resume=True,
                                    declared_total=total)
        return await _execute(run_id, run_dir, source, journal, args, kwargs,
                              budget_total=remaining, resume=True,
                              declared_total=total)

    if action == "status":
        run_id = str(args.get("run_id") or "")
        run_dir = get_omnimate_home() / ".workflows" / run_id
        if not run_dir.exists():
            return json.dumps({"error": f"run 不存在: {run_id}",
                               "error_type": "invalid_run_id"}, ensure_ascii=False)
        j = WorkflowJournal(run_dir)  # 只读加载——status 不得触发 hash 截断（终审修）
        meta = j.load_meta()
        return json.dumps({
            "run_id": run_id,
            "status": meta.get("status", "unknown"),
            "stats": meta.get("stats"),
            "budget_spent": meta.get("budget_spent"),
            "journal_entries": len(j),
            "active": run_id in _ACTIVE_RUNS,
        }, ensure_ascii=False)

    if action == "list":
        base = get_omnimate_home() / ".workflows"
        runs = []
        if base.exists():
            for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime,
                            reverse=True)[:20]:
                if not d.is_dir():
                    continue
                try:
                    meta = json.loads(
                        (d / "meta.json").read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                runs.append({
                    "run_id": d.name,
                    "status": meta.get("status", "unknown"),
                    "created_at": meta.get("created_at"),
                })
        return json.dumps({"runs": runs}, ensure_ascii=False)

    if action == "kill":
        run_id = str(args.get("run_id") or "")
        ev = _ACTIVE_RUNS.get(run_id)
        if ev is None:
            return json.dumps({"run_id": run_id, "killed": False,
                               "reason": "无活跃 run（可能已结束）"}, ensure_ascii=False)
        ev.set()
        return json.dumps({"run_id": run_id, "killed": True}, ensure_ascii=False)

    return json.dumps({"error": f"未知 action: {action}",
                       "error_type": "invalid_action"}, ensure_ascii=False)


async def _execute(run_id, run_dir, source, journal, args, kwargs, *,
                   budget_total=None, resume=False, declared_total=None,
                   cancel_event=None):
    """公共执行路径：注册取消通道 → 跑引擎 → meta 落盘。"""
    # 经模块属性调用（测试 monkeypatch agent.workflow_engine.make_agent_runner 可命中）
    import agent.workflow_engine as WE

    config = kwargs.get("config") or {}
    delegation = config.get("delegation") or {}
    budget = int(budget_total or args.get("budget_total")
                 or delegation.get("workflow_budget_total", 500_000))
    max_conc = int(args.get("max_concurrency")
                   or delegation.get("max_concurrent_children", 5))

    ev = cancel_event or threading.Event()  # C3：跨线程可 set；detached 预注册复用
    _ACTIVE_RUNS[run_id] = ev
    try:
        out = await WE.run_workflow(
            source,
            args=args.get("args") if isinstance(args.get("args"), dict) else {},
            agent_runner=WE.make_agent_runner(kwargs),
            # 不传全局 validator：引擎对无 schema 的 agent() 也会套 validator
            # （W1 契约），纯文本产出会被判 dead；结构化输出由引擎内建
            # _json_parseable 在 schema 调用上把关（蓝图 §5）。
            validator=None,
            budget_total=budget,
            max_concurrency=max_conc,
            journal=journal,
            cancel_event=ev,
        )
    finally:
        _ACTIVE_RUNS.pop(run_id, None)

    # 跨 resume 累计：读旧值 + 本次；declared_total 保 meta 记录的是声明总额（首跑值），
    # 不被 resume 时的剩余额覆盖
    cumulative = int(journal.load_meta().get("cumulative_spent") or 0)
    journal.save_meta({
        "status": "completed" if out.get("ok") else "failed",
        "stats": out.get("stats"),
        "budget_spent": out.get("budget_spent"),
        "budget_total": declared_total or budget,
        "cumulative_spent": cumulative + int(out.get("budget_spent") or 0),
        "error": out.get("error"),
        "finished_at": time.time(),
    })
    out["run_id"] = run_id
    if resume:
        out["resumed"] = True
    return json.dumps(out, ensure_ascii=False, default=str)


WORKFLOW_SCHEMA = {
    "name": "workflow",
    "description": (
        "确定性工作流编排：用受约束 Python 脚本（agent/parallel/pipeline/phase "
        "原语）并发驱动一批子代理，journal 断点续跑+预算封顶。"
        "DSL 写法：load_skill('workflow-dsl')。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["run", "resume", "list", "status", "kill"]},
            "name": {"type": "string",
                     "description": "workflows 目录里的脚本名（run 时与 script 二选一）"},
            "script": {"type": "string", "description": "内联脚本（含 async def main()）"},
            "run_id": {"type": "string", "description": "resume/status/kill 用"},
            "args": {"type": "object", "description": "传给脚本的参数（脚本内 args 取用）"},
            "budget_total": {"type": "integer",
                             "description": "token 预算上限（默认 delegation.workflow_budget_total=50 万）"},
            "max_concurrency": {"type": "integer", "description": "并发上限（默认 5）"},
            "wait": {"type": "boolean", "description": (
                "false=后台运行立即返回 run_id（不阻塞对话；完成后以后台通知送达，"
                "workflow status/kill 随时查进度/取消）。默认 true 前台等待")},
        },
        "required": ["action"],
    },
}

registry.register(
    name="workflow",
    toolset="core",
    schema=WORKFLOW_SCHEMA,
    handler=_handle_workflow,
    emoji="🕸️",
    isConcurrencySafe=False,  # 大量副作用（批量子代理 spawn），串行保守
)
