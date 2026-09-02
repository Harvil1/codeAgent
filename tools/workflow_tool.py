"""workflow 工具：确定性工作流引擎暴露给 LLM 的入口。

workflow 引擎（agent/workflow_engine.py）能用受约束的 Python 脚本
并发驱动一批子代理，本文件把它包装成一个工具给 LLM 调。共五个 action：
run（name=脚本名 或 script=内联代码）/ resume（只信 run 目录里的快照）/
list / status / kill。DSL 具体写法见 skills/workflow-dsl/SKILL.md
（窄腰原则：工具 description 只写一句话，详细文档让 LLM 按需 load_skill）。

run/resume 支持 wait=false——
放到后台线程 + 独立事件循环里跑，立刻返回 run_id 不堵住对话主循环；
跑完后经 delegation 队列以后台通知送达；同会话内可以 kill
（取消通道用 threading.Event，因为要跨线程触发）。run 目录超过
KEEP_MAX_RUNS=50 个时按"最久未用先删"清理。
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

# run_id -> threading.Event（正在跑的 run 的取消开关；只存在进程内存里）。
# 取舍：用 threading.Event 而不是 asyncio.Event——detached run
# 跑在后台线程的事件循环里，主循环线程要能跨线程 set 它
# （asyncio.Event 不能跨线程/跨事件循环用）
_ACTIVE_RUNS: dict = {}

# run 目录数量上限（按修改时间从旧到新删，跳过还在跑的）
_KEEP_MAX_RUNS = 50


def _cleanup_old_runs(base) -> int:
    """run 目录超过上限时删掉最旧的一批。

    run 目录会越攒越多，按"最久没动过先删"（LRU）控制总量；
    还在跑的 run 绝不能删。删失败了也不报错（fail-open）。

    参数：
        base: 存 run 目录的根路径（~/.codeAgent/.workflows）

    返回：
        实际删掉的目录数
    """
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
    """后台 run 跑完后，把结果推进 delegation 队列当后台通知（模型下一轮能看到）。

    detached run 不在主循环里等结果，靠通知机制把结果送回来。
    通知失败只记日志不报错（fail-open）。
    """
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
    """把 run 丢到后台线程跑，立刻返回"已启动"的 JSON（不堵主循环）。

    参数：
        run_id: 本次 run 的 ID
        run_dir: 本次 run 的目录（存脚本快照和 journal）
        source: workflow 脚本内容
        journal: 执行日志/断点对象（WorkflowJournal）
        args: LLM 传入的工具参数
        kwargs: dispatch 透传的上下文
        budget_total: token 预算上限（resume 时是剩余额度）
        resume: 是否断点续跑
        declared_total: 首跑时声明的总预算（resume 记账用）

    返回：
        JSON 字符串：{"ok": true, "run_id": ..., "detached": true, ...}
    """
    from agent import _spawn_detached

    # 顺序不能反：必须先注册取消通道再起线程——否则 kill 可能赶在
    # _execute 注册之前到达，变成"杀了个空气"
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
    """工具 handler：按 action 分发到 run / resume / status / list / kill 五条路。

    workflow 的所有 LLM 入口都走这里，五个 action 的详细语义见文件头说明。

    参数：
        args: LLM 传入的工具参数（action / name / script / run_id / wait 等）
        **kwargs: dispatch 透传的上下文（config / agent_ref 等）

    返回：
        JSON 字符串（各 action 的结果或错误）
    """
    from agent.workflow_engine import run_workflow, make_agent_runner
    from agent.workflow_journal import WorkflowJournal
    from agent.workflow_registry import load_workflow_scripts
    from constants import get_codeagent_home

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
        run_dir = get_codeagent_home() / ".workflows" / run_id
        journal = WorkflowJournal.create(run_dir, source)
        _cleanup_old_runs(run_dir.parent)  # 超上限时 LRU 清理旧目录
        if not bool(args.get("wait", True)):  # 后台模式立即返回
            return _launch_detached(run_id, run_dir, source, journal, args, kwargs)
        return await _execute(run_id, run_dir, source, journal, args, kwargs)

    if action == "resume":
        run_id = str(args.get("run_id") or "")
        run_dir = get_codeagent_home() / ".workflows" / run_id
        if not (run_dir / "script.py").exists():
            return json.dumps({"error": f"run 不存在: {run_id}",
                               "error_type": "invalid_run_id"}, ensure_ascii=False)
        # 只信磁盘快照，不信任任何缓存（防投毒）；脚本 hash 对不上时在 load 内截断
        source = (run_dir / "script.py").read_text(encoding="utf-8")
        journal = WorkflowJournal.load(run_dir)
        meta = journal.load_meta()
        # 剩余额度制：还能花的 = 总额度 - 历次累计已花，花完直接拒
        # （不然反复 resume 可以无限烧 token）
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
        if not bool(args.get("wait", True)):  # 后台模式 resume
            return _launch_detached(run_id, run_dir, source, journal, args, kwargs,
                                    budget_total=remaining, resume=True,
                                    declared_total=total)
        return await _execute(run_id, run_dir, source, journal, args, kwargs,
                              budget_total=remaining, resume=True,
                              declared_total=total)

    if action == "status":
        run_id = str(args.get("run_id") or "")
        run_dir = get_codeagent_home() / ".workflows" / run_id
        if not run_dir.exists():
            return json.dumps({"error": f"run 不存在: {run_id}",
                               "error_type": "invalid_run_id"}, ensure_ascii=False)
        # 只读加载——status 绝不能触发 hash 截断副作用
        j = WorkflowJournal(run_dir)
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
        base = get_codeagent_home() / ".workflows"
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
    """前台/后台共用的执行路径：注册取消开关 → 跑引擎 → 结果落盘 meta。

    参数：
        run_id: 本次 run 的 ID
        run_dir: run 目录（journal 写在这里）
        source: workflow 脚本内容
        journal: 断点/日志对象
        args: LLM 传入的工具参数
        kwargs: dispatch 透传上下文（子代理 runner 从这里构造）
        budget_total: token 预算上限（None 时按默认链兜底）
        resume: 是否续跑
        declared_total: 首跑声明的总预算（保账本口径用）
        cancel_event: 已注册的取消开关（detached 模式复用，前台自建）

    返回：
        JSON 字符串（引擎结果 + run_id，resume 时附 resumed=true）
    """
    # 必须经模块属性调用：这样测试 monkeypatch agent.workflow_engine.
    # make_agent_runner 才能命中（直接 from-import 会拷贝引用，patch 不上）
    import agent.workflow_engine as WE

    config = kwargs.get("config") or {}
    delegation = config.get("delegation") or {}
    budget = int(budget_total or args.get("budget_total")
                 or delegation.get("workflow_budget_total", 500_000))
    max_conc = int(args.get("max_concurrency")
                   or delegation.get("max_concurrent_children", 5))

    ev = cancel_event or threading.Event()  # 要跨线程可 set；detached 复用预注册的
    _ACTIVE_RUNS[run_id] = ev
    try:
        out = await WE.run_workflow(
            source,
            args=args.get("args") if isinstance(args.get("args"), dict) else {},
            agent_runner=WE.make_agent_runner(kwargs),
            # 故意不传全局 validator：引擎对没写 schema 的 agent() 也会套
            # validator，纯文本产出会被误判 dead；结构化输出由引擎内建的
            # _json_parseable 在有 schema 的调用上把关
            validator=None,
            budget_total=budget,
            max_concurrency=max_conc,
            journal=journal,
            cancel_event=ev,
        )
    finally:
        _ACTIVE_RUNS.pop(run_id, None)

    # 跨 resume 累计记账：旧值 + 本次花费。declared_total 保证 meta 里记的
    # 是"声明总额"（首跑时的值），不被 resume 时的剩余额度覆盖掉
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
    isConcurrencySafe=False,  # 副作用很重（成批 spawn 子代理），保守串行
)
