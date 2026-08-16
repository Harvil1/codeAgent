"""R21 S 类专项测试。

#46 auto 分类器增强（拒绝回落 + 危险前缀剥离）
#48 任务全清 + verification nudge
#23 记忆路径安全
#41 孤儿并行工具结果修复
#37 yolo 交接复审
#8 记忆检索并行 prefetch
#45 审批命令解释器
#39 粘贴引用协议
#42 全局输入历史
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_tools import ensure_tools_discovered
from tools.registry import registry

ensure_tools_discovered()

from agent.permission import (
    LLM_DENIAL_MAX_CONSECUTIVE,
    LLM_DENIAL_MAX_TOTAL,
    PermissionChecker,
    _is_dangerous_whitelist_entry,
)


# ---------------------------------------------------------------------------
# R21 #46：auto 分类器增强
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", [
    "python", "python3", "node", "npx", "bash", "sh", "ssh",
    "eval", "exec", "env", "xargs", "sudo", "curl", "wget",
    "python:*", "node:*",            # x:* 旧前缀形态
    "python -c", "npx serve",        # 首词形态
    "npm run", "npm run build",
])
def test_dangerous_whitelist_entries(entry):
    assert _is_dangerous_whitelist_entry(entry), f"应判危险: {entry}"


@pytest.mark.parametrize("entry", [
    "ls", "git status", "cat", "pip list", "echo", "", "  ",
])
def test_safe_whitelist_entries(entry):
    if entry.strip():
        assert not _is_dangerous_whitelist_entry(entry), f"不应判危险: {entry}"
    else:
        assert _is_dangerous_whitelist_entry(entry)  # 空条目排除


def _mk_classifier_checker(whitelist, verdicts):
    """构造闸门 4 全开的 checker（providers 注入 + mock aux）。"""
    calls = {"n": 0}

    class _Aux:
        async def chat_completions(self, msgs, **kw):
            calls["n"] += 1
            v = verdicts.pop(0) if verdicts else {"safe": True}
            content = json.dumps(v)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
            )])

    checker = PermissionChecker()
    checker.set_aux_llm_provider(lambda: _Aux())
    checker.set_config_provider(lambda: {
        "features": {"bash_llm_classifier": {"enabled": True, "whitelist": whitelist}},
    })
    return checker, calls


def test_dangerous_prefix_stripped_from_whitelist():
    """白名单含 python → python 命令不走快速通道（仍走 LLM 分类）。"""
    checker, calls = _mk_classifier_checker(
        ["ls", "python"], [{"safe": True}],
    )
    r = checker.check("python -c 'print(1)'")
    assert calls["n"] == 1  # 没被白名单快速通道跳过，走了 LLM
    assert r.allowed and r.gate == "llm_safe"
    # 安全前缀仍走快速通道（0 LLM）——用非只读命令（ls 会被 T7 只读通道先截走）
    checker2, calls2 = _mk_classifier_checker(["pip install"], [])
    r2 = checker2.check("pip install requests")
    assert r2.gate == "whitelist"
    assert calls2["n"] == 0


def test_denial_fallback_consecutive():
    """连续 3 次判 unsafe → 闸门 4 停用（后续 0 LLM 调用，默认通过）。"""
    checker, calls = _mk_classifier_checker(
        [], [{"safe": False, "reason": "x"}, {"safe": False, "reason": "x"},
             {"safe": False, "reason": "x"}, {"safe": False, "reason": "x"}],
    )
    for _ in range(3):
        r = checker.check("some command")
        assert not r.allowed and r.gate == "llm_unsafe"
    assert checker._llm_denial_consecutive == LLM_DENIAL_MAX_CONSECUTIVE
    # 第 4 次：闸门 4 已停用 → 0 新 LLM 调用 → 默认通过
    r4 = checker.check("some command")
    assert r4.allowed and r4.gate == "ok"
    assert calls["n"] == 3


def test_denial_total_fallback():
    """累计 20 次拒绝（穿插 safe 重置连续计数，连续阈值不触发）→ 停用。"""
    verdicts = []
    for _ in range(10):
        verdicts.extend([{"safe": False, "reason": "x"}, {"safe": False, "reason": "x"},
                         {"safe": True}])
    checker, calls = _mk_classifier_checker([], verdicts)
    for _ in range(30):
        checker.check("cmd")
    assert checker._llm_denial_total == 20
    assert checker._llm_denial_total >= LLM_DENIAL_MAX_TOTAL
    # 第 29 次 unsafe 后 total=20 → 闸门停用；第 30 次不再调 LLM
    # （verdicts 里的 safe 没被消费）→ consecutive 保持 2
    assert checker._llm_denial_consecutive == 2
    # 后续 0 新调用
    n_before = calls["n"]
    checker.check("cmd")
    assert calls["n"] == n_before


# ---------------------------------------------------------------------------
# R21 #48：任务全清 + verification nudge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_all_done_cleanup_and_nudge(tmp_path):
    """全 completed → 自动软删清空；≥3 活跃无验证字样 → reminder。"""
    from agent.task_store import get_task_store

    class _FakeAgent:
        hooks_registry = None
    home = str(tmp_path)
    store = get_task_store(home)
    t1 = store.create(subject="做 A", description="实现 A")["id"]
    t2 = store.create(subject="做 B", description="实现 B")["id"]
    t3 = store.create(subject="做 C", description="实现 C")["id"]

    # 完成 2 个（剩 1 个活跃 → 无 nudge 无清空）
    r1 = json.loads(await registry.dispatch("task_complete", {"id": t1},
                                            omnimate_home=home, agent_ref=_FakeAgent()))
    assert "all_tasks_cleared" not in r1
    assert "reminder" not in r1

    # 完成第 3 个 → 全 completed → all_done 标志（状态保留可查）
    await registry.dispatch("task_complete", {"id": t2},
                            omnimate_home=home, agent_ref=_FakeAgent())
    r3 = json.loads(await registry.dispatch("task_complete", {"id": t3},
                                            omnimate_home=home, agent_ref=_FakeAgent()))
    assert r3.get("all_done") is True
    assert r3.get("total") == 3
    # completed 状态持久可查（不软删——保持既有语义）
    statuses = {t["status"] for t in store.list_all()}
    assert statuses == {"completed"}
    assert store.list_all(status="pending") == []


@pytest.mark.asyncio
async def test_task_verification_nudge(tmp_path):
    """≥3 活跃且无验证字样 → reminder；描述含验证则不提醒。"""
    from agent.task_store import get_task_store

    class _FakeAgent:
        hooks_registry = None
    home = str(tmp_path)
    store = get_task_store(home)
    done = store.create(subject="D", description="已完成项")["id"]
    for i in range(3):
        store.create(subject=f"任务{i}", description=f"实现功能{i}")
    # 完成 done（剩 3 条活跃无验证字样）→ reminder 出现
    r = json.loads(await registry.dispatch("task_complete", {"id": done},
                                           omnimate_home=home, agent_ref=_FakeAgent()))
    assert "reminder" in r and "验证" in r["reminder"]

    # 描述含"测试"的活跃任务集 → 不提醒
    home2 = str(tmp_path / "s2")
    store2 = get_task_store(home2)
    d2 = store2.create(subject="D", description="x")["id"]
    for i in range(3):
        store2.create(subject=f"任务{i}", description=f"实现并测试功能{i}")
    r2 = json.loads(await registry.dispatch("task_complete", {"id": d2},
                                            omnimate_home=home2, agent_ref=_FakeAgent()))
    assert "reminder" not in r2


# ---------------------------------------------------------------------------
# R21 #23：记忆路径安全
# ---------------------------------------------------------------------------

from agent.memory_store import MemoryStore, validate_memory_dir


def test_validate_memory_dir_rules(tmp_path):
    """受保护路径拒绝 + 越界拒绝 + 正常路径通过。"""
    # 正常：home/.memory
    assert validate_memory_dir(tmp_path / ".memory", tmp_path) is None

    # 受保护路径（词法形式）
    assert validate_memory_dir(Path.home() / ".ssh" / "mem", tmp_path) is not None
    # 越界（realpath 不在 home 下）
    assert validate_memory_dir(Path("C:/elsewhere/mem"), tmp_path) is not None


def test_validate_memory_dir_symlink_escape(tmp_path):
    """.memory 是指向受保护路径的软链 → realpath 形式拒。"""
    link_dir = tmp_path / "evil"
    try:
        link_dir.symlink_to(Path.home() / ".ssh")
    except (OSError, NotImplementedError):
        pytest.skip("本环境无法创建符号链接")
    reason = validate_memory_dir(link_dir, tmp_path)
    assert reason is not None


def test_memory_store_rejects_bad_dir(tmp_path):
    """构造时校验失败 → ValueError（fail-closed）。"""
    with pytest.raises(ValueError, match="安全校验失败"):
        MemoryStore(omnimate_home=tmp_path, memory_dir=Path.home() / ".ssh" / "mem")
    # 越界目录同样拒
    with pytest.raises(ValueError):
        MemoryStore(omnimate_home=tmp_path, memory_dir=tmp_path.parent / "other")
    # 正常自定义目录（home 内）通过
    s = MemoryStore(omnimate_home=tmp_path, memory_dir=tmp_path / "custom_mem")
    assert s._memory_dir.name == "custom_mem"


# ---------------------------------------------------------------------------
# R21 #41：孤儿并行工具结果修复（resume 加载时）
# ---------------------------------------------------------------------------

def test_fix_tool_call_pairs_repair_semantics():
    """孤儿修复语义（正向补漏 + 反向删除），resume 复用同一函数。"""
    from agent.context_compressor import _fix_tool_call_pairs

    msgs = [
        {"role": "user", "content": "q"},
        # 正向孤儿：assistant 有 tool_calls 但 result 缺失
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        # 只有 c1 的 result，c2 悬空（并行批次保存中断）
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": 1}'},
        # 反向孤儿：无对应 tool_calls
        {"role": "tool", "tool_call_id": "cX", "content": '{"dangling": 1}'},
        {"role": "assistant", "content": "done"},
    ]
    fixed = _fix_tool_call_pairs(msgs)
    ids = [m.get("tool_call_id") for m in fixed if m.get("role") == "tool"]
    assert "c1" in ids and "c2" in ids   # c2 被补漏
    assert "cX" not in ids               # 反向孤儿被删
    # 补的 result 紧跟在 c1 后（immediately-after 语义）
    idx_c1 = next(i for i, m in enumerate(fixed) if m.get("tool_call_id") == "c1")
    assert fixed[idx_c1 + 1].get("tool_call_id") == "c2"


# ---------------------------------------------------------------------------
# R21 #37：yolo 交接复审
# ---------------------------------------------------------------------------

def test_review_handoff_flags_dangerous():
    """aux 判危险 → 结果前附警告；判安全/失败/短文本 → 原文。"""
    from tools.delegate_tool import _review_handoff

    class _Aux:
        def __init__(self, verdict):
            self.verdict = verdict

        async def chat_completions(self, msgs, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(self.verdict),
            ))])

    long_result = "子代理执行完成。" + "x" * 300

    # 判危险 → 警告前缀
    out = _review_handoff(long_result, SimpleNamespace(
        aux_llm_router=_Aux({"dangerous": True, "warning": "删除了系统目录"}),
    ))
    assert out.startswith("[⚠ 交接复审警告]")
    assert long_result in out

    # 判安全 → 原文
    out2 = _review_handoff(long_result, SimpleNamespace(
        aux_llm_router=_Aux({"dangerous": False}),
    ))
    assert out2 == long_result

    # 无 aux / 短文本 → 原文（不调用）
    assert _review_handoff(long_result, SimpleNamespace(aux_llm_router=None)) == long_result
    assert _review_handoff("短结果", SimpleNamespace(aux_llm_router=object())) == "短结果"

    # aux 抛异常 → 原文（fail-open）
    class _Broken:
        async def chat_completions(self, msgs, **kw):
            raise RuntimeError("aux down")
    assert _review_handoff(long_result, SimpleNamespace(aux_llm_router=_Broken())) == long_result


def test_handoff_review_config_default():
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["delegation"]["handoff_review_enabled"] is False


# ---------------------------------------------------------------------------
# R21 #8：记忆检索并行 prefetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_prefetch_consume():
    """prefetch task 被 _consume_memory_prefetch 一次性消费（append + 清空）。"""
    from agent import AIAgent

    async def fake_fetch():
        return {"role": "user", "content": "<relevant_memories>x</relevant_memories>"}

    a = AIAgent.__new__(AIAgent)
    a._memory_prefetch_task = None
    # 无 task → 原样
    out = await a._consume_memory_prefetch([{"role": "user", "content": "q"}])
    assert len(out) == 1

    import asyncio
    a._memory_prefetch_task = asyncio.create_task(fake_fetch())
    msgs = [{"role": "user", "content": "q"}]
    out2 = await a._consume_memory_prefetch(msgs)
    assert len(out2) == 2
    assert "relevant_memories" in out2[-1]["content"]
    assert a._memory_prefetch_task is None  # 已清（一次性）
    # 第二次消费：无 task 不追加
    out3 = await a._consume_memory_prefetch(out2)
    assert len(out3) == 2

    # task 异常 → fail-open 原样返回
    async def boom():
        raise RuntimeError("aux down")
    a._memory_prefetch_task = asyncio.create_task(boom())
    out4 = await a._consume_memory_prefetch([{"role": "user", "content": "q"}])
    assert len(out4) == 1
    assert a._memory_prefetch_task is None

    # task 返回 None → 不追加
    async def none_fetch():
        return None
    a._memory_prefetch_task = asyncio.create_task(none_fetch())
    out5 = await a._consume_memory_prefetch([{"role": "user", "content": "q"}])
    assert len(out5) == 1


# ---------------------------------------------------------------------------
# R21 #42：全局输入历史 + R21 #39：粘贴引用协议
# ---------------------------------------------------------------------------

from agent.input_history import (
    HISTORY_LIMIT,
    GlobalHistory,
    expand_paste_references,
    store_paste_if_large,
)


def test_global_history(tmp_path):
    h = GlobalHistory(tmp_path)
    assert h.recent() == []
    h.append("第一条")
    h.append("第二条")
    h.append("第二条")  # 与最近一条相同 → 不记
    assert h.recent(10) == ["第二条", "第一条"]
    assert h.get(1) == "第二条"
    assert h.get(2) == "第一条"
    assert h.get(3) is None  # 越界
    # 空文本不记
    h.append("   ")
    assert len(h.recent(10)) == 2

    # 跨实例（跨会话语义）
    h2 = GlobalHistory(tmp_path)
    assert h2.recent(1) == ["第二条"]


def test_global_history_trim(tmp_path):
    """超 2 倍上限触发软裁剪（保留最新 HISTORY_LIMIT 条 + 其后新增）。"""
    h = GlobalHistory(tmp_path)
    for i in range(HISTORY_LIMIT * 2 + 5):
        h.append(f"msg-{i}")
    items = h.recent(HISTORY_LIMIT + 10)
    # 裁剪发生在第 201 条（留 100），其后 4 次新增 → 104
    assert len(items) == HISTORY_LIMIT + 4
    assert items[0] == f"msg-{HISTORY_LIMIT * 2 + 4}"  # 最新在前


def test_store_paste_threshold(tmp_path):
    """≤1024 原样；>1024 外存 + 占位符；可往返展开。"""
    short = "短输入"
    text, path = store_paste_if_large(short, tmp_path)
    assert text == short and path is None

    long_text = "x" * 1025 + "\nsecond line"
    text2, path2 = store_paste_if_large(long_text, tmp_path)
    assert path2 is not None
    assert text2.startswith("[Pasted text #1 +2 lines]")
    # 展开往返
    expanded = expand_paste_references(text2, tmp_path)
    assert expanded == long_text
    # 第二次外存编号递增
    text3, _ = store_paste_if_large("y" * 2000, tmp_path)
    assert "#2" in text3


def test_expand_paste_missing_file(tmp_path):
    """占位符无对应文件 → 保留占位符（fail-open）。"""
    out = expand_paste_references("[Pasted text #99 +5 lines]", tmp_path)
    assert out == "[Pasted text #99 +5 lines]"
    # 无占位符文本原样
    assert expand_paste_references("普通消息", tmp_path) == "普通消息"


# ---------------------------------------------------------------------------
# R22 #27：TF-IDF 技能搜索
# ---------------------------------------------------------------------------

from tools.skill_tools import _skill_search_rank, _tokenize


def test_tokenize_mixed():
    """英文连字符词组 + 中文单字；停用词滤除。"""
    toks = _tokenize("How to deploy 这个项目 with Docker")
    assert "deploy" in toks
    assert "docker" in toks
    assert "how" not in toks and "to" not in toks and "with" not in toks
    assert "个" not in toks  # 中文停用字


def test_skill_search_rank():
    skills = {
        "docker-deploy": {"name": "docker-deploy", "description": "用 Docker 部署项目的流程"},
        "code-review": {"name": "code-review", "description": "审查代码变更"},
        "cherry-pick": {"name": "cherry-pick", "description": "git cherry-pick 到 release 分支"},
        "skillify": {"name": "skillify", "description": "把会话流程沉淀为技能"},
    }
    # 中英混合查询命中
    assert [s["name"] for s in _skill_search_rank(skills, "怎么用 docker 部署")] == ["docker-deploy"]
    assert [s["name"] for s in _skill_search_rank(skills, "审查代码")] == ["code-review"]
    # 英文查询
    assert [s["name"] for s in _skill_search_rank(skills, "docker deploy")] == ["docker-deploy"]
    # 不相关查询 → 空
    assert _skill_search_rank(skills, "数据库迁移") == []
    # 空 query → 空
    assert _skill_search_rank(skills, "") == []
    # name 加权：技能名命中 > 描述命中
    skills2 = {
        "a": {"name": "a", "description": "release 发布流程"},
        "b": {"name": "release-notes", "description": "x"},
    }
    ranked = _skill_search_rank(skills2, "release")
    assert ranked[0]["name"] == "release-notes"


@pytest.mark.asyncio
async def test_skills_list_query_dispatch(tmp_path):
    """dispatch 集成：query 命中返回排序（第一是目标）；无 query 全量（回归）。

    注：自定义 home 仍包含内置技能目录（设计如此），断言用首位而非唯一性。
    """
    sd = tmp_path / "skills"
    (sd / "docker-deploy").mkdir(parents=True)
    (sd / "docker-deploy" / "SKILL.md").write_text(
        '---\nname: docker-deploy\ndescription: "用 Docker 部署项目的流程"\n---\n正文',
        encoding="utf-8",
    )
    (sd / "code-review").mkdir(parents=True)
    (sd / "code-review" / "SKILL.md").write_text(
        '---\nname: code-review\ndescription: "审查代码变更"\n---\n正文',
        encoding="utf-8",
    )
    r = json.loads(await registry.dispatch(
        "skills_list", {"query": "docker 部署"},
        omnimate_home=str(tmp_path),
    ))
    assert r["matched"] >= 1
    assert r["skills"][0]["name"] == "docker-deploy"  # name 命中排首位

    # 无 query：全量（含自定义 + 内置 ≥2）
    r3 = json.loads(await registry.dispatch(
        "skills_list", {}, omnimate_home=str(tmp_path),
    ))
    assert len(r3["skills"]) >= 2


# ---------------------------------------------------------------------------
# R22 #11：队列命令消费（输入排队 + 工具批后回流）
# ---------------------------------------------------------------------------

def _mk_queue_agent(queue_obj=None, depth=0):
    from agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    a._input_queue = queue_obj
    a.spawn_depth = depth
    a._pending_ephemeral_messages = []
    return a


def test_drain_queued_input():
    """队列输入 → 合并 ephemeral 回流；空队列/无队列 no-op；子代理不回流。"""
    import queue as _q

    q = _q.Queue()
    a = _mk_queue_agent(q)
    q.put("先看这个")
    q.put("还有这个")
    a._drain_queued_input()
    assert len(a._pending_ephemeral_messages) == 1
    msg = a._pending_ephemeral_messages[0]
    assert "queued_user_input" in msg["content"]
    assert "先看这个" in msg["content"] and "还有这个" in msg["content"]
    assert msg["_ephemeral"] is True
    # 队列已清（再 drain no-op）
    a._drain_queued_input()
    assert len(a._pending_ephemeral_messages) == 1

    # 无队列 no-op
    b = _mk_queue_agent(None)
    b._drain_queued_input()
    assert b._pending_ephemeral_messages == []

    # 子代理（spawn_depth>0）不回流
    q2 = _q.Queue()
    q2.put("x")
    c = _mk_queue_agent(q2, depth=1)
    c._drain_queued_input()
    assert c._pending_ephemeral_messages == []

    # 纯空白输入被滤
    q3 = _q.Queue()
    q3.put("   ")
    d = _mk_queue_agent(q3)
    d._drain_queued_input()
    assert d._pending_ephemeral_messages == []


# ---------------------------------------------------------------------------
# R22 #32：Monitor 监视器
# ---------------------------------------------------------------------------

def test_monitor_mode_tee_and_no_stall(tmp_path):
    """monitor：stdout tee 落盘 + 豁免 stall 通知；退出照常通知。"""
    import time as _time
    from agent.background import BackgroundManager

    mgr = BackgroundManager(stall_timeout=45.0)
    # 轮询型"监视器"：每 0.2s 输出一行，1s 后退出（stall 45s 永不触发；
    # 常规任务若真停滞才会触发——这里 monitor 豁免逻辑靠 exit 验证不误报）
    tid = mgr.start(
        ["python", "-c",
         "import time\nfor i in range(5):\n    print(f'line {i}', flush=True)\n    time.sleep(0.15)"],
        timeout=30, monitor=True, monitor_dir=tmp_path,
    )
    # 等完成（轮询通知）
    deadline = _time.time() + 15
    notifications = []
    while _time.time() < deadline:
        notifications.extend(mgr.drain_notifications())
        task = mgr.status(tid)
        if task and task.status in ("completed", "failed"):
            break
        _time.sleep(0.2)
    notifications.extend(mgr.drain_notifications())
    task = mgr.status(tid)
    assert task.status == "completed"
    assert task.monitor is True
    # tee 文件含全部输出（不受 result cap 影响）
    assert task.output_file and Path(task.output_file).exists()
    tee_text = Path(task.output_file).read_text(encoding="utf-8")
    assert "line 0" in tee_text and "line 4" in tee_text
    # 退出通知（无 stall 标志）
    final = [n for n in notifications if n.get("task_id") == tid]
    assert final and final[-1].get("status") == "completed"
    assert not final[-1].get("stall")


def test_monitor_quiet_no_stall_notification(tmp_path):
    """monitor 安静期（无输出）不触发 stall 通知（安静是常态）。"""
    import time as _time
    from agent.background import BackgroundManager

    # stall_timeout 极短（0.5s）+ monitor 任务 1.2s 无输出——豁免则无 stall 通知
    mgr = BackgroundManager(stall_timeout=0.5)
    tid = mgr.start(
        ["python", "-c", "import time; time.sleep(1.2); print('done', flush=True)"],
        timeout=30, monitor=True, monitor_dir=tmp_path,
    )
    deadline = _time.time() + 15
    notifications = []
    while _time.time() < deadline:
        notifications.extend(mgr.drain_notifications())
        task = mgr.status(tid)
        if task and task.status in ("completed", "failed"):
            break
        _time.sleep(0.2)
    notifications.extend(mgr.drain_notifications())
    assert mgr.status(tid).status == "completed"
    stalls = [n for n in notifications if n.get("stall")]
    assert stalls == []  # monitor 豁免——同场景常规任务必报 stall


# ---------------------------------------------------------------------------
# R22 #40：compact 边界重链
# ---------------------------------------------------------------------------

def test_truncate_at_last_compact_boundary():
    """最后边界之前的旧消息裁掉；标记行剥掉保留摘要正文；无标记保守全量。"""
    from cli import _truncate_at_last_compact_boundary as trunc

    msgs = [
        {"role": "user", "content": "旧消息 1"},
        {"role": "assistant", "content": "旧回复"},
        {"role": "user", "content": "[COMPACT_BOUNDARY]\n[之前的对话已自动总结]\n\n摘要 A\n\n[以下是最近的对话]"},
        {"role": "user", "content": "新消息 1"},
        {"role": "assistant", "content": "新回复"},
        {"role": "user", "content": "[COMPACT_BOUNDARY]\n[之前的对话已自动总结]\n\n摘要 B\n\n[以下是最近的对话]"},
        {"role": "user", "content": "边界后消息"},
    ]
    out = trunc(msgs)
    # 从最后边界起（前 5 条丢弃：旧消息×2 + 第一边界 + 新消息×2）
    assert len(out) == 2
    assert out[0]["content"].startswith("[之前的对话已自动总结]")
    assert "摘要 B" in out[0]["content"]
    assert "[COMPACT_BOUNDARY]" not in out[0]["content"]  # 标记行已剥
    assert out[1]["content"] == "边界后消息"
    # 入参不被修改（浅拷贝保护）
    assert msgs[5]["content"].startswith("[COMPACT_BOUNDARY]")

    # 无标记 → 原样（旧会话保守路径）
    plain = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert trunc(plain) == plain


# ---------------------------------------------------------------------------
# R24 #26：动态技能目录发现
# ---------------------------------------------------------------------------

from agent.skill_commands import discover_skill_dirs_for_path


def test_discover_skill_dirs_nested(tmp_path):
    """深路径向上发现嵌套 .omnimate/skills；node_modules 投放跳过；cwd 外不适用。"""
    api_skills = tmp_path / "services" / "api" / ".omnimate" / "skills"
    api_skills.mkdir(parents=True)
    root_skills = tmp_path / ".claude" / "skills"
    root_skills.mkdir(parents=True)
    poisoned = tmp_path / "node_modules" / "pkg" / ".omnimate" / "skills"
    poisoned.mkdir(parents=True)

    deep = tmp_path / "services" / "api" / "src" / "main.py"
    found = discover_skill_dirs_for_path(str(deep), str(tmp_path))
    # 深路径优先：api 的目录在前，root 的 .claude/skills 也在链上
    assert found[0] == api_skills
    assert root_skills in found
    # 投放目录不收
    assert poisoned not in found

    # 文件在 cwd 外 → 空
    assert discover_skill_dirs_for_path("C:/elsewhere/x.py", str(tmp_path)) == []
    # 空 path → 空
    assert discover_skill_dirs_for_path("", str(tmp_path)) == []


def test_nested_dir_feeds_conditional_match(tmp_path):
    """嵌套目录里的条件技能可被 find_conditional_skill_matches 发现。"""
    import agent.skill_commands as sc
    sc._fm_summary_cache.clear()
    nested = tmp_path / "packages" / "web" / ".omnimate" / "skills"
    d = nested / "react-helper"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        '---\nname: react-helper\ndescription: "React 组件规范"\npaths: ["*.tsx"]\n---\n正文',
        encoding="utf-8",
    )
    from agent.workspace_context import workspace_cwd_context
    with workspace_cwd_context(str(tmp_path)):
        hits = sc.find_conditional_skill_matches(
            str(tmp_path / "packages" / "web" / "src" / "App.tsx"),
        )
        names = [h["name"] for h in hits]
        assert "react-helper" in names, names


# ---------------------------------------------------------------------------
# R24 #36：scratchpad 涂鸦区
# ---------------------------------------------------------------------------

from agent.scratchpad import (
    cleanup_old_scratchpads,
    ensure_scratchpad,
    scratchpad_context_block,
    scratchpad_dir,
)


def test_scratchpad_flow(tmp_path):
    """ensure 创建目录 + 白名单注入；context block 含路径；旧目录清理。"""
    from agent.permission import clear_extra_allowed_roots, list_extra_allowed_roots
    clear_extra_allowed_roots()
    try:
        d = ensure_scratchpad("sess-abc", tmp_path)
        assert d is not None and d.is_dir()
        assert d == tmp_path / ".scratchpad" / "sess-abc"
        # 白名单已注入（运行时）
        assert any(str(d) in str(r) for r in list_extra_allowed_roots())
        # context block 含路径与说明
        block = scratchpad_context_block("sess-abc", tmp_path)
        assert str(d) in block and "免权限" in block
    finally:
        clear_extra_allowed_roots()

    # 清理：旧目录（mtime 老）删、新目录留
    import os
    old = tmp_path / ".scratchpad" / "old-sess"
    old.mkdir(parents=True)
    (old / "note.txt").write_text("x", encoding="utf-8")
    os.utime(old, (0, 0))  # epoch = 必过期
    keep = ensure_scratchpad("new-sess", tmp_path)
    removed = cleanup_old_scratchpads(tmp_path, retention_days=7)
    assert not old.exists()
    assert keep is not None and keep.is_dir()


def test_coordinator_builtin_agent():
    """内置 coordinator 子代理存在（纯编排 + scratchpad 说明）。"""
    from agent.agent_defs import scan_agent_defs
    defs = scan_agent_defs() or {}
    d = defs.get("coordinator") if isinstance(defs, dict) else None
    assert d is not None, "coordinator 应在 scan_agent_defs 里"
    body = getattr(d, "prompt", "") or getattr(d, "system_prompt", "") or ""
    # 拿不到 body 就读文件
    if "Scratchpad" not in body:
        from agent.agent_defs import _builtin_agents_dir
        body = (_builtin_agents_dir() / "coordinator.md").read_text(encoding="utf-8")
    assert "Scratchpad" in body and "编排" in body
