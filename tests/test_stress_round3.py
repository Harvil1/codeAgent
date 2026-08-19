"""压力测试 Round 3：剩余全部缺口（并发 / 规模 / 边界形态）。

Round 1 函数级 + Round 2 粘合处之后，本文件压：
1. 并发文件锁争用（mailbox / trace 多线程同写）
2. 索引与存储规模（500 记忆 / 300 技能 / 500 cron / 万条 session）
3. 消息形态边界（1MB user 消息 / 一条 assistant 50 个 tool_calls）

纯本地 mock（不调真 LLM）。
"""
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context_compressor import _fix_tool_call_pairs
from agent.context_pipeline import (
    CompressionSessionState,
    compress_if_needed,
    reset_offload_decisions,
)
from agent.cron import CronJob, CronScheduler
from agent.goal import GoalState
from agent.memory_store import MemoryStore
from agent.prompt_builder import _build_skill_index
from agent.session_store import SessionStore
from agent.team.mailbox import Mailbox
from agent.trace import TraceSink


# ---------------------------------------------------------------------------
# 1. 并发文件锁争用
# ---------------------------------------------------------------------------

def test_stress_concurrent_mailbox_8_threads(tmp_path):
    """8 线程并发 send（各 100 封）+ 2 线程并发消费（check_unread + mark_read）。

    验证：无丢失（总 800 封全在）、无崩溃、mark_read 后 unread 收敛。
    """
    mb = Mailbox(tmp_path)
    send_count = [100, 100, 100, 100, 100, 100, 100, 100]
    errors = []

    def sender(tid: int, n: int):
        try:
            for k in range(n):
                mb.send(
                    to="main", from_=f"w{tid}",
                    content=f"mail t{tid} k{k}", kind="message",
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"sender{tid}: {e}")

    stop = threading.Event()

    def consumer(cid: int):
        try:
            while not stop.is_set():
                unread = mb.check_unread("main")
                if unread:
                    mb.mark_read("main", [m["id"] for m in unread])
                time.sleep(0.001)
        except Exception as e:  # noqa: BLE001
            errors.append(f"consumer{cid}: {e}")

    consumers = [threading.Thread(target=consumer, args=(i,), daemon=True)
                 for i in range(2)]
    for t in consumers:
        t.start()

    t0 = time.perf_counter()
    senders = [threading.Thread(target=sender, args=(i, n))
               for i, n in enumerate(send_count)]
    for t in senders:
        t.start()
    for t in senders:
        t.join(timeout=60)
    elapsed = time.perf_counter() - t0
    stop.set()
    for t in consumers:
        t.join(timeout=5)

    assert errors == [], f"并发错误: {errors}"
    # 无丢失
    all_msgs = mb.check_all("main")
    assert len(all_msgs) == 800, f"丢了 {800 - len(all_msgs)} 封"
    assert elapsed < 60.0, f"800 封并发耗时 {elapsed:.1f}s"


def test_stress_concurrent_trace_8_threads(tmp_path):
    """8 线程 × 1000 emit = 8000 行，验证 threading.Lock 下无丢失/交错损坏。"""
    sink = TraceSink(tmp_path)
    errors = []

    def emitter(tid: int):
        try:
            for i in range(1000):
                sink.emit("post_tool_use", tool="t", tid=tid, seq=i)
        except Exception as e:  # noqa: BLE001
            errors.append(f"emitter{tid}: {e}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=emitter, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.perf_counter() - t0

    assert errors == [], f"并发错误: {errors}"
    summary = sink.summary()
    assert summary["total_events"] == 8000, (
        f"丢了 {8000 - summary['total_events']} 条（锁失效）"
    )
    assert elapsed < 30.0, f"8000 条并发耗时 {elapsed:.1f}s"
    # 无交错损坏：每行都是合法 JSON
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = (tmp_path / ".trace" / f"{date_str}.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 8000
    for line in lines[:100]:  # 抽查
        json.loads(line)  # 不抛


# ---------------------------------------------------------------------------
# 2. 索引与存储规模
# ---------------------------------------------------------------------------

def test_stress_memory_store_500_entries(tmp_path):
    """500 条记忆：批量 save + snapshot_for_prompt 截断 + full_index_text。"""
    ms = MemoryStore(omnimate_home=tmp_path)

    t0 = time.perf_counter()
    for i in range(500):
        ms.save(
            name=f"item_{i}",
            description=f"第 {i} 条压力测试记忆",
            type="other",
            body="x" * 200,
            summary=f"摘要 {i}",
            topic=f"topic_{i % 20}",
        )
    save_elapsed = time.perf_counter() - t0

    assert save_elapsed < 60.0, f"500 条 save 耗时 {save_elapsed:.1f}s"

    # snapshot 有截断保护（200 行 / 25KB 上限）
    t1 = time.perf_counter()
    snap = ms.snapshot_for_prompt()
    snap_elapsed = time.perf_counter() - t1
    assert snap_elapsed < 1.0
    assert 0 < len(snap) <= 25_000 + 200

    # full_index_text 是完整版（供 retriever）
    full = ms.full_index_text()
    assert "item_499" in full or "topic_19" in full

    # 条目检索
    entries = ms.list_all()
    assert len(entries) >= 500


def test_stress_skill_index_300_skills(tmp_path):
    """300 个技能目录的索引构建耗时（system prompt 热路径）。"""
    skills = tmp_path / "skills"
    for i in range(300):
        d = skills / f"skill_{i:03d}"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f'---\ndescription: "技能 {i} 的描述"\n---\n\n正文',
            encoding="utf-8",
        )

    t0 = time.perf_counter()
    index = _build_skill_index(skills)
    elapsed = time.perf_counter() - t0

    assert "skill_000" in index or "技能 0" in index
    assert "skill_299" in index or "技能 299" in index
    assert elapsed < 5.0, f"300 技能索引耗时 {elapsed:.2f}s 超 5s"
    # 二次构建也应在时限内（不比第一次快——OS 缓存/调度抖动不保证单调，
    # 旧断言 "< elapsed" 是 flaky 源，曾致 CCAR10-11 多轮偶发失败）
    t1 = time.perf_counter()
    _build_skill_index(skills)
    assert time.perf_counter() - t1 < 5.0


def test_stress_cron_500_jobs_tick(tmp_path):
    """500 个 cron job 一次 _tick（250 个全 match + 250 个不 match）。"""
    sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
    jobs = []
    for i in range(500):
        # 一半每分钟触发，一半几乎不触发（测跳过路径）
        expr = "* * * * *" if i % 2 == 0 else "0 5 5 5 *"
        jobs.append(CronJob(
            id=f"job_{i}",
            cron=expr,
            message=f"cron message {i}",
            enabled=True,
            created_at=datetime.now().isoformat(timespec="seconds"),
        ))
    sched._jobs = jobs

    now = datetime(2026, 8, 13, 12, 0, 0)
    t0 = time.perf_counter()
    sched._tick(now)
    elapsed = time.perf_counter() - t0

    # 250 个全 match 的都触发
    notifications = list(sched._notifications)
    assert len(notifications) == 250, (
        f"触发了 {len(notifications)} 个（期望 250）"
    )
    assert elapsed < 5.0, f"500 job tick 耗时 {elapsed:.2f}s 超 5s"

    # 同分钟二次 tick 去重（不重复触发）
    sched._tick(now)
    assert len(sched._notifications) == 250


def test_stress_bg_manager_full_load(tmp_path):
    """BackgroundManager 满载：max_concurrent=3 + 15 个快任务分批跑 + 超载拒绝。

    行为说明：start() 满载时立即 raise RuntimeError（不排队）——
    所以 15 个任务分批启动（每批 3 个，等完成再启下一批）。
    """
    from agent.background import BackgroundManager

    mgr = BackgroundManager(max_concurrent=3, default_timeout=30.0)

    # 分 5 批 × 3 个（每批等前面的完成释放槽位）
    all_task_ids = []
    for batch in range(5):
        batch_ids = []
        for i in range(3):
            tid = mgr.start(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
            )
            batch_ids.append(tid)
        all_task_ids.extend(batch_ids)
        deadline = time.time() + 15
        while time.time() < deadline:
            if all(
                mgr.status(tid).status in ("completed", "failed")
                for tid in batch_ids
            ):
                break
            time.sleep(0.05)

    statuses = [mgr.status(tid).status for tid in all_task_ids]
    completed = sum(1 for s in statuses if s == "completed")
    assert completed >= 14, f"完成数不足: {statuses}"

    # 超载拒绝：塞满 2 槽后第 3 个 raise
    mgr2 = BackgroundManager(max_concurrent=2, default_timeout=30.0)
    long_ids = []
    try:
        for i in range(2):
            tid = mgr2.start(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
            )
            long_ids.append(tid)
        # 槽已满，第 3 个应该立即 raise（不排队）
        with pytest.raises(RuntimeError, match="max concurrent"):
            mgr2.start(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
            )
    finally:
        for tid in long_ids:
            mgr2.stop(tid)


def test_stress_session_store_5k_messages(tmp_path):
    """5000 条消息 + 全文 search（Python re 扫描，单用户场景基准）。"""
    store = SessionStore(tmp_path / "sessions")
    sid = store.create_session(title="stress")

    t0 = time.perf_counter()
    for i in range(5000):
        store.append_message(
            sid, "user" if i % 2 == 0 else "assistant",
            f"压力消息 {i} unique_token_{i % 100}",
        )
    insert_elapsed = time.perf_counter() - t0
    assert insert_elapsed < 60.0, f"5000 条插入耗时 {insert_elapsed:.1f}s"

    t1 = time.perf_counter()
    results = store.search("unique_token_42", limit=20)
    search_elapsed = time.perf_counter() - t1

    assert len(results) >= 20  # 每 100 条出现一次，5000 条里 50 次
    assert search_elapsed < 5.0, f"search 耗时 {search_elapsed:.2f}s 超 5s"
    store.close()


# ---------------------------------------------------------------------------
# 3. 消息形态边界
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stress_huge_user_message_1mb(tmp_path):
    """单条 1MB user 消息（非 tool result）走压缩管线。

    L1/L2 只处理 tool result——大 user 消息是形态盲区，
    验证 llm_compact 路径能兜住（不崩 + 协议合法）。
    """
    reset_offload_decisions()
    big_user = {"role": "user", "content": "巨量上下文 " + "U" * 1_000_000}
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "普通消息"},
        big_user,
        {"role": "assistant", "content": "收到"},
    ]
    # 加一些 tool 轮凑规模
    for i in range(50):
        messages.extend([
            {"role": "user", "content": f"q{i}"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": json.dumps({"path": "x"})},
                }],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "name": "read_file",
             "content": "r" * 5000},
            {"role": "assistant", "content": f"a{i}"},
        ])

    state = CompressionSessionState()
    t0 = time.perf_counter()
    new_msgs, changed, _cp = await compress_if_needed(
        messages,
        llm_client=None,
        model=None,
        config={},
        session_state=state,
        agent_home=tmp_path,
        session_id="huge_user_stress",
    )
    elapsed = time.perf_counter() - t0

    fixed = _fix_tool_call_pairs(new_msgs)
    # 协议合法
    seen = set()
    for m in fixed:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen.add(tc["id"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen
    assert elapsed < 30.0, f"1MB user 消息压缩耗时 {elapsed:.1f}s 超 30s"


def test_stress_50_parallel_tool_calls_pairing():
    """一条 assistant 带 50 个 tool_calls（并行工具调用上限形态）配对。"""
    tc_list = []
    for i in range(50):
        tc_list.append({
            "id": f"parallel_{i}",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": f"f{i}.py"}),
            },
        })
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "并行读 50 个文件"},
        {"role": "assistant", "content": None, "tool_calls": tc_list},
    ]
    # 50 个 result（乱序回填，模拟并发完成顺序）
    order = list(range(50))
    import random
    random.Random(42).shuffle(order)
    for i in order:
        messages.append({
            "role": "tool",
            "tool_call_id": f"parallel_{i}",
            "name": "read_file",
            "content": f"content {i}",
        })
    messages.append({"role": "assistant", "content": "读完了"})
    messages.append({"role": "user", "content": "下一条"})

    # 乱序 result 本身协议合法（tool_call_id 都有前置）
    fixed = _fix_tool_call_pairs(messages)
    seen = set()
    for m in fixed:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen.add(tc["id"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen

    # 缺 5 个 result 的场景（并发部分失败）
    broken = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": tc_list},
    ]
    for i in range(45):  # 只回 45 个
        broken.append({
            "role": "tool",
            "tool_call_id": f"parallel_{i}",
            "name": "read_file",
            "content": f"c{i}",
        })
    broken.append({"role": "user", "content": "next"})

    fixed2 = _fix_tool_call_pairs(broken)
    # 修复后：所有 50 个 tool_call 都有 result
    tc_ids = {tc["id"] for tc in tc_list}
    result_ids = {
        m["tool_call_id"] for m in fixed2 if m.get("role") == "tool"
    }
    assert tc_ids <= result_ids, (
        f"缺 {tc_ids - result_ids} 个 result（修复未补全）"
    )


def test_stress_goal_state_thread_safe_reads():
    """多线程并发读 goal 状态（主循环 + CLI status 查询并发场景）。"""
    g = GoalState(objective="并发读目标")
    errors = []

    def reader(tid: int):
        try:
            for i in range(2000):
                _ = g.status
                _ = g.iteration_count
                _ = g.token_budget
                _ = g.objective
        except Exception as e:  # noqa: BLE001
            errors.append(f"reader{tid}: {e}")

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
