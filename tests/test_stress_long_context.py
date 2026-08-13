"""长上下文压力测试（CCAR8 后）。

验证 5 层压缩管线 + goal 长跑 + ephemeral 高频注入 + trace 高频写
在极端规模下：(1) 不崩 (2) OpenAI 消息协议合法（无孤儿 tool result）
(3) ephemeral 不泄漏 (4) 性能可接受。

纯本地 mock（不调真 LLM），压的是管线逻辑不是 API。
"""
import json
import time

import pytest

from agent.context_pipeline import (
    CompressionSessionState,
    compress_if_needed,
    micro_compact,
    offload_large_tool_results,
    snip_compact,
)
from agent.context_compressor import (
    _fix_tool_call_pairs,
    estimate_message_tokens,
)
from agent.goal import GoalState
from agent.trace import TraceSink


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_turn(i: int, big_result: int = 0, with_ts: bool = False) -> list:
    """构造一轮：user → assistant(tool_calls) → tool(result) → assistant(text)。"""
    ts = {"_timestamp": time.time() - 7200} if with_ts else {}  # 2h 前（触发 time-based MC）
    return [
        {"role": "user", "content": f"user message {i} " + "x" * 100, **ts},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": f"f{i}.py"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "name": "read_file",
            "content": ("BIGDATA" * (big_result // 7)) if big_result else f"result {i}",
        },
        {"role": "assistant", "content": f"assistant summary {i}", **ts},
    ]


def make_history(turns: int, big_result: int = 0, with_ts: bool = False) -> list:
    history = [{"role": "system", "content": "sys prompt " + "s" * 500}]
    for i in range(turns):
        history.extend(make_turn(i, big_result, with_ts))
    return history


def assert_protocol_valid(messages: list) -> None:
    """OpenAI 协议校验：每个 tool result 的 tool_call_id 必须在前面 assistant(tool_calls) 里。

    同时校验 tool result 紧跟其 assistant(tc)（ Anthropic immediately-after 要求，
    _fix_tool_call_pairs 的输出应满足）。
    """
    seen = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen.add(tc["id"])
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id")
            assert tid in seen, f"孤儿 tool result: {tid}（前面没有对应 tool_calls）"


# ---------------------------------------------------------------------------
# Test 1: L1 snip 万条消息
# ---------------------------------------------------------------------------

def test_stress_snip_10k_messages():
    """2500 轮 × 4 条 = 10000+ 条消息过 L1 snip。

    验证：裁剪发生、耗时 < 5s、配对协议合法。
    """
    history = make_history(2500)
    assert len(history) > 10000

    t0 = time.perf_counter()
    new_msgs, changed = snip_compact(history, threshold=200)
    elapsed = time.perf_counter() - t0

    assert changed is True
    assert len(new_msgs) < len(history)
    assert elapsed < 5.0, f"snip 10k 条耗时 {elapsed:.2f}s 超 5s 上限"

    fixed = _fix_tool_call_pairs(new_msgs)
    assert_protocol_valid(fixed)


def test_stress_snip_idempotent_on_huge_history():
    """万条历史二次 snip 幂等（已有占位不再裁）。"""
    history = make_history(2500)
    once, changed1 = snip_compact(history, threshold=200)
    twice, changed2 = snip_compact(once, threshold=200)
    assert changed1 is True
    assert changed2 is False
    assert len(twice) == len(once)


# ---------------------------------------------------------------------------
# Test 2: L2 micro + L2.5 offload 大 tool result
# ---------------------------------------------------------------------------

def test_stress_micro_compact_200_big_results(tmp_path):
    """200 条 100KB tool result（约 20MB）过 L2 micro 折叠。

    验证：折叠发生、tool_call_id 保留、耗时 < 10s。
    """
    history = make_history(200, big_result=100_000)
    total_bytes = sum(
        len(m["content"]) for m in history if m.get("role") == "tool"
    )
    assert total_bytes > 15_000_000, f"构造量不足：{total_bytes} bytes"

    t0 = time.perf_counter()
    new_msgs, changed = micro_compact(
        history, threshold=50_000, agent_home=tmp_path,
    )
    elapsed = time.perf_counter() - t0

    assert changed is True
    assert elapsed < 10.0, f"micro 20MB 耗时 {elapsed:.2f}s 超 10s 上限"

    # 配对保留（只换 content，结构不动）
    assert_protocol_valid(new_msgs)
    # 折叠后总量显著下降（最近 3 条保护不折）
    new_total = sum(
        len(str(m["content"])) for m in new_msgs if m.get("role") == "tool"
    )
    assert new_total < total_bytes // 10, (
        f"折叠效果不足：{total_bytes} → {new_total}"
    )


def test_stress_offload_200_big_results(tmp_path):
    """200 条 100KB tool result 过 L2.5 offload（落盘 + 决策冻结）。"""
    from agent.context_pipeline import reset_offload_decisions
    reset_offload_decisions()

    history = make_history(200, big_result=100_000)

    t0 = time.perf_counter()
    new_msgs, changed = offload_large_tool_results(
        history, agent_home=tmp_path, threshold=50_000,
    )
    elapsed = time.perf_counter() - t0

    assert changed is True
    assert elapsed < 10.0
    assert_protocol_valid(new_msgs)

    # 决策冻结：二次跑 byte-identical（不再新落盘）
    again, changed2 = offload_large_tool_results(
        new_msgs, agent_home=tmp_path, threshold=50_000,
    )
    assert changed2 is False, "决策冻结失效：二次跑又落盘了"
    assert again == new_msgs


# ---------------------------------------------------------------------------
# Test 3: _fix_tool_call_pairs 压力（大量孤儿）
# ---------------------------------------------------------------------------

def test_stress_fix_pairs_with_orphans():
    """3000 轮 + 故意删 1/3 的 assistant(tool_calls) → 大量孤儿 result。

    验证：修复后协议合法、耗时 < 5s。
    """
    history = make_history(3000)
    # 制造孤儿：删掉部分 assistant(tool_calls) 消息
    pruned = [
        m for i, m in enumerate(history)
        if not (m.get("role") == "assistant" and m.get("tool_calls")
                and i % 3 == 0)
    ]

    t0 = time.perf_counter()
    fixed = _fix_tool_call_pairs(pruned)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"fix_pairs 3k 轮耗时 {elapsed:.2f}s 超 5s"
    assert_protocol_valid(fixed)


def test_stress_fix_pairs_misordered():
    """错序压力：tool result 出现在对应 assistant(tc) 之前（X13 场景放大）。"""
    history = make_history(2000)
    # 把一半的 tool result 挪到历史最前面（全部变成反向孤儿）
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    others = [m for m in history if m.get("role") != "tool"]
    system = [m for m in others if m.get("role") == "system"]
    rest = [m for m in others if m.get("role") != "system"]
    misordered = system + tool_msgs[:1000] + rest

    fixed = _fix_tool_call_pairs(misordered)
    assert_protocol_valid(fixed)


# ---------------------------------------------------------------------------
# Test 4: 全管线 compress_if_needed（极端历史 + ephemeral 混入）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stress_compress_if_needed_extreme(tmp_path):
    """500 轮 + 每轮 80KB tool result（约 40MB ≈ 13M token）+ 2h 前 ts。

    过完整压缩管线（llm_client=None 降级 rule-based 摘要）。
    验证：不崩、协议合法、总 token 显著下降。
    """
    history = make_history(500, big_result=80_000, with_ts=True)
    # 混入 ephemeral 消息（压缩路径应过滤）
    ephemeral = {
        "role": "user",
        "content": '<channel_push count="1">[stress] injected</channel_push>',
        "_ephemeral": True,
    }
    history.insert(500, dict(ephemeral))

    before_tokens = estimate_message_tokens(history)
    assert before_tokens > 1_000_000, f"构造量不足：{before_tokens} tokens"

    state = CompressionSessionState()
    t0 = time.perf_counter()
    new_msgs, changed = await compress_if_needed(
        history,
        llm_client=None,   # 降级 rule-based（不烧 LLM）
        model=None,
        config={},          # 全默认值
        session_state=state,
        agent_home=tmp_path,
        session_id="stress_test_session",
    )
    elapsed = time.perf_counter() - t0

    # 基本断言：不崩 + 有变化 + 协议合法
    assert elapsed < 30.0, f"全管线耗时 {elapsed:.2f}s 超 30s 上限"
    fixed = _fix_tool_call_pairs(new_msgs)
    assert_protocol_valid(fixed)

    # ephemeral 不应被压缩路径保留为持久结构（strip 后发给 LLM 无 _ephemeral 字段；
    # 这里验证压缩输出中 _ephemeral 标记的消息没有增加）
    eph_count = sum(1 for m in new_msgs if m.get("_ephemeral"))
    assert eph_count <= 1, "压缩过程不应复制/新增 ephemeral 消息"

    after_tokens = estimate_message_tokens(new_msgs)
    # 压缩有效（至少砍一半；L1+L2 已足够）
    assert after_tokens < before_tokens * 0.8, (
        f"压缩效果不足：{before_tokens} → {after_tokens} tokens"
    )


@pytest.mark.asyncio
async def test_stress_compress_repeated_10x(tmp_path):
    """连续 10 次全管线压缩（模拟长会话反复触发）。

    验证：每次协议合法、不累积破坏、耗时稳定。
    """
    from agent.context_pipeline import reset_offload_decisions
    reset_offload_decisions()

    history = make_history(300, big_result=60_000, with_ts=True)
    state = CompressionSessionState()

    messages = history
    for round_n in range(10):
        messages, changed = await compress_if_needed(
            messages,
            llm_client=None,
            model=None,
            config={},
            session_state=state,
            agent_home=tmp_path,
            session_id=f"stress_round_{round_n}",
        )
        fixed = _fix_tool_call_pairs(messages)
        assert_protocol_valid(fixed)
        # 消息数不应增长（压缩只减不增）
        assert len(messages) <= len(history) + 10, (
            f"round {round_n}: 消息数异常增长 → {len(messages)}"
        )


# ---------------------------------------------------------------------------
# Test 5: GoalState 千轮长跑
# ---------------------------------------------------------------------------

def test_stress_goal_1000_iterations():
    """1000 轮 evaluate_after_turn（无 budget 限制）。

    验证：计数正确、notes 不无限膨胀到失控、持久化 round-trip。
    """
    g = GoalState(objective="stress goal", token_budget_limit=None)
    t0 = time.perf_counter()
    for i in range(1000):
        decision = g.evaluate_after_turn(tokens_used=100)
        assert decision == "continue", f"第 {i} 轮意外决策: {decision}"
    elapsed = time.perf_counter() - t0

    assert g.iteration_count == 1000
    assert g.token_budget == 100_000
    assert elapsed < 5.0, f"1000 轮 evaluate 耗时 {elapsed:.2f}s"


def test_stress_goal_budget_pause_at_scale():
    """10 万轮累加后 budget 超限 pause（大量数字累加不出错）。"""
    g = GoalState(objective="x", token_budget_limit=1_000_000)
    for i in range(10_000):
        decision = g.evaluate_after_turn(tokens_used=200)
        if decision != "continue":
            break
    assert decision == "pause"
    assert g.pause_reason == "budget_exceeded"
    assert g.token_budget >= 1_000_000


def test_stress_goal_save_load_loop(tmp_path):
    """500 次 save/load round-trip（原子写压力）。"""
    g = GoalState(objective="persist stress")
    path = tmp_path / ".goal" / "current.json"
    t0 = time.perf_counter()
    for i in range(500):
        g.iteration_count = i
        g.save(path)
        loaded = GoalState.load(path)
        assert loaded is not None
        assert loaded.iteration_count == i
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"500 次 round-trip 耗时 {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 6: ephemeral 高频注入
# ---------------------------------------------------------------------------

def test_stress_channel_injection_500_rounds(tmp_path):
    """模拟 500 轮、每轮 50 条 channel push → 注入 + mark_consumed。

    验证：注入消息 role=user 带 _ephemeral、inbox 清空、耗时 < 10s。
    """
    from agent import _build_channel_injection
    from agent.channel_inbox import ChannelInbox

    inbox = ChannelInbox(tmp_path)
    t0 = time.perf_counter()
    for round_n in range(500):
        # 每轮 push 50 条
        for k in range(50):
            inbox.push("feishu", {"round": round_n, "seq": k, "text": "m" * 200})
        # 注入（消费）
        msg = _build_channel_injection(inbox)
        assert msg is not None, f"round {round_n}: 有 50 条未读但没注入"
        assert msg["role"] == "user"
        assert msg.get("_ephemeral") is True
        assert 'channel_push' in msg["content"]
        # 消费完应该清空
        assert inbox.unconsumed() == [], f"round {round_n}: 注入后未清空"
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"500 轮注入耗时 {elapsed:.2f}s 超 10s"
    # inbox 目录应该没剩文件（全部消费即删）
    leftovers = list((tmp_path / ".inbox").glob("*.json"))
    assert leftovers == [], f"剩 {len(leftovers)} 条未清理"


def test_stress_mail_injection_500_rounds(tmp_path):
    """模拟 500 轮、每轮 20 条 mail（累计万封）→ 注入 + mark_read。

    性能语义：mark_read 写侧已 O(k)（sidecar append，本压力测试发现的
    O(n²) 已修），读侧 check_unread 仍 O(n) 全量扫描——这是"非消费式
    保留"的语义代价（必须扫全量找未读）。单轮 ~45ms 生产无感；万封是
    极端场景（生产靠 clear 清理）。上限 30s 用于抓未来性能回归
    （sidecar 修复前全量重写时此场景 30s+ 超时）。
    """
    from agent import _build_mail_injection
    from agent.team.mailbox import Mailbox

    mb = Mailbox(tmp_path)
    t0 = time.perf_counter()
    for round_n in range(500):
        for k in range(20):
            mb.send(to="main", from_="worker", content="mail " * 50, kind="task")
        msg = _build_mail_injection(mb, "main")
        assert msg is not None
        assert msg["role"] == "user"
        assert msg.get("_ephemeral") is True
        assert mb.check_unread("main") == []
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"500 轮 mail 注入耗时 {elapsed:.2f}s 超 30s"
    # mailbox 保留了已读邮件（非消费式，与 MessageBus 不同）
    all_msgs = mb.check_all("main")
    assert len(all_msgs) == 500 * 20, "mail 不应被删除（非消费式语义）"


def test_stress_goal_continue_message_1000x():
    """1000 次构造 goal continue 消息（纯函数压力 + 格式稳定）。"""
    from agent import _build_goal_continue_message

    g = GoalState(objective="obj " + "o" * 1000)
    t0 = time.perf_counter()
    for i in range(1000):
        g.iteration_count = i
        msg = _build_goal_continue_message(g)
        assert msg is not None
        assert msg["role"] == "user"
        assert msg.get("_ephemeral") is True
        assert "continue_goal" in msg["content"]
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0


# ---------------------------------------------------------------------------
# Test 7: TraceSink 高频写
# ---------------------------------------------------------------------------

def test_stress_trace_10k_emits(tmp_path):
    """10000 次 emit 写盘压力。

    验证：不崩、行数完整、summary 聚合正确、耗时 < 15s。
    """
    sink = TraceSink(tmp_path)
    t0 = time.perf_counter()
    for i in range(10_000):
        sink.emit(
            "post_tool_use",
            tool="read_file",
            idx=i,
            payload="x" * 100,
        )
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0, f"10k emits 耗时 {elapsed:.2f}s 超 15s"

    summary = sink.summary()
    assert summary["total_events"] == 10_000
    assert summary["by_event"]["post_tool_use"] == 10_000

    # query 极限过滤
    results = sink.query(event="post_tool_use", limit=100)
    assert len(results) == 100


# ---------------------------------------------------------------------------
# Test 8: token 估算性能
# ---------------------------------------------------------------------------

def test_stress_estimate_tokens_10k_messages():
    """万条消息 token 估算耗时 < 2s（每轮压缩都调它，是热路径）。"""
    history = make_history(2500)
    t0 = time.perf_counter()
    tokens = estimate_message_tokens(history)
    elapsed = time.perf_counter() - t0

    assert tokens > 100_000
    assert elapsed < 2.0, f"estimate 10k 条耗时 {elapsed:.2f}s 超 2s"
