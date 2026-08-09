# tests/test_context_pipeline.py
"""分层压缩管线测试。"""
from unittest.mock import MagicMock

from agent.context_pipeline import snip_compact, _split_system, micro_compact, llm_compact, CompressionSessionState, reactive_compact, compress_if_needed
from agent.context_compressor import _fix_tool_call_pairs


def _check_immediate_pairing(messages):
    """验证 tool_use/tool_result 双向配对完整。

    Anthropic API 严格要求：
    1. forward：每个 assistant(tool_calls) 后立刻是一串连续 tool_result，覆盖所有 id
       （中间插入其他消息 → "tool_use ids found without tool_result blocks immediately after"）
    2. backward：每个 tool_result 前面必须有对应的 assistant(tool_calls)
       （没有 → "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"）

    返回 (ok, reason)。
    """
    # forward 检查
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
        if not ids:
            continue
        j = i + 1
        collected = set()
        while j < len(messages) and messages[j].get("role") == "tool":
            collected.add(messages[j].get("tool_call_id"))
            j += 1
        if collected != ids:
            return False, (
                f"forward: assistant@{i} ids={ids} immediately_following_results={collected} "
                f"(next non-tool msg @ {j}: {messages[j].get('role') if j < len(messages) else 'END'})"
            )
    # backward 检查
    seen_ids = set()
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    seen_ids.add(tc["id"])
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid and cid not in seen_ids:
                return False, (
                    f"backward: tool@{i} tool_call_id={cid} 无对应 assistant(tool_calls) "
                    f"在前面（孤儿 result）"
                )
    return True, ""


def _mk_msgs(n, with_system=True):
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": "sys"})
    for i in range(n):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_split_system_extracts_system():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    sys, conv = _split_system(msgs)
    assert sys == {"role": "system", "content": "s"}
    assert conv == [{"role": "user", "content": "u"}]


def test_split_system_no_system_returns_none():
    msgs = [{"role": "user", "content": "u"}]
    sys, conv = _split_system(msgs)
    assert sys is None
    assert conv == msgs


def test_snip_below_threshold_noop():
    msgs = _mk_msgs(20)  # 1 + 40 = 41 条 < 50
    out, changed = snip_compact(msgs, threshold=50)
    assert changed is False
    assert out == msgs


def test_snip_above_threshold_cuts_middle():
    msgs = _mk_msgs(40)  # 1 + 80 = 81 条 > 50
    out, changed = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    assert changed is True
    # 期望结构：system + 3 头 + 1 占位 + 47 尾 = 52
    assert len(out) == 52
    assert out[0]["role"] == "system"
    # 占位消息
    placeholder = out[4]  # system + 3 head + placeholder
    assert "snip_compact" in placeholder["content"]


def test_snip_placeholder_mentions_transcript_path():
    msgs = _mk_msgs(40)
    out, _ = snip_compact(msgs, threshold=50)
    placeholders = [m for m in out if "snip_compact" in m.get("content", "")]
    assert len(placeholders) == 1
    assert ".transcripts" in placeholders[0]["content"]


def test_snip_idempotent_after_release():
    """snip 后的消息数若低于 release 阈值，再调一次不二次裁剪。"""
    msgs = _mk_msgs(40)
    out1, _ = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    # out1 有 52 条 > 50，但占位消息已是裁剪结果
    out2, changed = snip_compact(out1, threshold=50, keep_first=3, keep_last=47)
    # 第二次不应再裁（已经是占位形态）—— 通过检测占位数量
    placeholders = [m for m in out2 if "snip_compact" in m.get("content", "")]
    assert len(placeholders) == 1


# ============ 成对保护测试（P0-2） ============

def _mk_with_tool_pair_at_head_boundary():
    """构造一个 head 切割点（keep_first=2）正好落在 tool_use/tool_result 对中间。

    conv 序列（system 之后）：
      idx=0 user
      idx=1 assistant(tool_calls=call_X)   ← keep_first=2 时 head 到这里
      idx=2 tool(tool_call_id=call_X)      ← 这条被切走，但 tool_call 在 head → 孤儿
      idx=3..N 后续消息（凑到 > threshold）
    """
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u0"})
    msgs.append({
        "role": "assistant",
        "tool_calls": [{"id": "call_X", "function": {"name": "t", "arguments": "{}"}}],
    })
    msgs.append({
        "role": "tool", "tool_call_id": "call_X", "name": "t",
        "content": "result_X",
    })
    for i in range(1, 60):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_snip_protects_tool_use_tool_result_pair():
    """裁剪不能把 assistant(tool_call) 留在 head，把 tool(result) 切走。

    没有成对保护 + keep_first=2 时：
      head = [u0, assistant(tool_calls=call_X)]
      切走 → tool(result_X) 落入省略区
      → OpenAI 协议报 "no tool_result for tool_call call_X"。
    成对保护后：head 应该往后挪一位把 tool result 也带上（变成 3 条）。
    """
    msgs = _mk_with_tool_pair_at_head_boundary()
    out, changed = snip_compact(msgs, threshold=50, keep_first=2, keep_last=47)
    assert changed is True

    # 找占位前的 head 部分（system 之后到占位之前）
    placeholder_idx = None
    for i, m in enumerate(out):
        if "snip_compact" in str(m.get("content", "")):
            placeholder_idx = i
            break
    assert placeholder_idx is not None
    head_part = out[1:placeholder_idx]  # 跳过 system

    has_tool_call = any(
        any(tc.get("id") == "call_X" for tc in (m.get("tool_calls") or []))
        for m in head_part
    )
    has_tool_result = any(
        m.get("tool_call_id") == "call_X"
        for m in head_part
    )
    # 不能孤儿：要么都在 head，要么都不在
    assert has_tool_call == has_tool_result, (
        f"孤儿！has_tool_call={has_tool_call}, has_tool_result={has_tool_result}"
    )


def test_snip_protects_pair_when_tail_boundary_splits():
    """tail 切割点同样不能落在 tool_use/tool_result 对中间。

    构造：在 tail 切割点前正好是 tool_use，让 tail 取最后 47 条时丢掉了它的 result。
    """
    msgs = [{"role": "system", "content": "s"}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    # 在中间偏后位置塞一对 tool_use/tool_result
    msgs.append({"role": "user", "content": "trigger"})
    msgs.append({
        "role": "assistant",
        "tool_calls": [{"id": "call_Y", "function": {"name": "t", "arguments": "{}"}}],
    })
    msgs.append({
        "role": "tool", "tool_call_id": "call_Y", "name": "t",
        "content": "result_Y",
    })
    # 尾部继续塞消息，让 tail 边界可能落在 tool_use/tool_result 中间
    for i in range(40):
        msgs.append({"role": "user", "content": f"tail_u{i}"})

    out, changed = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    assert changed is True

    # 整个输出里不能有孤儿
    all_tool_call_ids = set()
    for m in out:
        for tc in (m.get("tool_calls") or []):
            tid = tc.get("id")
            if tid:
                all_tool_call_ids.add(tid)
    all_tool_result_ids = {
        m.get("tool_call_id") for m in out if m.get("role") == "tool"
    }
    orphans = all_tool_call_ids - all_tool_result_ids
    assert not orphans, f"孤儿 tool_call: {orphans}"


# ============ 多工具 result 拆散修复（cli-dev handoff 问题 1） ============

def test_snip_protects_multi_result_pair_in_head():
    """head 边界不能把 assistant(tool_calls=[多 id]) 的部分 result 切走。

    构造（system 之后）：
      idx=0 user(u0)
      idx=1 assistant(tool_calls=[id1, id2, id3])   ← 一次调 3 个工具
      idx=2 tool(id1)
      idx=3 tool(id2)                                ← keep_first=4 时 head 边界落这里
      idx=4 tool(id3)
      idx=5..N 后续消息凑数到 > threshold

    当前 bug：conv[head_end-1]=conv[3]=tool(id2) 不是 assistant(tool_calls)，
    现有 head 扩边界逻辑不触发 → head=[u0, assistant(tc), tool(id1), tool(id2)]，
    把 tool(id3) 切到中段被裁 → assistant(tc) 后面立刻是 [tool(id1), tool(id2), placeholder]，
    缺 id3 → Anthropic API 400 "tool_use ids found without tool_result blocks immediately after"。
    """
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u0"})
    msgs.append({
        "role": "assistant",
        "tool_calls": [
            {"id": "id1", "function": {"name": "t", "arguments": "{}"}},
            {"id": "id2", "function": {"name": "t", "arguments": "{}"}},
            {"id": "id3", "function": {"name": "t", "arguments": "{}"}},
        ],
    })
    msgs.append({"role": "tool", "tool_call_id": "id1", "name": "t", "content": "r1"})
    msgs.append({"role": "tool", "tool_call_id": "id2", "name": "t", "content": "r2"})
    msgs.append({"role": "tool", "tool_call_id": "id3", "name": "t", "content": "r3"})
    # 凑到 > threshold
    for i in range(1, 60):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    out, changed = snip_compact(msgs, threshold=50, keep_first=4, keep_last=47)
    assert changed is True

    ok, reason = _check_immediate_pairing(out)
    assert ok, f"snip_compact 拆散了多工具 result 对：{reason}"


def test_snip_protects_multi_result_pair_at_tail():
    """tail 边界不能让一串 tool_result 进入 tail 但对应 assistant(tc) 在 tail 外。

    构造：tail_start 精确落在 result 序列第一条（tool(tid1)），
    prev 是 assistant(tc)。

    当前 bug（snip_compact line 82-84）：tail_start 是 tool_result 时，
    检查 prev 是不是 assistant(tc)；是 → break 不动。
    但 prev_idx = tail_start - 1 < tail_start，prev 在 tail 外，
    assistant(tc) 也不在 tail 内 → 这串 result 是孤儿，应切走。

    精确构造（conv 不含 system）：
      conv[0..28]  共 29 条凑数（让 assistant(tc) 落在 29）
      conv[29]     assistant(tc=[tid1,tid2,tid3])
      conv[30]     tool(tid1)  ← tail_start = len-keep_last = 77-47 = 30
      conv[31]     tool(tid2)
      conv[32]     tool(tid3)
      conv[33..76] 共 44 条凑数（让 len(conv)=77）
    """
    msgs = [{"role": "system", "content": "s"}]
    # 凑数 29 条（14 对 user/assistant + 1 个 user）
    for i in range(14):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append({"role": "user", "content": "u14"})
    # assistant(tc) @ conv[29]
    msgs.append({
        "role": "assistant",
        "tool_calls": [
            {"id": "tid1", "function": {"name": "t", "arguments": "{}"}},
            {"id": "tid2", "function": {"name": "t", "arguments": "{}"}},
            {"id": "tid3", "function": {"name": "t", "arguments": "{}"}},
        ],
    })
    msgs.append({"role": "tool", "tool_call_id": "tid1", "name": "t", "content": "r1"})  # @ 30
    msgs.append({"role": "tool", "tool_call_id": "tid2", "name": "t", "content": "r2"})  # @ 31
    msgs.append({"role": "tool", "tool_call_id": "tid3", "name": "t", "content": "r3"})  # @ 32
    # 后续 44 条凑数 @ 33..76（22 对 user/assistant）
    for i in range(22):
        msgs.append({"role": "user", "content": f"tu{i}"})
        msgs.append({"role": "assistant", "content": f"ta{i}"})

    # 验证构造正确（conv 长度 = 77，tail_start = 30 落在 tool(tid1)）
    conv_len = len(msgs) - 1  # 减 system
    assert conv_len == 77, f"conv_len={conv_len} 不是 77"

    out, changed = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    assert changed is True

    ok, reason = _check_immediate_pairing(out)
    assert ok, f"snip_compact tail 边界留下孤儿 tool_result：{reason}"


def test_fix_tool_call_pairs_inserts_immediately_after():
    """_fix_tool_call_pairs 补 forward orphan 时必须紧跟 assistant(tc)。

    当前 bug：补在 messages 末尾（context_compressor.py line 153-161）。
    assistant(tc) 在中间 → 紧跟的不是 tool_result 而是后续消息 → Anthropic 400。

    场景：assistant(tc=[id1,id2]) 后只有 tool(id1)（id2 丢失），
    然后是 user 消息 + assistant 文本消息。
    修复后：假 result 必须紧跟 tool(id1) 之后，user 消息之前。
    """
    messages = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "id1", "function": {"name": "t", "arguments": "{}"}},
                {"id": "id2", "function": {"name": "t", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "id1", "name": "t", "content": "r1"},
        # id2 的 result 丢失
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok"},
    ]
    fixed = _fix_tool_call_pairs(messages)

    ok, reason = _check_immediate_pairing(fixed)
    assert ok, f"_fix_tool_call_pairs 末尾补漏，不 immediately after：{reason}"


# ============ L2 micro_compact 测试 ============

def _mk_with_tools(n_tools, recent=3):
    """构造 n_tools 条 tool 消息（夹在 user/assistant 之间）。"""
    msgs = [{"role": "system", "content": "s"}]
    for i in range(n_tools):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_{i}", "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}", "name": "t",
            "content": f"result_{i}" * 100,  # 长内容
        })
    return msgs


def test_micro_below_threshold_noop():
    msgs = _mk_with_tools(3)
    out, changed = micro_compact(msgs, keep_recent=3)
    assert changed is False
    assert out == msgs


def test_micro_replaces_old_tool_content():
    # 对齐 Claude Code microCompact：按单条大小折叠（threshold=50，content 100 > 50）
    msgs = _mk_with_tools(5)  # 5 个 tool 消息，各 ~100 字符
    out, changed = micro_compact(msgs, keep_recent=3, threshold=50)
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 5  # 数量不变
    # 前 2 个被折叠（超 threshold 且不在最近 3 条），后 3 个保留原文
    assert "micro_compacted" in tool_msgs[0]["content"]
    assert "micro_compacted" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"] == "result_2" * 100
    assert tool_msgs[3]["content"] == "result_3" * 100
    assert tool_msgs[4]["content"] == "result_4" * 100


def test_micro_below_size_threshold_noop():
    """单条不超过 threshold → 不折叠（按大小，非按数量）。"""
    msgs = _mk_with_tools(5)
    out, changed = micro_compact(msgs, keep_recent=3, threshold=10000)
    assert changed is False  # 100 字符 < 10000


def test_micro_preserves_tool_call_id_and_name():
    """折叠只换 content，role/tool_call_id/name 不变（保配对）。"""
    msgs = _mk_with_tools(5)
    out, _ = micro_compact(msgs, keep_recent=3, threshold=50)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert tool_msgs[0]["name"] == "t"


def test_micro_idempotent():
    """已经是占位的不再二次折叠。"""
    msgs = _mk_with_tools(5)
    out1, _ = micro_compact(msgs, keep_recent=3, threshold=50)
    out2, changed = micro_compact(out1, keep_recent=3, threshold=50)
    assert changed is False  # 第二次无事可做


# ============ L4 llm_compact 测试 ============

class _FakeLLM:
    """模拟 OpenAI 兼容 client（async chat_completions，对齐 LLMClient async 接口）。"""
    async def chat_completions(self, msgs):
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="这是对话总结"))]
        return m


async def test_llm_below_threshold_noop():
    msgs = _mk_msgs(20)
    out, changed = await llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100000, msg_threshold=100,
    )
    assert changed is False


async def test_llm_over_token_threshold_compacts():
    # 对齐 Claude Code：压缩由 token 驱动。161 条消息 token ~266，设阈值 100 触发。
    msgs = _mk_msgs(80)  # 1 + 160 = 161 条
    out, changed = await llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 期望：system + summary placeholder + 10 keep_recent = 12
    assert len(out) == 12
    assert out[0]["role"] == "system"
    assert "总结" in out[1]["content"]


async def test_llm_no_client_falls_back_to_rule_based():
    """llm_client=None 时仍能工作（沿用现有 _rule_based_summary）。"""
    msgs = _mk_msgs(80)
    out, changed = await llm_compact(
        msgs, llm_client=None, model="x",
        token_threshold=100, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 占位消息应含规则提取内容
    assert "用户" in out[1]["content"] or "总结" in out[1]["content"]


async def test_llm_fixes_tool_call_pairs():
    """压缩后 _fix_tool_call_pairs 应补漏（无配对 tool_result 的 tool_call）。"""
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u"})
    msgs.append({
        "role": "assistant",
        "tool_calls": [{"id": "call_x", "function": {"name": "t", "arguments": "{}"}}],
    })
    # 故意不给 tool 消息（模拟压缩边界丢失）
    for i in range(120):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    out, _ = await llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=10**9, msg_threshold=100, keep_recent=10,
    )
    # 找回 tool_result 补漏（如果有未配对的 tool_call 留在 keep_recent 里）
    # 这里 keep_recent 是最后 10 条，不含 assistant(tool_calls)，所以应该不补
    # 主要验证不抛异常
    assert isinstance(out, list)


# ============ reactive_compact 测试 ============

def test_session_state_default():
    s = CompressionSessionState()
    assert s.reacted is False
    assert s.llm_compact_count == 0
    assert s.cooldown_ok(5) is True


def test_session_state_record_and_cooldown():
    s = CompressionSessionState()
    s.current_turn = 10
    s.record_llm_compact()
    assert s.llm_compact_count == 1
    assert s.last_llm_compact_turn == 10
    s.current_turn = 12
    assert s.cooldown_ok(5) is False  # 12-10=2 < 5
    s.current_turn = 16
    assert s.cooldown_ok(5) is True   # 16-10=6 >= 5


def test_reactive_truncates_to_last_5():
    msgs = _mk_msgs(40)  # 81 条
    state = CompressionSessionState()
    out, changed = reactive_compact(msgs, session_state=state)
    assert changed is True
    assert state.reacted is True
    # system + placeholder + 5 条
    assert len(out) == 7
    assert out[0]["role"] == "system"
    assert "紧急上下文压缩" in out[1]["content"]


def test_reactive_once_per_session():
    """session_state.reacted=True 时不再触发。"""
    msgs = _mk_msgs(40)
    state = CompressionSessionState(reacted=True)
    out, changed = reactive_compact(msgs, session_state=state)
    assert changed is False
    assert out == msgs


def test_reactive_short_history_kept_as_is():
    """消息少于 keep_recent 时全部保留（仍加占位标记已触发）。"""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}]
    state = CompressionSessionState()
    out, changed = reactive_compact(msgs, session_state=state, keep_recent=5)
    assert changed is True
    assert state.reacted is True
    # system + placeholder + conv 2 条（u1 + a1）
    assert len(out) == 4
    assert out[0]["role"] == "system"
    assert "紧急上下文压缩" in out[1]["content"]


# ============ compress_if_needed 编排器测试 ============

_DEFAULT_CFG = {
    "snip_message_threshold": 50,
    "snip_keep_first": 3,
    "snip_keep_last": 47,
    "micro_keep_recent_results": 3,
    "llm_compact_token_threshold": 100000,
    "llm_compact_message_threshold": 100,
    "llm_compact_keep_recent": 10,
    "llm_compact_cooldown_turns": 5,
    "max_compress_attempts": 3,
    "transcript_enabled": True,
    "transcript_retention": 20,
}


async def test_compress_runs_l1_only_for_medium_conv(tmp_path):
    """50 < 消息数 < 100 时只跑 L1+L2，不触发 L4。"""
    msgs = _mk_msgs(40)  # 81 条，触发 L1，不触发 L4
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    assert state.llm_compact_count == 0  # L4 未触发


async def test_compress_runs_l4_for_huge_conv(tmp_path):
    """token 超限时触发 L4（对齐 Claude Code：token 驱动）。"""
    # 降低 L4 token 阈值让测试能触发（161 条消息 token ~266 > 100）
    # 禁 L1（snip）避免先裁消息导致 token 估算变小、L4 不触发
    cfg = {**_DEFAULT_CFG, "llm_compact_token_threshold": 100,
           "snip_message_threshold": 10**9}
    msgs = _mk_msgs(80)  # 161 条
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    assert state.llm_compact_count == 1


async def test_compress_respects_max_attempts(tmp_path):
    """llm_compact_count >= max_compress_attempts 时不再 L4。

    C2 修复后，L4 预算用 session_state.llm_compact_count 而非 attempt_count。
    """
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    state.llm_compact_count = 3  # 已达上限
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config={**_DEFAULT_CFG, "max_compress_attempts": 3}, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    # L1+L2 仍跑，L4 被跳过
    assert state.llm_compact_count == 3  # 未增长


async def test_compress_respects_cooldown(tmp_path):
    """L4 触发后 cooldown 期内不再触发——cooldown 是唯一拦截原因。

    通过设 llm_compact_token_threshold=1 确保 over_threshold=True，
    这样 cooldown 成为唯一阻止 L4 触发的门控。
    """
    cfg = {**_DEFAULT_CFG, "llm_compact_token_threshold": 1}
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    state.current_turn = 10
    state.record_llm_compact()  # turn=10 触发
    state.current_turn = 12      # 只过了 2 轮 < 5

    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    # L4 被 cooldown 拦下（over_threshold=True 但 cooldown 未过）
    # L1+L2 仍可能跑（changed 可能仍 True）
    assert state.llm_compact_count == 1  # 未增长——cooldown 拦下了 L4


async def test_compress_no_change_when_small(tmp_path):
    msgs = _mk_msgs(10)
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is False


async def test_compress_writes_transcript_before_l4(tmp_path):
    """L4 触发前应落盘 transcript（force=True）。"""
    # 降低 L4 token 阈值让测试能触发；禁 L1 避免先裁消息
    cfg = {**_DEFAULT_CFG, "llm_compact_token_threshold": 100,
           "snip_message_threshold": 10**9}
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="sess_t",
    )
    transcripts = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(transcripts) >= 1


def _mk_big_msgs(n_turns, chars_per_msg=8000):
    """构造 n_turns 轮对话，每条消息约 chars_per_msg 字符。

    用于让 L1 snip 后的 conv 仍超 token_threshold（100000 est tokens = ~300000 chars）。
    L1 后剩 system + 3 head + 1 placeholder + 47 tail = 52 条，
    47 tail * 8000 chars ≈ 376000 chars ≈ 125000 tokens > 100000。
    """
    big_content = "x" * chars_per_msg
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": big_content})
        msgs.append({"role": "assistant", "content": big_content})
    return msgs


async def test_compress_triggers_l4_with_default_config_after_l1(tmp_path):
    """默认配置下，L1 snip 后仍超 token_threshold 时触发 L4。

    用大内容消息让 L1 后的 conv token 估算 > 100000（默认阈值），
    覆盖默认配置路径下 L4 的触发。
    """
    msgs = _mk_big_msgs(60, chars_per_msg=8000)  # 1 + 120 = 121 条
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    assert state.llm_compact_count >= 1  # L4 触发


# ============ L2.5 output_offload 主动扫描测试（P1-2） ============

from agent.context_pipeline import offload_large_tool_results


def _mk_msgs_with_big_tool_results(sizes):
    """构造带多个 tool 消息的对话，sizes 是每个 tool 消息 content 的字符数列表。

    每条 tool 消息都配有对应的 assistant(tool_calls)。
    """
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u0"})
    for i, size in enumerate(sizes):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}", "name": "t",
            "content": "x" * size,
        })
    return msgs


def test_offload_below_threshold_noop(tmp_path):
    """tool 消息都小于阈值时，offload_large_tool_results 不动它们。"""
    msgs = _mk_msgs_with_big_tool_results([100, 200, 300])
    out, changed = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    assert changed is False
    assert out == msgs


def test_offload_drops_large_tool_results(tmp_path):
    """超阈值的 tool 消息被落盘，content 替换为 JSON 预览+指针。"""
    msgs = _mk_msgs_with_big_tool_results([100, 50000, 200])
    out, changed = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    # 第二个 tool 消息被 offload（50000 > 30000）
    import json as _json
    parsed = _json.loads(tool_msgs[1]["content"])
    assert parsed.get("truncated") is True
    assert "full_at" in parsed
    assert "preview" in parsed
    # 文件确实落盘了
    from pathlib import Path
    p = Path(parsed["full_at"])
    assert p.exists()
    assert len(p.read_text(encoding="utf-8")) == 50000


def test_offload_skips_already_offloaded(tmp_path):
    """已是占位 JSON（含 truncated 字段）的 tool 消息不再二次落盘。"""
    import json as _json
    already = _json.dumps({
        "truncated": True, "preview": "x", "full_at": "/some/path",
    })
    msgs = _mk_msgs_with_big_tool_results([100, 50000, 200])
    # 手动把第二个 tool 消息改成"已 offload"
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"][1]
    msgs[tool_idx]["content"] = already
    out, changed = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    assert changed is False  # 已是占位，不动


def test_offload_preserves_non_tool_messages(tmp_path):
    """非 tool 消息（user/assistant/system）一律不动，哪怕很长。"""
    big = "y" * 100000
    msgs = [
        {"role": "system", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
    ]
    out, changed = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    assert changed is False
    # 内容没变（虽然超长，但不是 tool 消息所以不动）
    assert all(m.get("content") == big for m in msgs)


def test_offload_preserves_tool_call_id_and_name(tmp_path):
    """offload 只换 content，role/tool_call_id/name 不变。"""
    msgs = _mk_msgs_with_big_tool_results([50000])
    out, _ = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert tool_msgs[0]["name"] == "t"
    assert tool_msgs[0]["role"] == "tool"


def test_offload_handles_missing_tool_call_id(tmp_path):
    """没有 tool_call_id 的 tool 消息用 fallback 名（防 _sanitize 报错）。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "tool_calls": [{"id": "x",
                                              "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "name": "t", "content": "z" * 50000},  # 没有 tool_call_id
    ]
    # 不抛异常
    out, changed = offload_large_tool_results(
        msgs, agent_home=tmp_path, threshold=30000,
    )
    assert changed is True


async def test_compress_runs_offload_before_micro(tmp_path):
    """compress_if_needed 编排里，offload 应在 micro_compact 之前跑。

    顺序理由：先落盘大内容（无损），再折叠旧的（有损但占位小）。
    """
    # 构造 5 个 tool 消息，每个 50000 字符
    msgs = _mk_msgs_with_big_tool_results([50000] * 5)
    # 加更多消息让 L1 触发
    for i in range(60):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    # 禁 L1（snip 50 会先裁掉 tool 消息），让 L2 microCompact 单独处理大 tool 结果
    cfg = {**_DEFAULT_CFG, "output_offload_threshold": 30000,
           "snip_message_threshold": 10**9}
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    # L2 microCompact：至少一个 tool 消息被折叠落盘（content 是 JSON 含 truncated 字段）
    import json as _json
    offloaded_count = 0
    for m in out:
        if m.get("role") != "tool":
            continue
        try:
            parsed = _json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded_count += 1
        except (ValueError, TypeError):
            pass
    assert offloaded_count >= 1, "没有任何 tool 消息被 offload"


# ---------------------------------------------------------------------------
# 对齐 Claude Code：压缩频率（token 主导，接近窗口才压缩）
# ---------------------------------------------------------------------------

async def test_l4_not_triggered_below_msg_threshold():
    """150 条消息(>旧100 但 <新500)→ L4 不触发(非 1M 模型)。"""
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    from types import SimpleNamespace
    conv = [{"role": "user", "content": "m"} for _ in range(150)]
    msgs = [{"role": "system", "content": "sys"}] + conv
    state = CompressionSessionState()
    # token 阈值设很大,只看 msg 阈值
    config = {"llm_compact_token_threshold": 10**9, "llm_compact_message_threshold": 500,
              "snip_message_threshold": 10**9}
    out, changed = await compress_if_needed(msgs, llm_client=None, model="deepseek-chat",
                                      config=config, session_state=state,
                                      agent_home=".", session_id="t")
    assert not changed, "150 条消息不应触发 L4（msg 阈值已放宽到 500）"


async def test_l4_triggered_above_token():
    """token 超阈值（接近窗口）→ L4 触发（对齐 Claude Code：token 驱动）。"""
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    # 600 条 x ~1 字符 ≈ 200 token，设阈值 100 触发
    conv = [{"role": "user", "content": "m"} for _ in range(600)]
    msgs = [{"role": "system", "content": "sys"}] + conv
    state = CompressionSessionState()
    config = {"llm_compact_token_threshold": 100, "llm_compact_message_threshold": 500,
              "snip_message_threshold": 10**9}
    out, changed = await compress_if_needed(msgs, llm_client=None, model="deepseek-chat",
                                      config=config, session_state=state,
                                      agent_home=".", session_id="t")
    assert changed, "token 超阈值应触发 L4"


async def test_l4_keep_recent_30():
    """L4 压缩后保留最近 30 条完整消息。"""
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    conv = [{"role": "user", "content": "m"} for _ in range(600)]
    msgs = [{"role": "system", "content": "sys"}] + conv
    state = CompressionSessionState()
    config = {"llm_compact_token_threshold": 100, "llm_compact_message_threshold": 500,
              "llm_compact_keep_recent": 30, "snip_message_threshold": 10**9}
    out, changed = await compress_if_needed(msgs, llm_client=None, model="deepseek-chat",
                                      config=config, session_state=state,
                                      agent_home=".", session_id="t")
    assert changed
    system, new_conv = msgs[0], out[1:]
    # 最近 30 条原始消息应保留
    assert new_conv[-30:] == conv[-30:], "应保留最近 30 条"


async def test_l4_1m_model_msg_threshold_relaxed():
    """1M 模型 msg 阈值放宽到 2000：1500 条消息不触发。"""
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    conv = [{"role": "user", "content": "m"} for _ in range(1500)]
    msgs = [{"role": "system", "content": "sys"}] + conv
    state = CompressionSessionState()
    config = {"llm_compact_token_threshold": 10**9, "llm_compact_message_threshold": 500,
              "snip_message_threshold": 10**9}
    out, changed = await compress_if_needed(msgs, llm_client=None, model="deepseek-v4-pro[1m]",
                                      config=config, session_state=state,
                                      agent_home=".", session_id="t")
    # L1 snip 可能触发（1500>200），但 L4 不应触发（msg 放宽到 2000）
    assert not any(
        "[之前的对话已自动总结]" in m.get("content", "") for m in out
    ), "1M 模型 1500 条消息不应触发 L4（msg 放宽到 2000）"


# ---------------------------------------------------------------------------
# Task P1.1: L4 context_collapse 折叠（spec §7.1）
#
# 触发：估算 token / context_window > threshold_ratio（默认 0.8）
# 动作：保留 system + pinned + 最近 N 轮，中间段折叠成占位（可逆，原文见 transcript）
# 不动 system prompt（保护 prompt cache）
# ---------------------------------------------------------------------------

from agent.context_pipeline import apply_context_collapse


def _mk_long_msgs_turns(turns: int, chars_per_msg: int = 200) -> list:
    """构造 turns 轮 user/assistant 对话，每条消息约 chars_per_msg 字符。

    一轮 = user + assistant，共 2*turns 条对话消息。
    """
    msgs = [{"role": "system", "content": "sys prompt" + "x" * 200}]
    pad = "y" * (chars_per_msg - 5)
    for i in range(turns):
        msgs.append({"role": "user", "content": f"u{i}:" + pad})
        msgs.append({"role": "assistant", "content": f"a{i}:" + pad})
    return msgs


def test_context_collapse_short_conversation_noop():
    """短对话（token 远低于阈值）不触发折叠。"""
    msgs = _mk_long_msgs_turns(turns=5)  # 10 条对话
    out, changed = apply_context_collapse(
        msgs,
        threshold_ratio=0.8,
        context_window=128_000,
        keep_recent_turns=3,
    )
    assert changed is False
    assert out == msgs


def test_context_collapse_long_conversation_triggered():
    """长对话触发折叠：保留 system + 最近 3 轮 + 折叠占位。"""
    # 40 轮 × 2 条/轮 × ~200 chars/条 / 3 ≈ 5300 tokens
    # threshold = 0.8 × 1000 = 800 tokens → 触发
    msgs = _mk_long_msgs_turns(turns=40, chars_per_msg=200)
    out, changed = apply_context_collapse(
        msgs,
        threshold_ratio=0.8,
        context_window=1000,  # 故意调小以便触发
        keep_recent_turns=3,
    )
    assert changed is True

    # system prompt 保留且内容不变（保护 prompt cache）
    assert out[0] == msgs[0]

    # 最近 3 轮（6 条对话）保留在末尾
    last_six = out[-6:]
    expected_last_six = msgs[-6:]
    assert last_six == expected_last_six

    # 至少有一个折叠占位
    placeholders = [
        m for m in out
        if "[context_collapse:" in str(m.get("content", ""))
    ]
    assert len(placeholders) >= 1, "应该有折叠占位"


def test_context_collapse_preserves_pinned_messages():
    """pinned 消息（content 以 [pinned] 开头）不被折叠，保留原位。"""
    msgs = [{"role": "system", "content": "sys"}]
    # 构造长对话，第 2 条 user 是 pinned
    pad = "z" * 200
    for i in range(40):
        if i == 2:
            msgs.append({"role": "user", "content": "[pinned] 重要约束" + pad})
        else:
            msgs.append({"role": "user", "content": f"u{i}:" + pad})
        msgs.append({"role": "assistant", "content": f"a{i}:" + pad})

    out, _ = apply_context_collapse(
        msgs,
        threshold_ratio=0.8,
        context_window=1000,
        keep_recent_turns=3,
    )
    pinned_preserved = any(
        "[pinned]" in str(m.get("content", "")) for m in out
    )
    assert pinned_preserved, "pinned 消息应保留"


def test_context_collapse_does_not_touch_system_prompt():
    """system prompt 原封不动（保护 prompt cache key）。"""
    msgs = _mk_long_msgs_turns(turns=40, chars_per_msg=200)
    original_system = dict(msgs[0])
    out, _ = apply_context_collapse(
        msgs,
        threshold_ratio=0.8,
        context_window=1000,
        keep_recent_turns=3,
    )
    assert out[0] == original_system
    assert out[0]["content"] == original_system["content"]


def test_context_collapse_with_tool_calls_keeps_pairs():
    """折叠后 tool_call / tool_result 仍成对（_fix_tool_call_pairs 安全网）。"""
    msgs = [{"role": "system", "content": "sys"}]
    pad = "x" * 200
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}:" + pad})
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "foo", "arguments": "{}"},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"result_{i}:" + pad,
        })
        msgs.append({"role": "assistant", "content": f"a{i}:" + pad})

    out, changed = apply_context_collapse(
        msgs,
        threshold_ratio=0.8,
        context_window=800,
        keep_recent_turns=3,
    )
    assert changed is True
    ok, reason = _check_immediate_pairing(out)
    assert ok, f"折叠后 tool_call 配对失败: {reason}"


def test_context_collapse_flag_off_no_op():
    """flag 关闭时（compress_if_needed 不调用），保持原消息不变。

    这里直接验证 compress_if_needed：当 features.context_collapse.enabled=False 时不触发。
    """
    import asyncio

    async def _run():
        msgs = _mk_long_msgs_turns(turns=40, chars_per_msg=200)
        state = CompressionSessionState()
        cfg = {
            "llm_compact_token_threshold": 10**9,  # 禁 L4
            "snip_message_threshold": 10**9,  # 禁 L1
            "output_offload_threshold": 10**9,  # 禁 L2
            "features": {
                "context_collapse": {"enabled": False, "threshold_ratio": 0.8},
            },
            "context_collapse_context_window": 1000,  # 即便窗口小也不应触发
        }
        out, changed = await compress_if_needed(
            msgs, llm_client=None, model="x",
            config=cfg, session_state=state,
            agent_home=".", session_id="t",
        )
        # flag OFF → 不应有 context_collapse 占位
        has_collapse = any(
            "[context_collapse:" in str(m.get("content", "")) for m in out
        )
        assert not has_collapse, "flag OFF 时不应出现 context_collapse 占位"

    asyncio.run(_run())


def test_context_collapse_flag_on_integrates_via_compress_if_needed():
    """flag ON 时 compress_if_needed 自动触发 context_collapse。"""
    import asyncio

    async def _run():
        msgs = _mk_long_msgs_turns(turns=40, chars_per_msg=200)
        state = CompressionSessionState()
        cfg = {
            "llm_compact_token_threshold": 10**9,  # 禁 L4
            "snip_message_threshold": 10**9,  # 禁 L1
            "output_offload_threshold": 10**9,  # 禁 L2
            "features": {
                "context_collapse": {"enabled": True, "threshold_ratio": 0.8},
            },
            "context_collapse_context_window": 1000,  # 强制触发
        }
        out, changed = await compress_if_needed(
            msgs, llm_client=None, model="x",
            config=cfg, session_state=state,
            agent_home=".", session_id="t",
        )
        assert changed, "flag ON + 窗口小 → 应触发 context_collapse"
        has_collapse = any(
            "[context_collapse:" in str(m.get("content", "")) for m in out
        )
        assert has_collapse, "flag ON 时应出现 context_collapse 占位"

    asyncio.run(_run())
