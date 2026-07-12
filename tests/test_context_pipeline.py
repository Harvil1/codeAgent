# tests/test_context_pipeline.py
"""分层压缩管线测试。"""
from unittest.mock import MagicMock

from agent.context_pipeline import snip_compact, _split_system, micro_compact, llm_compact, CompressionSessionState, reactive_compact


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
    msgs = _mk_with_tools(5)  # 5 个 tool 消息
    out, changed = micro_compact(msgs, keep_recent=3)
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 5  # 数量不变
    # 前 2 个被折叠，后 3 个保留原文
    assert "micro_compacted" in tool_msgs[0]["content"]
    assert "micro_compacted" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"] == "result_2" * 100
    assert tool_msgs[3]["content"] == "result_3" * 100
    assert tool_msgs[4]["content"] == "result_4" * 100


def test_micro_preserves_tool_call_id_and_name():
    """折叠只换 content，role/tool_call_id/name 不变（保配对）。"""
    msgs = _mk_with_tools(5)
    out, _ = micro_compact(msgs, keep_recent=3)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert tool_msgs[0]["name"] == "t"


def test_micro_idempotent():
    """已经是占位的不再二次折叠。"""
    msgs = _mk_with_tools(5)
    out1, _ = micro_compact(msgs, keep_recent=3)
    out2, changed = micro_compact(out1, keep_recent=3)
    assert changed is False  # 第二次无事可做


# ============ L4 llm_compact 测试 ============

class _FakeLLM:
    """模拟 OpenAI 兼容 client。"""
    def chat_completions(self, msgs):
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="这是对话总结"))]
        return m


def test_llm_below_threshold_noop():
    msgs = _mk_msgs(20)
    out, changed = llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100000, msg_threshold=100,
    )
    assert changed is False


def test_llm_over_msg_threshold_compacts():
    msgs = _mk_msgs(80)  # 1 + 160 = 161 条 > 100
    out, changed = llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100000, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 期望：system + summary placeholder + 10 keep_recent = 12
    assert len(out) == 12
    assert out[0]["role"] == "system"
    assert "总结" in out[1]["content"]


def test_llm_no_client_falls_back_to_rule_based():
    """llm_client=None 时仍能工作（沿用现有 _rule_based_summary）。"""
    msgs = _mk_msgs(80)
    out, changed = llm_compact(
        msgs, llm_client=None, model="x",
        token_threshold=100000, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 占位消息应含规则提取内容
    assert "用户" in out[1]["content"] or "总结" in out[1]["content"]


def test_llm_fixes_tool_call_pairs():
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

    out, _ = llm_compact(
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
