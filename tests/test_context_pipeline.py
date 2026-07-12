# tests/test_context_pipeline.py
"""分层压缩管线测试。"""
from agent.context_pipeline import snip_compact, _split_system, micro_compact


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
