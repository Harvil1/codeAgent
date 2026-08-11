# tests/test_partial_compact.py
"""Task C: partial compact（双向 from / up_to）测试。

覆盖范围：
1. _summarize_conversation 加 from_idx/up_to_idx（默认 0/-1 = 全量）
2. llm_compact partial 模式（拼装 head + summary + tail）
3. compact 工具 schema 加 from_idx/up_to_idx
4. 太少不压（段 < 2 条返回空串）
5. tool_call 配对兜底（head/tail 切片破坏对时 _fix_tool_call_pairs 修复）
6. 端到端测试（通过 compress_if_needed 不传 partial 时全量回归）
7. 默认行为完全向后兼容
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent.context_compressor import (
    _summarize_conversation,
    reset_compact_circuit_breaker,
)
from agent.context_pipeline import (
    llm_compact,
    compress_if_needed,
    CompressionSessionState,
    reset_offload_decisions,
)
from tools.compact_tool import COMPACT_SCHEMA


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _make_llm(return_content="这是摘要"):
    """构造 mock async LLM client。"""
    client = MagicMock()
    m = MagicMock()
    m.choices = [MagicMock(message=MagicMock(content=return_content))]
    client.chat_completions = AsyncMock(return_value=m)
    return client


def _mk_conv(n=20):
    """构造 n 轮对话（不含 system）。每轮 user + assistant = 2 条。"""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"用户提问 {i}"})
        msgs.append({"role": "assistant", "content": f"助手回答 {i}"})
    return msgs


def _mk_conv_with_system(n=20):
    """构造带 system 的 n 轮对话。"""
    return [{"role": "system", "content": "sys"}] + _mk_conv(n)


def _mk_conv_with_tools():
    """构造含 tool_calls/tool_result 的对话（测配对兜底）。

    结构：
      system
      user(0)
      assistant(0, tool_calls=[tc1])
      tool(tc1_result)
      user(1)
      assistant(1, tool_calls=[tc2])
      tool(tc2_result)
      user(2)
      assistant(2)
    """
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0", "tool_calls": [
            {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "name": "read_file", "content": "result1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "tool_calls": [
            {"id": "tc2", "function": {"name": "write_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc2", "name": "write_file", "content": "result2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state_per_test():
    """每个测试前后重置模块级状态。"""
    reset_compact_circuit_breaker()
    reset_offload_decisions()
    yield
    reset_compact_circuit_breaker()
    reset_offload_decisions()


# ---------------------------------------------------------------------------
# 测试 1：_summarize_conversation 默认参数 = 全量（向后兼容）
# ---------------------------------------------------------------------------

async def test_summarize_default_is_full():
    """不传 from_idx/up_to_idx 时全量压缩（向后兼容）。"""
    client = _make_llm("全量摘要")
    msgs = _mk_conv(5)

    result = await _summarize_conversation(msgs, llm_client=client)

    assert result == "全量摘要"
    # 验证 LLM 收到的 dialog 含全部消息
    call_args = client.chat_completions.call_args
    sent_messages = call_args[0][0]
    user_msg_content = sent_messages[-1]["content"]
    # 全量消息内容应在 prompt 里
    assert "用户提问 0" in user_msg_content
    assert "用户提问 4" in user_msg_content


# ---------------------------------------------------------------------------
# 测试 2：_summarize_conversation partial 提取（from_idx/up_to_idx 切片）
# ---------------------------------------------------------------------------

async def test_summarize_partial_from_to():
    """from_idx=2 up_to_idx=6 只压第 2-5 条消息。"""
    client = _make_llm("中段摘要")
    msgs = _mk_conv(10)  # 20 条

    result = await _summarize_conversation(
        msgs, llm_client=client,
        from_idx=2, up_to_idx=6,
    )

    assert result == "中段摘要"
    # 验证 LLM 收到的是 messages[2:6]，不是全量
    call_args = client.chat_completions.call_args
    sent_messages = call_args[0][0]
    user_msg_content = sent_messages[-1]["content"]
    # _mk_conv(10) 结构: msgs[0]=user(0), msgs[1]=assistant(0), msgs[2]=user(1), ...
    # messages[2:6] = user(1), assistant(1), user(2), assistant(2)
    assert "用户提问 1" in user_msg_content
    assert "助手回答 2" in user_msg_content
    # 不应含 messages[0:2] 或 messages[6:] 的内容
    assert "用户提问 0" not in user_msg_content
    assert "用户提问 3" not in user_msg_content


async def test_summarize_partial_from_only():
    """from_idx=15 不传 up_to_idx（默认 -1=到末尾）。"""
    client = _make_llm("后段摘要")
    msgs = _mk_conv(10)  # 20 条

    result = await _summarize_conversation(
        msgs, llm_client=client,
        from_idx=15,
    )

    assert result == "后段摘要"
    call_args = client.chat_completions.call_args
    user_msg_content = call_args[0][0][-1]["content"]
    # messages[15:] = assistant(7) 到 assistant(9)
    assert "用户提问 0" not in user_msg_content
    assert "用户提问 8" in user_msg_content


async def test_summarize_partial_too_few_returns_empty():
    """段 < 2 条返回空串（不压）。"""
    client = _make_llm("不应被调用")
    msgs = _mk_conv(10)

    result = await _summarize_conversation(
        msgs, llm_client=client,
        from_idx=0, up_to_idx=1,  # 只 1 条
    )

    assert result == ""
    assert client.chat_completions.call_count == 0


# ---------------------------------------------------------------------------
# 测试 3：llm_compact partial 模式（拼装 head + summary + tail）
# ---------------------------------------------------------------------------

async def test_llm_compact_partial_middle():
    """partial 压中段：head + summary + tail 拼装正确。"""
    client = _make_llm("中段摘要内容")
    # 20 条 conv + system
    messages = _mk_conv_with_system(10)  # 21 条（1 system + 20 conv）

    # from_idx=4 up_to_idx=16 → 压 conv[4:16]，保留 conv[:4] + conv[16:]
    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=4, up_to_idx=16,
        token_threshold=0,  # 强制触发
    )

    assert changed is True
    # 去掉 system 看拼装
    conv = [m for m in new_msgs if m.get("role") != "system"]
    # head = conv[:4]（原 4 条）+ 1 summary + tail = conv[16:]（原 4 条）
    assert len(conv) == 4 + 1 + 4
    # head 应保留原文
    assert conv[0]["content"] == "用户提问 0"
    assert conv[3]["content"] == "助手回答 1"
    # summary msg
    assert "中段摘要内容" in conv[4]["content"]
    # tail 应保留原文
    assert conv[5]["content"] == "用户提问 8"
    assert conv[8]["content"] == "助手回答 9"


async def test_llm_compact_partial_front():
    """partial 压前段：from=0 up_to=10 → 保留后段原文。"""
    client = _make_llm("前段摘要")
    messages = _mk_conv_with_system(10)  # 21 条

    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=0, up_to_idx=10,
        token_threshold=0,
    )

    assert changed is True
    conv = [m for m in new_msgs if m.get("role") != "system"]
    # head = [] + 1 summary + tail = conv[10:]（10 条）
    assert len(conv) == 1 + 10
    assert "前段摘要" in conv[0]["content"]
    # tail 第一条应是 conv[10] = user(5)
    assert conv[1]["content"] == "用户提问 5"


async def test_llm_compact_partial_back():
    """partial 压后段：from=15 up_to=-1 → 保留前段原文。"""
    client = _make_llm("后段摘要")
    messages = _mk_conv_with_system(10)  # 21 条

    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=15, up_to_idx=-1,
        token_threshold=0,
    )

    assert changed is True
    conv = [m for m in new_msgs if m.get("role") != "system"]
    # head = conv[:15]（15 条）+ 1 summary + tail = []
    assert len(conv) == 15 + 1
    # head 原文保留
    # conv 结构（_mk_conv 产生的 10 轮 = 20 条 conv）：
    # conv[0]=user(0), conv[1]=assistant(0), conv[2]=user(1), ...
    # conv[14]=user(7)
    assert conv[0]["content"] == "用户提问 0"
    assert conv[14]["content"] == "用户提问 7"
    assert "后段摘要" in conv[15]["content"]


async def test_llm_compact_default_full_backward_compat():
    """不传 partial 参数 = 全量压缩（向后兼容）。"""
    client = _make_llm("全量摘要")
    messages = _mk_conv_with_system(10)

    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        keep_recent=4,
        token_threshold=0,
    )

    assert changed is True
    conv = [m for m in new_msgs if m.get("role") != "system"]
    # 全量：summary + keep_recent(4)
    assert len(conv) == 1 + 4
    assert "全量摘要" in conv[0]["content"]
    # keep_recent 4 条 = conv[-4:] = conv[16:]
    # conv[16] = user(8), conv[17] = assistant(8), conv[18] = user(9), conv[19] = assistant(9)
    assert conv[1]["content"] == "用户提问 8"
    assert conv[2]["content"] == "助手回答 8"
    assert conv[4]["content"] == "助手回答 9"


# ---------------------------------------------------------------------------
# 测试 4：tool_call 配对兜底
# ---------------------------------------------------------------------------

async def test_llm_compact_partial_tool_call_pairing():
    """partial 切片可能切断 tool_call/tool_result 对，验证 _fix_tool_call_pairs 兜底。"""
    client = _make_llm("partial 摘要")
    # 用带 tool_calls 的对话
    messages = _mk_conv_with_tools()  # 9 条（1 system + 8 conv）

    # from_idx=2 up_to_idx=6 → 切 conv[2:6]
    # conv[2] = assistant(0, tool_calls=[tc1])
    # conv[3] = tool(tc1_result)
    # conv[4] = user(1)
    # conv[5] = assistant(1, tool_calls=[tc2])
    # head = conv[:2] = [user(0), assistant(0, tool_calls=[tc1])]
    #   → head 末尾是 assistant(tc1) 但 head 里没有 tc1 的 result → 需补
    # tail = conv[6:] = [tool(tc2_result), user(2), assistant(2)]
    #   → tail 开头是 tool(tc2) 但 assistant(tc2) 被压了 → 反向孤儿 → 删
    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=2, up_to_idx=6,
        token_threshold=0,
    )

    assert changed is True

    # 验证 tool_call 配对完整（用 test_context_pipeline 的检查逻辑）
    conv = [m for m in new_msgs if m.get("role") != "system"]
    ok, reason = _check_immediate_pairing(conv)
    assert ok, f"partial compact 后 tool_call 配对被破坏: {reason}"


def _check_immediate_pairing(messages):
    """验证 tool_use/tool_result 双向配对完整（复用 test_context_pipeline 的逻辑）。"""
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
                f"forward: assistant@{i} ids={ids} immediately_following_results={collected}"
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
                    f"backward: tool@{i} tool_call_id={cid} 无对应 assistant(tool_calls)"
                )
    return True, ""


# ---------------------------------------------------------------------------
# 测试 5：compact 工具 schema 含 from_idx/up_to_idx
# ---------------------------------------------------------------------------

def test_compact_schema_has_partial_fields():
    """compact 工具 schema 应含 from_idx / up_to_idx 可选字段。"""
    props = COMPACT_SCHEMA["parameters"]["properties"]
    assert "from_idx" in props, "compact schema 缺 from_idx 字段"
    assert "up_to_idx" in props, "compact schema 缺 up_to_idx 字段"
    assert props["from_idx"]["type"] == "integer"
    assert props["up_to_idx"]["type"] == "integer"


# ---------------------------------------------------------------------------
# 测试 6：端到端通过 compress_if_needed 触发（不传 partial = 全量回归）
# ---------------------------------------------------------------------------

async def test_e2e_compress_if_needed_full_compat(tmp_path):
    """端到端：compress_if_needed 不传 partial，行为跟现有完全一致。"""
    big_content = "x" * 8000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(60):
        msgs.append({"role": "user", "content": big_content})
        msgs.append({"role": "assistant", "content": big_content})

    mock_client = SimpleNamespace()
    mock_client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="e2e 摘要", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    ))

    state = CompressionSessionState()
    cfg = {
        "llm_compact_token_threshold": 100,
        "snip_message_threshold": 10**9,
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "transcript_enabled": False,
    }

    out, changed = await compress_if_needed(
        msgs, llm_client=mock_client, model="test-model",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="e2e-partial",
    )

    assert changed is True
    assert state.llm_compact_count == 1


# ---------------------------------------------------------------------------
# 测试 7：llm_compact partial + keep_recent 交互
# ---------------------------------------------------------------------------

async def test_llm_compact_partial_overrides_keep_recent():
    """partial 模式时 from_idx/up_to_idx 决定切片，keep_recent 被忽略。"""
    client = _make_llm("partial 摘要")
    messages = _mk_conv_with_system(10)  # 21 条

    # partial 模式：from=0 up_to=10，keep_recent=4 应被忽略
    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=0, up_to_idx=10,
        keep_recent=4,  # 应被忽略
        token_threshold=0,
    )

    assert changed is True
    conv = [m for m in new_msgs if m.get("role") != "system"]
    # partial：head=[] + summary + tail=conv[10:]（10 条）
    # 如果 keep_recent 生效了会变成 summary + 4 条
    assert len(conv) == 1 + 10, (
        f"partial 模式应忽略 keep_recent，期望 11 条，实际 {len(conv)} 条"
    )


# ---------------------------------------------------------------------------
# 测试 8：llm_compact partial 段太少不压（返回不变）
# ---------------------------------------------------------------------------

async def test_llm_compact_partial_too_few_noop():
    """partial 段 < 2 条时 noop（不压，返回原 messages + changed=False）。"""
    client = _make_llm("不应调用")
    messages = _mk_conv_with_system(10)

    new_msgs, changed = await llm_compact(
        messages,
        llm_client=client,
        model="test",
        from_idx=0, up_to_idx=1,  # 只 1 条
        token_threshold=0,
    )

    assert changed is False
    assert client.chat_completions.call_count == 0


# ---------------------------------------------------------------------------
# 测试 9：端到端通过 compact 工具 handler 触发 partial
# ---------------------------------------------------------------------------

async def test_e2e_compact_tool_partial():
    """端到端：通过 compact 工具 handler 触发 partial 压缩，验证 agent 状态更新。"""
    import json
    from tools.compact_tool import _handle_compact

    # mock LLM client
    client = _make_llm("tool partial 摘要")

    # mock agent
    agent = MagicMock()
    agent.llm_client = client
    agent.model = "test-model"
    agent.conversation_history = _mk_conv(10)  # 20 条 conv
    agent._get_system_prompt = MagicMock(return_value="sys")
    agent.invalidate_system_prompt = MagicMock()

    result_str = await _handle_compact(
        {"from_idx": 4, "up_to_idx": 16, "focus": "中段压缩"},
        agent_ref=agent,
    )
    result = json.loads(result_str)

    assert result["success"] is True
    assert "partial" in result["mode"]
    assert result["from_idx"] == 4
    assert result["up_to_idx"] == 16
    # agent.conversation_history 应被更新
    assert len(agent.conversation_history) < 20
    # LLM 被调用
    assert client.chat_completions.call_count >= 1


async def test_e2e_compact_tool_full_compat():
    """端到端：compact 工具不传 partial 参数 = 全量压缩（向后兼容）。"""
    import json
    from tools.compact_tool import _handle_compact

    client = _make_llm("全量压缩摘要")

    agent = MagicMock()
    agent.llm_client = client
    agent.model = "test-model"
    # keep_recent=30 in _handle_compact, need > 30 conv msgs for full mode
    agent.conversation_history = _mk_conv(20)  # 40 条 conv
    agent._get_system_prompt = MagicMock(return_value="sys")
    agent.invalidate_system_prompt = MagicMock()

    result_str = await _handle_compact({}, agent_ref=agent)
    result = json.loads(result_str)

    assert result["success"] is True
    assert result["mode"] == "full"
    assert result["from_idx"] is None  # 全量模式不返回 partial 字段
    # agent 状态更新
    assert len(agent.conversation_history) < 40


async def test_e2e_compact_tool_too_short_partial():
    """端到端：partial 模式对话太短（< 2 条）返回失败。"""
    import json
    from tools.compact_tool import _handle_compact

    client = _make_llm("不应调用")

    agent = MagicMock()
    agent.llm_client = client
    agent.model = "test-model"
    agent.conversation_history = [{"role": "user", "content": "only 1 msg"}]
    agent._get_system_prompt = MagicMock(return_value="sys")
    agent.invalidate_system_prompt = MagicMock()

    result_str = await _handle_compact(
        {"from_idx": 0, "up_to_idx": -1},
        agent_ref=agent,
    )
    result = json.loads(result_str)

    assert result["success"] is False
    assert client.chat_completions.call_count == 0
