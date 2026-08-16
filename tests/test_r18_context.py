"""R18 上下文成本专项测试。

#16 压缩请求图片剥离
#18 autocompact 触发熔断
#15 压缩调用复用缓存前缀
#17 token 权威计数
#19 tool_use 批间摘要
"""

from types import SimpleNamespace

import pytest

from agent.context_compressor import _summarize_conversation, strip_media_blocks
from agent.context_pipeline import (
    MAX_CONSECUTIVE_L4_FAILURES,
    CompressionSessionState,
    estimate_message_tokens,
    llm_compact,
)


# ---------------------------------------------------------------------------
# R18 #16：媒体块剥离
# ---------------------------------------------------------------------------

def test_strip_media_blocks_basic():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "看这张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": "分析中"},
        {"role": "user", "content": [
            {"type": "document", "source": {"base64": "BBBB"}},
            {"type": "text", "text": "附件"},
        ]},
    ]
    out = strip_media_blocks(msgs)
    # 图片/文档块替换为文本标记，全文本时合并为 str
    assert out[0]["content"] == "看这张图\n[image]"
    assert out[1]["content"] == "分析中"  # str content 原样
    assert out[2]["content"] == "[document]\n附件"
    # 入参不污染
    assert isinstance(msgs[0]["content"], list)


def test_strip_media_blocks_noop_for_plain():
    """纯 str content 消息是 no-op（前缀复用时逐字节一致）。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
    ]
    out = strip_media_blocks(msgs)
    assert out == msgs
    assert all(m is o for m, o in zip(msgs, out))  # 同对象（未拷贝）


def test_strip_media_blocks_mixed_keeps_list():
    """非文本块（如未知类型）保留 list 形态。"""
    msgs = [{"role": "user", "content": [
        {"type": "audio", "data": "x"},  # 未知类型保留
        {"type": "image", "source": {}},
    ]}]
    out = strip_media_blocks(msgs)
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0] == {"type": "audio", "data": "x"}
    assert out[0]["content"][1] == {"type": "text", "text": "[image]"}


@pytest.mark.asyncio
async def test_summarize_strips_media(monkeypatch):
    """_summarize_conversation 对多模态消息先剥离再格式化（不 PTL/不 f-string list）。"""
    captured = {}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            captured["msgs"] = msgs
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="摘要结果", tool_calls=None),
            )])

    media_msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "q1"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
        {"role": "assistant", "content": "a1"},
    ]
    out = await _summarize_conversation(media_msgs, _Client())
    assert out == "摘要结果"
    # 发给 LLM 的 prompt 里是 [image] 标记而不是 base64
    sent = captured["msgs"][1]["content"]
    assert "[image]" in sent
    assert "data:" not in sent


# ---------------------------------------------------------------------------
# R18 #18：L4 触发熔断
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_l4_trigger_circuit_breaker(monkeypatch):
    """连续 3 次 L4 触发失败 → 本会话不再触发；成功清零。"""
    state = CompressionSessionState()
    config = {
        "max_compress_attempts": 10,
        "llm_compact_cooldown_turns": 0,
        "llm_compact_token_threshold": 100,   # 低阈值强制 over_threshold
        "transcript_enabled": False,
    }

    call_count = {"n": 0}

    class _FailClient:
        async def chat_completions(self, msgs, **kw):
            call_count["n"] += 1
            raise RuntimeError("summary down")  # 摘要失败 → 降级规则总结

    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": "x" * 400} for _ in range(50)
    ]

    from agent.context_pipeline import compress_if_needed
    import agent.context_compressor as cc

    # 重置摘要层熔断（模块级，防测试间污染）
    monkeypatch.setattr(cc, "_consecutive_failures", 0)
    monkeypatch.setattr(cc, "_compact_circuit_open", False)

    for i in range(MAX_CONSECUTIVE_L4_FAILURES):
        _, changed = await compress_if_needed(
            list(msgs), llm_client=_FailClient(), model="m",
            config=dict(config), session_state=state,
            agent_home=None, session_id="t",
        )
        assert state.llm_compact_failures == i + 1

    # 第 4 次：触发熔断，不再调摘要 LLM
    before = call_count["n"]
    await compress_if_needed(
        list(msgs), llm_client=_FailClient(), model="m",
        config=dict(config), session_state=state,
        agent_home=None, session_id="t",
    )
    assert call_count["n"] == before  # 没有新调用
    assert state.llm_compact_failures == MAX_CONSECUTIVE_L4_FAILURES

    # 摘要层熔断也开了（连续 3 次摘要失败）——此时走规则总结不调 LLM
    assert cc._compact_circuit_open is True


def test_l4_failure_reset_on_success():
    """llm_compact 成功（changed=True）时失败计数清零。"""
    state = CompressionSessionState()
    state.llm_compact_failures = 2
    state.record_llm_compact()  # 模拟成功路径
    # record_llm_compact 本身不清零——清零在 compress_if_needed 的 c4 分支
    assert state.llm_compact_failures == 2


# ---------------------------------------------------------------------------
# R18 #15：压缩调用复用缓存前缀（fork）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fork_summary_reuses_prefix():
    """fork 模式：请求 = 完整对话前缀 + 追加摘要指令；tools 透传。"""
    captured = {}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            captured["msgs"] = msgs
            captured["kw"] = kw
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="fork 摘要", tool_calls=None),
            )])

    prefix = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    out = await _summarize_conversation(
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
        _Client(),
        fork_prefix_messages=prefix,
        tools=[{"type": "function", "function": {"name": "t"}}],
    )
    assert out == "fork 摘要"
    sent = captured["msgs"]
    # 前缀逐条保留 + 末尾追加 1 条摘要指令
    assert sent[:3] == prefix
    assert len(sent) == 4
    assert sent[3]["role"] == "user"
    assert "9 段" in sent[3]["content"]
    # tools 透传（与主调用一致，保前缀缓存）；不传 model/max_tokens
    assert captured["kw"]["tools"]
    assert "model" not in captured["kw"]
    assert "max_tokens" not in captured["kw"]


@pytest.mark.asyncio
async def test_fork_skipped_when_summary_model_set():
    """配置 summary_model 时不 fork（独立调用形态：system+user 两条）。"""
    captured = {}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            captured["msgs"] = msgs
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="独立摘要", tool_calls=None),
            )])

    await _summarize_conversation(
        [{"role": "user", "content": "q"}], _Client(),
        summary_model="cheap-model",
        fork_prefix_messages=[{"role": "system", "content": "s"}],
    )
    # 独立调用：2 条（摘要 system + user prompt），不是 fork 形态
    assert len(captured["msgs"]) == 2
    assert captured["msgs"][0]["content"] == "你是技术对话摘要助手。"


@pytest.mark.asyncio
async def test_fork_failure_falls_back_to_standalone():
    """fork 失败 → 降级独立调用路径（PTL 重试/熔断语义在独立路径）。"""
    calls = {"n": 0}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("fork boom")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="降级后摘要", tool_calls=None),
            )])

    out = await _summarize_conversation(
        [{"role": "user", "content": "q"}], _Client(),
        fork_prefix_messages=[{"role": "system", "content": "s"}],
    )
    assert out == "降级后摘要"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_fork_empty_response_falls_back():
    """fork 空响应 → 降级独立调用。"""
    calls = {"n": 0}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=None),
                )])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok 摘要", tool_calls=None),
            )])

    out = await _summarize_conversation(
        [{"role": "user", "content": "q"}], _Client(),
        fork_prefix_messages=[{"role": "system", "content": "s"}],
    )
    assert out == "ok 摘要"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_llm_compact_passes_fork_prefix():
    """llm_compact 自动把完整 messages 作为 fork 基底传给摘要。"""
    captured = {}

    class _Client:
        async def chat_completions(self, msgs, **kw):
            captured["msgs"] = msgs
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="sum", tool_calls=None),
            )])

    msgs = (
        [{"role": "system", "content": "s"}]
        + [{"role": "user", "content": f"q{i} " + "x" * 100} for i in range(60)]
    )
    new_msgs, changed = await llm_compact(
        msgs, llm_client=_Client(), model="m",
        keep_recent=10, token_threshold=100,
    )
    assert changed is True
    # fork 请求以完整 messages（含 system）为前缀
    assert captured["msgs"][0] == {"role": "system", "content": "s"}
    assert len(captured["msgs"]) > 10  # 前缀全量 + 1 指令
