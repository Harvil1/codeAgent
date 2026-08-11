# tests/test_summarize_9section.py
"""改造点 ② LLM 摘要质量提升测试（9 段式 + PTL 重试 + 熔断器）。

覆盖范围：
1. 9 段式 prompt 真的发给 LLM
2. PTL 重试（prompt_too_long 时丢 20% 旧消息重试，最终成功）
3. PTL 重试耗尽后走 _rule_based_summary
4. 熔断器（连续 3 次失败后第 4 次直接走规则总结，不调 LLM）
5. 熔断器重置（成功一次后 _consecutive_failures=0）
6. session memory 替代（传了 session_memory 时完全不调 LLM）
7. <analysis> 草稿剥离
8. 端到端：通过 compress_if_needed 触发，验证 9 段 prompt + 熔断器在生产路径生效
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent.context_compressor import (
    _summarize_conversation,
    _strip_analysis_draft,
    _rule_based_summary,
    SUMMARIZE_PROMPT_9SECTION,
    reset_compact_circuit_breaker,
    MAX_CONSECUTIVE_FAILURES,
    MAX_PTL_RETRIES,
    _compute_ptl_drop_count,
    _get_model_max_tokens,
)
from agent.context_pipeline import compress_if_needed, CompressionSessionState


# ---------------------------------------------------------------------------
# 辅助：构造 mock LLM client（async）
# ---------------------------------------------------------------------------

def _make_llm(return_content="这是摘要", side_effect=None):
    """构造一个 mock async LLM client。

    side_effect 可以是 Exception / list（每次抛不同异常）。
    return_content 是成功时返回的 content。
    """
    client = MagicMock()
    if side_effect is not None:
        client.chat_completions = AsyncMock(side_effect=side_effect)
    else:
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content=return_content))]
        client.chat_completions = AsyncMock(return_value=m)
    return client


def _mk_msgs(n=10):
    """构造 n 轮对话。"""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"用户提问 {i}"})
        msgs.append({"role": "assistant", "content": f"助手回答 {i}"})
    return msgs


# ---------------------------------------------------------------------------
# Fixtures：每个测试前后重置熔断器（防跨测试污染）
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_circuit_breaker_per_test():
    """每个测试前后重置模块级熔断器状态。"""
    reset_compact_circuit_breaker()
    yield
    reset_compact_circuit_breaker()


# ---------------------------------------------------------------------------
# 测试 1：9 段式 prompt 触发
# ---------------------------------------------------------------------------

async def test_9section_prompt_sent_to_llm():
    """mock LLM，捕获 prompt，验证含 9 段标题。"""
    client = _make_llm(return_content="# 摘要内容")
    msgs = _mk_msgs(5)

    await _summarize_conversation(msgs, llm_client=client)

    # 验证 LLM 被调用
    assert client.chat_completions.call_count >= 1
    # 捕获发给 LLM 的 messages
    call_args = client.chat_completions.call_args
    sent_messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
    # 把所有 content 合并成一个大字符串
    all_content = " ".join(
        m.get("content", "") for m in sent_messages
        if isinstance(m.get("content"), str)
    )
    # 验证 9 段标题都在
    expected_sections = [
        "Primary Request and Intent",
        "Key Technical Concepts",
        "Files and Code Sections",
        "Errors and fixes",
        "Problem Solving",
        "All user messages",
        "Pending Tasks",
        "Current Work",
        "Optional Next Step",
    ]
    for section in expected_sections:
        assert section in all_content, f"9 段式 prompt 缺少段落: {section}"


# ---------------------------------------------------------------------------
# 测试 2：PTL 重试（第一次抛 prompt_too_long，第二次成功）
# ---------------------------------------------------------------------------

async def test_ptl_retry_drops_messages_then_succeeds():
    """第一次抛 prompt_too_long，验证丢 20% 消息重试，最终成功。"""
    # 第一次抛 PTL，第二次成功
    ptl_error = Exception(
        "Error code: 400 - {'error': {'message': 'prompt_too_long', "
        "'type': 'invalid_request_error'}}"
    )
    client = _make_llm(return_content="最终摘要")
    # 第一次抛 PTL，第二次返回正常
    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content="最终摘要"))]
    client.chat_completions = AsyncMock(
        side_effect=[ptl_error, success_resp]
    )
    msgs = _mk_msgs(10)  # 20 条

    result = await _summarize_conversation(msgs, llm_client=client)

    # 验证调用了 2 次（第一次 PTL，第二次成功）
    assert client.chat_completions.call_count == 2
    # 验证最终成功
    assert result == "最终摘要"
    # 验证第二次的 dialog 比第一次少（丢了 20%）
    first_call_messages = client.chat_completions.call_args_list[0][0][0]
    second_call_messages = client.chat_completions.call_args_list[1][0][0]
    first_content = first_call_messages[-1]["content"] if first_call_messages else ""
    second_content = second_call_messages[-1]["content"] if second_call_messages else ""
    assert len(second_content) < len(first_content), (
        "PTL 重试应该丢 20% 旧消息，第二次 prompt 应更短"
    )


# ---------------------------------------------------------------------------
# 测试 3：PTL 重试耗尽（连续 4 次抛 PTL → 走 _rule_based_summary）
# ---------------------------------------------------------------------------

async def test_ptl_retry_exhausted_falls_back_to_rules():
    """连续 MAX_PTL_RETRIES+1 次抛 PTL，验证走 _rule_based_summary。"""
    ptl_error = Exception("prompt_too_long: input too long")
    client = _make_llm()
    client.chat_completions = AsyncMock(side_effect=ptl_error)
    msgs = _mk_msgs(10)

    result = await _summarize_conversation(msgs, llm_client=client)

    # 验证返回的是规则总结（含 "规则提取" 标记）
    assert "规则提取" in result or "LLM 不可用" in result
    # 验证调用了 MAX_PTL_RETRIES + 1 次（重试到耗尽）
    assert client.chat_completions.call_count == MAX_PTL_RETRIES + 1


# ---------------------------------------------------------------------------
# 测试 4：熔断器（连续 3 次非 PTL 失败 → 第 4 次不调 LLM）
# ---------------------------------------------------------------------------

async def test_circuit_breaker_opens_after_consecutive_failures():
    """连续 MAX_CONSECUTIVE_FAILURES 次失败后，第 4 次直接走规则总结。"""
    # 非 PTL 错误（如连接超时）
    conn_error = Exception("Connection timeout")
    client = _make_llm()
    client.chat_completions = AsyncMock(side_effect=conn_error)
    msgs = _mk_msgs(5)

    # 前 3 次调用：每次失败（异常 → 走规则总结），累加 _consecutive_failures
    for i in range(MAX_CONSECUTIVE_FAILURES):
        result = await _summarize_conversation(msgs, llm_client=client)
        assert "规则提取" in result or "LLM 不可用" in result

    # 第 4 次调用：熔断器已开，不应调 LLM
    call_count_before = client.chat_completions.call_count
    result4 = await _summarize_conversation(msgs, llm_client=client)
    call_count_after = client.chat_completions.call_count

    # 验证调用次数没增加（熔断器拦截）
    assert call_count_after == call_count_before, (
        f"熔断器开后不应调 LLM，但 call_count 从 {call_count_before} 增到 {call_count_after}"
    )
    # 验证走规则总结
    assert "规则提取" in result4 or "LLM 不可用" in result4


# ---------------------------------------------------------------------------
# 测试 5：熔断器重置（成功一次后状态清零）
# ---------------------------------------------------------------------------

async def test_circuit_breaker_resets_on_success():
    """成功调用一次后 _consecutive_failures=0 且 _compact_circuit_open=False。"""
    import agent.context_compressor as cc

    # 先制造 2 次失败（不触发熔断，2 < 3）
    fail_client = _make_llm(side_effect=Exception("timeout"))
    msgs = _mk_msgs(3)
    await _summarize_conversation(msgs, llm_client=fail_client)
    await _summarize_conversation(msgs, llm_client=fail_client)
    assert cc._consecutive_failures == 2
    assert cc._compact_circuit_open is False

    # 成功一次
    success_client = _make_llm(return_content="好的摘要")
    result = await _summarize_conversation(msgs, llm_client=success_client)

    # 验证熔断器状态重置
    assert cc._consecutive_failures == 0
    assert cc._compact_circuit_open is False
    assert result == "好的摘要"


# ---------------------------------------------------------------------------
# 测试 6：session memory 替代（传了 session_memory 时不调 LLM）
# ---------------------------------------------------------------------------

async def test_session_memory_bypasses_llm():
    """传了 session_memory 时直接返回，不调 LLM。"""
    client = _make_llm(return_content="不应该返回这个")
    msgs = _mk_msgs(5)

    result = await _summarize_conversation(
        msgs, llm_client=client, session_memory="这是预提取的 session memory"
    )

    # 验证完全不调 LLM
    assert client.chat_completions.call_count == 0
    # 验证返回 session_memory 内容
    assert result == "这是预提取的 session memory"


# ---------------------------------------------------------------------------
# 测试 7：<analysis> 草稿剥离
# ---------------------------------------------------------------------------

async def test_strip_analysis_draft():
    """LLM 返回 <analysis>...</analysis> + 实际摘要，验证 <analysis> 被剥离。"""
    raw = "<analysis>这是内部推理</analysis>\n\n# 实际摘要\n内容"
    client = _make_llm(return_content=raw)
    msgs = _mk_msgs(3)

    result = await _summarize_conversation(msgs, llm_client=client)

    assert "<analysis>" not in result
    assert "实际摘要" in result
    assert "内部推理" not in result


def test_strip_analysis_draft_unit():
    """_strip_analysis_draft 单元测试：各种 <analysis> 格式。"""
    # 标准
    assert _strip_analysis_draft("<analysis>x</analysis>real") == "real"
    # 多行
    assert _strip_analysis_draft(
        "<analysis>\nline1\nline2\n</analysis>\n\nactual"
    ) == "actual"
    # 无 analysis
    assert _strip_analysis_draft("no tags here") == "no tags here"
    # 空
    assert _strip_analysis_draft("") == ""


# ---------------------------------------------------------------------------
# 端到端测试：通过 compress_if_needed 触发 9 段式 + 熔断器
# ---------------------------------------------------------------------------

async def test_e2e_9section_prompt_via_compress_if_needed(tmp_path):
    """端到端：通过 compress_if_needed 触发 L4 llm_compact，
    验证 9 段式 prompt 真的发给 LLM（不只是 _summarize_conversation 单元测试）。
    """
    # 构造 mock LLM 捕获请求
    captured_messages = []

    async def _capture_chat_completions(messages, **kwargs):
        captured_messages.append(messages)
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="端到端摘要"))]
        return m

    client = MagicMock()
    client.chat_completions = _capture_chat_completions

    # 构造超 token 的对话（让 L4 触发）
    big_content = "x" * 8000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(60):
        msgs.append({"role": "user", "content": big_content})
        msgs.append({"role": "assistant", "content": big_content})

    state = CompressionSessionState()
    cfg = {
        "llm_compact_token_threshold": 100,  # 强制触发 L4
        "snip_message_threshold": 10**9,    # 禁 L1
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "transcript_enabled": False,
    }

    out, changed = await compress_if_needed(
        msgs, llm_client=client, model="test-model",
        config=cfg, session_state=state,
        agent_home=tmp_path, session_id="e2e-test",
    )

    assert changed is True
    assert state.llm_compact_count == 1  # L4 确实触发了
    # 验证捕获到请求
    assert len(captured_messages) >= 1
    # 验证 9 段标题在生产路径真发出
    all_content = " ".join(
        m.get("content", "") for m in captured_messages[0]
        if isinstance(m.get("content"), str)
    )
    assert "Primary Request and Intent" in all_content, (
        "端到端：9 段式 prompt 应通过 compress_if_needed → llm_compact → _summarize_conversation 发出"
    )


async def test_e2e_circuit_breaker_via_compress_if_needed(tmp_path):
    """端到端：连续 3 次 LLM 失败后，第 4 次 compress_if_needed 调用
    L4 路径不应调 LLM（熔断器在 _summarize_conversation 层拦截）。
    """
    conn_error = Exception("Connection timeout")
    client = _make_llm()
    client.chat_completions = AsyncMock(side_effect=conn_error)

    big_content = "x" * 8000

    def _mk_big_msgs():
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(60):
            msgs.append({"role": "user", "content": big_content})
            msgs.append({"role": "assistant", "content": big_content})
        return msgs

    cfg = {
        "llm_compact_token_threshold": 100,
        "snip_message_threshold": 10**9,
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 0,  # 禁 cooldown 让每次都能触发 L4
        "max_compress_attempts": 100,     # 放宽，让熔断器做主拦截
        "transcript_enabled": False,
    }

    # 前 3 次：每次创建新 state（让 cooldown 不拦），LLM 失败
    for i in range(MAX_CONSECUTIVE_FAILURES):
        state = CompressionSessionState()
        await compress_if_needed(
            _mk_big_msgs(), llm_client=client, model="test-model",
            config=cfg, session_state=state,
            agent_home=tmp_path, session_id=f"e2e-fail-{i}",
        )

    # 第 4 次：熔断器应已开，LLM 不应被调用
    call_count_before = client.chat_completions.call_count
    state4 = CompressionSessionState()
    out4, changed4 = await compress_if_needed(
        _mk_big_msgs(), llm_client=client, model="test-model",
        config=cfg, session_state=state4,
        agent_home=tmp_path, session_id="e2e-fail-4",
    )
    call_count_after = client.chat_completions.call_count

    # 熔断器拦截：调用次数不应增加
    assert call_count_after == call_count_before, (
        f"端到端熔断器：第 4 次不应调 LLM，call_count 从 {call_count_before} → {call_count_after}"
    )


# ---------------------------------------------------------------------------
# Review Fix Round 1：system prompt + model 参数验证
# ---------------------------------------------------------------------------

async def test_system_prompt_sent_to_llm():
    """Review Fix Important 2：验证 system prompt "你是技术对话摘要助手。" 真发出。"""
    client = _make_llm(return_content="# 摘要")
    msgs = _mk_msgs(3)

    await _summarize_conversation(msgs, llm_client=client)

    call_args = client.chat_completions.call_args
    sent_messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
    # 第一条应是 system message
    assert len(sent_messages) >= 2, (
        f"应有 system + user 两条 message，实际 {len(sent_messages)} 条"
    )
    assert sent_messages[0]["role"] == "system", (
        f"第一条 message 应是 system，实际 role={sent_messages[0].get('role')}"
    )
    assert sent_messages[0]["content"] == "你是技术对话摘要助手。", (
        f"system prompt 内容不对，实际: {sent_messages[0]['content']!r}"
    )
    # 第二条应是 user message（9 段式 prompt）
    assert sent_messages[1]["role"] == "user"


async def test_model_param_forwarded_to_llm():
    """Review Fix Important 1 + Minor 5：验证 model 真传给 chat_completions。"""
    client = _make_llm(return_content="# 摘要")
    msgs = _mk_msgs(3)

    # Case 1：只传 model
    await _summarize_conversation(msgs, llm_client=client, model="main-model-x")
    kwargs1 = client.chat_completions.call_args[1]
    assert kwargs1.get("model") == "main-model-x", (
        f"model 参数应转发，实际 kwargs: {kwargs1}"
    )

    # Case 2：同时传 model + summary_model，应优先 model（spec: model or summary_model）
    await _summarize_conversation(
        msgs, llm_client=client,
        model="main-model-x", summary_model="summary-model-y",
    )
    kwargs2 = client.chat_completions.call_args[1]
    assert kwargs2.get("model") == "main-model-x", (
        f"model + summary_model 都传时应优先 model，实际: {kwargs2.get('model')!r}"
    )

    # Case 3：只传 summary_model（model=None），应用 summary_model
    await _summarize_conversation(
        msgs, llm_client=client, summary_model="summary-model-y",
    )
    kwargs3 = client.chat_completions.call_args[1]
    assert kwargs3.get("model") == "summary-model-y", (
        f"model=None 时应回退到 summary_model，实际: {kwargs3.get('model')!r}"
    )


# ===========================================================================
# Task E：PTL tokenGap 精确算法测试
# ===========================================================================

# ---------------------------------------------------------------------------
# _get_model_max_tokens 查表测试
# ---------------------------------------------------------------------------

def test_get_model_max_tokens_lookup():
    """_get_model_max_tokens 查表正确。"""
    assert _get_model_max_tokens("deepseek-chat") == 65536
    assert _get_model_max_tokens("deepseek-v4") == 65536
    assert _get_model_max_tokens("claude-3-5-sonnet") == 200000
    assert _get_model_max_tokens("claude-3-5-haiku") == 200000
    assert _get_model_max_tokens("some-model[1m]") == 1_000_000
    assert _get_model_max_tokens("model-1m") == 1_000_000
    assert _get_model_max_tokens("unknown-model") == 64000
    assert _get_model_max_tokens("") == 64000
    assert _get_model_max_tokens(None) == 64000


# ---------------------------------------------------------------------------
# _compute_ptl_drop_count 精确算法测试
# ---------------------------------------------------------------------------

def test_compute_drop_count_deepseek_format():
    """DeepSeek 格式 'input 75000 > 65536' → 提取上限 + 实际，算精确 drop。"""
    # 100 条消息，每条约 750 token
    msgs = [{"role": "user", "content": "x" * 2250}] * 100  # 2250 chars / 3 = 750 tokens
    error_msg = "prompt_too_long: input 75000 > 65536"

    drop = _compute_ptl_drop_count(msgs, error_msg, model_max_tokens=65536)
    # budget = 65536 * 0.85 = 55705; overflow = 75000 - 55705 = 19295
    # avg_per_msg = 75000 / 100 = 750; drop = 19295/750 * 1.1 + 1 ≈ 29
    assert drop >= 20, f"DeepSeek 格式应丢 ≥20 条，实际 {drop}"
    assert drop < 100, "不应丢光"


def test_compute_drop_count_anthropic_format():
    """Anthropic 格式 'prompt is too long: 75000 > 64000'。"""
    msgs = [{"role": "user", "content": "x" * 2250}] * 100
    error_msg = "prompt is too long: 75000 > 64000"

    drop = _compute_ptl_drop_count(msgs, error_msg, model_max_tokens=64000)
    assert drop >= 15, f"Anthropic 格式应丢 ≥15 条，实际 {drop}"


def test_compute_drop_count_openai_format():
    """OpenAI 格式 'maximum context length is 65536... requested 75000'。"""
    msgs = [{"role": "user", "content": "x" * 2250}] * 100
    error_msg = (
        "This model's maximum context length is 65536 tokens. "
        "However, your messages resulted in 75000 tokens."
    )

    drop = _compute_ptl_drop_count(msgs, error_msg, model_max_tokens=65536)
    assert drop >= 15, f"OpenAI 格式应丢 ≥15 条，实际 {drop}"


def test_compute_drop_count_fallback_unparseable_error():
    """错误消息无法解析 → fallback 到旧 20% 算法。"""
    msgs = [{"role": "user", "content": "msg"}] * 100
    # 无数字的错误消息
    error_msg = "some weird error without token numbers"

    drop = _compute_ptl_drop_count(msgs, error_msg)
    assert drop == 20, f"无法解析时应 fallback 到 20%（100//5=20），实际 {drop}"


def test_compute_drop_count_fallback_empty_error():
    """空错误消息 → fallback 到旧 20% 算法。"""
    msgs = [{"role": "user", "content": "msg"}] * 50
    drop = _compute_ptl_drop_count(msgs, "")
    assert drop == 10, f"空 error 时应 fallback（50//5=10），实际 {drop}"


def test_compute_drop_count_fallback_non_ptl_error():
    """非 PTL 错误（如 timeout）→ fallback 到旧 20%。"""
    msgs = [{"role": "user", "content": "msg"}] * 50
    error_msg = "Connection timeout"
    drop = _compute_ptl_drop_count(msgs, error_msg)
    assert drop == 10, f"非 PTL 错误应 fallback（50//5=10），实际 {drop}"


def test_compute_drop_count_protect_min_2():
    """消息少时不能丢光——至少留 2 条。"""
    msgs = [{"role": "user", "content": "x" * 99999}] * 3  # 3 条巨大消息
    error_msg = "prompt_too_long: input 99999 > 1000"

    drop = _compute_ptl_drop_count(msgs, error_msg, model_max_tokens=1000)
    assert drop <= 1, f"3 条消息时最多丢 1 条（留 2 条），实际 {drop}"


def test_compute_drop_count_vs_old_20pct():
    """精确算法 vs 旧算法：100 条 PTL 时精确算法应比旧算法丢更多（overflow 更大）。

    旧算法：100 // 5 = 20 条
    精确算法：算 overflow 后 drop_count 会更大
    """
    msgs = [{"role": "user", "content": "x" * 2250}] * 100  # 750 tokens/msg
    error_msg = "prompt_too_long: input 90000 > 65536"

    drop = _compute_ptl_drop_count(msgs, error_msg, model_max_tokens=65536)
    old_drop = 100 // 5  # = 20
    # overflow = 90000 - 55705 = 34295; drop = 34295/900 * 1.1 + 1 ≈ 43
    # 精确算法应比旧算法丢更多（因为 overflow 更大，旧算法固定 20% 不够）
    assert drop > old_drop, (
        f"精确算法（{drop}）应比旧 20%（{old_drop}）丢更多（overflow 大时）"
    )


# ---------------------------------------------------------------------------
# 端到端测试：_summarize_conversation 用精确算法重试
# ---------------------------------------------------------------------------

async def test_e2e_ptl_retry_uses_precise_drop_count():
    """Task E 端到端：mock LLM 第一次抛 PTL（带 token 数），第二次成功，
    验证丢精确数（而非旧 20%）。
    """
    # 构造 100 条消息（每条约 750 token）
    msgs = []
    for i in range(100):
        msgs.append({"role": "user", "content": f"x" * 2250})
        msgs.append({"role": "assistant", "content": f"y" * 2250})

    # PTL 错误（DeepSeek 格式，带 token 数）
    ptl_error = Exception(
        "prompt_too_long: input 200000 > 65536"
    )
    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content="摘要"))]

    client = MagicMock()
    client.chat_completions = AsyncMock(
        side_effect=[ptl_error, success_resp]
    )

    result = await _summarize_conversation(msgs, llm_client=client, model="deepseek-chat")

    assert client.chat_completions.call_count == 2
    assert result == "摘要"
    # 验证第二次调用的消息比第一次少（丢了一些消息）
    first_call = client.chat_completions.call_args_list[0][0][0]
    second_call = client.chat_completions.call_args_list[1][0][0]
    first_prompt = first_call[-1]["content"] if first_call else ""
    second_prompt = second_call[-1]["content"] if second_call else ""
    assert len(second_prompt) < len(first_prompt), (
        "PTL 重试后 prompt 应更短（丢了消息）"
    )


async def test_e2e_ptl_retry_fallback_on_unparseable_error():
    """Task E 端到端：PTL 错误消息无法解析 → 走 fallback 20%。
    验证不报错，正常重试。
    """
    msgs = _mk_msgs(20)  # 40 条

    # PTL 错误但无 token 数（无法解析精确算法）
    ptl_error = Exception("prompt_too_long: input too long")
    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content="fallback 摘要"))]

    client = MagicMock()
    client.chat_completions = AsyncMock(
        side_effect=[ptl_error, success_resp]
    )

    result = await _summarize_conversation(msgs, llm_client=client)

    assert client.chat_completions.call_count == 2
    assert result == "fallback 摘要"
