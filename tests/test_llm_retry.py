"""LLM 重试与错误恢复测试（适配新 llm_client 接口 + async call_with_retry）。

改造说明（T_D1 fix）：call_with_retry 改 async 后：
  - 调用方加 await（测试函数改 async def）
  - time.sleep → asyncio.sleep（monkeypatch 改 agent.llm_retry.asyncio.sleep）
  - llm_client.chat_completions 用 AsyncMock（被 await）
  - .call_count → .await_count
"""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm_retry import (
    is_retryable, get_retry_after, call_with_retry,
    DEFAULT_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# is_retryable（纯同步，不涉及 async）
# ---------------------------------------------------------------------------

def _make_api_error(status_code):
    try:
        from openai import APIStatusError
        err = APIStatusError.__new__(APIStatusError)
        err.status_code = status_code
        return err
    except ImportError:
        return None


def test_retryable_429():
    err = _make_api_error(429)
    if err:
        assert is_retryable(err) is True


def test_retryable_500():
    err = _make_api_error(500)
    if err:
        assert is_retryable(err) is True


def test_not_retryable_400():
    err = _make_api_error(400)
    if err:
        assert is_retryable(err) is False


def test_retryable_timeout_by_name():
    class MyTimeout(Exception):
        pass
    assert is_retryable(MyTimeout()) is True


# ---------------------------------------------------------------------------
# get_retry_after（纯同步）
# ---------------------------------------------------------------------------

def test_retry_after_from_attribute():
    err = SimpleNamespace(retry_after=2.5)
    assert get_retry_after(err) == 2.5


def test_retry_after_none():
    err = SimpleNamespace(retry_after=None)
    assert get_retry_after(err) is None


# ---------------------------------------------------------------------------
# 辅助：构造 async mock LLM client
# ---------------------------------------------------------------------------

def _mock_llm_response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
        )],
    )


def _mock_llm_client(response=None, side_effect=None):
    """构造 mock LLMClient，chat_completions 是 AsyncMock（call_with_retry 会 await）。"""
    client = MagicMock()
    if side_effect is not None:
        client.chat_completions = AsyncMock(side_effect=side_effect)
    else:
        client.chat_completions = AsyncMock(return_value=response or _mock_llm_response())
    return client


def _patch_sleep_noop(monkeypatch):
    """patch asyncio.sleep 成空操作（async 版）。

    call_with_retry 改 async 后用 await asyncio.sleep(...)，
    所以 monkeypatch agent.llm_retry.asyncio.sleep。
    """
    async def _no_op(_s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _no_op)


def _patch_sleep_collect(monkeypatch, sink: list):
    """patch asyncio.sleep，把收到的 sleep 值收集到 sink 列表。"""
    async def _collector(s):
        sink.append(s)
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _collector)


# ---------------------------------------------------------------------------
# call_with_retry（async，新接口：llm_client.chat_completions）
# ---------------------------------------------------------------------------

async def test_call_succeeds_first_try():
    """第一次就成功，不重试。"""
    llm_client = _mock_llm_client(_mock_llm_response())
    result = await call_with_retry(llm_client, [])
    assert result is not None
    assert llm_client.chat_completions.await_count == 1


async def test_call_retries_on_429(monkeypatch):
    """429 错误触发重试，最终成功。"""
    _patch_sleep_noop(monkeypatch)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=[
        err429, err429, _mock_llm_response(),
    ])

    result = await call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)
    assert result is not None
    assert llm_client.chat_completions.await_count == 3


async def test_call_not_retryable_raises_immediately(monkeypatch):
    """不可重试错误立即抛。"""
    _patch_sleep_noop(monkeypatch)
    err400 = _make_api_error(400)
    llm_client = _mock_llm_client(side_effect=[err400])

    with pytest.raises(Exception):
        await call_with_retry(llm_client, [], max_retries=5)

    assert llm_client.chat_completions.await_count == 1


async def test_call_exhausts_retries_then_raises(monkeypatch):
    """可重试错误重试耗尽后抛。"""
    _patch_sleep_noop(monkeypatch)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=err429)

    with pytest.raises(Exception):
        await call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)

    assert llm_client.chat_completions.await_count == 3


async def test_call_falls_back_to_fallback_client(monkeypatch):
    """主 client 失败后切换备用 client。"""
    _patch_sleep_noop(monkeypatch)
    err429 = _make_api_error(429)
    main_client = _mock_llm_client(side_effect=err429)
    fallback_client = _mock_llm_client(_mock_llm_response("fallback-ok"))

    result = await call_with_retry(
        main_client, [],
        max_retries=3, initial_backoff=0.01,
        fallback_llm_client=fallback_client,
    )

    assert main_client.chat_completions.await_count == 3
    assert fallback_client.chat_completions.await_count == 1


async def test_call_no_fallback_just_raises(monkeypatch):
    """无备用 client 时，主 client 失败直接抛。"""
    _patch_sleep_noop(monkeypatch)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=err429)

    with pytest.raises(Exception):
        await call_with_retry(
            llm_client, [],
            max_retries=2, initial_backoff=0.01,
            fallback_llm_client=None,
        )
    assert llm_client.chat_completions.await_count == 2


# ---------------------------------------------------------------------------
# 退避加抖动（jitter）—— 避免多实例雷击
# ---------------------------------------------------------------------------

async def test_backoff_has_jitter(monkeypatch):
    """退避值 = base + jitter，其中 jitter ∈ [0, base * jitter_ratio]。

    验证 sleep 收到的值严格大于 base（除非 jitter_ratio=0）。
    """
    sleeps = []
    _patch_sleep_collect(monkeypatch, sleeps)
    # 固定 random.uniform 返回最大抖动，便于断言
    monkeypatch.setattr("agent.llm_retry.random.uniform", lambda a, b: b)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=[
        err429, err429, _mock_llm_response(),
    ])
    await call_with_retry(llm_client, [], max_retries=3, initial_backoff=1.0)

    # attempt 0: base=1.0, sleep=1.0 + 0.25 = 1.25
    # attempt 1: base=2.0, sleep=2.0 + 0.50 = 2.50
    assert sleeps == [1.25, 2.50]


async def test_backoff_jitter_within_range(monkeypatch):
    """jitter ∈ [0, base * 0.25]，sleep ∈ [base, base * 1.25]。"""
    sleeps = []
    _patch_sleep_collect(monkeypatch, sleeps)
    # random.uniform 正常工作（不 patch）

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=[err429, _mock_llm_response()])
    await call_with_retry(llm_client, [], max_retries=2, initial_backoff=1.0)

    assert len(sleeps) == 1
    # base=1.0, jitter∈[0, 0.25]，sleep∈[1.0, 1.25]
    assert 1.0 <= sleeps[0] <= 1.25


async def test_backoff_zero_jitter_falls_back_to_pure_exponential(monkeypatch):
    """jitter_ratio=0 时退避等于纯指数（向后兼容）。"""
    sleeps = []
    _patch_sleep_collect(monkeypatch, sleeps)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=[err429, _mock_llm_response()])
    await call_with_retry(
        llm_client, [], max_retries=2,
        initial_backoff=1.0, jitter_ratio=0.0,
    )
    assert sleeps == [1.0]


async def test_retry_after_overrides_backoff_but_keeps_jitter(monkeypatch):
    """有 Retry-After header 时用其值，但仍加抖动。"""
    sleeps = []
    _patch_sleep_collect(monkeypatch, sleeps)
    monkeypatch.setattr("agent.llm_retry.random.uniform", lambda a, b: b)

    err = SimpleNamespace(
        retry_after=5.0,
        response_headers={"retry-after": "5"},
    )
    # 让 is_retryable 认为可重试：兜底走 type 名匹配
    class _RetryableErr(Exception):
        pass
    # _RetryableErr 名字不含 timeout/connection/temporary，得手动构造 openai 风
    err429 = _make_api_error(429)
    if err429:
        err429.retry_after = 5.0
        err429.response_headers = {"retry-after": "5"}
        to_raise = err429
    else:
        to_raise = err

    llm_client = _mock_llm_client(side_effect=[to_raise, _mock_llm_response()])
    await call_with_retry(llm_client, [], max_retries=2, initial_backoff=1.0)

    # base=5.0（Retry-After），jitter=5.0*0.25=1.25，sleep=6.25
    assert sleeps == [6.25]


# ---------------------------------------------------------------------------
# max_tokens 升级机制（P0-3）（纯同步类，不涉及 async）
# ---------------------------------------------------------------------------

from agent.llm_retry import (
    MaxTokensEscalator, detect_length_finish, DEFAULT_INITIAL_MAX_TOKENS,
    DEFAULT_ESCALATED_MAX_TOKENS,
)


def test_detect_length_finish_true():
    """finish_reason == 'length' → True。"""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length")],
    )
    assert detect_length_finish(resp) is True


def test_detect_length_finish_false_for_stop():
    """finish_reason == 'stop' → False。"""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop")],
    )
    assert detect_length_finish(resp) is False


def test_detect_length_finish_false_for_tool_calls():
    """finish_reason == 'tool_calls' → False（正常工具调用，不是截断）。"""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="tool_calls")],
    )
    assert detect_length_finish(resp) is False


def test_detect_length_finish_handles_malformed_response():
    """response 结构异常时不抛，返回 False（fail-open）。"""
    assert detect_length_finish(None) is False
    assert detect_length_finish(SimpleNamespace(choices=[])) is False
    assert detect_length_finish(SimpleNamespace()) is False


def test_escalator_initial_state():
    """新 escalator 未升级，get_next_max_tokens 返回 None（用 provider 默认）。"""
    esc = MaxTokensEscalator()
    assert esc.has_escalated is False
    assert esc.get_next_max_tokens() is None  # None = 不显式传，让 SDK 用默认


def test_escalator_escalates_once():
    """escalate() 后 has_escalated=True，max_tokens=升级值。"""
    esc = MaxTokensEscalator(initial=4096, escalated=32768)
    new_max = esc.escalate()
    assert new_max == 32768
    assert esc.has_escalated is True
    assert esc.get_next_max_tokens() == 32768


def test_escalator_idempotent():
    """已升级后再 escalate() 不变（防多轮循环重复升级）。"""
    esc = MaxTokensEscalator(initial=4096, escalatedated=32768) if False else MaxTokensEscalator(initial=4096, escalated=32768)
    first = esc.escalate()
    second = esc.escalate()
    assert first == second == 32768


def test_escalator_reset():
    """reset() 后回到初始状态（新会话用）。"""
    esc = MaxTokensEscalator()
    esc.escalate()
    assert esc.has_escalated is True
    esc.reset()
    assert esc.has_escalated is False
    assert esc.get_next_max_tokens() is None


def test_escalator_default_values():
    """默认值：初始 None（用 provider 默认），升级 8 倍常见上限。"""
    esc = MaxTokensEscalator()
    # 默认升级值给一个生产可用的数（不是无限大）
    assert DEFAULT_ESCALATED_MAX_TOKENS >= 16 * 1024  # 至少 16K


# ---------------------------------------------------------------------------
# call_with_retry 透传 max_tokens（async）
# ---------------------------------------------------------------------------

async def test_call_with_retry_passes_max_tokens():
    """call_with_retry 把 max_tokens 透传给 chat_completions。"""
    llm_client = _mock_llm_client(_mock_llm_response())
    await call_with_retry(llm_client, [], max_tokens=8192)
    # AsyncMock 的 call_args_kwargs 拿到 kwargs
    kwargs = llm_client.chat_completions.call_args.kwargs
    assert kwargs.get("max_tokens") == 8192


async def test_call_with_retry_without_max_tokens_does_not_pass_it():
    """不传 max_tokens 时，chat_completions 也不会收到 max_tokens=None（避免 SDK 误判）。"""
    llm_client = _mock_llm_client(_mock_llm_response())
    await call_with_retry(llm_client, [])
    kwargs = llm_client.chat_completions.call_args.kwargs
    assert "max_tokens" not in kwargs


# ---------------------------------------------------------------------------
# P1-1: 529 连续失败精确切模型（避免浪费重试次数）（async）
# ---------------------------------------------------------------------------

async def test_consecutive_529_switches_to_fallback_early(monkeypatch):
    """连续 N 次 529 立即切 fallback，不等耗尽所有重试。

    场景：主 client 连续返回 529（Anthropic 过载），达到阈值（默认 3）后立即切 fallback。
    避免在已知过载的 endpoint 上浪费重试次数。
    """
    _patch_sleep_noop(monkeypatch)
    err529 = _make_api_error(529)
    main_client = _mock_llm_client(side_effect=[err529, err529, err529])
    fallback_client = _mock_llm_client(_mock_llm_response("fallback-ok"))

    result = await call_with_retry(
        main_client, [],
        max_retries=10,  # 给足重试预算
        initial_backoff=0.001,
        fallback_llm_client=fallback_client,
        consecutive_529_threshold=3,
    )

    # 关键断言：主 client 只调了 3 次（达到阈值就切，没耗完 10 次）
    assert main_client.chat_completions.await_count == 3
    assert fallback_client.chat_completions.await_count == 1
    assert result is not None


async def test_529_below_threshold_keeps_retrying_main(monkeypatch):
    """529 没达到阈值（比如 2 次）时仍在主 client 重试，不切 fallback。"""
    _patch_sleep_noop(monkeypatch)
    err529 = _make_api_error(529)
    err429 = _make_api_error(429)
    ok = _mock_llm_response()
    # 序列：529, 529（未到阈值 3）, 429, 429, ok —— 主 client 重试到成功
    main_client = _mock_llm_client(side_effect=[err529, err529, err429, err429, ok])
    fallback_client = _mock_llm_client(_mock_llm_response("fallback"))

    await call_with_retry(
        main_client, [],
        max_retries=10, initial_backoff=0.001,
        fallback_llm_client=fallback_client,
        consecutive_529_threshold=3,
    )

    # fallback 没被调（529 计数被中间的 429 重置）
    assert fallback_client.chat_completions.await_count == 0


async def test_529_counter_resets_on_non_529_error(monkeypatch):
    """中间出现非 529 错误（如 429）时，529 连续计数清零。"""
    _patch_sleep_noop(monkeypatch)
    err529 = _make_api_error(529)
    err429 = _make_api_error(429)
    ok = _mock_llm_response()
    # 序列：529, 529, 429, 529, ok ——
    #   两次 529 计数到 2（未到 3），429 重置为 0，
    #   再 1 次 529 计数才到 1（远不到 3），ok 成成功。
    #   → 不切 fallback，主 client 走完 5 次。
    main_client = _mock_llm_client(side_effect=[
        err529, err529, err429, err529, ok,
    ])
    fallback_client = _mock_llm_client(_mock_llm_response("fallback"))

    await call_with_retry(
        main_client, [],
        max_retries=20, initial_backoff=0.001,
        fallback_llm_client=fallback_client,
        consecutive_529_threshold=3,
    )

    # 没切 fallback（529 没连续到 3 次）
    assert fallback_client.chat_completions.await_count == 0
    assert main_client.chat_completions.await_count == 5


async def test_529_threshold_disabled_when_set_to_zero(monkeypatch):
    """consecutive_529_threshold=0 时禁用提前切换，走完所有重试。"""
    _patch_sleep_noop(monkeypatch)
    err529 = _make_api_error(529)
    main_client = _mock_llm_client(side_effect=[err529] * 10)
    fallback_client = _mock_llm_client(_mock_llm_response("fallback"))

    await call_with_retry(
        main_client, [],
        max_retries=5, initial_backoff=0.001,
        fallback_llm_client=fallback_client,
        consecutive_529_threshold=0,  # 禁用
    )

    # 走完 5 次主 client，然后才切 fallback
    assert main_client.chat_completions.await_count == 5
    assert fallback_client.chat_completions.await_count == 1


def test_529_threshold_default_value():
    """默认阈值 = 3（从 DEFAULT_CONSECUTIVE_529_THRESHOLD 导出）。"""
    from agent.llm_retry import DEFAULT_CONSECUTIVE_529_THRESHOLD
    assert DEFAULT_CONSECUTIVE_529_THRESHOLD == 3
