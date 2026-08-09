"""call_with_retry async 行为测试。

本测试覆盖 async 改造后的 call_with_retry：
  - 是 coroutine function
  - llm_client.chat_completions 被 await
  - 429/5xx 触发重试（asyncio.sleep 替代 time.sleep）
  - 不可重试错误立即抛
  - fallback client 切换
  - 529 连续阈值早切
  - 退避抖动（jitter）

注意：改造后 call_with_retry 内部用 asyncio.sleep，因此测试 monkeypatch
agent.llm_retry.asyncio.sleep（不是 time.sleep）。
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm_retry import (
    call_with_retry,
    DEFAULT_CONSECUTIVE_529_THRESHOLD,
)


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------

def _make_api_error(status_code):
    """构造 openai.APIStatusError 兼容的假错误（避免真实 HTTP 请求）。"""
    try:
        from openai import APIStatusError
        err = APIStatusError.__new__(APIStatusError)
        err.status_code = status_code
        return err
    except ImportError:
        return None


def _mock_llm_response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )],
    )


def _mock_llm_client_async(response=None, side_effect=None):
    """构造 mock LLMClient，其 chat_completions 是 AsyncMock（async 方法）。

    call_with_retry 改 async 后会 await llm_client.chat_completions(...)，
    因此 mock 必须是 AsyncMock。
    """
    client = MagicMock()
    if side_effect is not None:
        client.chat_completions = AsyncMock(side_effect=side_effect)
    else:
        client.chat_completions = AsyncMock(return_value=response or _mock_llm_response())
    return client


# ---------------------------------------------------------------------------
# 契约测试：call_with_retry 必须是 coroutine function
# ---------------------------------------------------------------------------

def test_call_with_retry_is_coroutine():
    """call_with_retry 必须是 async def（coroutine function）。"""
    assert inspect.iscoroutinefunction(call_with_retry), \
        "call_with_retry 必须是 async def"


# ---------------------------------------------------------------------------
# 行为测试：await + retry + fallback + 529 + jitter
# ---------------------------------------------------------------------------

async def test_call_succeeds_first_try():
    """第一次就成功，chat_completions 被 await 一次。"""
    llm_client = _mock_llm_client_async(_mock_llm_response())
    result = await call_with_retry(llm_client, [])
    assert result is not None
    assert llm_client.chat_completions.await_count == 1


async def test_chat_completions_is_awaited_not_called():
    """llm_client.chat_completions 必须被 await（assert_awaited），不是裸调用。"""
    llm_client = _mock_llm_client_async(_mock_llm_response())
    await call_with_retry(llm_client, [])
    llm_client.chat_completions.assert_awaited_once()


async def test_call_retries_on_429(monkeypatch):
    """429 触发重试，最终成功。asyncio.sleep 被 patch 掉避免真睡。"""
    async_sleeps = []
    async def _fake_sleep(s):
        async_sleeps.append(s)
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=[
        err429, err429, _mock_llm_response(),
    ])

    result = await call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)
    assert result is not None
    assert llm_client.chat_completions.await_count == 3


async def test_call_not_retryable_raises_immediately(monkeypatch):
    """不可重试错误（400）立即抛，不进重试。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err400 = _make_api_error(400)
    llm_client = _mock_llm_client_async(side_effect=[err400])

    with pytest.raises(Exception):
        await call_with_retry(llm_client, [], max_retries=5)

    assert llm_client.chat_completions.await_count == 1


async def test_call_exhausts_retries_then_raises(monkeypatch):
    """可重试错误重试耗尽后抛。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=err429)

    with pytest.raises(Exception):
        await call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)

    assert llm_client.chat_completions.await_count == 3


async def test_call_falls_back_to_fallback_client(monkeypatch):
    """主 client 失败耗尽后切到 fallback client（也被 await）。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    main_client = _mock_llm_client_async(side_effect=err429)
    fallback_client = _mock_llm_client_async(_mock_llm_response("fallback-ok"))

    result = await call_with_retry(
        main_client, [],
        max_retries=3, initial_backoff=0.01,
        fallback_llm_client=fallback_client,
    )

    assert main_client.chat_completions.await_count == 3
    assert fallback_client.chat_completions.await_count == 1
    assert result is not None


async def test_consecutive_529_switches_to_fallback_early(monkeypatch):
    """连续 N 次 529 立即切 fallback（不等耗尽）。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err529 = _make_api_error(529)
    main_client = _mock_llm_client_async(side_effect=[err529, err529, err529])
    fallback_client = _mock_llm_client_async(_mock_llm_response("fallback-ok"))

    result = await call_with_retry(
        main_client, [],
        max_retries=10,
        initial_backoff=0.001,
        fallback_llm_client=fallback_client,
        consecutive_529_threshold=3,
    )

    assert main_client.chat_completions.await_count == 3
    assert fallback_client.chat_completions.await_count == 1
    assert result is not None


async def test_backoff_has_jitter(monkeypatch):
    """退避 = base + jitter（asyncio.sleep 收到的值校验）。"""
    async_sleeps = []
    async def _fake_sleep(s):
        async_sleeps.append(s)
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)
    # 固定 random.uniform 返回最大抖动，便于断言
    monkeypatch.setattr("agent.llm_retry.random.uniform", lambda a, b: b)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=[
        err429, err429, _mock_llm_response(),
    ])
    await call_with_retry(llm_client, [], max_retries=3, initial_backoff=1.0)

    # attempt 0: base=1.0, sleep=1.0 + 0.25 = 1.25
    # attempt 1: base=2.0, sleep=2.0 + 0.50 = 2.50
    assert async_sleeps == [1.25, 2.50]


async def test_max_tokens_is_passed_through():
    """max_tokens 透传给 chat_completions。"""
    llm_client = _mock_llm_client_async(_mock_llm_response())
    await call_with_retry(llm_client, [], max_tokens=8192)
    kwargs = llm_client.chat_completions.call_args.kwargs
    assert kwargs.get("max_tokens") == 8192


async def test_no_max_tokens_not_passed():
    """不传 max_tokens 时不会把 max_tokens=None 传给 SDK。"""
    llm_client = _mock_llm_client_async(_mock_llm_response())
    await call_with_retry(llm_client, [])
    kwargs = llm_client.chat_completions.call_args.kwargs
    assert "max_tokens" not in kwargs
