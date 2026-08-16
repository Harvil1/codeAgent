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

import httpx
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


# ===== R25 #4：后台调用遇 529 立即放弃（防放大）=====

class FakeConnection529Error(Exception):
    """名字含 connection 让 is_retryable 兜底命中；带 status_code=529。"""
    def __init__(self):
        super().__init__("overloaded")
        self.status_code = 529


class _Always529Client:
    async def chat_completions(self, messages, **kwargs):
        raise FakeConnection529Error()


class _Flaky529Client:
    """第 1 次 529，第 2 次成功。"""
    def __init__(self):
        self.calls = 0

    async def chat_completions(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise FakeConnection529Error()
        class _R:  # 最小响应桩
            class choices:
                class message:
                    content = "ok"
        return _R


class TestBackground529GiveUp:
    async def test_background_raises_immediately(self, monkeypatch):
        import asyncio
        from agent.llm_retry import call_with_retry
        sleeps = []
        async def fake_sleep(s):
            sleeps.append(s)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        with pytest.raises(FakeConnection529Error):
            await call_with_retry(
                _Always529Client(), [{"role": "user", "content": "x"}],
                max_retries=5, background=True,
            )
        assert sleeps == []  # 没有退避 sleep = 没重试

    async def test_foreground_still_retries(self, monkeypatch):
        import asyncio
        from agent.llm_retry import call_with_retry
        async def fake_sleep(s):
            pass
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        client = _Flaky529Client()
        resp = await call_with_retry(
            client, [{"role": "user", "content": "x"}],
            max_retries=5, background=False,
        )
        assert client.calls == 2  # 前台语义不变：重试后成功


# ===== R25 #5：长退避分片心跳 =====

class TestSleepWithHeartbeat:
    async def test_chunks_and_callbacks(self, monkeypatch):
        import asyncio
        from agent.llm_retry import _sleep_with_heartbeat
        slept = []
        async def fake_sleep(s):
            slept.append(s)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        beats = []
        await _sleep_with_heartbeat(70.0, lambda e, t: beats.append(e))
        assert slept == [30.0, 30.0, 10.0]
        assert beats == [30.0, 60.0]  # 结束前每次心跳；最后一片完成不叫

    async def test_no_cb_single_sleep(self, monkeypatch):
        import asyncio
        from agent.llm_retry import _sleep_with_heartbeat
        slept = []
        async def fake_sleep(s):
            slept.append(s)
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await _sleep_with_heartbeat(70.0, None)
        assert slept == [70.0]

    async def test_cb_exception_swallowed(self, monkeypatch):
        import asyncio
        from agent.llm_retry import _sleep_with_heartbeat
        async def fake_sleep(s):
            pass
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        def bad_cb(e, t):
            raise RuntimeError("boom")
        await _sleep_with_heartbeat(65.0, bad_cb)  # 不抛

    async def test_accepts_heartbeat_kwarg(self, monkeypatch):
        """call_with_retry 接受 heartbeat_cb 参数（冒烟）。"""
        import asyncio
        from agent.llm_retry import call_with_retry
        async def fake_sleep(s):
            pass
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        client = _Flaky529Client()
        await call_with_retry(
            client, [{"role": "user", "content": "x"}],
            max_retries=3, heartbeat_cb=lambda e, t: None,
        )


# ===== R26 #10：连接重置 → 重建 client 再重试 =====

class TestConnectionResetRecovery:
    async def test_is_connection_reset_by_cause(self):
        from agent.llm_retry import _is_connection_reset
        err = httpx.TransportError("boom")
        err.__cause__ = ConnectionResetError("reset by peer")
        assert _is_connection_reset(err) is True

    def test_is_connection_reset_by_name(self):
        from agent.llm_retry import _is_connection_reset
        class RemoteProtocolError(Exception):
            pass
        assert _is_connection_reset(RemoteProtocolError()) is True

    def test_not_connection_reset(self):
        from agent.llm_retry import _is_connection_reset
        assert _is_connection_reset(ValueError("x")) is False

    async def test_retry_calls_reset_client(self, monkeypatch):
        """连接重置后重试前调用了 reset_client。"""
        import asyncio
        from agent.llm_retry import call_with_retry

        class FlakyConnClient:
            def __init__(self):
                self.calls = 0
                self.resets = 0

            def reset_client(self):
                self.resets += 1

            async def chat_completions(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    err = httpx.TransportError("broken")
                    err.__cause__ = ConnectionResetError()
                    raise err
                class _R:
                    class choices:
                        class message:
                            content = "ok"
                return _R

        client = FlakyConnClient()
        async def fake_sleep(s):
            pass
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await call_with_retry(client, [{"role": "user", "content": "x"}], max_retries=3)
        assert client.resets == 1
        assert client.calls == 2

    async def test_base_client_reset_noop(self):
        from agent.llm_client import LLMClient
        LLMClient().reset_client()  # 不抛即过
