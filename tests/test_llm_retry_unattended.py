"""call_with_retry 持久重试（unattended）模式测试。

Task P2.1：当 `is_feature_enabled(config, "bash_unattended_retry")` 开启时：
  - max_retries 视为无上限（持续重试，不因计数耗尽退出）
  - 加 deadline（time.monotonic 起 + max_hours*3600），超过 deadline 退出
  - 普通 5xx/429 重试逻辑保留
  - flag OFF（或 config=None）→ 完全走原有 max_retries 逻辑不变

测试覆盖：
  1. flag ON + 429 持续 30 次不退出（最终成功）
  2. flag ON + 超过 max_hours（用 max_hours=0 立即到期模拟）→ 退出并抛错
  3. flag OFF → 原 max_retries 逻辑不变（3 次重试耗尽）
  4. config=None → 原 max_retries 逻辑不变（等价于 flag OFF）
  5. flag ON 但 400 不可重试错误仍立即抛（deadline 不影响不可重试语义）
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm_retry import call_with_retry


# ---------------------------------------------------------------------------
# 辅助构造（复用 test_llm_retry_async.py 的模式）
# ---------------------------------------------------------------------------

def _make_api_error(status_code):
    """构造 openai.APIStatusError 兼容的假错误。"""
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
    client = MagicMock()
    if side_effect is not None:
        client.chat_completions = AsyncMock(side_effect=side_effect)
    else:
        client.chat_completions = AsyncMock(return_value=response or _mock_llm_response())
    return client


def _flag_on_config(max_hours=24):
    """构造 bash_unattended_retry=ON 的 config。"""
    return {
        "features": {
            "bash_unattended_retry": {
                "enabled": True,
                "max_hours": max_hours,
            },
        },
    }


def _flag_off_config():
    """构造 bash_unattended_retry=OFF 的 config。"""
    return {
        "features": {
            "bash_unattended_retry": {
                "enabled": False,
                "max_hours": 24,
            },
        },
    }


# ---------------------------------------------------------------------------
# 测试 1：flag ON + 429 持续 30 次不退出，最终成功
# ---------------------------------------------------------------------------

async def test_unattended_retries_30_times_then_succeeds(monkeypatch):
    """flag ON 时 429 持续 30 次仍不退出，第 31 次成功。

    原 max_retries 默认 5，30 次远超上限 —— 验证 unattended 模式无限重试语义。
    """
    async def _fake_sleep(s):
        pass  # 不真睡
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    # 30 次 429 后第 31 次成功
    side_effects = [err429] * 30 + [_mock_llm_response("finally-ok")]
    llm_client = _mock_llm_client_async(side_effect=side_effects)

    result = await call_with_retry(
        llm_client, [],
        max_retries=5,  # 默认上限，unattended 模式应忽略
        initial_backoff=0.001,
        config=_flag_on_config(max_hours=24),
    )

    assert result is not None
    # 31 次调用（30 次失败 + 1 次成功）
    assert llm_client.chat_completions.await_count == 31


# ---------------------------------------------------------------------------
# 测试 2：flag ON + deadline 到期退出
# ---------------------------------------------------------------------------

async def test_unattended_exits_when_deadline_exceeded(monkeypatch):
    """flag ON 时超过 max_hours 退出。

    模拟时间推进：deadline 在第 1 次 LLM 调用失败后到期，
    第 2 次循环顶部检测到 deadline → break → 抛 last_error。
    """
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    # time.monotonic 序列（用 max_hours 很小但不为 0，让 deadline 在迭代后到期）：
    #   1. 算 deadline 起点（t0=1000.0）→ deadline = 1000.0 + 0.0001*3600 = 1000.36
    #   2. 第 1 次循环检查 → 1000.0（未超）→ 进入 LLM 调用 → 429
    #   3. 第 2 次循环检查 → 2000.0（已超 deadline 1000.36）→ break
    #   4+ 兜底再返回 5000.0（防 list 耗尽 IndexError）
    time_values = [1000.0, 1000.0, 2000.0]
    def _fake_monotonic():
        if time_values:
            return time_values.pop(0)
        return 5000.0  # 兜底：之后一直返回大值
    monkeypatch.setattr("agent.llm_retry.time.monotonic", _fake_monotonic)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=err429)

    with pytest.raises(Exception) as exc_info:
        await call_with_retry(
            llm_client, [],
            max_retries=5,
            initial_backoff=0.001,
            config=_flag_on_config(max_hours=0.0001),  # deadline ≈ t0 + 0.36s
        )

    # 验证：抛了 429（不是因 max_retries=5 耗尽退出，而是 deadline 到期）
    assert err429 is not None
    assert exc_info.value is err429
    # 验证：只调用了 1 次 LLM（第 1 次失败后 deadline 到期，不再重试）
    # （原 max_retries=5 模式下会调用 5 次，unattended deadline 模式只 1 次）
    assert llm_client.chat_completions.await_count == 1


# ---------------------------------------------------------------------------
# 测试 3：flag OFF → 原 max_retries 逻辑不变
# ---------------------------------------------------------------------------

async def test_unattended_flag_off_keeps_original_max_retries(monkeypatch):
    """flag OFF 时仍按 max_retries=3 耗尽退出。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=err429)

    with pytest.raises(Exception):
        await call_with_retry(
            llm_client, [],
            max_retries=3,
            initial_backoff=0.001,
            config=_flag_off_config(),
        )

    # 3 次后退出（不被 unattended 拉到无限）
    assert llm_client.chat_completions.await_count == 3


# ---------------------------------------------------------------------------
# 测试 4：config=None → 原 max_retries 逻辑不变
# ---------------------------------------------------------------------------

async def test_unattended_no_config_keeps_original_max_retries(monkeypatch):
    """config=None 时走原有 max_retries 逻辑（向后兼容关键测试）。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    llm_client = _mock_llm_client_async(side_effect=err429)

    with pytest.raises(Exception):
        await call_with_retry(
            llm_client, [],
            max_retries=3,
            initial_backoff=0.001,
            # config=None（默认）
        )

    assert llm_client.chat_completions.await_count == 3


# ---------------------------------------------------------------------------
# 测试 5：flag ON 但 400 不可重试错误仍立即抛
# ---------------------------------------------------------------------------

async def test_unattended_non_retryable_still_raises_immediately(monkeypatch):
    """flag ON 不影响 is_retryable 语义：400 仍立即抛。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err400 = _make_api_error(400)
    llm_client = _mock_llm_client_async(side_effect=[err400])

    with pytest.raises(Exception):
        await call_with_retry(
            llm_client, [],
            config=_flag_on_config(max_hours=24),
        )

    assert llm_client.chat_completions.await_count == 1


# ---------------------------------------------------------------------------
# 测试 6：flag ON 但 config 里没 max_hours → 用默认 24h 不崩
# ---------------------------------------------------------------------------

async def test_unattended_missing_max_hours_uses_default(monkeypatch):
    """flag ON 但配置里缺 max_hours 字段 → 用默认值 24h，不崩。"""
    async def _fake_sleep(s):
        pass
    monkeypatch.setattr("agent.llm_retry.asyncio.sleep", _fake_sleep)

    err429 = _make_api_error(429)
    success_resp = _mock_llm_response()
    llm_client = _mock_llm_client_async(side_effect=[err429, success_resp])

    # config 只声明 enabled=True，没 max_hours
    config = {"features": {"bash_unattended_retry": {"enabled": True}}}

    result = await call_with_retry(
        llm_client, [],
        max_retries=5,
        initial_backoff=0.001,
        config=config,
    )

    assert result is not None
    assert llm_client.chat_completions.await_count == 2
