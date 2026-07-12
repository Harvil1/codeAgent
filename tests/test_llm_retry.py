"""LLM 重试与错误恢复测试（适配新 llm_client 接口）。"""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.llm_retry import (
    is_retryable, get_retry_after, call_with_retry,
    DEFAULT_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# is_retryable
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
# get_retry_after
# ---------------------------------------------------------------------------

def test_retry_after_from_attribute():
    err = SimpleNamespace(retry_after=2.5)
    assert get_retry_after(err) == 2.5


def test_retry_after_none():
    err = SimpleNamespace(retry_after=None)
    assert get_retry_after(err) is None


# ---------------------------------------------------------------------------
# call_with_retry（新接口：llm_client.chat_completions）
# ---------------------------------------------------------------------------

def _mock_llm_response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
        )],
    )


def _mock_llm_client(response=None, side_effect=None):
    """构造 mock LLMClient（有 chat_completions 方法）。"""
    client = MagicMock()
    if side_effect is not None:
        client.chat_completions.side_effect = side_effect
    else:
        client.chat_completions.return_value = response or _mock_llm_response()
    return client


def test_call_succeeds_first_try():
    """第一次就成功，不重试。"""
    llm_client = _mock_llm_client(_mock_llm_response())
    result = call_with_retry(llm_client, [])
    assert result is not None
    assert llm_client.chat_completions.call_count == 1


def test_call_retries_on_429(monkeypatch):
    """429 错误触发重试，最终成功。"""
    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda s: None)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=[
        err429, err429, _mock_llm_response(),
    ])

    result = call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)
    assert result is not None
    assert llm_client.chat_completions.call_count == 3


def test_call_not_retryable_raises_immediately(monkeypatch):
    """不可重试错误立即抛。"""
    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda s: None)
    err400 = _make_api_error(400)
    llm_client = _mock_llm_client(side_effect=[err400])

    with pytest.raises(Exception):
        call_with_retry(llm_client, [], max_retries=5)

    assert llm_client.chat_completions.call_count == 1


def test_call_exhausts_retries_then_raises(monkeypatch):
    """可重试错误重试耗尽后抛。"""
    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda s: None)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=err429)

    with pytest.raises(Exception):
        call_with_retry(llm_client, [], max_retries=3, initial_backoff=0.01)

    assert llm_client.chat_completions.call_count == 3


def test_call_falls_back_to_fallback_client(monkeypatch):
    """主 client 失败后切换备用 client。"""
    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda s: None)
    err429 = _make_api_error(429)
    main_client = _mock_llm_client(side_effect=err429)
    fallback_client = _mock_llm_client(_mock_llm_response("fallback-ok"))

    result = call_with_retry(
        main_client, [],
        max_retries=3, initial_backoff=0.01,
        fallback_llm_client=fallback_client,
    )

    assert main_client.chat_completions.call_count == 3
    assert fallback_client.chat_completions.call_count == 1


def test_call_no_fallback_just_raises(monkeypatch):
    """无备用 client 时，主 client 失败直接抛。"""
    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda s: None)
    err429 = _make_api_error(429)
    llm_client = _mock_llm_client(side_effect=err429)

    with pytest.raises(Exception):
        call_with_retry(
            llm_client, [],
            max_retries=2, initial_backoff=0.01,
            fallback_llm_client=None,
        )
    assert llm_client.chat_completions.call_count == 2
