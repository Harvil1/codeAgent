"""LLMClient 基类的 async 契约测试。

所有子类必须满足这个契约：chat_completions 返回 awaitable，
chat_completions_stream 返回 async generator。
"""
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm_client import LLMClient, OpenAICompatClient


def test_chat_completions_is_coroutine_function():
    """基类 chat_completions 必须是 async def（coroutine function）。"""
    assert inspect.iscoroutinefunction(LLMClient.chat_completions), \
        "LLMClient.chat_completions 必须是 async def"


def test_chat_completions_stream_is_async_gen_function():
    """基类 chat_completions_stream 必须是 async generator function。"""
    assert inspect.isasyncgenfunction(LLMClient.chat_completions_stream), \
        "LLMClient.chat_completions_stream 必须是 async def（含 yield）"


async def test_subclass_must_implement_chat_completions():
    """直接调用基类 chat_completions 应抛 NotImplementedError。"""
    client = LLMClient()  # 直接实例化基类
    with pytest.raises(NotImplementedError):
        await client.chat_completions(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# OpenAICompatClient async 契约测试（Task B2）
# ---------------------------------------------------------------------------

async def test_openai_compat_uses_async_client():
    """OpenAICompatClient 应使用 AsyncOpenAI（不是 OpenAI）。"""
    with patch("agent.llm_client.AsyncOpenAI") as mock_async:
        client = OpenAICompatClient(
            base_url="https://api.deepseek.com/v1",
            api_key="fake-key",
            model="deepseek-chat",
        )
        mock_async.assert_called_once()


async def test_openai_compat_chat_completions_awaited():
    """chat_completions 返回的 response 必须是 await 后的（不是 coroutine）。"""
    with patch("agent.llm_client.AsyncOpenAI") as mock_async_cls:
        # 构造 mock client 链
        mock_instance = MagicMock()
        mock_create = AsyncMock(return_value=MagicMock(choices=[MagicMock()]))
        mock_instance.chat.completions.create = mock_create
        mock_async_cls.return_value = mock_instance

        client = OpenAICompatClient("https://x", "key", "model")
        await client.chat_completions(messages=[{"role": "user", "content": "hi"}])
        mock_create.assert_awaited_once()


async def test_openai_compat_stream_is_async_gen():
    """chat_completions_stream 必须是 async generator。"""
    with patch("agent.llm_client.AsyncOpenAI"):
        client = OpenAICompatClient("https://x", "key", "model")
    assert inspect.isasyncgenfunction(client.chat_completions_stream)
