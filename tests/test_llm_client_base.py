"""LLMClient 基类的 async 契约测试。

所有子类必须满足这个契约：chat_completions 返回 awaitable，
chat_completions_stream 返回 async generator。
"""
import inspect

import pytest

from agent.llm_client import LLMClient


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
