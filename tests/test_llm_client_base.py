"""LLMClient 基类的 async 契约测试。

所有子类必须满足这个契约：chat_completions 返回 awaitable，
chat_completions_stream 返回 async generator。
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm_client import LLMClient, OpenAICompatClient, AnthropicClient


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
# OpenAICompatClient async 契约测试
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


# ---------------------------------------------------------------------------
# AnthropicClient async 契约测试
# ---------------------------------------------------------------------------

def test_anthropic_chat_completions_is_coroutine():
    """AnthropicClient.chat_completions 必须是 coroutine function。"""
    assert inspect.iscoroutinefunction(AnthropicClient.chat_completions), \
        "AnthropicClient.chat_completions 必须是 async def"


def test_anthropic_stream_is_async_gen():
    """AnthropicClient.chat_completions_stream 必须是 async generator function。"""
    assert inspect.isasyncgenfunction(AnthropicClient.chat_completions_stream), \
        "AnthropicClient.chat_completions_stream 必须是 async def（含 yield）"


def test_anthropic_uses_async_anthropic_client():
    """__init__ 应使用 AsyncAnthropic（不是同步 Anthropic）。

    用 sys.modules 级 patch：拦截 anthropic 模块的 AsyncAnthropic 属性，
    确保 AnthropicClient 构造时取的是 AsyncAnthropic 而非 Anthropic。
    """
    import sys
    if "agent.llm_client" in sys.modules:
        # 已经 import 过，确保测试看到的是当前模块的 anthropic import
        pass
    with patch("anthropic.AsyncAnthropic") as mock_async_cls:
        client = AnthropicClient(api_key="fake-key", model="claude-3")
        mock_async_cls.assert_called_once()
        # 确认 self.client 来自 AsyncAnthropic 构造
        assert client.client is mock_async_cls.return_value


def test_anthropic_uses_auth_token_when_provided():
    """auth_token 优先于 api_key（DeepSeek Anthropic 端点走 Bearer）。"""
    with patch("anthropic.AsyncAnthropic") as mock_async_cls:
        AnthropicClient(
            api_key="key1",
            auth_token="token1",
            model="m",
        )
        kwargs = mock_async_cls.call_args.kwargs
        assert kwargs.get("auth_token") == "token1"
        assert "api_key" not in kwargs


async def test_anthropic_chat_completions_awaits_create():
    """chat_completions 应 await self.client.messages.create（不是直接同步调用）。"""
    with patch("anthropic.AsyncAnthropic") as mock_async_cls:
        mock_instance = MagicMock()
        mock_create = AsyncMock()
        # 构造一个最小 Anthropic 响应：content=[text block]，usage 有 input/output_tokens
        mock_create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello world")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        mock_instance.messages.create = mock_create
        mock_async_cls.return_value = mock_instance

        client = AnthropicClient(api_key="k", model="claude-3")
        result = await client.chat_completions(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        mock_create.assert_awaited_once()
        # 返回是 OpenAI 兼容包装：choices[0].message.content == "hello world"
        assert result.choices[0].message.content == "hello world"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3


async def test_anthropic_chat_completions_tool_use_wrapped():
    """工具调用响应应包装成 OpenAI tool_calls 格式（验证 _wrap_response 仍工作）。"""
    with patch("anthropic.AsyncAnthropic") as mock_async_cls:
        mock_instance = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="call_1", name="read_file",
                                input={"path": "/tmp/x"}),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_instance.messages.create = mock_create
        mock_async_cls.return_value = mock_instance

        client = AnthropicClient(api_key="k", model="claude-3")
        result = await client.chat_completions(messages=[{"role": "user", "content": "hi"}])
        tcs = result.choices[0].message.tool_calls
        assert tcs and len(tcs) == 1
        assert tcs[0].id == "call_1"
        assert tcs[0].type == "function"
        assert tcs[0].function.name == "read_file"
        # arguments 应是 JSON 字符串
        import json as _json
        assert _json.loads(tcs[0].function.arguments) == {"path": "/tmp/x"}


async def test_anthropic_chat_completions_stream_uses_async_with_and_async_for():
    """stream 应使用 async with + async for（不是同步 with/for）。

    对齐真实 AsyncMessageStream 语义：__aenter__ 返回 self（同一对象既是
    async iterator 又是 context manager），所以 stream 本身有
    __aiter__/__anext__ 和 get_final_message 协程。
    """
    with patch("anthropic.AsyncAnthropic") as mock_async_cls:
        mock_instance = MagicMock()

        # 构造同时是 async context manager + async iterator 的对象（对齐 SDK）
        class _FakeStream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration
            async def __aenter__(self):
                return self  # 关键：返回 self，和 AsyncMessageStream 一致
            async def __aexit__(self, *args):
                return False

        fake_stream = _FakeStream()
        # get_final_message 是协程，返回最小 final_message
        fake_stream.get_final_message = AsyncMock(return_value=SimpleNamespace(
            content=[],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        ))

        # messages.stream(...) 同步返回 fake_stream
        mock_instance.messages.stream = MagicMock(return_value=fake_stream)
        mock_async_cls.return_value = mock_instance

        client = AnthropicClient(api_key="k", model="claude-3")
        chunks = []
        async for chunk in client.chat_completions_stream(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        ):
            chunks.append(chunk)

        # 应该 yield 一次：空 tool_calls + finish_reason=stop + usage
        assert len(chunks) == 1
        final = chunks[-1]
        assert final["finish_reason"] == "stop"
        assert final["tool_calls"] == []
        assert final["usage"] is not None
        assert final["usage"]["prompt_tokens"] == 2
        assert final["usage"]["completion_tokens"] == 1
        # get_final_message 被 await 过
        fake_stream.get_final_message.assert_awaited_once()
