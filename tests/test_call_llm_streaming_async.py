"""_call_llm_streaming async 契约 + 行为测试（Task D2）。

关键事实（与 plan 描述的"async generator"不同，这里基于真实代码）：
- _call_llm_streaming 不是 generator，是**普通函数返回 response 对象**
- 内部通过 stream_callback 报告流式事件（不是 yield）
- 改造目标：def → async def；for → async for；call_with_retry 加 await
- 期望：inspect.iscoroutinefunction(_call_llm_streaming) == True（不是 isasyncgenfunction）

业务逻辑保留：
- chat_completions_stream 返回 async generator（T_B1/B2/B3 已改造）
- 流式失败 fallback 到 call_with_retry（已 async）
- max_tokens 升级（MaxTokensEscalator）保留
- callback 回放逻辑保留
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm_client import LLMClient


# ----------------------------------------------------------------------------
# 契约测试：函数类型
# ----------------------------------------------------------------------------

def test_call_llm_streaming_is_coroutine_function():
    """_call_llm_streaming 必须是 async def（coroutine function）。

    注意：它不是 async generator function（虽然名字含 streaming），
    因为内部用 callback 模式 + return response，不是 yield。
    """
    from agent import AIAgent
    assert inspect.iscoroutinefunction(AIAgent._call_llm_streaming), \
        "_call_llm_streaming 必须是 async def（T_D2 改造目标）"


def test_call_llm_streaming_is_not_async_gen_function():
    """_call_llm_streaming 不是 async generator function。

    防止误改成 async generator（业务上它是 async function 返回 response）。
    """
    from agent import AIAgent
    # 如果有人误改成 async generator，这个断言会失败提醒
    assert not inspect.isasyncgenfunction(AIAgent._call_llm_streaming), \
        "_call_llm_streaming 不应该是 async generator（用 callback + return 模式）"


# ----------------------------------------------------------------------------
# Mock 辅助：构造 async generator 风格的 chat_completions_stream
# ----------------------------------------------------------------------------

class _FakeAsyncStreamClient(LLMClient):
    """模拟 async generator 风格的 chat_completions_stream。

    T_B1/B2/B3 已把基类和子类的 chat_completions_stream 改为 async generator，
    所以 mock 必须用 `async def + yield`（不能用同步 `def + yield`）。
    """

    def __init__(self, chunks, non_stream_response=None, raise_on_stream=False):
        self._chunks = chunks
        self._non_stream_response = non_stream_response
        self._raise_on_stream = raise_on_stream
        self.non_stream_calls = 0
        self.last_max_tokens = "not_set"

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        self.non_stream_calls += 1
        self.last_max_tokens = kwargs.get("max_tokens", "not_set")
        if self._non_stream_response is not None:
            return self._non_stream_response
        # 默认非流式响应
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="fallback ok", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        if self._raise_on_stream:
            raise RuntimeError("stream broke")
        for c in self._chunks:
            yield c


def _build_minimal_agent(stream_callback=None, llm_client=None):
    """构造最小可用的 AIAgent 用于测试（参考 tests/test_streaming.py 的同款 helper）。"""
    from agent import AIAgent
    agent = AIAgent(
        base_url="x", api_key="x", model="test",
        enabled_toolsets=[],  # 不加载工具
        stream_callback=stream_callback,
        system_prompt_override="test",  # 跳过 prompt 构建
    )
    if llm_client is not None:
        agent.llm_client = llm_client
    return agent


# ----------------------------------------------------------------------------
# 行为测试：async 调用 + chunk 处理 + callback
# ----------------------------------------------------------------------------

async def test_streaming_async_call_returns_response():
    """_call_llm_streaming 用 await 拿 response（不是 for 消费 generator）。"""
    chunks = [
        {"content": "你好", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    client = _FakeAsyncStreamClient(chunks)
    agent = _build_minimal_agent(stream_callback=lambda e: None, llm_client=client)

    # 必须用 await 调用（不是 for）
    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )
    # 验证返回 response 结构（不是 chunk 列表）
    assert response is not None
    assert hasattr(response, "choices")
    assert response.choices[0].message.content == "你好"
    assert response.choices[0].finish_reason == "stop"


async def test_streaming_callback_receives_events():
    """stream_callback 收到 content + done 事件（业务逻辑不变）。"""
    chunks = [
        {"content": "你好", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "世界", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    events = []
    client = _FakeAsyncStreamClient(chunks)
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=client,
    )

    await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )

    content_events = [e for e in events if e["type"] == "content"]
    done_events = [e for e in events if e["type"] == "done"]
    assert len(content_events) == 2
    assert content_events[0]["delta"] == "你好"
    assert content_events[1]["accumulated"] == "你好世界"
    assert len(done_events) == 1
    assert done_events[0]["finish_reason"] == "stop"


async def test_streaming_tool_call_accumulation():
    """工具调用参数分片累积，合成 tool_calls 列表。"""
    tc1 = SimpleNamespace(
        index=0, id="call_1", type="function",
        function=SimpleNamespace(name="search", arguments='{"q": "hel'),
    )
    tc2 = SimpleNamespace(
        index=0, id=None, type="function",
        function=SimpleNamespace(name=None, arguments='lo"}'),
    )
    chunks = [
        {"content": "", "tool_calls": [tc1], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [tc2], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "tool_calls",
         "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    client = _FakeAsyncStreamClient(chunks)
    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=client,
    )

    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )

    # 合成 tool_calls
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    tc = response.choices[0].message.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function.name == "search"
    assert tc.function.arguments == '{"q": "hello"}'
    assert response.choices[0].finish_reason == "tool_calls"

    # tool_call_start 事件
    tc_starts = [e for e in events if e["type"] == "tool_call_start"]
    assert len(tc_starts) == 1
    assert tc_starts[0]["name"] == "search"


async def test_streaming_failure_falls_back_to_non_stream():
    """流式抛异常时，fallback 到 await call_with_retry（async 路径）。"""
    client = _FakeAsyncStreamClient(
        chunks=[],
        non_stream_response=SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="fallback ok", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        ),
        raise_on_stream=True,
    )
    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=client,
    )

    # 不应该抛
    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )
    assert response.choices[0].message.content == "fallback ok"

    # fallback 内容回放给 callback
    content_events = [e for e in events if e["type"] == "content"]
    assert any("fallback ok" in e["delta"] for e in content_events)


async def test_streaming_max_tokens_escalation():
    """finish_reason=length 时，触发 max_tokens 升级 + 非流式重试。"""
    stream_chunks = [
        {"content": "前半段", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "length", "usage": None},
    ]
    client = _FakeAsyncStreamClient(
        chunks=stream_chunks,
        non_stream_response=SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="前半段+后半段（完整）", tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=20, total_tokens=30,
            ),
        ),
    )
    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=client,
    )

    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )

    # 触发升级（流式 chat_completions_stream 一次 + 非流式 chat_completions 重试一次）
    assert client.non_stream_calls == 1
    assert client.last_max_tokens != "not_set"
    assert client.last_max_tokens >= 16 * 1024
    assert agent._max_tokens_escalator.has_escalated is True

    # response 是重试结果
    assert response.choices[0].message.content == "前半段+后半段（完整）"
    assert response.choices[0].finish_reason == "stop"

    # callback 收到重试内容回放
    content_events = [e for e in events if e["type"] == "content"]
    full_contents = [e["delta"] for e in content_events]
    assert any("完整" in c for c in full_contents)


async def test_streaming_no_escalate_on_stop():
    """finish_reason=stop 时，不触发 max_tokens 升级。"""
    chunks = [
        {"content": "完整内容", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    client = _FakeAsyncStreamClient(chunks)
    agent = _build_minimal_agent(llm_client=client)

    await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )

    # 没触发非流式 fallback
    assert client.non_stream_calls == 0
    assert agent._max_tokens_escalator.has_escalated is False


async def test_streaming_callback_exception_does_not_break():
    """callback 抛异常不影响主流程（fail-open）。"""
    chunks = [
        {"content": "a", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop", "usage": None},
    ]

    def bad_callback(event):
        if event["type"] == "content":
            raise ValueError("callback broke")

    client = _FakeAsyncStreamClient(chunks)
    agent = _build_minimal_agent(stream_callback=bad_callback, llm_client=client)

    # 不应该抛
    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )
    assert response.choices[0].message.content == "a"


# ----------------------------------------------------------------------------
# 边界：chat_completions_stream 必须用 async for 消费
# ----------------------------------------------------------------------------

async def test_streaming_uses_async_for_not_sync_for():
    """验证：如果 client.chat_completions_stream 是 async generator，
    _call_llm_streaming 内部必须用 async for（否则 TypeError）。

    用一个只支持 async iteration 的 client 测试。
    """
    chunks = [
        {"content": "x", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop", "usage": None},
    ]
    client = _FakeAsyncStreamClient(chunks)
    agent = _build_minimal_agent(llm_client=client)

    # 如果内部用 sync for，会在 async generator 上抛 TypeError
    response = await agent._call_llm_streaming(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    )
    assert response.choices[0].message.content == "x"
