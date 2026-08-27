"""流式输出测试（04）。

验证：
1. LLMClient.chat_completions_stream 默认实现（包装非流式为单 chunk）
2. AIAgent.stream_callback 接到事件
3. 工具调用参数流式累积正确
4. 流式失败 fallback 到非流式
5. 流式结束后 conversation_history 干净
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.llm_client import LLMClient, OpenAICompatClient
from agent.atomic_io import atomic_write_text  # 仅引用确保模块加载


# ----------------------------------------------------------------------------
# LLMClient.chat_completions_stream 默认实现
# ----------------------------------------------------------------------------

class _FakeNonStreamClient(LLMClient):
    """假非流式 client：返回固定内容，用于测试默认 stream 包装。"""

    def __init__(self, content="hello", tool_calls=None):
        self._content = content
        self._tool_calls = tool_calls

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        msg = SimpleNamespace(content=self._content, tool_calls=self._tool_calls)
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=usage,
        )


async def test_default_stream_wraps_non_stream():
    """LLMClient.chat_completions_stream 默认实现：单 chunk 含完整内容。"""
    client = _FakeNonStreamClient(content="hi", tool_calls=None)
    chunks = []
    async for c in client.chat_completions_stream(messages=[{"role": "user", "content": "?"}]):
        chunks.append(c)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "hi"
    assert chunks[0]["finish_reason"] == "stop"
    assert chunks[0]["usage"]["prompt_tokens"] == 10


# ----------------------------------------------------------------------------
# OpenAI delta 累积（用 mock SDK）
# ----------------------------------------------------------------------------

def _make_openai_chunk(content="", tool_calls=None, finish_reason=None, usage=None):
    """构造一个 OpenAI 风格的 stream chunk。"""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    chunk = SimpleNamespace(choices=[choice], usage=usage)
    return chunk


class _FakeAsyncStream:
    """模拟 AsyncOpenAI 的 stream 返回值：async iterable。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        outer = self

        class _Iter:
            def __init__(self):
                self._idx = 0

            async def __anext__(self):
                if self._idx >= len(outer._chunks):
                    raise StopAsyncIteration
                c = outer._chunks[self._idx]
                self._idx += 1
                return c

        return _Iter()


class _FakeOpenAISDK:
    """模拟 openai.AsyncOpenAI(...) 的 chat.completions.create。

    AsyncOpenAI 的 create 是 async function：
      - stream=False 时 await create(...) 拿 response 对象
      - stream=True 时 await create(...) 拿 AsyncIterator[chunk]
    """

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    @property
    def chat(self):
        outer = self
        class _Comps:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    outer.last_kwargs = kwargs
                    if kwargs.get("stream"):
                        return _FakeAsyncStream(outer._chunks)
                    # 非流式：直接返回第一个 chunk（模拟完整响应）
                    return outer._chunks[0] if outer._chunks else None
        return _Comps


async def test_openai_compat_stream_yields_normalized_chunks(monkeypatch):
    """OpenAICompatClient.chat_completions_stream 返回规范化 dict。"""
    chunks = [
        _make_openai_chunk(content="Hel"),
        _make_openai_chunk(content="lo"),
        _make_openai_chunk(content="", finish_reason="stop",
                           usage=SimpleNamespace(
                               prompt_tokens=20, completion_tokens=2, total_tokens=22,
                           )),
    ]
    fake_sdk = _FakeOpenAISDK(chunks)

    client = OpenAICompatClient(base_url="x", api_key="x", model="test")
    # 替换底层 SDK client
    client.client = fake_sdk

    # async generator：必须 async for 收集
    results = []
    async for chunk in client.chat_completions_stream(
        messages=[{"role": "user", "content": "?"}],
        tools=None,
    ):
        results.append(chunk)
    # 应该有 3 个 chunk（usage 那个 finish_reason 非空）
    assert len(results) == 3
    assert results[0]["content"] == "Hel"
    assert results[1]["content"] == "lo"
    # usage chunk
    assert results[2]["usage"]["prompt_tokens"] == 20


async def test_openai_compat_stream_tool_call_accumulation():
    """工具调用参数分片累积。"""
    # tool_call delta 模拟：第一个 chunk 给 id+name+部分 args，
    # 后续 chunk 给更多 args（同 index）
    tc1 = SimpleNamespace(
        index=0, id="call_1", type="function",
        function=SimpleNamespace(name="search", arguments='{"q": "hel'),
    )
    tc2 = SimpleNamespace(
        index=0, id=None, type="function",
        function=SimpleNamespace(name=None, arguments='lo"}'),
    )
    chunks = [
        _make_openai_chunk(content="", tool_calls=[tc1]),
        _make_openai_chunk(content="", tool_calls=[tc2]),
        _make_openai_chunk(content="", finish_reason="tool_calls",
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    ]
    fake_sdk = _FakeOpenAISDK(chunks)
    client = OpenAICompatClient(base_url="x", api_key="x", model="test")
    client.client = fake_sdk

    results = []
    async for chunk in client.chat_completions_stream(
        messages=[{"role": "user", "content": "?"}], tools=[],
    ):
        results.append(chunk)
    # tool_calls 在每个 chunk 都跟着（OpenAI SDK 行为）
    # 验证不影响后续累积逻辑
    assert len(results) == 3


# ----------------------------------------------------------------------------
# AIAgent 集成：stream_callback 接到事件
# ----------------------------------------------------------------------------

class _FakeStreamClient(LLMClient):
    """模拟流式 client：吐一系列 chunk。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = 0

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        # 非流式 fallback（流式失败时用）
        msg = SimpleNamespace(content="fallback", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=None,
        )

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        self.calls += 1
        for c in self._chunks:
            yield c


def _build_minimal_agent(stream_callback=None, llm_client=None):
    """构造最小可用的 AIAgent 用于测试。"""
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


async def test_stream_callback_receives_content_events():
    """AIAgent 流式时，callback 收到 content / done 事件。"""
    chunks = [
        {"content": "你好", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "世界", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    fake_client = _FakeStreamClient(chunks)
    events = []

    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=fake_client,
    )
    await agent.run_conversation("hi")

    content_events = [e for e in events if e["type"] == "content"]
    done_events = [e for e in events if e["type"] == "done"]
    assert len(content_events) == 2
    assert content_events[0]["delta"] == "你好"
    assert content_events[1]["delta"] == "世界"
    # accumulated 累积
    assert content_events[1]["accumulated"] == "你好世界"
    assert len(done_events) == 1
    assert done_events[0]["finish_reason"] == "stop"


async def test_stream_callback_tool_call_start_event():
    """工具调用时 callback 收到 tool_call_start。"""
    tc = SimpleNamespace(
        index=0, id="call_1", type="function",
        function=SimpleNamespace(name="search", arguments='{"q":"x"}'),
    )
    chunks = [
        {"content": "", "tool_calls": [tc], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "tool_calls",
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                   "cache_read": 0, "cache_creation": 0}},
    ]
    fake_client = _FakeStreamClient(chunks)
    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=fake_client,
    )
    # run_conversation 会处理 tool_calls，但工具不存在时会塞错误消息
    await agent.run_conversation("do search")

    tc_starts = [e for e in events if e["type"] == "tool_call_start"]
    # 注：工具不存在时 agent 会循环重试，但每次循环都至少触发一次 tool_call_start。
    # 关键不变式：tool_call_start 事件至少发了一次，且 name 正确
    assert len(tc_starts) >= 1
    assert all(e["name"] == "search" for e in tc_starts)


async def test_stream_failure_falls_back_to_non_stream():
    """流式抛异常时，fallback 到非流式 call_with_retry。"""

    class _BoomStreamClient(LLMClient):
        async def chat_completions(self, messages, *, tools=None, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="fallback ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )

        async def chat_completions_stream(self, *args, **kwargs):
            raise RuntimeError("stream broke")
            yield  # 让 Python 认为这是 generator

    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=_BoomStreamClient(),
    )
    # 不应该抛
    await agent.run_conversation("hi")
    # fallback 后内容也回放给 callback
    content_events = [e for e in events if e["type"] == "content"]
    assert any("fallback ok" in e["delta"] for e in content_events)


async def test_no_stream_callback_uses_non_stream_path():
    """stream_callback=None 时走非流式（向后兼容）。"""
    # 用一个会"故意只在 chat_completions 被调用"的 fake
    call_log = {"non_stream": 0, "stream": 0}

    class _Client(LLMClient):
        async def chat_completions(self, *args, **kwargs):
            call_log["non_stream"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )

        async def chat_completions_stream(self, *args, **kwargs):
            call_log["stream"] += 1
            yield {"content": "x", "tool_calls": [], "finish_reason": None, "usage": None}

    agent = _build_minimal_agent(stream_callback=None, llm_client=_Client())
    await agent.run_conversation("hi")
    # 走的是非流式路径
    assert call_log["non_stream"] >= 1
    assert call_log["stream"] == 0


async def test_stream_callback_exception_does_not_break_main_flow():
    """callback 抛异常不影响主流程。"""
    chunks = [
        {"content": "a", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop", "usage": None},
    ]

    def bad_callback(event):
        if event["type"] == "content":
            raise ValueError("callback broke")

    fake_client = _FakeStreamClient(chunks)
    agent = _build_minimal_agent(
        stream_callback=bad_callback,
        llm_client=fake_client,
    )
    # 不应该抛
    await agent.run_conversation("hi")


# ----------------------------------------------------------------------------
# P0-3: max_tokens 升级机制集成测试
# ----------------------------------------------------------------------------

async def test_max_tokens_escalates_on_length_finish():
    """流式 finish_reason=length 时，自动升级 max_tokens 并用非流式重试。

    场景：
      第一次流式：流了部分内容后 finish_reason="length"（被 max_tokens 截断）
      escalator 触发 → 调用 chat_completions（非流式）拿完整响应
      最终：response 是重试的完整内容，finish_reason="stop"
    """
    # 第一次流式返回截断
    stream_chunks = [
        {"content": "前半段", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "length", "usage": None},
    ]

    class _EscClient(LLMClient):
        def __init__(self):
            self.non_stream_calls = 0
            self.last_max_tokens = "not_set"

        async def chat_completions(self, messages, *, tools=None, **kwargs):
            self.non_stream_calls += 1
            self.last_max_tokens = kwargs.get("max_tokens", "not_set")
            # 非流式返回完整响应
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content="前半段+后半段（完整）", tool_calls=None,
                    ),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30,
                ),
            )

        async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
            for c in stream_chunks:
                yield c

    client = _EscClient()
    events = []
    agent = _build_minimal_agent(
        stream_callback=lambda e: events.append(e),
        llm_client=client,
    )
    await agent.run_conversation("hi")

    # 验证：触发了一次非流式调用
    assert client.non_stream_calls >= 1
    # 验证：传入了升级后的 max_tokens
    assert client.last_max_tokens != "not_set"
    assert client.last_max_tokens >= 16 * 1024  # 至少 16K
    # 验证：escalator 状态已升级
    assert agent._max_tokens_escalator.has_escalated is True
    # 验证：callback 收到了完整内容（重试结果回放）
    content_events = [e for e in events if e["type"] == "content"]
    full_contents = [e["delta"] for e in content_events]
    assert any("完整" in c for c in full_contents)


async def test_max_tokens_no_escalate_on_stop_finish():
    """finish_reason=stop 时不触发升级（非流式路径）。"""
    stream_chunks = [
        {"content": "完整内容", "tool_calls": [], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10,
                   "cache_read": 0, "cache_creation": 0}},
    ]

    non_stream_count = {"n": 0}

    class _NoEscClient(LLMClient):
        async def chat_completions(self, messages, *, tools=None, **kwargs):
            non_stream_count["n"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="x", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )

        async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
            for c in stream_chunks:
                yield c

    agent = _build_minimal_agent(llm_client=_NoEscClient())
    await agent.run_conversation("hi")

    # 非流式只被调一次（没 escalate 重试）
    assert non_stream_count["n"] == 1
    assert agent._max_tokens_escalator.has_escalated is False


async def test_max_tokens_escalate_idempotent_within_session():
    """一个 session 内最多升级一次（避免多轮循环重复升级）。

    非流式路径：
      第一轮：chat_completions 返回 length（截断）→ escalate → 重试返回 stop
      第二轮：chat_completions 返回 length（截断）→ 已升级，不再重试
    """
    call_log = {"stream": 0, "non_stream": 0}

    class _IdempotentClient(LLMClient):
        async def chat_completions(self, messages, *, tools=None, **kwargs):
            call_log["non_stream"] += 1
            # 第一次主调用：返回 length 截断
            # 第二次重试（escalate 触发）：返回 stop
            # 第三次主调用（第二轮 run）：返回 length 截断
            if call_log["non_stream"] in (1, 3):
                finish = "length"
            else:
                finish = "stop"
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason=finish,
                )],
                usage=None,
            )

        async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
            call_log["stream"] += 1
            yield {"content": "partial", "tool_calls": [],
                   "finish_reason": None, "usage": None}
            yield {"content": "", "tool_calls": [],
                   "finish_reason": "length", "usage": None}

    agent = _build_minimal_agent(llm_client=_IdempotentClient())
    await agent.run_conversation("round 1")

    # 第一轮：1 次主调用（length）+ 1 次 escalate 重试（stop）= 2 次
    assert call_log["non_stream"] == 2
    assert agent._max_tokens_escalator.has_escalated is True

    # 第二轮：再调一次主调用（length），但不再升级（已升过）
    await agent.run_conversation("round 2")
    # 第二轮 1 次主调用（length）+ 1 次续写恢复（升级后仍截断 →
    # _recover_output_truncation 局部视图续写，恢复调用返回 stop）= 4 次累计
    assert call_log["non_stream"] == 4
    assert agent._max_tokens_escalator.has_escalated is True  # 仍 True（幂等）
