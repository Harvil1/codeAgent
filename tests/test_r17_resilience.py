"""R17 韧性专项测试。

#12 流空闲看门狗（90s 无 chunk abort）
#10 max_tokens 完整恢复链（64k 升级 + 续写 3 次）
#13 max_tokens 400 溢出自适应重试
#44 unattended 退避帽
#14 终止原因枚举化
#9 扣留-恢复模式
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent.llm_client import (
    DEFAULT_STREAM_IDLE_TIMEOUT,
    LLMStreamIdleTimeout,
    OpenAICompatClient,
    _iterate_with_watchdog,
    create_llm_client,
)


# ---------------------------------------------------------------------------
# R17 #12：流空闲看门狗
# ---------------------------------------------------------------------------

class _FakeStream:
    """可注入停顿的假 async 流。"""

    def __init__(self, items, delay=0.0, stall_before_last=0.0):
        self.items = list(items)
        self.delay = delay
        self.stall_before_last = stall_before_last
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        if self.stall_before_last and len(self.items) == 1:
            await asyncio.sleep(self.stall_before_last)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.items.pop(0)

    async def close(self):
        self.closed = True


async def _collect(aiter):
    out = []
    async for x in aiter:
        out.append(x)
    return out


@pytest.mark.asyncio
async def test_watchdog_passthrough_normal():
    """正常流：看门狗不介入，元素透传。"""
    stream = _FakeStream([1, 2, 3])
    assert await _collect(
        _iterate_with_watchdog(stream, idle_timeout=5.0)
    ) == [1, 2, 3]
    assert not stream.closed


@pytest.mark.asyncio
async def test_watchdog_timeout_raises_and_closes():
    """空闲超时：抛 LLMStreamIdleTimeout 且尽力关流。"""
    stream = _FakeStream(["a", "b"], stall_before_last=0.3)
    with pytest.raises(LLMStreamIdleTimeout):
        await _collect(_iterate_with_watchdog(stream, idle_timeout=0.05))
    assert stream.closed


@pytest.mark.asyncio
async def test_watchdog_disabled_when_zero():
    """idle_timeout<=0 禁用（长停顿也不超时）。"""
    stream = _FakeStream([1], stall_before_last=0.1)
    assert await _collect(
        _iterate_with_watchdog(stream, idle_timeout=0)
    ) == [1]


@pytest.mark.asyncio
async def test_watchdog_resets_on_each_chunk():
    """每个 chunk 重置计时：慢但未超闲的流不被误杀。"""
    # 每 chunk 间隔 0.04s，idle_timeout 0.1s——总时长超 timeout 但每次间隔内
    stream = _FakeStream([1, 2, 3], delay=0.04)
    assert await _collect(
        _iterate_with_watchdog(stream, idle_timeout=0.1)
    ) == [1, 2, 3]


def test_client_default_and_config_watchdog():
    """client 构造默认 90s；create_llm_client 读 model_config 键。"""
    c = OpenAICompatClient(base_url="http://x", api_key="k", model="m")
    assert c.stream_idle_timeout == DEFAULT_STREAM_IDLE_TIMEOUT == 90.0

    c2 = create_llm_client({
        "format": "openai", "base_url": "http://x", "api_key": "k",
        "model": "m", "stream_idle_timeout": 30,
    })
    assert c2.stream_idle_timeout == 30

    c3 = create_llm_client({
        "format": "openai", "base_url": "http://x", "api_key": "k",
        "model": "m", "stream_idle_timeout": 0,
    })
    assert c3.stream_idle_timeout == 0  # 0 = 禁用

    c4 = create_llm_client({
        "format": "anthropic", "api_key": "k", "model": "m",
        "stream_idle_timeout": 45,
    })
    assert c4.stream_idle_timeout == 45


# ---------------------------------------------------------------------------
# R17 #10：max_tokens 完整恢复链（64k + 续写 3 次）
# ---------------------------------------------------------------------------

from agent.llm_retry import (
    DEFAULT_ESCALATED_MAX_TOKENS,
    DEFAULT_OUTPUT_RECOVERY_LIMIT,
    MaxTokensEscalator,
    _compute_backoff,
    call_with_retry,
    compute_overflow_max_tokens,
    parse_context_overflow,
)


def test_escalated_max_tokens_is_64k():
    """升级值对齐 CC ESCALATED_MAX_TOKENS=64k。"""
    assert DEFAULT_ESCALATED_MAX_TOKENS == 64000
    assert DEFAULT_OUTPUT_RECOVERY_LIMIT == 3
    esc = MaxTokensEscalator()
    assert esc.escalate() == 64000


def _mk_resp(content, finish="stop", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=tool_calls,
                reasoning_content=None, thinking_signature=None,
            ),
            finish_reason=finish,
        )],
        usage=None,
    )


def _mk_agent(monkeypatch, responses, escalate=True, config=None):
    """轻量 AIAgent（跳过 __init__），monkeypatch call_with_retry。"""
    from agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent._max_tokens_escalator = MaxTokensEscalator()
    if escalate:
        agent._max_tokens_escalator.escalate()
    agent.config = config or {}
    agent.llm_client = SimpleNamespace()
    agent.fallback_llm_client = None

    calls = []

    async def fake_retry(client, msgs, **kw):
        calls.append({"messages": [dict(m) for m in msgs], "kwargs": kw})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    import agent.llm_retry as lr
    monkeypatch.setattr(lr, "call_with_retry", fake_retry)
    # 方法内 from agent.llm_retry import ... 拿的是模块属性 → patch 生效
    return agent, calls


@pytest.mark.asyncio
async def test_recover_no_truncation_passthrough(monkeypatch):
    agent, calls = _mk_agent(monkeypatch, [])
    resp = _mk_resp("done")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out is resp
    assert calls == []


@pytest.mark.asyncio
async def test_recover_not_escalated_passthrough(monkeypatch):
    """未升级过的截断不接手（升级路径在 _call_llm_* 内部）。"""
    agent, calls = _mk_agent(monkeypatch, [], escalate=False)
    resp = _mk_resp("half...", finish="length")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out is resp
    assert calls == []


@pytest.mark.asyncio
async def test_recover_tool_calls_truncation_passthrough(monkeypatch):
    agent, calls = _mk_agent(monkeypatch, [])
    resp = _mk_resp(None, finish="length", tool_calls=[SimpleNamespace()])
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out is resp


@pytest.mark.asyncio
async def test_recover_success_first_retry(monkeypatch):
    """升级后仍截断 → 续写一次成功 → 拼接内容 + finish_reason=stop。"""
    agent, calls = _mk_agent(monkeypatch, [_mk_resp(" tail done")])
    resp = _mk_resp("head ", finish="length")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out.choices[0].message.content == "head  tail done"
    assert out.choices[0].finish_reason == "stop"
    # 局部视图：原消息 + 截断 assistant + 续写 meta user
    assert len(calls) == 1
    m = calls[0]["messages"]
    assert m[-2]["role"] == "assistant" and m[-2]["content"] == "head "
    assert m[-1]["role"] == "user" and "不要道歉" in m[-1]["content"]
    assert calls[0]["kwargs"]["max_tokens"] == 64000


@pytest.mark.asyncio
async def test_recover_exhausts_limit(monkeypatch):
    """续写 3 次仍 length → 返回拼接内容（finish_reason=length 标记）。"""
    agent, calls = _mk_agent(monkeypatch, [
        _mk_resp(" p1", finish="length"),
        _mk_resp(" p2", finish="length"),
        _mk_resp(" p3", finish="length"),
    ])
    resp = _mk_resp("h", finish="length")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out.choices[0].message.content == "h p1 p2 p3"
    assert out.choices[0].finish_reason == "length"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_recover_failure_fail_open(monkeypatch):
    """恢复调用抛异常 → 返回已拼接内容（fail-open）。"""
    agent, calls = _mk_agent(monkeypatch, [RuntimeError("boom")])
    resp = _mk_resp("h", finish="length")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out.choices[0].message.content == "h"
    assert out.choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_recover_limit_zero_disabled(monkeypatch):
    """llm.output_recovery_limit=0 关闭续写恢复。"""
    agent, calls = _mk_agent(
        monkeypatch, [], config={"llm": {"output_recovery_limit": 0}},
    )
    resp = _mk_resp("h", finish="length")
    out = await agent._recover_output_truncation(resp, [{"role": "user", "content": "q"}], [])
    assert out.choices[0].message.content == "h"
    assert calls == []


# ---------------------------------------------------------------------------
# R17 #13：400 溢出自适应
# ---------------------------------------------------------------------------

class _Fake400(Exception):
    status_code = 400


def test_parse_context_overflow_formats():
    exact = _Fake400(
        "Error: input length and `max_tokens` exceed context limit: 188059 + 20000 > 200000"
    )
    assert parse_context_overflow(exact) == (188059, 200000)

    loose = _Fake400(
        "This model's maximum context length is 131072 tokens. "
        "However, you requested 140000 tokens (120000 input tokens and 20000 max_tokens)"
    )
    assert parse_context_overflow(loose) == (120000, 131072)

    assert parse_context_overflow(_Fake400("bad request: invalid model")) is None


def test_compute_overflow_max_tokens():
    assert compute_overflow_max_tokens(120000, 131072) == 131072 - 120000 - 1000
    assert compute_overflow_max_tokens(130000, 131072) is None  # 剩余空间 < 3000


@pytest.mark.asyncio
async def test_call_with_retry_overflow_downgrade(monkeypatch):
    """400 溢出 → 下调 max_tokens 立即重试成功。"""
    err = _Fake400(
        "input length and `max_tokens` exceed context limit: 188059 + 20000 > 200000"
    )
    seen_kwargs = []

    class Client:
        async def chat_completions(self, msgs, **kw):
            seen_kwargs.append(dict(kw))
            if len(seen_kwargs) == 1:
                raise err
            return _mk_resp("ok")

    async def no_sleep(s):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    out = await call_with_retry(Client(), [{"role": "user", "content": "q"}],
                                max_tokens=20000)
    assert out.choices[0].message.content == "ok"
    assert seen_kwargs[0]["max_tokens"] == 20000
    # 200000 - 188059 - 1000 = 10941
    assert seen_kwargs[1]["max_tokens"] == 10941


@pytest.mark.asyncio
async def test_call_with_retry_overflow_too_big_raises(monkeypatch):
    """输入本身太大（剩余 < 3000）→ 不恢复，按 400 抛出。"""
    err = _Fake400(
        "input length and `max_tokens` exceed context limit: 199000 + 20000 > 200000"
    )

    class Client:
        async def chat_completions(self, msgs, **kw):
            raise err

    with pytest.raises(_Fake400):
        await call_with_retry(Client(), [{"role": "user", "content": "q"}],
                              max_tokens=20000)


@pytest.mark.asyncio
async def test_call_with_retry_plain_400_raises():
    """无溢出报文的 400 照旧立即抛。"""
    class Client:
        async def chat_completions(self, msgs, **kw):
            raise _Fake400("invalid model")

    with pytest.raises(_Fake400):
        await call_with_retry(Client(), [{"role": "user", "content": "q"}],
                              max_tokens=100)


# ---------------------------------------------------------------------------
# R17 #44：unattended 退避帽 5min
# ---------------------------------------------------------------------------

def test_compute_backoff_caps():
    """普通模式 60s 帽；unattended 5min 帽。"""
    assert _compute_backoff(
        attempt=30, initial_backoff=1.0, retry_after=None, jitter_ratio=0,
    ) == 60.0
    assert _compute_backoff(
        attempt=30, initial_backoff=1.0, retry_after=None, jitter_ratio=0,
        max_backoff=300.0,
    ) == 300.0
    # 大 retry_after 也被帽住
    assert _compute_backoff(
        attempt=0, initial_backoff=1.0, retry_after=9999.0, jitter_ratio=0,
        max_backoff=300.0,
    ) == 300.0
