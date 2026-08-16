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
