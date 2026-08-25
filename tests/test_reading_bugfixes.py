# -*- coding: utf-8 -*-
"""精读轮（桌面 OmniMate精读 附录 99）bugfix 回归测试。

  ★1 wrapper client 没有 close()：agent.cleanup() 的 getattr-close 循环
     对 OpenAICompatClient/AnthropicClient 是静默 no-op（X3 防泄漏修复失效）
  ★3 流式截断升级重试：重试结果没有 tool_calls 时，截断那次的半截
     tool_calls 残留进最终响应；且 usage 直接覆盖、截断那次的花费漏记
  ★4 retry_warning 注入没标 _ephemeral：当轮触发压缩时会混进正式历史落盘
"""
import asyncio
from types import SimpleNamespace


def _make_agent(tmp_path):
    from agent import AIAgent
    return AIAgent(
        base_url="http://127.0.0.1:1",
        api_key="test-key",
        model="test-model",
        omnimate_home=str(tmp_path),
        session_id="t",
    )


# ======================================================================
# ★1：wrapper client 必须有可调用的同步 close()
# ======================================================================

def test_close_on_wrapper_clients_exists_and_runs(tmp_path):
    from agent.llm_client import LLMClient, OpenAICompatClient, AnthropicClient

    # 基类：有无害的默认实现（getattr 可调用）
    assert callable(getattr(LLMClient(), "close", None))
    # 两个真包装类：类上确实定义了 close
    assert callable(getattr(OpenAICompatClient, "close", None))
    assert callable(getattr(AnthropicClient, "close", None))

    # 行为验证：同步调用真的会关掉底层 SDK client
    client = OpenAICompatClient(
        base_url="http://127.0.0.1:1", api_key="test-key", model="test-model",
    )
    closed = []

    class _FakeInner:
        async def close(self):
            closed.append(True)

    client.client = _FakeInner()
    client.close()  # 没有运行中的事件循环 → 走一次性 asyncio.run 分支
    assert closed == [True]


# ======================================================================
# ★3：流式截断升级重试要整体覆盖（tool_calls 清空 + usage 加总）
# ======================================================================

def test_stream_escalation_overrides_truncated_state(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    agent._stream_callback = lambda event: None

    def _frag(idx, id_, name, args):
        return SimpleNamespace(
            index=idx, id=id_,
            function=SimpleNamespace(name=name, arguments=args),
        )

    # 模拟一次被 max_tokens 掐断的流：工具参数只吐了一半 + finish=length
    chunks = [
        {"content": "", "tool_calls": [
            _frag(0, "call_1", "read_file", '{"path": "a'),
        ], "finish_reason": None, "usage": None},
        {"content": "", "tool_calls": [
            _frag(0, None, None, '.py"}'),
        ], "finish_reason": "length", "usage": {
            "prompt_tokens": 100, "completion_tokens": 10,
            "total_tokens": 110, "cache_read": 0, "cache_creation": 0,
        }},
    ]

    class _FakeStreamClient:
        async def chat_completions_stream(self, messages, *, tools=None, **kw):
            for c in chunks:
                yield c

    agent.llm_client = _FakeStreamClient()

    # 升级重试返回纯文本回答（没有 tool_calls）——旧版会把半截调用留下
    retried = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="完整回答", tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )

    async def _fake_retry(client, messages, **kw):
        return retried

    monkeypatch.setattr("agent.llm_retry.call_with_retry", _fake_retry)

    resp = asyncio.run(agent._call_llm_streaming(
        messages=[{"role": "user", "content": "hi"}], tools=None,
    ))

    msg = resp.choices[0].message
    assert msg.content == "完整回答"
    # 关键断言 1：半截 tool_calls 不许残留
    assert msg.tool_calls is None
    # 关键断言 2：两次调用的 token 都要记账（100+100 / 10+50）
    assert resp.usage.prompt_tokens == 200
    assert resp.usage.completion_tokens == 60


# ======================================================================
# ★4：retry_warning 必须是 ephemeral（不落盘）
# ======================================================================

def test_retry_warning_is_ephemeral(tmp_path):
    agent = _make_agent(tmp_path)
    agent._tool_failure_streak = 3
    agent._last_tool_error = "boom"

    msgs = [{"role": "user", "content": "hi"}]

    asyncio.run(agent._prepare_toolset_and_injections(msgs))

    warnings = [m for m in msgs if "retry_warning" in str(m.get("content", ""))]
    assert warnings, "应当注入一条 retry_warning"
    assert warnings[0].get("_ephemeral") is True, (
        "retry_warning 是临时提醒，必须带 _ephemeral 标记"
        "（否则当轮触发压缩会混进正式历史落盘）"
    )
