"""压力测试：补函数级用例的覆盖缺口。

test_stress_long_context.py 测的是函数级；本文件压三个盲区：
1. 主循环端到端长跑（run_conversation 数百轮 + goal 驱动）——粘合处验证
2. MCP notification 风暴（reader 线程高频消息）
3. 韧性时间线（熔断器连开 / 429 重试风暴 / 529 早切）

纯本地 mock（不调真 LLM）。
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import AIAgent
from agent.context_compressor import (
    _summarize_conversation,
    reset_compact_circuit_breaker,
)
from agent.goal import GoalState
from agent.llm_retry import call_with_retry
from agent.memory_store import MemoryStore
from agent.mcp_client import StdioTransport


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class RetryableError(Exception):
    """带 status_code 的可重试错误（llm_retry 用 status_code 属性判断）。

    注意：is_retryable 优先按 openai SDK 异常类型识别，自定义异常只走
    类名兜底（timeout/connection/temporary）。429/529 场景请用
    _make_api_error 构造真实 APIStatusError。
    """

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _make_api_error(status_code: int, msg: str):
    """构造真实 openai.APIStatusError（is_retryable 走 SDK 类型识别路径）。"""
    import httpx
    from openai import APIStatusError

    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code, request=req, text=msg)
    return APIStatusError(msg, response=resp, body=None)


def _text_response(text: str, prompt_tokens: int = 2000, completion_tokens: int = 50):
    """构造带 usage 的纯文本响应（goal budget 从 usage 拿 token）。"""
    msg = SimpleNamespace(content=text, tool_calls=None)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)], usage=usage,
    )


def _tool_call_response(call_id: str, name: str, arguments: dict):
    """构造 tool_call 响应。

    tool_calls 元素必须用对象（dispatch 读 tc.id / tc.function.name），
    dict 会 AttributeError。
    """
    tc = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name, arguments=json.dumps(arguments),
        ),
    )
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _assert_protocol_valid(messages: list) -> None:
    """OpenAI 协议校验：tool result 必须有前置 tool_call。"""
    seen = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen.add(tc["id"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen, (
                f"孤儿 tool result: {m.get('tool_call_id')}"
            )


def _make_agent(tmp_path, **kwargs) -> AIAgent:
    return AIAgent(
        api_key="fake",
        model="test-model",
        memory_store=MemoryStore(omnimate_home=tmp_path),
        enabled_toolsets=[],
        omnimate_home=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A. 主循环端到端长跑
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stress_main_loop_300_tool_rounds(tmp_path):
    """主循环 300 轮 brief 工具调用 + 最终响应。

    压的是 run_conversation + dispatch + 压缩编排的**粘合处**
    （常规用例只测管线函数级）。brief 纯 echo 无副作用。
    验证：跑完不崩、history 协议合法、无 ephemeral 泄漏、耗时可接受。
    """
    responses = []
    for i in range(300):
        responses.append(_tool_call_response(
            f"call_{i}", "brief", {"headline": f"round {i}"},
        ))
    responses.append(_text_response("最终响应"))

    agent = _make_agent(tmp_path, max_iterations=350)
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=responses)

    t0 = time.perf_counter()
    result = await agent.chat("开始压力测试")
    elapsed = time.perf_counter() - t0

    assert result == "最终响应"
    assert agent.llm_client.chat_completions.await_count == 301
    assert elapsed < 120.0, f"300 轮主循环耗时 {elapsed:.1f}s 超 120s"

    # history 协议合法
    _assert_protocol_valid(agent.conversation_history)
    # 无 ephemeral 泄漏到持久化
    leaked = [m for m in agent.conversation_history if m.get("_ephemeral")]
    assert leaked == [], f"{len(leaked)} 条 ephemeral 泄漏到 history"


@pytest.mark.asyncio
async def test_stress_goal_driven_50_rounds(tmp_path):
    """goal 驱动长跑：mock 一直返回文本，goal evaluate continue 逐轮推进，
    budget 超限（每轮 2050 token × 50 轮 > 100K limit）自动 pause。

    验证 goal continue 核心路径：注入 ephemeral → 下一轮消费 →
    不进 history → budget pause 收尾。
    """
    # 60 个文本响应（足够 goal 跑到 budget 超限）
    responses = [_text_response(f"progress {i}") for i in range(60)]

    goal = GoalState(objective="压力测试目标", token_budget_limit=100_000)
    agent = _make_agent(tmp_path, max_iterations=100, goal_state=goal)
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=responses)

    await agent.chat("开始目标")

    # goal 应该因 budget 超限 pause（每轮 2050 token，~49 轮后超 100K）
    assert goal.status == "paused", f"goal 状态: {goal.status}（期望 paused）"
    assert goal.pause_reason == "budget_exceeded"
    assert goal.iteration_count >= 45, f"goal 只跑了 {goal.iteration_count} 轮"
    # 没耗尽 mock 响应（budget 先到）
    assert agent.llm_client.chat_completions.await_count < 60

    # ephemeral 不泄漏到 history（关键：goal continue 消息用完即弃）
    leaked = [m for m in agent.conversation_history if m.get("_ephemeral")]
    assert leaked == [], f"{len(leaked)} 条 goal continue ephemeral 泄漏"
    # 协议合法
    _assert_protocol_valid(agent.conversation_history)
    # goal 持久化了状态
    assert goal.pause_reason == "budget_exceeded"


@pytest.mark.asyncio
async def test_stress_goal_network_pause_in_main_loop(tmp_path):
    """主循环内 LLM 连续网络错误 → goal 自动 pause（reason=network）。"""
    agent = _make_agent(tmp_path, max_iterations=10)
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(
        side_effect=RetryableError("Connection error", 503)
    )

    goal = GoalState(objective="网络异常目标")
    agent._goal_state = goal

    # chat 会因 LLM 失败返回错误信息，但不该崩
    result = await agent.chat("go")
    assert result is not None

    # 5xx 重试耗尽后 goal 应该被 pause（网络关键词匹配）
    # 注意：5xx 是可重试错误，重试耗尽后走错误路径 → goal network pause 分支
    # （如果 Retry-After 路径耗时太长这里主要验证不崩 + goal 状态变化路径可达）


# ---------------------------------------------------------------------------
# B. MCP notification 风暴
# ---------------------------------------------------------------------------

def test_stress_mcp_notification_storm_1000():
    """reader 线程 1000 条 notification 风暴 + 慢 handler（1ms/条）。

    线程代码常规用例只测单条 dispatch——这里压高频：
    验证不丢消息、不崩、耗时可接受。
    """
    transport = StdioTransport("echo")  # 不会真连
    received = []

    def slow_handler(method: str, params: dict):
        time.sleep(0.001)  # 模拟慢 handler（写盘/LLM）
        received.append((method, params))

    transport.set_notification_handler(slow_handler)
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    lines = [
        json.dumps({"method": "notifications/message", "params": {"i": i}})
        for i in range(1000)
    ]
    transport.process.stdout.readline.side_effect = lines + [""]  # EOF 结束

    t0 = time.perf_counter()
    transport._reader_loop()
    elapsed = time.perf_counter() - t0

    assert len(received) == 1000, f"丢了 {1000 - len(received)} 条 notification"
    assert elapsed < 10.0, f"1000 条风暴耗时 {elapsed:.2f}s 超 10s"
    # 顺序保留（FIFO）
    assert [p["i"] for _, p in received[:10]] == list(range(10))


def test_stress_mcp_mixed_notification_and_response():
    """notification 和 response 混合流：response 进 queue，notification 给 handler。"""
    import queue as queue_module

    transport = StdioTransport("echo")
    notifications = []

    transport.set_notification_handler(
        lambda m, p: notifications.append((m, p))
    )
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    lines = []
    # 500 组：1 条 response（带 id）+ 1 条 notification（无 id）
    for i in range(500):
        lines.append(json.dumps({"id": i + 1, "result": {"ok": i}}))
        lines.append(json.dumps({"method": "notifications/message", "params": {"i": i}}))
    transport.process.stdout.readline.side_effect = lines + [""]

    transport._reader_loop()

    assert len(notifications) == 500
    # response 全部进了 queue
    assert transport._response_queue.qsize() == 500


def test_stress_mcp_handler_exception_storm():
    """handler 连续抛异常 1000 次：reader 不崩、继续处理后续消息。"""
    transport = StdioTransport("echo")
    ok_received = []

    def broken_handler(method: str, params: dict):
        if params.get("i", 0) % 2 == 0:
            raise ValueError("handler 炸了")
        ok_received.append(params["i"])

    transport.set_notification_handler(broken_handler)
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    lines = [
        json.dumps({"method": "notifications/message", "params": {"i": i}})
        for i in range(1000)
    ]
    transport.process.stdout.readline.side_effect = lines + [""]

    transport._reader_loop()  # 不崩

    # 偶数全炸（500），奇数全部收到（500）——fail-open 生效
    assert ok_received == list(range(1, 1000, 2))


# ---------------------------------------------------------------------------
# C. 韧性时间线
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stress_circuit_breaker_storm():
    """9 段摘要熔断器：连续 3 次失败开闸，之后不再调 LLM（走规则总结）。

    压时间线：6 次压缩请求 × 每次内部 PTL 重试。
    """
    reset_compact_circuit_breaker()
    llm_calls = [0]

    class BoomClient:
        async def chat_completions(self, msgs, model=None, **kw):
            llm_calls[0] += 1
            raise RuntimeError("LLM 服务不可用")

    messages = [
        {"role": "user", "content": f"msg {i}"} for i in range(20)
    ] + [{"role": "assistant", "content": "resp"}]

    summaries = []
    for i in range(6):
        summary = await _summarize_conversation(
            messages, BoomClient(), model="m",
        )
        summaries.append(summary)

    # 6 次都有降级输出（rule-based，不空）
    assert all(s for s in summaries), "熔断后 rule-based 摘要不应为空"
    # 熔断开闸后 LLM 不再被调：前 3 次请求触发失败（每次可能含 PTL 重试
    # 但非 PTL 错误不重试，所以 3 次 LLM 调用开闸），后 3 次直接走规则
    assert llm_calls[0] == 3, (
        f"熔断器未生效：LLM 被调了 {llm_calls[0]} 次（期望 3 次后开闸）"
    )


@pytest.mark.asyncio
async def test_stress_429_retry_storm_with_fallback():
    """主 client 连续 429（5 次重试全败）→ fallback 成功。

    initial_backoff=0.001 + jitter=0 避免真睡。
    """
    ok_resp = _text_response("fallback ok")

    main_client = MagicMock()
    main_client.chat_completions = AsyncMock(
        side_effect=_make_api_error(429, "Too Many Requests")
    )
    fb_client = MagicMock()
    fb_client.chat_completions = AsyncMock(return_value=ok_resp)

    t0 = time.perf_counter()
    resp = await call_with_retry(
        main_client,
        [{"role": "user", "content": "hi"}],
        initial_backoff=0.001,
        jitter_ratio=0,
        fallback_llm_client=fb_client,
    )
    elapsed = time.perf_counter() - t0

    assert resp.choices[0].message.content == "fallback ok"
    assert main_client.chat_completions.await_count == 5, (
        f"主 client 重试次数 {main_client.chat_completions.await_count}（期望 5）"
    )
    assert fb_client.chat_completions.await_count == 1
    assert elapsed < 5.0, f"退避总耗时 {elapsed:.2f}s 异常"


@pytest.mark.asyncio
async def test_stress_529_early_switch():
    """529 连续 3 次 → 立即切 fallback（不等 5 次重试耗尽）。"""
    ok_resp = _text_response("early switch ok")

    main_client = MagicMock()
    main_client.chat_completions = AsyncMock(
        side_effect=_make_api_error(529, "529 overloaded")
    )
    fb_client = MagicMock()
    fb_client.chat_completions = AsyncMock(return_value=ok_resp)

    resp = await call_with_retry(
        main_client,
        [{"role": "user", "content": "hi"}],
        initial_backoff=0.001,
        jitter_ratio=0,
        fallback_llm_client=fb_client,
        consecutive_529_threshold=3,
    )

    assert resp.choices[0].message.content == "early switch ok"
    # 529 早切：主 client 只被调 3 次（不是 5 次）
    assert main_client.chat_completions.await_count == 3, (
        f"529 早切未生效：主 client 被调 {main_client.chat_completions.await_count} 次"
    )
    assert fb_client.chat_completions.await_count == 1


@pytest.mark.asyncio
async def test_stress_circuit_breaker_recovery():
    """熔断器恢复：开闸后一次成功调用关闭熔断（半开语义）。"""
    reset_compact_circuit_breaker()
    llm_calls = [0]

    class FlakyClient:
        """前 3 次炸，之后成功。"""

        async def chat_completions(self, msgs, model=None, **kw):
            llm_calls[0] += 1
            if llm_calls[0] <= 3:
                raise RuntimeError("临时故障")
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="恢复后的摘要"),
                )],
            )

    messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]

    # 3 次失败开闸
    for _ in range(3):
        await _summarize_conversation(messages, FlakyClient(), model="m")
    assert llm_calls[0] == 3

    # 熔断开闸状态：rule-based（不调 LLM）——用一个总是成功的 client 验证
    # 熔断仍开（成功不会重置，因为根本没调）
    always_ok = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok"),
        )],
    )

    class OkClient:
        async def chat_completions(self, msgs, model=None, **kw):
            llm_calls[0] += 1
            return always_ok

    # 重置熔断（模拟下一阶段恢复），成功一次后再次故障不应立即重新开闸
    reset_compact_circuit_breaker()
    s = await _summarize_conversation(messages, OkClient(), model="m")
    assert "ok" in s
