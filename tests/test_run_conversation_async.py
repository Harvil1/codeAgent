"""run_conversation async 契约测试（Task D4）。

验证：
1. run_conversation / chat / _call_llm_with_escalation 是 coroutine function
2. 最简单的 async 对话流程（mock LLM 返回无 tool_call）能跑通
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_run_conversation_is_coroutine():
    """run_conversation 应该是 async def。"""
    from agent import AIAgent
    assert inspect.iscoroutinefunction(AIAgent.run_conversation), (
        "run_conversation 必须是 async def（Task D4 要求）"
    )


def test_chat_is_coroutine():
    """chat 应该是 async def。"""
    from agent import AIAgent
    assert inspect.iscoroutinefunction(AIAgent.chat), (
        "chat 必须是 async def（Task D4 要求）"
    )


def test_call_llm_with_escalation_is_coroutine():
    """_call_llm_with_escalation 应该是 async def。"""
    from agent import AIAgent
    assert inspect.iscoroutinefunction(AIAgent._call_llm_with_escalation), (
        "_call_llm_with_escalation 必须是 async def（Task D4 要求）"
    )


def _make_mock_response(response_text="你好，我是助手。", tool_calls=None):
    """构造 mock LLM 响应（OpenAI 兼容结构）。"""
    msg = SimpleNamespace(content=response_text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_mock_llm_client(response_text="你好，我是助手。", tool_calls=None):
    """构造 async mock LLMClient。

    chat_completions 是 AsyncMock（对齐 T_B1 后的 async LLM client）。
    """
    response = _make_mock_response(response_text, tool_calls)
    client = SimpleNamespace()
    client.chat_completions = AsyncMock(return_value=response)
    client.chat_completions_stream = AsyncMock(return_value=response)
    client.model = "mock-model"
    return client


def _make_minimal_agent(tmp_path=None):
    """构造最小可用的 AIAgent（mock LLM client，无 tool_call 响应）。

    返回 (agent, mock_llm_client)。
    """
    from agent import AIAgent

    mock_llm = _make_mock_llm_client()

    kwargs = {
        "api_key": "fake",
        "model": "mock-model",
        "enabled_toolsets": [],
    }
    if tmp_path is not None:
        kwargs["omnimate_home"] = tmp_path

    agent = AIAgent(**kwargs)
    agent.llm_client = mock_llm
    return agent, mock_llm


async def test_run_conversation_minimal_flow(tmp_path):
    """最简单的 async 对话流程：mock LLM 返回无 tool_call 的最终响应。"""
    agent, mock_llm = _make_minimal_agent(tmp_path)

    result = await agent.run_conversation("你好")
    assert result == "你好，我是助手。"
    # 验证 LLM 被调用过
    assert mock_llm.chat_completions.called


async def test_chat_delegates_to_run_conversation(tmp_path):
    """chat 方法应该 await run_conversation。"""
    agent, mock_llm = _make_minimal_agent(tmp_path)

    result = await agent.chat("测试")
    assert result == "你好，我是助手。"


async def test_call_llm_with_escalation_non_stream_path(tmp_path):
    """非流式路径：_call_llm_with_escalation 返回 LLM 响应。"""
    agent, mock_llm = _make_minimal_agent(tmp_path)

    response = await agent._call_llm_with_escalation(
        messages=[{"role": "user", "content": "hi"}],
        tool_schemas=[],
        system_prompt="你是助手",
    )
    assert response is not None
    assert response.choices[0].message.content == "你好，我是助手。"
