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


# ============================================================================
# Task D4 fix 新增覆盖：async 压缩链 + 记忆检索链
# 验证这些路径能拿到真实 LLM 响应（chat_completions 被 await，不是返回 coroutine）
# ============================================================================

def test_compress_if_needed_is_coroutine():
    """compress_if_needed 必须是 async def（Task D4 fix）。"""
    from agent.context_pipeline import compress_if_needed
    assert inspect.iscoroutinefunction(compress_if_needed), (
        "compress_if_needed 必须是 async def（Task D4 fix）"
    )


def test_llm_compact_is_coroutine():
    """llm_compact 必须是 async def（Task D4 fix）。"""
    from agent.context_pipeline import llm_compact
    assert inspect.iscoroutinefunction(llm_compact), (
        "llm_compact 必须是 async def（Task D4 fix）"
    )


def test_summarize_conversation_is_coroutine():
    """_summarize_conversation 必须是 async def（Task D4 fix）。"""
    from agent.context_compressor import _summarize_conversation
    assert inspect.iscoroutinefunction(_summarize_conversation), (
        "_summarize_conversation 必须是 async def（Task D4 fix）"
    )


def test_retrieve_relevant_is_coroutine():
    """retrieve_relevant 必须是 async def（Task D4 fix）。"""
    from agent.memory_retriever import retrieve_relevant
    assert inspect.iscoroutinefunction(retrieve_relevant), (
        "retrieve_relevant 必须是 async def（Task D4 fix）"
    )


def test_aux_llm_router_chat_completions_is_coroutine():
    """AuxLLMRouter.chat_completions 必须是 async def（Task D4 fix）。"""
    from agent.aux_llm import AuxLLMRouter
    assert inspect.iscoroutinefunction(AuxLLMRouter.chat_completions), (
        "AuxLLMRouter.chat_completions 必须是 async def（Task D4 fix）"
    )


async def test_compress_llm_actually_awaited(tmp_path):
    """L4 压缩路径：chat_completions 被 await（能拿到真实响应）。

    之前 sync 调 async 方法返回 coroutine，except 捕获 TypeError 后降级。
    """
    from agent.context_pipeline import llm_compact

    # 构造 async mock client，记录被调用
    call_count = [0]

    class _AsyncClient:
        async def chat_completions(self, msgs, **kw):
            call_count[0] += 1
            m = SimpleNamespace(content="这是真实 LLM 摘要")
            return SimpleNamespace(choices=[SimpleNamespace(message=m)])

    # 构造超阈值 messages
    msgs = [{"role": "system", "content": "s"}]
    msgs += [{"role": "user", "content": f"u{i}"} for i in range(80)]
    msgs += [{"role": "assistant", "content": f"a{i}"} for i in range(80)]

    out, changed = await llm_compact(
        msgs, llm_client=_AsyncClient(), model="x",
        token_threshold=10, keep_recent=5,
    )
    assert changed is True
    assert call_count[0] == 1, "chat_completions 必须被 await 调用一次"
    # summary 应含真实摘要（而不是规则提取的"[规则提取]"前缀）
    placeholder_content = out[1]["content"]
    assert "真实 LLM 摘要" in placeholder_content, (
        "L4 应拿到真实 LLM 响应，不是降级到规则提取"
    )


async def test_memory_retrieval_llm_actually_awaited():
    """记忆检索：chat_completions 被 await（能拿到真实响应）。

    之前 sync 调 async 方法返回 coroutine，except 捕获 TypeError 后返回空 list。
    """
    from agent.memory_retriever import retrieve_relevant

    call_count = [0]

    class _AsyncClient:
        async def chat_completions(self, msgs, **kw):
            call_count[0] += 1
            m = SimpleNamespace(content='["general#123", "debug#456"]')
            return SimpleNamespace(choices=[SimpleNamespace(message=m)])

    result = await retrieve_relevant(
        query="如何配置 pytest",
        index_text="- [pytest](.memory/general.md) — pytest 配置",
        llm_client=_AsyncClient(),
        model="test",
    )
    assert call_count[0] == 1, "chat_completions 必须被 await 调用一次"
    assert result == ["general#123", "debug#456"], (
        "记忆检索应返回真实 LLM 响应，不是空 list"
    )


# ============================================================================
# Task P1.2: L5 reactive_compact 响应式回压 feature flag 测试
# 验证：
#   - flag OFF：context_length_exceeded 错误不触发响应式回压（返回 None）
#   - flag ON：context_length_exceeded 错误触发响应式回压（返回 _REACTIVE_RETRY）
#   - flag ON 但已达上限：不二次触发（max_per_session）
#   - Task D：flag ON + 冷却窗口外：可多次触发
# ============================================================================

def _make_prompt_too_long_llm_client():
    """构造一个 LLM client，chat_completions 永远抛 context_length_exceeded。"""
    client = SimpleNamespace()
    err = Exception(
        "Error code: 400 - {'error': {'message': 'This model's maximum context length is 128000 tokens. "
        "However, your messages resulted in 150000 tokens.', 'type': 'invalid_request_error', "
        "'code': 'context_length_exceeded'}}"
    )
    client.chat_completions = AsyncMock(side_effect=err)
    client.chat_completions_stream = AsyncMock(side_effect=err)
    client.model = "mock-model"
    return client


async def test_reactive_compact_flag_off_skips_retry(tmp_path):
    """flag OFF：context_length_exceeded 不触发响应式回压。

    默认 DEFAULT_CONFIG["features"]["reactive_compact"]["enabled"] = False，
    所以 AIAgent 默认配置下该路径应被跳过。
    """
    agent, _ = _make_minimal_agent(tmp_path)
    # 显式确认默认 flag 关（DEFAULT_CONFIG 已设 False，双保险）
    agent.config.setdefault("features", {})["reactive_compact"] = {"enabled": False}
    # 换成抛 context_length_exceeded 的 client
    agent.llm_client = _make_prompt_too_long_llm_client()
    agent._reacted = False

    response = await agent._call_llm_with_escalation(
        messages=[{"role": "user", "content": "hi"}],
        tool_schemas=[],
        system_prompt="你是助手",
    )
    # flag 关 → 不走响应式回压 → 返回 None（错误已塞回 history）
    assert response is None, (
        "reactive_compact flag OFF 时不应返回 _REACTIVE_RETRY，应返回 None"
    )
    # _reacted 不应被改（没触发回压）
    assert agent._reacted is False, (
        "flag OFF 时 _reacted 应保持 False（回压未触发）"
    )


async def test_reactive_compact_flag_on_triggers_retry(tmp_path):
    """flag ON：context_length_exceeded 触发响应式回压（返回 _REACTIVE_RETRY）。"""
    agent, _ = _make_minimal_agent(tmp_path)
    agent.config.setdefault("features", {})["reactive_compact"] = {"enabled": True}
    agent.llm_client = _make_prompt_too_long_llm_client()
    agent._reacted = False

    # 构造足够长的 messages（reactive_compact 会截到最近 5 条）
    msgs = [{"role": "system", "content": "你是助手"}]
    msgs += [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    msgs += [{"role": "assistant", "content": f"a{i}"} for i in range(20)]

    response = await agent._call_llm_with_escalation(
        messages=msgs,
        tool_schemas=[],
        system_prompt="你是助手",
    )
    assert response is agent._REACTIVE_RETRY, (
        "reactive_compact flag ON + context_length_exceeded 应返回 _REACTIVE_RETRY"
    )
    assert agent._reacted is True, "回压触发后 _reacted 应置 True"
    # conversation_history 应被压缩（远少于起始 41 条）
    assert len(agent.conversation_history) < 20, (
        f"压缩后 history 应远少于 20 条，实际 {len(agent.conversation_history)}"
    )


async def test_reactive_compact_max_per_session_reached(tmp_path):
    """flag ON 但本会话已达 max_per_session 上限：不二次触发（返回 None）。

    Task D：reactive_compact 改多次触发，gate 从 ``_reacted`` bool
    迁移到 ``session_state.reactive_count >= max_per_session``。
    """
    agent, _ = _make_minimal_agent(tmp_path)
    agent.config.setdefault("features", {})["reactive_compact"] = {"enabled": True}
    agent.llm_client = _make_prompt_too_long_llm_client()
    # 模拟已达到单会话上限（默认 5）
    agent._compress_session_state.reactive_count = 5

    response = await agent._call_llm_with_escalation(
        messages=[{"role": "user", "content": "hi"}],
        tool_schemas=[],
        system_prompt="你是助手",
    )
    assert response is None, (
        "本会话已达 reactive_compact 上限（reactive_count=5）应返回 None，不二次触发"
    )


async def test_reactive_compact_cooldown_blocks_within_window(tmp_path):
    """Task D：flag ON + 冷却窗口内（< 60s）→ 拒绝触发（返回 None）。"""
    import time as _time
    agent, _ = _make_minimal_agent(tmp_path)
    agent.config.setdefault("features", {})["reactive_compact"] = {"enabled": True}
    agent.llm_client = _make_prompt_too_long_llm_client()
    # 模拟 30s 前触发过一次（< 60s 冷却窗口）
    agent._compress_session_state.reactive_count = 1
    agent._compress_session_state.reactive_last_at = _time.time() - 30

    msgs = [{"role": "system", "content": "你是助手"}]
    msgs += [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    msgs += [{"role": "assistant", "content": f"a{i}"} for i in range(20)]

    response = await agent._call_llm_with_escalation(
        messages=msgs,
        tool_schemas=[],
        system_prompt="你是助手",
    )
    assert response is None, (
        "冷却窗口内（< 60s）不应触发 reactive_compact"
    )
