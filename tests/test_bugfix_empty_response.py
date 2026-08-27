"""空响应处理 bug 修复测试。

场景：LLM 流式返回 content=None + tool_calls=None，
当前 _finalize_response 静默返回空串让用户感到"突然断开"。

修复：
1. content 空 + reasoning_content 有值 → 用 reasoning 作为回复（思考模型纯 thinking）
2. 完全空（content + reasoning 都空） → 友好兜底消息
"""
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 场景 1：content 空 + reasoning_content 有值 → 用 reasoning 作为回复
# ---------------------------------------------------------------------------

def test_finalize_response_uses_reasoning_when_content_empty():
    """思考模型纯 thinking 响应：content=None 但 reasoning_content 有值。

    DeepSeek-reasoner 等模型有时只产出思考过程不给最终答案。
    reasoning_content 包含模型的推理过程，对用户有价值——应该作为回复显示。
    """
    from agent import AIAgent

    agent = _make_minimal_agent()
    assistant_msg = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning_content="让我分析一下坦克大战项目...\n当前进度阶段 7...",
        thinking_signature=None,
    )

    import asyncio as _aio
    # _finalize_response 是 async（STOP hook 移出事件循环线程）
    result = _aio.run(agent._finalize_response(assistant_msg, "继续做坦克大战"))
    # 应该用 reasoning_content 作为回复（不返回空串）
    assert result, f"回复不应为空（reasoning 有值时应作为回复），实际: {result!r}"
    assert "坦克大战" in result or "阶段" in result, (
        f"回复应含 reasoning 内容，实际: {result!r}"
    )


# ---------------------------------------------------------------------------
# 场景 2：完全空响应（content + reasoning 都空） → 友好兜底消息
# ---------------------------------------------------------------------------

def test_finalize_response_friendly_fallback_when_all_empty():
    """LLM 返回完全空响应（content=None + reasoning_content=None）。

    通常是网络抖动、流式断连、provider bug。
    不应静默返回空串让用户困惑，应给明确兜底消息。
    """
    from agent import AIAgent

    agent = _make_minimal_agent()
    assistant_msg = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning_content=None,
        thinking_signature=None,
    )

    import asyncio as _aio
    # _finalize_response 是 async（STOP hook 移出事件循环线程）
    result = _aio.run(agent._finalize_response(assistant_msg, "继续做坦克大战"))
    # 应返回友好兜底消息（不空）
    assert result, f"完全空响应时应返回友好兜底，不应是空串，实际: {result!r}"
    # 应含明确提示（"重试"/"空响应"/"网络"等关键词）
    assert any(kw in result for kw in ["重试", "空响应", "网络", "请", "retry", "empty"]), (
        f"兜底消息应明确告诉用户发生了啥，实际: {result!r}"
    )


# ---------------------------------------------------------------------------
# 场景 3：正常 content 不受影响
# ---------------------------------------------------------------------------

def test_finalize_response_normal_content_unchanged():
    """回归：正常 content 路径不受影响。"""
    from agent import AIAgent

    agent = _make_minimal_agent()
    assistant_msg = SimpleNamespace(
        content="我读了 level.js，下一步要写 game.js",
        tool_calls=None,
        reasoning_content=None,
        thinking_signature=None,
    )

    import asyncio as _aio
    # _finalize_response 是 async（STOP hook 移出事件循环线程）
    result = _aio.run(agent._finalize_response(assistant_msg, "继续做坦克大战"))
    assert result == "我读了 level.js，下一步要写 game.js"


# ---------------------------------------------------------------------------
# 场景 4：content 是空字符串（不是 None）也走 fallback
# ---------------------------------------------------------------------------

def test_finalize_response_empty_string_content_triggers_fallback():
    """content="" 也算空（与 None 等价处理）。"""
    from agent import AIAgent

    agent = _make_minimal_agent()
    assistant_msg = SimpleNamespace(
        content="",
        tool_calls=None,
        reasoning_content=None,
        thinking_signature=None,
    )

    import asyncio as _aio
    # _finalize_response 是 async（STOP hook 移出事件循环线程）
    result = _aio.run(agent._finalize_response(assistant_msg, "继续做坦克大战"))
    assert result, f"空串 content 也应触发兜底，实际: {result!r}"


# ---------------------------------------------------------------------------
# 场景 5：流式路径合成 message 时正确传递 reasoning_content
# ---------------------------------------------------------------------------

def test_streaming_message_passes_reasoning_content_through():
    """_call_llm_streaming 合成 message 时 reasoning_content 字段应保留。

    静态验证：源码合成的 SimpleNamespace 含 reasoning_content 字段。
    """
    from agent import AIAgent
    src = inspect.getsource(AIAgent._call_llm_streaming)
    # 合成 message 时应含 reasoning_content
    assert "reasoning_content" in src, (
        "_call_llm_streaming 合成 message 时应保留 reasoning_content 字段"
    )


# ---------------------------------------------------------------------------
# Helper：构造最小 AIAgent 实例（不走完整 __init__，只为测 _finalize_response）
# ---------------------------------------------------------------------------

def _make_minimal_agent():
    """构造最小 AIAgent 实例用于测 _finalize_response。"""
    from agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.conversation_history = []
    agent.on_response = None
    agent.memory_manager = None
    agent.hooks_registry = None
    agent.config = {"hooks": {"enabled": False}}
    agent._stop_fire_count = 0
    agent._stop_hook_forced = False
    agent.session_id = "test"
    agent._trigger_reflection_async = lambda: None
    agent._reflection_enabled = False
    agent._reflection_lock = __import__("threading").Lock()
    agent._last_reflection_turn = -1000
    agent._compress_session_state = MagicMock()
    return agent
