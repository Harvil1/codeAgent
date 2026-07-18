"""压缩后上下文重锚定（Post-Compress Re-anchoring）测试。

覆盖：
- compress_if_needed 返回 changed=True 时 messages 末尾出现 <post_compress_brief>
- changed=False 时无此消息
- todo 为空时 brief 省略"当前任务清单"行
- plan_mode=True/False 文案不同
- brief 不进 conversation_history（保护持久化）
"""
import json
from unittest.mock import MagicMock, patch

import pytest


def _make_minimal_agent(**overrides):
    """构造一个最小 mock 的 AIAgent（不连真 LLM）。"""
    from agent import AIAgent
    base = dict(
        base_url="http://localhost",
        api_key="test-key",
        model="test-model",
        enabled_toolsets=["core"],
    )
    base.update(overrides)
    with patch("agent.llm_client.create_llm_client") as mock:
        mock.return_value = MagicMock()
        return AIAgent(**base)


def _make_final_response(text="done"):
    """构造无 tool_calls 的最终响应。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = text
    mock_resp.choices[0].message.tool_calls = None
    return mock_resp


# ============================================================================
# Task 1: 压缩后 brief 注入
# ============================================================================

def test_brief_injected_when_compressed():
    """compress_if_needed 返回 changed=True 时，messages 末尾含 <post_compress_brief>。"""
    agent = _make_minimal_agent()

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        # 模拟压缩：返回原 messages + changed=True
        return messages, True

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("ok")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        agent.run_conversation("test")

    assert captured_messages_list, "LLM 未被调用"
    last_msgs = captured_messages_list[0]
    brief_found = any(
        "<post_compress_brief>" in (m.get("content") or "")
        for m in last_msgs
    )
    assert brief_found, "messages 末尾缺 <post_compress_brief>"


def test_no_brief_when_not_compressed():
    """compress_if_needed 返回 changed=False 时，messages 不含 <post_compress_brief>。"""
    agent = _make_minimal_agent()

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, False  # 未压缩

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("ok")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        agent.run_conversation("test")

    last_msgs = captured_messages_list[0]
    brief_found = any(
        "<post_compress_brief>" in (m.get("content") or "")
        for m in last_msgs
    )
    assert not brief_found, "未压缩时不应注入 brief"


def test_brief_omits_todo_when_empty():
    """todo 为空时，brief 不含"当前任务清单"行。"""
    agent = _make_minimal_agent()
    # 默认 todo_manager 为空（无 todo）

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("ok")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        agent.run_conversation("test")

    last_msgs = captured_messages_list[0]
    brief_msg = next(
        (m for m in last_msgs if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    assert brief_msg is not None, "缺 brief"
    assert "当前任务清单" not in brief_msg["content"], (
        "todo 为空时不应有'当前任务清单'行"
    )


def test_brief_marks_plan_mode_when_true():
    """plan_mode=True 时，brief 标"计划模式"。"""
    agent = _make_minimal_agent()
    agent.plan_mode = True

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("ok")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        agent.run_conversation("test")

    brief_msg = next(
        (m for m in captured_messages_list[0]
         if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    assert brief_msg is not None
    assert "计划模式" in brief_msg["content"], "plan_mode=True 应标'计划模式'"
    assert "只能调研" in brief_msg["content"], "应说明约束"


def test_brief_marks_normal_mode_when_false():
    """plan_mode=False 时，brief 标"正常执行模式"。"""
    agent = _make_minimal_agent()
    assert agent.plan_mode is False

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("ok")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        agent.run_conversation("test")

    brief_msg = next(
        (m for m in captured_messages_list[0]
         if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    assert brief_msg is not None
    assert "正常执行模式" in brief_msg["content"], "plan_mode=False 应标'正常执行模式'"


def test_brief_not_in_conversation_history():
    """brief 是临时消息，不进 conversation_history（保护持久化）。"""
    agent = _make_minimal_agent()

    def fake_compress(messages, **kwargs):
        return messages, True

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry") as mock_llm:
        mock_llm.return_value = _make_final_response("ok")
        agent.run_conversation("test")

    # history 里有 user("test") + assistant("ok")，不应有 brief
    for msg in agent.conversation_history:
        content = msg.get("content") or ""
        assert "<post_compress_brief>" not in content, (
            "brief 不应进 conversation_history"
        )
