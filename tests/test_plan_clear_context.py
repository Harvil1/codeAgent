"""T9（核心机制对齐第 9 项）：plan 批准后可选"清上下文执行"。

- 审批回调协议扩展：(approved, feedback, clear_context) 三元组（二元组向后兼容）
- 选清空：history 截断为 [post_plan_brief（含计划全文）]，system prompt
  context 层重建（invalidate_system_prompt）
- 选否：现状路径完全兼容
- 计划全文在批准点已拿到（exit_plan_mode 携带），空计划不提供该选项
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(tmp_path, **overrides):
    from agent import AIAgent
    base = dict(
        base_url="http://localhost", api_key="k", model="m",
        enabled_toolsets=["core"],
    )
    base.update(overrides)
    with patch("agent.llm_client.create_llm_client") as mock:
        mock.return_value = MagicMock()
        return AIAgent(**base)


def _plan_result(plan_text):
    return json.dumps({
        "error": "需要审批", "error_type": "plan_approval_required",
        "plan": plan_text,
    }, ensure_ascii=False)


PLAN = "1. 改 agent/foo.py\n2. 加测试 tests/test_foo.py\n3. 跑回归"


def test_approval_clear_context(tmp_path):
    """三元组 (True, "", True) → history 截断为计划 brief + system prompt 重建。"""
    agent = _make_agent(tmp_path, omnimate_home=tmp_path)
    agent.conversation_history = [
        {"role": "user", "content": "调研一下"},
        {"role": "assistant", "content": "我看完了 100 个文件……"},
        {"role": "user", "content": "继续"},
    ]
    agent.plan_mode = True
    agent._system_prompt_built = True  # 模拟已构建缓存
    agent.plan_approval_callback = lambda p: (True, "", True)

    tc = SimpleNamespace()
    content = agent._maybe_handle_plan_approval(tc, _plan_result(PLAN))

    data = json.loads(content)
    assert data.get("plan_approved") is True
    assert data.get("context_cleared") is True
    # history 只剩一条 post_plan_brief user 消息
    assert len(agent.conversation_history) == 1
    msg = agent.conversation_history[0]
    assert msg["role"] == "user"
    assert "<post_plan_brief>" in msg["content"]
    assert "agent/foo.py" in msg["content"]  # 计划全文在场
    # system prompt 缓存已失效（context 层将重建）
    assert agent._system_prompt_built is False
    # 计划全文已记录（compact 恢复源）
    assert agent._last_approved_plan == PLAN
    assert agent.plan_mode is False


def test_approval_no_clear_backcompat(tmp_path):
    """二元组 (True, "") → 现状路径（history 不动）。"""
    agent = _make_agent(tmp_path, omnimate_home=tmp_path)
    history_before = [
        {"role": "user", "content": "调研"},
        {"role": "assistant", "content": "done"},
    ]
    agent.conversation_history = list(history_before)
    agent.plan_mode = True
    agent.plan_approval_callback = lambda p: (True, "")

    content = agent._maybe_handle_plan_approval(
        SimpleNamespace(), _plan_result(PLAN),
    )
    data = json.loads(content)
    assert data.get("plan_approved") is True
    assert "context_cleared" not in data
    assert agent.conversation_history == history_before


def test_approval_reject_unchanged(tmp_path):
    """拒绝路径不受影响。"""
    agent = _make_agent(tmp_path, omnimate_home=tmp_path)
    agent.plan_mode = True
    agent.plan_approval_callback = lambda p: (False, "改一下步骤 2")

    content = agent._maybe_handle_plan_approval(
        SimpleNamespace(), _plan_result(PLAN),
    )
    data = json.loads(content)
    assert data.get("plan_rejected") is True
    assert agent.plan_mode is True


# ---------------------------------------------------------------------------
# CLI 回调
# ---------------------------------------------------------------------------

def test_cli_callback_clear_option(monkeypatch, capsys):
    """CLI 审批输入 c → (True, "", True)。"""
    import builtins
    import cli as cli_mod

    monkeypatch.setattr(builtins, "input", lambda *a: "c")
    approved, feedback, clear = cli_mod.cli_plan_approval_callback(PLAN)
    assert approved is True and clear is True

    monkeypatch.setattr(builtins, "input", lambda *a: "y")
    approved, feedback, clear = cli_mod.cli_plan_approval_callback(PLAN)
    assert approved is True and clear is False

    monkeypatch.setattr(builtins, "input", lambda *a: "n")
    approved, feedback, clear = cli_mod.cli_plan_approval_callback(PLAN)
    assert approved is False and clear is False


def test_cli_callback_prompts_mention_clear(capsys):
    """提示文案包含清空选项说明。"""
    import builtins
    import cli as cli_mod

    with patch("builtins.input", side_effect=EOFError):
        cli_mod.cli_plan_approval_callback("plan")
    out = capsys.readouterr().out
    assert "清空" in out or "clear" in out.lower()
