# tests/test_goal.py
"""GoalState 状态机 + 持久化的单元测试。

仅覆盖状态机本身，主循环集成见下方。
"""
import json
import logging
from pathlib import Path

from agent.goal import GoalState


def test_goal_state_initial():
    g = GoalState(objective="测试目标")
    assert g.status == "active"
    assert g.iteration_count == 0
    assert g.token_budget == 0
    assert g.pause_reason is None
    assert g.goal_id  # 非空


def test_goal_state_save_and_load(tmp_path: Path):
    g = GoalState(objective="完成 X", token_budget_limit=100_000)
    g.iteration_count = 3
    g.token_budget = 5000
    g.task_ids = ["task_1", "task_2"]
    g.save(tmp_path / ".goal" / "current.json")

    loaded = GoalState.load(tmp_path / ".goal" / "current.json")
    assert loaded is not None
    assert loaded.objective == "完成 X"
    assert loaded.iteration_count == 3
    assert loaded.token_budget == 5000
    assert loaded.token_budget_limit == 100_000
    assert loaded.task_ids == ["task_1", "task_2"]
    assert loaded.status == "active"


def test_goal_load_missing_file_returns_none(tmp_path: Path):
    """文件不存在时返回 None（调用方决定是否新建）。"""
    loaded = GoalState.load(tmp_path / "nonexistent.json")
    assert loaded is None


def test_goal_load_corrupted_returns_none(tmp_path: Path, caplog):
    """损坏文件返回 None + log warning。"""
    path = tmp_path / "bad.json"
    path.write_text("not json {{{", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = GoalState.load(path)
    assert loaded is None


def test_goal_save_atomic(tmp_path: Path):
    """save 用 tmp + rename 原子写。"""
    g = GoalState(objective="x")
    path = tmp_path / ".goal" / "current.json"
    g.save(path)
    assert path.exists()
    # 不应该有残留 tmp 文件
    # （注意：Path 对象 bool() 恒为 True，必须用 .exists()）
    assert not (tmp_path / ".goal" / "current.json.tmp").exists()


def test_goal_pause_and_resume():
    g = GoalState(objective="x")
    g.pause(reason="manual")
    assert g.status == "paused"
    assert g.pause_reason == "manual"
    g.resume()
    assert g.status == "active"
    assert g.pause_reason is None


def test_goal_complete():
    g = GoalState(objective="x")
    g.complete()
    assert g.status == "completed"


def test_goal_cancel():
    g = GoalState(objective="x")
    g.cancel()
    assert g.status == "cancelled"


def test_goal_evaluate_continue_no_conditions():
    """无 budget 限制 + 无完成条件 → continue。"""
    g = GoalState(objective="x")
    decision = g.evaluate_after_turn(tokens_used=100)
    assert decision == "continue"
    assert g.iteration_count == 1
    assert g.token_budget == 100


def test_goal_evaluate_pause_on_budget():
    """超 budget → pause。"""
    g = GoalState(objective="x", token_budget_limit=1000)
    decision = g.evaluate_after_turn(tokens_used=1500)
    assert decision == "pause"
    assert g.pause_reason == "budget_exceeded"


def test_goal_evaluate_complete_when_all_tasks_done():
    """所有关联任务完成 → complete（需要注入完成状态）。"""
    g = GoalState(objective="x")
    decision = g.evaluate_after_turn(tokens_used=100, all_tasks_done=True)
    assert decision == "complete"


def test_goal_pause_on_network():
    """网络异常 pause。"""
    g = GoalState(objective="x")
    g.pause(reason="network")
    assert g.status == "paused"
    assert g.pause_reason == "network"


def test_goal_notes_appended():
    """pause/resume 等事件追加到 notes。"""
    g = GoalState(objective="x")
    g.pause(reason="manual")
    assert any("manual" in n for n in g.notes)
    g.resume()
    assert any("resume" in n.lower() for n in g.notes)


# ============================================================================
# 主循环集成测试
# ============================================================================
# 策略：抽纯函数（_build_goal_continue_message / _build_channel_injection /
# _build_mail_injection）测，再用一个集成测试验证 AIAgent 字段接线。

import asyncio  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from agent.channel_inbox import ChannelInbox  # noqa: E402
from agent.team.mailbox import Mailbox  # noqa: E402


def _make_mock_llm_client(response_text="hello", tool_calls=None):
    """构造 mock LLMClient（async chat_completions，返回 OpenAI 兼容响应）。"""
    msg = SimpleNamespace(content=response_text, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5,
        prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
    )
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# 纯函数测试（不需要 AIAgent 实例）
# ---------------------------------------------------------------------------

def test_build_goal_continue_message_basic():
    """goal active 时构造 <continue_goal> ephemeral user 消息。"""
    from agent import _build_goal_continue_message
    g = GoalState(objective="完成 X", iteration_count=3)
    msg = _build_goal_continue_message(g)
    assert msg is not None
    assert msg["role"] == "user"
    assert msg.get("_ephemeral") is True
    content = msg["content"]
    assert "<continue_goal" in content
    assert "完成 X" in content
    assert 'iteration="3"' in content or "iteration=3" in content


def test_build_goal_continue_message_none_returns_none():
    """goal_state=None 时返回 None（调用方跳过注入）。"""
    from agent import _build_goal_continue_message
    assert _build_goal_continue_message(None) is None


def test_build_channel_injection_with_unconsumed(tmp_path):
    """channel inbox 有未消费消息时构造 <channel_push> ephemeral。"""
    from agent import _build_channel_injection
    inbox = ChannelInbox(tmp_path / ".inbox_base")
    inbox.push("server_A", {"event": "build_done"})
    msg = _build_channel_injection(inbox)
    assert msg is not None
    assert msg["role"] == "user"
    assert msg.get("_ephemeral") is True
    assert "<channel_push" in msg["content"]
    assert "server_A" in msg["content"]
    assert "build_done" in msg["content"]
    # 注入后已 mark_consumed
    assert inbox.unconsumed() == []


def test_build_channel_injection_empty_returns_none(tmp_path):
    """channel inbox 无消息时返回 None。"""
    from agent import _build_channel_injection
    inbox = ChannelInbox(tmp_path / ".inbox_base")
    assert _build_channel_injection(inbox) is None


def test_build_channel_injection_none_returns_none():
    """inbox=None 时返回 None。"""
    from agent import _build_channel_injection
    assert _build_channel_injection(None) is None


def test_build_mail_injection_with_unread(tmp_path):
    """mailbox 有未读邮件时构造 <mail> ephemeral。"""
    from agent import _build_mail_injection
    mailbox = Mailbox(tmp_path)
    mailbox.send(to="main", from_="alice", content="hello")
    msg = _build_mail_injection(mailbox, agent_name="main")
    assert msg is not None
    assert msg["role"] == "user"
    assert msg.get("_ephemeral") is True
    assert "<mail" in msg["content"]
    assert "alice" in msg["content"]
    assert "hello" in msg["content"]
    # 注入后已 mark_read
    assert mailbox.check_unread("main") == []


def test_build_mail_injection_empty_returns_none(tmp_path):
    """mailbox 无未读邮件时返回 None。"""
    from agent import _build_mail_injection
    mailbox = Mailbox(tmp_path)
    assert _build_mail_injection(mailbox, agent_name="main") is None


def test_build_mail_injection_none_returns_none():
    """mailbox=None 或 agent_name 为空时返回 None。"""
    from agent import _build_mail_injection
    assert _build_mail_injection(None, agent_name="main") is None


# ---------------------------------------------------------------------------
# AIAgent 字段接线测试（不跑主循环）
# ---------------------------------------------------------------------------

def test_ai_agent_has_goal_state_field(tmp_path):
    """AIAgent 构造后 goal_state 默认 None，setter 真生效。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    assert agent._goal_state is None
    g = GoalState(objective="x")
    agent.set_goal_state(g)
    assert agent._goal_state is g


def test_ai_agent_has_channel_inbox_field(tmp_path):
    """AIAgent 构造后 channel_inbox 默认 None，setter 真生效。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    assert agent._channel_inbox is None
    inbox = ChannelInbox(tmp_path / ".inbox_base")
    agent.set_channel_inbox(inbox)
    assert agent._channel_inbox is inbox


def test_ai_agent_has_mailbox_field(tmp_path):
    """AIAgent 构造后 mailbox 默认 None，setter 真生效。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    assert agent._mailbox is None
    assert agent._agent_name == "main"  # 默认值
    mailbox = Mailbox(tmp_path)
    agent.set_mailbox(mailbox, agent_name="worker-1")
    assert agent._mailbox is mailbox
    assert agent._agent_name == "worker-1"


# ---------------------------------------------------------------------------
# 集成测试：主循环 ephemeral 不进持久化 + system prompt 不变
# ---------------------------------------------------------------------------

async def test_goal_continue_does_not_modify_system_prompt(tmp_path):
    """goal active 时 system prompt 不变（保护 prompt cache）。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    agent.llm_client = _make_mock_llm_client(response_text="工作完成")
    sp_before = agent._get_system_prompt()

    g = GoalState(objective="测试目标")
    agent.set_goal_state(g)
    # 模拟 run_conversation 完成一轮（不会 continue，因为 mock 单轮就 return）
    await agent.chat("start")

    sp_after = agent._get_system_prompt()
    assert sp_before == sp_after, "system prompt 在 goal active 时不能变"


async def test_goal_continue_message_not_in_persisted_history(tmp_path):
    """<continue_goal> ephemeral 消息不进 conversation_history。"""
    from agent import AIAgent
    from agent import _build_goal_continue_message

    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    agent.llm_client = _make_mock_llm_client(response_text="ok")

    g = GoalState(objective="x")
    agent.set_goal_state(g)

    # 调用 helper（模拟主循环逻辑）
    msg = _build_goal_continue_message(g)
    # 关键断言：helper 返回的消息有 _ephemeral 标记
    assert msg is not None
    assert msg.get("_ephemeral") is True

    # 模拟主循环：把消息加到 messages（发给 LLM 的）但不加到 conversation_history
    # 这里直接验证 conversation_history 里没有 <continue_goal>
    await agent.chat("hi")
    for m in agent.conversation_history:
        content = m.get("content", "")
        assert "<continue_goal" not in str(content), \
            "ephemeral 消息不能进 conversation_history"


async def test_channel_injection_in_assemble_turn_messages(tmp_path):
    """_assemble_turn_messages 注入 <channel_push> ephemeral，不进 history。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    inbox = ChannelInbox(tmp_path / ".inbox_base")
    inbox.push("server_X", {"event": "ready"})
    agent.set_channel_inbox(inbox)

    # 调 _assemble_turn_messages 前需要先把 user 消息塞 history
    agent.conversation_history.append({"role": "user", "content": "hi"})

    messages = agent._assemble_turn_messages("sys_prompt", {})

    # 找出 channel_push 注入
    found_channel = None
    for m in messages:
        if "<channel_push" in str(m.get("content", "")):
            found_channel = m
            break
    assert found_channel is not None, "channel_push 消息应注入 messages"
    # 但 conversation_history 不含
    for m in agent.conversation_history:
        assert "<channel_push" not in str(m.get("content", ""))


async def test_mail_injection_in_assemble_turn_messages(tmp_path):
    """_assemble_turn_messages 注入 <mail> ephemeral，不进 history。"""
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    mailbox = Mailbox(tmp_path)
    mailbox.send(to="main", from_="bob", content="ping")
    agent.set_mailbox(mailbox, agent_name="main")

    agent.conversation_history.append({"role": "user", "content": "hi"})

    messages = agent._assemble_turn_messages("sys_prompt", {})

    found_mail = None
    for m in messages:
        if "<mail" in str(m.get("content", "")):
            found_mail = m
            break
    assert found_mail is not None, "<mail> 消息应注入 messages"
    for m in agent.conversation_history:
        assert "<mail" not in str(m.get("content", ""))


async def test_goal_auto_pause_on_network_error(tmp_path):
    """_call_llm_with_escalation 网络异常时 goal 自动 pause。"""
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    # mock LLM 抛网络异常
    err_client = MagicMock()
    err_client.chat_completions = AsyncMock(
        side_effect=ConnectionError("connection timeout"),
    )
    agent.llm_client = err_client

    g = GoalState(objective="x")
    agent.set_goal_state(g)

    # 跑主循环（会因 LLM 异常退出，但 goal 应已 pause）
    await agent.chat("start")

    assert g.status == "paused"
    assert g.pause_reason == "network"


async def test_goal_not_paused_on_non_network_error(tmp_path):
    """非网络异常不触发 goal pause（如 400 参数错误）。"""
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
    )
    err_client = MagicMock()
    err_client.chat_completions = AsyncMock(
        side_effect=ValueError("invalid parameter"),
    )
    agent.llm_client = err_client

    g = GoalState(objective="x")
    agent.set_goal_state(g)

    await agent.chat("start")

    # 非网络异常不 pause goal
    assert g.status == "active"


# ============================================================================
# pause() 集中通知
# ============================================================================

def test_pause_notifies_for_all_reasons():
    """pause() 集中通知（network/budget/manual 三原因一处接）。

    lazy import 下 patch agent.notifier.notify（函数内 import 每次调用时
    从 agent.notifier 模块取名字，patch 模块属性即可命中）。
    """
    from unittest.mock import patch
    g = GoalState(objective="x")
    with patch("agent.notifier.notify") as mock_notify:
        g.pause(reason="network")
        g.pause(reason="budget_exceeded")  # resume 后再 pause
        g.resume()
        g.pause(reason="manual")
    assert mock_notify.call_count == 3


def test_pause_notify_failure_does_not_break_state_machine():
    """notify 抛异常绝不炸状态机（fail-open，goal.py 被大量单测直接调）。"""
    from unittest.mock import patch
    g = GoalState(objective="x")
    with patch("agent.notifier.notify", side_effect=RuntimeError("boom")):
        g.pause(reason="network")
    # 状态机语义完整保留
    assert g.status == "paused"
    assert g.pause_reason == "network"
    assert "paused: reason=network" in g.notes[-1]


# ============================================================================
# 预算未用完"踢一脚"（nudge）
# ============================================================================

class TestShouldNudge:
    """预算未用完 + 最近有进展 → 该踢一脚让它继续。"""

    def _state(self, limit=100_000):
        from agent.goal import GoalState
        return GoalState(objective="修 lint", token_budget_limit=limit)

    def test_nudge_when_budget_left_and_progress(self):
        gs = self._state()
        gs.iteration_count = 3
        gs.token_budget = 50_000
        assert gs.should_nudge(recent_tool_success=True) is True

    def test_no_nudge_without_progress(self):
        gs = self._state()
        gs.iteration_count = 3
        gs.token_budget = 50_000
        assert gs.should_nudge(recent_tool_success=False) is False

    def test_no_nudge_when_budget_mostly_used(self):
        gs = self._state()
        gs.iteration_count = 3
        gs.token_budget = 95_000
        assert gs.should_nudge(recent_tool_success=True) is False

    def test_no_nudge_without_limit_or_early(self):
        gs = self._state(limit=None)
        gs.iteration_count = 3
        assert gs.should_nudge(recent_tool_success=True) is False
        gs2 = self._state()
        gs2.iteration_count = 1  # 刚开始，模型还没收尾过，无需踢
        assert gs2.should_nudge(recent_tool_success=True) is False


# ============================================================================
# nudge 主循环接线（防 silent-dead-code：单元测试过 ≠ 生产路径生效）
# ============================================================================

async def test_goal_nudge_wiring_in_run_conversation(tmp_path):
    """宣布完成 + 预算剩很多 → 注入 [goal nudge] 继续，第二次才放行 complete。

    LLM 响应序列：
      1. tool_calls（fake 工具成功 → _last_turn_had_tool_success=True）
      2. 最终响应（iteration 0→1，任务未全完 → <continue_goal>）
      3. 最终响应（iteration 1→2，任务未全完 → <continue_goal>）
      4. 最终响应（iteration=2 + 任务全完 + 预算剩 → nudge！不 complete）
      5. 最终响应（_nudged_this_turn 已置 → 正常 evaluate complete）
    """
    from agent import AIAgent
    from tools.registry import registry

    tool_name = "_r26_fake_ok_tool"
    registry.register(
        name=tool_name,
        toolset="test",
        schema={"name": tool_name, "description": "fake ok", "parameters": {}},
        handler=lambda args, **kw: json.dumps({"ok": True}, ensure_ascii=False),
        isConcurrencySafe=True,
    )
    try:
        agent = AIAgent(
            api_key="fake", model="test",
            enabled_toolsets=[], omnimate_home=tmp_path,
        )

        def _resp(text="", tool_calls=None):
            msg = SimpleNamespace(content=text, tool_calls=tool_calls)
            usage = SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=msg, finish_reason="stop",
                )],
                usage=usage,
            )

        tc = SimpleNamespace(
            id="call_r26_1", type="function",
            function=SimpleNamespace(name=tool_name, arguments="{}"),
        )
        responses = [_resp(tool_calls=[tc])] + [
            _resp(f"done {i}") for i in range(4)
        ]
        recorded = []

        async def _fake_cc(messages, *a, **kw):
            recorded.append([dict(m) for m in messages])
            return responses.pop(0)

        client = MagicMock()
        client.chat_completions = _fake_cc
        client.model = "mock-model"
        agent.llm_client = client

        # 任务"全完成"只在第 3 次 _check_all_goal_tasks_done 调用起为 True
        # （前两个最终响应轮任务未全完 → <continue_goal>；第 3 个最终响应轮
        # 恰好是 all_done + 预算剩 → nudge 拦截）
        check_calls = {"n": 0}

        def fake_check():
            check_calls["n"] += 1
            return check_calls["n"] >= 3

        agent._check_all_goal_tasks_done = fake_check

        g = GoalState(objective="修 20 个文件", token_budget_limit=100_000)
        g.token_budget = 50_000  # 预算剩一半
        agent.set_goal_state(g)

        await agent.chat("start")

        # 第二次宣布完成被放行（nudge 只拦一次）
        assert g.status == "completed"
        # nudge 轮不进 evaluate_after_turn：2 次 continue + 1 次 complete = 3
        assert g.iteration_count == 3
        # nudge 消息确实发给了 LLM（ephemeral 进 messages），
        # 且不进 conversation_history（保护持久化 + prompt cache）
        seen_nudge = any(
            "[goal nudge]" in str(m.get("content", ""))
            for msgs in recorded for m in msgs
        )
        assert seen_nudge, "[goal nudge] 应注入发给 LLM 的 messages"
        for m in agent.conversation_history:
            content = str(m.get("content", ""))
            assert "[goal nudge]" not in content, "nudge 不能进 conversation_history"
    finally:
        with registry._lock:
            registry._tools.pop(tool_name, None)
