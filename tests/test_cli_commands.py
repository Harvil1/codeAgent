# tests/test_cli_commands.py
"""CCAR8 Task 12：6 个新 CLI 命令的单元测试。

覆盖：
- /goal status（无 active goal）
- /goal pause/resume/clear（需要先有 active goal）
- /poor on/off/status
- /trace today（空 trace）
- /inbox（空 mailbox）
- /mailbox send/check/clear（CLI 包装层）
- /resume（无参数列表 + 有参数加载）

策略：构造最小 FakeRT（不跑主循环，不依赖 LLM）直接调 _handle_command。
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.goal import GoalState
from agent.team.mailbox import Mailbox
from agent.handoff import HandoffStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeAgent:
    """最小 AIAgent mock（/goal 命令读写它）。"""

    def __init__(self, tmp_path: Path):
        self.conversation_history = []
        self.session_id = "fake-session"
        self._goal_state = None
        self._mailbox = None
        self._agent_name = "main"
        self.goal_state_path = tmp_path / ".goal" / "current.json"

    def set_goal_state(self, gs):
        self._goal_state = gs

    def set_mailbox(self, mb, agent_name=None):
        self._mailbox = mb
        if agent_name:
            self._agent_name = agent_name


class _FakeRT:
    """最小 RuntimeContext mock。"""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path
        self.config = {
            "model": {"name": "deepseek-chat", "provider": "deepseek"},
            "goal": {"enabled": True, "default_token_budget": 100_000,
                     "reflection_interval": 5},
            "trace": {"enabled": True, "retention_days": 7},
        }
        self.agent = _FakeAgent(tmp_path)
        self.session_id = "fake-session"
        self.handoff_store = HandoffStore(tmp_path / ".handoff")
        self.mailbox = Mailbox(tmp_path / ".team")
        self.agent_name = "main"
        self.trace_sink = None  # 测试里按需建
        self.aux_llm_client = None
        self.aux_model = None
        self._poor_mode_on = False

        # 把 mailbox 挂到 agent（/mailbox 命令通过 agent._mailbox 读）
        self.agent._mailbox = self.mailbox


# ---------------------------------------------------------------------------
# /goal
# ---------------------------------------------------------------------------

def test_goal_status_no_active_goal(tmp_path, capsys):
    """/goal status 在无 active goal 时显示提示。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/goal status", rt)
    assert handled is True
    out = capsys.readouterr().out
    # 输出包含"无 active goal"或"无目标"等提示
    assert "无" in out or "no active" in out.lower() or "未启用" in out


def test_goal_start_sets_goal_state_and_budget(tmp_path, capsys):
    """/goal <objective> 启动新 goal：agent._goal_state 被设 + token_budget_limit 被设。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/goal 完成报告文档", rt)
    assert handled is True
    # agent._goal_state 被设
    gs = rt.agent._goal_state
    assert gs is not None
    assert gs.objective == "完成报告文档"
    assert gs.status == "active"
    # token_budget_limit 从 config 读
    assert gs.token_budget_limit == 100_000
    # 持久化文件存在
    assert (tmp_path / ".goal" / "current.json").exists()


def test_goal_pause_then_clear(tmp_path, capsys):
    """先 /goal <obj> 启动，再 /goal clear 取消。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _handle_command("/goal 完成测试", rt)
    assert rt.agent._goal_state is not None

    handled = _handle_command("/goal clear", rt)
    assert handled is True
    # clear 后 goal_state 被清掉
    assert rt.agent._goal_state is None


def test_goal_pause_subcommand(tmp_path, capsys):
    """/goal pause 把 goal 状态改 paused。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _handle_command("/goal 跑测试", rt)
    handled = _handle_command("/goal pause", rt)
    assert handled is True
    assert rt.agent._goal_state.status == "paused"


def test_goal_resume_subcommand(tmp_path, capsys):
    """/goal resume 把 paused goal 恢复 active。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _handle_command("/goal 跑测试", rt)
    _handle_command("/goal pause", rt)
    handled = _handle_command("/goal resume", rt)
    assert handled is True
    assert rt.agent._goal_state.status == "active"


def test_goal_pause_old_when_start_new(tmp_path, capsys):
    """启动新 goal 时自动 pause 旧 goal（不丢历史）。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _handle_command("/goal 第一个目标", rt)
    first_gs = rt.agent._goal_state
    _handle_command("/goal 第二个目标", rt)
    # 旧 goal 被 pause
    assert first_gs.status == "paused"
    # 当前是新 goal
    assert rt.agent._goal_state is not first_gs
    assert rt.agent._goal_state.objective == "第二个目标"


# ---------------------------------------------------------------------------
# /poor
# ---------------------------------------------------------------------------

def test_poor_status_default_off(tmp_path, capsys):
    """/poor status 默认 OFF。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/poor status", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "OFF" in out


def test_poor_on_then_off(tmp_path, capsys):
    """/poor on 应用 preset，/poor off 关闭。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/poor on", rt)
    assert handled is True
    assert rt._poor_mode_on is True
    # config 被改（reflection.enabled = False）
    assert rt.config["reflection"]["enabled"] is False

    handled = _handle_command("/poor off", rt)
    assert handled is True
    assert rt._poor_mode_on is False


def test_poor_invalid_arg(tmp_path, capsys):
    """/poor xxx 显示用法。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/poor invalid_arg", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "用法" in out or "on|off|status" in out


# ---------------------------------------------------------------------------
# /trace
# ---------------------------------------------------------------------------

def test_trace_today_empty(tmp_path, capsys):
    """/trace today 无 trace_sink 时返回未启用提示。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    rt.trace_sink = None
    handled = _handle_command("/trace today", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "Trace" in out or "未启用" in out


def test_trace_today_with_sink(tmp_path, capsys):
    """/trace today 有 sink 时输出 summary。"""
    from cli import _handle_command
    from agent.trace import TraceSink
    rt = _FakeRT(tmp_path)
    rt.trace_sink = TraceSink(base_dir=tmp_path)
    rt.trace_sink.emit("pre_llm_call", input_tokens=10)
    handled = _handle_command("/trace today", rt)
    assert handled is True
    out = capsys.readouterr().out
    # summary 输出包含 total_events 字段
    assert "total_events" in out or "pre_llm_call" in out


# ---------------------------------------------------------------------------
# /inbox
# ---------------------------------------------------------------------------

def test_inbox_empty(tmp_path, capsys):
    """/inbox 无 channel 消息时显示"收件箱为空"。"""
    from cli import _handle_command
    from agent.channel_inbox import ChannelInbox
    rt = _FakeRT(tmp_path)
    # 给 agent 挂一个空 channel_inbox
    rt.agent._channel_inbox = ChannelInbox(tmp_path / ".inbox")
    handled = _handle_command("/inbox", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "空" in out or "empty" in out.lower()


def test_inbox_no_channel_inbox(tmp_path, capsys):
    """/inbox 在 channel_inbox 未初始化时给出提示。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/inbox", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "未初始化" in out or "无 MCP" in out


def test_inbox_with_message(tmp_path, capsys):
    """/inbox 有 channel 推送消息时显示。"""
    from cli import _handle_command
    from agent.channel_inbox import ChannelInbox
    rt = _FakeRT(tmp_path)
    inbox = ChannelInbox(tmp_path / ".inbox")
    inbox.push("server_A", {"event": "build_done", "msg": "hello inbox"})
    rt.agent._channel_inbox = inbox
    handled = _handle_command("/inbox", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "server_A" in out or "build_done" in out


# ---------------------------------------------------------------------------
# /mailbox
# ---------------------------------------------------------------------------

def test_mailbox_send_cli(tmp_path, capsys):
    """/mailbox send <to> <content> 通过 CLI 发邮件。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/mailbox send alice hi from cli", rt)
    assert handled is True
    msgs = rt.mailbox.check_unread("alice")
    assert len(msgs) == 1
    assert "hi from cli" in msgs[0]["content"]


def test_mailbox_check_cli(tmp_path, capsys):
    """/mailbox check 列出未读邮件。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    rt.mailbox.send(to="main", from_="x", content="m1")
    handled = _handle_command("/mailbox check", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "m1" in out


def test_mailbox_clear_cli(tmp_path, capsys):
    """/mailbox clear 清空邮箱。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    rt.mailbox.send(to="main", from_="x", content="m1")
    rt.mailbox.send(to="main", from_="x", content="m2")
    handled = _handle_command("/mailbox clear", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "2" in out  # 清掉 2 条
    assert rt.mailbox.check_unread("main") == []


def test_mailbox_no_subcommand_shows_help(tmp_path, capsys):
    """/mailbox 无参数显示用法。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/mailbox", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "send" in out or "用法" in out


# ---------------------------------------------------------------------------
# /resume
# ---------------------------------------------------------------------------

def test_resume_list_no_bundles(tmp_path, capsys):
    """/resume_bundle 无 bundle 时显示空提示。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/resume_bundle", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "无" in out or "0" in out or "empty" in out.lower() or "没有" in out


def test_resume_list_with_bundles(tmp_path, capsys):
    """/resume_bundle 有 bundle 时列出。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    rt.handoff_store.save(
        transcript=[{"role": "user", "content": "hi"}],
        source_session_id=None,
        model={"name": "x", "provider": "x"},
        title="测试 bundle",
    )
    handled = _handle_command("/resume_bundle", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "测试 bundle" in out


def test_resume_load_bundle_into_history(tmp_path, capsys):
    """/resume_bundle <id> 加载 bundle 注入 conversation_history。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    bid = rt.handoff_store.save(
        transcript=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        source_session_id=None,
        model={"name": "x", "provider": "x"},
        title="要加载的",
    )
    handled = _handle_command(f"/resume_bundle {bid}", rt)
    assert handled is True
    # conversation_history 被填充
    assert len(rt.agent.conversation_history) >= 2
    contents = [m.get("content", "") for m in rt.agent.conversation_history]
    assert any("hello" in c for c in contents)
    assert any("world" in c for c in contents)
