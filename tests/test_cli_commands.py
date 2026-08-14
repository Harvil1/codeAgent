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


# ---------------------------------------------------------------------------
# CCAR11 Task 2: /compact + /context
# ---------------------------------------------------------------------------

class _FakeCompactLLM:
    """压缩摘要用 mock LLM client（chat_completions 返回固定摘要）。"""

    def __init__(self, summary: str = "这是 L4 压缩摘要（测试）"):
        self.summary = summary
        self.calls = 0

    async def chat_completions(self, messages, model=None, **kwargs):
        self.calls += 1
        msg = MagicMock()
        msg.content = self.summary
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp


def _make_history(n_msgs: int) -> list:
    """构造 n 条交替 user/assistant 历史（每条 content 有长度）。"""
    hist = []
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        hist.append({"role": role, "content": f"消息 {i} " + "x" * 100})
    return hist


def _setup_compact_agent(rt, n_msgs: int = 6, llm: bool = True):
    """给 _FakeAgent 挂上 /compact 需要的最小属性。"""
    from agent.context_pipeline import CompressionSessionState
    import agent.context_compressor as _cc
    # 重置摘要熔断器全局（防其他测试污染）
    _cc._consecutive_failures = 0
    _cc._compact_circuit_open = False
    rt.agent.conversation_history = _make_history(n_msgs)
    rt.agent.llm_client = _FakeCompactLLM() if llm else None
    rt.agent.model = "deepseek-chat"
    rt.agent._compress_session_state = CompressionSessionState()
    rt.agent._pending_ephemeral_messages = []
    rt.config["context"] = {"llm_compact_keep_recent": 2}


def test_compact_command_runs_llm_compact(tmp_path, capsys):
    """/compact --yes 触发 L4 压缩：history 缩短 + llm_compact_count +1。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _setup_compact_agent(rt, n_msgs=6)
    before = len(rt.agent.conversation_history)
    handled = _handle_command("/compact --yes", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "压缩完成" in out
    # history 被压缩：1 个摘要占位 + keep_recent(2) = 3 条 < 原 6 条
    assert len(rt.agent.conversation_history) < before
    assert rt.agent._compress_session_state.llm_compact_count == 1
    # 打印了前后 token 对比
    assert "tokens" in out.lower()


def test_compact_confirms_before_force(tmp_path, capsys, monkeypatch):
    """/compact 无 --yes 时先确认，拒绝则不压。"""
    import cli as cli_mod
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _setup_compact_agent(rt, n_msgs=6)
    before = list(rt.agent.conversation_history)
    monkeypatch.setattr(cli_mod.console, "input", lambda *a, **k: "n")
    handled = _handle_command("/compact", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "已取消" in out
    # 未压缩
    assert rt.agent.conversation_history == before
    assert rt.agent._compress_session_state.llm_compact_count == 0


def test_compact_confirmed_yes_runs(tmp_path, capsys, monkeypatch):
    """确认输入 y 时执行压缩。"""
    import cli as cli_mod
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _setup_compact_agent(rt, n_msgs=6)
    monkeypatch.setattr(cli_mod.console, "input", lambda *a, **k: "y")
    handled = _handle_command("/compact", rt)
    assert handled is True
    assert rt.agent._compress_session_state.llm_compact_count == 1


def test_compact_fallback_snip_when_no_llm(tmp_path, capsys):
    """LLM client 为 None 时降级 snip_compact（无损裁剪）。"""
    from cli import _handle_command
    rt = _FakeRT(tmp_path)
    _setup_compact_agent(rt, n_msgs=20, llm=False)
    handled = _handle_command("/compact --yes", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "snip" in out.lower()
    assert len(rt.agent.conversation_history) < 20


def test_context_command_renders_table(tmp_path, capsys):
    """/context 渲染 token 分布表。"""
    from cli import _handle_command
    from agent.context_pipeline import CompressionSessionState
    rt = _FakeRT(tmp_path)
    rt.agent.conversation_history = [
        {"role": "user", "content": "你好" * 50},
        {"role": "assistant", "content": "在的" * 50},
        {"role": "user", "content": "继续"},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
    ]
    rt.agent._compress_session_state = CompressionSessionState()
    rt.agent._pending_ephemeral_messages = []
    handled = _handle_command("/context", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "消息数" in out
    assert "tokens" in out.lower()
    assert "current_turn" in out
    assert "llm_compact_count" in out
    assert "reactive_count" in out


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


# ---------------------------------------------------------------------------
# CCAR11 Task 3: /status + /doctor + /diff
# ---------------------------------------------------------------------------

class _FakeMCPClient:
    def __init__(self, connected: bool):
        self.is_connected = connected


class _FakeMCPManager:
    def __init__(self, clients: dict):
        self._clients = clients


def test_status_basic(tmp_path, capsys, monkeypatch):
    """/status 显示模型/aux/goal/项目键/MCP/工具数。"""
    from cli import _handle_command
    import agent.mcp_client as mcp_mod

    rt = _FakeRT(tmp_path)
    rt._statusline_project_key = "proj-abc123"
    gs = GoalState(objective="写一份完整的测试报告文档", iteration_count=3)
    rt.agent._goal_state = gs
    monkeypatch.setattr(
        mcp_mod, "get_mcp_manager",
        lambda: _FakeMCPManager({
            "server_a": _FakeMCPClient(True),
            "server_b": _FakeMCPClient(False),
        }),
    )

    handled = _handle_command("/status", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "deepseek-chat" in out          # 主模型
    assert "aux" in out.lower()            # aux 段
    assert "写一份完整的测试报告文档"[:30] in out  # goal objective
    assert "active" in out                 # goal status
    assert "3" in out                      # iteration
    assert "proj-abc123" in out            # 项目键
    assert "server_a" in out and "server_b" in out  # MCP servers
    assert "工具" in out                   # 工具总数段


def test_status_no_goal_no_mcp(tmp_path, capsys, monkeypatch):
    """/status 无 goal / 无 MCP server / aux 未配置时给出提示而不是报错。"""
    from cli import _handle_command
    import agent.mcp_client as mcp_mod

    rt = _FakeRT(tmp_path)
    monkeypatch.setattr(
        mcp_mod, "get_mcp_manager", lambda: _FakeMCPManager({}),
    )
    handled = _handle_command("/status", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "无" in out  # 无 active goal / 无 MCP server


def test_status_section_failure_fail_open(tmp_path, capsys, monkeypatch):
    """/status 某段数据源抛异常时其余段仍然输出（fail-open）。"""
    from cli import _handle_command
    import agent.mcp_client as mcp_mod

    rt = _FakeRT(tmp_path)
    # _statusline_project_key 是 property 且抛异常 → 该段显示读取失败
    monkeypatch.setattr(
        _FakeRT, "_statusline_project_key",
        property(lambda self: (_ for _ in ()).throw(RuntimeError("boom"))),
        raising=False,
    )
    monkeypatch.setattr(
        mcp_mod, "get_mcp_manager", lambda: _FakeMCPManager({}),
    )
    handled = _handle_command("/status", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "deepseek-chat" in out  # 模型段不受影响


def test_doctor_all_pass(tmp_path, capsys, monkeypatch):
    """/doctor 全部通过时显示 6/6。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    handled = _handle_command("/doctor", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "6/6" in out
    assert "✓" in out


def test_doctor_missing_api_key(tmp_path, capsys, monkeypatch):
    """/doctor API key env 未设置时该项 ✗ 且汇总 < 6/6。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    handled = _handle_command("/doctor", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "✗" in out
    assert "6/6" not in out


def test_doctor_config_broken(tmp_path, capsys, monkeypatch):
    """/doctor load_config 抛异常时第 1 项 ✗（fail-open 不崩）。"""
    from cli import _handle_command
    import config as config_mod

    rt = _FakeRT(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def _boom(*a, **kw):
        raise RuntimeError("bad config")

    monkeypatch.setattr(config_mod, "load_config", _boom)
    handled = _handle_command("/doctor", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "✗" in out
    assert "6/6" not in out


def test_diff_no_checkpoint(tmp_path, capsys):
    """/diff 无 checkpoint manager（或无追踪记录）时提示。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)  # 无 checkpoint_mgr
    handled = _handle_command("/diff", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "无" in out


def test_diff_tracked_files(tmp_path, capsys):
    """/diff 列出 checkpoint 追踪的本会话改动文件。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)

    class _FakeCkpt:
        def tracked_files(self):
            return ["D:/proj/a.py", "D:/proj/b.md"]

        def list_snapshots(self):
            return [{"id": "s1", "files": ["D:/proj/a.py"]}]

    rt.checkpoint_mgr = _FakeCkpt()
    handled = _handle_command("/diff", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "a.py" in out and "b.md" in out
    assert "1" in out  # 快照数


def test_help_contains_three_commands(tmp_path, capsys):
    """/help 帮助文本包含三个新命令。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)
    _handle_command("/help", rt)
    out = capsys.readouterr().out
    assert "/status" in out
    assert "/doctor" in out
    assert "/diff" in out
