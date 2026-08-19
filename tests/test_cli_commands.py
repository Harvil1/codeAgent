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

def test_mailbox_tools_in_core_tools():
    """mailbox_* 三工具必须进 _CORE_TOOLS 才对 LLM 可见（发现 ≠ 可见）。

    历史 bug：注册 toolset="team" 但 TOOLSETS["team"]["tools"] 未列，
    导致只有 CLI /mailbox 可用，LLM 调不到（silent-dead-code）。
    """
    from toolsets import _CORE_TOOLS
    for name in ("mailbox_send", "mailbox_check", "mailbox_clear"):
        assert name in _CORE_TOOLS, f"{name} 不在 _CORE_TOOLS，LLM 不可见"


def test_mailbox_send_async_disallowed():
    """mailbox_send 对 async 子代理禁用（与 team_send 同理：影响其他 agent）。

    mailbox_check/clear 只动自己的邮箱，不进黑名单。
    """
    from toolsets import ASYNC_AGENT_DISALLOWED_TOOLS
    assert "mailbox_send" in ASYNC_AGENT_DISALLOWED_TOOLS
    assert "mailbox_check" not in ASYNC_AGENT_DISALLOWED_TOOLS
    assert "mailbox_clear" not in ASYNC_AGENT_DISALLOWED_TOOLS


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


# ---------------------------------------------------------------------------
# CCAR11 Task 4: /add-dir 运行时白名单 + 持久化
# ---------------------------------------------------------------------------

def _clear_extra_roots():
    """清空运行时额外白名单（防其他测试污染）。"""
    from agent.permission import clear_extra_allowed_roots
    clear_extra_allowed_roots()


def test_add_dir_no_args_lists_whitelist(tmp_path, capsys):
    """/add-dir 无参数列出当前 safe_path 写白名单。"""
    from cli import _handle_command
    _clear_extra_roots()
    rt = _FakeRT(tmp_path)
    handled = _handle_command("/add-dir", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "白名单" in out
    # 默认白名单包含 cwd
    assert str(Path.cwd().resolve()) in out


def test_add_dir_rejects_missing_dir(tmp_path, capsys):
    """/add-dir 指向不存在的目录时拒绝并提示。"""
    from cli import _handle_command
    _clear_extra_roots()
    rt = _FakeRT(tmp_path)
    handled = _handle_command(f"/add-dir {tmp_path / 'no_such_dir'}", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "不存在" in out


def test_add_dir_runtime_and_persist(tmp_path, capsys, monkeypatch):
    """/add-dir 添加后 safe_path 放行 + 路径出现在 settings.json 里。"""
    from cli import _handle_command
    import constants
    from agent.permission import safe_path, clear_extra_allowed_roots

    clear_extra_allowed_roots()
    home = tmp_path / "omnimate_home"
    home.mkdir()
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: home)

    target = tmp_path / "extra_root"
    target.mkdir()
    probe = target / "x.txt"

    # 添加前：白名单外写入被拒
    assert safe_path(probe, write=True).allowed is False

    rt = _FakeRT(tmp_path)
    handled = _handle_command(f"/add-dir {target}", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "已添加" in out

    # 运行时生效：safe_path 放行（Task 4 核心断言）
    assert safe_path(probe, write=True).allowed is True

    # 持久化：settings.json 的 security.extra_allowed_roots 出现该路径
    settings_file = home / "settings.json"
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert str(target.resolve()) in data["security"]["extra_allowed_roots"]


def test_add_dir_idempotent(tmp_path, capsys, monkeypatch):
    """重复 /add-dir 同一目录幂等：运行时不重复 + settings.json 只存一份。"""
    from cli import _handle_command
    import constants
    from agent.permission import (
        clear_extra_allowed_roots, list_extra_allowed_roots,
    )

    clear_extra_allowed_roots()
    home = tmp_path / "omnimate_home"
    home.mkdir()
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: home)
    target = tmp_path / "dup_root"
    target.mkdir()

    rt = _FakeRT(tmp_path)
    _handle_command(f"/add-dir {target}", rt)
    capsys.readouterr()
    handled = _handle_command(f"/add-dir {target}", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "已在白名单" in out  # 第二次提示幂等跳过

    # 运行时只挂一份
    resolved = target.resolve()
    assert list_extra_allowed_roots().count(resolved) == 1
    # settings.json 里也只存一份
    data = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    roots = data["security"]["extra_allowed_roots"]
    assert roots.count(str(resolved)) == 1


def test_persist_extra_root_preserves_other_keys(tmp_path, monkeypatch):
    """持久化读-改-写不破坏 settings.json 已有的其他字段。"""
    import constants
    import cli as cli_mod

    home = tmp_path / "omnimate_home"
    home.mkdir()
    # 预置用户已有配置（非默认值，验证不被覆盖）
    (home / "settings.json").write_text(json.dumps({
        "security": {"command_approval": "never"},
        "display": {"show_tool_progress": False},
        "enabled_toolsets": ["core"],
    }), encoding="utf-8")
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: home)
    target = tmp_path / "keep_root"
    target.mkdir()

    assert cli_mod._persist_extra_root(str(target)) is True
    data = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    assert data["security"]["command_approval"] == "never"
    assert data["display"]["show_tool_progress"] is False
    assert data["enabled_toolsets"] == ["core"]
    assert str(target) in data["security"]["extra_allowed_roots"]
    # 幂等：重复持久化返回 False 且不重复写
    assert cli_mod._persist_extra_root(str(target)) is False
    data2 = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    assert data2["security"]["extra_allowed_roots"].count(str(target)) == 1


def test_add_dir_end_to_end_via_load_config(tmp_path, monkeypatch):
    """端到端闭环：_persist_extra_root 写 settings.json → load_config 真实函数
    读回该路径 → _load_persisted_extra_roots 灌回运行时白名单。

    防"写错轨"回归：此前写 config.yaml，而 load_config 默认只读
    settings.json（首次启动还会把 config.yaml 迁走改名 .bak），
    真实部署灌回 0 条。
    """
    import constants
    from config import load_config
    import cli as cli_mod
    from agent.permission import (
        clear_extra_allowed_roots, list_extra_allowed_roots, safe_path,
    )

    home = tmp_path / "omnimate_home"
    home.mkdir()
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: home)

    clear_extra_allowed_roots()
    target = tmp_path / "e2e_root"
    target.mkdir()

    # 1) /add-dir 持久化写入 settings.json
    assert cli_mod._persist_extra_root(str(target)) is True

    # 2) 用 load_config 真实函数（默认路径 = settings.json）读回该路径
    cfg = load_config()
    assert str(target) in cfg["security"]["extra_allowed_roots"]

    # 3) 启动灌回：RuntimeContext 启动路径 → 运行时白名单生效
    n = cli_mod._load_persisted_extra_roots(cfg)
    assert n == 1
    assert any(r == target.resolve() for r in list_extra_allowed_roots())
    assert safe_path(target / "z.txt", write=True).allowed is True
    clear_extra_allowed_roots()


def test_load_persisted_extra_roots_at_startup(tmp_path):
    """启动加载：config security.extra_allowed_roots → 运行时 safe_path 白名单。"""
    import cli as cli_mod
    from agent.permission import (
        clear_extra_allowed_roots, list_extra_allowed_roots, safe_path,
    )

    clear_extra_allowed_roots()
    target = tmp_path / "boot_root"
    target.mkdir()

    n = cli_mod._load_persisted_extra_roots(
        {"security": {"extra_allowed_roots": [str(target)]}}
    )
    assert n == 1
    assert any(r == target.resolve() for r in list_extra_allowed_roots())
    assert safe_path(target / "y.txt", write=True).allowed is True
    clear_extra_allowed_roots()


def test_help_contains_add_dir(tmp_path, capsys):
    """/help 帮助文本包含 /add-dir。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)
    _handle_command("/help", rt)
    out = capsys.readouterr().out
    assert "/add-dir" in out


# ---------------------------------------------------------------------------
# CCAR11 Task 5: /paste 剪贴板图片
# ---------------------------------------------------------------------------

def _fake_ps_run_factory(returncode: int, create_file: bool = True, exc: Exception = None):
    """构造假的 subprocess.run（模拟 PowerShell 剪贴板读取）。

    - 从 PowerShell 命令文本里抠出 $img.Save('<path>') 的输出路径
    - create_file=True 时预创建该文件（模拟保存成功）
    - exc 非 None 时直接抛（模拟 PowerShell 不存在/超时等）
    """
    saved_paths = []

    def _fake_run(cmd, **kwargs):
        if exc is not None:
            raise exc
        ps_text = cmd[-1]
        out_path = Path(ps_text.split("$img.Save('")[1].split("')")[0])
        if create_file:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\x89PNG-fake")
        saved_paths.append(out_path)
        r = MagicMock()
        r.returncode = returncode
        return r

    _fake_run.saved_paths = saved_paths
    return _fake_run


def test_paste_saves_clipboard_image(tmp_path, capsys, monkeypatch):
    """/paste 剪贴板有图片时保存到 <workspace>/.paste/img_*.png 并打印路径。"""
    import cli as cli_mod
    from cli import _handle_command
    from agent.workspace_context import workspace_cwd_context

    fake_run = _fake_ps_run_factory(returncode=0, create_file=True)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.sys, "platform", "win32")

    rt = _FakeRT(tmp_path)
    with workspace_cwd_context(str(tmp_path)):
        handled = _handle_command("/paste", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "已保存" in out
    # 提示里包含保存路径 + image_analyze 引导
    assert ".paste" in out
    assert "image_analyze" in out
    # 文件落在 workspace cwd 的 .paste/ 下（不是进程 cwd）
    assert len(fake_run.saved_paths) == 1
    saved = fake_run.saved_paths[0]
    assert saved.parent == tmp_path / ".paste"
    assert saved.name.startswith("img_")
    assert saved.suffix == ".png"
    assert saved.exists()


def test_paste_no_image_in_clipboard(tmp_path, capsys, monkeypatch):
    """PowerShell 退出码 2（剪贴板无图片）→ 提示而非报错。"""
    import cli as cli_mod
    from cli import _handle_command
    from agent.workspace_context import workspace_cwd_context

    fake_run = _fake_ps_run_factory(returncode=2, create_file=False)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.sys, "platform", "win32")

    rt = _FakeRT(tmp_path)
    with workspace_cwd_context(str(tmp_path)):
        handled = _handle_command("/paste", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "剪贴板中没有图片" in out


def test_paste_fail_open_on_error(tmp_path, capsys, monkeypatch):
    """PowerShell 抛异常（超时/不存在）→ fail-open 提示手动给路径，不崩。"""
    import cli as cli_mod
    from cli import _handle_command
    from agent.workspace_context import workspace_cwd_context

    fake_run = _fake_ps_run_factory(
        returncode=1, create_file=False,
        exc=RuntimeError("powershell not found"),
    )
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.sys, "platform", "win32")

    rt = _FakeRT(tmp_path)
    with workspace_cwd_context(str(tmp_path)):
        handled = _handle_command("/paste", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "粘贴失败" in out
    assert "手动" in out  # 引导手动保存后给路径


def test_paste_fail_open_on_returncode(tmp_path, capsys, monkeypatch):
    """PowerShell 退出码非 0/2 且文件未生成 → 同样 fail-open 提示。"""
    import cli as cli_mod
    from cli import _handle_command
    from agent.workspace_context import workspace_cwd_context

    fake_run = _fake_ps_run_factory(returncode=1, create_file=False)
    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_mod.sys, "platform", "win32")

    rt = _FakeRT(tmp_path)
    with workspace_cwd_context(str(tmp_path)):
        handled = _handle_command("/paste", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "粘贴失败" in out


def test_paste_non_windows_fail_open(tmp_path, capsys, monkeypatch):
    """非 Windows 平台直接走 fail-open 提示（不调用 PowerShell）。"""
    import cli as cli_mod
    from cli import _handle_command
    from agent.workspace_context import workspace_cwd_context

    called = []

    def _spy_run(cmd, **kwargs):
        called.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr(cli_mod.subprocess, "run", _spy_run)
    monkeypatch.setattr(cli_mod.sys, "platform", "linux")

    rt = _FakeRT(tmp_path)
    with workspace_cwd_context(str(tmp_path)):
        handled = _handle_command("/paste", rt)
    assert handled is True
    out = capsys.readouterr().out
    assert "粘贴失败" in out or "Windows" in out
    assert called == []  # 非 Windows 不启动 PowerShell 子进程


def test_help_contains_paste(tmp_path, capsys):
    """/help 帮助文本包含 /paste。"""
    from cli import _handle_command

    rt = _FakeRT(tmp_path)
    _handle_command("/help", rt)
    out = capsys.readouterr().out
    assert "/paste" in out


# ---------------------------------------------------------------------------
# T5（核心机制对齐第 5 项）：审批"总是允许"档 + /approved remove-root
# ---------------------------------------------------------------------------

def test_approval_callback_path_always(tmp_path, monkeypatch, capsys):
    """CLI 审批回调：路径分支输入 a → 返回 "always"（持久化由 checker 统一处理）。"""
    import cli as cli_mod

    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    cb = cli_mod._make_approval_callback()

    monkeypatch.setattr(
        cli_mod.console, "input", lambda *a, **kw: "a",
    )
    result = cb(f"文件写入审批: {outside / 'x.txt'}")
    assert result == "always"

    # y 仍是"本次允许"（True）
    monkeypatch.setattr(cli_mod.console, "input", lambda *a, **kw: "y")
    assert cb(f"文件写入审批: {outside / 'y.txt'}") is True
    # 其他输入拒绝
    monkeypatch.setattr(cli_mod.console, "input", lambda *a, **kw: "")
    assert cb(f"文件写入审批: {outside / 'n.txt'}") is False


def test_manage_whitelist_lists_and_removes_roots(tmp_path, monkeypatch, capsys):
    """/approved 列出持久化写入根目录；/approved remove-root <n> 移除（settings + 运行时）。"""
    from agent.settings import load_settings, save_settings
    from agent.permission import add_extra_allowed_root, clear_extra_allowed_roots
    import cli as cli_mod

    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    root_a = tmp_path / "dirA"
    root_a.mkdir()
    root_b = tmp_path / "dirB"
    root_b.mkdir()

    data = load_settings()
    data.setdefault("security", {})["extra_allowed_roots"] = [str(root_a), str(root_b)]
    save_settings(data)
    clear_extra_allowed_roots()
    add_extra_allowed_root(root_a)
    add_extra_allowed_root(root_b)

    rt = _FakeRT(tmp_path)
    # 列出（含写入根目录节）
    cli_mod._manage_whitelist(rt, "")
    out = capsys.readouterr().out
    assert "dirA" in out and "dirB" in out

    # remove-root 0 → 移除 root_a（settings.json + 运行时）
    cli_mod._manage_whitelist(rt, "remove-root 0")
    data2 = load_settings()
    roots = data2.get("security", {}).get("extra_allowed_roots", [])
    assert str(root_a) not in roots
    assert str(root_b) in roots
    from agent.permission import list_extra_allowed_roots
    assert root_a not in list_extra_allowed_roots()
    assert root_b in list_extra_allowed_roots()


# ---------------------------------------------------------------------------
# R30 审计 High-3：Ctrl+C 哨兵去重窗口
# ---------------------------------------------------------------------------

def test_interrupt_sentinel_dedup_window():
    """输入线程的 Ctrl+C 哨兵与主线程回合内中断去重。

    同一次 Ctrl+C 可能同时被两个消费者收到：主线程（KeyboardInterrupt →
    agent.interrupt()，记录时间戳）与输入线程（console.input 抛
    KeyboardInterrupt → _INTERRUPT_SENTINEL 入队）。窗口内到达的哨兵视为
    同一次按键的重复消费（不退出 REPL）；空闲提示符下的 Ctrl+C（无近期
    回合内中断）保持原退出语义。
    """
    from cli import _should_exit_on_interrupt_sentinel

    now = 1000.0
    # 空闲提示符下 Ctrl+C（近期无回合内中断）→ 退出
    assert _should_exit_on_interrupt_sentinel(0.0, now) is True
    # 0.3s 前刚发生过回合内中断 → 同一次按键的重复消费 → 不退出
    assert _should_exit_on_interrupt_sentinel(now - 0.3, now) is False
    # 超过窗口（1.5s 前的中断）→ 视为新的独立 Ctrl+C → 退出
    assert _should_exit_on_interrupt_sentinel(now - 1.5, now) is True


# ---------------------------------------------------------------------------
# R30 审计 Medium-8：审批回调的路径判定启发式
# ---------------------------------------------------------------------------

def test_is_path_item_heuristic():
    """命令/路径判定：多 token 命令不再被误判为路径审批。

    旧启发式 `"/" in item ...` 把 `del /s /q tmp`、`rm -rf build/` 这类
    含斜杠的**命令**判成路径——提示语变成"即将写入路径"且出现
    "a=总是允许并记住"选项（走的是命令分支，语义完全错位）；
    另有三元优先级问题（len<3 时整体 False，"~" 单字符漏判）。
    新规则：多 token 一律命令；~ 开头 / 盘符 / 单 token 含分隔符才算路径。
    """
    from cli import _is_path_item

    # 命令形态（含 flag、含路径参数——都不是"路径审批"）
    assert _is_path_item("del /s /q tmp") is False
    assert _is_path_item("rm -rf build/") is False
    assert _is_path_item("git reset --hard src/") is False
    assert _is_path_item("rmdir /s /q build") is False
    assert _is_path_item("git") is False
    # 路径形态
    assert _is_path_item("文件写入审批: D:" + chr(92) + "x") is True
    assert _is_path_item("D:" + chr(92) + "project" + chr(92) + "x") is True
    assert _is_path_item("C:/Users/a") is True
    assert _is_path_item("~/.ssh/config") is True
    assert _is_path_item("~") is True
    assert _is_path_item("/tmp/data") is True
    assert _is_path_item("src/lib") is True
    assert _is_path_item("build" + chr(92) + "sub") is True
    assert _is_path_item("") is False


# ---------------------------------------------------------------------------
# R30 审计 Medium-7：/new 会话级状态清理
# ---------------------------------------------------------------------------

def test_new_session_resets_session_scoped_state(tmp_path, monkeypatch):
    """/new 后会话级状态不得跨会话泄漏。

    旧行为只做三件事（新 session_id + 清 history + invalidate prompt）：
    - checkpoint_mgr 仍指向旧会话 → 此后快照全写进旧目录，/rewind 回滚错对象
      （对照 resume_session 会重建——同文件内不对称，漏项）
    - 压缩状态（cooldown/熔断计数）、auto_extract 游标（history 已清但游标
      仍指旧长度）、记忆注入去重、context_tip、中断残留等全部继承
    """
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")
    from cli import RuntimeContext

    rt = RuntimeContext()
    rt.initialize()
    old_session = rt.session_id
    old_ckpt = rt.checkpoint_mgr

    # 污染一批会话级状态（模拟旧会话积累）
    rt.agent._context_tip_shown = True
    rt.agent._auto_extract_cursor = 7
    rt.agent._auto_extract_turn_count = 9
    rt.agent._interrupt_requested = True
    rt.agent._pending_ephemeral_messages = [{"role": "user", "content": "旧"}]
    rt.agent._pending_tool_batch_summary = "旧摘要"
    rt.agent._queued_cli_commands = ["/old"]
    rt.agent._compress_session_state.llm_compact_failures = 3
    rt.agent._compress_session_state.last_llm_compact_turn = 42
    rt.agent._surfaced_memory_ids = {"old-mem"}
    rt.agent.conversation_history = [{"role": "user", "content": "旧对话"}]

    rt.new_session()

    assert rt.session_id != old_session
    assert rt.agent.session_id == rt.session_id
    # checkpoint 绑定新会话
    assert rt.checkpoint_mgr is not old_ckpt
    assert rt.checkpoint_mgr._session_id == rt.session_id
    # 会话级状态全部重置
    assert rt.agent.conversation_history == []
    assert rt.agent._context_tip_shown is False
    assert rt.agent._auto_extract_cursor == 0
    assert rt.agent._auto_extract_turn_count == 0
    assert rt.agent._interrupt_requested is False
    assert rt.agent._pending_ephemeral_messages == []
    assert rt.agent._pending_tool_batch_summary is None
    assert rt.agent._queued_cli_commands == []
    assert rt.agent._surfaced_memory_ids == set()
    assert rt.agent._compress_session_state.llm_compact_failures == 0
    assert rt.agent._compress_session_state.last_llm_compact_turn == -10**6
