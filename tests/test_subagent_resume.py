"""subagent_resume 工具测试。

关键约束：
- subagent_persistence 的函数不接 base_dir 参数（走 _sessions_dir()），
  所以测试用 monkeypatch 替换 _sessions_dir 指向 tmp_path
- transcript 消息格式：{"role": "user|assistant", "content": "..."}
- _spawn_resumed_agent 是接缝（生产实现真跑子代理，测试 patch 掉）
"""
import inspect
import json

import pytest


# ---------------------------------------------------------------------------
# 辅助：把 subagent_persistence 的 _sessions_dir 重定向到 tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """把 subagent_persistence._sessions_dir 指向 tmp_path/.agent-sessions。

    返回 tmp_path，让测试可直接读写验证。
    """
    sessions_dir = tmp_path / ".agent-sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from agent import subagent_persistence as sp
    monkeypatch.setattr(sp, "_sessions_dir", lambda: sessions_dir)
    return tmp_path


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_resume_appends_to_same_transcript(isolated_sessions):
    """resume 跑完后 transcript 续写同一文件。"""
    from agent import subagent_persistence as sp
    agent_id = "sa_test001"
    sp.write_metadata(agent_id, {
        "status": "running",
        "created_at": "2026-08-14T00:00:00",
    })
    sp.append_message(agent_id, {"role": "user", "content": "先干点活"})
    sp.append_message(agent_id, {"role": "assistant", "content": "干了一半"})

    from tools.subagent_resume_tool import _run_resume
    with pytest.MonkeyPatch().context() as mp:
        # patch 掉接缝，避免真跑子代理
        from tools import subagent_resume_tool as mod
        mp.setattr(mod, "_spawn_resumed_agent",
                   lambda messages, instruction, **kw: "续跑完成的结果")
        result = _run_resume(agent_id, "继续")

    data = json.loads(result)
    assert data["result"] == "续跑完成的结果"
    assert data["agent_id"] == agent_id

    # transcript 有新消息（续写：原 2 条 + user 指令 + assistant 结果 = 4 条）
    msgs = sp.load_transcript(agent_id)
    assert len(msgs) >= 4
    # 末尾两条是续写的
    assert msgs[-2]["role"] == "user"
    assert msgs[-2]["content"] == "继续"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "续跑完成的结果"

    # metadata 状态被 mark_completed 更新
    meta = sp.load_metadata(agent_id)
    assert meta["status"] == "completed"


def test_resume_nonexistent_agent_error(isolated_sessions):
    """不存在的 agent_id → 错误 JSON（不崩）。"""
    from tools.subagent_resume_tool import _run_resume
    result = _run_resume("sa_ghost", "继续")
    data = json.loads(result)
    assert "error" in data
    assert data.get("error_type") == "empty_transcript"


def test_resume_empty_transcript_error(isolated_sessions):
    """有 metadata 但无 transcript → empty_transcript。"""
    from agent import subagent_persistence as sp
    sp.write_metadata("sa_empty", {"status": "running"})

    from tools.subagent_resume_tool import _run_resume
    data = json.loads(_run_resume("sa_empty", "继续"))
    assert "error" in data
    assert data.get("error_type") == "empty_transcript"


def test_resume_spawn_failure_returns_error(isolated_sessions):
    """_spawn_resumed_agent 抛异常 → error JSON，不崩。"""
    from agent import subagent_persistence as sp
    agent_id = "sa_fail001"
    sp.write_metadata(agent_id, {"status": "running"})
    sp.append_message(agent_id, {"role": "user", "content": "hi"})
    sp.append_message(agent_id, {"role": "assistant", "content": "yo"})

    from tools.subagent_resume_tool import _run_resume
    with pytest.MonkeyPatch().context() as mp:
        from tools import subagent_resume_tool as mod

        def _boom(messages, instruction, **kw):
            raise RuntimeError("子代理崩了")

        mp.setattr(mod, "_spawn_resumed_agent", _boom)
        data = json.loads(_run_resume(agent_id, "继续"))

    assert "error" in data
    assert data.get("error_type") == "resume_failed"
    assert data["agent_id"] == agent_id


def test_handler_dispatch_contract():
    """签名契约：handler(args, **kwargs)——防 silent-dead-code。"""
    from tools.subagent_resume_tool import _handle_subagent_resume
    sig = inspect.signature(_handle_subagent_resume)
    params = list(sig.parameters.values())
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


def test_handler_missing_agent_id():
    """agent_id 为空 → invalid_args。"""
    from tools.subagent_resume_tool import _handle_subagent_resume
    data = json.loads(_handle_subagent_resume({}))
    assert data["error_type"] == "invalid_args"


def test_handler_passes_instruction_default():
    """handler 未传 instruction 时用默认值。"""
    from tools import subagent_resume_tool as mod

    captured = {}

    def _fake_run(agent_id, instruction, **kw):
        captured["agent_id"] = agent_id
        captured["instruction"] = instruction
        return json.dumps({"result": "ok"})

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(mod, "_run_resume", _fake_run)
        mod._handle_subagent_resume({"agent_id": "sa_abc"})
    assert captured["agent_id"] == "sa_abc"
    assert "继续" in captured["instruction"]  # 默认值


def test_tool_registered_in_core_toolset():
    """工具已注册到 core toolset。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()
    entry = registry.get("subagent_resume")
    assert entry is not None, "subagent_resume 未注册"
    assert entry.isConcurrencySafe is False, "resume 重资源，必须串行"
    # schema 字段
    schema = entry.schema
    assert schema["name"] == "subagent_resume"
    assert "agent_id" in schema["parameters"]["properties"]
    assert "agent_id" in schema["parameters"]["required"]


# ===========================================================================
# /resumable CLI 命令测试
#
# 关键 API 约束（与上方工具测试一致）：
# - subagent_persistence 的 list_resumable/write_metadata **不接 base_dir 参数**
#   （走模块级 _sessions_dir()），所以测试用 monkeypatch 替换 _sessions_dir
#   指向 tmp_path/.agent-sessions
# - _run_resume 是恢复入口，_spawn_resumed_agent 是接缝（测试 patch 隔离）
# ===========================================================================


def test_resumable_cli_list_empty(isolated_sessions, capsys):
    """CLI 列表：无记录时提示"无中断的子代理可恢复"。"""
    from cli import _handle_resumable_command

    class _FakeRt:
        pass

    rt = _FakeRt()
    result = _handle_resumable_command("", rt)
    assert result is True
    captured = capsys.readouterr()
    # 关键：空列表分支走"无可恢复"提示
    assert "无" in captured.out and "可恢复" in captured.out


def test_resumable_cli_list_with_items(isolated_sessions, capsys):
    """CLI 列表：有记录显示 agent_id/status/消息数/时间。"""
    from agent import subagent_persistence as sp

    # 构造 2 条 running 记录（含 transcript 才算"可恢复"语义）
    for aid in ("sa_a001", "sa_a002"):
        sp.write_metadata(aid, {
            "status": "running",
            "agent_type": "explore",
            "created_at": "2026-08-14T00:00:00",
        })
        sp.append_message(aid, {"role": "user", "content": "干点活"})
        sp.append_message(aid, {"role": "assistant", "content": "干完了"})

    from cli import _handle_resumable_command

    class _FakeRt:
        pass

    rt = _FakeRt()
    assert _handle_resumable_command("", rt) is True
    captured = capsys.readouterr()
    # 两条都显示
    assert "sa_a001" in captured.out
    assert "sa_a002" in captured.out
    # 用法提示
    assert "/resumable" in captured.out


def test_resumable_cli_list_filters_out_terminal_status(isolated_sessions, capsys):
    """CLI 列表：status=completed/interrupted 的不显示（list_resumable 只返 running）。"""
    from agent import subagent_persistence as sp

    sp.write_metadata("sa_running", {"status": "running"})
    sp.append_message("sa_running", {"role": "user", "content": "x"})
    sp.write_metadata("sa_done", {"status": "completed", "completed_at": 1.0})
    sp.append_message("sa_done", {"role": "user", "content": "y"})

    from cli import _handle_resumable_command

    class _FakeRt:
        pass

    _handle_resumable_command("", _FakeRt())
    captured = capsys.readouterr()
    assert "sa_running" in captured.out
    assert "sa_done" not in captured.out


def test_resumable_cli_list_notes_semantic_boundary(isolated_sessions, capsys):
    """列表提示注明语义边界：只有存了 transcript 的子代理可恢复。"""
    from agent import subagent_persistence as sp

    sp.write_metadata("sa_x", {"status": "running"})
    sp.append_message("sa_x", {"role": "user", "content": "x"})

    from cli import _handle_resumable_command

    class _FakeRt:
        pass

    _handle_resumable_command("", _FakeRt())
    captured = capsys.readouterr()
    # 关键：语义边界明示——只列 status=running 的，真正中断的可能无 transcript
    assert "running" in captured.out or "status" in captured.out.lower()


def test_resumable_cli_resume_success(isolated_sessions, capsys):
    """CLI /resumable <agent_id>：调 _run_resume 成功，绿色显示结果前 2000 字符。"""
    from agent import subagent_persistence as sp

    agent_id = "sa_resume_ok"
    sp.write_metadata(agent_id, {"status": "running"})
    sp.append_message(agent_id, {"role": "user", "content": "开局"})
    sp.append_message(agent_id, {"role": "assistant", "content": "中场"})

    from cli import _handle_resumable_command
    from tools import subagent_resume_tool as mod

    class _FakeRt:
        agent = object()  # 非 None 即可，_run_resume 会 patch 掉
        config = {"model": {"model": "test"}}

    with pytest.MonkeyPatch().context() as mp:
        # patch 接缝（_spawn_resumed_agent）避免真跑子代理
        mp.setattr(mod, "_spawn_resumed_agent",
                   lambda messages, instruction, **kw: "续命完成的结果")
        result = _handle_resumable_command(agent_id, _FakeRt())

    assert result is True
    captured = capsys.readouterr()
    assert "续命完成的结果" in captured.out
    # 绿色提示（rich tag）
    assert "恢复完成" in captured.out or "完成" in captured.out


def test_resumable_cli_resume_long_result_truncated(isolated_sessions, capsys):
    """结果超 2000 字符只显示前 2000。"""
    from agent import subagent_persistence as sp

    agent_id = "sa_long"
    sp.write_metadata(agent_id, {"status": "running"})
    sp.append_message(agent_id, {"role": "user", "content": "x"})
    sp.append_message(agent_id, {"role": "assistant", "content": "y"})

    from cli import _handle_resumable_command
    from tools import subagent_resume_tool as mod

    long_text = "Z" * 5000

    class _FakeRt:
        agent = object()
        config = {}

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(mod, "_spawn_resumed_agent",
                   lambda messages, instruction, **kw: long_text)
        _handle_resumable_command(agent_id, _FakeRt())

    captured = capsys.readouterr()
    # Rich console 会软换行，把 Z 字符打散到多行；
    # 关键断言：①截断提示出现 ②全文 Z 数远小于 5000
    assert "已截断到 2000 字符" in captured.out
    # 去掉所有空白后数 Z 个数（应等于 2000，不是 5000）
    z_count = captured.out.count("Z")
    assert z_count == 2000, f"应截到 2000，实得 {z_count}"


def test_resumable_cli_resume_error_red_output(isolated_sessions, capsys):
    """_run_resume 返 error JSON → 红色提示。"""
    from agent import subagent_persistence as sp

    # 不写 transcript → _run_resume 返 empty_transcript
    agent_id = "sa_no_transcript"
    sp.write_metadata(agent_id, {"status": "running"})

    from cli import _handle_resumable_command

    class _FakeRt:
        agent = object()
        config = {}

    result = _handle_resumable_command(agent_id, _FakeRt())
    assert result is True  # 命令本身处理了（不崩），error 只是显示
    captured = capsys.readouterr()
    assert "恢复失败" in captured.out or "失败" in captured.out


def test_resumable_cli_dispatched_from_handle_command(isolated_sessions):
    """_handle_command 能分发 /resumable（无参走列表分支，不崩）。"""
    from cli import _handle_command

    class _FakeRt:
        agent = None
        config = {}

    # /resumable 无参 → True
    assert _handle_command("/resumable", _FakeRt()) is True
    # /resumable <id> 也被处理（True，具体结果走 _run_resume）
    assert _handle_command("/resumable sa_ghost_xyz", _FakeRt()) is True


def test_spawn_resumed_agent_passes_memory_store(isolated_sessions):
    """follow-up：_spawn_resumed_agent 接受 memory_store 参数并透传给 AIAgent。

    关键：dispatch_kwargs 的 agent_ref.memory_store 会被取出来传给 spawn，
    spawn 再传给 AIAgent（让续跑子代理复用父记忆）。
    """
    from tools.subagent_resume_tool import _spawn_resumed_agent

    captured = {}

    class _FakeAIAgent:
        def __init__(self, **kw):
            captured.update(kw)
            self.conversation_history = []

        async def chat(self, prompt):
            return "ok"

    import asyncio

    class _FakeAgentRef:
        spawn_depth = 0
        memory_store = "FAKE_MEM_STORE"

    # patch AIAgent 构造，避免真造
    import agent as agent_mod
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(agent_mod, "AIAgent", _FakeAIAgent)
        # LLM 配置走 config fallback（避免 RuntimeError）
        mp.setattr("config.load_config",
                   lambda: {"model": {"model": "test",
                                      "base_url": "http://x",
                                      "api_key": "k"}})

        _spawn_resumed_agent(
            [{"role": "user", "content": "hi"}],
            "继续",
            agent_ref=_FakeAgentRef(),
            memory_store="FAKE_MEM_STORE",
        )

    # 关键断言：memory_store 透传到了 AIAgent
    assert captured.get("memory_store") == "FAKE_MEM_STORE"


def test_run_resume_passes_memory_store_from_agent_ref(isolated_sessions):
    """follow-up：_run_resume 从 dispatch_kwargs.agent_ref 取 memory_store 传给 spawn。

    场景：工具 handler 收到 agent_ref（AIAgent 实例），_run_resume 应从中取
    memory_store 并透传给 _spawn_resumed_agent（让续跑子代理复用父记忆库）。
    """
    from agent import subagent_persistence as sp

    agent_id = "sa_mem_test"
    sp.write_metadata(agent_id, {"status": "running"})
    sp.append_message(agent_id, {"role": "user", "content": "x"})
    sp.append_message(agent_id, {"role": "assistant", "content": "y"})

    from tools.subagent_resume_tool import _run_resume
    from tools import subagent_resume_tool as mod

    captured = {}

    def _fake_spawn(messages, instruction, **kw):
        captured.update(kw)
        return "ok"

    class _FakeAgentRef:
        spawn_depth = 0
        memory_store = "PARENT_MEM_STORE"

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(mod, "_spawn_resumed_agent", _fake_spawn)
        _run_resume(agent_id, "继续", agent_ref=_FakeAgentRef())

    # 关键断言：_run_resume 把 agent_ref.memory_store 取出透传
    assert captured.get("memory_store") == "PARENT_MEM_STORE"
