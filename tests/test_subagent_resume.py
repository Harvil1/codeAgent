"""subagent_resume 工具测试（CCAR10 Task 4，补 CCAR5-I Phase 2）。

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
    """签名契约：handler(args, **kwargs)（CCAR8 教训防 silent-dead-code）。"""
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
    assert "agent_id" in schema["inputSchema"]["properties"]
    assert "agent_id" in schema["inputSchema"]["required"]
