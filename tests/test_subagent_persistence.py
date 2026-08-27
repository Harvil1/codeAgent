"""Task I: 子代理 sidechain transcript 持久化测试。

测试维度：
1. agent_id 格式：generate_agent_id 返回 sub-xxx-timestamp-rand8
2. metadata write/read round-trip
3. append/load transcript
4. list_resumable：只返回 status=running
5. cleanup_old：N 天前的已完成记录被清理
6. mark_completed：状态从 running 转 completed
7. fail-open：IO 异常不让主流程崩
8. cleanup_stale_subagents：所有 running 变 interrupted
9. 端到端：通过 _run_child 起 mock 子代理，验证 transcript 落盘 + 状态变化
"""

import json
import time
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_sessions_dir(tmp_path, monkeypatch):
    """隔离 .agent-sessions 目录到 tmp_path，避免污染真实文件系统。"""
    fake_home = tmp_path / "OmniMate"
    fake_home.mkdir()
    monkeypatch.setenv("OMNIMATE_HOME", str(fake_home))
    # 清理已 import 的 constants 缓存（get_omnimate_home 读 env）
    import importlib
    import constants
    importlib.reload(constants)
    # 重新 import 目标模块，使其 _sessions_dir 拿到新的 home
    import agent.subagent_persistence as sp
    importlib.reload(sp)
    return fake_home


# ---------------------------------------------------------------------------
# generate_agent_id
# ---------------------------------------------------------------------------

class TestGenerateAgentId:
    def test_format_sub_prefix(self, isolated_sessions_dir):
        from agent.subagent_persistence import generate_agent_id
        aid = generate_agent_id(parent_session_id="abcdef1234567890")
        assert aid.startswith("sub-"), f"应以 sub- 开头: {aid}"

    def test_format_contains_parent_id_prefix(self, isolated_sessions_dir):
        from agent.subagent_persistence import generate_agent_id
        aid = generate_agent_id(parent_session_id="abcdef1234567890")
        # sub-{parent_session_id 前 8 位}-...
        assert "abcdef12" in aid, f"应含 parent_session_id 前 8 位: {aid}"

    def test_format_orphan_when_no_parent(self, isolated_sessions_dir):
        from agent.subagent_persistence import generate_agent_id
        aid = generate_agent_id()
        assert "orphan" in aid, f"无 parent 时应含 orphan: {aid}"

    def test_format_random_suffix_8chars(self, isolated_sessions_dir):
        from agent.subagent_persistence import generate_agent_id
        aid = generate_agent_id(parent_session_id="test1234")
        # sub-test1234-YYYYMMDD-HHMMSS-XXXXXXXX
        parts = aid.split("-")
        assert len(parts) >= 4, f"应至少 4 段: {aid}"
        last = parts[-1]
        assert len(last) == 8, f"随机后缀应 8 字符: {last}"

    def test_unique(self, isolated_sessions_dir):
        from agent.subagent_persistence import generate_agent_id
        ids = {generate_agent_id() for _ in range(20)}
        assert len(ids) == 20, "连续调 20 次应都不同"


# ---------------------------------------------------------------------------
# metadata write/read
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_write_load_roundtrip(self, isolated_sessions_dir):
        from agent.subagent_persistence import write_metadata, load_metadata, generate_agent_id
        aid = generate_agent_id(parent_session_id="test")
        meta = {
            "agent_type": "general-purpose",
            "parent_session_id": "test",
            "status": "running",
            "created_at": time.time(),
        }
        write_metadata(aid, meta)
        loaded = load_metadata(aid)
        assert loaded is not None
        assert loaded["agent_type"] == "general-purpose"
        assert loaded["status"] == "running"
        assert loaded["agent_id"] == aid
        assert "updated_at" in loaded

    def test_load_nonexistent_returns_none(self, isolated_sessions_dir):
        from agent.subagent_persistence import load_metadata
        assert load_metadata("nonexistent-id") is None

    def test_write_metadata_fail_open(self, isolated_sessions_dir, monkeypatch):
        """write_metadata 异常不抛。"""
        from agent.subagent_persistence import write_metadata, generate_agent_id
        aid = generate_agent_id()

        def boom(*a, **kw):
            raise OSError("disk full")
        monkeypatch.setattr("pathlib.Path.write_text", boom)
        # 不抛
        write_metadata(aid, {"status": "running"})


# ---------------------------------------------------------------------------
# append/load transcript
# ---------------------------------------------------------------------------

class TestTranscript:
    def test_append_and_load(self, isolated_sessions_dir):
        from agent.subagent_persistence import append_message, load_transcript, generate_agent_id
        aid = generate_agent_id()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "bye"},
        ]
        for m in msgs:
            append_message(aid, m)
        loaded = load_transcript(aid)
        assert len(loaded) == 3
        assert loaded[0]["content"] == "hello"
        assert loaded[2]["content"] == "bye"

    def test_load_nonexistent_returns_empty(self, isolated_sessions_dir):
        from agent.subagent_persistence import load_transcript
        assert load_transcript("nonexistent") == []

    def test_append_fail_open(self, isolated_sessions_dir, monkeypatch):
        from agent.subagent_persistence import append_message, generate_agent_id
        aid = generate_agent_id()

        def boom(*a, **kw):
            raise OSError("no space")
        monkeypatch.setattr("builtins.open", boom)
        # 不抛
        append_message(aid, {"role": "user", "content": "test"})

    def test_load_fail_open(self, isolated_sessions_dir, monkeypatch):
        from agent.subagent_persistence import load_transcript, write_metadata, generate_agent_id
        aid = generate_agent_id()
        # 先正常写一条
        write_metadata(aid, {"status": "running"})
        # 写一条 transcript
        from agent.subagent_persistence import append_message
        append_message(aid, {"role": "user", "content": "test"})

        # 破坏 load（json parse 出错 → 返回空列表）
        def boom(*a, **kw):
            raise ValueError("corrupted")
        monkeypatch.setattr("json.loads", boom)
        result = load_transcript(aid)
        assert result == []


# ---------------------------------------------------------------------------
# list_resumable
# ---------------------------------------------------------------------------

class TestListResumable:
    def test_only_running_returned(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, list_resumable, mark_completed,
        )
        # 3 running + 1 completed
        for i in range(3):
            aid = generate_agent_id()
            write_metadata(aid, {"status": "running", "agent_type": "test"})
        completed_id = generate_agent_id()
        write_metadata(completed_id, {"status": "running"})
        mark_completed(completed_id, "completed")

        resumable = list_resumable()
        assert len(resumable) == 3
        for m in resumable:
            assert m["status"] == "running"

    def test_empty_when_none_running(self, isolated_sessions_dir):
        from agent.subagent_persistence import list_resumable
        assert list_resumable() == []


# ---------------------------------------------------------------------------
# mark_completed
# ---------------------------------------------------------------------------

class TestMarkCompleted:
    def test_running_to_completed(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, mark_completed, load_metadata,
        )
        aid = generate_agent_id()
        write_metadata(aid, {"status": "running"})
        mark_completed(aid, "completed")
        loaded = load_metadata(aid)
        assert loaded["status"] == "completed"
        assert "completed_at" in loaded

    def test_running_to_failed(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, mark_completed, load_metadata,
        )
        aid = generate_agent_id()
        write_metadata(aid, {"status": "running"})
        mark_completed(aid, "failed")
        loaded = load_metadata(aid)
        assert loaded["status"] == "failed"

    def test_running_to_interrupted(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, mark_completed, load_metadata,
        )
        aid = generate_agent_id()
        write_metadata(aid, {"status": "running"})
        mark_completed(aid, "interrupted")
        loaded = load_metadata(aid)
        assert loaded["status"] == "interrupted"


# ---------------------------------------------------------------------------
# cleanup_old
# ---------------------------------------------------------------------------

class TestCleanupOld:
    def test_old_completed_cleaned(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, append_message, generate_agent_id, mark_completed,
            cleanup_old,
        )
        # 10 天前完成的
        old_time = time.time() - 10 * 86400
        aid_old = generate_agent_id()
        write_metadata(aid_old, {"status": "running", "created_at": old_time})
        mark_completed(aid_old, "completed")
        # 手动改 completed_at 到 10 天前
        from agent.subagent_persistence import load_metadata
        meta = load_metadata(aid_old)
        meta["completed_at"] = old_time
        write_metadata(aid_old, meta)
        append_message(aid_old, {"role": "user", "content": "old"})

        # 今天的（不应该被清理）
        aid_new = generate_agent_id()
        write_metadata(aid_new, {"status": "running"})
        mark_completed(aid_new, "completed")

        # 验证清理前 transcript 确实存在（保证测试有意义）
        from agent.subagent_persistence import load_transcript
        assert len(load_transcript(aid_old)) == 1, "清理前应有 1 条 transcript"

        cleaned = cleanup_old(days=7)
        assert cleaned == 1

        # 验证旧 meta 被删
        from agent.subagent_persistence import load_metadata
        assert load_metadata(aid_old) is None
        assert load_metadata(aid_new) is not None

        # 验证旧 transcript (jsonl) 也被删（不只 meta）
        assert load_transcript(aid_old) == [], "jsonl transcript 应一并删除"

    def test_running_not_cleaned(self, isolated_sessions_dir):
        """即使很老的 running 也不被 cleanup_old 删（只清终态）。"""
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, cleanup_old, load_metadata,
        )
        old_time = time.time() - 30 * 86400
        aid = generate_agent_id()
        write_metadata(aid, {
            "status": "running",
            "created_at": old_time,
            "completed_at": old_time,  # 即使 completed_at 旧
        })
        cleaned = cleanup_old(days=7)
        assert cleaned == 0
        assert load_metadata(aid) is not None


# ---------------------------------------------------------------------------
# cleanup_stale_subagents
# ---------------------------------------------------------------------------

class TestCleanupStaleSubagents:
    def test_all_running_become_interrupted(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, list_resumable,
        )
        from agent.subagent_persistence import cleanup_stale_subagents

        # 2 running 记录
        for i in range(2):
            aid = generate_agent_id()
            write_metadata(aid, {"status": "running"})

        # 启动时清理
        cleaned = cleanup_stale_subagents()
        assert cleaned == 2

        # 所有 running 变 interrupted
        remaining = list_resumable()
        assert len(remaining) == 0  # running 的没有了

    def test_completed_not_affected(self, isolated_sessions_dir):
        from agent.subagent_persistence import (
            write_metadata, generate_agent_id, mark_completed, cleanup_stale_subagents,
            load_metadata,
        )
        aid_running = generate_agent_id()
        write_metadata(aid_running, {"status": "running"})

        aid_completed = generate_agent_id()
        write_metadata(aid_completed, {"status": "running"})
        mark_completed(aid_completed, "completed")

        cleaned = cleanup_stale_subagents()
        assert cleaned == 1

        # completed 的状态不变
        assert load_metadata(aid_completed)["status"] == "completed"


# ---------------------------------------------------------------------------
# 端到端：通过 _run_child 验证落盘
# ---------------------------------------------------------------------------

class TestEndToEndRunChild:
    """端到端：_run_child 起 mock 子代理，验证 transcript 落盘 + 状态变化。"""

    def test_transcript_persisted_on_success(self, isolated_sessions_dir, monkeypatch):
        """_run_child 成功完成时，transcript 落盘 + status=completed。"""
        import agent.subagent_persistence as sp

        # 记录生成的 agent_id
        generated_ids = []
        orig_gen = sp.generate_agent_id

        def capture_gen(parent_session_id=""):
            aid = orig_gen(parent_session_id)
            generated_ids.append(aid)
            return aid
        monkeypatch.setattr(sp, "generate_agent_id", capture_gen)

        # Mock AIAgent
        mock_child = MagicMock()
        mock_child.llm_client = MagicMock()
        mock_child.model = "fake"

        captured_on_response = []

        async def chat_side_effect(msg):
            return "子代理执行成功"

        mock_child.chat.side_effect = chat_side_effect

        with patch("agent.AIAgent", return_value=mock_child) as mock_ctor:
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}):
                with patch("config.load_config", return_value={
                    "model": {
                        "name": "fake-model",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "http://fake",
                    }
                }):
                    from tools.delegate_tool import _run_child
                    result = _run_child(
                        "test goal", "", "leaf",
                        session_id="parent-session-1234",
                    )

        # 1. 验证 agent_id 被生成
        assert len(generated_ids) == 1
        child_agent_id = generated_ids[0]
        assert "parent-s" in child_agent_id  # parent session id 前 8 位 = "parent-s"

        # 2. 验证 metadata 状态 = completed
        meta = sp.load_metadata(child_agent_id)
        assert meta is not None
        assert meta["status"] == "completed"
        assert meta["agent_type"] in ("general-purpose", "leaf")
        assert "completed_at" in meta

        # 3. 验证 AIAgent 构造时收到了独立 hooks_registry：
        #    轮级 transcript 走 POST_LLM_CALL 程序式 hook，不用 on_response
        assert mock_ctor.called
        _, kwargs = mock_ctor.call_args
        assert kwargs.get("hooks_registry") is not None
        # on_response 已删除（每轮已记，避免双写）
        assert not kwargs.get("on_response")

    def test_transcript_persisted_on_failure(self, isolated_sessions_dir, monkeypatch):
        """_run_child 异常时 status=failed。"""
        import agent.subagent_persistence as sp

        generated_ids = []
        orig_gen = sp.generate_agent_id

        def capture_gen(parent_session_id=""):
            aid = orig_gen(parent_session_id)
            generated_ids.append(aid)
            return aid
        monkeypatch.setattr(sp, "generate_agent_id", capture_gen)

        mock_child = MagicMock()
        mock_child.llm_client = MagicMock()
        mock_child.model = "fake"

        async def chat_side_effect(msg):
            raise RuntimeError("子代理崩溃了")
        mock_child.chat.side_effect = chat_side_effect

        with patch("agent.AIAgent", return_value=mock_child):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}):
                with patch("config.load_config", return_value={
                    "model": {
                        "name": "fake-model",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "http://fake",
                    }
                }):
                    from tools.delegate_tool import _run_child
                    with pytest.raises(RuntimeError):
                        _run_child("test goal", "", "leaf")

        # 验证 status = failed
        assert len(generated_ids) == 1
        meta = sp.load_metadata(generated_ids[0])
        assert meta is not None
        assert meta["status"] == "failed"

    def test_on_response_appends_transcript(self, isolated_sessions_dir, monkeypatch):
        """轮级 hook 把每轮 assistant 文本写入 transcript。

        子代理拿独立 hooks_registry，每轮 LLM 响应（POST_LLM_CALL）都会
        append（on_response 语义只记最终响应，给不了完整轨迹）。
        """
        import agent.subagent_persistence as sp

        generated_ids = []
        orig_gen = sp.generate_agent_id

        def capture_gen(parent_session_id=""):
            aid = orig_gen(parent_session_id)
            generated_ids.append(aid)
            return aid
        monkeypatch.setattr(sp, "generate_agent_id", capture_gen)

        # Mock AIAgent，构造时拿到 hooks_registry
        mock_child = MagicMock()
        mock_child.llm_client = MagicMock()
        mock_child.model = "fake"

        async def chat_side_effect(msg):
            return "结果"
        mock_child.chat.side_effect = chat_side_effect

        with patch("agent.AIAgent", return_value=mock_child) as mock_ctor:
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}):
                with patch("config.load_config", return_value={
                    "model": {
                        "name": "fake-model",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "http://fake",
                    }
                }):
                    from tools.delegate_tool import _run_child
                    _run_child("test", "", "leaf")

        # 从构造参数拿子代理的 hooks_registry，模拟 2 轮 LLM 响应
        # （真实流程中 AIAgent._run_post_llm_call_hook 每次 LLM 调用后触发）
        _, kwargs = mock_ctor.call_args
        hooks = kwargs.get("hooks_registry")
        assert hooks is not None

        def _resp(text):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=text, tool_calls=None),
                )],
            )

        out1 = hooks.run_post_llm_call(_resp("first response"))
        out2 = hooks.run_post_llm_call(_resp("second response"))
        # hook 不修改 response（返回 None 表示不修改，run 链保持原对象）
        assert out1 is not None and out2 is not None

        child_agent_id = generated_ids[0]
        transcript = sp.load_transcript(child_agent_id)
        # user 指令 + 2 轮 assistant = 3 条
        assert len(transcript) == 3
        assert transcript[0]["role"] == "user"
        assert transcript[1]["content"] == "first response"
        assert transcript[2]["content"] == "second response"


# ---------------------------------------------------------------------------
# 轮级 transcript（POST_LLM_CALL 程序式 hook，每轮 append）
# ---------------------------------------------------------------------------

def _spawn_mock_child(monkeypatch, goal="test goal", context=""):
    """公共 helper：起 mock _run_child，返回 (agent_id, mock_ctor, mock_child)。

    mock AIAgent 不真正跑 LLM；轮级行为由测试拿 ctor kwargs 的
    hooks_registry 手动触发（模拟 AIAgent._run_post_llm_call_hook）。
    """
    import agent.subagent_persistence as sp

    generated_ids = []
    orig_gen = sp.generate_agent_id

    def capture_gen(parent_session_id=""):
        aid = orig_gen(parent_session_id)
        generated_ids.append(aid)
        return aid
    monkeypatch.setattr(sp, "generate_agent_id", capture_gen)

    mock_child = MagicMock()
    mock_child.llm_client = MagicMock()
    mock_child.model = "fake"

    async def chat_side_effect(msg):
        return "子代理执行成功"
    mock_child.chat.side_effect = chat_side_effect

    with patch("agent.AIAgent", return_value=mock_child) as mock_ctor:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake"}):
            with patch("config.load_config", return_value={
                "model": {
                    "name": "fake-model",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "http://fake",
                }
            }):
                from tools.delegate_tool import _run_child
                _run_child(goal, context, "leaf", session_id="parent-session-1234")
    return generated_ids[0], mock_ctor, mock_child


def _llm_resp(content, tool_calls=None):
    """构造 OpenAI 风格的 mock LLM response（POST_LLM_CALL hook 的入参）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
        )],
    )


class TestPerTurnTranscript:
    """subagent transcript 每轮 append（完整轨迹，可 resume 中断代理）。"""

    def test_transcript_records_multiple_turns(self, isolated_sessions_dir, monkeypatch):
        """mock 2 轮 LLM 响应 → transcript ≥3 条（user + 2 assistant）。"""
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)

        _, kwargs = mock_ctor.call_args
        hooks = kwargs.get("hooks_registry")
        assert hooks is not None, "子代理应拿到独立 hooks_registry"

        hooks.run_post_llm_call(_llm_resp("第一轮：我先查一下文件"))
        hooks.run_post_llm_call(_llm_resp("最终结果：任务完成"))

        transcript = sp.load_transcript(aid)
        assert len(transcript) >= 3
        # 第 1 条是 user 指令（轨迹开头，resume 的对话起点）
        assert transcript[0]["role"] == "user"
        assert "test goal" in transcript[0]["content"]
        assistants = [m for m in transcript if m["role"] == "assistant"]
        assert len(assistants) == 2
        assert assistants[0]["content"] == "第一轮：我先查一下文件"
        assert assistants[1]["content"] == "最终结果：任务完成"

    def test_user_directive_with_context(self, isolated_sessions_dir, monkeypatch):
        """带 context 时 user 指令含上下文（对齐子代理 directive 格式）。"""
        import agent.subagent_persistence as sp
        aid, _, _ = _spawn_mock_child(
            monkeypatch, goal="查日志", context="项目是 HermesAgent")

        transcript = sp.load_transcript(aid)
        assert transcript[0]["role"] == "user"
        assert "查日志" in transcript[0]["content"]
        assert "HermesAgent" in transcript[0]["content"]

    def test_tool_calls_only_turn_not_appended(self, isolated_sessions_dir, monkeypatch):
        """纯 tool_calls 轮（content=None）不 append——没有文本可记。"""
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        hooks = kwargs["hooks_registry"]

        before = len(sp.load_transcript(aid))
        hooks.run_post_llm_call(_llm_resp(None, tool_calls=[{"id": "call_1"}]))
        hooks.run_post_llm_call(_llm_resp("", tool_calls=[{"id": "call_2"}]))
        after = sp.load_transcript(aid)

        assert len(after) == before  # 无文本轮不产生记录
        # 关键契约：轨迹永不带 tool_calls（无配对 tool result 会造孤儿 → API 400）
        assert all("tool_calls" not in m for m in after)

    def test_anthropic_list_content_blocks(self, isolated_sessions_dir, monkeypatch):
        """Anthropic 风格 content blocks（list）只拼 text 块。"""
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        hooks = kwargs["hooks_registry"]

        hooks.run_post_llm_call(_llm_resp([
            {"type": "text", "text": "先分析"},
            {"type": "tool_use", "id": "tu_1"},
            {"type": "text", "text": "再执行"},
        ]))

        assistants = [m for m in sp.load_transcript(aid) if m["role"] == "assistant"]
        assert len(assistants) == 1
        assert "先分析" in assistants[0]["content"]
        assert "再执行" in assistants[0]["content"]

    def test_hook_fail_open_on_append_error(self, isolated_sessions_dir, monkeypatch):
        """append_message 抛异常时 hook 不崩、不修改 response。"""
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        hooks = kwargs["hooks_registry"]

        def _raise(agent_id, message):
            raise OSError("disk full")
        monkeypatch.setattr(sp, "append_message", _raise)

        resp = _llm_resp("这轮写入会失败")
        out = hooks.run_post_llm_call(resp)  # 不应抛
        assert out is resp  # response 原样返回

    def test_malformed_response_fail_open(self, isolated_sessions_dir, monkeypatch):
        """畸形 response（choices 空/属性缺失）安全跳过。"""
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        hooks = kwargs["hooks_registry"]

        before = len(sp.load_transcript(aid))
        hooks.run_post_llm_call(SimpleNamespace(choices=[]))
        hooks.run_post_llm_call(SimpleNamespace(choices=None))
        hooks.run_post_llm_call(object())  # 任意怪对象
        assert len(sp.load_transcript(aid)) == before

    def test_final_response_no_double_write(self, isolated_sessions_dir, monkeypatch):
        """on_response 路径已删——最终响应只由轮级 hook 记一次。

        mock chat 返回后（真实场景 on_response 曾在这里补记最终响应），
        轮级 hook 再触发最终轮，transcript 中该文本只出现一次。
        """
        import agent.subagent_persistence as sp
        aid, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        assert not kwargs.get("on_response"), "on_response 应已从 _run_child 移除"

        hooks = kwargs["hooks_registry"]
        hooks.run_post_llm_call(_llm_resp("子代理执行成功"))

        transcript = sp.load_transcript(aid)
        contents = [m["content"] for m in transcript]
        assert contents.count("子代理执行成功") == 1  # 不双写

    def test_hooks_registry_not_shared_with_parent(self, isolated_sessions_dir, monkeypatch):
        """子代理的 hooks_registry 是独立新建实例（不共享主 agent，零污染）。"""
        from agent.hooks import HookEvent, HookRegistry
        _, mock_ctor, _ = _spawn_mock_child(monkeypatch)
        _, kwargs = mock_ctor.call_args
        hooks = kwargs["hooks_registry"]
        assert isinstance(hooks, HookRegistry)
        # 只注册了 POST_LLM_CALL 一个程序式 hook（轮级 transcript），别的事件为空
        assert len(hooks._hooks[HookEvent.POST_LLM_CALL]) == 1


# ---------------------------------------------------------------------------
# encoding 测试（Windows 必须 utf-8）
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_utf8_content_persisted(self, isolated_sessions_dir):
        """中文内容能正确读写。"""
        from agent.subagent_persistence import (
            write_metadata, load_metadata, append_message, load_transcript,
            generate_agent_id,
        )
        aid = generate_agent_id()
        write_metadata(aid, {
            "status": "running",
            "description": "子代理测试中文描述",
        })
        meta = load_metadata(aid)
        assert "子代理测试中文描述" in meta["description"]

        append_message(aid, {"role": "user", "content": "你好世界"})
        msgs = load_transcript(aid)
        assert msgs[0]["content"] == "你好世界"
