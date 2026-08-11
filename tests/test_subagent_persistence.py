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

        cleaned = cleanup_old(days=7)
        assert cleaned == 1

        # 验证旧文件被删
        from agent.subagent_persistence import load_metadata
        assert load_metadata(aid_old) is None
        assert load_metadata(aid_new) is not None

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
            # 模拟 on_response 回调（_run_child 传给 AIAgent 的）
            # AIAgent 构造时接收 on_response；在 chat 中会调用
            # 我们这里不直接跑 AIAgent 内部，只验证 on_response 被传入了
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

        # 3. 验证 AIAgent 构造时收到了 on_response 回调
        assert mock_ctor.called
        _, kwargs = mock_ctor.call_args
        assert "on_response" in kwargs
        assert callable(kwargs["on_response"])

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
        """on_response 回调把 message 写入 transcript。"""
        import agent.subagent_persistence as sp

        generated_ids = []
        orig_gen = sp.generate_agent_id

        def capture_gen(parent_session_id=""):
            aid = orig_gen(parent_session_id)
            generated_ids.append(aid)
            return aid
        monkeypatch.setattr(sp, "generate_agent_id", capture_gen)

        # Mock AIAgent，构造时拿到 on_response 并手动调一下
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

        # 手动调 on_response（模拟 child 内部调用）
        _, kwargs = mock_ctor.call_args
        on_response = kwargs.get("on_response")
        assert on_response is not None

        child_agent_id = generated_ids[0]
        # 模拟两条 message
        on_response("first response")
        on_response("second response")

        # 验证 transcript 文件有两条
        transcript = sp.load_transcript(child_agent_id)
        # on_response 回调签名是 on_response(final_content: str)
        # 它内部构造 message dict 并 append
        assert len(transcript) == 2


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
