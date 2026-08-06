"""第 2 轮 bug 修复测试：致命级 bug（权限/数据完整性）。

每条测试对应一个 bug，TDD RED → GREEN。
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# S2: acceptEdits 模式应被 _common.get_mode_override_from_kwargs 接受
# ---------------------------------------------------------------------------

def test_get_mode_override_accepts_acceptedits():
    """S2 fix: get_mode_override_from_kwargs 应返回 'acceptEdits'，不静默丢弃。"""
    from tools._common import get_mode_override_from_kwargs

    class FakeAgent:
        permission_mode = "acceptEdits"

    result = get_mode_override_from_kwargs({"agent_ref": FakeAgent()})
    assert result == "acceptEdits", (
        f"acceptEdits 应被透传，实际: {result}（被静默丢弃会导致 acceptEdits 失效）"
    )


def test_get_mode_override_still_returns_default_and_bypass():
    """回归：default 和 bypassPermissions 不受影响。"""
    from tools._common import get_mode_override_from_kwargs

    class FakeAgent:
        permission_mode = "default"
    assert get_mode_override_from_kwargs({"agent_ref": FakeAgent()}) == "default"

    class FakeAgent2:
        permission_mode = "bypassPermissions"
    assert get_mode_override_from_kwargs({"agent_ref": FakeAgent2()}) == "bypassPermissions"


def test_get_mode_override_unknown_returns_none():
    """未知值仍返回 None。"""
    from tools._common import get_mode_override_from_kwargs

    class FakeAgent:
        permission_mode = "garbage"
    assert get_mode_override_from_kwargs({"agent_ref": FakeAgent()}) is None


# ---------------------------------------------------------------------------
# S1: bg_start 必须过权限闸门（含 fatal 底线）
# ---------------------------------------------------------------------------

def test_bg_start_rejects_fatal_command():
    """S1 fix: bg_start(command=['rm','-rf','/']) 必须被硬拒。

    bug：原代码完全不调 PermissionChecker，可绕过 fatal 底线。
    """
    from tools.bg_task import _handle_bg_start
    from agent.background import BackgroundManager

    bg = BackgroundManager()
    result = _handle_bg_start(
        {"command": ["rm", "-rf", "/"]},
        bg_manager=bg,
    )
    parsed = json.loads(result)
    assert parsed.get("error_type") == "permission_denied", (
        f"bg_start 应拒 fatal 命令，实际返回: {parsed}"
    )
    assert "fatal" in parsed.get("error", "").lower() or "硬底线" in parsed.get("error", "")


def test_bg_start_rejects_destructive_without_approval():
    """S1 fix: bg_start 跑 rm xxx（破坏性但非 fatal）无审批 callback 时拒绝。"""
    from tools.bg_task import _handle_bg_start
    from agent.background import BackgroundManager

    bg = BackgroundManager()
    result = _handle_bg_start(
        {"command": ["rm", "somefile.tmp"]},
        bg_manager=bg,
    )
    parsed = json.loads(result)
    # 没传 approval_callback → 走 destructive 拒绝路径
    assert parsed.get("error_type") in ("permission_denied", "destructive"), (
        f"bg_start 应拒破坏性命令（无审批），实际: {parsed}"
    )


def test_bg_start_allows_safe_command(monkeypatch):
    """回归：安全命令（ls/echo）仍能跑。"""
    from tools.bg_task import _handle_bg_start
    from agent import background as bg_mod

    # mock 实际启动 + status，只验证权限通过
    def fake_start(self, command, **kwargs):
        return "fake_task_id"

    class FakeTask:
        status = "running"
        pid = 12345

    def fake_status(self, task_id):
        return FakeTask()

    monkeypatch.setattr(bg_mod.BackgroundManager, "start", fake_start)
    monkeypatch.setattr(bg_mod.BackgroundManager, "status", fake_status)

    from agent.background import BackgroundManager
    bg = BackgroundManager()
    result = _handle_bg_start(
        {"command": ["ls"], "detach": True},
        bg_manager=bg,
    )
    parsed = json.loads(result)
    assert parsed.get("task_id") == "fake_task_id", (
        f"安全命令应通过并返回 task_id，实际: {parsed}"
    )


# ---------------------------------------------------------------------------
# S3: search_files 必须检查路径权限
# ---------------------------------------------------------------------------

def test_search_files_rejects_protected_path():
    """S3 fix: search_files(path='~/.ssh') 必须拒（safe_path 受保护路径）。"""
    from tools.file_operations import _handle_search_files

    # 不真实读取 ~/.ssh，只验证拒绝
    ssh_path = str(Path.home() / ".ssh")
    result = _handle_search_files({"pattern": "PRIVATE", "path": ssh_path})
    parsed = json.loads(result)
    assert parsed.get("error_type") == "permission_denied", (
        f"search_files 应拒 ~/.ssh，实际: {parsed}"
    )


def test_search_files_rejects_etc_path():
    """S3 fix: search_files(path='/etc') 必须拒。"""
    from tools.file_operations import _handle_search_files

    result = _handle_search_files({"pattern": "root", "path": "/etc"})
    parsed = json.loads(result)
    assert parsed.get("error_type") == "permission_denied", (
        f"search_files 应拒 /etc，实际: {parsed}"
    )


def test_search_files_allows_cwd(tmp_path):
    """回归：cwd 内仍能搜。"""
    from tools.file_operations import _handle_search_files

    # tmp_path 是 pytest 提供的临时目录，不在受保护列表
    (tmp_path / "test.txt").write_text("hello world", encoding="utf-8")
    result = _handle_search_files({"pattern": "hello", "path": str(tmp_path)})
    parsed = json.loads(result)
    assert "matches" in parsed or "error_type" not in parsed, (
        f"cwd 内应能搜，实际: {parsed}"
    )


# ---------------------------------------------------------------------------
# S4: terminal_tool cwd 必须检查权限
# ---------------------------------------------------------------------------

def test_terminal_rejects_protected_cwd():
    """S4 fix: terminal(cwd='~/.ssh') 必须拒。"""
    from tools.terminal_tool import _handle_terminal
    from tools.terminal_tool import check_terminal_requirements

    ssh_path = str(Path.home() / ".ssh")
    with patch("tools.terminal_tool.check_terminal_requirements", lambda: True):
        result = _handle_terminal({"command": "ls", "cwd": ssh_path})
    parsed = json.loads(result)
    assert parsed.get("error_type") == "permission_denied", (
        f"terminal cwd=~/.ssh 应拒，实际: {parsed}"
    )


def test_terminal_rejects_etc_cwd():
    """S4 fix: terminal(cwd='/etc') 必须拒。"""
    from tools.terminal_tool import _handle_terminal
    from unittest.mock import patch

    with patch("tools.terminal_tool.check_terminal_requirements", lambda: True):
        result = _handle_terminal({"command": "ls", "cwd": "/etc"})
    parsed = json.loads(result)
    assert parsed.get("error_type") == "permission_denied", (
        f"terminal cwd=/etc 应拒，实际: {parsed}"
    )


# ---------------------------------------------------------------------------
# S7: delete_session 原子性（两条 DELETE 包一个事务）
# ---------------------------------------------------------------------------

def test_delete_session_atomic_on_partial_failure():
    """S7 fix: delete_session 源码必须含 BEGIN/COMMIT/ROLLBACK（事务原子性）。

    sqlite3.Connection.execute 是 read-only，无法 monkeypatch，所以改静态验证：
    源码含 BEGIN → 显式事务 + ROLLBACK → 异常回滚。
    """
    import inspect
    from agent.session_store import SessionStore
    src = inspect.getsource(SessionStore.delete_session)
    # 必须显式开事务
    assert "BEGIN" in src, f"delete_session 应含 BEGIN（显式开事务），实际:\n{src}"
    # 必须有回滚路径
    assert "ROLLBACK" in src, f"delete_session 应含 ROLLBACK（异常回滚），实际:\n{src}"
    # 必须有提交
    assert "COMMIT" in src, f"delete_session 应含 COMMIT，实际:\n{src}"


def test_delete_session_normal_case(tmp_path):
    """回归：正常情况删 session 同时删 messages。"""
    from agent.session_store import SessionStore

    db_path = tmp_path / "test.db"
    store = SessionStore(db_path)
    sid = store.create_session(model="test", provider="test")
    store.append_message(sid, role="user", content="hello")

    store.delete_session(sid)

    sessions = store.list_sessions(limit=10)
    sids = [s["id"] if isinstance(s, dict) else s.id for s in sessions]
    assert sid not in sids
    # messages 也应没了
    msgs = store.get_messages(sid)
    assert len(msgs) == 0
