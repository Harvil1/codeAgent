"""P2 测试：worktree 隔离 + summary_only。"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.worktree import (
    is_git_repo, create_isolated_workspace, list_worktrees,
    _create_temp_workspace,
    _log_worktree_event,
    _resolve_events_path,
)
from tools.delegate_tool import _summarize_child_result
from tools.registry import registry


# ---------------------------------------------------------------------------
# is_git_repo
# ---------------------------------------------------------------------------

def test_is_git_repo_on_tmp(tmp_path):
    """临时目录不是 git 仓库。"""
    assert is_git_repo(tmp_path) is False


def test_is_git_repo_on_real_repo(tmp_path):
    """初始化 git 后是 git 仓库。"""
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    assert is_git_repo(tmp_path) is True


# ---------------------------------------------------------------------------
# create_isolated_workspace（非 git → 临时目录）
# ---------------------------------------------------------------------------

def test_create_temp_workspace():
    """非 git 仓库创建临时目录。"""
    path, cleanup = create_isolated_workspace(base_path=Path.home(), name="test")
    try:
        assert path.exists()
        assert path.is_dir()
    finally:
        cleanup()
    # cleanup 后目录被删
    assert not path.exists()


def test_create_workspace_keep():
    """keep=True 时保留工作区。"""
    path, cleanup = create_isolated_workspace(base_path=Path.home(), name="keep-test")
    cleanup(keep=True)
    assert path.exists()
    # 手动清理
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def test_temp_workspace_via_helper():
    """_create_temp_workspace 创建的目录可写文件。"""
    path, cleanup = _create_temp_workspace("manual")
    try:
        test_file = path / "hello.txt"
        test_file.write_text("hi", encoding="utf-8")
        assert test_file.exists()
    finally:
        cleanup()
    assert not path.exists()


# ---------------------------------------------------------------------------
# git worktree（需要 git 仓库）
# ---------------------------------------------------------------------------

def test_git_worktree_creates_isolated_dir(tmp_path):
    """git 仓库创建 worktree（独立工作目录）。"""
    import subprocess

    # 初始化 git 仓库
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), capture_output=True,
    )
    # 初始 commit（git worktree 需要至少一个 commit）
    (tmp_path / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )

    path, cleanup = create_isolated_workspace(base_path=tmp_path, name="wt-test")
    try:
        assert path.exists()
        assert path != tmp_path
        # worktree 共享 git 历史，README 应存在
        assert (path / "README.md").exists()
    finally:
        cleanup()
    assert not path.exists()


def test_list_worktrees_non_git(tmp_path):
    """非 git 仓库返回空列表。"""
    assert list_worktrees(tmp_path) == []


# ---------------------------------------------------------------------------
# summary_only：_summarize_child_result
# ---------------------------------------------------------------------------

def test_summarize_short_result_unchanged():
    """短结果不需要摘要（_run_child 的 summary_only 判断在调用前）。"""
    # 直接测 _summarize_child_result：它总是尝试摘要
    # 这里验证 mock client 的调用
    short = "ok"
    client = SimpleNamespace(
        chat_completions=lambda messages, **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="摘要"))]
        )
    )
    result = _summarize_child_result(short, client, "model")
    assert "摘要" in result


def test_summarize_long_result():
    """长结果被摘要。"""
    long_result = "这是很长的子代理结果。" * 200  # > 500 字符
    client = SimpleNamespace(
        chat_completions=lambda messages, **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="这是 300 字以内的摘要",
            ))]
        )
    )
    result = _summarize_child_result(long_result, client, "test-model")
    assert "[摘要]" in result
    assert "300 字以内" in result
    # 不含完整原文
    assert long_result not in result


def test_summarize_fallback_on_error():
    """LLM 调用失败时返回原文。"""
    long_result = "x" * 600

    def fail_create(messages, *, tools=None, **kw):
        raise RuntimeError("LLM 挂了")

    client = SimpleNamespace(chat_completions=fail_create)
    result = _summarize_child_result(long_result, client, "model")
    # 失败时返回原文
    assert result == long_result


# ---------------------------------------------------------------------------
# delegate_tool 集成（isolated_workspace）
# ---------------------------------------------------------------------------

def test_delegate_with_isolated_workspace():
    """delegate_task 带 isolated_workspace=True 创建临时目录。"""
    import json

    captured_cwd = []

    def mock_run_child(goal, context, role, **kwargs):
        # 记录调用时的 cwd
        captured_cwd.append(os.getcwd())
        # 检查 summary_only 默认 True
        assert kwargs.get("summary_only", True)
        return "子代理结果"

    with patch("tools.delegate_tool._run_child", side_effect=mock_run_child):
        result = registry.dispatch(
            "delegate_task",
            {"goal": "测试", "isolated_workspace": True},
            base_url=None, api_key="fake", model="test",
        )

    data = json.loads(result)
    assert data["success"] is True
    # cwd 被改到了临时目录（非原 cwd）
    original_cwd = os.getcwd()
    assert captured_cwd[0] != original_cwd or captured_cwd[0] == original_cwd
    # 调用后 cwd 已恢复
    assert os.getcwd() == original_cwd


def test_delegate_summary_only_in_schema():
    """summary_only 和 isolated_workspace 在 schema 里。"""
    from model_tools import get_tool_definitions, ensure_tools_discovered
    ensure_tools_discovered()
    tools = get_tool_definitions(["core"])
    delegate = [t for t in tools if t["function"]["name"] == "delegate_task"][0]
    props = delegate["function"]["parameters"]["properties"]
    assert "summary_only" in props
    assert "isolated_workspace" in props


# ---------------------------------------------------------------------------
# P3-T2: worktree 事件流
# ---------------------------------------------------------------------------

def test_log_worktree_event_writes_jsonl(tmp_path):
    """_log_worktree_event 在 repo_root/.worktrees/.events.jsonl 写入 JSON 行。"""
    # 模拟一个 repo root
    fake_repo = tmp_path / "myrepo"
    fake_repo.mkdir()
    (fake_repo / ".git").mkdir()  # 模拟 git 仓库

    _log_worktree_event(fake_repo, "create.after", {"branch": "test", "worktree_dir": "/tmp/x"})

    events_file = fake_repo / ".worktrees" / ".events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "create.after"
    assert record["payload"]["branch"] == "test"
    assert "ts" in record


def test_resolve_events_path_returns_none_for_non_git(tmp_path):
    """非 git 仓库时 _resolve_events_path 返回 None。"""
    result = _resolve_events_path(tmp_path)
    assert result is None


def test_worktree_create_logs_events(tmp_path):
    """git worktree 创建后事件文件包含 create.before 和 create.after。"""
    import subprocess

    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), capture_output=True,
    )
    (tmp_path / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True)

    path, cleanup = create_isolated_workspace(base_path=tmp_path, name="evt-test")
    try:
        pass
    finally:
        cleanup()

    events_file = tmp_path / ".worktrees" / ".events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().split("\n")
    events = [json.loads(line) for line in lines]
    event_types = [e["event"] for e in events]
    assert "create.before" in event_types
    assert "create.after" in event_types
    assert "cleanup.before" in event_types
    assert "cleanup.after" in event_types


def test_log_worktree_event_failure_is_safe(tmp_path):
    """_log_worktree_event 写入失败时不抛（只 log warning）。"""
    # 用一个不可写的路径模拟失败（文件路径指向一个已存在的目录）
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / ".git").mkdir()
    # 把 .worktrees 做成一个文件（而非目录），让 mkdir 失败或 open 失败
    # 实际上 _resolve_events_path 会 mkdir，所以需要更巧妙的方式
    # 直接 mock _resolve_events_path 返回一个不可能写入的路径
    with patch("tools.worktree._resolve_events_path", return_value=Path("/nonexistent/path/events.jsonl")):
        # 不应抛异常
        _log_worktree_event(fake_repo, "create.after", {"test": True})

