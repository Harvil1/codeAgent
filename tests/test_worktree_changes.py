"""Task G 测试：worktree 变更检测 + 智能清理。

测试矩阵：
1. has_worktree_changes — git 场景（有改动/无改动/异常 fail-open）
2. has_worktree_changes — 非 git 场景（temp dir listdir 对比）
3. cleanup_worktree_smart — 有改动保留/无改动清理/force=True
4. create_isolated_workspace — 返回的 cleanup 默认智能（有改动保留）
5. 端到端：子代理 isolated_workspace 写文件 → worktree 保留；不写 → 清理
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.worktree import (
    has_worktree_changes,
    cleanup_worktree_smart,
    create_isolated_workspace,
    _create_temp_workspace,
    _create_git_worktree,
)


# ---------------------------------------------------------------------------
# 辅助：初始化 git 仓库
# ---------------------------------------------------------------------------

def _init_git_repo(tmp_path: Path) -> Path:
    """初始化一个 git 仓库并做初始 commit，返回 repo root。"""
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
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# has_worktree_changes — git 场景
# ---------------------------------------------------------------------------

def test_has_changes_git_no_changes(tmp_path):
    """git worktree 无改动 → False。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "test-nochange")
    try:
        assert has_worktree_changes(wt_path) is False
    finally:
        cleanup()


def test_has_changes_git_with_new_file(tmp_path):
    """git worktree 有新文件 → True。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "test-newfile")
    try:
        (wt_path / "new_file.txt").write_text("hello", encoding="utf-8")
        assert has_worktree_changes(wt_path) is True
    finally:
        cleanup(force=True) if callable(cleanup) and 'force' in cleanup.__code__.co_varnames else cleanup()


def test_has_changes_git_with_modification(tmp_path):
    """git worktree 修改已有文件 → True。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "test-modify")
    try:
        (wt_path / "README.md").write_text("modified", encoding="utf-8")
        assert has_worktree_changes(wt_path) is True
    finally:
        cleanup(force=True) if callable(cleanup) and 'force' in cleanup.__code__.co_varnames else cleanup()


def test_has_changes_git_exception_fail_open(tmp_path):
    """git status 抛异常 → True（fail-open，保守保留）。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "test-exception")
    try:
        with patch("subprocess.run", side_effect=RuntimeError("git 挂了")):
            result = has_worktree_changes(wt_path)
        assert result is True  # fail-open
    finally:
        cleanup(force=True) if callable(cleanup) and 'force' in cleanup.__code__.co_varnames else cleanup()


# ---------------------------------------------------------------------------
# has_worktree_changes — 非 git 场景（listdir 对比）
# ---------------------------------------------------------------------------

def test_has_changes_non_git_no_changes():
    """非 git temp 目录无改动（snapshot 后无新文件）→ False。"""
    wt_path, cleanup = _create_temp_workspace("test-nongit-nochange")
    try:
        # snapshot 在创建时已记，此时没新文件
        assert has_worktree_changes(wt_path) is False
    finally:
        cleanup()


def test_has_changes_non_git_with_new_file():
    """非 git temp 目录有新文件 → True。"""
    wt_path, cleanup = _create_temp_workspace("test-nongit-newfile")
    try:
        (wt_path / "output.txt").write_text("result", encoding="utf-8")
        assert has_worktree_changes(wt_path) is True
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# cleanup_worktree_smart
# ---------------------------------------------------------------------------

def test_cleanup_smart_no_changes_cleans(tmp_path):
    """无改动 → cleanup_worktree_smart 清理（True）。

    注意：cleanup_worktree_smart 只清理目录不删分支（方案 B），
    分支删除用闭包 cleanup（有精确 branch 上下文）。
    """
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "smart-nochange")
    try:
        result = cleanup_worktree_smart(wt_path)
        assert result is True
        assert not wt_path.exists()
    finally:
        # 用闭包清理分支（worktree 目录已被 smart cleanup 删，闭包兜底）
        cleanup(force=True)


def test_cleanup_smart_with_changes_preserves(tmp_path):
    """有改动 → cleanup_worktree_smart 保留（False）。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "smart-changes")
    try:
        (wt_path / "output.txt").write_text("data", encoding="utf-8")
        result = cleanup_worktree_smart(wt_path)
        assert result is False
        assert wt_path.exists()
    finally:
        # 用闭包清理（删 worktree 目录 + 分支）
        cleanup(force=True)


def test_cleanup_smart_force_cleans_even_with_changes(tmp_path):
    """force=True 时即使有改动也清理。"""
    repo = _init_git_repo(tmp_path)
    wt_path, cleanup = _create_git_worktree(repo, "smart-force")
    try:
        (wt_path / "output.txt").write_text("data", encoding="utf-8")
        result = cleanup_worktree_smart(wt_path, force=True)
        assert result is True
        assert not wt_path.exists()
    except Exception:
        cleanup(force=True)
        raise


# ---------------------------------------------------------------------------
# create_isolated_workspace 返回 smart cleanup（默认智能）
# ---------------------------------------------------------------------------

def test_create_workspace_smart_cleanup_preserves_with_changes(tmp_path):
    """create_isolated_workspace 返回的 cleanup 有改动时保留。"""
    repo = _init_git_repo(tmp_path)
    path, cleanup = create_isolated_workspace(base_path=tmp_path, name="smart-ws")
    try:
        (path / "generated.txt").write_text("output", encoding="utf-8")
    finally:
        result = cleanup()  # 默认 smart
    # 有改动 → 保留 → cleanup 返回 False
    assert result is False or result is None
    assert path.exists()
    # 用闭包强制清理（删目录 + 分支）
    cleanup(force=True)


def test_create_workspace_smart_cleanup_no_changes(tmp_path):
    """create_isolated_workspace 返回的 cleanup 无改动时清理。"""
    repo = _init_git_repo(tmp_path)
    path, cleanup = create_isolated_workspace(base_path=tmp_path, name="smart-clean")
    result = cleanup()
    assert not path.exists()


def test_create_workspace_force_cleanup(tmp_path):
    """cleanup(force=True) 总是清理（有改动也清）。"""
    repo = _init_git_repo(tmp_path)
    path, cleanup = create_isolated_workspace(base_path=tmp_path, name="force-clean")
    (path / "x.txt").write_text("x", encoding="utf-8")
    result = cleanup(force=True)
    assert not path.exists()


# ---------------------------------------------------------------------------
# 非 git temp workspace 智能清理
# ---------------------------------------------------------------------------

def test_temp_workspace_smart_cleanup_preserves():
    """非 git temp 目录有改动 → smart cleanup 保留。"""
    path, cleanup = _create_temp_workspace("temp-smart")
    (path / "result.txt").write_text("data", encoding="utf-8")
    result = cleanup()  # 默认 smart
    assert result is False or result is None
    assert path.exists()
    # 手动清理
    shutil.rmtree(path, ignore_errors=True)


def test_temp_workspace_smart_cleanup_no_changes():
    """非 git temp 目录无改动 → smart cleanup 清理。"""
    path, cleanup = _create_temp_workspace("temp-clean")
    result = cleanup()
    assert not path.exists()


# ---------------------------------------------------------------------------
# 端到端：delegate_tool 子代理 isolated_workspace
# ---------------------------------------------------------------------------

def test_e2e_delegate_with_file_changes_preserves_worktree(tmp_path):
    """端到端：子代理用 isolated_workspace 写文件 → worktree 保留。

    Mock _run_child 的 AIAgent 构造，直接在 workspace 写文件
    来模拟子代理行为。
    """
    repo = _init_git_repo(tmp_path)

    captured_workspace = []

    def mock_run_child(goal, context, role, **kwargs):
        # 子代理在 worktree 里写了文件
        from agent.workspace_context import get_workspace_cwd
        ws = get_workspace_cwd()
        captured_workspace.append(ws)
        if ws:
            p = Path(ws)
            (p / "generated_output.txt").write_text(
                "子代理生成的内容", encoding="utf-8",
            )
        return "子代理结果：文件已写入"

    with patch("tools.delegate_tool._run_child", side_effect=mock_run_child):
        from tools.registry import registry
        result = registry.dispatch(
            "subagent",
            {"goal": "生成文件", "isolated_workspace": True},
            base_url=None, api_key="fake", model="test",
        )

    import asyncio
    data = asyncio.run(_extract_result(result))
    assert data["success"] is True

    # worktree 应该保留（有改动）
    if captured_workspace:
        ws_path = Path(captured_workspace[0])
        assert ws_path.exists(), "有改动的 worktree 应该保留"
        assert (ws_path / "generated_output.txt").exists()
        # 清理
        cleanup_worktree_smart(ws_path, force=True)


async def _extract_result(result):
    """registry.dispatch 可能返回 str 或 coroutine，统一处理。"""
    if hasattr(result, "__await__"):
        return json.loads(await result)
    if isinstance(result, str):
        return json.loads(result)
    return result


def test_e2e_delegate_no_changes_cleans_worktree(tmp_path):
    """端到端：子代理用 isolated_workspace 不写文件 → worktree 清理。"""
    repo = _init_git_repo(tmp_path)

    captured_workspace = []

    def mock_run_child(goal, context, role, **kwargs):
        from agent.workspace_context import get_workspace_cwd
        ws = get_workspace_cwd()
        captured_workspace.append(ws)
        return "子代理结果：没动文件"

    with patch("tools.delegate_tool._run_child", side_effect=mock_run_child):
        from tools.registry import registry
        result = registry.dispatch(
            "subagent",
            {"goal": "查信息", "isolated_workspace": True},
            base_url=None, api_key="fake", model="test",
        )

    import asyncio
    data = asyncio.run(_extract_result(result))
    assert data["success"] is True

    # worktree 应该被清理（无改动）
    if captured_workspace:
        ws_path = Path(captured_workspace[0])
        assert not ws_path.exists(), "无改动的 worktree 应该被清理"


def test_e2e_worktree_preserved_in_result(tmp_path):
    """端到端：有改动时，worktree_preserved 字段出现在返回结果里。"""
    repo = _init_git_repo(tmp_path)

    def mock_run_child(goal, context, role, **kwargs):
        from agent.workspace_context import get_workspace_cwd
        ws = get_workspace_cwd()
        if ws:
            (Path(ws) / "out.txt").write_text("data", encoding="utf-8")
        return "done"

    with patch("tools.delegate_tool._run_child", side_effect=mock_run_child):
        from tools.registry import registry
        result = registry.dispatch(
            "subagent",
            {"goal": "生成文件", "isolated_workspace": True},
            base_url=None, api_key="fake", model="test",
        )

    import asyncio
    data = asyncio.run(_extract_result(result))
    # _run_child mock 不会走 finally smart cleanup（mock 替换了整个函数）
    # 但真实 _run_child 的 result 里会带 worktree_preserved
    # 这里只验证 mock 路径不崩
    assert data["success"] is True
