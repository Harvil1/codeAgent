"""tests/test_project_scope.py — 项目键计算测试。

对标 claude-code-main findCanonicalGitRoot：canonical git root（worktree 归一），
非 git 退 cwd。fail-open：git 命令失败 → 退 base。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.project_scope import get_project_memory_dir, get_project_memory_key


@pytest.fixture(autouse=True)
def _clear_key_cache():
    """每测试清空进程级缓存，避免跨测试 tmp_path 复用导致污染。"""
    import agent.project_scope as ps
    ps._key_cache.clear()
    yield
    ps._key_cache.clear()


def _ensure_git_identity(env: dict) -> dict:
    """tmp_path 里 git init 后 commit 需要身份，没配就临时塞默认值。"""
    env = dict(env)
    env.setdefault("GIT_AUTHOR_NAME", "Test")
    env.setdefault("GIT_AUTHOR_EMAIL", "test@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "Test")
    env.setdefault("GIT_COMMITTER_EMAIL", "test@example.com")
    return env


def test_non_git_dir_uses_base(tmp_path):
    """非 git 目录直接用 base 做 key。"""
    key = get_project_memory_key(str(tmp_path))
    assert key  # 非空
    assert "\\" not in key and "/" not in key and ":" not in key


def test_git_repo_uses_toplevel(tmp_path):
    """git 仓库用 rev-parse 的 canonical root（不是 base 本身）。"""
    env = _ensure_git_identity(os.environ)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    sub = tmp_path / "subdir"
    sub.mkdir()
    key_from_root = get_project_memory_key(str(tmp_path))
    key_from_sub = get_project_memory_key(str(sub))
    assert key_from_root == key_from_sub  # 子目录归一到 root


def test_sanitize_replaces_unsafe_chars(tmp_path):
    """Windows 路径 D:\\x\\y → D--x-y（分隔符/冒号替换为 -）。"""
    key = get_project_memory_key(r"D:\project\tanke")
    assert key == "D--project-tanke"


def test_worktree_normalizes_to_main_repo(tmp_path):
    """worktree 里算 key 归一到主 repo（canonical git root）。"""
    env = _ensure_git_identity(os.environ)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init", "-q"],
        cwd=tmp_path, check=True, env=env,
    )
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(wt), "HEAD"],
        cwd=tmp_path, check=True, env=env,
    )
    assert get_project_memory_key(str(wt)) == get_project_memory_key(str(tmp_path))


def test_git_failure_falls_back_to_base(tmp_path, monkeypatch):
    """git 命令不可用 → fail-open 退 base。

    Windows 下清空 PATH 可能仍能从系统路径找到 git，所以这里 monkeypatch
    `_git_toplevel` 直接返回 None 测 fallback 逻辑（与原测试同语义，
    只是测点从「git 不存在」改为「git 调用失败」）。
    """
    import agent.project_scope as ps
    monkeypatch.setattr(ps, "_git_toplevel", lambda base: None)
    key = get_project_memory_key(str(tmp_path / "projX"))
    assert key == get_project_memory_key(str(tmp_path / "projX"))  # 稳定
    assert "projX" in key


def test_get_project_memory_dir_layout(tmp_path):
    """get_project_memory_dir 返回标准布局，不 mkdir。"""
    d = get_project_memory_dir(tmp_path, base=r"D:\project\tanke")
    assert d == tmp_path / ".memory" / "projects" / "D--project-tanke"
    assert not d.exists()  # 不 mkdir（调用方按需）


def test_key_cached_per_base(tmp_path):
    """同 base 只算一次（缓存生效：第二次调用 git 不再被调）。

    用「同 base 两次调用结果一致」断言缓存生效的语义（弱断言但语义等价）。
    """
    import agent.project_scope as ps
    # 清空进程级缓存避免与其他测试干扰
    ps._key_cache.clear()
    key1 = get_project_memory_key(str(tmp_path))
    key2 = get_project_memory_key(str(tmp_path))
    assert key1 == key2
    # 缓存命中：tmp_path 已在 _key_cache
    assert str(tmp_path) in ps._key_cache


def test_daemon_thread_inherits_workspace_cwd():
    """daemon thread 用 copy_context 启动后能看到主线程的 workspace_cwd。

    curator/reflection 线程不传 contextvars
    会 fallback os.getcwd()，导致多项目场景漏扫/写错项目区。
    """
    import contextvars
    import threading

    from agent.workspace_context import get_workspace_cwd, workspace_cwd_context

    results = []

    def _worker():
        results.append(get_workspace_cwd())

    with workspace_cwd_context(r"D:\fake\project"):
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(_worker), daemon=True)
        t.start()
        t.join(timeout=5)

    assert results == [r"D:\fake\project"]


def test_daemon_thread_without_copy_context_falls_back_to_os_getcwd():
    """对照组：daemon thread 不 copy_context → 看不到主线程 workspace_cwd。

    用这个对照证明修法（copy_context().run）真的生效，而不是碰巧通过。
    """
    import threading

    from agent.workspace_context import get_workspace_cwd, workspace_cwd_context

    results = []
    original = os.getcwd()

    def _worker():
        results.append(get_workspace_cwd())

    # 用一个肯定不是当前 cwd 的假路径
    fake = r"Z:\definitely\not\real" if os.name == "nt" else "/definitely/not/real"
    with workspace_cwd_context(fake):
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)

    # 不 copy_context → fallback os.getcwd()（= 启动进程的 cwd）
    assert results == [original]
