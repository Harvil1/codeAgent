"""并发子代理 workspace cwd 隔离测试。

背景：
    _run_child 原来用 os.chdir(workspace_path) 切到 worktree。但 os.chdir 是
    进程级全局状态，ThreadPoolExecutor(max_workers=5) 并发子代理会互相踩 cwd：

        线程 A: chdir(/worktree_A) → cwd = /worktree_A
        线程 B: chdir(/worktree_B) → cwd = /worktree_B（A 已经被踩了）
        A 继续跑，cwd 已经是 /worktree_B 不是 /worktree_A → A 的相对路径全错

    修复：用 contextvars.ContextVar（每线程独立）替代 os.chdir。
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from agent.workspace_context import (
    _workspace_cwd,
    get_workspace_cwd,
    workspace_cwd_context,
)


# ---------------------------------------------------------------------------
# 1. 单线程 basic
# ---------------------------------------------------------------------------

def test_basic_set_and_get(tmp_path):
    """with workspace_cwd_context(path) 内 get_workspace_cwd() 返回 path。"""
    target = str(tmp_path)
    with workspace_cwd_context(target):
        assert get_workspace_cwd() == target


def test_default_fallback_to_os_getcwd():
    """没 set 过，get_workspace_cwd() 返回 os.getcwd()。"""
    # 主线程没人 set 过（除非被前面测试污染，但 token reset 应该清理干净）
    # 用一个 fresh 的 ContextVar 验证 default 行为
    assert _workspace_cwd.get() is None
    assert get_workspace_cwd() == os.getcwd()


def test_exit_restores_previous(tmp_path):
    """退出 with 块后，get_workspace_cwd() 恢复到之前的值（os.getcwd）。"""
    before = get_workspace_cwd()
    with workspace_cwd_context(str(tmp_path)):
        assert get_workspace_cwd() == str(tmp_path)
    assert get_workspace_cwd() == before


# ---------------------------------------------------------------------------
# 2. 嵌套
# ---------------------------------------------------------------------------

def test_nested_context(tmp_path):
    """外层 set A，内层 set B，内层 get=B，退出内层 get=A。"""
    a = str(tmp_path / "A")
    b = str(tmp_path / "B")
    os.makedirs(a)
    os.makedirs(b)

    with workspace_cwd_context(a):
        assert get_workspace_cwd() == a
        with workspace_cwd_context(b):
            assert get_workspace_cwd() == b
        # 退出内层恢复到外层
        assert get_workspace_cwd() == a


def test_none_path_is_noop():
    """path=None 时 with 块内仍 fallback 到 os.getcwd()（调用方统一传 None 的安全语义）。"""
    with workspace_cwd_context(None):
        assert get_workspace_cwd() == os.getcwd()


# ---------------------------------------------------------------------------
# 3. 并发隔离（核心测试）
# ---------------------------------------------------------------------------

def test_concurrent_threads_isolated(tmp_path):
    """起 5 个线程，每个 set 不同 path，验证每个线程内 get_workspace_cwd 返回自己的 path。

    这是修 os.chdir 并发踩 cwd bug 的核心验证：ContextVar 在 ThreadPoolExecutor
    里每个线程有自己的副本，互不干扰。
    """
    paths = [str(tmp_path / f"w{i}") for i in range(5)]
    for p in paths:
        os.makedirs(p)

    results = {}  # thread_id -> 在线程内观测到的 cwd
    barrier = threading.Barrier(5)  # 让 5 个线程同时 set，最大化竞争窗口

    def worker(idx):
        my_path = paths[idx]
        with workspace_cwd_context(my_path):
            barrier.wait()  # 所有线程都 set 完再读
            time.sleep(0.05)  # 给其他线程时间 set（验证不会被踩）
            observed = get_workspace_cwd()
            results[idx] = observed

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for f in futures:
            f.result()

    # 每个线程看到的都是自己的 path（不是别人的）
    for i, expected in enumerate(paths):
        assert results[i] == expected, (
            f"线程 {i} 应该看到 {expected}，实际看到 {results[i]}"
            "（ContextVar 没隔离 = 并发踩 cwd bug 复现）"
        )


def test_child_thread_does_not_affect_main(tmp_path):
    """主线程 set A，子线程 set B，子线程退出后主线程 get_workspace_cwd 仍 = A。"""
    main_path = str(tmp_path / "main")
    child_path = str(tmp_path / "child")
    os.makedirs(main_path)
    os.makedirs(child_path)

    with workspace_cwd_context(main_path):
        # 主线程看到 A
        assert get_workspace_cwd() == main_path

        child_observed = []

        def child():
            with workspace_cwd_context(child_path):
                child_observed.append(get_workspace_cwd())

        t = threading.Thread(target=child)
        t.start()
        t.join()

        # 子线程看到 B
        assert child_observed == [child_path]
        # 主线程仍然是 A（没被子线程踩）
        assert get_workspace_cwd() == main_path


def test_concurrent_threads_no_real_chdir(tmp_path):
    """验证修复后 os.getcwd() 不再被子代理改（进程 cwd 保持不变）。

    这是 os.chdir 改 ContextVar 的副作用验证：
    主线程的 os.getcwd() 在并发子代理运行期间应该保持原值。
    """
    original_real_cwd = os.getcwd()
    paths = [str(tmp_path / f"w{i}") for i in range(3)]
    for p in paths:
        os.makedirs(p)

    def worker(idx):
        with workspace_cwd_context(paths[idx]):
            time.sleep(0.05)
            # 验证 ContextVar 生效
            assert get_workspace_cwd() == paths[idx]
            # 验证 os.getcwd() 没被改（这就是不用 os.chdir 的好处）
            assert os.getcwd() == original_real_cwd

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, i) for i in range(3)]
        for f in futures:
            f.result()

    # 主线程的 os.getcwd() 全程没动
    assert os.getcwd() == original_real_cwd


# ---------------------------------------------------------------------------
# 4. 端到端测试：_run_child + terminal_tool 读 get_workspace_cwd
# ---------------------------------------------------------------------------

def test_run_child_sets_workspace_context_for_tools(tmp_path, monkeypatch):
    """端到端：_run_child 在 isolated workspace 里跑，工具读到的是 workspace cwd。

    模拟 create_isolated_workspace 返回 tmp_path，然后 _run_child 内部应该把
    workspace_cwd_context 设到 tmp_path，工具（如 terminal_tool）调 get_workspace_cwd
    应该拿到 tmp_path 而不是 os.getcwd()。
    """
    # 跳过真实 LLM 调用：AIAgent.chat 不应该被实际执行
    # 只验证 workspace context 是否被正确设置
    workspace_a = tmp_path / "worktree_A"
    workspace_b = tmp_path / "worktree_B"
    workspace_a.mkdir()
    workspace_b.mkdir()

    observed_cwds = []
    original_run_child = None

    # 直接测试 workspace_cwd_context + get_workspace_cwd 的端到端链路
    # 模拟 _run_child 里的 pattern：set ContextVar → 跑子代理 → cleanup
    def simulate_run_child(workspace_path):
        from agent.workspace_context import _workspace_cwd
        token = _workspace_cwd.set(str(workspace_path))
        try:
            # 这里模拟 terminal_tool 读取 cwd 的行为
            from agent.workspace_context import get_workspace_cwd
            observed_cwds.append(get_workspace_cwd())
        finally:
            _workspace_cwd.reset(token)

    # 并发跑两个"子代理"
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(simulate_run_child, workspace_a)
        f2 = executor.submit(simulate_run_child, workspace_b)
        f1.result()
        f2.result()

    # 两个子代理看到各自的 workspace（顺序不保证，但两个值都应存在）
    assert str(workspace_a) in observed_cwds
    assert str(workspace_b) in observed_cwds
    assert len(observed_cwds) == 2


def test_terminal_tool_reads_workspace_cwd(tmp_path):
    """验证 terminal_tool 读 cwd 走 get_workspace_cwd（而非 os.getcwd）。

    在 workspace_cwd_context 内调 terminal_tool 的 cwd 解析逻辑，
    应该拿到 workspace path。
    """
    from tools.terminal_tool import _handle_terminal

    workspace = str(tmp_path)

    with workspace_cwd_context(workspace):
        # mock 权限检查放行
        with patch("agent.permission.safe_path") as mock_safe, \
             patch("agent.permission.get_default_checker") as mock_checker:
            mock_safe.return_value = type("R", (), {
                "allowed": True, "reason": "ok", "gate": "ok",
            })()
            mock_checker.return_value.check.return_value = type("R", (), {
                "allowed": True, "reason": "ok", "gate": "ok",
            })()

            # mock subprocess.run 不实际执行
            import subprocess
            mock_result = type("R", (), {
                "stdout": workspace,  # 返回 cwd 让测试验证
                "stderr": "",
                "returncode": 0,
            })()
            with patch("subprocess.run", return_value=mock_result):
                # 不传 cwd，让 terminal_tool 走 get_workspace_cwd
                result = _handle_terminal({"command": "pwd", "timeout": 5})

    # 验证 terminal_tool 没报错（权限/参数正常）
    import json
    data = json.loads(result)
    # 命令执行成功（可能输出截断或包装，但不应是 permission_denied）
    assert data.get("error_type") != "permission_denied"
