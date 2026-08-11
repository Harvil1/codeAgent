"""子代理 workspace cwd 上下文（替代 os.chdir）。

用 contextvars.ContextVar 让每个并发子代理线程有自己的 cwd，
互不干扰（修 os.chdir 进程级全局踩 cwd 的 bug）。

背景：
    _run_child 用 os.chdir(workspace_path) 切到 worktree。但 os.chdir 是
    进程级全局状态，ThreadPoolExecutor 并发子代理会互相踩 cwd：

        线程 A: chdir(/worktree_A) → cwd = /worktree_A
        线程 B: chdir(/worktree_B) → cwd = /worktree_B（A 已经被踩了）

    ContextVar 在 ThreadPoolExecutor 里每个线程有自己的副本（线程隔离），
    不会互相干扰。

用法：
    # 设置当前 context 的 workspace cwd
    with workspace_cwd_context("/path/to/worktree"):
        # 在这个 context 内，get_workspace_cwd() 返回 /path/to/worktree
        run_child_agent()

    # 工具/权限层读 cwd 一律走 get_workspace_cwd()
    cwd = get_workspace_cwd()
"""

import contextvars
import os
from contextlib import contextmanager
from typing import Optional

# 每线程独立的 workspace cwd（None = fallback 到 os.getcwd()）
# ThreadPoolExecutor 的新线程会以主线程当前 context 的 copy 启动，
# 但子线程内 ContextVar 的 set/reset 只影响自己（token 作用域线程局部）。
_workspace_cwd: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "omnimate_workspace_cwd", default=None,
)


def get_workspace_cwd() -> str:
    """获取当前 context 的 workspace cwd。

    优先级：ContextVar > os.getcwd()。
    并发子代理（ThreadPoolExecutor）每线程独立 ContextVar，互不干扰。
    主线程或未设置时 fallback 到 os.getcwd()（向后兼容）。
    """
    return _workspace_cwd.get() or os.getcwd()


@contextmanager
def workspace_cwd_context(path: Optional[str]):
    """临时设置当前 context 的 workspace cwd。

    用法：
        with workspace_cwd_context("/path/to/worktree"):
            # 在这个 context 内，get_workspace_cwd() 返回 /path/to/worktree
            ...

    退出 with 块后自动恢复到之前的值（ContextVar token 机制）。
    path=None 时等价于"不设置"（仍走 os.getcwd()），方便调用方统一传。
    """
    token = _workspace_cwd.set(path)
    try:
        yield
    finally:
        _workspace_cwd.reset(token)
