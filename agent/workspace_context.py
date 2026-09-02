"""子代理的工作目录（cwd）隔离机制——用 ContextVar 取代 os.chdir。

为什么需要：
    子代理跑在独立的 git worktree（工作副本目录）里，而 os.chdir 改的是
    整个进程共用的"当前目录"，就好比全家共用一块白板：线程 A 刚写上自己
    的目录，线程 B 一擦写上自己的，A 的就被踩了。ThreadPoolExecutor
    并发跑子代理时必然互相打架：

        线程 A: chdir(/worktree_A) → cwd = /worktree_A
        线程 B: chdir(/worktree_B) → cwd = /worktree_B（A 已经被踩了）

    解法：contextvars.ContextVar 相当于给每个线程发一块自己的小白板，
    各写各的，互看不见（线程隔离），并发就安全了。

用法：
    # 临时设置当前 context 的 workspace cwd
    with workspace_cwd_context("/path/to/worktree"):
        # 在这个 context 内，get_workspace_cwd() 返回 /path/to/worktree
        run_child_agent()

    # 工具/权限层读 cwd 一律走 get_workspace_cwd()，不要再直接 os.getcwd()
    cwd = get_workspace_cwd()
"""

import contextvars
import os
from contextlib import contextmanager
from typing import Optional

# 每个线程各自一份的 workspace cwd（None = 没设置，退回 os.getcwd()）。
# 细节：ThreadPoolExecutor 开新线程时会拷贝一份主线程当前的 context 启动，
# 但线程内部对 ContextVar 的 set/reset 只影响自己这份数据（token 的作用
# 域是线程局部的），所以不会漏到别的线程去。
_workspace_cwd: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "codeagent_workspace_cwd", default=None,
)


def get_workspace_cwd() -> str:
    """问一句"我现在在哪个目录干活"。

    返回：
        当前目录路径。

    优先级：先看本线程的 ContextVar 设没设；没设就用 os.getcwd()
    （向后兼容）。并发子代理每线程一份 ContextVar，
    各拿各的答案互不干扰。
    """
    return _workspace_cwd.get() or os.getcwd()


@contextmanager
def workspace_cwd_context(path: Optional[str]):
    """在 with 块范围内临时切换 workspace cwd，出了块自动切回来。

    参数：
        path：要切到的目录；传 None 等于"啥也不设"（继续走 os.getcwd()），
            方便调用方不用写 if 判断、统一传参。

    用法：
        with workspace_cwd_context("/path/to/worktree"):
            # 这个范围内 get_workspace_cwd() 返回 /path/to/worktree
            ...

    为什么能自动恢复：set 时会拿到一个 token（回程票），with 块结束用
    它把值还原——就算块里抛异常，finally 也会恢复。
    """
    token = _workspace_cwd.set(path)
    try:
        yield
    finally:
        _workspace_cwd.reset(token)


# ---------------------------------------------------------------------------
# 会话级切换
# ---------------------------------------------------------------------------
# 跟上面的 workspace_cwd_context（with 块、出了块就还原）不同，下面这对
# 函数是"长效开关"：一切换，所有 get_workspace_cwd() 的调用方都跟着换，
# 直到显式 clear 或会话结束。
#
# ⚠️ 并发局限（回程票存在模块级全局变量里的固有边界——这是刻意文档化，
# 不是能修的 bug）：
#   1. 回程票（token）整个进程只有一张，所以只给**主对话单线程**用
#      （worktree_enter/exit 工具本来就在主循环里串行执行）。
#   2. set 和 clear 必须在同一个 context（同一线程环境）里配对调用，
#      否则 ContextVar.reset 会抛 "Token was created in a different Context"。
#   3. 子代理的 workspace_cwd_context with 块不受影响（它们各拿各的局部
#      票，出块即还原）；但子代理**启动时**会拷贝主 context 的一份，所以
#      enter 之后才 spawn 的子代理会继承切换后的会话 cwd——这正是
#      "会话级切换"想要的效果。
_session_token: Optional[contextvars.Token] = None


def set_session_workspace_cwd(path: Optional[str]) -> None:
    """整个会话切换 cwd（进 worktree 后全 agent 跟着搬）。

    参数：
        path：要切去的目录；传 None 表示回到"没有会话级覆盖"的状态。

    这是长效 set（不是 with 块那种临时的）：切完之后所有
    get_workspace_cwd() 调用方都跟随，直到 clear_session_workspace_cwd
    或会话结束。重复 set 时旧回程票直接作废（理解为"又搬了一次家"，
    不会再搬回中间那个旧地址）。
    """
    global _session_token
    _session_token = _workspace_cwd.set(path)


def clear_session_workspace_cwd() -> None:
    """撤销会话级切换：把 cwd 恢复到 set 之前的样子。

    重复调用也安全（没设置过时就是什么都不做，不报错——幂等）。
    """
    global _session_token
    if _session_token is not None:
        _workspace_cwd.reset(_session_token)
        _session_token = None
