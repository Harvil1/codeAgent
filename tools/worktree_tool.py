"""worktree 工具（CCAR12 Task 6）：会话级 worktree 进出。

对齐 CCB EnterWorktree/ExitWorktree 语义——LLM 可以把**当前主对话**切进
一个 git worktree（隔离目录 + 独立分支），后续所有工具的
get_workspace_cwd() 都指向 worktree，直到 worktree_exit。

与 subagent(isolated_workspace=True) 的区别：
    subagent 隔离是**子代理 with 块**作用域（跑完即恢复）；
    这里是**会话级**长效切换（set_session_workspace_cwd，直到显式 exit）。

复用 CCAR5 基建（tools/worktree.py）：
    - 创建：git worktree add（分支 omnimate/<name>/<short_id>，目录
      <repo_root>/.worktrees/<name>）；非 git 目录降级系统临时目录
    - 清理：has_worktree_changes 智能检测 + cleanup_worktree_smart
      （有改动保留 + 提示，无改动删除）

接线（dispatch_kwargs）：
    - agent_ref：取 hooks_registry 触发 CWD_CHANGED（agent_ref 为 None 或
      无 hooks_registry 时跳过 hook，fail-open）
    - session_id：透传给 hook payload

两工具 isConcurrencySafe=False（改会话级全局 cwd + 建/删 worktree，串行）。
两 handler 是 async def（is_async=True）——会话 cwd 的 ContextVar set/reset
必须在主循环 context 里执行（dispatch 直接 await），走 to_thread 的
context 拷贝会导致 enter 静默失效 + exit reset 跨 context 抛 ValueError。
"""
import json
import logging
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from agent.workspace_context import (
    clear_session_workspace_cwd,
    get_workspace_cwd,
    set_session_workspace_cwd,
)
from tools.registry import registry
from tools.worktree import (
    cleanup_worktree_smart,
    get_repo_root,
    has_worktree_changes,
    is_git_repo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 会话状态（module-level，主对话单线程使用——见 workspace_context 并发局限）
# ---------------------------------------------------------------------------

class _SessionWorktree:
    """当前会话进入的 worktree 记录（enter 时写入，exit 时消费）。"""

    def __init__(self, path: Path, branch: Optional[str],
                 workspace_type: str, reused: bool,
                 repo_root: Optional[Path] = None):
        self.path = Path(path)
        self.branch = branch        # git 模式新建时记录；复用/降级时 None（不删分支）
        self.workspace_type = workspace_type  # "git" | "temp"
        self.reused = reused
        # enter 时记 repo root（exit 删分支用它——exit 前 cwd 已恢复，
        # 不能依赖恢复后的进程 cwd 反查 repo）
        self.repo_root = Path(repo_root) if repo_root else None


# 当前会话的 worktree（None = 不在 worktree 中）
_session_worktree: Optional[_SessionWorktree] = None

# name 白名单：只允许字母/数字/._-（防路径穿越和奇怪字符）
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _reset_session_worktree() -> None:
    """清空会话 worktree 状态（测试用；生产路径走 worktree_exit）。"""
    global _session_worktree
    _session_worktree = None
    clear_session_workspace_cwd()


# ---------------------------------------------------------------------------
# schema（OpenAI "parameters" 键——CCAR11 契约）
# ---------------------------------------------------------------------------

WORKTREE_ENTER_SCHEMA = {
    "name": "worktree_enter",
    "description": (
        "进入会话级 worktree：创建（或复用已存在的）.worktrees/<name> 并把"
        "当前会话的 cwd 切换过去。之后所有文件/命令操作都发生在 worktree 里，"
        "直到 worktree_exit。适合做实验性/破坏性改动而不污染主工作区。"
        "不传 name 时默认 wt-<时间戳>。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "worktree 名（也是分支名后缀）。省略时自动生成 wt-<时间戳>"
                ),
            },
        },
        "required": [],
    },
}

WORKTREE_EXIT_SCHEMA = {
    "name": "worktree_exit",
    "description": (
        "退出会话级 worktree，恢复原 cwd。keep=True（默认）保留 worktree 供"
        "查看；keep=False 时智能清理——有未提交改动则保留并提示，无改动则删除。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "keep": {
                "type": "boolean",
                "default": True,
                "description": "True 保留 worktree；False 无改动时删除（有改动仍保留）",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _err(tool_name: str, message: str, error_type: str, **extra) -> str:
    payload = {"error": message, "error_type": error_type, "tool": tool_name}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _fire_cwd_changed(dispatch_kwargs: dict, old: str, new: str) -> None:
    """触发 CWD_CHANGED hook（fail-open：无 hooks_registry / 异常都跳过）。"""
    agent_ref = dispatch_kwargs.get("agent_ref")
    hooks_registry = getattr(agent_ref, "hooks_registry", None) if agent_ref else None
    if hooks_registry is None:
        return
    try:
        hooks_registry.run_cwd_changed({
            "session_id": dispatch_kwargs.get("session_id", ""),
            "old": old,
            "new": new,
        })
    except Exception as e:
        logger.warning("CWD_CHANGED hook 异常（fail-open）: %s", e)


def _create_session_worktree(name: str):
    """建 worktree：git 仓库 → .worktrees/<name>（存在复用）；非 git → 系统临时目录。

    返回 _SessionWorktree。git worktree add 失败时降级临时目录（CCAR5 语义）。
    """
    base = Path(get_workspace_cwd())

    if is_git_repo(base):
        try:
            repo_root = get_repo_root(base) or base
            wt_dir = repo_root / ".worktrees" / name
            if wt_dir.exists():
                # 复用：目录已存在（上次会话留下）。分支未知 → exit 时不删分支。
                return _SessionWorktree(wt_dir, None, "git", reused=True,
                                        repo_root=repo_root)

            short_id = uuid.uuid4().hex[:8]
            branch = f"omnimate/{name}/{short_id}"
            result = subprocess.run(
                ["git", "worktree", "add", "-b", branch, str(wt_dir)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git worktree add 失败: {result.stderr.strip()}")
            logger.info("已创建会话 worktree: %s（分支 %s）", wt_dir, branch)
            return _SessionWorktree(wt_dir, branch, "git", reused=False,
                                    repo_root=repo_root)
        except Exception as e:
            logger.warning("git worktree 创建失败，降级到临时目录: %s", e)

    # 非 git（或 git 失败降级）：系统临时目录（CCAR5 _create_temp_workspace 语义）
    tmp = Path(tempfile.mkdtemp(prefix=f"omnimate-wt-{name}-"))
    return _SessionWorktree(tmp, None, "temp", reused=False)


async def _handle_worktree_enter(args: dict, **dispatch_kwargs) -> str:
    """进入会话级 worktree：建/复用目录 + set_session_workspace_cwd + CWD_CHANGED hook。

    为什么 async def（CCAR12 Task 6 fix，review Critical）：
    registry.dispatch 对 sync handler 走 asyncio.to_thread——会把当前
    context **拷贝**到 worker 线程，`set_session_workspace_cwd` 的 ContextVar
    set 只改拷贝，不回透主循环（enter 静默失效）；exit 的
    clear_session_workspace_cwd 在拷贝 context 里 reset 主 context 的
    token → ValueError → 永远 tool_exception（状态机死锁）。
    async handler dispatch 直接 await——同 task 同 context，set 生效 +
    token 同 context 可 reset。内部逻辑同步 IO（worktree 创建是本地
    git 命令），async 里直接跑即可。
    """
    global _session_worktree

    # CCAR13 A1（CCAR12 final review follow-up）：子代理（spawn_depth>0）不得
    # 切换会话级 worktree——ContextVar 是进程级共享，子代理 enter 会劫持
    # 主对话的 cwd，且 exit 在子代理结束时未必发生（会话级语义被滥用）。
    # isinstance 守卫：agent_ref 可能是 MagicMock（无 spawn_depth 属性时
    # getattr 返回 auto-attribute，不是 int——按 0 处理不误伤）。
    agent_ref = dispatch_kwargs.get("agent_ref")
    spawn_depth = getattr(agent_ref, "spawn_depth", 0)
    if isinstance(spawn_depth, int) and spawn_depth > 0:
        return _err(
            "worktree_enter",
            "子代理（spawn_depth>0）不能切换会话级 worktree，请由主代理调用",
            "permission_denied",
        )

    if _session_worktree is not None:
        return _err(
            "worktree_enter",
            f"当前已在 worktree 中（{_session_worktree.path}），先 worktree_exit",
            "already_in_worktree",
            path=str(_session_worktree.path),
        )

    raw_name = (args.get("name") or "").strip() or f"wt-{int(time.time())}"
    name = _NAME_RE.sub("-", raw_name)
    while ".." in name:  # ".." 单字符合法但拼起来是路径穿越 + git 拒绝的分支名
        name = name.replace("..", ".")
    name = name.strip(".-") or f"wt-{int(time.time())}"

    old_cwd = get_workspace_cwd()
    try:
        wt = _create_session_worktree(name)
    except Exception as e:
        logger.warning("worktree_enter 创建失败: %s", e, exc_info=True)
        return _err("worktree_enter", str(e), "create_failed")

    set_session_workspace_cwd(str(wt.path))
    _session_worktree = wt

    _fire_cwd_changed(dispatch_kwargs, old_cwd, str(wt.path))

    return json.dumps(
        {
            "path": str(wt.path),
            "reused": wt.reused,
            "workspace_type": wt.workspace_type,
            "branch": wt.branch,
        },
        ensure_ascii=False,
    )


async def _handle_worktree_exit(args: dict, **dispatch_kwargs) -> str:
    """退出会话级 worktree：clear_session_workspace_cwd + keep=False 时智能清理。

    async def 理由同 _handle_worktree_enter：clear 的 ContextVar reset
    必须和 set 同 context（to_thread 拷贝 context 里 reset 必炸）。
    """
    global _session_worktree
    if _session_worktree is None:
        return _err("worktree_exit", "当前不在 worktree 中", "not_in_worktree")

    keep = bool(args.get("keep", True))

    wt = _session_worktree
    _session_worktree = None
    clear_session_workspace_cwd()

    if keep:
        return json.dumps(
            {"path": str(wt.path), "kept": True, "cleaned": False,
             "reason": "keep"},
            ensure_ascii=False,
        )

    # keep=False：智能清理（有改动保留 + 提示，无改动删）
    if has_worktree_changes(wt.path):
        return json.dumps(
            {
                "path": str(wt.path),
                "kept": True,
                "cleaned": False,
                "reason": "has_changes",
                "hint": (
                    "worktree 有未提交改动已保留；确认后可手动删除，或提交/合并"
                    "分支后用 git worktree remove 清理"
                ),
            },
            ensure_ascii=False,
        )

    cleaned = cleanup_worktree_smart(wt.path)
    # git 模式新建的分支一并删（cleanup_worktree_smart 不删分支——CCAR5-G 分工：
    # 分支删除由有 branch 上下文的调用方负责，这里就是）。
    # repo 用 enter 时记的 wt.repo_root——此刻 cwd 已恢复到 worktree 之前的
    # 位置，不能依赖恢复后的进程 cwd 反查（Minor 2）
    if wt.branch and wt.workspace_type == "git":
        repo_root = wt.repo_root or get_repo_root(Path.cwd())
        if repo_root is not None:
            try:
                subprocess.run(
                    ["git", "branch", "-D", wt.branch],
                    cwd=str(repo_root),
                    capture_output=True,
                    timeout=10,
                )
            except Exception as e:
                logger.debug("删除分支 %s 失败: %s", wt.branch, e)

    return json.dumps(
        {"path": str(wt.path), "kept": False, "cleaned": cleaned},
        ensure_ascii=False,
    )


# 模块级注册（import 时自动执行）
# is_async=True：handler 是 async def——dispatch 直接 await（同 task 同 context），
# 不走 to_thread 的 context 拷贝（set_session_workspace_cwd 的 set/reset 必须在
# 主循环 context 里执行，否则 enter 静默失效 + exit reset 跨 context 炸 ValueError）
registry.register(
    name="worktree_enter",
    toolset="core",
    schema=WORKTREE_ENTER_SCHEMA,
    handler=_handle_worktree_enter,
    emoji="🌳",
    is_async=True,
    isConcurrencySafe=False,  # 改会话级全局 cwd + 建 worktree，必须串行
)
registry.register(
    name="worktree_exit",
    toolset="core",
    schema=WORKTREE_EXIT_SCHEMA,
    handler=_handle_worktree_exit,
    emoji="🌳",
    is_async=True,
    isConcurrencySafe=False,  # 清 cwd 状态 + 可能删 worktree，必须串行
)
