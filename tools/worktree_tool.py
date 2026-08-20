"""worktree 的 LLM 工具（历史出处 CCAR12 Task 6）：让主对话整个搬进搬出 worktree。

这个文件是干嘛的：把 tools/worktree.py 的隔离能力包装成两个 LLM 能调的
工具——worktree_enter（搬进去）和 worktree_exit（搬出来）。语义对齐 CCB
的 EnterWorktree/ExitWorktree：LLM 可以把**当前主对话**切进一个 git
worktree（隔离目录 + 独立分支，好比给主对话临时开了间独立办公室），之后
所有工具问「当前目录在哪」（get_workspace_cwd()）得到的都是 worktree 的
路径，直到调 worktree_exit 才搬回来。

和 subagent(isolated_workspace=True) 的区别（两者别混）：
    subagent 那种隔离是**临时出差**——子代理跑完这块活就自动搬回原地；
    这里是**长期搬家**——会话级切换（set_session_workspace_cwd），不显式
    调 exit 就一直在 worktree 里待着。

底层复用 CCAR5 轮建好的基建（tools/worktree.py）：
    - 创建：git worktree add（分支叫 omnimate/<名字>/<8位短ID>，目录在
      <仓库根>/.worktrees/<名字>）；非 git 目录降级用系统临时目录
    - 清理：先聪明检测有没有改动（has_worktree_changes）再决定
      （有改动就保留并提示，干干净净才删）

系统接线（dispatch_kwargs 传进来的上下文）：
    - agent_ref：从它身上拿 hooks_registry，目录切换时触发 CWD_CHANGED
      钩子（没传 agent_ref 或没配钩子就跳过，fail-open）
    - session_id：捎带给钩子 payload 的会话 ID

两个关键设计（都是历史踩坑换来的）：
    1. isConcurrencySafe=False：这俩工具改的是会话级全局目录 + 建/删
       worktree，必须排队执行，不能并发。
    2. handler 是 async def（注册时 is_async=True）：会话目录底层是
       ContextVar（线程/协程各自的局部变量），set 和 reset 必须在主循环
       的 context 里执行。走 to_thread 的话 context 会被**拷贝**一份，
       在拷贝里 set 不回透主循环（enter 静默失效），exit 的 reset 更会
       跨 context 直接抛 ValueError（状态机死锁）。
"""
import json
import logging
import re
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
    _run_git,
    cleanup_worktree_smart,
    get_repo_root,
    has_worktree_changes,
    is_git_repo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 会话状态（放在模块级，只有主对话单线程用它——并发局限详见 workspace_context 模块）
# ---------------------------------------------------------------------------

class _SessionWorktree:
    """当前会话搬进了哪个 worktree 的登记信息（enter 时记下，exit 时用掉）。

    好比搬家时在旧家门口贴的条子：新家地址、新配的钥匙（分支名）、
    老家的位置，搬回来清理时全靠这张条子。
    """

    def __init__(self, path: Path, branch: Optional[str],
                 workspace_type: str, reused: bool,
                 repo_root: Optional[Path] = None):
        self.path = Path(path)
        self.branch = branch        # 只在 git 模式且新建时记录；复用已有/降级临时目录时是 None（None = exit 时不删分支）
        self.workspace_type = workspace_type  # "git"（真 worktree）| "temp"（临时目录）
        self.reused = reused
        # 进门时就记下仓库根目录：exit 删分支时要用。因为 exit 时 cwd 已搬回
        # 原处，再靠「当前目录」反查仓库就查错地方了
        self.repo_root = Path(repo_root) if repo_root else None


# 当前会话所在的 worktree（None = 还没搬进去）
_session_worktree: Optional[_SessionWorktree] = None

# 名字白名单：只放行字母/数字/. _ -，其他字符一律替换掉（防路径穿越和怪字符钻空子）
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _reset_session_worktree() -> None:
    """一键清空会话 worktree 状态。

    给测试用的快捷复位；生产路径不走这里（走 worktree_exit 的正规流程）。

    返回：无。
    """
    global _session_worktree
    _session_worktree = None
    clear_session_workspace_cwd()


# ---------------------------------------------------------------------------
# schema（工具说明书；注意参数键必须是 OpenAI 的 "parameters"——历史踩坑
# CCAR11 契约：用错键 LLM 就看不见参数定义，第 5 次翻车后立了规矩）
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
# 工具函数（schema 说明书下面这些是真正干活的实现）
# ---------------------------------------------------------------------------

def _err(tool_name: str, message: str, error_type: str, **extra) -> str:
    """拼一个统一格式的 JSON 错误返回。

    参数：
        tool_name：出错的工具名（帮 LLM 认出是谁报的错）。
        message：人话错误描述。
        error_type：错误类型标签（如 permission_denied）。
        **extra：想额外捎带的键值对（如 path）。

    返回：JSON 字符串。
    """
    payload = {"error": message, "error_type": error_type, "tool": tool_name}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _fire_cwd_changed(dispatch_kwargs: dict, old: str, new: str) -> None:
    """通知一声「目录换了」：触发 CWD_CHANGED 钩子。

    fail-open：没配钩子登记本、或钩子自己抛异常，都直接跳过，
    绝不影响目录切换本身。

    参数：
        dispatch_kwargs：系统上下文（从里面拿 agent_ref → hooks_registry
        和 session_id）。
        old：切换前的目录。
        new：切换后的目录。

    返回：无。
    """
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
    """建一个会话用的 worktree 目录。

    规则：在 git 仓库里就建 .worktrees/<名字>（目录已经存在就直接拿来
    复用——可能是上次会话留下的）；不在 git 仓库（或 git 建砸了）就退化用
    系统临时目录。降级语义沿用 CCAR5 轮定下的规矩。

    参数：
        name：worktree 名字（已提前洗干净，拼接安全）。

    返回：_SessionWorktree 登记信息（含路径、分支、类型、是否复用）。
    """
    base = Path(get_workspace_cwd())

    if is_git_repo(base):
        try:
            repo_root = get_repo_root(base) or base
            wt_dir = repo_root / ".worktrees" / name
            if wt_dir.exists():
                # 复用：目录已存在（多半是上次会话留下的）。不知道对应哪个
                # 分支，所以 branch 记 None——exit 时不会去删分支
                return _SessionWorktree(wt_dir, None, "git", reused=True,
                                        repo_root=repo_root)

            short_id = uuid.uuid4().hex[:8]
            branch = f"omnimate/{name}/{short_id}"
            result = _run_git(["worktree", "add", "-b", branch, str(wt_dir)],
                              repo_root, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"git worktree add 失败: {result.stderr.strip()}")
            logger.info("已创建会话 worktree: %s（分支 %s）", wt_dir, branch)
            return _SessionWorktree(wt_dir, branch, "git", reused=False,
                                    repo_root=repo_root)
        except Exception as e:
            logger.warning("git worktree 创建失败，降级到临时目录: %s", e)

    # 非 git（或 git 建砸了降级）：系统临时目录（沿用 CCAR5 _create_temp_workspace 的语义）
    tmp = Path(tempfile.mkdtemp(prefix=f"omnimate-wt-{name}-"))
    return _SessionWorktree(tmp, None, "temp", reused=False)


async def _handle_worktree_enter(args: dict, **dispatch_kwargs) -> str:
    """worktree_enter 的实现：把主对话搬进一个 worktree（建/复用 + 切目录 + 发通知）。

    为什么必须是 async def（历史踩坑，CCAR12 Task 6 修复，评审定级 Critical）：
    registry.dispatch 对同步 handler 会走 asyncio.to_thread——那会把当前
    context **拷贝**一份到工作线程。`set_session_workspace_cwd` 底层的
    ContextVar 一 set，改的只是拷贝，主循环毫无感知（enter 看似成功实则
    静默失效）；更糟的是 exit 的 clear 在拷贝 context 里去 reset 主
    context 的 token → 直接 ValueError → 永远 tool_exception（状态机死锁）。
    async handler 由 dispatch 直接 await——同一个任务同一个 context，
    set 生效、token 也能正常 reset。函数内部只是本地 git 命令这种同步
    IO，在 async 里直接跑就行，不必专门包线程。

    参数：
        args：LLM 传的参数——name（worktree 名，可选，不传自动生成）。
        dispatch_kwargs：系统上下文（agent_ref、session_id）。

    返回：JSON 字符串，含新目录路径、是否复用、类型、分支名；
    子代理调用、已在 worktree 中、创建失败各有对应错误。
    """
    global _session_worktree

    # 历史踩坑（CCAR13 A1，CCAR12 终审 follow-up）：子代理（spawn_depth>0，
    # 即派生层级大于 0 的分身）不许切会话级 worktree——ContextVar 是整个
    # 进程共享的，子代理一 enter 就把主对话的目录劫持了，而且子代理结束时
    # 未必会 exit（会话级语义被滥用）。
    # isinstance 守卫的缘故：agent_ref 可能是测试用的 MagicMock，没有
    # spawn_depth 属性时 getattr 会返回自动生成的假属性（不是 int）——
    # 按 0 处理，不误伤测试。
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
    while ".." in name:  # 单个 "." 是合法字符，但拼成 ".." 就是路径穿越，git 也会拒绝这种分支名
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
    """worktree_exit 的实现：把主对话搬回原目录，按需清理 worktree。

    为什么 async def：理由同 _handle_worktree_enter——clear 底层的
    ContextVar reset 必须和当初的 set 在同一个 context 里执行，否则
    在 to_thread 的拷贝 context 里 reset 必炸。

    参数：
        args：LLM 传的参数——keep（默认 True 保留 worktree 给人看；
        False 才动清理，且有改动仍会保留并提示）。
        dispatch_kwargs：系统上下文。

    返回：JSON 字符串，说明保留了还是删了（带原因/提示）；
    本来就不在 worktree 里则返回错误。
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

    # keep=False 才清理，而且是聪明清理：有改动保留 + 给提示，干净才删
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
    # git 模式新建的分支要连着一起删。分工规矩（CCAR5-G 定的）：
    # cleanup_worktree_smart 只删目录不删分支——分支删除由知道分支名的
    # 调用方负责，正好这里知道。
    # 仓库位置用 enter 时记下的 wt.repo_root——此刻 cwd 已经搬回原处，
    # 再靠「当前目录」反查仓库就查错地方了（历史出处：Minor 2 修复）
    if wt.branch and wt.workspace_type == "git":
        repo_root = wt.repo_root or get_repo_root(Path.cwd())
        if repo_root is not None:
            try:
                _run_git(["branch", "-D", wt.branch], repo_root)
            except Exception as e:
                logger.debug("删除分支 %s 失败: %s", wt.branch, e)

    return json.dumps(
        {"path": str(wt.path), "kept": False, "cleaned": cleaned},
        ensure_ascii=False,
    )


# 模块级注册（import 这个模块时自动把两个工具登记进中央注册表，LLM 才看得见）
# is_async=True 的原因：handler 是 async def——dispatch 会直接 await（同一个
# 任务同一个 context），不走 to_thread 的 context 拷贝。会话目录的
# set/reset 必须在主循环 context 里执行，否则 enter 静默失效 + exit 的
# reset 跨 context 直接炸 ValueError（详见 handler docstring 里的历史踩坑）
registry.register(
    name="worktree_enter",
    toolset="core",
    schema=WORKTREE_ENTER_SCHEMA,
    handler=_handle_worktree_enter,
    emoji="🌳",
    is_async=True,
    isConcurrencySafe=False,  # 改的是会话级全局目录 + 建 worktree，必须排队执行
)
registry.register(
    name="worktree_exit",
    toolset="core",
    schema=WORKTREE_EXIT_SCHEMA,
    handler=_handle_worktree_exit,
    emoji="🌳",
    is_async=True,
    isConcurrencySafe=False,  # 清目录状态 + 可能删 worktree，必须排队执行
)
