"""工作区隔离：给并行的活儿各开一个互不打扰的独立工作目录。

这个文件是干嘛的：当多个子代理（主对话派出去帮忙干活的分身）同时干活时，
如果都挤在同一个目录里改文件，会互相把对方的改动踩掉。本模块提供
「一人一间屋」的 worktree 隔离机制。

两种情况：
    git 仓库：用 git worktree 建（worktree = git 自带的「一库多目录」功能，
    多个目录共享同一份历史，但各自有独立分支和文件）
    非 git 目录：只能建一个普通临时目录凑合用

create_isolated_workspace 还有一个可选参数
hook_registry（钩子登记本，用户配置的附加动作），在 worktree 创建/清理时
触发 WORKTREE_CREATE / WORKTREE_REMOVE 钩子。这两个钩子是通知型的，
给审计/清理脚本用的；钩子出异常也不影响 worktree 本身（fail-open：
附加功能挂了就挂了，主流程照走）。

用法示例：
    path, cleanup = create_isolated_workspace(name="task-x")
    try:
        # 注意：千万别用 os.chdir 切目录——它是整个进程共享的全局
        # 开关，并发的子代理会互相踩对方的当前目录。
        # 要用 workspace_cwd_context（contextvars.ContextVar 实现，
        # 线程之间互相看不见对方的值，天然隔离）
        from agent.workspace_context import workspace_cwd_context
        with workspace_cwd_context(str(path)):
            ...  # 在隔离工作区执行任务
    finally:
        cleanup()
"""

import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run_git(argv: List[str], cwd, timeout: float = 10) -> subprocess.CompletedProcess:
    """跑一条 git 命令的统一入口（本模块和 worktree_tool 的 git 操作都从这走）。

    Windows 命令行默认用 GBK 编码，git 输出里的中文会变乱码，所以强制
    utf-8 文本模式，遇到解码不了的坏字节就用替换符顶替，不让程序崩。

    参数：
        argv：git 子命令及参数（如 ["worktree", "add", ...]）。
        cwd：在哪个目录下执行。
        timeout：超时秒数，默认 10。

    返回：subprocess.CompletedProcess（含返回码、stdout、stderr）。
    """
    return subprocess.run(
        ["git", *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 变更检测
# ---------------------------------------------------------------------------

def has_worktree_changes(worktree_path: Path) -> bool:
    """看一眼 worktree 里有没有改动过的东西。

    判断方法分两种：
        git 目录：跑 git status --porcelain（机器友好格式，有输出 = 有改动）；
        非 git 目录：退化成看目录里有没有文件（临时工作区刚建时是空的，
        里面有东西就说明子代理写过）。

    设计原则（fail-open，宁可错保不可错删）：检查过程出任何异常都当作
    「有改动」处理——宁可多留一个目录，也不能把用户辛苦改的代码当垃圾删了。

    参数：
        worktree_path：worktree 的目录路径。

    返回：True = 有改动（或检查挂了拿不准）；False = 干干净净没东西。
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return False  # 目录都不存在了，自然没东西可清理

    # 先试 git status 这条路
    try:
        result = _run_git(["status", "--porcelain"], wt)
        if result.returncode == 0:
            return bool(result.stdout.strip())
        # 返回码非 0 多半说明这不是 git 仓库 → 换下一种判断方式
    except Exception as e:
        logger.debug("git status 不可用，fallback 到 listdir 检测: %s", e)

    # 非 git 目录的退路：里面有文件就当作有改动
    # （临时工作区刚建时是空的，子代理写过文件后里面才会有东西）
    try:
        return any(wt.iterdir())
    except Exception:
        return True  # fail-open


def cleanup_worktree_smart(worktree_path: Path, force: bool = False) -> bool:
    """聪明地清理 worktree：有改动就留着，没改动（或调用方强制）才删。

    「聪明」在哪：删之前先看目录里有没有没保存的工作成果——有就保留并记
    一条日志，绝不销毁用户可能要的东西；干净的目录才直接删。

    适用场景：拿到路径就能调的独立场景（手里没有创建时的上下文信息）。
    注意分工：**这个函数只管检测改动 + 删目录，不删 git 分支**——分支删除
    由 _create_git_worktree 返回的 cleanup 闭包负责（闭包里存着精确的分支
    名，而这里并不知道这个目录对应哪个分支，瞎删会误伤其他并发 worktree
    的分支）。

    参数：
        worktree_path：worktree 的目录路径。
        force：True 时不看有没有改动，直接删（调用方明确要删时用）。

    返回：True = 已清理；False = 因为有改动而保留了。
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return True  # 目录本来就没有，视为「清理完成」

    if not force and has_worktree_changes(wt):
        logger.info("worktree %s 有改动，保留（cleanup_worktree_smart）", wt)
        return False

    # 如果这是 git 仓库的一部分，优先用 git 自己的 worktree 清理命令
    # 注意：这里不删分支——分支删除由 _create_git_worktree 的闭包负责（它才知道对应哪个分支）
    repo_root = get_repo_root(wt)
    if repo_root is not None:
        try:
            _run_git(["worktree", "remove", "--force", str(wt)], repo_root)
        except Exception as e:
            logger.warning("git worktree remove 失败: %s", e)

    # 兜底手段：不管 git 说什么，直接把目录删了（删不掉也不报错）
    shutil.rmtree(wt, ignore_errors=True)
    return not wt.exists()


# ---------------------------------------------------------------------------
# 事件流（worktree 一生大事的流水账：何时建、何时删、何时保留）
# ---------------------------------------------------------------------------

def _resolve_events_path(workspace_or_repo) -> Path:
    """算出事件流水账文件该放哪。

    worktree 的创建/清理事件统一记在仓库根目录下的
    .worktrees/.events.jsonl（一行一条 JSON，追加写入）。

    参数：
        workspace_or_repo：worktree 路径或仓库路径（两种都接受，
        会先找到所属的仓库根目录）。

    返回：事件文件完整路径；非 git 目录或找不到仓库根时返回 None
    （没地方记就不记）。
    """
    p = Path(workspace_or_repo) if workspace_or_repo else None
    repo_root = None
    if p is not None:
        # 传进来的要是本身就是仓库根目录（里面有 .git），就省得再反查了
        if (p / ".git").exists():
            repo_root = p
        else:
            repo_root = get_repo_root(p)
    if repo_root is None:
        return None
    events_dir = repo_root / ".worktrees"
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir / ".events.jsonl"


def _log_worktree_event(repo_root, event_type: str, payload: dict) -> None:
    """往流水账文件里追加一条 worktree 事件。

    定位：这是审计辅助功能，写不进去也不能拖垮 worktree 的正常创建/清理
    ——失败时只记一条 warning 日志，不往外抛异常。

    参数：
        repo_root：仓库根目录（据此定位事件文件；None 时啥也不做）。
        event_type：事件类型（如 "create.before" / "remove.after"）。
        payload：事件的具体内容（分支名、目录路径等）。

    返回：无。任何失败都静默吞掉（只留日志）。
    """
    events_file = _resolve_events_path(repo_root)
    if events_file is None:
        return
    record = {
        "event": event_type,
        "payload": payload,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("写入 worktree 事件失败: %s", e)


def is_git_repo(path=None) -> bool:
    """判断一个路径是不是在 git 仓库里面。

    参数：
        path：待查路径，不传就看当前目录。

    返回：True = 在 git 仓库里；False = 不在（或 git 命令执行出错，
    出错按「不是」处理，走临时目录那条退路）。
    """
    path = Path(path) if path else Path.cwd()
    try:
        result = _run_git(["rev-parse", "--is-inside-work-tree"], path, timeout=5)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def get_repo_root(path=None) -> Optional[Path]:
    """找到路径所属 git 仓库的根目录（.git 所在的那一层）。

    参数：
        path：从哪个路径往上找，不传就从当前目录找。

    返回：仓库根目录的 Path；不在 git 仓库里（或 git 出错）返回 None。
    """
    path = Path(path) if path else Path.cwd()
    try:
        result = _run_git(["rev-parse", "--show-toplevel"], path, timeout=5)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        logger.warning("异常被吞(fail-open)", exc_info=True)
    return None


def create_isolated_workspace(
    base_path=None,
    name: str = "workspace",
    *,
    hook_registry=None,
    session_id: str = "",
) -> Tuple[Path, Callable]:
    """创建一个隔离工作区（本模块的门面入口，外部都调它）。

    返回两个东西：(workspace_path 工作区路径, cleanup_fn 清理函数)。
    cleanup_fn(keep=False) 清掉工作区；keep=True 则原样留着给人看。

    怎么建看环境：
        git 仓库：用 git worktree 建独立分支 + 独立目录；
        非 git：建一个空临时目录（不复制任何文件，就是个空屋）。

    参数：
        base_path：以哪个目录为基准建（不传用当前目录）。
        name：工作区名字（会用在分支名和目录名里）。
        hook_registry：可选，钩子登记本。
        传了它，创建/清理时会触发 WORKTREE_CREATE / WORKTREE_REMOVE
        两个通知钩子（fail-open，钩子挂了不影响主流程）。
        session_id：可选，触发钩子时捎带给钩子的会话 ID。

    返回：(工作区路径, 清理函数) 二元组。git 建失败会自动降级成临时目录。
    """
    base = Path(base_path) if base_path else Path.cwd()

    if is_git_repo(base):
        try:
            return _create_git_worktree(
                base, name, hook_registry=hook_registry, session_id=session_id,
            )
        except Exception as e:
            logger.warning("git worktree 创建失败，降级到临时目录: %s", e)

    return _create_temp_workspace(
        name, hook_registry=hook_registry, session_id=session_id,
    )


def _create_git_worktree(base: Path, name: str, *,
                         hook_registry=None, session_id: str = "") -> Tuple[Path, Callable]:
    """用 git worktree 建独立工作区（内部分支：git 环境专用）。

    干的事：建一个 codeagent/<名字>/<8位短ID> 的新分支 + 对应的 worktree
    目录（放在仓库旁边的 .codeAgent-worktrees/ 下，不混进项目目录），并
    返回一个配套的清理闭包。

    hook_registry 不为 None 时，建好/删完会触发
    WORKTREE_CREATE / WORKTREE_REMOVE 通知钩子（fail-open）。

    参数：
        base：基准目录（在它所属的仓库里建 worktree）。
        name：工作区名（拼进分支名）。
        hook_registry：可选的钩子登记本。
        session_id：可选，捎带给钩子的会话 ID。

    返回：(worktree 目录路径, cleanup 清理闭包)。git worktree add 命令
    失败时抛 RuntimeError（外层会接住降级成临时目录）。
    """
    repo_root = get_repo_root(base) or base
    short_id = uuid.uuid4().hex[:8]
    branch = f"codeagent/{name}/{short_id}"

    # worktree 目录放在仓库隔壁的 .codeAgent-worktrees/ 下（记得 gitignore 它）
    worktree_dir = repo_root.parent / ".codeAgent-worktrees" / f"{name}-{short_id}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # 流水账：动手建之前先记一笔
    _log_worktree_event(repo_root, "create.before", {
        "branch": branch,
        "worktree_dir": str(worktree_dir),
        "name": name,
    })

    result = _run_git(["worktree", "add", "-b", branch, str(worktree_dir)],
                      repo_root, timeout=30)
    if result.returncode != 0:
        # 流水账：建砸了也记一笔（带错误信息）
        _log_worktree_event(repo_root, "create.failed", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
            "error": result.stderr.strip(),
        })
        raise RuntimeError(f"git worktree add 失败: {result.stderr}")

    logger.info("已创建 git worktree: %s（分支 %s）", worktree_dir, branch)

    # 流水账：建成了记一笔
    _log_worktree_event(repo_root, "create.after", {
        "branch": branch,
        "worktree_dir": str(worktree_dir),
        "name": name,
    })

    # 触发 WORKTREE_CREATE 钩子（fail-open）
    _fire_worktree_hook(hook_registry, "create", {
        "session_id": session_id,
        "path": str(worktree_dir),
        "branch": branch,
        "workspace_type": "git",
    })

    def cleanup(keep: bool = False, force: bool = False):
        """清理这个 worktree（闭包记住了分支名，所以能连分支一起删干净）。

        三种用法：
            keep=True：原样保留，什么都不删（用户想自己看看时用）；
            force=True：不管有没有改动，强制删；
            默认（都是 False）：聪明地清——有改动就保留，干净才删。

        参数：
            keep：True 表示保留给人看。
            force：True 表示强行删除。

        返回：True = 已清理；False = 保留了（因为有改动，或 keep=True）。
        """
        if keep:
            logger.info("保留 worktree: %s", worktree_dir)
            # 流水账：主动保留也记一笔
            _log_worktree_event(repo_root, "remove.keep", {
                "branch": branch,
                "worktree_dir": str(worktree_dir),
            })
            return False
        # 聪明清理：没带 force 时先看有没有改动，有就舍不得删
        if not force and has_worktree_changes(worktree_dir):
            logger.info("worktree %s 有改动，保留（智能清理）", worktree_dir)
            _log_worktree_event(repo_root, "remove.keep", {
                "branch": branch,
                "worktree_dir": str(worktree_dir),
                "reason": "has_changes",
            })
            return False
        # 流水账：真要删了，删之前记一笔
        _log_worktree_event(repo_root, "remove.before", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
        })
        try:
            _run_git(["worktree", "remove", "--force", str(worktree_dir)], repo_root)
            _run_git(["branch", "-D", branch], repo_root)
            logger.info("已清理 worktree: %s", worktree_dir)
        except Exception as e:
            logger.warning("清理 worktree 失败: %s", e)
        # 兜底手段：git 命令没删干净就直接删目录
        shutil.rmtree(worktree_dir, ignore_errors=True)
        # 流水账：删完了记一笔
        _log_worktree_event(repo_root, "remove.after", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
        })
        # 触发 WORKTREE_REMOVE 钩子（fail-open）
        _fire_worktree_hook(hook_registry, "remove", {
            "session_id": session_id,
            "path": str(worktree_dir),
            "branch": branch,
        })
        return True

    return worktree_dir, cleanup


def _create_temp_workspace(name: str, *,
                           hook_registry=None, session_id: str = "") -> Tuple[Path, Callable]:
    """非 git 仓库（或 git 路子走不通）时的退路：建一个空临时目录。

    参数：
        name：工作区名（用在临时目录名前缀里）。
        hook_registry：可选的钩子登记本。
        session_id：可选，捎带给钩子的会话 ID。

    返回：(临时目录路径, cleanup 清理闭包)。
    """
    prefix = f"codeAgent-{name}-"
    tmp = Path(tempfile.mkdtemp(prefix=prefix))

    # 流水账：临时目录也记一笔（repo_root 为 None 时写不进文件，只走钩子）
    _log_worktree_event(None, "create.after", {
        "worktree_dir": str(tmp),
        "name": name,
        "type": "temp",
    })

    # 触发 WORKTREE_CREATE 钩子（fail-open）
    _fire_worktree_hook(hook_registry, "create", {
        "session_id": session_id,
        "path": str(tmp),
        "workspace_type": "temp",
    })

    def cleanup(keep: bool = False, force: bool = False):
        """清理这个临时工作区。

        三种用法：
            keep=True：原样保留，什么都不删；
            force=True：不管有没有东西，强制删；
            默认：聪明地清——里面有文件就保留（可能是干活的成果），
            空的才删。

        参数：
            keep：True 表示保留给人看。
            force：True 表示强行删除。

        返回：True = 已清理；False = 保留了。
        """
        if keep:
            return False
        # 聪明清理：没带 force 时先看有没有文件
        if not force and has_worktree_changes(tmp):
            logger.info("temp workspace %s 有改动，保留（智能清理）", tmp)
            return False
        shutil.rmtree(tmp, ignore_errors=True)
        # 触发 WORKTREE_REMOVE 钩子（fail-open）
        _fire_worktree_hook(hook_registry, "remove", {
            "session_id": session_id,
            "path": str(tmp),
        })
        return True

    return tmp, cleanup


def _fire_worktree_hook(hook_registry, action: str, payload: dict) -> None:
    """触发 worktree 的创建/删除通知钩子。

    定位：纯通知性质的附加动作，worktree 的正事不能被它拖累——钩子抛出
    的任何异常都在这里吞掉，只留一条 warning 日志。

    参数：
        hook_registry：钩子登记本；None（没配置）就直接返回什么都不做。
        action："create"（刚建好）或 "remove"（刚删掉），决定触发哪个钩子。
        payload：捎给钩子的具体内容（路径、分支、会话 ID 等）。

    返回：无。
    """
    if hook_registry is None:
        return
    try:
        if action == "create":
            hook_registry.run_worktree_create(payload)
        elif action == "remove":
            hook_registry.run_worktree_remove(payload)
    except Exception as e:
        logger.warning("WORKTREE_%s hook 触发异常（fail-open）: %s",
                       action.upper(), e)


def list_worktrees(base_path=None) -> list:
    """列出当前 git 仓库里的所有 worktree（给用户查看/清理用）。

    参数：
        base_path：从哪个路径找仓库，不传用当前目录。

    返回：列表，每项一个 dict（含 path 路径、branch 分支名）；
    非 git 目录或命令失败返回空列表（查不到就当没有）。
    """
    base = Path(base_path) if base_path else Path.cwd()
    if not is_git_repo(base):
        return []
    try:
        result = _run_git(["worktree", "list", "--porcelain"], base, timeout=5)
        if result.returncode != 0:
            return []
        worktrees = []
        current = {}
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current:
                    worktrees.append(current)
                current = {"path": line.split(" ", 1)[1]}
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1]
            elif not line.strip() and current:
                worktrees.append(current)
                current = {}
        if current:
            worktrees.append(current)
        return worktrees
    except Exception:
        return []
