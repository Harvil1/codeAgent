"""工作区隔离：为并行任务提供独立工作目录。

git 仓库：用 git worktree 创建（共享历史，独立分支和文件）
非 git：创建临时目录

借鉴 业界 的 worktree-task-isolation 机制，用于多子代理并行
时不互相干扰文件。

P3.4 新增：create_isolated_workspace 接受可选的 hook_registry 参数，
在 worktree 创建/清理时触发 WORKTREE_CREATE / WORKTREE_REMOVE hook
（通知型，审计/清理注册用，hook 异常 fail-open 不影响 worktree 主流程）。

用法：
    path, cleanup = create_isolated_workspace(name="task-x")
    try:
        # 注意：不要用 os.chdir（进程级全局，并发子代理会互相踩 cwd）
        # 用 workspace_cwd_context（contextvars.ContextVar，线程隔离）
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
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 事件流（worktree lifecycle 审计日志）
# ---------------------------------------------------------------------------

def _resolve_events_path(workspace_or_repo) -> Path:
    """解析事件文件路径。

    放在仓库根目录下的 .worktrees/.events.jsonl。
    非 git / 无法确定 repo root 时返回 None。
    """
    p = Path(workspace_or_repo) if workspace_or_repo else None
    repo_root = None
    if p is not None:
        # 如果传入的已经是 repo root（包含 .git），直接用
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
    """写入一条 worktree 事件到 jsonl 文件。

    失败时只 log warning，不抛（事件流是审计辅助，不影响主流程）。
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
    """检查路径是否在 git 仓库内。"""
    path = Path(path) if path else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def get_repo_root(path=None) -> Optional[Path]:
    """获取 git 仓库根目录。"""
    path = Path(path) if path else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def create_isolated_workspace(
    base_path=None,
    name: str = "workspace",
    *,
    hook_registry=None,
    session_id: str = "",
) -> Tuple[Path, Callable]:
    """创建隔离工作区。

    返回 (workspace_path, cleanup_fn)。
    cleanup_fn(keep=False) 清理工作区；keep=True 保留供后续查看。

    git 仓库：用 git worktree 创建独立分支和工作目录。
    非 git：创建空临时目录（不拷贝文件）。

    P3.4 新增可选参数：
        hook_registry: HookRegistry 实例。传入时在创建/清理时触发
                       WORKTREE_CREATE / WORKTREE_REMOVE 事件（fail-open）。
        session_id: 触发 hook 时透传的 session_id（可选）。
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
    """用 git worktree 创建独立工作区。

    P3.4: hook_registry 非 None 时触发 WORKTREE_CREATE / WORKTREE_REMOVE 事件（fail-open）。
    """
    repo_root = get_repo_root(base) or base
    short_id = uuid.uuid4().hex[:8]
    branch = f"omnimate/{name}/{short_id}"

    # worktree 放在 .omnimate-worktrees/ 下（gitignore 它）
    worktree_dir = repo_root.parent / ".omnimate-worktrees" / f"{name}-{short_id}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # 事件：create.before
    _log_worktree_event(repo_root, "create.before", {
        "branch": branch,
        "worktree_dir": str(worktree_dir),
        "name": name,
    })

    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_dir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        # 事件：create.failed
        _log_worktree_event(repo_root, "create.failed", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
            "error": result.stderr.strip(),
        })
        raise RuntimeError(f"git worktree add 失败: {result.stderr}")

    logger.info("已创建 git worktree: %s（分支 %s）", worktree_dir, branch)

    # 事件：create.after
    _log_worktree_event(repo_root, "create.after", {
        "branch": branch,
        "worktree_dir": str(worktree_dir),
        "name": name,
    })

    # P3.4: 触发 WORKTREE_CREATE hook（fail-open）
    _fire_worktree_hook(hook_registry, "create", {
        "session_id": session_id,
        "path": str(worktree_dir),
        "branch": branch,
        "workspace_type": "git",
    })

    def cleanup(keep: bool = False):
        if keep:
            logger.info("保留 worktree: %s", worktree_dir)
            # 事件：remove.keep
            _log_worktree_event(repo_root, "remove.keep", {
                "branch": branch,
                "worktree_dir": str(worktree_dir),
            })
            return
        # 事件：remove.before
        _log_worktree_event(repo_root, "remove.before", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
        })
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=str(repo_root),
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(repo_root),
                capture_output=True,
                timeout=10,
            )
            logger.info("已清理 worktree: %s", worktree_dir)
        except Exception as e:
            logger.debug("清理 worktree 失败: %s", e)
        # 兜底删除目录
        shutil.rmtree(worktree_dir, ignore_errors=True)
        # 事件：remove.after
        _log_worktree_event(repo_root, "remove.after", {
            "branch": branch,
            "worktree_dir": str(worktree_dir),
        })
        # P3.4: 触发 WORKTREE_REMOVE hook（fail-open）
        _fire_worktree_hook(hook_registry, "remove", {
            "session_id": session_id,
            "path": str(worktree_dir),
            "branch": branch,
        })

    return worktree_dir, cleanup


def _create_temp_workspace(name: str, *,
                           hook_registry=None, session_id: str = "") -> Tuple[Path, Callable]:
    """非 git 仓库时创建空临时目录。"""
    prefix = f"omnimate-{name}-"
    tmp = Path(tempfile.mkdtemp(prefix=prefix))

    # 事件：create.after（temp workspace 也记录，但 repo_root 为 None 时不写文件）
    _log_worktree_event(None, "create.after", {
        "worktree_dir": str(tmp),
        "name": name,
        "type": "temp",
    })

    # P3.4: 触发 WORKTREE_CREATE hook（fail-open）
    _fire_worktree_hook(hook_registry, "create", {
        "session_id": session_id,
        "path": str(tmp),
        "workspace_type": "temp",
    })

    def cleanup(keep: bool = False):
        if keep:
            return
        shutil.rmtree(tmp, ignore_errors=True)
        # P3.4: 触发 WORKTREE_REMOVE hook（fail-open）
        _fire_worktree_hook(hook_registry, "remove", {
            "session_id": session_id,
            "path": str(tmp),
        })

    return tmp, cleanup


def _fire_worktree_hook(hook_registry, action: str, payload: dict) -> None:
    """P3.4: 触发 WORKTREE_CREATE / WORKTREE_REMOVE hook（fail-open）。

    action: "create" | "remove"
    hook_registry 为 None 时无操作。任何异常都吞掉（worktree 主流程不能被 hook 打断）。
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
    """列出当前 git 仓库的所有 worktree。"""
    base = Path(base_path) if base_path else Path.cwd()
    if not is_git_repo(base):
        return []
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=5,
        )
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
