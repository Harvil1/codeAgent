"""工作区隔离：为并行任务提供独立工作目录。

git 仓库：用 git worktree 创建（共享历史，独立分支和文件）
非 git：创建临时目录

借鉴 Claude Code 的 worktree-task-isolation 机制，用于多子代理并行
时不互相干扰文件。

用法：
    path, cleanup = create_isolated_workspace(name="task-x")
    try:
        os.chdir(path)
        # 在隔离工作区执行任务
    finally:
        cleanup()
"""

import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)


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
) -> Tuple[Path, Callable]:
    """创建隔离工作区。

    返回 (workspace_path, cleanup_fn)。
    cleanup_fn(keep=False) 清理工作区；keep=True 保留供后续查看。

    git 仓库：用 git worktree 创建独立分支和工作目录。
    非 git：创建空临时目录（不拷贝文件）。
    """
    base = Path(base_path) if base_path else Path.cwd()

    if is_git_repo(base):
        try:
            return _create_git_worktree(base, name)
        except Exception as e:
            logger.warning("git worktree 创建失败，降级到临时目录: %s", e)

    return _create_temp_workspace(name)


def _create_git_worktree(base: Path, name: str) -> Tuple[Path, Callable]:
    """用 git worktree 创建独立工作区。"""
    repo_root = get_repo_root(base) or base
    short_id = uuid.uuid4().hex[:8]
    branch = f"harvil/{name}/{short_id}"

    # worktree 放在 .harvil-worktrees/ 下（gitignore 它）
    worktree_dir = repo_root.parent / ".harvil-worktrees" / f"{name}-{short_id}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_dir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add 失败: {result.stderr}")

    logger.info("已创建 git worktree: %s（分支 %s）", worktree_dir, branch)

    def cleanup(keep: bool = False):
        if keep:
            logger.info("保留 worktree: %s", worktree_dir)
            return
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

    return worktree_dir, cleanup


def _create_temp_workspace(name: str) -> Tuple[Path, Callable]:
    """非 git 仓库时创建空临时目录。"""
    prefix = f"harvil-{name}-"
    tmp = Path(tempfile.mkdtemp(prefix=prefix))

    def cleanup(keep: bool = False):
        if keep:
            return
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, cleanup


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
