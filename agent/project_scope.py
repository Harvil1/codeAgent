"""agent/project_scope.py — 记忆项目区键计算（CCAR9，对标 CCB findCanonicalGitRoot）。

分层隔离（用户决策）：user/feedback 类全局共享，project/reference 类
按项目分区。项目键 = canonical git root（worktree 归一），非 git 退 cwd。
"""
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

_git_timeout = 2.0
_key_cache: Dict[str, str] = {}


def _sanitize(path_str: str) -> str:
    """把路径里的非安全字符（分隔符/冒号等）替换成 -，做目录名。"""
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", path_str)


def _git_toplevel(base: str) -> Optional[str]:
    """base 所在 git repo 的 root；非 git / 失败返回 None（fail-open）。

    canonical root：用 `git rev-parse --git-common-dir` 取「公共 git 目录」
    （worktree 场景返回主 repo 的 .git，非 worktree 返回当前 .git），
    再取父目录得到主 repo root。这样同 repo 的不同 worktree 共享同一项目键，
    对齐 CCB findCanonicalGitRoot 语义。

    单用 `--show-toplevel` 在 worktree 里返回的是 worktree 自己的路径
    （非归一），所以不能直接用。

    注意：`--git-common-dir` 可能返回相对路径（如 `.git`），要基于 base
    解析（而不是主进程 cwd，否则会把相对路径拼到错误的目录上）。
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=base, capture_output=True, text=True,
            timeout=_git_timeout, encoding="utf-8", errors="replace",
        )
        if r.returncode == 0 and r.stdout.strip():
            raw = r.stdout.strip()
            common_dir = Path(raw)
            if not common_dir.is_absolute():
                # 相对路径：相对 subprocess 的 cwd（base），不是主进程 cwd
                common_dir = Path(base) / common_dir
            common_dir = common_dir.resolve()
            # common_dir 通常是 <root>/.git；父目录就是 repo root
            return str(common_dir.parent)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_project_memory_key(base: Optional[str] = None) -> str:
    """计算当前项目分区键（进程级缓存，同 base 只算一次）。

    base=None → get_workspace_cwd()（子代理 worktree 场景正确）。
    git 仓库 → canonical root（worktree 里跑 rev-parse 返回主 repo root，
    同 repo 共享项目区）；非 git → base 本身。
    """
    if base is None:
        from agent.workspace_context import get_workspace_cwd
        base = get_workspace_cwd()
    cached = _key_cache.get(base)
    if cached is not None:
        return cached
    root = _git_toplevel(base) or base
    key = _sanitize(root)
    _key_cache[base] = key
    return key


def get_project_memory_dir(agent_home, base: Optional[str] = None) -> Path:
    """项目区目录（不 mkdir，调用方按需创建）。

    返回 <agent_home>/.memory/projects/<sanitized_key>。
    """
    return (
        Path(agent_home) / ".memory" / "projects"
        / get_project_memory_key(base)
    )
