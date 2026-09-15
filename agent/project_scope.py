"""记忆项目区的分区键计算。

记忆（AI 沉淀的事实条目）分两类存放——user/feedback 类是用户全局的，
所有项目共享；project/reference 类按项目分家，各项目互相看不见。
本文件负责算"这是哪个项目的"那把钥匙。

项目键怎么定：取项目所在 git 仓库的"正身根目录"（canonical root——
worktree 也要归一到主仓库，避免同一仓库分出两个项目区）；
不是 git 仓库就用当前目录本身。
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

_git_timeout = 2.0
_key_cache: Dict[str, str] = {}


def _sanitize(path_str: str) -> str:
    """把路径字符串变成安全的目录名：分隔符、冒号等危险字符一律换成 -。

    参数：
    - path_str：原始路径字符串

    返回：只含字母数字和 _ . - 的字符串。
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", path_str)


def _git_toplevel(base: str) -> Optional[str]:
    """返回 base 所在 git 仓库的正身根目录；不是 git / 出错返回 None（fail-open）。

    为什么不用 `--show-toplevel`：在 worktree 里它返回的是 worktree 自己
    的路径（没有归一），同仓库的不同 worktree 会算出不同的项目键——所以
    不能直接用。

    正确做法：用
    `git rev-parse --git-common-dir` 取"公共 git 目录"（worktree 场景返回
    主仓库的 .git，普通场景返回当前 .git），再取它的父目录就是主仓库根。
    这样同仓库的所有 worktree 共享同一个项目键。

    参数：
    - base：要查询的目录

    返回：主仓库根路径字符串；失败 None。

    注意：`--git-common-dir` 可能返回相对路径（如 `.git`），必须
    基于 base 解析——用主进程 cwd 去拼会拼到错误的目录上。
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
                # 相对路径是相对 subprocess 的 cwd（即 base）说的，不是主进程 cwd
                common_dir = Path(base) / common_dir
            common_dir = common_dir.resolve()
            # common_dir 通常是 <仓库根>/.git，父目录就是仓库根
            return str(common_dir.parent)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_project_memory_key(base: Optional[str] = None) -> str:
    """算出当前项目的分区键（进程级缓存，同一 base 只算一次）。

    参数：
    - base：基准目录；传 None 时自动取 get_workspace_cwd()
      （工作区感知的 cwd——子代理在 worktree 里跑也能拿对）

    返回： sanitize 后的项目键字符串。规则：git 仓库 → canonical root
    （worktree 归一到主仓库，同仓库共享项目区）；非 git → base 本身。
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
    """项目区目录路径（只算路径，不创建目录——要不要建由调用方决定）。

    参数：
    - agent_home：agent 数据根目录（如 ~/.codeAgent）
    - base：基准目录（None 时同 get_project_memory_key 的规则）

    返回：<agent_home>/.memory/projects/<分区键> 的 Path。

    Windows 大小写归一：盘符大小写不同的两种进入方式（D:\\Foo 和 d:\\foo）
    会算出两个键、裂成两个互不可见的项目区。这里在 win32 上把目录名折叠
    成小写；若已存在大小写不同的老目录（历史遗留分区），沿用老目录——
    读写不分家，比硬迁移安全。
    """
    key = get_project_memory_key(base)
    dir_path = Path(agent_home) / ".memory" / "projects" / key
    if sys.platform == "win32":
        folded = key.lower()
        if folded != key:
            projects_root = dir_path.parent
            folded_dir = projects_root / folded
            if folded_dir.exists():
                return folded_dir
            try:
                for child in projects_root.iterdir():
                    if child.is_dir() and child.name.lower() == folded:
                        return child  # 大小写不同的老目录：沿用，防读写分家
            except OSError:
                pass
            return folded_dir
    return dir_path
