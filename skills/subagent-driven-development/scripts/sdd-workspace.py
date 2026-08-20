#!/usr/bin/env python3
"""找到（必要时创建）SDD 工作目录，并把它的绝对路径打印出来。

SDD = Subagent-Driven Development（子代理驱动开发）流程的工作目录，
用来放任务简报、复审包等中间产物。本脚本是 superpowers 项目里
sdd-workspace（bash 版）的 Python 翻版。

设计取舍：工作目录放在仓库工作区内（不在 .git/ 里），并在里面写一个
内容为 "*" 的 .gitignore 把自己整个忽略掉——这样 git status 保持干净，
中间产物不会被误提交。
"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """程序入口：定位 git 仓库根目录 → 确保 .superpowers/sdd 存在并自忽略 → 打印路径。

    返回：进程退出码，0 表示成功（git 命令失败会直接抛异常，算失败）。
    """
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True,
        encoding="utf-8", errors="replace",
    ).strip()
    workdir = Path(root) / ".superpowers" / "sdd"
    workdir.mkdir(parents=True, exist_ok=True)
    # 用 "*" 忽略整个目录自身，保持 git status 干净
    (workdir / ".gitignore").write_text("*\n", encoding="utf-8")
    print(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
