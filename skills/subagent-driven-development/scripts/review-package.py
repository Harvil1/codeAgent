#!/usr/bin/env python3
"""生成一份「复审包」文件：包含提交列表、改动统计和完整 diff。

用法：review-package BASE HEAD [OUTFILE]
      （起点提交 终点提交 [输出文件，可选]）
不指定输出文件时，默认写到 <仓库根>/.superpowers/sdd/review-<短hash>..<短hash>.diff。
本脚本是 superpowers 项目里 review-package（bash 版）的 Python 翻版。
给代码复审的子代理看：一个文件里就能看到这批改动的一切。
"""
import subprocess
import sys
from pathlib import Path


def _git(args: list[str]) -> str:
    """跑一条 git 命令并把输出按文本返回。

    参数：
        args  git 子命令及参数列表，如 ["log", "--oneline"]

    返回：命令的标准输出文本。
    注意：Windows 上 text=True 默认按 GBK 解码，git 输出里的
    UTF-8 中文 commit message 会乱码甚至崩溃，所以必须显式指定 utf-8。
    """
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8", errors="replace"
    )


def _ref_ok(ref: str) -> bool:
    """检查一个 git 引用（分支名/commit hash/HEAD 等）是否真实存在。

    参数：
        ref  待检查的引用字符串

    返回：True 存在；False 不存在。
    """
    try:
        _git(["rev-parse", "--verify", "--quiet", ref])
        return True
    except subprocess.CalledProcessError:
        return False


def _sdd_dir() -> Path:
    """调用 sdd-workspace.py 拿到 SDD 工作目录的路径。

    返回：工作目录的 Path（目录由那个脚本负责创建，这里只取路径）。
    """
    out = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve().parent / "sdd-workspace.py")],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    return Path(out)


def main() -> int:
    """程序入口：校验两个提交引用 → 拼装提交列表/统计/diff → 写出复审包文件。

    参数（从 sys.argv 取）：
        argv[1]  起点提交（BASE），复审范围不包含它
        argv[2]  终点提交（HEAD），复审范围包含它
        argv[3]  可选的输出文件路径，不给就按短 hash 命名放 SDD 目录

    返回：进程退出码——0 成功；2 用法错误或引用不存在。
    """
    if not (3 <= len(sys.argv) <= 4):
        print("usage: review-package BASE HEAD [OUTFILE]", file=sys.stderr)
        return 2
    base, head = sys.argv[1], sys.argv[2]
    for ref, name in ((base, "BASE"), (head, "HEAD")):
        if not _ref_ok(ref):
            print(f"bad {name}: {ref}", file=sys.stderr)
            return 2
    if len(sys.argv) == 4:
        out = Path(sys.argv[3])
    else:
        short_base = _git(["rev-parse", "--short", base]).strip()
        short_head = _git(["rev-parse", "--short", head]).strip()
        out = _sdd_dir() / f"review-{short_base}..{short_head}.diff"
    content = "\n".join([
        f"# Review package: {base}..{head}",
        "",
        "## Commits",
        _git(["log", "--oneline", f"{base}..{head}"]).rstrip(),
        "",
        "## Files changed",
        _git(["diff", "--stat", f"{base}..{head}"]).rstrip(),
        "",
        "## Diff",
        _git(["diff", "-U10", f"{base}..{head}"]).rstrip(),
        "",
    ])
    out.write_text(content, encoding="utf-8")
    commits = _git(["rev-list", "--count", f"{base}..{head}"]).strip()
    print(f"wrote {out}: {commits} commit(s), {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
