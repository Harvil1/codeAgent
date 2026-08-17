#!/usr/bin/env python3
"""Generate a review package (commit list, stat, full diff) to a file.

Usage: review-package BASE HEAD [OUTFILE]
Default OUTFILE: <repo-root>/.superpowers/sdd/review-<base7>..<head7>.diff
Mirror of superpowers scripts/review-package (bash).
"""
import subprocess
import sys
from pathlib import Path


def _git(args: list[str]) -> str:
    # Windows 上 text=True 默认用 GBK,git 输出含 UTF-8 中文会崩 → 显式 utf-8
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8", errors="replace"
    )


def _ref_ok(ref: str) -> bool:
    try:
        _git(["rev-parse", "--verify", "--quiet", ref])
        return True
    except subprocess.CalledProcessError:
        return False


def _sdd_dir() -> Path:
    out = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve().parent / "sdd-workspace.py")],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    return Path(out)


def main() -> int:
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
