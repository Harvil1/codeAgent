#!/usr/bin/env python3
"""Resolve/ensure the SDD workspace dir and print its absolute path.

Mirror of superpowers scripts/sdd-workspace (bash). Workspace lives in the
working tree (not under .git/), self-gitignored so `git status` stays clean.
"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    workdir = Path(root) / ".superpowers" / "sdd"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / ".gitignore").write_text("*\n", encoding="utf-8")
    print(workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
