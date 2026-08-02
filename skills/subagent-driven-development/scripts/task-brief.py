#!/usr/bin/env python3
"""Extract one task's full text from a plan file into a brief file.

Usage: task-brief PLAN_FILE TASK_NUMBER [OUTFILE]
Default OUTFILE: <repo-root>/.superpowers/sdd/task-<N>-brief.md
Mirror of superpowers scripts/task-brief (bash). Code-fence headings are skipped.
"""
import re
import subprocess
import sys
from pathlib import Path

_TASK_HEADING = re.compile(r"^#{1,6}\s+Task\s+(\d+)(?:[^0-9]|$)", re.IGNORECASE)


def _sdd_dir() -> Path:
    out = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve().parent / "sdd-workspace.py")],
        text=True,
    ).strip()
    return Path(out)


def main() -> int:
    if not (3 <= len(sys.argv) <= 4):
        print("usage: task-brief PLAN_FILE TASK_NUMBER [OUTFILE]", file=sys.stderr)
        return 2
    plan, n = Path(sys.argv[1]), sys.argv[2]
    if not plan.is_file():
        print(f"no such plan file: {plan}", file=sys.stderr)
        return 2
    out = Path(sys.argv[3]) if len(sys.argv) == 4 else _sdd_dir() / f"task-{n}-brief.md"

    target = int(n)
    buf: list[str] = []
    in_task, in_fence = False, False
    for ln in plan.read_text(encoding="utf-8").splitlines():
        if ln.startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m = _TASK_HEADING.match(ln)
            if m:
                in_task = int(m.group(1)) == target
        if in_task:
            buf.append(ln)
    out.write_text("\n".join(buf) + "\n", encoding="utf-8")
    if not buf:
        print(f"task {n} not found in {plan} (no heading matching 'Task {n}')", file=sys.stderr)
        return 3
    print(f"wrote {out}: {len(buf)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
