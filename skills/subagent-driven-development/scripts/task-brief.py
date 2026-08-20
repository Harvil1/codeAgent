#!/usr/bin/env python3
"""从计划文件里抽出某一个任务的完整文字，单独存成一份「任务简报」文件。

用法：task-brief PLAN_FILE TASK_NUMBER [OUTFILE]
      （计划文件 任务编号 [输出文件，可选]）
不指定输出文件时，默认写到 <仓库根>/.superpowers/sdd/task-<N>-brief.md。
本脚本是 superpowers 项目里 task-brief（bash 版）的 Python 翻版。

设计取舍：识别任务标题时会跳过代码围栏（``` 包住的代码块）里的行——
代码示例里写的 "# Task 3" 不是真标题，不能当成任务边界。
"""
import re
import subprocess
import sys
from pathlib import Path

# 匹配「# 到 ###### 任意级别标题 + Task + 数字」；数字后面必须不是数字
# （避免 "Task 1" 误匹配到 "Task 12" 的前半段）
_TASK_HEADING = re.compile(r"^#{1,6}\s+Task\s+(\d+)(?:[^0-9]|$)", re.IGNORECASE)


def _sdd_dir() -> Path:
    """调用 sdd-workspace.py 拿到 SDD 工作目录的路径。

    返回：工作目录的 Path（目录本身由那个脚本负责创建，这里只取路径）。
    """
    out = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve().parent / "sdd-workspace.py")],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    return Path(out)


def main() -> int:
    """程序入口：解析命令行参数 → 从计划文件里截取目标任务 → 写出简报文件。

    参数（从 sys.argv 取）：
        argv[1]  计划文件路径（Markdown，里面有 "## Task 1"、"## Task 2" 等标题）
        argv[2]  要抽取的任务编号
        argv[3]  可选的输出文件路径，不给就用默认路径

    返回：进程退出码——0 成功；2 用法错误/计划文件不存在；3 计划里找不到该任务。
    """
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
    # in_task：当前行是否属于目标任务；in_fence：是否在代码块里
    in_task, in_fence = False, False
    for ln in plan.read_text(encoding="utf-8").splitlines():
        if ln.startswith("```"):
            in_fence = not in_fence
        # 只在代码块之外识别任务标题（代码里的 "#" 是注释不是标题）
        if not in_fence:
            m = _TASK_HEADING.match(ln)
            if m:
                in_task = int(m.group(1)) == target
        if in_task:
            buf.append(ln)
    out.write_text("\n".join(buf) + "\n", encoding="utf-8")
    # 一行都没收集到 = 计划里没有这个任务的标题
    if not buf:
        print(f"task {n} not found in {plan} (no heading matching 'Task {n}')", file=sys.stderr)
        return 3
    print(f"wrote {out}: {len(buf)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
