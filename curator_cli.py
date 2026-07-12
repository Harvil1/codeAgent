"""curator CLI：harvilagent curator <verb>

verbs：
  status    - 查看 curator 状态
  run       - 手动触发审查（--dry-run 预览）
  pause     - 暂停自动触发
  resume    - 恢复自动触发
  pin       - pin 一个技能（免疫自动转换）
  unpin     - 取消 pin
  restore   - 从 .archive/ 恢复技能
"""

import sys
from pathlib import Path

from agent.curator import (
    load_state, save_state, run_curator_review,
)
from tools.skill_usage import load_usage, set_pinned, restore_skill


def curator_cli(args: list, skills_dir: Path = None):
    """curator 子命令入口。"""
    if skills_dir is None:
        from constants import skills_dir as _sd
        skills_dir = _sd()

    if not args:
        _show_status(skills_dir)
        return

    verb = args[0]

    if verb == "status":
        _show_status(skills_dir)

    elif verb == "run":
        dry_run = "--dry-run" in args
        _run_curator(skills_dir, dry_run=dry_run)

    elif verb == "pause":
        _pause_curator(skills_dir)

    elif verb == "resume":
        _resume_curator(skills_dir)

    elif verb == "pin":
        if len(args) < 2:
            print("用法: curator pin <skill-name>")
            return
        set_pinned(skills_dir, args[1], True)
        print(f"已 pin: {args[1]}")

    elif verb == "unpin":
        if len(args) < 2:
            print("用法: curator unpin <skill-name>")
            return
        set_pinned(skills_dir, args[1], False)
        print(f"已 unpin: {args[1]}")

    elif verb == "restore":
        if len(args) < 2:
            print("用法: curator restore <skill-name>")
            return
        ok, msg = restore_skill(skills_dir, args[1])
        print(msg)

    else:
        print(f"未知命令: {verb}")
        print("可用命令: status, run, pause, resume, pin, unpin, restore")


def _show_status(skills_dir: Path):
    """显示 curator 状态。"""
    state = load_state(skills_dir)
    usage = load_usage(skills_dir)

    agent_skills = [
        (n, r) for n, r in usage.items()
        if r.get("created_by") == "agent"
    ]

    print("=== Curator 状态 ===")
    print(f"上次运行: {state.get('last_run_at', '从未')}")
    print(f"上次总结: {state.get('last_run_summary', '无')}")
    print(f"已暂停: {'是' if state.get('paused') else '否'}")
    print(f"\n管理的技能数: {len(agent_skills)}")

    by_state = {"active": 0, "stale": 0, "archived": 0}
    for _, rec in agent_skills:
        s = rec.get("state", "active")
        by_state[s] = by_state.get(s, 0) + 1

    for state_name, count in by_state.items():
        print(f"  {state_name}: {count}")


def _run_curator(skills_dir: Path, dry_run: bool = False):
    """手动触发 curator 审查。"""
    print(f"{'[DRY RUN] ' if dry_run else ''}运行 curator...")
    report = run_curator_review(skills_dir, dry_run=dry_run)
    print(f"\n转换计数: {report['transitions']}")
    print(f"合并/清理: {report['consolidations']}")
    print(f"耗时: {report['duration_seconds']:.1f}s")


def _pause_curator(skills_dir: Path):
    """暂停 curator。"""
    state = load_state(skills_dir)
    state["paused"] = True
    save_state(skills_dir, state)
    print("curator 已暂停")


def _resume_curator(skills_dir: Path):
    """恢复 curator。"""
    state = load_state(skills_dir)
    state["paused"] = False
    save_state(skills_dir, state)
    print("curator 已恢复")


def main():
    """curator CLI 入口：python -m curator_cli <verb> ..."""
    if len(sys.argv) < 2:
        _show_status(None)
        return
    curator_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
