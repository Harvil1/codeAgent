"""curator CLI：harvilagent curator <verb>

verbs：
  status    - 查看 curator 状态
  run       - 手动触发审查（--dry-run 预览）
  pause     - 暂停自动触发
  resume    - 恢复自动触发
  pin       - pin 一个技能（免疫自动转换）
  unpin     - 取消 pin
  restore   - 从 .archive/ 恢复技能
  memory    - Memory Curator 子命令
              status / run [--dry-run] [--no-llm] / pause / resume
"""

import sys
from pathlib import Path

from agent.curator import (
    load_state, save_state, run_curator_review,
)
from tools.skill_usage import load_usage, set_pinned, restore_skill


def _cmd_memory(args):
    """memory 子命令:curator memory status|run [--dry-run] [--no-llm]|pause|resume

    第 1 阶段跑 apply_automatic_transitions(纯状态转换),不依赖 LLM。
    第 2 阶段调用 run_memory_review(主模型 LLM 合并 + 矛盾检测),
    --no-llm 跳过第 2 阶段。状态写到 <agent_home>/.memory/.curator_state.json。
    """
    import datetime
    from constants import get_agent_home
    from agent.memory_curator import (
        apply_automatic_transitions,
        load_memory_curator_state,
        save_memory_curator_state,
    )

    memory_dir = get_agent_home() / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    sub = args[0] if args else "status"

    if sub == "status":
        state = load_memory_curator_state(memory_dir)
        print("=== Memory Curator 状态 ===")
        print(f"上次运行: {state.get('last_run_at', '从未')}")
        print(f"上次总结: {state.get('last_run_summary', '无')}")
        print(f"已暂停: {'是' if state.get('paused') else '否'}")
        return

    if sub == "run":
        dry_run = "--dry-run" in args
        no_llm = "--no-llm" in args  # 跳过 LLM 阶段
        print(f"{'[DRY RUN] ' if dry_run else ''}运行 Memory Curator...")

        # 第 1 阶段
        if not dry_run:
            counts = apply_automatic_transitions(memory_dir)
        else:
            # dry-run 不改记忆文件,只预览(与 skill curator 一致)
            counts = {
                "checked": 0, "marked_stale": 0,
                "archived": 0, "reactivated": 0,
            }
        print(f"\n第 1 阶段转换: {counts}")

        # 第 2 阶段(可选 LLM review)
        review_summary = "skipped"
        if not dry_run and not no_llm:
            from agent.memory_curator import run_memory_review
            from cli import RuntimeContext
            try:
                rt = RuntimeContext()
                factory = rt._make_memory_review_agent_factory()
                report = run_memory_review(memory_dir, agent_factory=factory)
                review_summary = (
                    f"reviewed={report['buckets_reviewed']}, "
                    f"actions={report['executed_actions']}, "
                    f"errors={report['errors']}"
                )
                print(f"第 2 阶段 LLM: {review_summary}")
            except Exception as e:
                review_summary = f"failed: {e}"
                print(f"第 2 阶段失败: {e}")

        # 写状态
        if not dry_run:
            state = load_memory_curator_state(memory_dir)
            state["last_run_at"] = datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat()
            state["last_run_summary"] = (
                f"第 1 阶段: {counts}; 第 2 阶段: {review_summary}"
            )
            state["paused"] = state.get("paused", False)
            save_memory_curator_state(memory_dir, state)
        return

    if sub == "pause":
        state = load_memory_curator_state(memory_dir)
        state["paused"] = True
        save_memory_curator_state(memory_dir, state)
        print("Memory Curator 已暂停")
        return

    if sub == "resume":
        state = load_memory_curator_state(memory_dir)
        state["paused"] = False
        save_memory_curator_state(memory_dir, state)
        print("Memory Curator 已恢复")
        return

    print(f"未知子命令: {sub}")
    print("可用: status, run [--dry-run], pause, resume")


def curator_cli(args: list, skills_dir: Path = None):
    """curator 子命令入口。"""
    if skills_dir is None:
        from constants import skills_dir as _sd
        skills_dir = _sd()

    if not args:
        _show_status(skills_dir)
        return

    verb = args[0]

    # memory 子命令走独立的 Memory Curator 管线(第 1 阶段 dry-run/实跑)
    if verb == "memory":
        _cmd_memory(args[1:])
        return

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


def main(args=None):
    """curator CLI 入口：python -m curator_cli <verb> ...

    args=None 时从 sys.argv 读取(生产入口);传入 list 时直接使用(测试入口)。
    """
    if args is None:
        args = sys.argv[1:]
    if not args:
        _show_status(None)
        return
    curator_cli(args)


if __name__ == "__main__":
    main()
