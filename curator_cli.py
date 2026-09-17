"""Curator（维护工人）的命令行入口：python -m curator_cli <命令>。

Curator 是后台维护程序，负责整理技能库和记忆库（归档旧技能、合并重复
记忆等）。平时它按周期自动跑，这个 CLI 让你能手动查看/触发/暂停它。
位于项目顶层，供命令行和测试直接调用。

命令（verb）一览：
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
    """处理 memory 子命令：curator memory status|run [--dry-run|--no-llm]|pause|resume。

    Memory Curator（记忆维护工人）分两个阶段干活——
    第 1 阶段跑 apply_automatic_transitions：按固定规则做状态转换
    （比如长期没用的记忆标记为过期、归档），纯机械逻辑，不花钱调 LLM。
    第 2 阶段调用 run_memory_review：让主模型做记忆合并和矛盾检测，
    这一步要花 token；加 --no-llm 可以跳过它只跑第 1 阶段。
    运行状态记在 <agent_home>/.memory/.curator_state.json 里。

    参数：
        args  子命令及参数列表，如 ["run", "--dry-run"]；空列表默认当 status
    """
    import datetime
    from constants import get_codeagent_home
    from agent.memory_curator import (
        apply_automatic_transitions,
        load_memory_curator_state,
        save_memory_curator_state,
    )

    memory_dir = get_codeagent_home() / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    # 本进程内只建一个 MemoryStore 全程共用（构造带副作用：老格式搬家 +
    # 索引重建），两个阶段都传它——各阶段各自现建的话，两把实例锁互不
    # 互斥，1/2 阶段并发写同一 topic.jsonl 会互相覆盖
    from agent.memory_store import MemoryStore
    _store = MemoryStore(codeagent_home=get_codeagent_home())
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
        no_llm = "--no-llm" in args  # 加了这个开关就不跑要花 token 的 LLM 阶段
        print(f"{'[DRY RUN] ' if dry_run else ''}运行 Memory Curator...")

        # 第 1 阶段：纯规则状态转换，不依赖 LLM
        if not dry_run:
            counts = apply_automatic_transitions(memory_dir, store=_store)
        else:
            # dry-run 只预览不动真格：不修改任何记忆文件（和技能 curator 的语义保持一致）
            counts = {
                "checked": 0, "marked_stale": 0,
                "archived": 0, "reactivated": 0,
            }
        print(f"\n第 1 阶段转换: {counts}")

        # 第 2 阶段：可选的 LLM 审查（合并相似记忆、找矛盾说法）
        review_summary = "skipped"
        if not dry_run and not no_llm:
            from agent.memory_curator import run_memory_review
            from cli import RuntimeContext
            try:
                rt = RuntimeContext()
                factory = rt._make_memory_review_agent_factory()
                report = run_memory_review(
                    memory_dir, agent_factory=factory, store=_store,
                )
                review_summary = (
                    f"reviewed={report['buckets_reviewed']}, "
                    f"actions={report['executed_actions']}, "
                    f"errors={report['errors']}"
                )
                print(f"第 2 阶段 LLM: {review_summary}")
            except Exception as e:
                review_summary = f"failed: {e}"
                print(f"第 2 阶段失败: {e}")

        # 把本次运行结果记进状态文件，供下次 status 展示；dry-run 不落盘
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
    print("可用: status, run [--dry-run|--no-llm], pause, resume")


def curator_cli(args: list, skills_dir: Path = None):
    """curator 命令的总分发器：看第一个词是什么，转给对应的处理函数。

    CLI 层（cli.py / main）收到 curator 命令后调这里；每个动词
    （status/run/pause/...）各有一个小处理函数，这里只做路由。

    参数：
        args        命令参数列表，第一个元素是动词，如 ["run", "--dry-run"]；
                    空列表时默认显示状态
        skills_dir  技能库目录；不传就用 constants 里默认的技能目录

    返回：无（结果直接打印到终端）。
    """
    if skills_dir is None:
        from constants import skills_dir as _sd
        skills_dir = _sd()

    if not args:
        _show_status(skills_dir)
        return

    verb = args[0]

    # memory 是独立管线（管记忆库而不是技能库），单独转给 _cmd_memory
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
    """打印 curator 的运行状态：上次运行时间/总结、是否暂停、技能数量分布。

    参数：
        skills_dir  技能库目录，状态和用量数据都从这底下读

    返回：无（直接打印到终端）。
    """
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
    """立刻手动跑一轮技能审查（curator 平时按 7 天周期自动跑），并打印报告。

    参数：
        skills_dir  技能库目录
        dry_run     True 时只预览会发生什么转换，不真正动文件

    返回：无（报告直接打印到终端）。
    """
    print(f"{'[DRY RUN] ' if dry_run else ''}运行 curator...")
    report = run_curator_review(skills_dir, dry_run=dry_run)
    print(f"\n转换计数: {report['transitions']}")
    print(f"合并/清理: {report['consolidations']}")
    print(f"耗时: {report['duration_seconds']:.1f}s")


def _pause_curator(skills_dir: Path):
    """暂停 curator 的自动周期（写进状态文件，之后想恢复用 resume）。

    参数：
        skills_dir  技能库目录，状态文件存在这底下

    返回：无。
    """
    state = load_state(skills_dir)
    state["paused"] = True
    save_state(skills_dir, state)
    print("curator 已暂停")


def _resume_curator(skills_dir: Path):
    """恢复被 pause 暂停的 curator 自动周期。

    参数：
        skills_dir  技能库目录，状态文件存在这底下

    返回：无。
    """
    state = load_state(skills_dir)
    state["paused"] = False
    save_state(skills_dir, state)
    print("curator 已恢复")


def main(args=None):
    """命令行入口：python -m curator_cli <verb> ...

    参数：
        args  参数列表；不传（None）时从 sys.argv 读取——这是生产入口的
              路径；测试可以直接传一个 list 进来，不碰命令行

    返回：无。
    """
    if args is None:
        args = sys.argv[1:]
    if not args:
        # 无参 = 看状态；skills_dir 缺省值由 curator_cli 的分发逻辑解析，
        # 直接传 None 会一路传到 Path(None) 炸掉
        from constants import skills_dir as _default_skills_dir
        _show_status(_default_skills_dir())
        return
    curator_cli(args)


if __name__ == "__main__":
    main()
