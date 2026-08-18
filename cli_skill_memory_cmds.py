"""技能/记忆命令簇（R30 从 cli.py 机械抽离，行为不变）。

/skills /memory /skill-learning 相关处理函数。
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

from rich.table import Table

from tools.skill_usage import load_usage
from constants import get_omnimate_home, skills_dir
from cli_ui import console

logger = logging.getLogger(__name__)



def _handle_skill_learning_command(args: str, rt) -> bool:
    """/skill-learning status|start|stop|evolve|prune（CCAR15 Task 4）。

    status：enabled/observer + instinct 总数 / global 簇数 / 已进化技能数
    start/stop：翻 runtime config 的 skill_learning.enabled（重启失效；
                提示走 config_set skill_learning.enabled 持久化）
    evolve：手动触发簇达标演化（global scope，门槛读 config）
    prune：清过期低置信 instinct（store.prune）
    """
    from agent.skill_learning.store import InstinctStore

    sub = (args or "status").strip().lower() or "status"
    home = Path(rt.home)
    cfg = rt.config if isinstance(rt.config, dict) else {}
    sl_cfg = cfg.setdefault("skill_learning", {})

    if sub in ("start", "stop"):
        enabled = (sub == "start")
        sl_cfg["enabled"] = enabled
        # rt 与 agent 通常共享同一 config dict（cli initialize 传入）；
        # 若 agent 持有独立 dict（测试/自定义装配）也同步一份
        agent_cfg = getattr(getattr(rt, "agent", None), "config", None)
        if isinstance(agent_cfg, dict) and agent_cfg is not cfg:
            agent_cfg.setdefault("skill_learning", {})["enabled"] = enabled
        console.print(
            f"[green]skill_learning 已{'开启' if enabled else '关闭'}（runtime）[/green] "
            f"[dim]（重启失效；持久化用 config_set skill_learning.enabled "
            f"{'true' if enabled else 'false'}）[/dim]"
        )
        return True

    if sub == "status":
        try:
            store = InstinctStore(home / ".skill-learning")
            instincts = store.list_all()
            clusters = store.cluster("global")
            skills_dir = home / "skills"
            learned = (list(skills_dir.glob("learned-*"))
                       if skills_dir.is_dir() else [])
            console.print(
                f"[cyan]enabled:[/cyan] {sl_cfg.get('enabled', False)}  "
                f"[cyan]observer:[/cyan] {sl_cfg.get('observer', 'heuristic')}"
            )
            console.print(
                f"[cyan]instinct 总数:[/cyan] {len(instincts)}  "
                f"[cyan]global 簇数:[/cyan] {len(clusters)}  "
                f"[cyan]已进化技能:[/cyan] {len(learned)}"
            )
        except Exception as e:
            console.print(f"[red]读取 skill_learning 状态失败：[/red]{e}")
        return True

    if sub == "evolve":
        try:
            from agent.skill_learning import maybe_evolve
            store = InstinctStore(home / ".skill-learning")
            generated = maybe_evolve(
                store, "global", home / "skills",
                min_avg_confidence=sl_cfg.get("evolve_threshold", 0.75),
                min_members=sl_cfg.get("evolve_min_cluster", 3),
            )
            if generated:
                names = ", ".join(p.parent.name for p in generated)
                console.print(
                    f"[green]本次演化生成 {len(generated)} 个技能：[/green]{names}"
                )
            else:
                console.print("[yellow]没有达标的簇（无新技能生成）[/yellow]")
        except Exception as e:
            console.print(f"[red]evolve 失败：[/red]{e}")
        return True

    if sub == "prune":
        try:
            store = InstinctStore(home / ".skill-learning")
            removed = store.prune()
            console.print(
                f"[green]prune 完成：清理 {removed} 条过期低置信 instinct[/green]"
            )
        except Exception as e:
            console.print(f"[red]prune 失败：[/red]{e}")
        return True

    console.print(
        "[yellow]用法: /skill-learning status|start|stop|evolve|prune[/yellow]"
    )
    return True
def _handle_skills_command(rt: RuntimeContext, args: str):
    """处理 /skills [list|rate|recommend] 子命令。"""
    parts = args.split(None, 1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("list", "ls", ""):
        _list_skills(rt)
        return

    if sub == "rate":
        _rate_skill(rt, rest)
        return

    if sub in ("recommend", "rec"):
        _recommend_skills(rt)
        return

    console.print(f"[yellow]未知子命令：{sub}（用 list / rate / recommend）[/yellow]")
def _list_skills(rt: RuntimeContext):
    sd = skills_dir()
    usage = load_usage(sd)

    table = Table(title="技能列表")
    table.add_column("命令", style="cyan")
    table.add_column("描述")
    table.add_column("使用", justify="right")
    table.add_column("评分", justify="right")

    found = False
    for skill_md in sorted(sd.glob("*/SKILL.md")):
        name = skill_md.parent.name
        rec = usage.get(name, {})
        if rec.get("state") == "archived":
            continue
        found = True

        # 从 frontmatter 读描述
        desc = ""
        try:
            content = skill_md.read_text(encoding="utf-8")
            from agent.skill_commands import parse_frontmatter
            fm, _ = parse_frontmatter(content)
            desc = fm.get("description", "")
        except Exception:
            logger.debug("读取技能 frontmatter 失败", exc_info=True)

        rating = rec.get("rating")
        rating_str = f"{rating}★" if rating else "-"
        table.add_row(f"/{name}", desc, str(rec.get("use_count", 0)), rating_str)

    if found:
        console.print(table)
        console.print(
            "\n[dim]/skills rate <name> <1-5> 打分 | "
            "/skills recommend 推荐[/dim]"
        )
    else:
        console.print("[yellow]暂无技能。用 skill_manage 工具创建。[/yellow]")
def _rate_skill(rt: RuntimeContext, args: str):
    """/skills rate <name> <1-5>"""
    parts = args.split()
    if len(parts) != 2:
        console.print("[yellow]用法：/skills rate <技能名> <1-5>[/yellow]")
        return

    name, rating_str = parts
    try:
        rating = int(rating_str)
    except ValueError:
        console.print(f"[red]评分必须是整数：{rating_str}[/red]")
        return

    from tools.skill_usage import set_rating
    ok, msg = set_rating(skills_dir(), name, rating)
    if ok:
        console.print(f"[green]✓ {msg}[/green]")
    else:
        console.print(f"[red]{msg}[/red]")
def _recommend_skills(rt: RuntimeContext):
    """/skills recommend — 基于使用次数 + 评分的综合推荐。"""
    from tools.skill_usage import get_recommendations

    recs = get_recommendations(skills_dir(), limit=5)
    if not recs:
        console.print("[yellow]暂无技能可推荐。[/yellow]")
        return

    console.print(f"[bold]推荐技能 Top {len(recs)}：[/bold]")
    console.print(
        "[dim]综合分 = 使用次数 × 1.0 + 评分 × 2.0 + 查看次数 × 0.1"
        "（pinned +10）[/dim]\n"
    )

    table = Table()
    table.add_column("#", style="dim", justify="right")
    table.add_column("技能", style="cyan")
    table.add_column("综合分", justify="right")
    table.add_column("使用", justify="right")
    table.add_column("评分", justify="right")
    table.add_column("描述")

    for i, r in enumerate(recs, 1):
        pin = "📌 " if r["pinned"] else ""
        rating_str = f"{r['rating']}★" if r["rating"] else "-"
        desc = (r["description"] or "")[:50]
        table.add_row(
            str(i),
            f"{pin}/{r['name']}",
            str(r["score"]),
            str(r["use_count"]),
            rating_str,
            desc,
        )
    console.print(table)
def _show_memory(rt: RuntimeContext):
    if not rt.memory_store:
        console.print("[yellow]记忆系统未启用[/yellow]")
        return

    # 修复（2026-08-17）：原实现访问不存在的 memory_entries/user_entries 属性
    # （/memory 必崩 AttributeError 的既有 bug）。改用 list_all() 按类型分组。
    entries = rt.memory_store.list_all()
    agent_entries = [e for e in entries if e.type not in ("user", "feedback")]
    user_entries = [e for e in entries if e.type in ("user", "feedback")]

    console.print("[bold]MEMORY.md（agent 笔记）：[/bold]")
    for entry in agent_entries:
        desc = (entry.description or "")[:60]
        console.print(f"  - [{entry.type}] {entry.name}: {desc}")
    if not agent_entries:
        console.print("  [dim]（空）[/dim]")

    console.print("\n[bold]USER.md（用户画像）：[/bold]")
    for entry in user_entries:
        desc = (entry.description or "")[:60]
        console.print(f"  - [{entry.type}] {entry.name}: {desc}")
    if not user_entries:
        console.print("  [dim]（空）[/dim]")

    # 菜单：编辑 MEMORY.md / USER.md（对齐 Claude Code /memory 命令体验）
    console.print(
        "\n[dim]输入 [cyan]m[/cyan] 编辑 MEMORY.md，[cyan]u[/cyan] 编辑 USER.md，"
        "其他键返回[/dim]"
    )
    choice = console.input("> ").strip().lower()
    if choice == "m":
        _open_in_editor(get_omnimate_home() / "MEMORY.md")
    elif choice == "u":
        _open_in_editor(get_omnimate_home() / "USER.md")


def _open_in_editor(path: Path) -> None:
    """用 $EDITOR（Windows fallback notepad）打开文件。"""
    import subprocess
    editor = os.environ.get("EDITOR") or ("notepad" if sys.platform == "win32" else "vi")
    try:
        subprocess.Popen([editor, str(path)])
        console.print(f"[green]已用 {editor} 打开 {path}[/green]")
        console.print("[dim]编辑保存后会话重新加载生效（保护 prompt cache）。[/dim]")
    except Exception as e:
        console.print(f"[red]打开编辑器失败: {e}[/red]")
        console.print(f"[yellow]手动编辑：{path}[/yellow]")
def _quick_save_memory(rt: RuntimeContext, text: str) -> None:
    """`#` 快捷写记忆：弹菜单选类型，直接调 MemoryStore.save。

    对齐 Claude Code 的 `#` shortcut 体验。
    """
    if not rt.memory_store:
        console.print("[yellow]记忆系统未启用，无法保存[/yellow]")
        return
    if not text:
        return

    console.print(f"[dim]内容：[/dim] {text}")
    console.print(
        "[bold]选择类型：[/bold] "
        "[cyan]1[/cyan]=user  [cyan]2[/cyan]=feedback  "
        "[cyan]3[/cyan]=project  [cyan]4[/cyan]=reference  [cyan]5[/cyan]=other"
    )
    choice = console.input("> ").strip()
    type_map = {
        "1": "user", "2": "feedback", "3": "project",
        "4": "reference", "5": "other",
    }
    mtype = type_map.get(choice, "other")
    # name 用前 30 字符（同 topic 同 name 会更新而非新建）
    name = text[:30].replace("\n", " ")
    try:
        entry_id = rt.memory_store.save(
            name=name,
            description=text,
            type=mtype,
            body=text,
            topic="quick",
        )
        console.print(
            f"[green]已保存记忆（type={mtype}, id={entry_id}）[/green]\n"
            f"[dim]下次会话注入生效（保护 prompt cache）。[/dim]"
        )
    except Exception as e:
        console.print(f"[red]保存失败: {e}[/red]")
