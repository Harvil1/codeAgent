"""技能/记忆类命令集。

这里放的是"管理 agent 知识库"的命令处理函数：/skills 看技能列表打分推荐、
/memory 看和编辑记忆、/skill-learning 管控行为学习管线。被 cli.py 的主
分发调用，输出统一走 cli_ui 的共享 console。
"""
# 注解延迟求值：rt: RuntimeContext 的 RuntimeContext 定义在 cli.py，
# 直接 import 会循环依赖；3.14+ 天然延迟，低版本靠 future 注解
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from rich.table import Table

from tools.skill_usage import load_usage
from constants import get_codeagent_home, skills_dir
from cli_commands import slash_command
from cli_ui import console

logger = logging.getLogger(__name__)



def _handle_skill_learning_command(args: str, rt) -> bool:
    """/skill-learning 命令：管理 skillLearning 行为学习管线。

    skillLearning 是"agent 从使用中攒经验"的系统——平时观察用户行为
    存成 instinct（本能条目），攒够一簇相似的就演化成正式技能。这个命令
    管五个子命令：
    - status：看开关状态 + instinct 总数 / global 簇数 / 已进化技能数；
    - start/stop：开/关开关。注意只改内存里的配置，重启就失效；要长期
      生效得用 config_set skill_learning.enabled 持久化（会提示用户）；
    - evolve：手动触发一次"簇达标就演化"（只针对 global 范围，门槛值
      从 config 读）；
    - prune：清掉过期且置信度低的 instinct。

    参数：
        args：子命令串（status/start/stop/evolve/prune，空默认 status）
        rt：RuntimeContext（取 home 路径、config 和 agent 用）

    返回：
        bool —— True 表示命令已处理
    """
    from agent.skill_learning.store import InstinctStore

    sub = (args or "status").strip().lower() or "status"
    home = Path(rt.home)
    cfg = rt.config if isinstance(rt.config, dict) else {}
    sl_cfg = cfg.setdefault("skill_learning", {})

    if sub in ("start", "stop"):
        enabled = (sub == "start")
        sl_cfg["enabled"] = enabled
        # rt 和 agent 通常是同一份 config dict（cli 初始化时传的就是同一个）；
        # 但万一 agent 自己拿了一份独立的（测试或自定义装配的场景），
        # 也要同步改一份，否则开关对 agent 不生效
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
    """/skills 命令分发器：按第一个词分发给 list / rate / recommend。

    参数：
        rt：RuntimeContext（传给子处理函数）
        args：子命令串（空默认 list）

    返回：
        无
    """
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
    """/skills list 列表体：用 Rich 表展示所有未归档的技能。

    列出每个技能的命令名、描述、使用次数和用户评分；已归档的
    （state=archived）不显示。

    参数：
        rt：RuntimeContext（本函数实际只用 skills_dir 的全局函数，
        参数保持签名一致）

    返回：
        无（没技能时提示用 skill_manage 工具创建）
    """
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

        # 描述从 SKILL.md 头部的 frontmatter（--- 包住的元数据块）里读
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
    """/skills rate 子命令：给某个技能打 1-5 星的评分。

    评分记进使用统计，参与推荐排序（pinned 的技能免疫自动归档）。

    参数：
        rt：RuntimeContext（保持签名一致）
        args："<技能名> <1-5>" 形式的参数串

    返回：
        无（成功/失败都直接打印提示）
    """
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
    """/skills recommend 子命令：按综合分推荐 Top 5 技能。

    综合分 = 使用次数 × 1.0 + 评分 × 2.0 + 查看次数 × 0.1（pinned 额外 +10）。

    参数：
        rt：RuntimeContext（保持签名一致，实际数据从使用统计读）

    返回：
        无（没有可推荐的技能时提示一句）
    """
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
    """/memory 命令主体：展示记忆条目 + 提供编辑入口。

    记忆分两摊展示——agent 笔记（MEMORY.md 相关，project/reference
    等类型）和用户画像（USER.md 相关，user/feedback 类型）。看完后弹出
    小菜单：按 m 编辑 MEMORY.md、按 u 编辑 USER.md。

    参数：
        rt：RuntimeContext（取 memory_store 用）

    返回：
        无（记忆系统未启用时提示后返回）
    """
    if not rt.memory_store:
        console.print("[yellow]记忆系统未启用[/yellow]")
        return

    # memory_store 上没有按类分好的现成属性，用 list_all() 拿全量
    # 再按类型自己分组。
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

    # 菜单：让用户直接打开 MEMORY.md / USER.md 编辑
    console.print(
        "\n[dim]输入 [cyan]m[/cyan] 编辑 MEMORY.md，[cyan]u[/cyan] 编辑 USER.md，"
        "其他键返回[/dim]"
    )
    choice = console.input("> ").strip().lower()
    if choice == "m":
        _open_in_editor(get_codeagent_home() / "MEMORY.md")
    elif choice == "u":
        _open_in_editor(get_codeagent_home() / "USER.md")


def _open_in_editor(path: Path) -> None:
    """用系统编辑器打开一个文件：优先 $EDITOR 环境变量，Windows 上没设
    就用记事本，其他系统用 vi。

    参数：
        path：要打开的文件路径

    返回：
        无（编辑器是后台拉起的，不等它关；打不开就提示手动路径）
    """
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
    """输入以 `#` 开头时的快捷存记忆：弹菜单选类型，直接存 MemoryStore
    （一句话就能存，不用等模型来调工具）。

    参数：
        rt：RuntimeContext（取 memory_store 用）
        text：`#` 后面跟的正文（就是要存的内容）

    返回：
        无（存完/失败都打印提示）
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
    # 条目名取前 30 字：同 topic 同名会走"更新"而不是重复新建
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


# ---------------------------------------------------------------------------
# 注册命令（技能/记忆类）——装饰器在 import 时自登记进 cli_commands 注册表。
# 注意：转调 cli.py 里定义的实现（如 _handle_paste_command）必须用函数内
# 延迟 import——模块级 import 会循环依赖（cli.py 顶部正在 import 本模块）。
# ---------------------------------------------------------------------------

@slash_command(name="/skills", category="技能/记忆",
               usage="/skills [rate <name> <1-5> | recommend]",
               summary="列出技能（可打分/推荐）")
def cmd_skills(args: str, rt) -> bool:
    _handle_skills_command(rt, args)
    return True


@slash_command(name="/memory", category="技能/记忆", usage="/memory",
               summary="查看记忆（m/u 键编辑 MEMORY.md/USER.md）")
def cmd_memory(args: str, rt) -> bool:
    _show_memory(rt)
    return True


@slash_command(name="/paste", category="技能/记忆", usage="/paste",
               summary="保存剪贴板图片到 .paste/（Windows）")
def cmd_paste(args: str, rt) -> bool:
    from cli import _handle_paste_command
    return _handle_paste_command(args, rt)


@slash_command(name="/skill-learning", category="技能/记忆",
               usage="/skill-learning [status|start|stop|evolve|prune]",
               summary="行为直觉学习管线管理")
def cmd_skill_learning(args: str, rt) -> bool:
    return _handle_skill_learning_command(args, rt)


@slash_command(name="/agents", category="技能/记忆", usage="/agents",
               summary="列出自定义子代理定义")
def cmd_agents(args: str, rt) -> bool:
    # E2 新增：列出自定义子代理定义（~/.codeAgent/agents + ./.codeAgent/agents）
    from agent.agent_defs import scan_agent_defs
    defs = scan_agent_defs()
    if not defs:
        console.print(
            "[yellow]无自定义子代理。[/yellow] "
            "在 [cyan]~/.codeAgent/agents/[/cyan] 或 [cyan]./.codeAgent/agents/[/cyan] "
            "放 .md 文件（frontmatter 含 name/description/tools/maxTurns 等）。"
        )
        return True
    console.print(f"[green]共 {len(defs)} 个自定义子代理:[/green]")
    for n, d in defs.items():
        tools = ",".join(d.tools) if d.tools else "(默认)"
        model_str = d.model or "继承"
        perm_str = d.permission_mode or "default"
        max_str = d.max_turns if d.max_turns else "默认"
        console.print(
            f"  [cyan]{n}[/cyan]: {d.description} "
            f"[tools={tools}, model={model_str}, perm={perm_str}, maxTurns={max_str}]"
        )
    return True
