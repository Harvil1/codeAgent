"""会话/历史类命令集。

这里放的是"翻旧账"的命令处理函数：/resume 恢复历史会话、/sessions 列
会话、/search 搜历史对话、/history 回放最近消息。依赖靠参数传入：每个
函数都带一个 rt（RuntimeContext，cli.py 聚合的运行时上下文，里面有
session_store / agent 等），终端输出统一走 cli_ui 的共享 console。
"""
# 注解延迟求值：rt: RuntimeContext 的 RuntimeContext 定义在 cli.py，
# 直接 import 会循环依赖；3.14+ 天然延迟，低版本靠 future 注解
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from agent.cross_project import list_recent_bundles_across_projects
from cli_commands import slash_command
from cli_ui import console

logger = logging.getLogger(__name__)



def _handle_resume_command(args: str, rt) -> bool:
    """/resume_bundle 命令：跨项目列出或加载"会话移交包"（bundle）。

    bundle 是把一段对话打包存的文件（handoff 机制），可以跨项目/跨机器
    接着聊。这个命令管两种用法：
    - 不带参数：列出最近 10 个 bundle（扫描所有项目，不分当前项目）；
    - 带 id：加载指定 bundle，把里面的对话记录直接替换当前会话历史。

    参数：
        args：命令参数（bundle id，支持只打前几位的前缀匹配）
        rt：RuntimeContext（取 handoff_store 和 agent 用）

    返回：
        bool —— True 表示命令已处理
    """
    handoff_store = getattr(rt, "handoff_store", None)
    if handoff_store is None:
        console.print("[red]handoff 存储未初始化[/red]")
        return True

    args = (args or "").strip()
    if not args:
        # 无参数：列出最近的 bundle
        try:
            from agent.cross_project import list_recent_bundles_across_projects
            bundles = list_recent_bundles_across_projects(handoff_store, limit=10)
        except Exception as e:
            console.print(f"[red]列 bundle 失败：[/red]{e}")
            return True
        if not bundles:
            console.print("[yellow]无 bundle（跨项目）[/yellow]")
            return True
        console.print(f"[cyan]最近 {len(bundles)} 个 bundle：[/cyan]")
        for b in bundles:
            title = b.title or "(无标题)"
            cwd = b.source_cwd or ""
            cwd_short = Path(cwd).name if cwd else ""
            auto_tag = " [auto]" if getattr(b, "auto_saved", False) else ""
            # 注意 created_at 是 datetime 对象不是字符串，要自己格式化
            ca = b.created_at
            ca_str = ca.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ca, "strftime") else str(ca)[:19]
            console.print(
                f"  [cyan]{b.bundle_id[:12]}[/cyan] {ca_str} "
                f"[{cwd_short}]{auto_tag} {title}"
            )
        console.print("[dim]用法: /resume_bundle <id> 加载某个 bundle[/dim]")
        return True

    # 有参数：加载指定的 bundle（id 支持只打前几位）
    bid = args
    # 短 id 前缀匹配：全量列表里找开头吻合的
    try:
        all_bundles = handoff_store.list_bundles()
        matches = [b for b in all_bundles if b.bundle_id.startswith(bid)]
        if not matches:
            console.print(f"[red]找不到 bundle：{bid}[/red]")
            return True
        if len(matches) > 1:
            console.print(
                f"[yellow]ID 前缀歧义（{len(matches)} 个匹配），"
                f"请用更长的前缀[/yellow]"
            )
            return True
        bid = matches[0].bundle_id
        bundle = handoff_store.load(bid)
    except Exception as e:
        console.print(f"[red]load bundle 失败：[/red]{e}")
        return True

    # 把 bundle 里的对话记录直接替换当前会话历史
    # HandoffBundle 是 dataclass，transcript 是它的一个字段
    transcript = getattr(bundle, "transcript", None) or []
    if not transcript:
        console.print("[yellow]bundle 无 transcript（空 bundle）[/yellow]")
        return True

    rt.agent.conversation_history = list(transcript)
    console.print(
        f"[green]✓ 已加载 bundle {bid[:12]}（{len(transcript)} 条消息）[/green] "
        "[dim]（已覆盖当前会话历史）[/dim]"
    )
    return True
def _list_sessions(rt: RuntimeContext):
    """/sessions 列表体：用 Rich 表展示最近 20 个历史会话。

    参数：
        rt：RuntimeContext（取 session_store 和当前 session_id——
        当前会话会在标题旁标绿"(当前)"）

    返回：
        无（直接打印表格；session_store 未启用时提示后返回）
    """
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    sessions = rt.session_store.list_sessions(limit=20)
    if not sessions:
        console.print("[yellow]暂无历史会话[/yellow]")
        return

    table = Table(title="历史会话")
    table.add_column("ID", style="dim")
    table.add_column("标题")
    table.add_column("消息数", justify="right")
    table.add_column("更新时间")

    for s in sessions:
        sid = s["id"]
        title = s.get("title") or "(无标题)"
        if sid == rt.session_id:
            title = f"[green]{title} (当前)[/green]"
        table.add_row(
            sid[:8] + "...",
            title,
            str(s.get("message_count", 0)),
            (s.get("updated_at") or "")[:19],
        )

    console.print(table)
def _resume_and_cleanup_empty(rt: RuntimeContext, target_session_id: str) -> None:
    """恢复到指定会话，并顺手删掉刚创建的那个空会话（不删会污染会话列表）。

    删除失败不阻塞——顶多列表里多一条空记录，留条 debug 日志方便排查。

    参数：
        rt：RuntimeContext（从中拿当前 session_id 并执行恢复）
        target_session_id：要恢复到的目标会话 id

    返回：
        无
    """
    empty_id = rt.session_id
    rt.resume_session(target_session_id)
    if empty_id and empty_id != rt.session_id:
        try:
            rt.session_store.delete_session(empty_id)
        except Exception:
            logger.warning("清理空 session 失败: %s", empty_id, exc_info=True)
def _resume_session_interactive(rt: RuntimeContext, args: str):
    """/resume 命令主体：交互式恢复一个历史会话。

    三种用法：
      /resume           列出历史，让用户输序号选
      /resume 0         直接恢复序号 0（就是最近的一个）
      /resume <id前缀>  按会话 id 的前几位匹配恢复

    参数：
        rt：RuntimeContext（取 session_store 用）
        args：命令参数（序号 / id 前缀 / 空）

    返回：
        无（恢复动作通过 rt.resume_session 完成）
    """
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    sessions = rt.session_store.list_sessions(limit=20)
    if not sessions:
        console.print("[yellow]暂无历史会话[/yellow]")
        return

    arg = args.strip()

    # 参数是数字序号：直接恢复对应会话
    if arg.isdigit():
        idx = int(arg)
        if 0 <= idx < len(sessions):
            rt.resume_session(sessions[idx]["id"])
        else:
            console.print(f"[yellow]序号超出范围（0-{len(sessions) - 1}）[/yellow]")
        return

    # 参数是 id 前缀：在列表里找开头吻合的会话
    if arg:
        for s in sessions:
            if s["id"].startswith(arg):
                rt.resume_session(s["id"])
                return
        console.print(f"[yellow]未找到匹配 '{arg}' 的会话[/yellow]")
        return

    # 无参数：列出会话让用户挑
    _list_sessions(rt)
    console.print(
        "\n输入 [bold]序号[/bold]恢复对应会话"
        "（0 是最近），或直接回车取消："
    )
    try:
        choice = console.input("[bold]序号>[/bold] ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not choice:
        return
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(sessions):
            rt.resume_session(sessions[idx]["id"])
        else:
            console.print(f"[yellow]序号超出范围（0-{len(sessions) - 1}）[/yellow]")
    else:
        console.print("[yellow]请输入数字序号[/yellow]")
def _search_sessions(rt: RuntimeContext, query: str):
    """/search 命令主体：在历史对话里搜关键词。

    搜索词后面还能跟 key=value 形式的过滤条件（可组合）：
      role=<user|assistant|tool>       只看某种角色的消息
      tool=<tool_name>                 只看调过某工具的（如 tool=terminal）
      since=<YYYY-MM-DD>               起始日期
      until=<YYYY-MM-DD>               截止日期
      limit=<N>                        最多返回几条（默认 10）

    例：
      /search 报错 role=user
      /search python tool=terminal since=2026-07-01

    参数：
        rt：RuntimeContext（取 session_store 用）
        query：用户输入的"关键词 + 过滤条件"整串

    返回：
        无（结果直接打印；没关键词时打印用法提示）
    """
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    # 把输入拆开：带 = 的当过滤条件，其余的拼成关键词
    parts = query.split()
    keywords = []
    filters = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            filters[k.lower()] = v
        else:
            keywords.append(p)

    if not keywords:
        console.print("[yellow]用法：/search <关键词> [key=value ...][/yellow]")
        console.print(
            "[dim]过滤：role= user/assistant/tool | tool= <name> | "
            "since= YYYY-MM-DD | until= YYYY-MM-DD | limit= N[/dim]"
        )
        return

    keyword = " ".join(keywords)
    limit = int(filters.get("limit", "10"))
    role = filters.get("role")
    tool_name = filters.get("tool")
    since = filters.get("since")
    until = filters.get("until")

    # 日期自动补全：只给日期没给时刻时，起始补零点、截止补当天最后一秒，
    # 这样"某一天"的边界才罩得住
    if since and "T" not in since:
        since = since + "T00:00:00"
    if until and "T" not in until:
        until = until + "T23:59:59"

    results = rt.session_store.search(
        keyword,
        limit=limit,
        role=role,
        tool_name=tool_name,
        since=since,
        until=until,
    )
    if not results:
        console.print(f"[yellow]未找到匹配 '{keyword}' 的对话[/yellow]")
        return

    # 把生效的过滤条件回显出来，让用户知道搜的时候带了什么筛子
    active_filters = []
    if role:
        active_filters.append(f"role={role}")
    if tool_name:
        active_filters.append(f"tool={tool_name}")
    if since:
        active_filters.append(f"since={since[:10]}")
    if until:
        active_filters.append(f"until={until[:10]}")
    filter_str = f" [dim]({', '.join(active_filters)})[/dim]" if active_filters else ""

    console.print(f"[bold]搜索 '{keyword}' 的结果（{len(results)} 条）：[/bold]{filter_str}")
    for r in results:
        title = r.get("title") or "(无标题)"
        snippet = r.get("snippet", "")
        role_tag = f"[dim][{r.get('role', '?')}][/dim] "
        console.print(f"\n[cyan]{title}[/cyan] [dim]({r.get('timestamp', '')[:19]})[/dim]")
        console.print(f"  {role_tag}{snippet}")
# ---------------------------------------------------------------------------

def _print_message_list(msgs, *, char_limit: int = 300, header: Optional[str] = None):
    """把一组对话消息回放到终端：用户的话青色、AI 的话绿色（tool 消息跳过）。
    恢复会话、看移交包等多处共用的统一回放函数。

    参数：
        msgs：消息列表（role/content 结构的 dict）
        char_limit：单条消息最多显示多少字符（超长截断加 ...，默认 300）
        header：可选的标题行（传了就先打印）

    返回：
        无
    """
    if not msgs:
        return
    if header:
        console.print(f"\n[cyan]{header}[/cyan]\n")
    for msg in msgs:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if len(content) > char_limit:
            content = content[: char_limit - 3] + "..."
        if role == "user":
            console.print(f"[bold cyan]你:[/bold cyan] {content}")
        elif role == "assistant":
            console.print(f"[bold green]AI:[/bold green] {content}")
    console.print()
def _auto_resume_last(rt: RuntimeContext):
    """命令行带 -c/--continue 时用：跳过询问，直接恢复最近一个有消息的会话。

    参数：
        rt：RuntimeContext（取 session_store 用）

    返回：
        无（没有可恢复的就提示一句，开新对话）
    """
    if not rt.session_store:
        return

    # 先取大候选池再滤空会话：裸启动每次都会留下一个空会话档案，
    # limit=5 先截断的话攒 4 个空会话就把窗口占满——明明库里有非空
    # 历史却报"没有可恢复的会话"
    sessions = rt.session_store.list_sessions(limit=50)
    history = [
        s for s in sessions
        if s["id"] != rt.session_id and (s.get("message_count") or 0) > 0
    ]
    if not history:
        # 索引滞后兜底：崩溃/强退后 index.json 的 message_count 可能滞后
        # 到 0（去抖落盘没跑），真实非空的会话被误判成空——对最近的
        # 几个"零计数"会话实读一次消息数核实（最多查 5 个，不扫全库）
        _checked = 0
        for s in sessions:
            if s["id"] == rt.session_id or _checked >= 5:
                continue
            if (s.get("message_count") or 0) > 0:
                continue
            _checked += 1
            try:
                if len(rt.session_store.get_messages(s["id"])) > 0:
                    history = [s]
                    break
            except Exception:
                continue
    if not history:
        console.print("[yellow]没有可恢复的历史会话，已开始新对话[/yellow]")
        return

    last = history[0]
    # 恢复 + 概览回显都在 resume_session 里做（分类报数 + 尾部预览）；
    # 这里不再重复回放（曾经两处各打一遍，屏上出现两份「最近 N 条」）
    _resume_and_cleanup_empty(rt, last["id"])


# ---------------------------------------------------------------------------
# 注册命令（会话类）——装饰器在 import 时自登记进 cli_commands 注册表。
# 注意：转调 cli.py 里定义的实现（如 _handle_rewind_command）必须用函数内
# 延迟 import——模块级 import 会循环依赖（cli.py 顶部正在 import 本模块）。
# ---------------------------------------------------------------------------

@slash_command(name="/new", category="会话", usage="/new",
               summary="开始新对话")
def cmd_new(args: str, rt) -> bool:
    rt.new_session()
    console.print("[green][已开始新对话][/green]")
    return True


@slash_command(name="/sessions", category="会话", usage="/sessions",
               summary="列出历史会话")
def cmd_sessions(args: str, rt) -> bool:
    _list_sessions(rt)
    return True


@slash_command(name="/resume", category="会话", usage="/resume [序号]",
               summary="恢复历史会话")
def cmd_resume(args: str, rt) -> bool:
    _resume_session_interactive(rt, args)
    return True


@slash_command(name="/search", category="会话", usage="/search <关键词>",
               summary="搜索历史对话")
def cmd_search(args: str, rt) -> bool:
    if not args:
        console.print("[yellow]用法：/search <关键词>[/yellow]")
        return True
    _search_sessions(rt, args)
    return True


@slash_command(name="/history", category="会话", usage="/history [N]",
               summary="全局输入历史（/history N 看第 N 条原文）")
def cmd_history(args: str, rt) -> bool:
    # 全局输入历史（/history 列最近 20 条；/history N 打印第 N 条完整原文）
    try:
        from agent.input_history import GlobalHistory
        h = GlobalHistory(rt.home)
        if args.strip().isdigit():
            item = h.get(int(args.strip()))
            if item:
                console.print(Panel.fit(item[:2000], title="输入历史（复制后可直接粘贴使用）"))
            else:
                console.print("[yellow]没有第 %s 条历史[/yellow]" % args.strip())
            return True
        items = h.recent(20)
        if not items:
            console.print("[dim]暂无输入历史[/dim]")
            return True
        lines = [
            f"[cyan]{i}[/cyan]. {t[:80].replace(chr(10), ' ')}"
            + ("…" if len(t) > 80 else "")
            for i, t in enumerate(items, 1)
        ]
        console.print(Panel.fit("\n".join(lines), title="输入历史（/history N 看原文）"))
    except Exception as e:
        console.print(f"[red]历史读取失败: {e}[/red]")
    return True


@slash_command(name="/rewind", category="会话", usage="/rewind",
               summary="回退到历史某个节点")
def cmd_rewind(args: str, rt) -> bool:
    from cli import _handle_rewind_command
    _handle_rewind_command(rt, args)
    return True


@slash_command(name="/handoff", category="会话",
               usage="/handoff [save|load|list|show|delete|export|import]",
               summary="会话移交（保存/加载/查看/导入导出）")
def cmd_handoff(args: str, rt) -> bool:
    from cli import _handle_handoff_command
    return _handle_handoff_command(args, rt)


@slash_command(name="/resumable", category="会话", usage="/resumable [agent_id]",
               summary="列出/恢复可续跑的子代理")
def cmd_resumable(args: str, rt) -> bool:
    from cli import _handle_resumable_command
    return _handle_resumable_command(args, rt)


@slash_command(name="/resume_bundle", category="会话", usage="/resume_bundle [id]",
               summary="跨项目恢复 bundle（/resume 已被恢复会话占用）")
def cmd_resume_bundle(args: str, rt) -> bool:
    # /resume 这个名字已被"恢复会话"占用，跨项目的 bundle 恢复
    # 另起 /resume_bundle 加以区分
    return _handle_resume_command(args, rt)


@slash_command(name="/inbox", category="会话", usage="/inbox",
               summary="显示 ChannelInbox 未消费消息")
def cmd_inbox(args: str, rt) -> bool:
    from cli import _handle_inbox_command
    return _handle_inbox_command(args, rt)


@slash_command(name="/mailbox", category="会话", usage="/mailbox [send|check|clear]",
               summary="队友邮箱（发消息/收信/清空）")
def cmd_mailbox(args: str, rt) -> bool:
    from cli import _handle_mailbox_command
    return _handle_mailbox_command(args, rt)
