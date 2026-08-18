"""会话/历史命令簇（R30 从 cli.py 机械抽离，行为不变）。

/resume /sessions /search /history 相关处理函数。
依赖注入：函数签名带 rt（RuntimeContext），console 用 cli_ui 共享实例。
"""
import json
import logging
import re
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from agent.cross_project import list_recent_bundles_across_projects
from cli_ui import console

logger = logging.getLogger(__name__)



def _handle_resume_command(args: str, rt) -> bool:
    """/resume_bundle [id|--cwd <path>]：跨项目列出/加载 bundle。

    无参数：列出最近 10 个 bundle（跨所有项目）。
    有参数：load bundle 注入 conversation_history（覆盖当前会话）。
    """
    handoff_store = getattr(rt, "handoff_store", None)
    if handoff_store is None:
        console.print("[red]handoff 存储未初始化[/red]")
        return True

    args = (args or "").strip()
    if not args:
        # 列出
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
            # created_at 是 datetime 对象（不是字符串）
            ca = b.created_at
            ca_str = ca.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ca, "strftime") else str(ca)[:19]
            console.print(
                f"  [cyan]{b.bundle_id[:12]}[/cyan] {ca_str} "
                f"[{cwd_short}]{auto_tag} {title}"
            )
        console.print("[dim]用法: /resume_bundle <id> 加载某个 bundle[/dim]")
        return True

    # 加载指定 bundle
    bid = args
    # 支持短 id 前缀匹配
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

    # 注入 conversation_history（覆盖当前）
    # HandoffBundle 是 dataclass，transcript 是字段
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
def _maybe_prompt_resume(rt: RuntimeContext):
    """启动时检测历史会话，提示用户是否恢复最近的会话。"""
    if not rt.session_store:
        return

    sessions = rt.session_store.list_sessions(limit=5)
    # 排除当前刚创建的空 session，只看有消息的历史
    history = [
        s for s in sessions
        if s["id"] != rt.session_id and (s.get("message_count") or 0) > 0
    ]
    if not history:
        return

    console.print("\n[cyan]发现历史会话：[/cyan]")
    for i, s in enumerate(history):
        title = s.get("title") or "(无标题)"
        cnt = s.get("message_count", 0)
        t = (s.get("updated_at") or "")[:19]
        console.print(f"  [{i}] {title}（{cnt} 条，{t}）")
    console.print("  [n] 开始新对话（默认）")

    try:
        choice = console.input("[bold]恢复哪个？[/bold] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "n" or not choice:
        return  # 保持刚创建的新 session

    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(history):
            # 恢复选中的，删除刚创建的空 session
            _resume_and_cleanup_empty(rt, history[idx]["id"])
        else:
            console.print(f"[yellow]序号超出范围，已开始新对话[/yellow]")
def _resume_and_cleanup_empty(rt: RuntimeContext, target_session_id: str) -> None:
    """恢复到指定 session，并清理掉当前的空 session（如果有）。

    用户在新会话里选择了恢复历史时，刚创建的空 session 需要删掉避免污染列表。
    清理失败不阻塞（只是多留一个空记录），但留 debug 日志便于排查。
    """
    empty_id = rt.session_id
    rt.resume_session(target_session_id)
    if empty_id and empty_id != rt.session_id:
        try:
            rt.session_store.delete_session(empty_id)
        except Exception:
            logger.debug("清理空 session 失败: %s", empty_id, exc_info=True)
def _resume_session_interactive(rt: RuntimeContext, args: str):
    """交互式恢复历史会话。

    用法：
      /resume         列出历史，输入序号选择
      /resume 0       直接恢复序号 0（最近）
      /resume <id前缀> 按 session_id 前缀匹配
    """
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    sessions = rt.session_store.list_sessions(limit=20)
    if not sessions:
        console.print("[yellow]暂无历史会话[/yellow]")
        return

    arg = args.strip()

    # 数字序号：直接恢复
    if arg.isdigit():
        idx = int(arg)
        if 0 <= idx < len(sessions):
            rt.resume_session(sessions[idx]["id"])
        else:
            console.print(f"[yellow]序号超出范围（0-{len(sessions) - 1}）[/yellow]")
        return

    # session_id 前缀匹配
    if arg:
        for s in sessions:
            if s["id"].startswith(arg):
                rt.resume_session(s["id"])
                return
        console.print(f"[yellow]未找到匹配 '{arg}' 的会话[/yellow]")
        return

    # 无参数：列出 + 让用户选
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
    """搜索历史对话。

    支持的过滤参数（key=value 形式，可组合）：
      role=<user|assistant|tool>       按角色过滤
      tool=<tool_name>                 按工具调用过滤（如 tool=terminal）
      since=<YYYY-MM-DD>               起始时间
      until=<YYYY-MM-DD>               截止时间
      limit=<N>                        结果数（默认 10）

    例：
      /search 报错 role=user
      /search python tool=terminal since=2026-07-01
    """
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    # 解析 query 和 filters
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

    # 日期补全（since 自动加 T00:00:00，until 自动加 T23:59:59）
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

    # 显示当前过滤条件
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
    """把消息列表按 user/assistant 不同颜色打印出来（tool 消息跳过）。

    用于恢复会话、handoff show 等多处场景的统一回放。
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
def _show_history_messages(rt: RuntimeContext, limit: int = 6):
    """恢复会话后，回放最近 N 条消息让用户看到上下文。"""
    if not rt.session_store or not rt.session_id:
        return
    msgs = rt.session_store.get_messages(rt.session_id, limit=limit)
    if not msgs:
        return
    _print_message_list(msgs, char_limit=300, header=f"最近 {len(msgs)} 条历史消息：")
def _auto_resume_last(rt: RuntimeContext):
    """-c/--continue 参数触发：自动恢复最近的有消息会话。"""
    if not rt.session_store:
        return

    sessions = rt.session_store.list_sessions(limit=5)
    history = [
        s for s in sessions
        if s["id"] != rt.session_id and (s.get("message_count") or 0) > 0
    ]
    if not history:
        console.print("[yellow]没有可恢复的历史会话，已开始新对话[/yellow]")
        return

    last = history[0]
    _resume_and_cleanup_empty(rt, last["id"])
    # 显示历史消息
    _show_history_messages(rt)
