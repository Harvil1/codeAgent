"""诊断/状态类命令集（R30 给 cli.py 瘦身时原样搬过来的，行为没变）。

这里放的是"给用户看病"的命令处理函数：/status 看当前状态、/doctor 自检
环境、/context 看上下文占用、/compact 手动压缩上下文、/usage 看 token 花
销、/stats 看跨会话统计。它们都被 cli.py 的主分发调用，输出统一走
cli_ui 的共享 console。
"""
import importlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
import asyncio
from typing import Optional

from rich.table import Table

from constants import get_omnimate_home
from cli_ui import console

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# CCAR11 Task 2 新增：/compact + /context 两个命令
# ---------------------------------------------------------------------------

def _sync_history_after_compact(agent, new_messages: list) -> None:
    """压缩做完后，把新历史写回 agent 并作废 system prompt 缓存。

    背景：手动压缩后对话内容变了，但 agent 身上的旧历史和旧 prompt 缓存
    还留着，不处理的话下一次调用 LLM 用的还是压缩前的旧东西。这里的收尾
    动作和主循环里自动压缩（AIAgent._run_context_compression）保持一致：
    - 去掉开头的 system 消息（压缩函数的输入本就不该含 system，这里是
      防御性兜底）；
    - 去掉带 ``_ephemeral`` 标记的消息（临时消息只活一轮，不能进持久历史）；
    - 作废 system prompt 缓存（对话前缀变了，缓存已经不对了，必须重建，
      否则下次发给 LLM 的 system prompt 还是旧缓存里的）。

    参数：
        agent：AIAgent 实例（要更新它的 conversation_history）
        new_messages：压缩函数产出的新消息列表

    返回：
        无（直接改 agent 的字段）
    """
    msgs = list(new_messages or [])
    if msgs and msgs[0].get("role") == "system":
        # 防御性兜底：正常压缩输入不含 system，万一有就去掉
        msgs = msgs[1:]
    agent.conversation_history = [m for m in msgs if not m.get("_ephemeral")]
    # 用 getattr 探测而不是直接调：老版本 agent 可能还没有这个方法
    invalidate = getattr(agent, "invalidate_system_prompt", None)
    if callable(invalidate):
        try:
            invalidate()
        except Exception:
            pass
def _print_compact_delta(
    before_msgs: int, before_tokens: int,
    after_msgs: int, after_tokens: int,
    *, mode: str,
) -> None:
    """打印压缩前后的对比：消息数和 token 数各省了多少。

    参数：
        before_msgs / before_tokens：压缩前的消息条数 / 估算 token 数
        after_msgs / after_tokens：压缩后的消息条数 / 估算 token 数
        mode：压缩方式名（如 "llm_compact（L4 摘要）"），仅用于展示

    返回：
        无（直接打印到终端）
    """
    saved = max(0, before_tokens - after_tokens)
    console.print(f"[green]压缩完成（{mode}）[/green]")
    console.print(
        f"  消息数 {before_msgs} → {after_msgs}，"
        f"估算 tokens ~{before_tokens} → ~{after_tokens}"
        f"（节省 ~{saved}）"
    )
def _handle_compact_cli(args: str, rt) -> bool:
    """/compact 命令：用户手动触发一次上下文压缩（CCAR11 Task 2）。

    背景：上下文（对话历史）太长会撑爆模型窗口、烧钱。平时是自动压缩，
    这个命令让用户"现在就压"。三条路：
    - ``--yes`` 参数跳过确认；不加就交互式问一句（EOF 或输入异常当作拒绝）；
    - 正常路径用 llm_compact 走 L4（用 LLM 把早期对话总结成摘要），
      token_threshold 传 0 绕过"够不够长才压"的自动判定——手动压缩的
      意思就是无条件压，跟自动压缩的"快到窗口上限才压"不是一回事；
    - LLM 不可用时（client 为空 / 调用报错 / 已在事件循环里跑不了）
      降级用 snip_compact——它不调 LLM、无损地裁掉中段，同样强制执行；
    - 压缩完同步 agent 状态并把本次记入 llm_compact_count（和自动压缩
      的收尾完全一致）。

    参数：
        args：命令后跟的参数串（目前只认 --yes）
        rt：RuntimeContext（cli.py 聚合的运行时上下文，取 agent 和 config 用）

    返回：
        bool —— True 表示命令已处理（保持统一的命令处理签名）
    """
    agent = getattr(rt, "agent", None)
    if agent is None:
        console.print("[yellow]Agent 未初始化，无法压缩[/yellow]")
        return True

    # --yes 跳过确认，否则交互式问一句（输入异常当作拒绝，宁可不压）
    arg_parts = (args or "").split()
    if "--yes" not in arg_parts:
        try:
            ans = console.input("压缩会用 LLM 摘要总结早期对话，继续？(y/n) ")
        except Exception:
            ans = "n"
        if str(ans).strip().lower() not in ("y", "yes"):
            console.print("[yellow]已取消压缩[/yellow]")
            return True

    from agent.context_compressor import estimate_message_tokens
    from agent.context_pipeline import llm_compact, snip_compact

    history = list(agent.conversation_history)
    before_msgs = len(history)
    before_tokens = estimate_message_tokens(history)
    ctx_cfg = (getattr(rt, "config", None) or {}).get("context", {}) or {}
    keep_recent = int(ctx_cfg.get("llm_compact_keep_recent", 30))

    compacted = False
    new_messages = history
    llm_client = getattr(agent, "llm_client", None)
    if llm_client is not None:
        try:
            new_messages, compacted = asyncio.run(llm_compact(
                history,
                llm_client=llm_client,
                model=getattr(agent, "model", None),
                keep_recent=keep_recent,
                token_threshold=0,  # 传 0 = 无条件压（绕过"够长才压"的判定）
                precomputed_tokens=before_tokens,  # 复用已算好的 token 数；空历史时 0>0 自然短路
            ))
        except RuntimeError:
            # asyncio.run 在已有事件循环的嵌套环境里会抛 RuntimeError——
            # 走不了 LLM 压缩就降级 snip（和 /init 的桥接做法一致）
            console.print("[yellow]事件循环冲突，降级为 snip_compact（无损裁剪）[/yellow]")
        except Exception as e:
            console.print(f"[red]LLM 压缩失败：[/red]{e}")

    if compacted:
        # L4 摘要成功：把新历史写回 agent（收尾动作和自动压缩一致）
        _sync_history_after_compact(agent, new_messages)
        state = getattr(agent, "_compress_session_state", None)
        if state is not None:
            try:
                state.record_llm_compact()
            except Exception:
                pass
        after_tokens = estimate_message_tokens(agent.conversation_history)
        _print_compact_delta(
            before_msgs, before_tokens,
            len(agent.conversation_history), after_tokens,
            mode="llm_compact（L4 摘要）",
        )
        return True

    # 降级路径：snip_compact 不调 LLM、无损裁剪中段，threshold=0 强制执行
    new_messages, snipped = snip_compact(
        history, keep_first=3, keep_last=10, threshold=0,
    )
    if not snipped:
        console.print("[yellow]历史太短，无需压缩[/yellow]")
        return True
    console.print("[yellow]LLM 不可用，已降级 snip_compact（无损裁剪）[/yellow]")
    _sync_history_after_compact(agent, new_messages)
    after_tokens = estimate_message_tokens(agent.conversation_history)
    _print_compact_delta(
        before_msgs, before_tokens,
        len(agent.conversation_history), after_tokens,
        mode="snip_compact（无损）",
    )
    return True
def _handle_context_cli(args: str, rt) -> bool:
    """/context 命令：用一张表显示当前上下文（对话历史）的占用情况。

    背景：用户想知道"现在上下文有多满、压缩发生了几次"，这张 Rich 表给
    出全景：
    - 消息按角色（system/user/assistant/tool）各有多少条；
    - 用 estimate_message_tokens 估算的总 token 数；
    - 压缩会话状态（当前第几轮 / LLM 压缩次数 / 被动压缩次数）；
    - 待注入的临时消息（_pending_ephemeral_messages）还剩几条。

    参数：
        args：命令参数（本命令不使用，保持签名统一）
        rt：RuntimeContext（取 agent 用）

    返回：
        bool —— True 表示命令已处理
    """
    from collections import Counter

    from agent.context_compressor import estimate_message_tokens

    agent = getattr(rt, "agent", None)
    history = list(getattr(agent, "conversation_history", None) or []) \
        if agent is not None else []
    role_counts = Counter(m.get("role", "?") for m in history)
    total_tokens = estimate_message_tokens(history)

    table = Table(title="Context 状态")
    table.add_column("指标", style="cyan")
    table.add_column("值", justify="right")
    table.add_row("消息数", str(len(history)))
    for role in ("system", "user", "assistant", "tool"):
        table.add_row(f"  role={role}", str(role_counts.get(role, 0)))
    other = sum(
        v for k, v in role_counts.items()
        if k not in ("system", "user", "assistant", "tool")
    )
    if other:
        table.add_row("  role=其他", str(other))
    table.add_row("估算 tokens", f"~{total_tokens}")

    state = getattr(agent, "_compress_session_state", None) if agent else None
    if state is not None:
        table.add_row("current_turn", str(getattr(state, "current_turn", 0)))
        table.add_row("llm_compact_count", str(getattr(state, "llm_compact_count", 0)))
        table.add_row("reactive_count", str(getattr(state, "reactive_count", 0)))
    else:
        table.add_row("压缩会话状态", "未初始化")

    pending = getattr(agent, "_pending_ephemeral_messages", None) if agent else None
    table.add_row(
        "pending_ephemeral",
        str(len(pending)) if pending is not None else "0",
    )
    console.print(table)
    return True
# ---------------------------------------------------------------------------
# CCAR11 Task 3 新增：/status + /doctor + /diff 三个命令
# ---------------------------------------------------------------------------

def _status_row(label: str, fn):
    """给 /status 表格安全地造一行：这一格读挂了就显示"读取失败"，
    不拖垮其他行。

    参数：
        label：行标题（如"主模型"）
        fn：取值函数（调用它拿到要显示的内容）

    返回：
        (标题, 值) 二元组；fn 抛异常时值是红色的"读取失败：原因"。
    """
    try:
        return (label, fn())
    except Exception as e:
        logger.debug("/status 段 %s 读取失败: %s", label, e)
        return (label, f"[red]读取失败：{e}[/red]")
def _handle_status_cli(args: str, rt) -> bool:
    """/status 命令：一张表看全当前运行状态（CCAR11 Task 3）。

    背景：排查问题时用户需要一眼看到"用的什么模型、MCP 连没连上"。这张
    Rich 表逐项展示，每一项单独兜底——某一项读挂了只影响那一行：
    - 主模型（rt.config 的 model 段）和 aux LLM（辅助小模型）配没配；
    - goal（目标驱动状态：目标前 30 字 + 状态 + 迭代次数）；
    - 当前项目的记忆键（rt._statusline_project_key，标识记忆隔离用的项目）；
    - MCP 各 server 连接状态（逐个看已连接/断开）；
    - 注册的工具总数。

    参数：
        args：命令参数（本命令不使用）
        rt：RuntimeContext（取 config 和 agent 用）

    返回：
        bool —— True 表示命令已处理
    """
    cfg = getattr(rt, "config", None) or {}
    agent = getattr(rt, "agent", None)

    def _model_desc() -> str:
        m = cfg.get("model", {}) or {}
        return f"{m.get('name', '?')}（provider: {m.get('provider', '?')}）"

    def _aux_desc() -> str:
        aux = getattr(agent, "aux_llm_router", None)
        return "已配置" if aux is not None else "未配置"

    def _goal_desc() -> str:
        gs = getattr(agent, "_goal_state", None)
        if gs is None:
            return "无 active goal"
        obj = str(getattr(gs, "objective", ""))[:30]
        return (
            f"{obj}（status={getattr(gs, 'status', '?')}, "
            f"iter={getattr(gs, 'iteration_count', 0)}）"
        )

    def _project_desc() -> str:
        key = getattr(rt, "_statusline_project_key", "") or ""
        return key if key else "（未获取 / 非 git 项目）"

    def _mcp_desc() -> str:
        from agent.mcp_client import get_mcp_manager
        clients = dict(getattr(get_mcp_manager(), "_clients", {}) or {})
        if not clients:
            return "无已注册 server"
        parts = []
        for name, client in sorted(clients.items()):
            ok = bool(getattr(client, "is_connected", False))
            parts.append(
                f"{name} {'[green]已连接[/green]' if ok else '[red]断开[/red]'}"
            )
        return "  ".join(parts)

    def _tools_desc() -> str:
        from tools.registry import registry
        return str(len(registry.list_all()))

    table = Table(title="Status 一览")
    table.add_column("项", style="cyan")
    table.add_column("值")
    for label, value in (
        _status_row("主模型", _model_desc),
        _status_row("aux LLM", _aux_desc),
        _status_row("goal", _goal_desc),
        _status_row("项目记忆键", _project_desc),
        _status_row("MCP", _mcp_desc),
        _status_row("工具总数", _tools_desc),
    ):
        table.add_row(label, value)
    console.print(table)
    return True
def _handle_doctor_cli(args: str, rt) -> bool:
    """/doctor 命令：给环境做 6 项体检，帮用户定位"为什么跑不起来"。

    背景：类似医生问诊，一项项查常见病因，每项独立判定（一项挂不影响
    其他项继续查），最后给出通过/失败汇总：
    1. 配置能正常加载（load_config 不抛错）；
    2. 模型服务商的 API key 环境变量已设置（读 config 的
       model.api_key_env，默认 DEEPSEEK_API_KEY）；
    3. agent home 目录（~/.OmniMate）可写（写个临时文件再删掉试试）；
    4. .mcp.json 能解析成 JSON（只在文件存在时才查）；
    5. 关键依赖库装齐了没（rich / httpx / openai 逐个 import）；
    6. sessions / skills 目录可用（顺手自动创建，创建了也算通过）。

    参数：
        args：命令参数（本命令不使用）
        rt：RuntimeContext（取 config 和 home 路径用）

    返回：
        bool —— True 表示命令已处理
    """
    import importlib

    cfg = getattr(rt, "config", None) or {}
    home = Path(getattr(rt, "home", None) or get_omnimate_home())
    results = []  # 收集 6 项结果：[(是否通过, 标题, 详情)]

    # 1. config 可加载
    try:
        from config import load_config
        load_config()
        results.append((True, "配置加载", "load_config OK"))
    except Exception as e:
        results.append((False, "配置加载", f"load_config 失败：{e}"))

    # 2. API key env
    try:
        env_name = (cfg.get("model", {}) or {}).get("api_key_env") or "DEEPSEEK_API_KEY"
        if os.environ.get(env_name):
            results.append((True, "API key", f"{env_name} 已设置"))
        else:
            results.append((False, "API key", f"{env_name} 未设置"))
    except Exception as e:
        results.append((False, "API key", f"检查失败：{e}"))

    # 3. agent home 可写
    try:
        probe = home / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        results.append((True, "agent home 可写", str(home)))
    except Exception as e:
        results.append((False, "agent home 可写", f"{home} 不可写：{e}"))

    # 4. .mcp.json 解析（存在才查）
    try:
        mcp_path = home / ".mcp.json"
        if not mcp_path.exists():
            results.append((True, ".mcp.json", "未配置（跳过）"))
        else:
            json.loads(mcp_path.read_text(encoding="utf-8"))
            results.append((True, ".mcp.json", "解析 OK"))
    except Exception as e:
        results.append((False, ".mcp.json", f"解析失败：{e}"))

    # 5. 关键依赖 import
    try:
        missing = []
        for mod in ("rich", "httpx", "openai"):
            try:
                importlib.import_module(mod)
            except Exception:
                missing.append(mod)
        if missing:
            results.append((False, "关键依赖", f"缺失：{', '.join(missing)}"))
        else:
            results.append((True, "关键依赖", "rich / httpx / openai OK"))
    except Exception as e:
        results.append((False, "关键依赖", f"检查失败：{e}"))

    # 6. sessions / skills 目录（自动创建也算 ✓）
    try:
        for d in (home / ".sessions", home / "skills"):
            d.mkdir(parents=True, exist_ok=True)
        results.append((True, "sessions/skills 目录", "存在（必要时已创建）"))
    except Exception as e:
        results.append((False, "sessions/skills 目录", f"创建失败：{e}"))

    for ok, title, detail in results:
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"{mark} {title}：{detail}")
    passed = sum(1 for ok, _, _ in results if ok)
    color = "green" if passed == len(results) else "yellow"
    console.print(
        f"[{color}]汇总：{passed}/{len(results)} 项通过[/{color}]"
    )
    return True
def _show_usage(rt: RuntimeContext):
    """/usage 命令的展示体：当前会话的迭代预算、历史长度和 token 花销。

    背景：用户想知道"这个会话烧了多少 token"。逐块展示（都是
    有数据才显示，出错只写 debug 日志不炸整个命令）：
    - 迭代预算剩余（agent 还能跑几轮）和对话历史条数；
    - LLM token 用量：调用次数、输入/输出/Cache 命中/Cache 写入 tokens；
    - 按模型分账的用量（usage_tracker 存在时才有；只统计 token 不算钱）。

    参数：
        rt：RuntimeContext（取 agent、config、session_store、session_id）

    返回：
        无（直接打印到终端）
    """
    if rt.agent:
        console.print(f"迭代预算剩余: [bold]{rt.agent.iteration_budget.remaining}[/bold]"
                      f"/{rt.agent.iteration_budget.total}")
        console.print(f"对话历史长度: [bold]{len(rt.agent.conversation_history)}[/bold] 条消息")
        # batch1-T2 加的：LLM token 用量统计
        stats = rt.agent.llm_usage_stats
        if stats["total_calls"] > 0:
            console.print(f"\n[bold]LLM Token 用量：[/bold]")
            console.print(f"  调用次数:          [bold]{stats['total_calls']}[/bold]")
            console.print(f"  输入 tokens:       [bold]{stats['total_prompt_tokens']:,}[/bold]")
            console.print(f"  输出 tokens:       [bold]{stats['total_completion_tokens']:,}[/bold]")
            console.print(f"  Cache 命中 tokens: [bold]{stats['total_cache_read_tokens']:,}[/bold]")
            console.print(f"  Cache 写入 tokens: [bold]{stats['total_cache_creation_tokens']:,}[/bold]")
            total_in = stats["total_prompt_tokens"]
            if total_in > 0:
                hit_rate = stats["total_cache_read_tokens"] / total_in * 100
                console.print(f"  Cache 命中率:      [bold]{hit_rate:.1f}%[/bold]")

            # R30f-H8 加的：按模型分账的用量。只在 usage_tracker
            # 被注入时才有；辅助模型与主模型分开计（只统计 token 不算钱）
            tracker = getattr(rt, "usage_tracker", None)
            if tracker is not None:
                try:
                    s = tracker.summary()
                    if s.get("models"):
                        console.print(f"\n[bold]按模型（本会话持久累计）：[/bold]")
                        for name, row in s["models"].items():
                            console.print(
                                f"  [cyan]{name}[/cyan]: {row['calls']} 次 | "
                                f"in {row['prompt']:,} / out {row['completion']:,} | "
                                f"cache r{row['cache_read']:,}/w{row['cache_creation']:,}"
                            )
                except Exception as e:
                    logger.debug("per-model 用量展示失败（fail-open）: %s", e)
    if rt.session_store and rt.session_id:
        info = rt.session_store.get_session(rt.session_id)
        if info:
            console.print(f"当前会话消息数: [bold]{info['message_count']}[/bold]")
def _show_stats(rt: RuntimeContext):
    """/stats 命令的展示体：跨所有会话的汇总统计。

    背景：单看一个会话不够，用户还想看"我总共聊了多少、哪个工具用得最勤"。
    从 session_store 聚合出：会话/消息总数、时间跨度、消息角色分布、
    最长会话 Top 5、工具调用 Top 10（带条形图）。

    参数：
        rt：RuntimeContext（取 session_store、agent、config）

    返回：
        无（直接打印到终端；session_store 未初始化时提示后返回）
    """
    if not rt.session_store:
        console.print("[yellow]session_store 未初始化[/yellow]")
        return

    stats = rt.session_store.get_stats()

    # === 总览 ===
    console.print(Panel(
        f"[cyan]会话总数:[/cyan]  [bold]{stats['sessions']}[/bold]\n"
        f"[cyan]消息总数:[/cyan]  [bold]{stats['messages']}[/bold]\n"
        f"[cyan]最早会话:[/cyan]  {stats['earliest'] or '(无)'}\n"
        f"[cyan]最新会话:[/cyan]  {stats['latest'] or '(无)'}",
        title="[bold]总览[/bold]",
        border_style="blue",
    ))

    # === 角色分布 ===
    if stats["role_distribution"]:
        roles = stats["role_distribution"]
        total = sum(roles.values()) or 1
        lines = []
        for role, cnt in sorted(roles.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            lines.append(f"  {role:10}  [bold]{cnt}[/bold]  [dim]({pct:.1f}%)[/dim]")
        console.print(f"\n[bold]角色分布：[/bold]")
        console.print("\n".join(lines))

    # === Top 5 最长会话 ===
    if stats["top_sessions"]:
        console.print(f"\n[bold]最长会话 Top 5：[/bold]")
        table = Table()
        table.add_column("#", style="dim", justify="right")
        table.add_column("消息数", justify="right")
        table.add_column("模型")
        table.add_column("标题")
        table.add_column("更新时间")
        for i, s in enumerate(stats["top_sessions"], 1):
            title = (s.get("title") or "(无标题)")[:30]
            model = s.get("model") or "?"
            updated = (s.get("updated_at") or "")[:19]
            table.add_row(str(i), str(s.get("message_count", 0)), model, title, updated)
        console.print(table)

    # === 工具调用 Top 10 ===
    if stats["tool_calls"]:
        console.print(f"\n[bold]工具调用 Top {len(stats['tool_calls'])}：[/bold]")
        table = Table()
        table.add_column("#", style="dim", justify="right")
        table.add_column("工具名", style="cyan")
        table.add_column("次数", justify="right")
        max_count = stats["tool_calls"][0]["count"] or 1
        for i, t in enumerate(stats["tool_calls"], 1):
            bar_len = int(t["count"] / max_count * 20)
            bar = "█" * bar_len
            table.add_row(str(i), t["name"], str(t["count"]),
                          f"[dim]{bar}[/dim]")
        console.print(table)
    else:
        console.print(f"\n[dim]暂无工具调用记录[/dim]")
# ---------------------------------------------------------------------------
# CCAR10 Task 3：statusline（每轮 AI 回答结束后在底部打的一行小状态）
# ---------------------------------------------------------------------------
# 设计取舍：等整轮 AI 响应完全输出完（而不是流式输出过程中）才用暗色打
# 一行。不用 rich.Live 的原因：Windows 下 Live 和 input() 抢终端会打架，
# 而且 Live 的"原地刷新"会把滚动历史刷掉。兜底策略：_render_statusline
# 出任何异常都返回空串，主循环看到非空才打印（fail-open）。

def _format_tokens(n: int) -> str:
    """把 token 数缩写成好读的形式：12300 → '12.3K'；1234567 → '1.2M'；0 → '0'。

    背景：statusline 地方小，几万几十万的数字太占宽度。

    参数：
        n：token 数（容错：传了不能转成 int 的东西就当 0）

    返回：
        缩写后的字符串。1000 以下直接显示原数——不然 900 会变成 "0.9K"，
        反而更难读。
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)
