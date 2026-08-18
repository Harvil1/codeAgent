"""诊断/状态命令簇（R30 从 cli.py 机械抽离，行为不变）。

/status /doctor /context /compact /usage /stats 相关处理函数。
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
# CCAR11 Task 2 NEW: /compact + /context
# ---------------------------------------------------------------------------

def _sync_history_after_compact(agent, new_messages: list) -> None:
    """压缩后同步 agent.conversation_history + 失效 system prompt 缓存。

    对齐 AIAgent._run_context_compression 的收尾逻辑：
    - strip 头部 system（压缩函数输入是纯 history，正常无 system，防御性处理）
    - strip ``_ephemeral`` 标记消息（ephemeral 不进持久化 history）
    - invalidate_system_prompt（前缀已变，必须重建，否则下次发 LLM 的
      system prompt 还是旧缓存）
    """
    msgs = list(new_messages or [])
    if msgs and msgs[0].get("role") == "system":
        msgs = msgs[1:]
    agent.conversation_history = [m for m in msgs if not m.get("_ephemeral")]
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
    """打印压缩前后 token 对比。"""
    saved = max(0, before_tokens - after_tokens)
    console.print(f"[green]压缩完成（{mode}）[/green]")
    console.print(
        f"  消息数 {before_msgs} → {after_msgs}，"
        f"估算 tokens ~{before_tokens} → ~{after_tokens}"
        f"（节省 ~{saved}）"
    )
def _handle_compact_cli(args: str, rt) -> bool:
    """/compact 手动触发 L4 压缩（CCAR11 Task 2）。

    - ``--yes`` 跳过确认，否则交互确认（EOF/输入异常视为拒绝）
    - 正常路径：llm_compact 强制走 L4（token_threshold=0 绕过自动阈值判定，
      手动压缩语义 = "现在就压"，与自动压缩的"接近窗口才压"不同）
    - LLM 不可用（client None / 调用异常 / 事件循环冲突）→ 降级 snip_compact
      （无损裁剪，threshold=0 强制）+ 提示
    - 压缩后同步 agent 状态 + 记录 llm_compact_count（对齐自动压缩收尾）
    """
    agent = getattr(rt, "agent", None)
    if agent is None:
        console.print("[yellow]Agent 未初始化，无法压缩[/yellow]")
        return True

    # --yes 跳过确认，否则交互确认（EOF/异常视为拒绝）
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
                token_threshold=0,  # 强制触发（手动压缩不受自动阈值限制）
                precomputed_tokens=before_tokens,  # 空历史时 0>0 自然短路
            ))
        except RuntimeError:
            # 已在事件循环内（嵌入环境）——降级 snip（对齐 /init 的桥接模式）
            console.print("[yellow]事件循环冲突，降级为 snip_compact（无损裁剪）[/yellow]")
        except Exception as e:
            console.print(f"[red]LLM 压缩失败：[/red]{e}")

    if compacted:
        # L4 成功：同步 agent 状态（对齐 _run_context_compression 收尾）
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

    # 降级路径：snip_compact（无损，threshold=0 强制触发）
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
    """/context 显示当前上下文 token 分布表（CCAR11 Task 2）。

    Rich Table 展示：
    - role 分布（system/user/assistant/tool 计数）
    - estimate_message_tokens 估算总量
    - 压缩会话状态（current_turn / llm_compact_count / reactive_count）
    - _pending_ephemeral_messages 长度（待注入的临时消息）
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
# CCAR11 Task 3 NEW: /status + /doctor + /diff
# ---------------------------------------------------------------------------

def _status_row(label: str, fn):
    """安全构建 /status 表格行：段内异常 → 显示"读取失败"，不影响其他段。"""
    try:
        return (label, fn())
    except Exception as e:
        logger.debug("/status 段 %s 读取失败: %s", label, e)
        return (label, f"[red]读取失败：{e}[/red]")
def _handle_status_cli(args: str, rt) -> bool:
    """/status 状态一览（CCAR11 Task 3）。

    Rich Table 展示（每段独立 try，一段挂了不影响其他段）：
    - 主模型（rt.config model 段）/ aux LLM 有无
    - goal（objective 前 30 字 + status + iteration）
    - 当前项目记忆键（rt._statusline_project_key）
    - MCP 各 server 连接状态（get_mcp_manager 遍历 _clients 的 is_connected）
    - 工具总数（registry.list_all）
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
    """/doctor 自诊断 6 项（CCAR11 Task 3，fail-open 每项独立）。

    1. load_config() 成功
    2. provider API key env 已设置（config model.api_key_env，默认 DEEPSEEK_API_KEY）
    3. agent home 可写（tmp 文件写删）
    4. .mcp.json 可解析（存在才查）
    5. 关键依赖可 import（rich / httpx / openai）
    6. sessions / skills 目录可用（自动创建也算 ✓）
    """
    import importlib

    cfg = getattr(rt, "config", None) or {}
    home = Path(getattr(rt, "home", None) or get_omnimate_home())
    results = []  # [(通过?, 标题, 详情)]

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
    if rt.agent:
        console.print(f"迭代预算剩余: [bold]{rt.agent.iteration_budget.remaining}[/bold]"
                      f"/{rt.agent.iteration_budget.total}")
        console.print(f"对话历史长度: [bold]{len(rt.agent.conversation_history)}[/bold] 条消息")
        # batch1-T2: LLM token 用量统计
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

            # R30f-H8：per-model 用量 + 成本（tracker 注入时才有；aux 模型
            # 与主模型分开计，金额复用 agent/pricing.py 价表）
            tracker = getattr(rt, "usage_tracker", None)
            if tracker is not None:
                try:
                    s = tracker.summary()
                    if s.get("models"):
                        console.print(f"\n[bold]按模型（本会话持久累计）：[/bold]")
                        for name, row in s["models"].items():
                            cost = (f"  ${row['cost_usd']:.4f}"
                                    if "cost_usd" in row else "  $?")
                            console.print(
                                f"  [cyan]{name}[/cyan]: {row['calls']} 次 | "
                                f"in {row['prompt']:,} / out {row['completion']:,} | "
                                f"cache r{row['cache_read']:,}/w{row['cache_creation']:,}"
                                f"{cost}"
                            )
                except Exception as e:
                    logger.debug("per-model 用量展示失败（fail-open）: %s", e)

            # 成本估算（批次 2：A3）
            try:
                from agent.pricing import estimate_cost_usd
                model_cfg = rt.config.get("model", {})
                est = estimate_cost_usd(
                    provider=model_cfg.get("provider", ""),
                    model=model_cfg.get("name", ""),
                    prompt_tokens=stats["total_prompt_tokens"],
                    completion_tokens=stats["total_completion_tokens"],
                    cache_read_tokens=stats["total_cache_read_tokens"],
                    cache_creation_tokens=stats["total_cache_creation_tokens"],
                )
                if est is not None:
                    console.print(f"\n[bold]成本估算：[/bold]")
                    console.print(f"  总成本:    [bold green]${est['cost_usd']:.4f}[/bold green]")
                    bk = est["breakdown"]
                    console.print(
                        f"  [dim]输入:     ${bk['input']:.4f}"
                        f" | Cache 命中: ${bk['cache_hit']:.4f}"
                        f" | Cache 写入: ${bk['cache_write']:.4f}"
                        f" | 输出: ${bk['output']:.4f}[/dim]"
                    )
                else:
                    console.print(
                        f"\n[dim]成本估算：未知模型 "
                        f"{model_cfg.get('provider', '?')}/"
                        f"{model_cfg.get('name', '?')}（pricing.py 未收录）[/dim]"
                    )
            except Exception as e:
                logger.debug("成本估算失败（fail-open）: %s", e)
    if rt.session_store and rt.session_id:
        info = rt.session_store.get_session(rt.session_id)
        if info:
            console.print(f"当前会话消息数: [bold]{info['message_count']}[/bold]")
def _show_stats(rt: RuntimeContext):
    """跨会话聚合统计：会话/消息/工具调用/角色分布 + 当前会话成本。"""
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

    # === 当前会话成本（复用 pricing） ===
    if rt.agent:
        agent_stats = rt.agent.llm_usage_stats
        if agent_stats.get("total_calls", 0) > 0:
            try:
                from agent.pricing import estimate_cost_usd
                model_cfg = rt.config.get("model", {})
                est = estimate_cost_usd(
                    provider=model_cfg.get("provider", ""),
                    model=model_cfg.get("name", ""),
                    prompt_tokens=agent_stats.get("total_prompt_tokens", 0),
                    completion_tokens=agent_stats.get("total_completion_tokens", 0),
                    cache_read_tokens=agent_stats.get("total_cache_read_tokens", 0),
                    cache_creation_tokens=agent_stats.get("total_cache_creation_tokens", 0),
                )
                if est:
                    console.print(
                        f"\n[bold]当前会话成本：[/bold] "
                        f"[bold green]${est['cost_usd']:.4f}[/bold green] "
                        f"[dim]({agent_stats['total_calls']} 次调用)[/dim]"
                    )
            except Exception as e:
                logger.debug("stats 成本估算失败: %s", e)
# ---------------------------------------------------------------------------
# CCAR10 Task 3: statusline（每轮尾部状态行）
# ---------------------------------------------------------------------------
# 设计：在每轮 AI 响应完全输出后（不是流式中）console.print 一行 dim。
# 不用 rich.Live —— Windows + input() 冲突，且 Live 会刷掉滚动历史。
# fail-open：_render_statusline 任何异常返回 ""，主循环 if line 才打印。

def _format_tokens(n: int) -> str:
    """token 数格式化。12300 → '12.3K'；1234567 → '1.2M'；0 → '0'。

    1000 以下直接显示原数，避免 "0.9K" 这种短数过度缩写。
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
