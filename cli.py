"""CLI 交互层（完整版）。

启动时初始化的组件：
  - MemoryStore（MEMORY.md + USER.md，frozen snapshot）
  - MemoryManager（编排器，可选外部 provider）
  - SessionStore（SQLite + FTS5，会话持久化）
  - AIAgent（注入所有组件）
  - curator 检查（should_run_now 后台触发）

支持的 slash 命令：
  /help              显示帮助
  /new               开始新对话（清空历史 + 新建 session）
  /skills            列出所有技能
  /memory            查看当前记忆
  /sessions          列出历史会话
  /search <kw>       搜索历史对话
  /usage             显示工具用量
  /quit              退出
"""

import logging
import os
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import AIAgent
from agent.memory_store import MemoryStore
from agent.memory_manager import MemoryManager
from agent.session_store import SessionStore
from agent.skill_commands import scan_skill_commands, execute_skill
from agent.title_generator import maybe_set_title
from agent.curator import should_run_now, run_curator_review
from config import load_config
from constants import get_agent_home, skills_dir, sessions_db_path
from tools.skill_usage import bump_use, load_usage

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 运行时初始化
# ---------------------------------------------------------------------------

class RuntimeContext:
    """聚合 agent 运行时的所有组件。"""

    def __init__(self):
        self.config = load_config()
        self.home = get_agent_home()
        self.memory_store = None
        self.memory_manager = None
        self.session_store = None
        self.agent = None
        self.session_id = None
        self.skill_commands = {}

    def initialize(self):
        """初始化所有组件。"""
        # 0. 设置权限检查器（注入破坏性命令审批 callback + 持久化白名单）
        try:
            from agent.permission import set_default_checker, PermissionChecker
            from agent.settings import approved_commands_path
            set_default_checker(PermissionChecker(
                approval_callback=_make_approval_callback(),
                whitelist_file=str(approved_commands_path()),
            ))
        except Exception as e:
            logger.debug("权限检查器初始化失败（用默认）: %s", e)

        # 1. 记忆系统
        if self.config.get("memory", {}).get("enabled", True):
            self.memory_store = MemoryStore(
                self.home,
                memory_char_limit=self.config.get("memory", {}).get(
                    "memory_char_limit", 2200),
                user_char_limit=self.config.get("memory", {}).get(
                    "user_char_limit", 1375),
            )
            self.memory_manager = MemoryManager(self.memory_store)

        # 2. 会话存储
        if self.config.get("sessions", {}).get("auto_save", True):
            db = self.config.get("sessions", {}).get("db_path")
            db_path = Path(db) if db else sessions_db_path()
            self.session_store = SessionStore(db_path)

        # 3. 创建会话
        if self.session_store:
            self.session_id = self.session_store.create_session(
                model=self.config["model"]["name"],
                provider=self.config["model"]["provider"],
            )

        # 4. 创建 agent
        self.agent = self._create_agent()

        # 5. 扫描技能命令
        self.skill_commands = scan_skill_commands(skills_dir())

        # 6. 后台触发 curator（不阻塞启动）
        self._maybe_trigger_curator()

    def _create_agent(self) -> AIAgent:
        """根据配置创建 agent。

        优先用 settings.json 里的 api_key 字段；为空时 fallback 到环境变量。
        """
        model_cfg = self.config.get("model", {})
        api_key = model_cfg.get("api_key") or ""

        # Fallback：JSON 里没填 key 时，尝试 provider 专属环境变量
        if not api_key:
            provider = (model_cfg.get("provider") or "").upper()
            api_key_env = model_cfg.get("api_key_env") or ""
            candidates = [
                api_key_env,
                f"{provider}_API_KEY" if provider else None,
            ]
            for cand in candidates:
                if cand and os.environ.get(cand):
                    api_key = os.environ.get(cand)
                    break

        if not api_key:
            console.print("[red]未设置 API key！[/red]")
            console.print(
                f"请在 [bold]{self.home / 'settings.json'}[/bold] 的 "
                f"models.<name>.api_key 填入，或设置环境变量。"
            )
            raise SystemExit(1)

        return AIAgent(
            base_url=model_cfg.get("base_url"),
            api_key=api_key,
            model=model_cfg["name"],
            max_iterations=self.config.get("agent", {}).get("max_iterations", 90),
            enabled_toolsets=self.config.get("enabled_toolsets", ["core"]),
            session_id=self.session_id,
            memory_store=self.memory_store,
            memory_manager=self.memory_manager,
            session_store=self.session_store,
            harvil_home=self.home,
            on_tool_call=_on_tool_call,
            config=self.config,
        )

    def _maybe_trigger_curator(self):
        """后台检查 curator 是否该运行（非阻塞）。"""
        if not self.config.get("curator", {}).get("enabled", True):
            return

        def _check():
            try:
                if should_run_now(skills_dir()):
                    console.print("[dim]后台 curator 触发：整理技能库...[/dim]")
                    run_curator_review(skills_dir())
            except Exception as e:
                logger.debug("curator 触发失败: %s", e)

        # 守护线程，不阻塞主循环
        t = threading.Thread(target=_check, daemon=True)
        t.start()

    def new_session(self):
        """开始新会话。"""
        if self.session_store:
            self.session_id = self.session_store.create_session(
                model=self.config["model"]["name"],
                provider=self.config["model"]["provider"],
            )
            self.agent.session_id = self.session_id
        self.agent.conversation_history = []
        self.agent.invalidate_system_prompt()

    def resume_session(self, session_id: str) -> bool:
        """恢复历史会话：加载消息到 agent.conversation_history。

        agent 的 system prompt 会重建（记忆快照从当前 .md 重新读）。
        返回是否成功。
        """
        if not self.session_store:
            return False

        info = self.session_store.get_session(session_id)
        if not info:
            console.print(f"[red]会话不存在: {session_id}[/red]")
            return False

        msgs = self.session_store.get_messages(session_id)
        # conversation_history 不含 system（system 由 prompt_builder 生成）
        self.agent.conversation_history = [m for m in msgs if m.get("role") != "system"]
        self.session_id = session_id
        self.agent.session_id = session_id
        self.agent.invalidate_system_prompt()

        title = info.get("title") or "(无标题)"
        console.print(
            f"[green][已恢复会话: {title}（{len(msgs)} 条消息）][/green]"
        )

        # 显示最近几条消息让用户看到上下文（不显示 tool 消息，太碎）
        recent = [m for m in msgs[-6:] if (m.get("content") or "").strip()]
        if recent:
            console.print(f"\n[cyan]最近 {len(recent)} 条历史消息：[/cyan]\n")
            for m in recent:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if len(content) > 300:
                    content = content[:297] + "..."
                if role == "user":
                    console.print(f"[bold cyan]你:[/bold cyan] {content}")
                elif role == "assistant":
                    console.print(f"[bold green]AI:[/bold green] {content}")
            console.print()

        return True


# ---------------------------------------------------------------------------
# 回调
# ---------------------------------------------------------------------------

def _make_approval_callback():
    """创建命令审批 callback（破坏性命令触发时询问用户 y/n）。

    同意后自动加入持久化白名单（~/.agent/approved_commands.json），
    跨会话不再询问同样命令。用 /approved 命令管理白名单。
    """
    def callback(command: str) -> bool:
        console.print(f"[yellow]⚠️ 即将执行破坏性命令：[/yellow]")
        console.print(f"[bold]{command}[/bold]")
        try:
            answer = console.input(
                "[bold]允许执行？(y/N):[/bold] [dim]（同意后此命令不再询问）[/dim] ",
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        return answer in ("y", "yes")
    return callback


def _on_tool_call(name: str, args: dict):
    """工具调用时的回调（打印进度）。"""
    if not load_config().get("display", {}).get("show_tool_progress", True):
        return
    short_args = {}
    for k, v in (args or {}).items():
        s = str(v)
        short_args[k] = s if len(s) <= 80 else s[:77] + "..."
    console.print(f"[dim]→ 调用工具: {name} {short_args}[/dim]")


# ---------------------------------------------------------------------------
# Slash 命令处理
# ---------------------------------------------------------------------------

def _handle_command(cmd: str, rt: RuntimeContext) -> bool:
    """处理 slash 命令。返回 True 表示已处理。"""
    parts = cmd.split(None, 1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit"):
        raise SystemExit(0)

    if name == "/help":
        _show_help()
        return True

    if name == "/new":
        rt.new_session()
        console.print("[green][已开始新对话][/green]")
        return True

    if name == "/skills":
        _list_skills(rt)
        return True

    if name == "/memory":
        _show_memory(rt)
        return True

    if name == "/sessions":
        _list_sessions(rt)
        return True

    if name == "/resume":
        _resume_session_interactive(rt, args)
        return True

    if name == "/search":
        if not args:
            console.print("[yellow]用法：/search <关键词>[/yellow]")
            return True
        _search_sessions(rt, args)
        return True

    if name == "/usage":
        _show_usage(rt)
        return True

    if name == "/model":
        _switch_model(rt, args)
        return True

    if name == "/approved":
        _manage_whitelist(rt, args)
        return True

    return False


def _show_help():
    console.print(Panel(
        "[bold]可用命令[/bold]\n\n"
        "[cyan]/new[/cyan]       开始新对话\n"
        "[cyan]/skills[/cyan]    列出技能\n"
        "[cyan]/memory[/cyan]    查看记忆\n"
        "[cyan]/sessions[/cyan]  列出历史会话\n"
        "[cyan]/resume[/cyan]    恢复历史会话（/resume [序号]）\n"
        "[cyan]/search[/cyan]    搜索历史对话（/search <关键词>）\n"
        "[cyan]/usage[/cyan]     显示工具用量\n"
        "[cyan]/model[/cyan]     切换模型（/model [name]）\n"
        "[cyan]/approved[/cyan]  管理审批白名单\n"
        "[cyan]/help[/cyan]      显示本帮助\n"
        "[cyan]/quit[/cyan]      退出\n\n"
        "[dim]输入 /技能名 触发对应技能[/dim]",
        border_style="blue",
    ))


def _list_skills(rt: RuntimeContext):
    sd = skills_dir()
    usage = load_usage(sd)

    table = Table(title="技能列表")
    table.add_column("命令", style="cyan")
    table.add_column("描述")
    table.add_column("使用次数", justify="right")

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
            pass

        table.add_row(f"/{name}", desc, str(rec.get("use_count", 0)))

    if found:
        console.print(table)
    else:
        console.print("[yellow]暂无技能。用 skill_manage 工具创建。[/yellow]")


def _show_memory(rt: RuntimeContext):
    if not rt.memory_store:
        console.print("[yellow]记忆系统未启用[/yellow]")
        return

    console.print("[bold]MEMORY.md（agent 笔记）：[/bold]")
    for entry in rt.memory_store.memory_entries:
        console.print(f"  - {entry}")
    if not rt.memory_store.memory_entries:
        console.print("  [dim]（空）[/dim]")

    console.print("\n[bold]USER.md（用户画像）：[/bold]")
    for entry in rt.memory_store.user_entries:
        console.print(f"  - {entry}")
    if not rt.memory_store.user_entries:
        console.print("  [dim]（空）[/dim]")


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
            empty_id = rt.session_id
            rt.resume_session(history[idx]["id"])
            if empty_id and empty_id != rt.session_id:
                try:
                    rt.session_store.delete_session(empty_id)
                except Exception:
                    pass
        else:
            console.print(f"[yellow]序号超出范围，已开始新对话[/yellow]")


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
    if not rt.session_store:
        console.print("[yellow]会话存储未启用[/yellow]")
        return

    results = rt.session_store.search(query, limit=10)
    if not results:
        console.print(f"[yellow]未找到匹配 '{query}' 的对话[/yellow]")
        return

    console.print(f"[bold]搜索 '{query}' 的结果：[/bold]")
    for r in results:
        title = r.get("title") or "(无标题)"
        snippet = r.get("snippet", "")
        console.print(f"\n[cyan]{title}[/cyan] [dim]({r.get('timestamp', '')[:19]})[/dim]")
        console.print(f"  {snippet}")


def _manage_whitelist(rt: RuntimeContext, args: str):
    """管理审批白名单（/approved）。

    /approved             列出所有已批准命令
    /approved remove <n>  按序号移除
    /approved remove <命令前缀>  按命令移除
    """
    from agent.permission import get_default_checker
    checker = get_default_checker()
    whitelist = checker.list_whitelist()

    if not args.strip():
        if not whitelist:
            console.print("[yellow]白名单为空（破坏性命令每次都会询问）[/yellow]")
        else:
            console.print(f"[bold]已批准命令（{len(whitelist)} 条，不再询问）：[/bold]")
            for i, cmd in enumerate(whitelist):
                # 截断长命令
                display = cmd if len(cmd) <= 80 else cmd[:77] + "..."
                console.print(f"  [{i}] {display}")
        console.print("\n用法：[cyan]/approved remove <序号或命令>[/cyan]")
        return

    parts = args.split(None, 1)
    action = parts[0].lower()
    target = parts[1].strip() if len(parts) > 1 else ""

    if action == "remove" and target:
        if target.isdigit():
            idx = int(target)
            if 0 <= idx < len(whitelist):
                target = whitelist[idx]
            else:
                console.print(f"[red]序号超出范围（0-{len(whitelist) - 1}）[/red]")
                return
        if checker.remove_from_whitelist(target):
            console.print(f"[green]已从白名单移除: {target[:80]}[/green]")
        else:
            console.print(f"[red]白名单中未找到: {target[:80]}[/red]")
    else:
        console.print(f"[yellow]用法：/approved remove <序号或命令>[/yellow]")


def _switch_model(rt: RuntimeContext, args: str):
    """切换当前激活模型。

    /model          列出所有模型
    /model <name>   切换到指定模型
    """
    from agent.settings import list_models, set_default_model, get_current_model_config

    models = list_models()
    if not models:
        console.print("[yellow]未配置任何模型。在 settings.json 的 models 段添加。[/yellow]")
        return

    current = get_current_model_config().get("name")

    if not args.strip():
        # 列出所有模型
        console.print("[bold]可用模型：[/bold]")
        for name, cfg in models.items():
            mark = "[green]*[/green]" if name == current else " "
            fmt = cfg.get("format", "openai")
            model_id = cfg.get("model", "?")
            has_key = "✓" if cfg.get("api_key") else "[red]无 key[/red]"
            console.print(
                f"  {mark} [cyan]{name}[/cyan] "
                f"({fmt}/{model_id}) {has_key}"
            )
        console.print(f"\n[dim]当前: {current}[/dim]")
        console.print("用 [cyan]/model <name>[/cyan] 切换")
        return

    target = args.strip()
    if target not in models:
        console.print(f"[red]未知模型: {target}[/red]")
        console.print(f"可用: {', '.join(models.keys())}")
        return

    if target == current:
        console.print(f"[yellow]已经在用 {target}[/yellow]")
        return

    # 持久化切换
    if not set_default_model(target):
        console.print(f"[red]切换失败[/red]")
        return

    # 重建 agent 的 llm_client
    try:
        from agent.llm_client import create_llm_client
        model_cfg = get_current_model_config()
        if not model_cfg.get("api_key"):
            console.print(f"[red]{target} 未配置 api_key[/red]")
            return
        rt.agent.llm_client = create_llm_client(model_cfg)
        rt.agent.model = model_cfg.get("model", target)
        rt.agent.model_format = model_cfg.get("format", "openai")
        console.print(f"[green][已切换到 {target}（{model_cfg.get('model')}）][/green]")
    except Exception as e:
        console.print(f"[red]重建 client 失败: {e}[/red]")
        logger.exception("切换模型失败")


def _show_usage(rt: RuntimeContext):
    if rt.agent:
        console.print(f"迭代预算剩余: [bold]{rt.agent.iteration_budget.remaining}[/bold]"
                      f"/{rt.agent.iteration_budget.total}")
        console.print(f"对话历史长度: [bold]{len(rt.agent.conversation_history)}[/bold] 条消息")
    if rt.session_store and rt.session_id:
        info = rt.session_store.get_session(rt.session_id)
        if info:
            console.print(f"当前会话消息数: [bold]{info['message_count']}[/bold]")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _show_history_messages(rt: RuntimeContext, limit: int = 6):
    """恢复会话后，回放最近 N 条消息让用户看到上下文。"""
    if not rt.session_store or not rt.session_id:
        return
    msgs = rt.session_store.get_messages(rt.session_id, limit=limit)
    if not msgs:
        return

    console.print(f"\n[cyan]最近 {len(msgs)} 条历史消息：[/cyan]\n")
    for msg in msgs:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # 截断长消息
        if len(content) > 300:
            content = content[:297] + "..."

        if role == "user":
            console.print(f"[bold cyan]你:[/bold cyan] {content}")
        elif role == "assistant":
            console.print(f"[bold green]AI:[/bold green] {content}")
        # tool 消息跳过（太长太碎）
    console.print()


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
    empty_id = rt.session_id
    rt.resume_session(last["id"])
    if empty_id and empty_id != rt.session_id:
        try:
            rt.session_store.delete_session(empty_id)
        except Exception:
            pass
    # 显示历史消息
    _show_history_messages(rt)


def run_interactive(resume_last: bool = False):
    """启动交互式 CLI。

    参数：
        resume_last: True 时自动恢复最近会话（-c/--continue 触发）；
                     False 时提示用户选择。
    """
    console.print(Panel(
        "[bold blue]自学习 AI Agent[/bold blue]\n"
        "输入消息开始对话。[cyan]/help[/cyan] 查看命令，[cyan]/quit[/cyan] 退出。",
        border_style="blue",
    ))

    try:
        rt = RuntimeContext()
        rt.initialize()
    except SystemExit:
        return
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        logger.exception("初始化失败")
        return

    # 启动时恢复历史会话
    if resume_last:
        _auto_resume_last(rt)
    else:
        _maybe_prompt_resume(rt)

    # 主循环
    while True:
        try:
            user_input = console.input("[bold cyan]你:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            break

        if not user_input:
            continue

        # 1. 处理 slash 命令
        if user_input.startswith("/"):
            # 先检查是否是技能命令
            cmd_name = user_input.split()[0]
            if cmd_name in rt.skill_commands:
                pass  # 走技能触发逻辑
            elif _handle_command(user_input, rt):
                continue

        # 2. 检查是否触发技能
        cmd_name = user_input.split()[0] if " " in user_input else user_input
        if cmd_name in rt.skill_commands:
            skill_info = rt.skill_commands[cmd_name]
            rest_msg = user_input[len(cmd_name):].strip()
            user_input = execute_skill(
                skill_info["skill_md_path"],
                rest_msg or "(执行此技能)",
            )
            # 记录技能使用
            bump_use(skills_dir(), skill_info["name"])
            console.print(f"[dim][已触发技能: {skill_info['name']}][/dim]")

        # 3. 保存用户消息到 session
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "user", user_input)

            # 第一轮后自动生成标题
            session_info = rt.session_store.get_session(rt.session_id)
            maybe_set_title(
                rt.session_store, rt.session_id,
                user_input,
                session_info.get("title") if session_info else None,
            )

        # 4. 调用 agent
        try:
            response = rt.agent.run_conversation(user_input)
            console.print("[bold green]AI:[/bold green]")
            console.print(response)

            # 5. 保存助手响应到 session
            if rt.session_store and rt.session_id:
                rt.session_store.append_message(
                    rt.session_id, "assistant", response,
                )
        except KeyboardInterrupt:
            rt.agent.interrupt()
            console.print("[yellow]\n[已中断][/yellow]")
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            logger.exception("agent 运行错误")


def run_one_shot(message: str):
    """非交互模式：发一条消息，打印响应。"""
    try:
        rt = RuntimeContext()
        rt.initialize()
    except SystemExit:
        return
    except Exception as e:
        print(f"初始化失败: {e}", file=sys.stderr)
        return

    try:
        response = rt.agent.run_conversation(message)
        print(response)
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "user", message)
            rt.session_store.append_message(rt.session_id, "assistant", response)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        logger.exception("one-shot 运行错误")
