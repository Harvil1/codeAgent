"""CLI（command-line interface，命令行界面——用户在黑窗口里打字交互）主入口。

这个文件是整个程序的"前台总调度"：main() 在这里解析命令行参数，决定是否
自动恢复最近会话；RuntimeContext 类在这里把所有零件
（记忆、会话存档、AI 本体、定时器、后台任务……）装配到一起。

启动时初始化的组件：
  - MemoryStore（记忆仓库：多个 JSONL 文件存记忆条目，MEMORY.md 当索引；
    对话中按需临时注入，注入完就丢、不留在历史里）
  - MemoryManager（记忆编排器，可以接外部 provider）
  - SessionStore（会话存档：JSONL 文件保存对话；传 .db 路径会自动落到
    .sessions/ 目录）
  - AIAgent（AI 本体，所有组件都塞给它）
  - curator 检查（知识库"维护工人"，到期后在后台线程自动整理）

支持的 slash 命令（斜杠开头的特殊指令，如 /help）：
  /help              显示帮助
  /new               开始新对话（清空历史 + 新建 session）
  /skills            列出所有技能
  /memory            查看当前记忆
  /sessions          列出历史会话
  /search <kw>       搜索历史对话
  /usage             显示工具用量
  /quit              退出
"""

import asyncio
import contextvars
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import AIAgent
from agent.memory_store import MemoryStore
from agent.memory_manager import MemoryManager
from agent.session_store import SessionStore
from agent.skill_commands import scan_skill_commands, execute_skill, scan_bundle_commands, execute_bundle
from agent.title_generator import maybe_set_title
from agent.curator import should_run_now, run_curator_review
from config import load_config
from constants import get_omnimate_home, skills_dir, sessions_db_path, all_skills_dirs
from tools.skill_usage import bump_use, load_usage
from agent.handoff import (
    HandoffStore,
    HandoffError,
    BundleNotFoundError,
    AmbiguousBundleIDError,
    SecretDetectedError,
    BundleTooLargeError,
    BundleCorruptedError,
)

from cli_ui import console

logger = logging.getLogger(__name__)


from cli_session_cmds import (  # noqa: F401（回导入：测试/内部引用兼容）
    _list_sessions,
    _maybe_prompt_resume,
    _resume_and_cleanup_empty,
    _resume_session_interactive,
    _search_sessions,
    _auto_resume_last,
    _handle_resume_command,
    _show_history_messages,
    _print_message_list,
)
from cli_skill_memory_cmds import (  # noqa: F401（回导入：测试/内部引用兼容）
    _handle_skills_command,
    _list_skills,
    _rate_skill,
    _recommend_skills,
    _show_memory,
    _quick_save_memory,
    _handle_skill_learning_command,
    _open_in_editor,
)
from cli_diag_cmds import (  # noqa: F401（回导入：测试/内部引用兼容）
    _status_row,
    _handle_status_cli,
    _handle_doctor_cli,
    _handle_context_cli,
    _handle_compact_cli,
    _print_compact_delta,
    _sync_history_after_compact,
    _format_tokens,
    _show_usage,
    _show_stats,
)


# ---------------------------------------------------------------------------
# 启动期校验/容错
# ---------------------------------------------------------------------------

def _validate_model_config(config: dict) -> None:
    """启动最早检查模型配置齐不齐，缺了就用红字告诉用户怎么补，然后退出。

    缺 name/provider 或值为空 → SystemExit(2)（带退出码 2 的程序终止）
    + 红字提示该编辑哪个文件、填什么（避免裸 KeyError traceback 让用户
    不知道改哪里）。配置完整就什么都不做。

    参数：
        config: 整个配置字典（load_config() 读出来的）
    """

    model_cfg = config.get("model") or {}
    if not isinstance(model_cfg, dict):
        console.print("[red]配置错误：settings.json 的 model 字段不是对象[/red]")
        raise SystemExit(2)

    name = model_cfg.get("name") or ""
    provider = model_cfg.get("provider") or ""

    if not name:
        console.print(
            "[red]配置错误：settings.json 缺少 model.name（模型名）[/red]\n"
            f"[dim]请编辑 [bold]{get_omnimate_home() / 'settings.json'}[/bold] "
            "填 models.<name>.name，例如 \"deepseek-chat\"[/dim]"
        )
        raise SystemExit(2)
    if not provider:
        console.print(
            f"[red]配置错误：settings.json 缺少 model.provider（厂商）[/red]\n"
            f"[dim]model.name={name} 已就绪，请补 provider，如 \"deepseek\"/\"anthropic\"[/dim]"
        )
        raise SystemExit(2)


def _run_memory_curator_once(memory_dir, *, config: dict, store=None) -> None:
    """记忆维护工人（memory curator）跑一轮，无论成败都把"上次运行时间"写盘。

    try/finally 兜底：跑完、跑挂都先写盘（否则中途崩掉，下次启动会
    重复跑一遍，白烧 LLM tokens）。

    store 参数要直接用主 agent 的记忆实例——
    两个实例各拿各的锁，锁就不生效了（跨实例 race）。

    参数：
        memory_dir: 记忆数据目录（~/.OmniMate/.memory）
        config: 配置字典（顺便通过它把 review agent 工厂传进来，见下）
        store: 主 agent 的 MemoryStore 实例（None 时第 1 阶段自己建）
    """
    import datetime
    from agent.memory_curator import (
        apply_automatic_transitions,
        load_memory_curator_state,
        save_memory_curator_state,
        run_memory_review,
    )

    state = load_memory_curator_state(memory_dir)
    review_summary = ""

    try:
        counts = apply_automatic_transitions(memory_dir, store=store)
        review_summary = f"第 1 阶段: {counts}"

        # 第 2 阶段（主模型 review）——失败时保留第 1 阶段结果
        try:
            # factory 由调用方在 RuntimeContext 注入；这里若没注入就跳过
            # 通过参数透传：本函数只负责"主体逻辑 + state 保存"
            # （漏传会导致第 2 阶段静默失效）
            factory = config.pop("_curator_factory", None) if isinstance(config, dict) else None
            if factory is not None:
                review_report = run_memory_review(
                    memory_dir, agent_factory=factory, config=config,
                )
                review_summary = (
                    f"第 1 阶段: {counts}; "
                    f"第 2 阶段: reviewed={review_report['buckets_reviewed']}, "
                    f"actions={review_report['executed_actions']}, "
                    f"errors={review_report['errors']}"
                )
        except Exception as e:
            logger.warning("第 2 阶段失败(保留第 1 阶段结果): %s", e)
            review_summary = f"第 1 阶段: {counts}; 第 2 阶段失败: {e}"

    except Exception as e:
        # 第 1 阶段就崩——也要把失败原因记进 state，方便排查
        logger.warning("Memory Curator 第 1 阶段失败: %s", e)
        review_summary = f"第 1 阶段失败: {e}"

    finally:
        # 无论中途是否崩，state 都写盘（防重复跑）
        state["last_run_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        state["last_run_summary"] = review_summary
        try:
            save_memory_curator_state(memory_dir, state)
            logger.info("Memory Curator state 已保存: %s", review_summary)
        except Exception as save_err:
            # 写盘本身失败只能记日志，不能因此卡住整个程序
            logger.error("Memory Curator state 保存失败: %s", save_err)


# ---------------------------------------------------------------------------
# 运行时初始化
# ---------------------------------------------------------------------------

def _thread_llm_client(config: dict):
    """给"后台线程"专用的 LLM 连接对象：每次调用独立建连，跨事件循环安全。

    主 LLM client 绑定在主线程的事件循环上，后台线程（如 curator）借用它
    会在别的循环里调异步代码而出错，所以用 ThreadedLLMClient——每次请求
    单独建连接，跟哪个循环都不绑定。

    注意：这里是主 client 构造的"镜像"。改 RuntimeContext 里主 client
    的构造逻辑时，这里要同步改，否则两边的模型配置会对不上。

    参数：
        config: 配置字典（从 model 段取连接参数）
    """
    from agent.llm_client import ThreadedLLMClient
    mc = (config or {}).get("model", {}) or {}
    # api_key 推导逻辑与主 client 完全同源（见 RuntimeContext._derive_api_key），
    # 保证线程拿到的凭证跟主连接一致
    api_key = RuntimeContext._derive_api_key(mc)
    return ThreadedLLMClient({
        "format": mc.get("format", "openai"),
        "base_url": mc.get("base_url"),
        "model": mc.get("name"),
        "api_key": api_key,
        "auth_token": mc.get("auth_token") or "",
    })


class RuntimeContext:
    """把 agent 运行时要用的所有零件装到一个筐里，随身携带。

    为什么需要它：组件多（配置、记忆、会话、AI 本体、定时器……），
    分散在各处容易漏传、顺序错乱。统一在 __init__ / initialize 里
    按依赖顺序装配，之后到处只传这一个对象。
    """

    def __init__(self):
        """只做"轻量启动"：读配置 + 建不依赖别人的组件。重的活留给 initialize()。"""
        self.config = load_config()
        self.home = get_omnimate_home()
        # === /add-dir 持久化白名单启动加载 ===
        # 把 settings.json 里 security.extra_allowed_roots 记过的额外可写目录
        # 灌进运行时白名单。宽松失败策略（fail-open）：某一条坏了就跳过，
        # 不让程序起不来。
        try:
            n = _load_persisted_extra_roots(self.config)
            if n:
                logger.info("已从 settings.json 加载 %d 个额外白名单目录", n)
        except Exception as e:
            logger.warning("加载 extra_allowed_roots 失败（跳过）: %s", e)
        self.memory_store = None
        self.memory_manager = None
        self.session_store = None
        self.agent = None
        self.session_id = None
        self.skill_commands = {}
        self.bundle_commands = {}  # 技能束斜杠命令：一个命令连发多个技能
        self.quit_requested = False  # 用户敲了 /quit 的标志（走正常关机流程，不是硬杀）
        self.checkpoint_mgr = None  # Checkpoint 管理器：文件快照/回滚
        # === Hooks 系统（事件钩子——特定事件发生时自动执行用户配置的动作）===
        from agent.hooks import HookRegistry
        self.hooks_registry = HookRegistry()

        # === 后台任务管理器 ===
        from agent.background import BackgroundManager
        bg_cfg = self.config.get("bg_task", {})
        self.bg_manager = BackgroundManager(
            max_concurrent=bg_cfg.get("max_concurrent", 5),
            notification_stdout_cap=bg_cfg.get("notification_stdout_cap", 500),
            result_stdout_cap=bg_cfg.get("result_stdout_cap", 5000),
            default_timeout=bg_cfg.get("default_timeout", 600),
            stall_timeout=bg_cfg.get("stall_timeout", 45.0),  # 停滞看门狗——45 秒没有新输出就通知用户（防"卡死没动静"）
        )

        # === Cron 调度器（定时任务，像闹钟一样到点触发）===
        from agent.cron import CronScheduler
        cron_cfg = self.config.get("cron", {})
        if cron_cfg.get("enabled", True):
            cron_path = cron_cfg.get("jobs_path")
            if cron_path is None:
                cron_path = Path(self.home) / ".cron" / "jobs.json"
            try:
                self.cron_scheduler = CronScheduler(
                    jobs_path=Path(cron_path),
                    poll_interval_seconds=cron_cfg.get("poll_interval_seconds", 30.0),
                    enabled=True,
                    max_age_days=cron_cfg.get("max_age_days", 7),  # 循环任务过期天数
                )
                self.cron_scheduler.start()
            except Exception as e:
                logger.error("CronScheduler 启动失败: %s", e)
                self.cron_scheduler = None
        else:
            self.cron_scheduler = None

        # === Agent Teams（多代理协作：一个"队长"带多个队友分工）===
        self.team_bus = None
        self.team_coordinator = None
        team_cfg = self.config.get("team", {})
        if team_cfg.get("enabled", True):
            team_path = team_cfg.get("team_dir") or (
                Path(self.home) / ".team"
            )
            try:
                from agent.team.bus import MessageBus
                from agent.team.coordinator import TeamCoordinator
                self.team_bus = MessageBus(team_dir=Path(team_path))
                self.team_coordinator = TeamCoordinator(
                    team_dir=Path(team_path),
                    omnimate_home=Path(self.home),
                    config=self.config,
                )
                # 主 agent 把自己注册成队长（仅当名册里还没有活着的 main 时才注册）
                members = self.team_coordinator.list_members()
                if not any(m.name == "main" and m.status == "running" for m in members):
                    try:
                        self.team_coordinator.register(
                            name="main", role="lead", status="running",
                        )
                    except ValueError:
                        pass  # 已存在，跳过
            except Exception as e:
                logger.error("Team 系统初始化失败: %s", e)
                self.team_bus = None
                self.team_coordinator = None

        # === Handoff bundle 存储（会话移交包裹：把当前对话打包存档/带走）===
        self.handoff_store = None  # 在 initialize() 中真正初始化

        # === mailbox + agent_name + trace_sink ===
        # mailbox：队友间的异步邮箱
        # agent_name：当前 agent 的名字（邮箱收件人，默认 "main"）
        # trace_sink：本地运行轨迹记录器（/trace 命令读这个）
        # 这些字段先置 None，在 initialize() 中真正填充
        # 注意：不要在这加 aux_llm_client / aux_model 之类的字段——
        # 没人读的死字段；辅助模型调用统一走 agent_ref.aux_llm_router
        self.mailbox = None
        self.agent_name = "main"
        self.trace_sink = None
        self._poor_mode_on = False  # /poor 省钱模式的开关状态
        # 状态行用的项目分区键（initialize 里赋值一次，取不到就空着不报错）
        self._statusline_project_key = None

        # 用 atexit 兜底关机——即使主循环崩了或 SystemExit，也会清理锁文件等资源
        import atexit
        atexit.register(self.shutdown)

    def _make_memory_review_agent_factory(self):
        """造一个"工厂函数"：调用它就新建一个专门做记忆复审的 AI 实例。

        记忆维护的第 2 阶段要 LLM 复审记忆内容，用一次性实例（不借用主
        对话的 agent，避免污染主对话历史）：

        - 用主模型
        - 不带任何工具（纯文本问答，LLM 输出 YAML 指令、由 Python 执行）
        - 独立的消息历史（跟主对话完全隔离）

        参数：无。
        返回：一个零参函数，调用即返回新的 AIAgent 实例。
        """

        def factory():
            from agent import AIAgent
            from agent.settings import load_settings, get_current_model_config

            try:
                settings = load_settings()
                model_cfg = get_current_model_config(settings)
                return AIAgent(
                    base_url=model_cfg.get("base_url"),
                    api_key=model_cfg.get("api_key", ""),
                    auth_token=model_cfg.get("auth_token", ""),
                    effort_level=model_cfg.get("effort_level"),
                    model=model_cfg.get("model", ""),
                    model_format=model_cfg.get("format", "anthropic"),
                    enabled_toolsets=[],  # 不给工具，纯文本交互
                    omnimate_home=self.home,
                    config=self.config,
                )
            except Exception as e:
                logger.warning("构造 memory review agent 失败: %s", e)
                raise

        return factory

    def initialize(self):
        """按依赖顺序把所有组件真正建起来（先存储、再 agent、最后后台服务）。

        参数：无。返回：无（结果都挂在 self 的字段上）。
        """
        # 启动最早校验 model.name/provider（缺字段时友好提示）
        _validate_model_config(self.config)

        # 0. 设置权限检查器（注入"破坏性命令要问用户"的回调 + 持久化白名单）
        # perm_mode 放 try 外面定义，因为后面构造 AIAgent 时还要用
        perm_mode = self.config.get("security", {}).get("permission_mode", "default")
        try:
            from agent.permission import set_default_checker, PermissionChecker
            from agent.settings import approved_commands_path, approved_paths_path
            set_default_checker(PermissionChecker(
                approval_callback=_make_approval_callback(
                    # 审批时按 e 可以让辅助模型解释命令。
                    # 延迟获取——aux_llm_router 要到更晚才构造好
                    aux_provider=lambda: getattr(self, "aux_llm_router", None),
                ),
                whitelist_file=str(approved_commands_path()),
                paths_whitelist_file=str(approved_paths_path()),
                mode=perm_mode,
                hooks_registry=self.hooks_registry,  # 权限审计 hook（每次权限判定都留痕）
            ))
            # 把 config["security"]["sandbox_mode"] 灌进检查器
            from agent.permission import get_default_checker
            _checker = get_default_checker()
            if _checker is not None:
                sandbox_mode = self.config.get("security", {}).get("sandbox_mode", "off")
                if sandbox_mode in ("off", "on"):
                    _checker.set_sandbox_mode(sandbox_mode)
        except Exception as e:
            logger.debug("权限检查器初始化失败（用默认）: %s", e)

        # 0.5 加载声明式 hooks（用户在配置文件里声明的事件钩子；如果启用）
        if self.config.get("hooks", {}).get("enabled", True):
            from agent.hook_loader import load_declarative_hooks
            settings_path = self.config.get("hooks", {}).get("settings_path")
            if settings_path is None:
                settings_path = Path(self.home) / ".hooks" / "settings.json"
            else:
                settings_path = Path(settings_path)
            try:
                n = load_declarative_hooks(self.hooks_registry, settings_path)
                if n > 0:
                    logger.info("加载了 %d 个声明式 hooks 自 %s", n, settings_path)
            except Exception as e:
                logger.error("加载声明式 hooks 失败: %s", e)

        # 1. 记忆系统
        # 1a. 先建 SessionStore（会话存档库；记忆/任务的自动双写也要用到它）
        if self.config.get("sessions", {}).get("auto_save", True):
            db = self.config.get("sessions", {}).get("db_path")
            db_path = Path(db) if db else sessions_db_path()
            self.session_store = SessionStore(db_path)

        # 1b. MemoryStore（记忆仓库，纯文件存储，无 SQLite 双写）
        if self.config.get("memory", {}).get("enabled", True):
            self.memory_store = MemoryStore(
                omnimate_home=self.home,
            )
            # memory_manager 的 LLM 连接等 agent 建好后再注入
            # （要跟 aux_llm_router 共用同一个）
            self.memory_manager = MemoryManager(self.memory_store)

        # 1c. TaskStore 全局单例（任务清单存储，纯文件）
        try:
            from agent.task_store import get_task_store
            get_task_store(self.home)
        except Exception as e:
            logger.warning("TaskStore 初始化失败: %s", e)

        # 3. 创建会话
        if self.session_store:
            self.session_id = self.session_store.create_session(
                model=self.config["model"]["name"],
                provider=self.config["model"]["provider"],
            )

        # Checkpoint：文件快照/回滚（按会话隔离）
        try:
            from agent.checkpoint import CheckpointManager
            ckpt_root = Path(self.home) / ".checkpoints"
            self.checkpoint_mgr = CheckpointManager(
                ckpt_root, self.session_id or "nosession",
                max_snapshots=self.config.get("checkpoint", {}).get(
                    "max_snapshots", 100),
            )
        except Exception as e:
            logger.warning("CheckpointManager 初始化失败: %s", e)
            self.checkpoint_mgr = None

        # 4. 创建 agent（AI 本体）
        self.agent = self._create_agent()
        # 按模型分别记账的用量追踪（注入 agent；/usage 按模型展示）
        try:
            from agent.usage_tracker import UsageTracker
            self.usage_tracker = UsageTracker(
                self.home, self.session_id or "default",
            )
            self.agent.set_usage_tracker(self.usage_tracker)
        except Exception as e:
            logger.warning("UsageTracker 初始化失败（fail-open）: %s", e)
            self.usage_tracker = None

        # 5. 扫描技能命令（内置 + 用户两个目录都扫，同名时用户目录优先）
        self.skill_commands = scan_skill_commands(all_skills_dirs())
        # 扫描技能束命令（只在用户目录扫，不扫内置）
        self.bundle_commands = scan_bundle_commands(skills_dir())

        # 7. Handoff 存储
        try:
            handoff_dir = Path(self.home) / ".handoff"
            self.handoff_store = HandoffStore(handoff_dir)
        except Exception as e:
            logger.warning("HandoffStore 初始化失败: %s", e)
            self.handoff_store = None

        # === 初始化 mailbox + trace_sink，注入 agent ===
        # mailbox 接线：用 team 目录，跟 team_bus 共享同一个邮箱根目录
        try:
            from agent.team.mailbox import Mailbox
            mb_dir = Path(self.home) / ".team"
            mb_dir.mkdir(parents=True, exist_ok=True)
            self.mailbox = Mailbox(mb_dir)
            self.agent_name = "main"
            # 把 mailbox + agent_name 挂到 AIAgent（mailbox 工具通过 agent_ref 读这两个）
            self.agent.set_mailbox(self.mailbox, self.agent_name)
            logger.info("mailbox 已注入 AIAgent（agent_name=%s）", self.agent_name)
        except Exception as e:
            logger.warning("mailbox 初始化失败（fail-open）: %s", e)
            self.mailbox = None

        # trace_sink 接线：从 config.trace.enabled 读开关
        trace_cfg = self.config.get("trace", {})
        if trace_cfg.get("enabled", True):
            try:
                from agent.trace import TraceSink
                self.trace_sink = TraceSink(base_dir=Path(self.home))
                # 回填到 agent（让 trace 钩子注册函数能拿到它）
                if self.agent is not None:
                    self.agent._trace_sink = self.trace_sink
                    from agent.trace import _register_trace_hooks
                    if self.agent.hooks_registry is not None:
                        _register_trace_hooks(self.agent.hooks_registry, self.trace_sink)
            except Exception as e:
                logger.warning("TraceSink 初始化失败（fail-open）: %s", e)
                self.trace_sink = None

        # 6. 后台触发 curator（在守护线程里跑，不拖慢启动）
        self._maybe_trigger_curator()

        # === Memory Curator 后台触发 ===
        try:
            from constants import get_omnimate_home
            from agent.memory_curator import should_run_now_memory
            memory_dir = get_omnimate_home() / ".memory"
            if memory_dir.exists() and should_run_now_memory(memory_dir, config=self.config):
                import threading

                def _run_memory_curator():
                    # 调 _run_memory_curator_once，
                    # 内部 try/finally 保证 state 写盘（即使中途崩溃）。
                    # 传主 store 实例避免跨实例竞争
                    #（threading.Lock 只对同一个实例生效，两个实例等于没锁）。
                    # factory 通过 config 字典临时捎带（避免改函数签名）。
                    cfg_copy = dict(self.config) if isinstance(self.config, dict) else {}
                    try:
                        cfg_copy["_curator_factory"] = self._make_memory_review_agent_factory()
                    except Exception:
                        pass  # factory 创建失败也能跑第 1 阶段（只是跳过第 2 阶段）
                    try:
                        _run_memory_curator_once(
                            memory_dir, config=cfg_copy, store=self.memory_store,
                        )
                    except Exception as e:
                        logger.warning("Memory Curator 后台运行失败: %s", e)

                # 重要：daemon 线程不会自动继承主线程的
                # contextvars（上下文变量——Python 3.12 以下的 threading.Thread
                # 不拷贝 context）。项目分区键依赖 workspace_cwd 这个 ContextVar，
                # 不带过去就会退化用 os.getcwd()——多项目场景下，别的项目的
                # project/reference 记忆就永远不会被维护到。
                # 做法：在主线程里 copy_context()，线程入口用 ctx.run 包一层。
                _curator_ctx = contextvars.copy_context()
                threading.Thread(
                    target=lambda: _curator_ctx.run(_run_memory_curator),
                    daemon=True,
                    name="memory-curator",
                ).start()
        except Exception as e:
            logger.debug("Memory Curator 触发检查失败(不阻塞): %s", e)

        # === 清理僵尸子代理记录 + 过期记录清理 ===
        # 启动时把"标着 running 但进程早就没了"的残留（上次崩溃留下的）改成
        # interrupted，并清掉超过保留天数的老记录。
        _delegation_cfg = self.config.get("delegation", {})
        if _delegation_cfg.get("subagent_persistence_enabled", True):
            try:
                from agent.subagent_persistence import (
                    cleanup_stale_subagents, cleanup_old,
                )
                stale_n = cleanup_stale_subagents()
                retention_days = _delegation_cfg.get(
                    "subagent_persistence_retention_days", 7,
                )
                old_n = cleanup_old(days=retention_days)
                if stale_n or old_n:
                    logger.info(
                        "子代理持久化清理：stale=%d, expired=%d", stale_n, old_n,
                    )
            except Exception as e:
                logger.debug("子代理持久化清理失败（不阻塞）: %s", e)

        # === statusline 项目分区键（赋值一次，取不到就空着）===
        # 放在 initialize 末尾（所有依赖就绪后），失败不影响主流程
        try:
            from agent.project_scope import get_project_memory_key
            self._statusline_project_key = get_project_memory_key()
        except Exception as e:
            logger.debug("statusline 项目键获取失败（不阻塞）: %s", e)
            self._statusline_project_key = ""

        # === Hooks: SESSION_START（会话已建立、声明式 hooks 已加载完毕）===
        self._fire_session_start()

    def _fire_session_start(self) -> None:
        """触发"会话开始"事件钩子（会话建立后调用一次）。

        参数：无。返回：无（钩子抛异常只记日志，不影响主流程）。
        """
        if not getattr(self, "hooks_registry", None):
            return
        try:
            self.hooks_registry.run_session_start({
                "session_id": self.session_id or "",
            })
        except Exception as e:
            logger.warning("SESSION_START hook 触发异常: %s", e)

    def _fire_session_end(self) -> None:
        """触发"会话结束"事件钩子（会话关闭前、资源还没清理时调用）。

        参数：无。返回：无（钩子抛异常只记日志，不影响主流程）。
        """
        if not getattr(self, "hooks_registry", None):
            return
        try:
            self.hooks_registry.run_session_end({
                "session_id": self.session_id or "",
            })
        except Exception as e:
            logger.warning("SESSION_END hook 触发异常: %s", e)

    def _create_agent(self) -> AIAgent:
        """按配置创建 AIAgent 实例，并把所有外围组件挂上去。

        全文件最重的装配点——凭证推导、辅助模型路由、流式输出、邮箱、
        视觉模型、MCP 收件箱……全在这一步接好线。
        凭证优先用 settings.json 里的 api_key 字段；为空时退回环境变量找。

        参数：无。
        返回：装配完成的 AIAgent 实例。
        """
        model_cfg = self.config.get("model", {})
        # api_key 和 auth_token 是两种不同的"门禁卡"，分别取、不混用：
        # DeepSeek 的 Anthropic 端点认 auth_token（Bearer 方式），
        # 拿 api_key（x-api-key 方式）去敲会被拒之门外
        api_key = model_cfg.get("api_key") or ""
        auth_token = model_cfg.get("auth_token") or ""

        # 兜底：JSON 里没填 key 时，依次尝试各家专属的环境变量
        if not api_key and not auth_token:
            provider = (model_cfg.get("provider") or "").upper()
            api_key_env = model_cfg.get("api_key_env") or ""
            candidates = [
                api_key_env,
                f"{provider}_API_KEY" if provider else None,
            ]
            # 新模式 llm 扁平配置下 provider 是档位名（opus/haiku/sonnet），
            # <档位>_API_KEY 这种环境变量通常不存在；再按 base_url 域名 +
            # 常见厂商的环境变量兜底（如默认 DeepSeek 端点 → DEEPSEEK_API_KEY）
            base_url = str(model_cfg.get("base_url") or "").lower()
            for _host in ("deepseek", "openai", "anthropic", "openrouter"):
                if _host in base_url:
                    candidates.append(f"{_host.upper()}_API_KEY")
            candidates.extend(
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
            )
            for cand in candidates:
                if cand and os.environ.get(cand):
                    api_key = os.environ.get(cand)
                    break

        if not api_key and not auth_token:
            console.print("[red]未设置 API key！[/red]")
            console.print(
                f"请在 [bold]{self.home / 'settings.json'}[/bold] 的 "
                f"models.<name>.api_key 或 auth_token 填入，或设置环境变量。"
            )
            raise SystemExit(1)

        # === 创建辅助模型路由器（AuxLLMRouter）===
        # 辅助模型 = 干杂活的便宜小模型（起标题、解释命令等），跟主模型分开
        aux_llm_router = None
        aux_cfg = self.config.get("aux_model")
        # 优先读 aux_llm.endpoints 端点列表
        aux_llm_cfg = self.config.get("aux_llm", {})
        endpoints_cfg = aux_llm_cfg.get("endpoints", []) if aux_llm_cfg else []

        if endpoints_cfg or (aux_cfg and isinstance(aux_cfg, dict) and aux_cfg.get("model")):
            try:
                from agent.aux_llm import AuxLLMRouter, LLMEndpoint
                # router 的降级兜底也可能从线程里调
                # （分类器/curator）——用每次独立连接的 client，
                # 别建绑定主循环的临时连接池
                from agent.llm_client import ThreadedLLMClient
                main_client = ThreadedLLMClient({
                    "format": model_cfg.get("format", "openai"),
                    "base_url": model_cfg.get("base_url"),
                    "api_key": api_key,
                    "model": model_cfg["name"],
                })
                # 配置了 endpoints 列表就优先用它
                endpoints = None
                if endpoints_cfg:
                    endpoints = [
                        LLMEndpoint(
                            name=ep_cfg["name"],
                            base_url=ep_cfg["base_url"],
                            api_key_env=ep_cfg.get("api_key_env", ""),
                            api_key_default=ep_cfg.get("api_key_default", ""),
                            model=ep_cfg["model"],
                            priority=ep_cfg.get("priority", 100),
                            enabled=ep_cfg.get("enabled", True),
                            format=ep_cfg.get("format", "openai"),
                        )
                        for ep_cfg in endpoints_cfg
                    ]
                aux_llm_router = AuxLLMRouter(
                    main_client=main_client,
                    main_model=model_cfg["name"],
                    aux_config=aux_cfg if not endpoints else None,
                    endpoints=endpoints,
                )
            except Exception as e:
                logger.warning("AuxLLMRouter 创建失败，辅助任务用主模型: %s", e)
                aux_llm_router = None
                # ThreadedLLMClient 没有常驻连接池可泄漏，不用 close
                main_client = None

        # === 流式输出回调 ===
        # config["streaming"]["enabled"] 默认 True（打字机效果：边生成边打印）
        streaming_cfg = self.config.get("streaming", {})
        stream_callback = None
        if streaming_cfg.get("enabled", True):
            stream_callback = _make_cli_stream_callback()

        # 给 memory_manager 注入 LLM 连接（有辅助模型就用辅助的）
        if self.memory_manager:
            if aux_llm_router:
                self.memory_manager._llm_client = aux_llm_router
                self.memory_manager._llm_model = (
                    aux_cfg.get("model") if aux_cfg else None
                )
            else:
                # 没配置辅助模型时用主连接（等 agent 创建后再补注入）
                pass

        # === 把辅助模型路由器注入给 hook 执行器 ===
        # 声明式 hook 的 prompt/agent 两类 handler 要靠辅助模型做评估。
        # hook_exec 是模块级的注入点，没配置辅助模型时 lambda 返回 None
        #（宽松失败：hook 相关评估静默跳过，不报错）。
        from agent.hook_exec import set_aux_router_provider, set_config_provider
        set_aux_router_provider(lambda: aux_llm_router)
        # 注入 config 提供者，让 hook 分发器能读功能开关——
        # http / mcp_tool / agent 三种 handler 类型都受开关门控
        set_config_provider(lambda: self.config)

        # 注：不要往 RuntimeContext 回填 aux_llm_client / aux_model——
        # 是没人读的死字段；辅助模型的唯一来源是 agent_ref.aux_llm_router。

        # === 给权限检查器注入辅助模型 + config 提供者 ===
        # 权限闸门第 4 道（用辅助模型给命令分类）需要这两个提供者。
        # 权限检查器比辅助路由器先构造（前面行 294 那里），所以这里回头补上
        # （跟上面 hook_exec 的做法同款）。没配置辅助模型时 lambda 返回
        # None → 闸门 4 跳过（宽松失败，不拦命令）。
        try:
            from agent.permission import get_default_checker
            _perm_checker = get_default_checker()
            if _perm_checker is not None:
                _perm_checker.set_aux_llm_provider(lambda: aux_llm_router)
                _perm_checker.set_config_provider(lambda: self.config)
        except Exception as e:
            logger.debug("PermissionChecker provider 注入失败（闸门 4 将跳过）: %s", e)

        agent = AIAgent(
            base_url=model_cfg.get("base_url"),
            api_key=api_key,
            auth_token=auth_token,
            effort_level=model_cfg.get("effort_level"),
            model=model_cfg["name"],
            model_format=model_cfg.get("format", "anthropic"),
            max_iterations=self.config.get("agent", {}).get("max_iterations", 200),
            enabled_toolsets=self.config.get("enabled_toolsets", ["core"]),
            session_id=self.session_id,
            memory_store=self.memory_store,
            memory_manager=self.memory_manager,
            session_store=self.session_store,
            omnimate_home=self.home,
            on_tool_call=_make_tool_call_callback(self.config),
            config=self.config,
            hooks_registry=self.hooks_registry,  # hook 注册表
            bg_manager=self.bg_manager,  # 后台任务管理器
            cron_scheduler=self.cron_scheduler,  # 定时调度器
            team_bus=self.team_bus,  # 团队消息总线
            team_coordinator=self.team_coordinator,  # 团队协调器
            team_name="main",  # 本 agent 的团队名
            aux_llm_router=aux_llm_router,  # 辅助模型路由
            plan_approval_callback=cli_plan_approval_callback,  # 计划审批回调（Plan Mode）
            stream_callback=stream_callback,  # 流式输出
            ask_user_bridge=_make_ask_user_bridge(),  # ask_user 工具的 CLI 桥接（在黑窗口里向用户提问）
            checkpoint_manager=self.checkpoint_mgr,  # 快照管理器
            permission_mode=self.config.get("security", {}).get("permission_mode", "default"),  # 权限模式透传给 AIAgent
            # 流空闲看门狗（llm.stream_idle_timeout_seconds，默认 90 秒没新内容就掐掉重试，<=0 禁用）
            stream_idle_timeout=(self.config.get("llm") or {}).get("stream_idle_timeout_seconds"),
        )

        # 如果 memory_manager 还没分到 LLM 连接，就用 agent 的主连接
        if self.memory_manager and self.memory_manager._llm_client is None:
            self.memory_manager._llm_client = agent.llm_client
            self.memory_manager._llm_model = agent.model

        # === 接线 MCP 推送 → 收件箱 ===
        # 把"收件箱推送函数"注册成传输层的 notification handler——
        # 不接线的话 MCP server 推消息时 handler 是 None，消息直接被扔掉，
        # /inbox 永远是空的。
        # 宽松失败：任何异常只记 warning，不影响 agent/MCP 本身。
        try:
            from agent.channel_inbox import ChannelInbox
            from agent.mcp_client import get_mcp_manager
            channel_inbox = ChannelInbox(base_dir=self.home)
            mgr = get_mcp_manager()
            # 遍历所有已连上的 MCP client，给底层传输层注册 handler
            # 闭包变量捕获技巧：server_name 用默认参数绑定
            #（不这么做的话循环变量在回调触发时可能已经"漂移"成别的值）
            with mgr._lock:
                clients_snapshot = list(mgr._clients.items())
            for _server_name, _client in clients_snapshot:
                try:
                    _transport = getattr(_client, "_transport", None)
                    if _transport is None:
                        continue
                    # handler 收到 (方法名, 参数)，把来源 server + 内容推进收件箱
                    # 故意不过滤消息类型：所有 notifications/* 都进收件箱
                    #（收件箱本身不做筛选，由 LLM 拿到摘要后自己判断重要性）
                    def _make_handler(sname, inbox):
                        def _handler(method, params):
                            try:
                                inbox.push(sname, {"method": method, **(params or {})})
                            except Exception as e:
                                logger.warning(
                                    "channel_inbox.push fail-open (server=%s): %s",
                                    sname, e,
                                )
                        return _handler
                    _transport.set_notification_handler(
                        _make_handler(_server_name, channel_inbox)
                    )
                except Exception as e:
                    logger.warning(
                        "MCP server %s notification handler 注册失败（fail-open）: %s",
                        _server_name, e,
                    )
            # 把收件箱挂到 agent（主循环组装注入消息时会读这个）
            agent.set_channel_inbox(channel_inbox)
            if clients_snapshot:
                logger.info(
                    "ChannelInbox 已接线 %d 个 MCP server",
                    len(clients_snapshot),
                )
        except Exception as e:
            logger.warning("ChannelInbox 接线失败（fail-open）: %s", e)

        return agent

    @staticmethod
    def _derive_api_key(model_cfg: dict) -> str:
        """推导 API 密钥：配置文件里的值优先，为空时按厂商/网址猜环境变量兜底。

        跟 _create_agent 的主推导链是同一套逻辑（供 _thread_llm_client 复用，
        保证线程里建的连接拿到的凭证跟主连接完全一致）。

        参数：
            model_cfg: model 段的配置字典
        返回：推导出的密钥字符串（可能为空串）。
        """
        api_key = model_cfg.get("api_key") or ""
        auth_token = model_cfg.get("auth_token") or ""
        if not api_key and not auth_token:
            provider = (model_cfg.get("provider") or "").upper()
            candidates = [
                model_cfg.get("api_key_env") or "",
                f"{provider}_API_KEY" if provider else None,
            ]
            base_url = str(model_cfg.get("base_url") or "").lower()
            for _host in ("deepseek", "openai", "anthropic", "openrouter"):
                if _host in base_url:
                    candidates.append(f"{_host.upper()}_API_KEY")
            candidates.extend(
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
            )
            for cand in candidates:
                if cand and os.environ.get(cand):
                    api_key = os.environ.get(cand)
                    break
        # 与主连接构造保持一致：只配了 auth_token 时就用 auth_token 当凭证
        return api_key or auth_token

    def _maybe_trigger_curator(self):
        """在后台线程检查技能维护工人（curator）到没到干活周期，到了就开跑。

        参数：无。返回：无（不阻塞——守护线程里跑，主线程继续启动）。
        """
        if not self.config.get("curator", {}).get("enabled", True):
            return

        def _check():
            try:
                if should_run_now(skills_dir()):
                    console.print("[dim]后台 curator 触发：整理技能库...[/dim]")
                    # 要用的组件现场取——session/memory store 用
                    # initialize 前段建好的那两个实例（同实例才防得了跨实例竞争）；
                    # LLM 优先用辅助模型（便宜），没有就退回主连接
                    #（跟 reflection 复盘的取法同一模式）。
                    # 缺哪个组件，run_curator_review 内部自己跳过并记日志。
                    _agent = getattr(self, "agent", None)
                    _aux = getattr(_agent, "aux_llm_router", None)
                    if _aux is not None and _aux.is_aux_configured:
                        _llm = _aux
                    else:
                        # 线程里绝不借用绑定主循环的主连接
                        #——用 ThreadedLLMClient 每次独立建连
                        _llm = _thread_llm_client(self.config)
                    run_curator_review(
                        skills_dir(),
                        session_store=self.session_store,
                        memory_store=self.memory_store,
                        llm=_llm,
                    )
            except Exception as e:
                logger.debug("curator 触发失败: %s", e)

        # 守护线程：程序退出时它自动跟着结束，不会拖住主循环
        t = threading.Thread(target=_check, daemon=True)
        t.start()

    def new_session(self):
        """开始新会话：新建会话档案 + 把 agent 的会话级状态全部清回初始。

        会话级状态要清理到位，漏一项就出错——
        - checkpoint 管理器必须重建绑定新会话，否则还指向旧会话，之后的快照
          全写进旧目录、/rewind 会回滚错对象（做法跟 resume_session 一致）
        - 压缩状态 / 自动提取游标 / 记忆注入去重 / 上下文提示 /
          中断残留 / 临时消息队列等也都要归零
        - 故意不清：审批缓存——用户批过的命令跨 /new 还反复问，坏处大于
          好处（用户的明确判断优先于算法，见 CLAUDE.md 设计原则 4）。

        参数：无。返回：无。
        """
        if self.session_store:
            self.session_id = self.session_store.create_session(
                model=self.config["model"]["name"],
                provider=self.config["model"]["provider"],
            )
            self.agent.session_id = self.session_id
        # checkpoint 管理器重建（跟 resume_session 的做法对齐）
        try:
            from agent.checkpoint import CheckpointManager
            ckpt_root = Path(self.home) / ".checkpoints"
            self.checkpoint_mgr = CheckpointManager(
                ckpt_root, self.session_id,
                max_snapshots=self.config.get("checkpoint", {}).get(
                    "max_snapshots", 100,
                ),
            )
        except Exception as e:
            logger.warning("checkpoint 重建失败: %s", e)
        # 会话级 agent 状态重置
        a = self.agent
        a.conversation_history = []
        try:
            from agent.context_pipeline import CompressionSessionState
            a._compress_session_state = CompressionSessionState()
        except Exception as e:
            logger.debug("压缩状态重置失败（忽略）: %s", e)
        a._interrupt_requested = False
        a._auto_extract_cursor = 0
        a._auto_extract_turn_count = 0
        a._context_tip_shown = False
        a._pending_ephemeral_messages = []
        a._pending_tool_batch_summary = None
        a._queued_cli_commands = []
        a._surfaced_memory_ids = set()
        try:
            from agent.memory_injection import reset_injection_cache
            reset_injection_cache()
        except Exception as e:
            logger.debug("记忆注入缓存重置失败（忽略）: %s", e)
        a.invalidate_system_prompt()

    def resume_session(self, session_id: str) -> bool:
        """恢复历史会话：把存档里的消息装回 agent 的对话历史。

        agent 的 system prompt（系统提示词）会重建——记忆快照从当前的
        .md 文件重新读，保证恢复后看到的是最新记忆。

        参数：
            session_id: 要恢复的会话 ID
        返回：True=恢复成功；False=没有会话库或会话不存在。
        """
        if not self.session_store:
            return False

        info = self.session_store.get_session(session_id)
        if not info:
            console.print(f"[red]会话不存在: {session_id}[/red]")
            return False

        msgs = self.session_store.get_messages(session_id)
        # 对话历史不含 system 消息（system 由 prompt_builder 现场生成）
        conv = [m for m in msgs if m.get("role") != "system"]
        # 按最后一次压缩的边界裁掉更早的旧消息
        #（会话库只追加不删改，不裁的话会载入全部旧历史；
        # 旧会话没有边界标记就保守全量载入）
        conv = _truncate_at_last_compact_boundary(conv)
        # 清理多余的摘要占位（只留最近一个）——长会话可能存了
        # 一堆"[之前的对话已自动总结]"占位，全塞进上下文会撑爆且混乱
        conv = _cleanup_redundant_summaries(conv)
        # 孤儿工具结果修复
        # 保存时中断可能留下"缺了一半"的悬空工具结果（assistant 发了
        # tool_calls 但对应的 result 缺失，或反过来的孤儿 result）。
        # 主循环里的 _fix_tool_call_pairs 只在发送前临时修（不写回盘）；
        # 这里在加载时就修好并回写，让存档和后续轮次都干净。
        try:
            from agent.context_compressor import _fix_tool_call_pairs
            fixed = _fix_tool_call_pairs(conv)
            if fixed != conv:
                logger.info("resume: 修复了孤儿 tool_call/result 配对")
                conv = fixed
        except Exception as e:
            logger.warning("resume 孤儿修复失败（忽略，发送前还会兜底）: %s", e)
        self.agent.conversation_history = conv
        self.session_id = session_id
        self.agent.session_id = session_id
        self.agent.invalidate_system_prompt()

        # 重建 checkpoint 管理器（绑定恢复的这个会话 ID）
        try:
            from agent.checkpoint import CheckpointManager
            ckpt_root = Path(self.home) / ".checkpoints"
            self.checkpoint_mgr = CheckpointManager(
                ckpt_root, session_id,
                max_snapshots=self.config.get("checkpoint", {}).get(
                    "max_snapshots", 100),
            )
            self.agent.checkpoint_manager = self.checkpoint_mgr
        except Exception as e:
            logger.warning("CheckpointManager 重建失败: %s", e)
            self.checkpoint_mgr = None

        title = info.get("title") or "(无标题)"
        console.print(
            f"[green][已恢复会话: {title}（{len(msgs)} 条消息）][/green]"
        )

        # 显示最近几条消息让用户想起上下文（不显示 tool 消息——太碎了）
        recent = [m for m in msgs[-6:]
                  if (m.get("content") or "").strip() and m.get("role") != "tool"]
        if recent:
            _print_message_list(recent, char_limit=300, header=f"最近 {len(recent)} 条历史消息：")

        return True

    def shutdown(self):
        """关机清理：触发会话结束钩子 + 停后台任务、定时器、团队，关连接。

        后台任务和定时器不关会变"僵尸进程"；Windows 上会话库连接不关会
        锁住文件。

        参数：无。返回：无（每一步都容错，单步失败不影响其余清理）。
        """
        # === Hooks: SESSION_END（在资源清理前触发，此时会话上下文还在）===
        self._fire_session_end()

        # 把技能使用统计从内存缓存刷到磁盘
        try:
            from tools.skill_usage import flush_usage
            flush_usage()
        except Exception:
            logger.debug("flush 技能使用统计失败", exc_info=True)

        if hasattr(self, "bg_manager") and self.bg_manager:
            try:
                self.bg_manager.shutdown()
            except Exception as e:
                logger.warning("bg_manager shutdown 失败: %s", e)

        # 关闭会话库的常驻连接（Windows 上不关会锁住文件）
        if hasattr(self, "session_store") and self.session_store:
            try:
                self.session_store.close()
            except Exception as e:
                logger.warning("session_store close 失败: %s", e)

        if hasattr(self, "cron_scheduler") and self.cron_scheduler:
            try:
                self.cron_scheduler.shutdown()
            except Exception as e:
                logger.warning("cron_scheduler shutdown 失败: %s", e)

        # === 主 agent 标记 stopped + 清理它拉起的子进程 ===
        if hasattr(self, "team_coordinator") and self.team_coordinator:
            try:
                self.team_coordinator.shutdown_all()
            except Exception as e:
                logger.warning("team_coordinator shutdown_all 失败: %s", e)
            try:
                self.team_coordinator.update_status("main", "completed")
            except Exception as e:
                logger.warning("team_coordinator shutdown 失败: %s", e)

        # === agent 自己的资源清理（MCP 连接等）===
        if hasattr(self, "agent") and self.agent:
            try:
                self.agent.cleanup()
            except Exception as e:
                logger.warning("agent.cleanup 失败: %s", e)


# ---------------------------------------------------------------------------
# 回调（把"CLI 怎么跟用户互动"做成函数，传给 agent 内部调用）
# ---------------------------------------------------------------------------

def _make_approval_callback(aux_provider=None):
    """造一个"问用户批不批准"的回调（危险命令执行前、白名单外写文件前都会用到）。

    供权限检查器使用（它不懂怎么跟用户对话）。回调收到一个字符串，
    自动判断是命令还是路径，显示不同的提问。

    批准的效果范围（如实说明，别夸大）：
    - 命令 → 同意后记入 ~/.OmniMate/approved_commands.json，
      跨会话不再重复询问同一条命令
    - 路径 → 三档：y=本次允许（父目录进会话缓存，同目录后续写入不再问）；
      a=总是允许（父目录持久化到 settings.json 的 security.extra_allowed_roots，
      跨会话生效，与 /add-dir 同一通道）；N=拒绝

    命令审批多一个 e 选项——让辅助模型解释这条命令干什么用 +
    LOW/MEDIUM/HIGH 风险等级（aux_provider 注入；没配辅助模型就隐藏该选项）。
    宽松失败：解释挂了不阻塞审批。

    参数：
        aux_provider: 一个零参函数，调用返回辅助模型路由器（或 None）
    返回：审批回调函数 callback(item)。
    """
    def _explain(command: str):
        """用辅助模型解释命令（干什么用 + 风险等级）。宽松失败。

        参数：
            command: 要解释的命令字符串
        """
        aux = aux_provider() if aux_provider else None
        if aux is None:
            console.print("[dim]（解释器不可用——未配置 aux_llm）[/dim]")
            return
        try:
            import asyncio as _aio
            resp = _aio.run(aux.chat_completions([{"role": "user", "content": (
                "用中文解释下面这条 shell 命令做什么。输出两行：\n"
                "第 1 行：用途（一句话）\n"
                "第 2 行：风险等级：LOW / MEDIUM / HIGH + 一句话理由\n\n"
                f"命令：{command}"
            )}]))
            text = (resp.choices[0].message.content or "").strip()
            if text:
                console.print(Panel.fit(text, title="🔍 命令解释", border_style="cyan"))
        except Exception as e:
            console.print(f"[dim]（解释失败: {e}）[/dim]")

    def callback(item: str):
        # 审批入口：命令/路径的判定用 _is_path_item
        #（check_path 发来的内容恒带"文件写入审批: "前缀，按这个约定识别）
        if _is_path_item(item):
            console.print(f"[yellow]⚠️ 即将写入路径(白名单外)：[/yellow]")
            console.print(f"[bold]{item}[/bold]")
            try:
                answer = console.input(
                    "[bold]允许？(y=本次 / a=总是允许并记住 / N=拒绝):[/bold] ",
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return False
            if answer in ("a", "always"):
                # 返回约定的哨兵值，持久化由 PermissionChecker.check_path 统一做
                #（不在回调里直接写文件，职责分开）
                return "always"
            return answer in ("y", "yes")
        else:
            console.print(f"[yellow]⚠️ 即将执行破坏性命令：[/yellow]")
            console.print(f"[bold]{item}[/bold]")
            explain_hint = "[dim] e=解释[/dim]" if aux_provider else ""
            while True:
                try:
                    answer = console.input(
                        f"[bold]允许执行？(y/N):[/bold]{explain_hint} "
                        "[dim]（同意后此命令不再询问）[/dim] ",
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    return False
                if answer == "e" and aux_provider:
                    _explain(item)
                    continue  # 解释完再问一遍
                return answer in ("y", "yes")
    return callback


def _make_ask_user_bridge():
    """造 ask_user 工具的 CLI 桥接：AI 想问用户选择题时，由它画面板、收答案。

    agent 内部的 ask_user 工具不懂终端交互，这个桥接负责把问题渲染成
    漂亮面板、读用户输入的序号。
    bridge(qdata) 返回选中选项的文字列表。
    异常（EOFError/KeyboardInterrupt——输入流关闭/用户按 Ctrl+C）由
    ask_user 的 handler 统一捕获。

    参数：无。
    返回：桥接函数 bridge(qdata)。
    """
    def bridge(qdata):
        question = qdata.get("question", "")
        options = qdata.get("options") or []
        multi = qdata.get("multi", False)

        n = len(options)
        lines = [f"[bold]{question}[/bold]", ""]
        for i, opt in enumerate(options):
            label = opt.get("label", "")
            desc = opt.get("description", "")
            lines.append(
                f"[cyan]{i + 1}[/cyan]. {label}" + (f" — {desc}" if desc else "")
            )
        # 永远给一个自己输入的口子——预设选项不可能穷尽用户的想法
        lines.append(f"[cyan]{n + 1}[/cyan]. ✍ 其他（自己输入）")
        console.print(Panel.fit(
            "\n".join(lines),
            title="❓ 需要你选择",
            border_style="cyan",
        ))

        if multi:
            # 多选：逗号分隔，数字段选选项、文字段算自定义答案（混着用也行）
            raw = console.input(
                "[bold]选择序号（逗号分隔，也可直接输入文字）> [/bold] ",
            ).strip()
            answers = []
            for part in raw.replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if part.isdigit() and 1 <= int(part) <= n:
                    answers.append(options[int(part) - 1]["label"])
                elif not part.isdigit():
                    answers.append(part)
            return answers

        # 单选：数字选选项；选「其他」的序号或直接输入文字都算自定义答案
        raw = console.input("[bold]选择序号（或直接输入你的答案）> [/bold] ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= n:
                return [options[idx - 1]["label"]]
            if idx == n + 1:
                custom = console.input("[bold]请输入你的答案 > [/bold] ").strip()
                return [custom] if custom else []
            return []
        return [raw] if raw else []
    return bridge


def _ts() -> str:
    """生成"[时:分:秒]"格式的时间戳前缀，用在工具调用进度输出里。

    参数：无。返回：如 "[14:30:05]" 的字符串。
    """
    from datetime import datetime
    return f"[{datetime.now().strftime('%H:%M:%S')}]"


def _make_tool_call_callback(config: dict):
    """造工具调用进度回调——把"要不要显示进度"的配置读一次存进闭包
    （免得每次工具调用都现读 settings.json）。

    参数：
        config: 配置字典（读 display.show_tool_progress 开关）
    返回：回调函数 callback(name, args)。
    """
    show = (config or {}).get("display", {}).get("show_tool_progress", True)

    def callback(name: str, args: dict):
        """每次工具被调用时在屏幕打一行灰字进度（开关关了就什么都不打）。

        参数：
            name: 工具名
            args: 工具参数（超 80 字符的值截断显示）
        """
        if not show:
            return
        short_args = {}
        for k, v in (args or {}).items():
            s = str(v)
            short_args[k] = s if len(s) <= 80 else s[:77] + "..."
        console.print(f"[dim]{_ts()} → 调用工具: {name} {short_args}[/dim]")

    return callback


# 向后兼容的接口：模块级函数仍然能用，但每次调用都重读配置文件（不推荐新代码用）
def _on_tool_call(name: str, args: dict):
    """工具调用时的进度打印（兼容接口：每次现读 config）。

    保留它只为兼容既有引用；新代码请用 _make_tool_call_callback。

    参数：
        name: 工具名
        args: 工具参数（超 80 字符的值截断显示）
    """
    if not load_config().get("display", {}).get("show_tool_progress", True):
        return
    short_args = {}
    for k, v in (args or {}).items():
        s = str(v)
        short_args[k] = s if len(s) <= 80 else s[:77] + "..."
    console.print(f"[dim]{_ts()} → 调用工具: {name} {short_args}[/dim]")


def _make_cli_stream_callback():
    """造 CLI 的流式输出回调（04）：模型每吐一段字就立刻打到屏幕上。

    打字机效果——第一个字出现 < 500 毫秒，不用等全部生成完。
    工具调用开始时打一行简短提示。
    流结束时不打印（换行收尾留给主流程）。

    参数：无。返回：回调函数 cb(event)。
    """
    import sys

    def cb(event: dict) -> None:
        """流事件处理器：content 打字 / tool_call_start 换行提示 / progress 心跳。

        参数：
            event: {"type": "content"|"tool_call_start"|"progress"|..., ...}
        """
        etype = event.get("type")
        if etype == "content":
            delta = event.get("delta") or ""
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        elif etype == "tool_call_start":
            # 正在流式打字时若切去调工具，先换行收尾再打提示
            print()  # noqa: T201
            name = event.get("name", "?")
            console.print(f"[dim]{_ts()} ⟳ 准备调用 {name}...[/dim]")
        elif etype == "progress":
            # 子代理运行中的周期性进度——告诉用户"还在干活，不是卡死了"
            msg = (event.get("message") or "").strip()
            elapsed = int(event.get("elapsed_seconds") or 0)
            if msg:
                print()  # noqa: T201
                console.print(
                    f"[dim]{_ts()} ⟳ 子代理[{elapsed}s] {msg}[/dim]"
                )
        # "done" 事件不打任何东西：收尾换行留给主流程
    return cb


# ---------------------------------------------------------------------------
# Slash 命令处理（斜杠开头指令的分发和各命令的实现）
# ---------------------------------------------------------------------------

def _handle_handoff_command(args: str, rt) -> bool:
    """处理 /handoff <子命令> [参数]——会话移交（把当前对话打包带走/装回）。

    子命令：save 保存 / list 列表 / load 装载 / show 详情 /
    delete 删除 / export 导出文件 / import 从文件导入 / help 帮助

    参数：
        args: 命令后面的全部文字（子命令 + 参数）
        rt: RuntimeContext（拿 handoff 存储和当前对话）
    返回：True（表示这条输入已处理完，不再发给模型）。
    """
    parts = args.split(None, 1)
    sub = parts[0].lower() if parts else "help"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("help", "h", "?", ""):
        console.print(Panel(
            "[bold]/handoff 子命令[/bold]\n\n"
            "[cyan]/handoff save [title][/cyan]    保存当前会话为 bundle\n"
            "[cyan]/handoff list[/cyan]           列出所有 bundle\n"
            "[cyan]/handoff load <id|idx>[/cyan]  加载 bundle（覆盖当前会话）\n"
            "[cyan]/handoff show <id|idx>[/cyan]  查看 bundle 详情\n"
            "[cyan]/handoff delete <id|idx>[/cyan] 软删除 bundle\n"
            "[cyan]/handoff export <id> <path>[/cyan]  导出 bundle 到路径\n"
            "[cyan]/handoff import <path>[/cyan]  从路径导入 bundle\n",
            border_style="blue",
        ))
        return True

    if rt.handoff_store is None:
        console.print("[red]handoff 存储未初始化[/red]")
        return True

    store = rt.handoff_store

    if sub == "save":
        title = rest.strip() or None
        try:
            bundle_id = store.save(
                transcript=rt.agent.conversation_history,
                source_session_id=rt.session_id,
                model=rt.config.get("model", {}),
                title=title,
            )
            meta = next((m for m in store.list_bundles()
                         if m.bundle_id == bundle_id), None)
            count = meta.message_count if meta else "?"
            size = meta.file_size if meta else 0
            console.print(
                f"[green]✓ bundle {bundle_id} 已保存[/green] "
                f"[dim]({count} 条消息，{size} bytes)[/dim]"
            )
        except SecretDetectedError as e:
            console.print(f"[red]检测到 {len(e.matches)} 处疑似密钥，拒绝保存[/red]")
            for m in e.matches[:3]:
                console.print(
                    f"[dim]  - msg#{m['message_index']} ({m['role']}): "
                    f"{m['pattern']}[/dim]"
                )
        except BundleTooLargeError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]建议先 /compress 压缩上下文[/dim]")
        return True

    if sub in ("list", "ls"):
        metas = store.list_bundles()
        if not metas:
            console.print("[dim]暂无 handoff bundle。用 /handoff save 创建。[/dim]")
            return True

        table = Table(title=f"Handoff Bundles（{len(metas)}）")
        table.add_column("#", style="dim", justify="right")
        table.add_column("Bundle ID", style="cyan")
        table.add_column("标题")
        table.add_column("消息数", justify="right")
        table.add_column("状态")
        table.add_column("创建时间")

        for idx, m in enumerate(metas):
            table.add_row(
                str(idx),
                m.bundle_id,
                m.title or "(无标题)",
                str(m.message_count),
                m.handoff_state,
                m.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)
        return True

    if sub == "show":
        if not rest:
            console.print("[yellow]用法：/handoff show <id|index>[/yellow]")
            return True
        try:
            bundle = store.load(rest.strip())
        except BundleNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return True
        except AmbiguousBundleIDError as e:
            console.print(f"[yellow]前缀匹配多个 bundle：[/yellow]")
            for c in e.candidates:
                console.print(f"[dim]  - {c}[/dim]")
            return True

        console.print(Panel(
            f"[bold]{bundle.title or '(无标题)'}[/bold]\n\n"
            f"[cyan]bundle_id:[/cyan]      {bundle.bundle_id}\n"
            f"[cyan]created_at:[/cyan]     {bundle.created_at.isoformat()}\n"
            f"[cyan]source_platform:[/cyan] {bundle.source_platform}\n"
            f"[cyan]model:[/cyan]          {bundle.model}\n"
            f"[cyan]state:[/cyan]          {bundle.handoff_state}\n"
            f"[cyan]messages:[/cyan]       {len(bundle.transcript)}\n"
            f"[cyan]memory_ptrs:[/cyan]    {len(bundle.memory_pointers)}\n"
            f"[cyan]checksum:[/cyan]       {bundle.schema_checksum[:20]}...",
            title="Bundle 详情",
            border_style="blue",
        ))
        # 顺便预览最近 6 条消息
        recent = bundle.transcript[-6:]
        if recent:
            _print_message_list(recent, char_limit=200, header="最近消息：")
        return True

    if sub == "load":
        if not rest:
            console.print("[yellow]用法：/handoff load <id|index>[/yellow]")
            return True
        try:
            bundle = store.load(rest.strip())
        except (BundleNotFoundError, AmbiguousBundleIDError) as e:
            console.print(f"[red]{e}[/red]")
            return True

        # 当前会话非空时先问一句（装载会覆盖现有对话）
        current_count = len(getattr(rt.agent, "conversation_history", []))
        if current_count > 0:
            console.print(
                f"[yellow]将覆盖当前会话（{current_count} 条消息）。"
                f"是否先 /handoff save? 直接 load 输入 y：[/yellow]"
            )
            try:
                answer = input("(y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                console.print("[dim]已取消[/dim]")
                return True

        # 用 bundle 内容替换当前对话历史 + 新建一个 session 记录这次装载
        rt.agent.conversation_history = list(bundle.transcript)
        if hasattr(rt.agent, "invalidate_system_prompt"):
            rt.agent.invalidate_system_prompt()
        if getattr(rt, "session_store", None):
            new_sid = rt.session_store.create_session(
                model=bundle.model.get("name", ""),
                provider=bundle.model.get("provider", ""),
            )
            rt.agent.session_id = new_sid
            rt.session_id = new_sid

        store.mark_completed(bundle.bundle_id)
        console.print(
            f"[green]✓ 已加载 bundle（{len(bundle.transcript)} 条消息，"
            f"标题：{bundle.title or '(无标题)'}）[/green]"
        )
        if bundle.memory_pointers:
            # 逐条检查 bundle 引用的记忆在本机存不存在（跨机器搬运可能缺货）
            mem_store = getattr(rt, "memory_store", None)
            if mem_store is not None:
                existing = sum(
                    1 for mid in bundle.memory_pointers
                    if mem_store.load_body(mid) is not None
                )
                total = len(bundle.memory_pointers)
                color = "green" if existing == total else "yellow"
                console.print(
                    f"[{color}]memory 引用：{existing}/{total} 本机存在[/{color}]"
                )
                if existing < total:
                    missing = [
                        mid for mid in bundle.memory_pointers
                        if mem_store.load_body(mid) is None
                    ]
                    console.print(
                        f"[dim]缺失：{', '.join(missing[:5])}"
                        f"{' ...' if len(missing) > 5 else ''}[/dim]"
                    )
            else:
                console.print(
                    f"[dim]引用了 {len(bundle.memory_pointers)} 条 memory"
                    f"（本机 memory_store 未初始化，无法验证）[/dim]"
                )
        return True

    if sub in ("delete", "rm"):
        if not rest:
            console.print("[yellow]用法：/handoff delete <id|index>[/yellow]")
            return True
        try:
            archived = store.delete(rest.strip())
            console.print(f"[green]✓ 已软删除到 {archived}[/green]")
        except BundleNotFoundError as e:
            console.print(f"[red]{e}[/red]")
        except AmbiguousBundleIDError as e:
            console.print(f"[yellow]前缀匹配多个 bundle：[/yellow]")
            for c in e.candidates:
                console.print(f"[dim]  - {c}[/dim]")
        return True

    if sub == "export":
        parts2 = rest.split(None, 1)
        if len(parts2) < 2:
            console.print("[yellow]用法：/handoff export <id|index> <path>[/yellow]")
            return True
        target_id, dest_str = parts2[0], parts2[1]
        try:
            dest = store.export_to(target_id, Path(dest_str).expanduser())
            console.print(f"[green]✓ 已导出到 {dest}[/green]")
        except (BundleNotFoundError, AmbiguousBundleIDError) as e:
            console.print(f"[red]{e}[/red]")
        return True

    if sub == "import":
        if not rest:
            console.print("[yellow]用法：/handoff import <path>[/yellow]")
            return True
        try:
            new_id = store.import_from(Path(rest.strip()).expanduser())
            console.print(f"[green]✓ 已导入 bundle {new_id}[/green]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
        except BundleCorruptedError as e:
            console.print(f"[red]bundle 损坏：{e}[/red]")
        return True

    console.print(f"[yellow]未知子命令：{sub}（用 /handoff help 查看）[/yellow]")
    return True


def cli_plan_approval_callback(plan: str) -> tuple:
    """计划模式（Plan Mode）的审批回调：打印 AI 写的计划，问用户批不批。

    计划模式下 AI 只做调研、不动手，调研完把计划交给用户审。
    返回 (是否批准, 修订意见, 是否清空上下文) 三元组：
    - y/yes → (True, "", False) 批准，保留上下文继续执行
    - c/clear → (True, "", True) 批准并清空上下文再执行（设计考量：调研
      过程的对话全部丢弃、只留计划指令，执行阶段不用再烧调研的 token；
      完整历史仍存在轨迹/会话库里随时可查）
    - edit → 收集一行修订意见 → (False, 意见, False)
    - 其他（n/空/任意键）→ (False, "用户拒绝", False)

    参数：
        plan: AI 提交的计划全文
    """
    print("\n" + "=" * 60)
    print("Agent 提交了以下计划，请审批：")
    print("=" * 60)
    print(plan)
    print("=" * 60)
    print("\n批准？[y=批准 / c=批准并清空上下文执行 / N=拒绝 / edit=修订]")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False, "用户中断输入", False
    if choice in ("y", "yes"):
        return True, "", False
    if choice in ("c", "clear"):
        return True, "", True
    if choice == "edit":
        print("请输入修订建议（单行）：")
        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return False, "用户中断输入", False
        return False, feedback or "用户未输入修订建议", False
    return False, "用户拒绝（未提供原因）", False


def _handle_command(cmd: str, rt: RuntimeContext) -> bool:
    """slash 命令总分发：认出哪个命令就转给对应的处理函数。

    参数：
        cmd: 用户敲的整条命令（如 "/model opus"）
        rt: RuntimeContext
    返回：True=这是已知命令、已处理；False=不认识（由调用方决定发模型还是报错）。
    """
    parts = cmd.split(None, 1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit"):
        # 故意不 raise SystemExit——那会跳过关机清理和 SESSION_END 钩子，
        # 直接把程序掐死。改成设标志，主循环看到后 break 走正常退出。
        rt.quit_requested = True
        return True

    if name == "/help":
        _show_help()
        return True

    if name == "/new":
        rt.new_session()
        console.print("[green][已开始新对话][/green]")
        return True

    if name == "/skills":
        _handle_skills_command(rt, args)
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

    if name == "/stats":
        _show_stats(rt)
        return True

    if name == "/model":
        _switch_model(rt, args)
        return True

    if name == "/plan":
        if args.strip() == "off":
            if rt.agent.plan_mode:
                rt.agent.plan_mode = False
                console.print("[green]已强制退出计划模式。[/green]")
            else:
                console.print("[yellow]当前不在计划模式。[/yellow]")
        else:
            if not rt.agent.plan_mode:
                rt.agent.plan_mode = True
                console.print(
                    "[green]已进入计划模式。[/green] "
                    "Agent 只能调研，完成调研后调 exit_plan_mode 等待审批。"
                )
            else:
                console.print("[yellow]已经在计划模式中。[/yellow]")
        return True

    if name == "/permission":
        # /permission [default|bypass|acceptEdits]：切换权限模式
        # 不带参数 → 显示当前模式；带参数 → 切换（要同步改两处：检查器的
        # mode + rt.agent.permission_mode，改一处会不同步）
        arg = args.strip().lower() if args else ""
        from agent.permission import get_default_checker
        checker = get_default_checker()
        if not arg:
            current = getattr(checker, "mode", "default")
            console.print(f"当前权限模式: [cyan]{current}[/cyan]")
            console.print(
                "[dim]用法: /permission default 切回默认（带审批闸门） | "
                "/permission bypass 切到 bypassPermissions（跳过审批，仍挡 fatal 根删除） | "
                "/permission accept 切到 acceptEdits（自动批 cwd 内编辑/fs 命令）[/dim]"
            )
            return True
        if arg in ("default", "bypass", "bypasspermissions", "acceptedits", "accept"):
            if arg == "default":
                new_mode = "default"
            elif arg in ("bypass", "bypasspermissions"):
                new_mode = "bypassPermissions"
            else:  # acceptedits / accept
                new_mode = "acceptEdits"
            if checker is not None:
                checker.mode = new_mode
            if getattr(rt, "agent", None) is not None:
                rt.agent.permission_mode = new_mode
            console.print(f"[green]权限模式切换为: {new_mode}[/green]")
        else:
            console.print("[yellow]用法: /permission [default|bypass|acceptEdits][/yellow]")
        return True

    if name == "/sandbox":
        # OS 沙箱开关
        # 用法：/sandbox on | off | status（无参数 = status）
        arg = args.strip().lower() if args else ""
        from agent.sandbox_runner import (
            is_available, availability_reason, sandbox_description,
        )
        from agent.permission import get_default_checker
        checker = get_default_checker()

        if arg in ("on", "enable"):
            if not is_available():
                console.print(
                    f"[yellow]⚠️  沙箱不可用：{availability_reason()}\n"
                    "仍会切换到 on 模式（fail-open 降级，命令照常执行）[/yellow]"
                )
            # 有些简化版检查器没有 set_sandbox_mode 方法，先探测防崩
            if hasattr(checker, "set_sandbox_mode"):
                checker.set_sandbox_mode("on")
                console.print(
                    f"[green]sandbox: on[/green]\n"
                    f"[dim]机制：{sandbox_description()}。"
                    "Linux/macOS 写文件被限制在 cwd + ~/.OmniMate + 配置的 "
                    "sandbox_writable_roots；Windows Job Object 为进程管控"
                    "（文件防线=safe_path 白名单层）。[/dim]"
                )
            else:
                console.print(
                    "[red]无法切换 sandbox：当前 PermissionChecker 不支持 set_sandbox_mode[/red]"
                )
        elif arg in ("off", "disable"):
            if hasattr(checker, "set_sandbox_mode"):
                checker.set_sandbox_mode("off")
                console.print("[green]sandbox: off[/green]")
            else:
                console.print(
                    "[red]无法切换 sandbox：当前 PermissionChecker 不支持 set_sandbox_mode[/red]"
                )
        else:  # status 或无参数
            mode = getattr(checker, "sandbox_mode", "off")
            if is_available():
                avail = f"[green]available[/green] — {sandbox_description()}"
            else:
                avail = f"[red]unavailable[/red] ({availability_reason()})"
            console.print(f"sandbox: [cyan]{mode}[/cyan]  ({avail})")
            console.print(
                "[dim]用法: /sandbox on 开启 | /sandbox off 关闭 | /sandbox status 查看状态[/dim]"
            )
        return True

    if name == "/hooks":
        # 展示会话启动那一刻锁定的 hook 快照 + 对比磁盘上的改动
        from agent.hook_loader import get_snapshot, get_disk_version
        snap = get_snapshot()
        if not snap:
            console.print(
                "[yellow]无声明式 hook（~/.OmniMate/.hooks/settings.json 未配置或为空）[/yellow]"
            )
            return True
        console.print(
            "[bold]当前会话生效的 hook（启动时锁定，运行期改配置不立即生效 — 防篡改）：[/bold]"
        )
        for event, hook_list in snap.items():
            console.print(f"  [cyan]{event}[/cyan] ({len(hook_list)} 个)")
            for h in hook_list:
                htype = h.get("type", "command")
                hname = h.get("name", "?")
                console.print(f"    - {hname} (type={htype})")
        # 对比磁盘版本，检测会话期间配置文件是否被改过
        disk = get_disk_version()
        if disk != snap:
            console.print(
                "\n[yellow]⚠ 磁盘 settings.json 与会话快照不一致[/yellow]\n"
                "[dim]提示：hook 配置在会话启动时锁定，运行期修改不会立即生效。"
                "重启会话才会加载新配置（防篡改）。[/dim]"
            )
        else:
            console.print("[dim]（磁盘配置与会话快照一致）[/dim]")
        return True

    if name == "/agents":
        # E2 新增：列出自定义子代理定义（~/.OmniMate/agents + ./.omnimate/agents）
        from agent.agent_defs import scan_agent_defs
        defs = scan_agent_defs()
        if not defs:
            console.print(
                "[yellow]无自定义子代理。[/yellow] "
                "在 [cyan]~/.OmniMate/agents/[/cyan] 或 [cyan]./.omnimate/agents/[/cyan] "
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

    if name == "/handoff":
        return _handle_handoff_command(args, rt)

    if name == "/approved":
        _manage_whitelist(rt, args)
        return True

    if name == "/rewind":
        _handle_rewind_command(rt, args)
        return True

    if name == "/cache-stats":
        try:
            from agent.cache_monitor import get_stats
            stats = get_stats()
            console.print(f"[cyan]本次会话 cache 累计 break 次数：[/cyan]{stats['total_breaks']}")
            if stats['last_break']:
                lb = stats['last_break']
                console.print(
                    f"[cyan]最近 break：[/cyan]cache read {lb['from']} → {lb['to']}"
                    f"（降 {lb['drop']} tokens）"
                )
                console.print(f"[cyan]根因：[/cyan]{lb['root_cause']}")
                # 顺便告诉用户 diff 文件存哪了
                if lb.get('diff_path'):
                    console.print(
                        f"[cyan]diff 文件：[/cyan]{lb['diff_path']}"
                        f"（read_file 看详细变化）"
                    )
            if stats['last_cache_read'] is not None:
                console.print(
                    f"[cyan]最近一次 cache read：[/cyan]{stats['last_cache_read']} tokens"
                )
        except Exception as e:
            console.print(f"[red]读取 cache 统计失败：[/red]{e}")
        return True

    # === goal / poor / output-style / trace / history / mailbox / inbox / resume_bundle ===
    if name == "/goal":
        return _handle_goal_command(args, rt)
    if name == "/poor":
        return _handle_poor_command(args, rt)
    if name == "/output-style":
        return _handle_output_style_command(args, rt)
    if name == "/trace":
        return _handle_trace_command(args, rt)
    if name == "/history":
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
    if name == "/mailbox":
        return _handle_mailbox_command(args, rt)
    if name == "/inbox":
        return _handle_inbox_command(args, rt)
    if name == "/resume_bundle":
        # /resume 这个名字已被"恢复会话"占用，跨项目的 bundle 恢复
        # 另起 /resume_bundle 加以区分
        return _handle_resume_command(args, rt)

    # === /init 生成 OMNIMATE.md ===
    if name == "/init":
        return _handle_init_command(rt, args)

    # === /resumable 列出/恢复可续跑的子代理 ===
    if name == "/resumable":
        return _handle_resumable_command(args, rt)

    # === /compact 手动压缩上下文 + /context 看 token 分布 ===
    if name == "/compact":
        return _handle_compact_cli(args, rt)
    if name == "/context":
        return _handle_context_cli(args, rt)

    # === /status 状态一览 + /doctor 自诊断 + /diff 本会话文件改动 ===
    if name == "/status":
        return _handle_status_cli(args, rt)
    if name == "/doctor":
        return _handle_doctor_cli(args, rt)
    if name == "/diff":
        return _handle_diff_cli(args, rt)

    # === /add-dir 追加可写路径白名单（运行时生效 + 持久化）===
    if name == "/add-dir":
        return _handle_add_dir_cli(args, rt)

    # === /paste 读剪贴板图片存 .paste/ 目录 ===
    if name == "/paste":
        return _handle_paste_command(args, rt)

    # === /skill-learning 行为直觉学习链路管理 ===
    if name == "/skill-learning":
        return _handle_skill_learning_command(args, rt)

    return False


def _truncate_at_last_compact_boundary(msgs: list) -> list:
    """恢复会话时，从最后一条"[COMPACT_BOUNDARY]"（压缩边界标记）起截断。

    标记之前的旧消息已被总结进摘要，载入只会撑上下文，全部裁掉；
    标记行本身剥掉，摘要正文保留。找不到标记就原样返回。

    参数：
        msgs: 从会话库读出的消息列表
    返回：裁剪后的新列表（不修改原列表）。
    """
    last_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str) and content.startswith("[COMPACT_BOUNDARY]"):
            last_idx = i
            break
    if last_idx < 0:
        return msgs
    kept = [dict(m) for m in msgs[last_idx:]]
    # 剥掉首条的标记行（摘要正文保留）
    first = kept[0]
    content = first.get("content", "")
    first["content"] = content.replace("[COMPACT_BOUNDARY]\n", "", 1)
    logger.info(
        "resume：按 compact 边界裁剪（丢弃 %d 条 pre-compact 消息）",
        last_idx,
    )
    return kept


def _cleanup_redundant_summaries(msgs: list) -> list:
    """清理历史里多余的摘要占位（只保留最近一个）。

    多个 "[之前的对话已自动总结]" 占位同时存在时只留最近一个——
    更早的内容已被新摘要覆盖。

    参数：
        msgs: 消息列表
    返回：清理后的新列表（不修改原列表）。
    """
    summary_idx = [
        i for i, m in enumerate(msgs)
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith(
            ("[之前的对话已自动总结]", "[紧急上下文压缩")
        )
    ]
    if len(summary_idx) <= 1:
        return msgs
    drop = set(summary_idx[:-1])  # 保留最后一个摘要，其余删
    return [m for i, m in enumerate(msgs) if i not in drop]


def _handle_rewind_command(rt: RuntimeContext, args: str) -> None:
    """/rewind：列出存档点（checkpoint 快照），选一个回滚（恢复文件和/或对话）。

    存档点来自每条用户消息发出前对 agent 改过文件的自动快照。

    参数：
        rt: RuntimeContext（拿 checkpoint 管理器和 agent）
        args: 暂未使用（交互式选序号）
    """
    mgr = getattr(rt, "checkpoint_mgr", None)
    if not mgr:
        console.print("[yellow]Checkpoint 不可用（会话未初始化）[/yellow]")
        return
    snaps = mgr.list_snapshots()
    if not snaps:
        console.print("[yellow]没有 checkpoint 快照（至少一个用户 prompt 后才有）[/yellow]")
        return

    console.print("[bold]Checkpoint 快照：[/bold]")
    for i, s in enumerate(snaps):
        files = ", ".join(s["files"][:3]) + ("..." if len(s["files"]) > 3 else "")
        console.print(
            f"  [cyan]{i}[/cyan] {s['ts'][:19]} | "
            f"{len(s['files'])} 个文件 | {s['msg_count']} 条消息 | {files}"
        )
    try:
        raw = console.input(
            "[bold]回滚序号 / s+序号做摘要 / 回车取消 > [/bold] "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    # 摘要模式：s / s3 / s 0 —— 把该存档点之后的对话压成一段摘要
    if raw.lower().startswith("s"):
        idx_part = raw[1:].strip()
        if not idx_part.isdigit():
            try:
                idx_part = console.input(
                    "[bold]对哪个序号做摘要？ > [/bold] "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return
        if not idx_part.isdigit():
            return
        idx = int(idx_part)
        if idx < 0 or idx >= len(snaps):
            console.print("[red]序号越界[/red]")
            return
        _summarize_rewind(rt, snaps[idx]["id"])
        return

    if not raw.isdigit():
        return
    idx = int(raw)
    if idx < 0 or idx >= len(snaps):
        console.print("[red]序号越界[/red]")
        return
    sid = snaps[idx]["id"]

    # 4 种恢复模式菜单
    console.print(
        f"[bold]快照 {sid[:19]} 含 {len(snaps[idx]['files'])} 文件 / "
        f"{snaps[idx]['msg_count']} 条对话。恢复模式：[/bold]\n"
        "  [cyan]1[/cyan] 全恢复（代码+对话）\n"
        "  [cyan]2[/cyan] 只恢复对话\n"
        "  [cyan]3[/cyan] 只恢复代码\n"
        "  [cyan]4[/cyan] 从此压缩（截断到该点 + LLM 摘要）"
    )
    try:
        mode = console.input("[bold]选 [1-4] / 回车取消 > [/bold] ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if mode not in ("1", "2", "3", "4"):
        return

    if mode == "1" or mode == "3":
        # 恢复文件（把快照里的文件内容写回去）
        restored = mgr.restore_files(sid)
        if restored:
            console.print(f"[green]已恢复 {len(restored)} 个文件[/green]")
            for p in restored:
                console.print(f"  [dim]{p}[/dim]")
        else:
            console.print("[yellow]该快照没有可恢复的文件[/yellow]")

    if mode == "1" or mode == "2":
        # 恢复对话（把快照时的对话历史装回去）
        conv = mgr.get_conversation(sid)
        if conv and rt.agent:
            rt.agent.conversation_history = conv
            rt.agent.invalidate_system_prompt()
            console.print(f"[green]已恢复对话（{len(conv)} 条消息）[/green]")
        else:
            console.print("[yellow]该快照没有对话副本[/yellow]")

    if mode == "4":
        _summarize_rewind(rt, sid)


def _summarize_rewind(rt: RuntimeContext, sid: str) -> None:
    """把选中存档点之后的对话压成一段摘要。

    做法：保留存档点当时的对话，其后新追加的消息交给 LLM 总结；
    最终对话历史 = 存档点对话 + 一条带摘要的 user 消息。

    参数：
        rt: RuntimeContext
        sid: checkpoint 快照 ID
    """
    mgr = getattr(rt, "checkpoint_mgr", None)
    if not mgr or not rt.agent:
        console.print("[yellow]Checkpoint 不可用[/yellow]")
        return

    ckpt_conv = mgr.get_conversation(sid)
    current = rt.agent.conversation_history

    # 存档点之后 = 当前历史里、存档对话后面追加的那部分（按前缀匹配切）
    after = current
    if (ckpt_conv and len(current) > len(ckpt_conv)
            and current[:len(ckpt_conv)] == ckpt_conv):
        after = current[len(ckpt_conv):]
    if not after:
        console.print("[yellow]该 checkpoint 之后没有新对话[/yellow]")
        return

    try:
        from agent.context_compressor import _summarize_conversation
        # _summarize_conversation 是 async（因为底层 LLM 调用是 async）。
        # 本函数是同步的（被同步的命令处理函数调用），用
        # asyncio.run 桥接（跟 reflection.py:150 的处理方式相同）。
        summary = asyncio.run(_summarize_conversation(after, rt.agent.llm_client))
    except Exception as e:
        console.print(f"[red]摘要失败: {e}[/red]")
        return

    rt.agent.conversation_history = ckpt_conv + [{
        "role": "user",
        "content": f"[该点之后的对话已总结]\n{summary}",
    }]
    rt.agent.invalidate_system_prompt()
    console.print(
        f"[green]已把 checkpoint 之后的 {len(after)} 条消息压成摘要[/green]"
    )


def _show_help():
    """打印 /help 帮助面板（列出所有可用命令）。参数：无。"""
    console.print(Panel(
        "[bold]可用命令[/bold]\n\n"
        "[cyan]/new[/cyan]       开始新对话\n"
        "[cyan]/skills[/cyan]    列出技能（/skills rate <name> <1-5> | /skills recommend）\n"
        "[cyan]/memory[/cyan]    查看记忆（输入 m 编辑 MEMORY.md / u 编辑 USER.md）\n"
        "[cyan]/sessions[/cyan]  列出历史会话\n"
        "[cyan]/resume[/cyan]    恢复历史会话（/resume [序号]）\n"
        "[cyan]/search[/cyan]    搜索历史对话（/search <关键词>）\n"
        "[cyan]/usage[/cyan]     显示工具用量\n"
        "[cyan]/stats[/cyan]     会话统计（跨会话聚合）\n"
        "[cyan]/model[/cyan]     切换模型（/model [name]）\n"
        "[cyan]/plan[/cyan]      进入计划模式（/plan off 强制退出）\n"
        "[cyan]/permission[/cyan]  查看或切换权限模式（/permission [default|bypass|acceptEdits]）\n"
        "[cyan]/sandbox[/cyan]    开启/关闭 OS 沙箱（/sandbox [on|off|status]，Linux 用 bwrap、macOS 用 sandbox-exec、Windows 用 Job Object）\n"
        "[cyan]/hooks[/cyan]    查看会话启动时锁定的 hook 快照（含磁盘 diff 检测）\n"
        "[cyan]/agents[/cyan]   列出自定义子代理（来自 ~/.OmniMate/agents/*.md）\n"
        "[cyan]/approved[/cyan]  管理审批白名单\n"
        "[cyan]/rewind[/cyan]    回滚到某个 checkpoint（恢复文件 + 可选对话）\n"
        "[cyan]/handoff[/cyan]   会话移交（save/load/list/show/delete/export/import）\n"
        "[cyan]/goal[/cyan]      目标驱动多轮（/goal <obj>|status|pause|resume|clear|tasks）\n"
        "[cyan]/poor[/cyan]      穷鬼模式（on|off|status，一键关烧钱功能）\n"
        "[cyan]/trace[/cyan]     本地 trace（today|yesterday|<date>|tail [N]）\n"
        "[cyan]/mailbox[/cyan]   队友邮箱（send|check|clear）\n"
        "[cyan]/inbox[/cyan]     显示 ChannelInbox 未消费消息\n"
        "[cyan]/resume_bundle[/cyan]  跨项目恢复 bundle（/resume_bundle [id]）\n"
        "[cyan]/resumable[/cyan] 列出/恢复可续跑子代理（/resumable [agent_id]）\n"
        "[cyan]/compact[/cyan]   手动压缩上下文（L4 摘要；--yes 跳过确认）\n"
        "[cyan]/context[/cyan]   显示上下文 token 分布与压缩状态\n"
        "[cyan]/status[/cyan]    状态一览（模型/goal/MCP/工具数）\n"
        "[cyan]/doctor[/cyan]    自诊断 6 项（配置/API key/目录/依赖）\n"
        "[cyan]/diff[/cyan]      本会话文件改动（checkpoint 追踪）\n"
        "[cyan]/add-dir[/cyan]   追加 safe_path 写白名单（无参数列出；运行时生效 + 持久化到 config）\n"
        "[cyan]/paste[/cyan]     保存剪贴板图片到 .paste/（Windows；之后在消息中引用路径让 AI 分析）\n"
        "[cyan]/init[/cyan]      生成当前项目的 OMNIMATE.md（已存在不覆盖，--force 覆盖）\n"
        "[cyan]/cache-stats[/cyan]  prompt cache 命中统计与 break 根因\n"
        "[cyan]/skill-learning[/cyan]  行为学习（status|start|stop|evolve|prune）\n"
        "[cyan]/help[/cyan]      显示本帮助\n"
        "[cyan]/quit[/cyan]      退出\n\n"
        "[dim]输入 /技能名 触发对应技能[/dim]",
        border_style="blue",
    ))


# ---------------------------------------------------------------------------
# goal / poor / output-style / trace / history / mailbox 等命令的处理函数
# ---------------------------------------------------------------------------

def _goal_state_path(rt) -> Path:
    """goal（目标驱动状态）的持久化文件路径：~/.OmniMate/.goal/current.json。

    参数：
        rt: RuntimeContext
    返回：Path 对象。
    """
    return Path(rt.home) / ".goal" / "current.json"


def _goal_status_output(rt, capsys_safe: bool = True) -> None:
    """打印当前 goal 的状态（目标、进度、token 预算等，直接打到屏幕）。

    参数：
        rt: RuntimeContext
        capsys_safe: 兼容参数（测试捕获 stdout 场景）
    """
    gs = getattr(rt.agent, "_goal_state", None)
    if gs is None:
        console.print("[yellow]无 active goal（未启用目标驱动）[/yellow]")
        console.print(
            "[dim]用法: /goal <objective> 启动新目标 | "
            "/goal status 看状态 | /goal clear 取消[/dim]"
        )
        return
    console.print(f"[cyan]objective:[/cyan] {gs.objective}")
    console.print(f"[cyan]status:[/cyan] {gs.status}")
    console.print(f"[cyan]iteration:[/cyan] {gs.iteration_count}")
    console.print(f"[cyan]token_budget:[/cyan] {gs.token_budget}/{gs.token_budget_limit or '∞'}")
    if gs.pause_reason:
        console.print(f"[cyan]pause_reason:[/cyan] {gs.pause_reason}")
    if gs.task_ids:
        console.print(f"[cyan]task_ids:[/cyan] {len(gs.task_ids)} 个")


def _start_new_goal(rt, objective: str) -> None:
    """启动新 goal（目标驱动：给 AI 一个目标，它多轮自主推进直到完成）。

    核心三步（暂停旧 goal / 建新 GoalState / 持久化 + 挂到 agent）走
    agent/goal.start_goal_agent 共享函数（跟 LLM 的
    goal_start 工具同一来源）；CLI 这层只负责屏幕输出 / 辅助模型拆解 /
    往对话历史注入目标。

    参数：
        rt: RuntimeContext
        objective: 用户输入的目标描述
    """
    from agent.goal import start_goal_agent

    # 0. 先记下旧 goal（共享函数会暂停它，这里记下来只为打印提示）
    old_gs = getattr(rt.agent, "_goal_state", None)
    will_pause_old = old_gs is not None and old_gs.status == "active"

    # 1-2. 核心三步：暂停旧 goal + 建新 GoalState + 持久化 + 挂到 agent
    goal_cfg = rt.config.get("goal", {}) if rt.config else {}
    budget_limit = goal_cfg.get("default_token_budget", 200_000)
    gs = start_goal_agent(
        rt.agent, objective, token_budget=budget_limit,
        persist_path=_goal_state_path(rt),
    )

    if will_pause_old:
        console.print(f"[dim]已自动 pause 旧 goal: {old_gs.objective}[/dim]")

    # 3. 尝试用辅助模型把目标拆解成子任务（宽松失败：没配辅助模型就跳过）
    # 直接走 agent.aux_llm_router（不要走 RuntimeContext 上
    # 的死字段；工具分发路径早已统一到 agent_ref.aux_llm_router）
    aux_client = getattr(rt.agent, "aux_llm_router", None)
    if aux_client is not None:
        try:
            # 拆解函数是异步的：CLI 同步路径用 asyncio.run 包一层
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # 已经在事件循环里（比如测试环境）——再 asyncio.run 会报错，跳过
                    task_ids = []
                else:
                    task_ids = _asyncio.run(
                        _goal_decompose_safe(gs, objective, aux_client)
                    )
            except RuntimeError:
                task_ids = _asyncio.run(
                    _goal_decompose_safe(gs, objective, aux_client)
                )
            if task_ids:
                gs.save(_goal_state_path(rt))
                console.print(f"[dim]aux_llm 拆出 {len(task_ids)} 个子任务[/dim]")
        except Exception as e:
            logger.warning("goal decompose 失败（fail-open）: %s", e)

    # 4. 把目标作为下一轮的 user 输入塞进对话历史（让 agent 开始追目标）
    # 设计：直接追加到 conversation_history 末尾，主循环下一轮自然看到。
    # ⚠️ 只能在 CLI 层（会话循环外）这么做；工具路径（goal_start 工具）里
    # 在 assistant(tool_calls) 和 tool 结果之间插 user 消息会破坏
    # "user/assistant 严格交替"的消息协议。
    if hasattr(rt.agent, "conversation_history"):
        rt.agent.conversation_history.append({
            "role": "user",
            "content": f"[goal_start] {objective}",
        })

    console.print(
        f"[green]✓ goal 已启动：{objective}[/green] "
        f"[dim](budget={budget_limit}, iteration=0)[/dim]"
    )


async def _goal_decompose_safe(gs, objective: str, aux_client):
    """给"用 LLM 拆解目标"套一层保险：出任何异常都返回空列表而不是炸出去。

    参数：
        gs: GoalState 目标状态对象
        objective: 目标描述
        aux_client: 辅助模型客户端
    返回：拆出的子任务 ID 列表（失败为空列表）。
    """
    from agent.goal import decompose_with_llm
    try:
        return await decompose_with_llm(gs, objective, aux_client)
    except Exception as e:
        logger.warning("goal decompose 异常（fail-open）: %s", e)
        return []


def _handle_goal_command(args: str, rt) -> bool:
    """处理 /goal 命令（目标驱动的总开关）。

    用法：
    /goal <objective>       启动新 goal
    /goal status            查看状态
    /goal pause [reason]    手动暂停
    /goal resume            恢复
    /goal continue          立即触发下一轮（暂停 → 进行中）
    /goal clear             取消
    /goal tasks             列出关联任务

    参数：
        args: 子命令 + 参数
        rt: RuntimeContext
    返回：True（已处理）。
    """
    parts = args.split(None, 1) if args else []
    if not parts:
        _goal_status_output(rt)
        return True
    sub = parts[0]

    if sub == "status":
        _goal_status_output(rt)
        return True

    if sub == "clear":
        gs = getattr(rt.agent, "_goal_state", None)
        if gs is None:
            console.print("[yellow]无 active goal[/yellow]")
            return True
        gs.cancel()
        gs.save(_goal_state_path(rt))
        rt.agent.set_goal_state(None)
        # 同时删掉持久化文件，不留僵尸状态
        try:
            p = _goal_state_path(rt)
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning("goal 持久化文件删除失败（忽略）: %s", e)
        console.print("[green]goal 已取消[/green]")
        return True

    if sub == "pause":
        gs = getattr(rt.agent, "_goal_state", None)
        if gs is None:
            console.print("[yellow]无 active goal[/yellow]")
            return True
        reason = parts[1] if len(parts) > 1 else "manual"
        gs.pause(reason=reason)
        gs.save(_goal_state_path(rt))
        console.print(f"[green]goal 已 pause（reason={reason}）[/green]")
        return True

    if sub in ("resume", "continue"):
        gs = getattr(rt.agent, "_goal_state", None)
        if gs is None:
            console.print("[yellow]无 active goal[/yellow]")
            return True
        gs.resume()
        gs.save(_goal_state_path(rt))
        console.print("[green]goal 已 resume[/green]")
        return True

    if sub == "tasks":
        gs = getattr(rt.agent, "_goal_state", None)
        if gs is None or not gs.task_ids:
            console.print("[yellow]无关联 task（goal 未启动或未拆解）[/yellow]")
            return True
        try:
            from agent.task_store import get_task_store
            store = get_task_store()
            for tid in gs.task_ids:
                t = store.get(tid)
                if t:
                    console.print(
                        f"  [cyan]{tid}[/cyan] [{t['status']}] {t['subject']}"
                    )
                else:
                    console.print(f"  [dim]{tid}（已删）[/dim]")
        except Exception as e:
            console.print(f"[red]列 task 失败：[/red]{e}")
        return True

    # 没匹配到任何子命令 → 把整串当作目标描述启动新 goal
    objective = args.strip()
    if not objective:
        console.print("[yellow]用法: /goal <objective> | status | pause | resume | clear | tasks[/yellow]")
        return True
    _start_new_goal(rt, objective)
    return True


def _handle_output_style_command(args: str, rt) -> bool:
    """/output-style：列出/切换/关闭输出风格。

    输出风格 = 一段预设提示词。目录：`<项目>/.omnimate/output-styles/`
    （优先，覆盖同名）+ `~/.OmniMate/output-styles/`。切换时写
    settings.json 顶层的 output_style 并作废缓存的 system prompt
    （下一条消息生效）。

    参数：
        args: 风格名 / off / 空（空=列表）
        rt: RuntimeContext
    返回：True（已处理）。
    """
    from agent.output_styles import discover_output_styles

    cwd = os.getcwd()
    styles = discover_output_styles(cwd, rt.home)
    current = (rt.config or {}).get("output_style")

    arg = args.strip() if args else ""
    if not arg:
        if not styles:
            console.print(
                "[dim]暂无输出风格。创建：~/.OmniMate/output-styles/<名>.md"
                " 或 .omnimate/output-styles/<名>.md（正文即提示词）[/dim]"
            )
            return True
        lines = []
        for name in sorted(styles):
            mark = " [cyan](当前)[/cyan]" if name == current else ""
            desc = f" — {styles[name].description}" if styles[name].description else ""
            lines.append(f"- {name}{desc}{mark}")
        console.print(Panel.fit("\n".join(lines), title=f"输出风格（当前: {current or '默认'}）"))
        return True

    if arg == "off":
        _set_output_style(rt, None)
        console.print("[green]输出风格已关闭（默认输出）[/green]")
        return True

    if arg not in styles:
        console.print(f"[red]没有找到输出风格: {arg}[/red]（可用: {sorted(styles)}）")
        return True

    _set_output_style(rt, arg)
    console.print(
        f"[green]输出风格已切换: {arg}[/green] "
        "[dim]（下一条消息生效；system prompt 已重建）[/dim]"
    )
    return True


def _set_output_style(rt, value):
    """切换输出风格：改运行时配置 + 写 settings.json + 作废缓存的 system prompt。

    宽松失败：写盘失败只记日志（当前会话内仍生效）。

    参数：
        rt: RuntimeContext
        value: 风格名（None = 关闭）
    """
    from agent.settings import load_settings, save_settings
    rt.config["output_style"] = value
    try:
        data = load_settings()
        data["output_style"] = value
        save_settings(data)
    except Exception as e:
        logger.warning("output_style 持久化失败（会话内仍生效）: %s", e)
    try:
        rt.agent.invalidate_system_prompt()
    except Exception:
        pass


def _handle_poor_command(args: str, rt) -> bool:
    """处理 /poor on|off|status——省钱模式（一键关掉所有烧 token 的功能）。

    参数：
        args: on / off / status（空 = status）
        rt: RuntimeContext
    返回：True（已处理）。
    """
    arg = args.strip().lower() if args else ""
    if not arg or arg == "status":
        is_on = getattr(rt, "_poor_mode_on", False)
        console.print(f"Poor Mode: [cyan]{'ON' if is_on else 'OFF'}[/cyan]")
        return True
    if arg == "on":
        import copy as _copy_mod
        from agent.poor_mode import apply_poor_preset
        # 开启前先给配置拍快照，关闭时完整还原
        if not getattr(rt, "_poor_config_snapshot", None):
            rt._poor_config_snapshot = _copy_mod.deepcopy(rt.config)
        rt.config = apply_poor_preset(rt.config, on=True)
        rt._poor_mode_on = True
        console.print(
            "[green]Poor Mode 已开启（runtime）[/green] "
            "[dim]（reflection/9段摘要/cache监控等已关；/poor off 可回滚）[/dim]"
        )
        return True
    if arg == "off":
        rt._poor_mode_on = False
        snapshot = getattr(rt, "_poor_config_snapshot", None)
        if snapshot is not None:
            rt.config = snapshot
            rt._poor_config_snapshot = None
            console.print(
                "[green]Poor Mode 已关闭，配置已回滚到开启前[/green]"
            )
        else:
            console.print(
                "[green]Poor Mode 已关闭（runtime 标记）[/green] "
                "[dim]（本会话未开启过，无配置需要回滚）[/dim]"
            )
        return True
    console.print("[yellow]用法: /poor on|off|status[/yellow]")
    return True




def _handle_trace_command(args: str, rt) -> bool:
    """/trace——查看本地运行轨迹（今天的/昨天的/指定日期的，或最近 N 条）。

    参数：
        args: today / yesterday / YYYY-MM-DD / tail [N]（空 = today）
        rt: RuntimeContext
    返回：True（已处理）。
    """
    sink = getattr(rt, "trace_sink", None)
    if sink is None:
        console.print("[yellow]Trace 未启用（config.trace.enabled=False 或未初始化）[/yellow]")
        return True
    parts = (args or "today").split()
    when = parts[0] if parts else "today"

    if when == "tail":
        n = 10
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                pass
        records = sink.query(limit=n)
        if not records:
            console.print("[yellow]无 trace 记录[/yellow]")
            return True
        for r in records:
            console.print(f"  [{r.get('ts', '')[:19]}] {r.get('event', '?')}")
        return True

    # 汇总路径：today / yesterday / 具体日期
    import datetime as _dt
    if when == "today":
        date_str = _dt.datetime.now().strftime("%Y-%m-%d")
    elif when == "yesterday":
        date_str = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        date_str = when  # 其他情况：假定用户直接给了 YYYY-MM-DD 日期

    try:
        summary = sink.summary(date_str=date_str)
    except Exception as e:
        console.print(f"[red]trace 查询失败：[/red]{e}")
        return True

    if not summary.get("total_events"):
        console.print(f"[yellow]{date_str} 无 trace 记录[/yellow]")
        return True

    console.print(f"[cyan]{date_str} trace summary[/cyan]")
    console.print(f"  total_events: {summary['total_events']}")
    console.print(f"  input_tokens: {summary['total_input_tokens']}")
    console.print(f"  output_tokens: {summary['total_output_tokens']}")
    console.print(f"  error_count: {summary['error_count']}")
    console.print(f"  by_event: {summary['by_event']}")
    return True


def _handle_init_command(rt, args: str) -> bool:
    """/init——给当前项目生成一份"项目说明书"OMNIMATE.md。

    流程：收集项目信息（目录树/关键文件/类型统计，缺哪跳哪）→
    主 LLM 按四段式生成（项目本质/常用命令/架构/约定）→
    写到 cwd/OMNIMATE.md（下次会话自动注入 system prompt，AI 进门就懂项目）。

    - /init            已存在不覆盖
    - /init --force    覆盖重新生成

    参数：
        rt: RuntimeContext
        args: 可含 --force
    返回：True（已处理）。
    """
    # 局部 import：跟 codebase 其他地方保持一致（agent_defs/prompt_builder 等都这么干）
    from agent.workspace_context import get_workspace_cwd

    force = "--force" in (args or "")
    cwd = Path(get_workspace_cwd())
    target = cwd / "OMNIMATE.md"
    if target.exists() and not force:
        console.print(
            "[yellow]OMNIMATE.md 已存在[/yellow] "
            "[dim]（/init --force 覆盖重新生成）[/dim]"
        )
        return True

    # ── 收集项目信息（宽松失败：哪一步收集不了就跳过）──
    parts = []

    # 1. 目录结构（顶层 + 部分二层）
    try:
        skip = {
            "node_modules", ".git", "__pycache__", ".venv",
            "dist", "build", ".next", "target",
        }
        entries = sorted(
            e.name for e in cwd.iterdir()
            if e.name not in skip and not e.name.startswith(".")
        )
        sub_tree = []
        for name in entries[:20]:
            if (cwd / name).is_dir():
                try:
                    subs = [
                        f"{name}/{s}" for s in
                        sorted(p.name for p in (cwd / name).iterdir())[:10]
                    ]
                    sub_tree.extend(subs[:10])
                except OSError:
                    pass
            else:
                sub_tree.append(name)
        parts.append(
            "## 目录结构（顶层 + 部分二层）\n"
            + "\n".join(entries + sub_tree[:60])
        )
    except Exception as e:
        logger.debug("收集目录结构失败（跳过）: %s", e)

    # 2. 关键配置文件（每个只读前 4KB，够 LLM 认出项目类型了）
    for fname in (
        "README.md", "README.rst", "pyproject.toml",
        "package.json", "requirements.txt", "Makefile",
        "setup.py", "go.mod", "Cargo.toml",
    ):
        f = cwd / fname
        if f.exists():
            try:
                content = f.read_text(encoding="utf-8", errors="replace")[:4096]
                parts.append(f"## {fname}\n{content}")
            except Exception:
                continue

    # 3. 文件类型统计（前 5）
    try:
        from collections import Counter
        exts = Counter(
            p.suffix for p in cwd.rglob("*")
            if p.is_file() and p.suffix
            and ".git" not in str(p) and "node_modules" not in str(p)
        )
        top = ", ".join(f"{e}({c})" for e, c in exts.most_common(5))
        parts.append(f"## 文件类型统计（前 5）\n{top}")
    except Exception as e:
        logger.debug("文件类型统计失败（跳过）: %s", e)

    info = "\n\n".join(parts) or "（空项目，无可用信息）"

    # ── 拼好提示词 → 主 LLM 按四段式生成 ──
    prompt = (
        "根据以下项目信息生成 OMNIMATE.md（项目指导文件，给 AI 编程助手看）。"
        "输出 Markdown，含且仅含这四节：\n"
        "## 项目本质（一句话 + 核心技术栈）\n"
        "## 常用命令（运行/测试/构建，从配置文件推断）\n"
        "## 架构（模块划分 + 依赖方向）\n"
        "## 约定（语言/编码/测试等能从项目验证的规则）\n\n"
        "要求：只写能从信息中验证的事实，不猜测；"
        "命令给出具体形式（如 uv run pytest tests/）；中文书写。\n\n"
        f"# 项目信息\n{info}"
    )

    async def _gen():
        resp = await rt.agent.llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    # asyncio.run 在已有事件循环时会抛 RuntimeError（测试环境/交互式解释器）
    # —— 用 try/except 兜底，跟 _start_new_goal 的处理方式相同
    text = ""
    try:
        text = asyncio.run(_gen())
    except RuntimeError:
        # 已在事件循环里（如 pytest-asyncio 接管时）→ 放弃生成
        #（对齐 _start_new_goal：循环已跑时再驱动会冲突；
        #  正常同步 CLI 路径不会进这个分支；测试环境走 mock 不依赖真 LLM）
        logger.warning("init 生成跳过（已有运行中的事件循环）")
        text = ""
    except Exception as e:
        logger.warning("init 生成失败（LLM 调用异常）: %s", e)
        text = ""

    if not text.strip():
        console.print("[red]OMNIMATE.md 生成失败（LLM 返回空）[/red]")
        return True

    try:
        target.write_text(text.strip() + "\n", encoding="utf-8")
    except Exception as e:
        console.print(f"[red]写入失败：[/red]{e}")
        return True

    console.print(
        f"[green]✓ OMNIMATE.md 已生成[/green] "
        f"[dim]（{target}，下次会话自动注入 system prompt）[/dim]"
    )
    return True


def _handle_inbox_command(args: str, rt) -> bool:
    """/inbox——显示收件箱里还没消费的消息（MCP 外部工具服务推送的内容）。

    本命令只做只读展示；"标记已消费"由主循环组装轮次消息时完成。

    参数：
        args: 未使用
        rt: RuntimeContext
    返回：True（已处理）。
    """
    # 收件箱挂在 AIAgent._channel_inbox（由 setter 注入）
    inbox = getattr(rt.agent, "_channel_inbox", None)
    if inbox is None:
        console.print("[yellow]ChannelInbox 未初始化（无 MCP server 推送）[/yellow]")
        return True
    try:
        msgs = inbox.unconsumed()
    except Exception as e:
        console.print(f"[red]读取 inbox 失败：[/red]{e}")
        return True
    if not msgs:
        console.print("[green]收件箱为空[/green]")
        return True
    console.print(f"[cyan]未消费消息 {len(msgs)} 条：[/cyan]")
    for m in msgs:
        server = m.get("server", "?")
        ts = m.get("ts", "")
        payload = m.get("payload", {})
        payload_str = json.dumps(payload, ensure_ascii=False)
        if len(payload_str) > 200:
            payload_str = payload_str[:200] + "..."
        console.print(f"  [{ts}] [{server}] {payload_str}")
    return True


def _handle_mailbox_command(args: str, rt) -> bool:
    """/mailbox send|check|clear——队友邮箱的命令行包装（给队友发信/查信/清空）。

    参数：
        args: 子命令 + 参数（send <收件人> <内容> / check / clear）
        rt: RuntimeContext
    返回：True（已处理）。
    """
    mailbox = getattr(rt, "mailbox", None)
    if mailbox is None:
        console.print("[red]mailbox 未初始化[/red]")
        return True
    agent_name = getattr(rt, "agent_name", "main")

    parts = args.split(None, 1)
    sub = parts[0].lower() if parts else "help"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("help", "h", "?", ""):
        console.print(Panel(
            "[bold]/mailbox 子命令[/bold]\n\n"
            "[cyan]/mailbox send <to> <content>[/cyan]  发邮件给另一个 agent\n"
            "[cyan]/mailbox check[/cyan]                列自己邮箱全部邮件（含已读）\n"
            "[cyan]/mailbox check --unread[/cyan]       只看未读\n"
            "[cyan]/mailbox clear[/cyan]                清空自己邮箱\n",
            border_style="blue",
        ))
        return True

    if sub == "send":
        send_parts = rest.split(None, 1)
        if len(send_parts) < 2:
            console.print("[yellow]用法: /mailbox send <to> <content>[/yellow]")
            return True
        to, content = send_parts[0], send_parts[1]
        msg_id = mailbox.send(to=to, from_=agent_name, content=content)
        console.print(f"[green]✓ 已投递给 {to}（id={msg_id}）[/green]")
        return True

    if sub in ("check", "ls", "list"):
        # 默认读全部（含已读）。不能默认只看未读：主循环每轮查完未读
        # 就立刻标记已读——等用户敲 /mailbox check 时邮件早被清空，
        # 永远显示空邮箱。加 --unread / -u 才过滤未读；"all" 关键字向后兼容。
        rest_lower = rest.lower()
        if "--unread" in rest_lower or "-u" in rest_lower.split():
            unread_only = True
        elif "all" in rest_lower:
            unread_only = False
        else:
            unread_only = False  # 默认显示全部（理由见上）
        if unread_only:
            msgs = mailbox.check_unread(agent_name)
            label = "未读"
        else:
            msgs = mailbox.check_all(agent_name)
            label = "全部"
        if not msgs:
            console.print(f"[green]{agent_name} 邮箱无{label}邮件[/green]")
            return True
        console.print(f"[cyan]{agent_name} 邮箱{label}邮件 {len(msgs)} 条：[/cyan]")
        for m in msgs:
            ts = m.get("ts", "")
            frm = m.get("from", "?")
            content = m.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            read_flag = "" if not m.get("read") else "[已读]"
            console.print(f"  [{ts}] from={frm}{read_flag}: {content}")
        return True

    if sub == "clear":
        count = mailbox.clear(agent_name)
        console.print(f"[green]已清空 {count} 条邮件[/green]")
        return True

    console.print(f"[yellow]未知子命令：{sub}（send/check/clear）[/yellow]")
    return True




def _handle_resumable_command(args: str, rt) -> bool:
    """/resumable [agent_id]——列出/恢复可续跑的子代理。

    - 无参：列出还标着 running 的子代理（agent_id/状态/消息数/时间），
      并提示语义边界——只有存了运行轨迹（transcript）的子代理才能真正恢复
    - 有参：按 agent_id 调恢复函数续命，成功用绿色显示结果前 2000 字，
      失败红色提示

    实现要点：
    - subagent_persistence 的 list_resumable() 不接收 base_dir 参数
      （它内部用模块级的 _sessions_dir()），所以本函数也不传；
      测试通过 monkeypatch _sessions_dir 来隔离环境
    - LLM 配置和 agent_ref 都从 rt 取（跟 /trace、/poor 同款模式）

    参数：
        args: 空 或 agent_id（可后附自定义续跑指令）
        rt: RuntimeContext
    返回：True（已处理）。
    """
    import json as _json
    from agent import subagent_persistence as sp

    parts = (args or "").split()

    # ── 无参：列清单分支 ──
    if not parts:
        try:
            items = sp.list_resumable()
        except Exception as e:
            console.print(f"[red]列出子代理失败：[/red]{e}")
            return True

        if not items:
            console.print("[dim]无中断的子代理可恢复[/dim]")
            return True

        console.print(f"[cyan]可恢复的子代理（{len(items)} 个，status=running）：[/cyan]")
        for it in items:
            aid = it.get("agent_id", "?")
            status = it.get("status", "?")
            # 消息数：现场读轨迹文件数一遍（元数据里没存）
            try:
                n_msgs = len(sp.load_transcript(aid))
            except Exception:
                n_msgs = "?"
            # 时间：优先 updated_at，其次 created_at
            ts = it.get("updated_at") or it.get("created_at") or ""
            if isinstance(ts, (int, float)):
                import time as _time
                ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(ts))
            # agent_type 纯展示用
            atype = it.get("agent_type", "")
            atype_tag = f"[dim]({atype})[/dim] " if atype else ""
            # 关键细节：agent_id 里的 [ 要转义——不转义会被 Rich 渲染库
            # 当成样式标签吃掉导致显示乱
            console.print(
                f"  \\[{aid}] {atype_tag}status={status} msgs={n_msgs} {ts}"
            )
        console.print(
            "[dim]用 /resumable <agent_id> 恢复。"
            "（注：只有存了 transcript 的子代理可恢复，"
            "真正中断可能没有轨迹）[/dim]"
        )
        return True

    # ── 有参：恢复分支 ──
    agent_id = parts[0]
    instruction = "继续完成剩余工作"
    # 支持用户在 agent_id 后面附带自定义的续跑指令
    if len(parts) > 1:
        instruction = " ".join(parts[1:])

    from tools.subagent_resume_tool import _run_resume

    # 从 rt 取上下文（LLM 配置 + agent 引用），跟 /poor、/trace 的取法一致
    agent_ref = getattr(rt, "agent", None)
    config = getattr(rt, "config", None)
    memory_store = getattr(rt, "memory_store", None)

    try:
        result_json = _run_resume(
            agent_id,
            instruction,
            agent_ref=agent_ref,
            config=config,
            memory_store=memory_store,
        )
    except Exception as e:
        console.print(f"[red]恢复失败：[/red]{e}")
        return True

    try:
        data = _json.loads(result_json)
    except Exception:
        # 不是 JSON 就原样显示（理论上不会——恢复函数必定返回 JSON）
        console.print(f"[yellow]{result_json[:2000]}[/yellow]")
        return True

    if "error" in data:
        err = data.get("error", "")
        console.print(f"[red]恢复失败：[/red]{err}")
        return True

    # 成功：绿色显示结果前 2000 字
    text = data.get("result", "")
    truncated_tag = " [dim](已截断到 2000 字符)[/dim]" if len(text) > 2000 else ""
    console.print(f"[green]恢复完成：[/green]{text[:2000]}{truncated_tag}")
    return True
















def _handle_diff_cli(args: str, rt) -> bool:
    """/diff——列出本会话改过哪些文件。

    基于快照管理器已有的数据：列出被编辑工具改过的文件清单 + 快照数。
    没有快照管理器或没有改动记录时给出提示。

    参数：
        args: 未使用
        rt: RuntimeContext
    返回：True（已处理）。
    """
    mgr = getattr(rt, "checkpoint_mgr", None)
    if mgr is None:
        console.print(
            "[yellow]本会话无 checkpoint 记录（文件快照未启用）[/yellow]"
        )
        return True

    try:
        files = list(mgr.tracked_files())
    except Exception as e:
        console.print(f"[red]读取改动记录失败：{e}[/red]")
        return True

    if not files:
        console.print(
            "[yellow]本会话无文件改动记录（write_file/str_replace 修改过的文件会出现在这里）[/yellow]"
        )
        return True

    console.print(f"[cyan]本会话改动文件（{len(files)}）：[/cyan]")
    for f in files:
        console.print(f"  [red]M[/red] {f}")
    try:
        snaps = list(mgr.list_snapshots())
        console.print(f"[dim]checkpoint 快照数：{len(snaps)}（/rewind 可回滚）[/dim]")
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# /add-dir 运行时白名单 + 持久化
# ---------------------------------------------------------------------------

def _persist_extra_root(root: str) -> bool:
    """把一个额外可写目录持久化到 settings.json 的 security.extra_allowed_roots。

    写法是"读-改-写"：先读现有内容 → 追加（去重）→ 原子写回。
    必须走 settings.json 这条真实配置轨——
    load_config 只读 settings.json，config.yaml 首次启动就被改名 .bak，
    往 yaml 写等于写进黑洞，灌不回来。

    参数：
        root: 要持久化的目录路径字符串
    返回：True=新写入；False=已存在（幂等，不重复写）。
    宽松失败：读/写失败抛异常给调用方（命令层捕获提示，不影响运行时白名单）。
    """
    from agent.settings import persist_extra_allowed_root

    # 逻辑下沉到 agent/settings.py（跟写路径审批"总是允许"档共用同一通道）
    return persist_extra_allowed_root(root)


def _load_persisted_extra_roots(config: dict) -> int:
    """启动时把 settings.json 里记的额外可写目录灌进运行时白名单。

    由 RuntimeContext.__init__ 调用（config 来自 load_config()，默认读
    settings.json——跟 _persist_extra_root 写的是同一个文件，读写闭环）。

    参数：
        config: 配置字典
    返回：成功加载的条数（宽松失败：单条失败跳过）。
    """
    from agent.permission import add_extra_allowed_root
    sec = (config or {}).get("security") or {}
    roots = sec.get("extra_allowed_roots") or []
    loaded = 0
    for root in roots:
        if not isinstance(root, str):
            continue
        try:
            if add_extra_allowed_root(root):
                loaded += 1
        except Exception as e:
            logger.warning("加载 extra_allowed_root 失败（跳过 %s）: %s", root, e)
    return loaded


def _handle_add_dir_cli(args: str, rt) -> bool:
    """/add-dir <目录>——把一个目录加进"AI 可写范围"白名单。

    - 无参数：列出当前白名单（默认的 cwd + ~/.OmniMate，加上追加过的）
    - 带目录：校验目录真实存在 → 运行时立即生效（去重幂等）→
      持久化到 settings.json（下次启动自动加载）
    - 安全底线不变：受保护路径（~/.ssh 等）和项目代码写保护在
      safe_path 里排在白名单检查之前，加白名单绕不过这些底线。

    参数：
        args: 空 或 目录路径
        rt: RuntimeContext
    返回：True（已处理）。
    """
    from agent.permission import (
        default_allowed_roots, list_extra_allowed_roots,
        add_extra_allowed_root,
    )

    parts = (args or "").split()
    if not parts:
        # 无参：列出当前白名单
        extras = {str(r) for r in list_extra_allowed_roots()}
        console.print("[bold]当前 safe_path 写白名单：[/bold]")
        for r in default_allowed_roots():
            mark = "  [cyan](/add-dir 追加)[/cyan]" if str(r) in extras else ""
            console.print(f"  {r}{mark}")
        console.print(
            "[dim]用法: /add-dir <目录> 追加白名单（运行时生效 + 持久化到 settings.json）[/dim]"
        )
        return True

    target = Path(parts[0]).expanduser().resolve()
    if not target.is_dir():
        console.print(f"[red]目录不存在:[/red] {target}")
        return True

    # 1) 运行时生效：追加进默认可写根目录
    added = add_extra_allowed_root(target)
    # 2) 持久化：写进 settings.json（读-改-写）
    try:
        persisted = _persist_extra_root(str(target))
    except Exception as e:
        persisted = False
        console.print(f"[yellow]⚠️  持久化到 settings.json 失败（运行时仍生效）: {e}[/yellow]")

    if added or persisted:
        console.print(f"[green]已添加白名单:[/green] {target}")
        if not persisted:
            console.print("[dim]（该目录已在 settings.json 中，未重复写入）[/dim]")
    else:
        console.print(f"[yellow]已在白名单中（幂等跳过）:[/yellow] {target}")
    return True


# ---------------------------------------------------------------------------
# /paste 剪贴板图片保存
# ---------------------------------------------------------------------------

def _handle_paste_command(args: str, rt) -> bool:
    """/paste——把剪贴板里的图片存到工作区 .paste/ 目录。

    小而美的 Windows 快捷路径：用 PowerShell 的剪贴板 API 读图 →
    存成 PNG → 打印路径。只保存、不自动分析（用户可能想配段文字再发给 AI）。

    宽松失败约束（设计如此，不是 bug）：
    - 非 Windows 平台不起子进程，直接提示手动保存
    - PowerShell 失败/超时/剪贴板里没图 → 提示手动给路径，不崩
    - 保存路径用 get_workspace_cwd()（让 worktree 子代理各存各的工作区）

    参数：
        args: 未使用
        rt: RuntimeContext
    返回：True（已处理）。
    """
    from datetime import datetime

    from agent.workspace_context import get_workspace_cwd

    paste_dir = Path(get_workspace_cwd()) / ".paste"
    out = paste_dir / f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
        "if ($img -eq $null) { exit 2 } "
        f"$img.Save('{out}'); exit 0"
    )
    try:
        if sys.platform != "win32":
            # 非 Windows 没有 PowerShell 剪贴板，直接走宽松失败提示
            raise RuntimeError("仅支持 Windows（其他平台请手动保存图片后给路径）")
        paste_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, timeout=15,
        )
        if r.returncode == 2:
            console.print("[yellow]剪贴板中没有图片[/yellow]")
            return True
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"PowerShell 退出码 {r.returncode}")
        console.print(
            f"[green]已保存剪贴板图片:[/green] {out}\n"
            "[dim]在消息中引用该路径即可让 AI 分析[/dim]"
        )
    except Exception as e:
        console.print(
            f"[yellow]粘贴失败（{e}）。可手动保存图片后在消息中给路径。[/yellow]"
        )
    return True


























def _manage_whitelist(rt: RuntimeContext, args: str):
    """管理审批白名单（/approved）——查看/删除各种"免审批"记录。

    用法：
    /approved                    列出已批准命令 + 前缀规则 + 持久化写入根目录
    /approved remove <序号|命令>   按序号/命令移除已批准命令
    /approved remove-root <序号|路径>  移除持久化写入根目录（同时删
                                       settings.json 记录和运行时白名单）
    /approved remove-prefix <p序号|前缀>  移除前缀规则（前缀规则
                                          只从安全的精选表派生，见 agent/command_prefix.py）

    参数：
        rt: RuntimeContext
        args: 子命令 + 参数（空 = 列表）
    """
    from agent.permission import get_default_checker, list_extra_allowed_roots
    checker = get_default_checker()
    whitelist = checker.list_whitelist()

    if not args.strip():
        if not whitelist:
            console.print("[yellow]命令白名单为空（破坏性命令每次都会询问）[/yellow]")
        else:
            console.print(f"[bold]已批准命令（{len(whitelist)} 条，不再询问）：[/bold]")
            for i, cmd in enumerate(whitelist):
                # 长命令截断显示，免得撑爆屏幕
                display = cmd if len(cmd) <= 80 else cmd[:77] + "..."
                console.print(f"  [{i}] {display}")
        # 前缀规则展示
        prefixes = sorted(getattr(checker, "_persistent_prefixes", set()) or set())
        if prefixes:
            console.print("\n[bold]前缀规则（同前缀命令免审批）：[/bold]")
            for i, p in enumerate(prefixes, 1):
                console.print(f"  p{i}. {p}")
        # 持久化写入根目录（审批时选"总是允许"落盘的条目）
        extra_roots = list_extra_allowed_roots()
        console.print(f"\n[bold]写入根目录白名单（{len(extra_roots)} 条，来自 /add-dir 与审批「总是允许」）：[/bold]")
        if not extra_roots:
            console.print("  [dim]（无）[/dim]")
        for i, root in enumerate(extra_roots):
            console.print(f"  [{i}] {root}")
        console.print(
            "\n用法：[cyan]/approved remove <序号或命令>[/cyan] | "
            "[cyan]/approved remove-root <序号或路径>[/cyan] | "
            "[cyan]/approved remove-prefix <p序号或前缀>[/cyan]"
        )
        return

    parts = args.split(None, 1)
    action = parts[0].lower()
    target = parts[1].strip() if len(parts) > 1 else ""

    if action == "remove-root":
        extra_roots = list_extra_allowed_roots()
        if not target:
            console.print("[yellow]用法：/approved remove-root <序号或路径>[/yellow]")
            return
        if target.isdigit():
            idx = int(target)
            if 0 <= idx < len(extra_roots):
                target = str(extra_roots[idx])
            else:
                console.print(f"[red]序号超出范围（0-{len(extra_roots) - 1}）[/red]")
                return
        from agent.settings import remove_extra_allowed_root as _remove_setting_root
        from agent.permission import remove_extra_allowed_root as _remove_runtime_root
        removed_setting = _remove_setting_root(target)
        removed_runtime = _remove_runtime_root(target)
        if removed_setting or removed_runtime:
            console.print(f"[green]已移除写入根目录白名单: {target[:80]}[/green]")
        else:
            console.print(f"[red]写入根目录白名单中未找到: {target[:80]}[/red]")
        return

    # 前缀规则移除（支持 p序号 或完整前缀字符串）
    if action == "remove-prefix":
        if not target:
            console.print("[yellow]用法：/approved remove-prefix <p序号或前缀>[/yellow]")
        else:
            prefixes = sorted(getattr(checker, "_persistent_prefixes", set()) or set())
            if target.startswith("p") and target[1:].isdigit():
                idx = int(target[1:]) - 1
                if 0 <= idx < len(prefixes):
                    target = prefixes[idx]
            if target in getattr(checker, "_persistent_prefixes", set()):
                checker._persistent_prefixes.discard(target)
                checker._save_whitelist()
                console.print(f"[green]已移除前缀规则：{target}[/green]")
            else:
                console.print(f"[yellow]未找到前缀规则：{target}[/yellow]")
        return

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
        console.print(f"[yellow]用法：/approved remove <序号或命令> | /approved remove-root <序号或路径> | /approved remove-prefix <p序号或前缀>[/yellow]")


def _switch_model(rt: RuntimeContext, args: str):
    """/model——查看可用模型或切换当前模型。

    用法：
    /model          列出所有模型
    /model <name>   切换到指定模型

    参数：
        rt: RuntimeContext
        args: 空 或 模型名
    """
    from agent.settings import list_models, set_default_model, get_current_model_config

    models = list_models()
    if not models:
        console.print("[yellow]未配置任何模型。在 settings.json 的 llm 段或 models 段添加。[/yellow]")
        return

    current = get_current_model_config().get("name")

    if not args.strip():
        # 无参：列出所有模型
        console.print("[bold]可用模型：[/bold]")
        for name, cfg in models.items():
            mark = "[green]*[/green]" if name == current else " "
            fmt = cfg.get("format", "anthropic")
            model_id = cfg.get("model", "?")
            has_key = "✓" if (cfg.get("api_key") or cfg.get("auth_token")) else "[red]无 key[/red]"
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

    # 先持久化切换（写进配置）
    if not set_default_model(target):
        console.print(f"[red]切换失败[/red]")
        return

    # 再给 agent 重建 LLM 连接（用新模型的配置）
    try:
        from agent.llm_client import create_llm_client
        model_cfg = get_current_model_config()
        if not (model_cfg.get("api_key") or model_cfg.get("auth_token")):
            console.print(f"[red]{target} 未配置 api_key 或 auth_token[/red]")
            return
        rt.agent.llm_client = create_llm_client(model_cfg)
        rt.agent.model = model_cfg.get("model", target)
        rt.agent.model_format = model_cfg.get("format", "openai")
        console.print(f"[green][已切换到 {target}（{model_cfg.get('model')}）][/green]")

        # === Hooks: CONFIG_CHANGE（配置变更事件——这里是模型切换）===
        if getattr(rt, "hooks_registry", None):
            try:
                rt.hooks_registry.run_config_change({
                    "session_id": rt.session_id or "",
                    "changed_keys": ["default_model"],
                })
            except Exception as e:
                logger.warning("CONFIG_CHANGE hook 触发异常: %s", e)
    except Exception as e:
        console.print(f"[red]重建 client 失败: {e}[/red]")
        logger.exception("切换模型失败")












def _render_statusline(rt, agent) -> str:
    """拼一行状态摘要，每轮回答后打在屏幕上。

    段的顺序：模型 │ 本会话 token 用量 │ goal 状态 │ 项目名
    返回值约定：非空字符串 → 主循环打印它；空串 → 什么都不打。
    任何异常都吞掉返回空串（状态行绝不能把主流程搞挂）。

    参数：
        rt: RuntimeContext
        agent: AIAgent 实例
    返回：状态行字符串（可能为空）。
    """
    try:
        # 1. 先查开关（默认开启）
        cfg = (getattr(rt, "config", None) or {})
        sl_cfg = cfg.get("statusline", {}) if isinstance(cfg, dict) else {}
        if not sl_cfg.get("enabled", True):
            return ""

        segs = []

        # 2. 模型段（agent.model 是实例字段，前缀 ⚡ 只是视觉锚点）
        model = getattr(agent, "model", "") or ""
        if model:
            segs.append(f"⚡{model}")

        # 3. token 段：优先用实时累加的用量统计（含缓存部分）；
        #    没有就退回 session_total_tokens 老字段（兼容 mock/旧实例）。
        usage_stats = getattr(agent, "_llm_usage_stats", None)
        if usage_stats and isinstance(usage_stats, dict):
            tokens = (
                int(usage_stats.get("total_prompt_tokens", 0) or 0)
                + int(usage_stats.get("total_completion_tokens", 0) or 0)
            )
        else:
            tokens = int(getattr(agent, "session_total_tokens", 0) or 0)
        segs.append(f"会话 {_format_tokens(tokens)} tok")

        # 4. goal 段：进行中/暂停/完成/失败才显示；已取消的视为废弃不显示
        goal = getattr(agent, "_goal_state", None)
        if goal is not None:
            gstatus = getattr(goal, "status", "") or ""
            if gstatus == "active":
                iter_cnt = getattr(goal, "iteration_count", 0) or 0
                segs.append(f"goal:进行中#{iter_cnt}")
            elif gstatus == "paused":
                segs.append("goal:已暂停")
            elif gstatus == "completed":
                segs.append("goal:已完成")
            elif gstatus == "failed":
                segs.append("goal:失败")
            # cancelled / 未知状态 → 不显示

        # 5. 项目段：取项目分区键的最后一段（就是项目名）
        proj_key = getattr(rt, "_statusline_project_key", "") or ""
        if proj_key:
            tail = proj_key.rsplit("-", 1)[-1]
            if tail:
                segs.append(f"项目:{tail}")

        return " │ ".join(segs)
    except Exception:
        return ""


def _is_path_item(item: str) -> bool:
    """判断审批回调收到的是"路径"还是"命令"。

    把命令误判成路径会拿到错误的"写入路径"审批语义，判定须保守。

    规则：带空格的多词组合一律按命令处理（del /s /q tmp、rm -rf build/）；
    只有 ~ 开头 / Windows 盘符（C:\\ 或 C:/）/ 单个词里含分隔符
    （src/lib、/tmp）才算路径。

    参数：
        item: 审批回调收到的字符串
    返回：True=是路径；False=是命令。
    """
    s = (item or "").strip()
    if not s:
        return False
    # permission.check_path 的既有约定：路径审批的内容恒带"文件写入审批: "
    # 前缀（permission.py:1740）——直接认这个标记，不再瞎猜
    if s.startswith("文件写入审批"):
        return True
    if s.startswith("~"):
        return True
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        return True  # Windows 盘符（如 C:\ 或 D:/）
    if " " in s:
        return False  # 带空格 = 命令形态（含参数/路径参数的都不是"路径审批"）
    # 单个词：含分隔符就是路径（命令不可能是"单个词且带 / 或 \"的形态）
    return "/" in s or "\\" in s


def _should_exit_on_interrupt_sentinel(
    last_interrupt_ts: float, now: float, window: float = 1.0,
) -> bool:
    """判断输入线程收到的 Ctrl+C 信号该不该退出程序。

    同一次按键可能同时被主线程（跑模型时中断本轮，记下时间戳）和输入线程
    （塞一个中断信号）收到：1 秒窗口内刚发生过"回合内中断"就视为重复
    消费（本意是中断本轮，不是退出）→ False；空闲等输入时的 Ctrl+C
    保持退出语义 → True。

    参数：
        last_interrupt_ts: 最近一次回合内中断的时间戳（monotonic 时钟）
        now: 当前时间戳
        window: 判重窗口秒数（默认 1.0）
    返回：True=应该退出；False=视为重复消费，忽略。
    """
    return (now - last_interrupt_ts) > window


def run_interactive(resume_last: bool = False, cli_agents: dict = None):
    """启动交互式命令行聊天（一问一答的常驻循环）。

    参数：
        resume_last: True 时自动恢复最近一次会话（命令行 -c/--continue 触发）；
                     False 时提示用户自己选
        cli_agents: `--agents '{json}'` 传入的子代理定义，
                    优先级介于用户级和项目级之间
    """
    # 把 CLI 传入的子代理定义注入发现链
    if cli_agents:
        from agent.agent_defs import inject_cli_agents
        n = inject_cli_agents(cli_agents)
        console.print(f"[dim]已从 CLI --agents 注入 {n} 个子代理[/dim]")

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

    # 输入线程 + 队列——模型干活时用户敲的字先排队，不打断当前
    # 回答；等工具批结束后由 agent 的排队输入回流机制以"临时消息"消化
    import queue as _queue_mod
    import threading as _threading_mod
    _input_q = _queue_mod.Queue()
    _input_stop = _threading_mod.Event()
    # 用"对象"当信号而不是字符串——防止用户真的输入 "__EOF__"
    # 这几个字导致误退出（对象地址唯一，用户敲不出来）
    _EOF_SENTINEL = object()
    _INTERRUPT_SENTINEL = object()
    # 最近一次"回合内中断"的时间戳——主线程的 Ctrl+C
    # 处理器记录；输入线程消费中断信号时用它去重同一次按键
    _last_ctrl_c = 0.0

    def _input_reader():
        """守护线程：不停读键盘输入塞进队列（读到文件末尾/出错就收工）。

        参数：无（闭包捕获队列等）。
        """
        while not _input_stop.is_set():
            try:
                line = console.input("[bold cyan]你:[/bold cyan] ")
                _input_q.put(line.strip())
            except EOFError:
                _input_q.put(_EOF_SENTINEL)
                return
            except KeyboardInterrupt:
                # 保持老语义：等输入时按 Ctrl+C = 退出程序
                _input_q.put(_INTERRUPT_SENTINEL)
                return
            except Exception:
                return
    _input_thread = _threading_mod.Thread(target=_input_reader, daemon=True)
    _input_thread.start()
    # 把队列交给 agent（排队输入的回流通道）
    try:
        rt.agent.set_input_queue(_input_q)
    except Exception:
        pass

    # === idle wake（后台唤醒）：空闲时后台任务/异步子代理完成 → 自动激活主循环 ===
    # 机制：生产端（bg 盯梢线程 / 委托线程）把完成通知入队后敲回调，回调把
    # 哨兵塞进输入队列（与 EOF 哨兵同款：对象地址唯一，用户敲不出来）。
    # 自愈性：回合还在跑时哨兵会被排队输入回流丢弃——但此时通知本来就会
    # 被当轮消费，无需唤醒；多余的哨兵由主循环预检跳过（不烧 LLM）。
    _BG_WAKE_SENTINEL = object()
    _BG_WAKE_MESSAGE = (
        "[后台唤醒] 后台任务已完成，请处理本轮注入的 "
        "task_notification/delegation_completion 通知：继续未完成的工作，"
        "或向用户汇报结果。"
    )

    def _on_bg_wake():
        """后台完成的唤醒回调（在盯梢/委托线程里执行，必须便宜、非阻塞）。"""
        try:
            if not ((rt.config.get("bg_task") or {}).get("idle_wake", True)):
                return  # 配置关了：恢复"等用户下次发消息"的老行为
        except Exception:
            pass  # 配置读不出来也照常唤醒（宁可多一次预检，不丢通知）
        _input_q.put(_BG_WAKE_SENTINEL)

    # 注册到两个生产端：bg 任务管理器 + 主 agent 的委托结果信箱
    for _producer in (
        getattr(rt.agent, "bg_manager", None),
        getattr(rt, "bg_manager", None),
        getattr(rt.agent, "_delegation_queue", None),
    ):
        try:
            if _producer is not None and hasattr(_producer, "set_wake_callback"):
                _producer.set_wake_callback(_on_bg_wake)
        except Exception:
            pass

    # 主循环：读输入 → 处理 → 调 agent → 显示，循环往复
    while True:
        # 优先消费模型运行期间排队的 slash 命令（agent 的排队
        # 机制会把它们分流到 _queued_cli_commands；对话一结束就在这里
        # 按正常命令执行掉）
        _deferred = getattr(rt.agent, "_queued_cli_commands", None)
        if _deferred:
            user_input = _deferred.pop(0)
        else:
            user_input = _input_q.get()
        if user_input is _EOF_SENTINEL:
            console.print("\n再见！")
            break
        if user_input is _INTERRUPT_SENTINEL:
            # 同一次 Ctrl+C 可能同时被输入线程（本信号）和
            # 主线程（回合内中断，已记 _last_ctrl_c）两边消费——窗口内到达的
            # 信号是重复消费，吞掉不退出；空闲等输入时的 Ctrl+C 保持退出语义
            if _should_exit_on_interrupt_sentinel(_last_ctrl_c, time.monotonic()):
                console.print("\n再见！")
                break
            continue

        # idle wake（后台唤醒）：后台任务/异步子代理完成时塞进来的哨兵。
        # 预检没货（通知已被正在跑的回合消费掉了）就静默跳过——防哨兵
        # 风暴空烧 LLM。这不是用户输入：跳过粘贴转存/输入历史/slash/
        # 技能等所有用户输入分支，直接走对话执行。
        if user_input is _BG_WAKE_SENTINEL:
            if not rt.agent.has_pending_wake_payload():
                continue
            console.print("[dim][后台任务完成，自动继续][/dim]")
            # 唤醒消息按普通 user 消息入会话库（审计可见、恢复后上下文连贯）
            if rt.session_store and rt.session_id:
                rt.session_store.append_message(
                    rt.session_id, "user", _BG_WAKE_MESSAGE,
                )
            try:
                console.print("[bold green]AI:[/bold green]")
                response = asyncio.run(rt.agent.run_conversation(_BG_WAKE_MESSAGE))
                # 显示逻辑与普通消息分支一致（流式已实时显示，兜底文案补打）
                if not getattr(rt.agent, "_stream_callback", None):
                    console.print(response)
                elif response and response.startswith(
                    ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数",
                     "[模型只产出了思考过程", "[LLM 返回了空响应")
                ):
                    console.print(f"[yellow]{response}[/yellow]")
                if rt.session_store and rt.session_id:
                    rt.session_store.append_message(
                        rt.session_id, "assistant", response,
                    )
                try:
                    _sl = _render_statusline(rt, rt.agent)
                    if _sl:
                        console.print(f"[dim]{_sl}[/dim]")
                except Exception as _e:
                    logger.debug("statusline 渲染失败（不阻塞）: %s", _e)
            except KeyboardInterrupt:
                rt.agent.interrupt()
                try:
                    from tools.delegate_tool import cancel_all_subagents
                    cancel_all_subagents("用户中断（后台唤醒轮）")
                except Exception:
                    pass
                _last_ctrl_c = time.monotonic()  # 给中断信号去重当锚点
                console.print("[yellow]\n[已中断][/yellow]")
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")
                logger.exception("agent 运行错误（后台唤醒轮）")
            continue

        if not user_input:
            continue

        # 大段粘贴内容转存外部文件 + 留占位符（会话库里只存
        # 占位符省空间，发送时再展开）
        from agent.input_history import store_paste_if_large
        user_input, _pasted_to = store_paste_if_large(user_input, rt.home)

        # 记入全局输入历史（跨会话可召回，宽松失败）。
        # 必须先转外存再记历史——历史里存的是占位符而不是大原文
        #（顺序反了的话 history.jsonl 会存进未替换的大原文，越滚越大）
        try:
            from agent.input_history import GlobalHistory
            GlobalHistory(rt.home).append(user_input)
        except Exception:
            pass

        # 0. `#` 开头 = 快捷写记忆
        if user_input.startswith("#"):
            text = user_input[1:].strip()
            if text:
                _quick_save_memory(rt, text)
            else:
                console.print("[yellow]用法：# <记忆内容>（如 # 项目用 pytest）[/yellow]")
            continue

        # 1. 处理 slash 命令（/ 开头的指令）
        if user_input.startswith("/"):
            # 先看是不是技能束/技能命令（命令名统一按 split()[0]
            # 解析，与下面技能束/技能触发用同一套规则）
            cmd_name = user_input.split()[0]
            if cmd_name in rt.bundle_commands:
                pass  # 是技能束 → 跳过分发，走下面的技能束触发逻辑
            elif cmd_name in rt.skill_commands:
                pass  # 是技能 → 跳过分发，走下面的技能触发逻辑
            elif _handle_command(user_input, rt):
                if rt.quit_requested:
                    break  # /quit 请求：跳出循环走正常关机（rt.shutdown()）
                continue
            else:
                # 不认识的 slash 命令——形如命令名（/word）时报错
                # 不发给模型（否则 /sesion 这类敲错的命令会整条静默
                # 发给 LLM）。路径形态（如 "/etc/passwd 是什么"）不拦，
                # 正常当消息发送。
                _name = cmd_name[1:]
                _is_cmd_like = (
                    _name
                    and _name[0].isalpha()
                    and all(c.isalnum() or c in "_-" for c in _name)
                )
                if _is_cmd_like:
                    console.print(
                        f"[yellow]未知命令 {cmd_name}（/help 查看命令列表；"
                        "要作为消息发送请调整开头写法）[/yellow]"
                    )
                    continue

        # 2. 检查是否触发技能束（命令名解析规则与步骤 1 相同）
        cmd_name = user_input.split()[0]
        if cmd_name in rt.bundle_commands:
            bundle_info = rt.bundle_commands[cmd_name]
            rest_msg = user_input[len(cmd_name):].strip()
            user_input = execute_bundle(
                bundle_info["name"],
                rest_msg or "(执行此技能束中的所有技能)",
                skills_dir(),
            )
            console.print(f"[dim][已触发技能束: {bundle_info['name']}（{len(bundle_info['skills'])} 个技能）][/dim]")
        # 3. 检查是否触发技能
        elif cmd_name in rt.skill_commands:
            skill_info = rt.skill_commands[cmd_name]
            rest_msg = user_input[len(cmd_name):].strip()
            # 声明了 context:fork 的技能放到隔离子代理里跑
            if skill_info.get("context") == "fork" and getattr(rt, "agent", None) is not None:
                from pathlib import Path as _P
                from agent.skill_commands import parse_frontmatter as _pf
                from agent.skill_fork import run_skill_in_fork
                _raw = _P(skill_info["skill_md_path"]).read_text(encoding="utf-8")
                _, _body_only = _pf(_raw)
                _fork_result = run_skill_in_fork(
                    skill_name=skill_info["name"],
                    skill_body=_body_only,
                    user_query=rest_msg or "(执行此技能)",
                    agent_ref=rt.agent,
                )
                user_input = (
                    f"[技能 {skill_info['name']} 在隔离子代理执行完毕]\n\n"
                    f"{_fork_result}"
                )
            else:
                user_input = execute_skill(
                    skill_info["skill_md_path"],
                    rest_msg or "(执行此技能)",
                )
            # 记一次技能使用（用于统计和推荐）
            bump_use(skills_dir(), skill_info["name"])
            console.print(f"[dim][已触发技能: {skill_info['name']}][/dim]")

        # 3. 保存用户消息到 session
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "user", user_input)

            # 第一条消息后自动给会话起标题（方便 /sessions 列表辨认）
            session_info = rt.session_store.get_session(rt.session_id)
            maybe_set_title(
                rt.session_store, rt.session_id,
                user_input,
                session_info.get("title") if session_info else None,
            )

        # 4. 调用 agent（把消息交给 AI，拿回答）
        try:
            # 发给 agent 前展开粘贴引用（会话库里存的是占位符）
            from agent.input_history import expand_paste_references
            agent_input = expand_paste_references(user_input, rt.home)
            # Checkpoint：每条用户消息发出前拍快照（/rewind 可回滚）
            if rt.checkpoint_mgr:
                try:
                    rt.checkpoint_mgr.create_snapshot(
                        conversation=list(rt.agent.conversation_history),
                    )
                except Exception as e:
                    logger.warning("checkpoint 快照失败: %s", e)
            # 先打 "AI:" 前缀，流式输出会接在这个前缀后面显示
            console.print("[bold green]AI:[/bold green]")
            # run_conversation 是 async，但 run_interactive 保持同步签名
            # （run_skill_in_fork 等下游依赖同步上下文），所以每轮用
            # asyncio.run 驱动一次完整的异步对话。
            response = asyncio.run(rt.agent.run_conversation(agent_input))
            # 流式模式（设了流式回调）下内容在对话过程中已经实时显示过，
            # 不重复打印。非流式模式才打印 response。
            # 但 LLM 失败/预算耗尽这类兜底文案不走流式（没有内容增量），
            # 必须主动打印——否则用户会看到"没反应就断了"。
            if not getattr(rt.agent, "_stream_callback", None):
                console.print(response)
            elif response and response.startswith(
                ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数",
                 "[模型只产出了思考过程", "[LLM 返回了空响应")
            ):
                # 流式模式下的兜底消息（空响应/纯思考/预算耗尽/LLM 失败）
                # 不经过流式回调，必须主动打印——不然用户看到"AI:"后面一片空白
                console.print(f"[yellow]{response}[/yellow]")

            # 5. 保存助手响应到 session
            if rt.session_store and rt.session_id:
                rt.session_store.append_message(
                    rt.session_id, "assistant", response,
                )

            # 每轮回答完打印状态行（模型/token/goal/项目）
            # 放在回答完整输出之后、不接 Live 动态刷新（Windows 上跟 input()
            # 有冲突）；中断/异常路径都不打（用户主动断开就别再追加信息了）。
            try:
                _sl = _render_statusline(rt, rt.agent)
                if _sl:
                    console.print(f"[dim]{_sl}[/dim]")
            except Exception as _e:
                logger.debug("statusline 渲染失败（不阻塞）: %s", _e)
        except KeyboardInterrupt:
            rt.agent.interrupt()
            # 批量/异步子代理跑在线程池里收不到 Ctrl+C 信号，
            # 必须在这里显式按下它们的取消旗，否则进程退出被吊死
            try:
                from tools.delegate_tool import cancel_all_subagents
                cancel_all_subagents("用户中断")
            except Exception:
                pass
            _last_ctrl_c = time.monotonic()  # 记下时间戳，给信号去重当锚点
            console.print("[yellow]\n[已中断][/yellow]")
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            logger.exception("agent 运行错误")

    # === 退出前清理后台任务 ===
    # 再按一轮所有子代理的取消旗（中断分支已按过；正常退出路径在这里兜底）
    try:
        from tools.delegate_tool import cancel_all_subagents
        if cancel_all_subagents("退出清理") > 0:
            console.print("[dim]正在停止后台子代理…[/dim]")
    except Exception:
        pass
    rt.shutdown()


# ---------------------------------------------------------------------------
# 主入口函数
# ---------------------------------------------------------------------------
# 设计决策：参数解析 + 分发逻辑放在 cli.main，
# main.py 只负责 stdout 编码 + MCP 初始化 + 调 cli.main，职责干净。
#
# 为什么不在 cli.main 外面再套一层 asyncio.run：
#   run_interactive 保持同步签名，内部用 asyncio.run 驱动异步的
#   run_conversation（避免破坏 run_skill_in_fork 等下游同步调用链）。
#   如果 cli.main 再套一层 asyncio.run，会和内部的 asyncio.run
#   嵌套报错："asyncio.run() cannot be called from a running event loop"
#   （事件循环已在跑时不能再开新的）。所以 asyncio.run 只出现在
#   run_interactive 内部（紧贴异步调用点），cli.main 本身只是个
#   同步分发器。

def main(argv: list = None) -> None:
    """CLI 主入口（main.py 只负责调用它）。

    参数：
        argv: 命令行参数列表（None 时用 sys.argv，便于测试时传假参数）

    支持的调用形式：
        python main.py                         # 交互模式
        python main.py -c / --continue         # 自动恢复最近会话
        python main.py --agents '{json}'       # CLI 注入子代理
    """
    if argv is None:
        argv = sys.argv
    args = argv[1:]
    cli_agents_raw = None

    # 先提取 --agents 参数（拿掉它，不破坏旧的 -c/--continue 逻辑）
    if "--agents" in args:
        idx = args.index("--agents")
        if idx + 1 >= len(args):
            print("--agents 需要一个 JSON 参数", file=sys.stderr)
            sys.exit(2)
        try:
            import json
            cli_agents_raw = json.loads(args[idx + 1])
        except json.JSONDecodeError as e:
            print(f"--agents 参数不是合法 JSON: {e}", file=sys.stderr)
            sys.exit(2)
        # 从 args 里删掉 --agents 及其值，让旧逻辑照常工作
        args = args[:idx] + args[idx + 2:]

    # 交互模式：检查 -c / --continue 标志（有就自动恢复最近会话）
    resume_last = "-c" in args or "--continue" in args
    run_interactive(resume_last=resume_last, cli_agents=cli_agents_raw)
