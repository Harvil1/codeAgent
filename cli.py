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
import concurrent.futures
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
from agent.wake_budget import WakeBudget
from config import load_config
from constants import get_codeagent_home, skills_dir, sessions_db_path, all_skills_dirs, APP_VERSION
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

from cli_commands import slash_command
from cli_ui import console

logger = logging.getLogger(__name__)


from cli_session_cmds import (  # noqa: F401（回导入：测试/内部引用兼容）
    _list_sessions,
    _resume_and_cleanup_empty,
    _resume_session_interactive,
    _search_sessions,
    _auto_resume_last,
    _handle_resume_command,
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
import cli_skin  # noqa: F401（import 即登记 /skin 命令 + 激活皮肤引擎）
import cli_events  # 模块级 _execute_turn 要用（顶层只 import 标准库，无循环风险）
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
from cli_plugin_cmds import _handle_plugin_command  # noqa: F401（回导入同上）


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
            f"[dim]请编辑 [bold]{get_codeagent_home() / 'settings.json'}[/bold] "
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
        memory_dir: 记忆数据目录（~/.codeAgent/.memory）
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
                    store=store,
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

class RuntimeContext:
    """把 agent 运行时要用的所有零件装到一个筐里，随身携带。

    为什么需要它：组件多（配置、记忆、会话、AI 本体、定时器……），
    分散在各处容易漏传、顺序错乱。统一在 __init__ / initialize 里
    按依赖顺序装配，之后到处只传这一个对象。
    """

    def __init__(self):
        """只做"轻量启动"：读配置 + 建不依赖别人的组件。重的活留给 initialize()。"""
        self.config = load_config()
        self.home = get_codeagent_home()
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
        self._shutdown_done = False  # 关机只跑一次的幂等标志（atexit + 显式调用双通道）
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
            # 注册表目录：任务表/通知队列纯内存，落一份盘进程重启后
            # 恢复注入才知道"上次还有哪些后台任务没跑完/结果没送达"
            registry_dir=Path(self.home) / ".task_outputs",
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
                    codeagent_home=Path(self.home),
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
                    codeagent_home=self.home,
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
                    # 延迟获取——aux router 挂在 agent 身上（项目约定：
                    # rt 上不放 aux 字段），审批发生时 agent 早已建好
                    aux_provider=lambda: getattr(
                        getattr(self, "agent", None), "aux_llm_router", None),
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
            logger.warning("权限检查器初始化失败（用默认）: %s", e)

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
                codeagent_home=self.home,
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
        # 任务面板的会话过滤域（任务库全局共享，面板只认本会话的）
        try:
            import cli_live
            cli_live.set_task_scope(self.session_id)
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)

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
            from constants import get_codeagent_home
            from agent.memory_curator import should_run_now_memory
            memory_dir = get_codeagent_home() / ".memory"
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
            logger.warning("Memory Curator 触发检查失败(不阻塞): %s", e)

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
                logger.warning("子代理持久化清理失败（不阻塞）: %s", e)

        # === 落盘产物清理：tool-results（mtime）+ scratchpad ===
        # scratchpad 的 cleanup_old_scratchpads 此前定义了但从未被调用
        # （死接线，文档承诺的 7 天清理实际没跑），这里一并接上
        try:
            from agent.output_offload import cleanup_old_tool_outputs
            _to_n = cleanup_old_tool_outputs(
                self.home,
                retention_days=self.config.get("context", {}).get(
                    "tool_output_retention_days", 14,
                ),
            )
            from agent.scratchpad import cleanup_old_scratchpads
            _sp_n = cleanup_old_scratchpads(self.home)
            if _to_n or _sp_n:
                logger.info(
                    "落盘清理：tool-results=%d, scratchpad=%d", _to_n, _sp_n,
                )
        except Exception as e:
            logger.warning("落盘产物清理失败（不阻塞）: %s", e)

        # === statusline 项目分区键（赋值一次，取不到就空着）===
        # 放在 initialize 末尾（所有依赖就绪后），失败不影响主流程
        try:
            from agent.project_scope import get_project_memory_key
            self._statusline_project_key = get_project_memory_key()
        except Exception as e:
            logger.warning("statusline 项目键获取失败（不阻塞）: %s", e)
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
            import datetime as _dt
            self.hooks_registry.run_session_start({
                "session_id": self.session_id or "",
                # 契约字段补齐（hooks.py 文档承诺的 payload 形状）：
                # 声明式 hook 的模板/脚本要引用这些字段
                "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "agent_home": str(self.home),
                "message_count": 0,
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
            import datetime as _dt
            self.hooks_registry.run_session_end({
                "session_id": self.session_id or "",
                # 契约字段补齐（同上）
                "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "reason": "shutdown",
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
            for _host in ("deepseek", "openai", "anthropic", "openrouter",
                          "bigmodel", "zhipu"):
                if _host in base_url:
                    candidates.append(f"{_host.upper()}_API_KEY")
            candidates.extend(
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                 "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"]
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
                # router 的降级兜底也会从后台线程（分类器/curator/进度
                # 播报）里调——这些消费线程都已走 loop_host.run_async，
                # 实际跑在同一个常驻宿主循环上，所以兜底直接用持久
                # client（一次构造、连接池绑宿主循环），不再需要旧的
                # 「每次请求现造现关」补丁
                from agent.llm_client import create_llm_client
                main_client = create_llm_client({
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
                    aux_config=aux_cfg if not endpoints else None,
                    endpoints=endpoints,
                    # owns_main=True：上面这个 main_client 是专为 router 现造的
                    # 专用 client（agent 主对话用的是 AIAgent 自建的另一套），
                    # 所有权明确归 router——shutdown 时由 router.close() 一并关
                    owns_main=True,
                )
            except Exception as e:
                logger.warning("AuxLLMRouter 创建失败，辅助任务用主模型: %s", e)
                aux_llm_router = None
                # 刚造的持久 client 还没发过请求（连接池要到第一次调用
                # 才真正分配连接），直接丢弃不泄漏，不用 close
                main_client = None

        # === 流式输出回调 ===
        # config["streaming"]["enabled"] 默认 False（不用打字机：回答整段
        # 出——claude code 交互节奏：等待期 spinner 转着，回答一次性落屏）
        streaming_cfg = self.config.get("streaming", {})
        stream_callback = None
        if streaming_cfg.get("enabled", False):
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
            logger.warning("PermissionChecker provider 注入失败（闸门 4 将跳过）: %s", e)

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
            codeagent_home=self.home,
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

        跟 _create_agent 的主推导链是同一套逻辑（curator 兜底 client 等
        「镜像构造」复用它，保证拿到的凭证跟主连接完全一致）。

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
            for _host in ("deepseek", "openai", "anthropic", "openrouter",
                          "bigmodel", "zhipu"):
                if _host in base_url:
                    candidates.append(f"{_host.upper()}_API_KEY")
            candidates.extend(
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                 "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY"]
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
                    # 标记 _llm 是不是本函数现造的 fallback client：
                    # 只有现造的才需要（也才允许）跑完就关——aux router
                    # 是全局共享的常驻组件，关了别人就没法用了
                    _is_fallback = not (
                        _aux is not None and _aux.is_aux_configured
                    )
                    if not _is_fallback:
                        _llm = _aux
                    else:
                        # 没配辅助模型就按主模型参数现造一个持久 client：
                        # curator 的 LLM 调用都经 loop_host.run_async 跑在
                        # 常驻宿主循环上（连接池绑得稳），一次构造、整轮
                        # 复用即可
                        from agent.llm_client import create_llm_client
                        _mc = (self.config or {}).get("model", {}) or {}
                        _llm = create_llm_client({
                            "format": _mc.get("format", "openai"),
                            "base_url": _mc.get("base_url"),
                            "model": _mc.get("name"),
                            "api_key": RuntimeContext._derive_api_key(_mc),
                            "auth_token": _mc.get("auth_token") or "",
                        })
                    try:
                        run_curator_review(
                            skills_dir(),
                            session_store=self.session_store,
                            memory_store=self.memory_store,
                            llm=_llm,
                        )
                    finally:
                        # 用完就关：这个 fallback client 是本函数现造的（无
                        # aux router 时才有），跑完审查留在宿主循环上占着
                        # 连接池——旧 ThreadedLLMClient 是每调用即关，退役
                        # 后这里补上（fail-open：关失败只记日志，不影响主流程）
                        if _is_fallback:
                            try:
                                from agent.loop_host import loop_host
                                from agent.llm_client import aclose_llm_client
                                loop_host.run_async(
                                    aclose_llm_client(_llm)
                                )
                            except Exception as e:
                                logger.warning(
                                    "curator fallback client 关闭失败"
                                    "（fail-open）: %s", e
                                )
                    # 审查顺利跑完才报完毕（中途抛异常会被下面的
                    # except 接住跳过这行，不会谎报"完毕"）
                    console.print("[dim]整理技能库完毕[/dim]")
            except Exception as e:
                logger.warning("curator 触发失败: %s", e)

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
        # 任务面板过滤域跟随新会话
        try:
            import cli_live
            cli_live.set_task_scope(self.session_id)
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
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
            logger.warning("压缩状态重置失败（忽略）: %s", e)
        a._interrupt_requested = False
        a._auto_extract_cursor = 0
        a._auto_extract_turn_count = 0
        a._context_tip_shown = False
        a._pending_ephemeral_messages = []
        a._pending_tool_batch_summary = None
        a._queued_cli_commands = []
        a._surfaced_memory_ids = set()
        # STOP hook 续命计数按会话算（"本会话最多续命几次"）——agent 侧和
        # registry 侧两份都要清，否则上个会话耗掉的额度让新会话静默哑火
        a._stop_fire_count = 0
        if self.hooks_registry is not None:
            self.hooks_registry.reset_stop_budget()
        # max_tokens 升级账本也按会话翻页——不清的话上局升过级，
        # 新会话第一次截断会因 has_escalated 跳过"调大上限重发"
        try:
            if getattr(a, "_max_tokens_escalator", None) is not None:
                a._max_tokens_escalator.reset()
        except Exception as e:
            logger.warning("max_tokens 升级器重置失败（忽略）: %s", e)
        try:
            from agent.memory_injection import reset_injection_cache
            reset_injection_cache()
        except Exception as e:
            logger.warning("记忆注入缓存重置失败（忽略）: %s", e)
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
        # 重放用全量档案（用户规矩：看到什么恢复什么）——必须在
        # 压缩边界裁剪**之前**留一份，AI 的上下文裁剪归裁剪
        raw_for_replay = list(conv)
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
        # resume 预热：先切好会话 id（预热真触发 L4 时压缩事务标记要落
        # 进「恢复的这个会话」而不是刚建的空会话），再把「盖回
        # _timestamp + 无损压缩前移」的预热结果装回对话历史
        self.session_id = session_id
        self.agent.session_id = session_id
        # 任务面板过滤域切到恢复的会话（待续清单也按它取）
        try:
            import cli_live
            cli_live.set_task_scope(session_id)
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        self.agent.conversation_history = _resume_warmup(self, conv)
        self.agent.invalidate_system_prompt()
        # STOP hook 续命计数按会话算——恢复的是"别的会话"，额度重新起算
        #（agent 侧 + registry 侧两份都清，与 new_session 同款）
        self.agent._stop_fire_count = 0
        if self.hooks_registry is not None:
            self.hooks_registry.reset_stop_budget()
        # max_tokens 升级账本同款翻页（与 new_session 一致）
        try:
            if getattr(self.agent, "_max_tokens_escalator", None) is not None:
                self.agent._max_tokens_escalator.reset()
        except Exception as e:
            logger.warning("max_tokens 升级器重置失败（忽略）: %s", e)
        # 长任务进度外存回读（ephemeral，第一次对话组装时消费）
        _inject_progress_recovery(self, session_id)
        # 老任务认领：session_id 特性之前建的任务没有归属字段，直接按
        # 会话过滤会把它们全藏掉（用户看到的就是"任务列表没了"）。
        # 本会话档案里出现过的 task_/bg_ 编号就是本会话的——回填归属
        _adopt_legacy_tasks(self, session_id, msgs)
        # 待办任务清单恢复：面板重拉 + 模型侧注入（续接执行不靠自觉）
        _resume_todo_lines = _inject_task_recovery(self)
        # 后台任务 + 可续跑子代理：上次会话没跑完/没送达的，注入让模型知道
        _inject_bg_recovery(self, session_id)

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
        # 概览行：分类报数——你几条、AI 几条、工具过程几条，一眼确认
        # 上下文全在（以前只回放最后几条 AI 文字，30 条会话看着像
        # 3 条，用户以为「好多没恢复」；工具消息是 AI 记忆的一部分，
        # 数出来才安心）
        n_user = sum(1 for m in msgs if m.get("role") == "user")
        n_ai = sum(1 for m in msgs if m.get("role") == "assistant")
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")
        bits = [f"你 {n_user}", f"AI {n_ai}"]
        if n_tool:
            bits.append(f"工具 {n_tool}")
        console.print(
            f"[green][已恢复会话: {title}（{len(msgs)} 条消息："
            f"{' · '.join(bits)}）][/green]"
        )

        # 完整重放对话流长相（看到什么恢复什么）：● 头行/⎿ 结果块/
        # 子代理行都按当时的格式重画。重放失败不拦恢复本身（fail-open）
        try:
            from cli_events import replay_session_transcript
            replay_session_transcript(raw_for_replay)
        except Exception as e:
            logger.warning("会话重放失败（恢复本身不受影响）: %s", e)

        # 重放之后补一块「当前待办」：任务存在全局 task_store 里跨会话
        # 存活，但 live 面板黑板是纯内存——不重拉的话恢复后面板空空，
        # 用户以为任务丢了。还有未完成任务才打（全做完了不打空块）
        try:
            if _resume_todo_lines:
                from cli_events import print_style_lines
                print_style_lines(
                    [("dim", f"  待续任务 {len(_resume_todo_lines)} 条：")]
                    + [("dim", f"  {ln}") for ln in _resume_todo_lines])
        except Exception as e:
            logger.warning("resume 待办回显失败（fail-open）: %s", e)

        return True

    def shutdown(self):
        """关机清理：触发会话结束钩子 + 停后台任务、定时器、团队，关连接。

        后台任务和定时器不关会变"僵尸进程"；Windows 上会话库连接不关会
        锁住文件。

        参数：无。返回：无（每一步都容错，单步失败不影响其余清理）。
        """
        # 幂等闸：shutdown 走两条路（run_interactive 收尾显式调 + atexit 兜底），
        # 不拦的话 SESSION_END 钩子正常退出会跑两遍（通知/审计类 hook 双份副作用）
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        # === Hooks: SESSION_END（在资源清理前触发，此时会话上下文还在）===
        self._fire_session_end()

        # 把技能使用统计从内存缓存刷到磁盘
        try:
            from tools.skill_usage import flush_usage
            flush_usage()
        except Exception:
            logger.warning("flush 技能使用统计失败", exc_info=True)

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

        # === 关 aux router 的 endpoint/兜底 client ===
        # router 挂在 agent 身上（aux_llm_router），不在 rt 上（rt 上的
        # aux 字段是没人读的死字段，项目约定不加）；close 是 async 协程，
        # 必须趁宿主循环还活着、经 loop_host.run_async 跑——所以这段
        # 必须在 loop_host.stop() 之前。
        _aux_router = getattr(getattr(self, "agent", None), "aux_llm_router", None)
        if _aux_router is not None:
            try:
                from agent.loop_host import loop_host
                loop_host.run_async(_aux_router.close())
            except Exception as e:
                logger.warning("aux router 关闭失败（fail-open）: %s", e)

        # 最后停常驻事件循环宿主（client 已关、生产已停；daemon 属性
        # 保证异常路径也不吊死进程）
        try:
            from agent.loop_host import loop_host
            loop_host.stop()
        except Exception as e:
            logger.warning("loop_host 停机失败（daemon 兜底）: %s", e)


# ---------------------------------------------------------------------------
# 回调（把"CLI 怎么跟用户互动"做成函数，传给 agent 内部调用）
# ---------------------------------------------------------------------------

def _make_approval_callback(aux_provider=None):
    """造一个"问用户批不批准"的回调（危险命令执行前、白名单外写文件前都会用到）。

    供权限检查器使用（它不懂怎么跟用户对话）。回调收到一个字符串，
    自动判断是命令还是路径，显示不同的提问。

    批准的效果范围（如实说明，别夸大）：
    - 命令 → 同意后记入 ~/.codeAgent/approved_commands.json，
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
            # 防死锁保险：审批发生在事件循环线程内时不能同步等（自己等自己）
            _aio.get_running_loop()
            console.print("[dim]（解释器不可用——事件循环线程内无法等待 LLM）[/dim]")
            return
        except RuntimeError:
            pass  # 普通工作线程，可以安全阻塞等
        try:
            # chat_completions 是 async 且连接池绑常驻宿主循环——
            # asyncio.run 另起炉灶会踩跨循环错误，统一走 loop_host
            from agent.loop_host import loop_host
            resp = loop_host.run_async(aux.chat_completions([{"role": "user", "content": (
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

    def _ask_via_selector(question, options, header="确认"):
        """用选择器面板做审批（跟 ask_user 同款交互，↑↓ 选 + Enter 确认）。

        文本 y/N 提示在滚动区里根本不显眼（用户看着像"没让我确认就执行
        了"）——面板独占终端、高亮选中行，想看不见都难。选择器起不来时
        降级回文本提示（无头/终端不支持的老路）。
        """
        try:
            from cli_question import run_selector
            from cli_ui import run_with_input_bridge
            res = run_with_input_bridge(
                lambda: run_selector(question, header, options, False)
            ) or {}
            if res.get("cancelled"):
                return None  # 取消 = 拒绝
            picked = (res.get("answers") or [])
            return picked[0] if picked else None
        except Exception:
            return None  # 选择器不可用 → 走文本降级

    def callback(item: str):
        # 审批入口：命令/路径的判定用 _is_path_item
        #（check_path 发来的内容恒带"文件写入审批: "前缀，按这个约定识别）
        # 所有审批统一用同款三选面板：允许（本次）/ 总是允许并记住 / 拒绝
        if _is_path_item(item):
            console.print(f"[yellow]⚠️ 即将写入路径(白名单外)：[/yellow]")
            console.print(f"[bold]{item}[/bold]")
        else:
            console.print(f"[yellow]⚠️ 即将执行破坏性命令：[/yellow]")
            _shown = item if len(item) <= 200 else (
                item[:200] + f"\n…（共 {len(item)} 字符，已截断）")
            console.print(f"[bold]{_shown}[/bold]")

        # 统一选项（命令审批多一个"解释"选项，有辅助模型才显示）
        opts = [
            {"label": "允许（本次）", "description": "仅这一次"},
            {"label": "总是允许并记住", "description":
                "路径→父目录进白名单 / 命令→加入审批白名单"},
            {"label": "拒绝", "description": "不执行"},
        ]
        if not _is_path_item(item) and aux_provider:
            opts.insert(2, {
                "label": "先解释这条命令",
                "description": "让辅助模型说明用途和风险",
            })

        _header = "路径审批" if _is_path_item(item) else "命令审批"
        _q = (f"写入 {item}，允许吗？" if _is_path_item(item)
              else "允许执行以上命令吗？")
        _picked = _ask_via_selector(_q, opts, header=_header)

        if _picked == "先解释这条命令":
            _explain(item)
            # 解释完再问一遍（同款面板）
            _picked = _ask_via_selector(
                "现在允许执行吗？",
                [
                    {"label": "允许（本次）", "description": "仅这一次"},
                    {"label": "总是允许并记住", "description": "下次同命令不再问"},
                    {"label": "拒绝", "description": "不执行"},
                ],
                header=_header,
            )

        if _picked == "总是允许并记住":
            return "always"
        return _picked == "允许（本次）"
    return callback


def _make_ask_user_bridge():
    """造 ask_user 工具的 CLI 桥接：AI 问一批选择题 → 逐题放 CC 同款面板。

    界面跟 claude code 的 AskUserQuestion 同款：顶部问题标签行显示
    进度，每题选项 + 行内自填（Type something.光标落上直接打字）+
    多选 Submit 行 + Chat about this（退出问卷用自己的话聊）。

    bridge(qdata) 把整批 questions 交给 cli_question.ask_via_selector
    （经 cli_ui 的 input 桥独占终端），返回
    {"answers": [...], "chat": str|None, "cancelled": bool}；
    答完打一条汇总回显。
    """
    def bridge(qdata):
        questions = qdata.get("questions") or []

        from cli_question import ask_via_selector
        from cli_ui import run_with_input_bridge

        def _do_ask():
            return ask_via_selector(
                questions,
                fallback_input=lambda prompt="": console.input(prompt),
            )

        result = run_with_input_bridge(_do_ask) or {}
        answers = result.get("answers") or []
        chat = result.get("chat")
        # 汇总回显：短标题 → 答案，一行一问
        # 取消不打「已答完」汇总（模型收到的是 user_interrupt，别骗眼睛）
        if not result.get("cancelled"):
            _headers = [q.get("header") or "" for q in questions]
            cli_events.print_style_lines(
                cli_events.format_ask_user_echo_batch(
                    answers, chat, headers=_headers))
        return result
    return bridge


def _make_cli_stream_callback():
    """造 CLI 的流式输出回调：回答画进圆角框、思考画暗色框（hermes 同款）。

    大白话：模型吐字 → StreamBoxRenderer 接进 ╭─╮ 框逐行摆好；
    切去调工具先关框再打准备行；done 事件关框清板。框的配色跟
    当前皮肤走。回调本身只做转发——渲染状态机在 cli_stream 里，
    可以脱离终端单测。

    参数：无。返回：回调函数 cb(event)。
    """
    from cli_stream import StreamBoxRenderer

    renderer = StreamBoxRenderer()

    def cb(event: dict) -> None:
        """流事件转发（渲染器内部吞一切异常，绝不反咬流）。"""
        renderer.on_event(event)
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


# ---------------------------------------------------------------------------
# 注册命令（交互配置类）——装饰器在 import 时自登记进 cli_commands 注册表。
# 会话/诊断/技能类命令的注册壳分别在 cli_session_cmds / cli_diag_cmds /
# cli_skill_memory_cmds 里，这里只留交互/配置类。转调的实现函数（如
# _switch_model）定义在本文件下方，调用时才解析名字，先后顺序无关。
# ---------------------------------------------------------------------------

@slash_command(name="/quit", category="交互配置", usage="/quit",
               summary="退出（/exit 等效；走正常关机流程）", aliases=["/exit"])
def cmd_quit(args: str, rt) -> bool:
    # 故意不 raise SystemExit——那会跳过关机清理和 SESSION_END 钩子，
    # 直接把程序掐死。改成设标志，主循环看到后 break 走正常退出。
    rt.quit_requested = True
    return True


@slash_command(name="/help", category="交互配置", usage="/help",
               summary="显示本帮助")
def cmd_help(args: str, rt) -> bool:
    import cli_commands as cc
    console.print(cc.help_renderable())
    console.print("[dim]提示：技能命令（/技能名）也会出现在 Tab 补全里[/dim]")
    return True


@slash_command(name="/model", category="交互配置", usage="/model [name]",
               summary="切换模型")
def cmd_model(args: str, rt) -> bool:
    _switch_model(rt, args)
    return True


@slash_command(name="/plan", category="交互配置", usage="/plan [off]",
               summary="进入/退出计划模式")
def cmd_plan(args: str, rt) -> bool:
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


@slash_command(name="/goal", category="交互配置",
               usage="/goal <obj>|status|pause|resume|clear|tasks",
               summary="目标驱动多轮",
               arg_completer=lambda text: [
                   "status", "pause", "resume", "continue", "clear", "tasks",
               ])
def cmd_goal(args: str, rt) -> bool:
    return _handle_goal_command(args, rt)


@slash_command(name="/poor", category="交互配置", usage="/poor [on|off|status]",
               summary="穷鬼模式（一键关烧钱功能）",
               arg_completer=lambda text: ["on", "off", "status"])
def cmd_poor(args: str, rt) -> bool:
    return _handle_poor_command(args, rt)


@slash_command(name="/output-style", category="交互配置",
               usage="/output-style [风格名|off]",
               summary="列出/切换输出风格")
def cmd_output_style(args: str, rt) -> bool:
    return _handle_output_style_command(args, rt)


@slash_command(name="/compact", category="交互配置", usage="/compact [--yes]",
               summary="手动压缩上下文（L4 摘要；--yes 跳过确认）")
def cmd_compact(args: str, rt) -> bool:
    return _handle_compact_cli(args, rt)


@slash_command(name="/init", category="交互配置", usage="/init [--force]",
               summary="生成当前项目的 CODEAGENT.md")
def cmd_init(args: str, rt) -> bool:
    return _handle_init_command(rt, args)


@slash_command(name="/approved", category="交互配置", usage="/approved",
               summary="管理审批白名单（无参数列出，可删除条目）")
def cmd_approved(args: str, rt) -> bool:
    _manage_whitelist(rt, args)
    return True


@slash_command(name="/add-dir", category="交互配置", usage="/add-dir [路径]",
               summary="追加 safe_path 写白名单（运行时生效 + 持久化）")
def cmd_add_dir(args: str, rt) -> bool:
    return _handle_add_dir_cli(args, rt)


def _handle_command(cmd: str, rt: RuntimeContext) -> bool:
    """slash 命令总分发：查注册表，查到转 handler，查不到返回 False。

    参数：
        cmd: 用户敲的整条命令（如 "/model opus"）
        rt: RuntimeContext

    返回：bool——True 表示命令已被处理；False 表示注册表里没有
        （调用方走未知命令/技能束/技能命令分支）。
    """
    import cli_commands as cc
    result = cc.dispatch(cmd, rt)
    return result is not False and result is not None


def _is_orphan_compact_start(m: dict) -> bool:
    """判断一条消息是不是孤立的 [COMPACT_START] 事务标记（整条 user 消息就是 START 开头）。

    恢复载入时这类消息要剔除：它是「压缩开了头但没跑完」的事务痕迹，
    当正文发给模型只有干扰。集中放一个小函数，裁剪的两个出口共用。
    """
    return (
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m.get("content").startswith("[COMPACT_START]")
    )


def _truncate_at_last_compact_boundary(msgs: list) -> list:
    """恢复会话时，从最后一条"[COMPACT_BOUNDARY]"（压缩边界标记）起截断。

    标记之前的旧消息已被总结进摘要，载入只会撑上下文，全部裁掉；
    标记行本身剥掉，摘要正文保留。孤立落库的 [COMPACT_START]（压缩
    被中断的事务痕迹）一并剔除；找不到边界标记就全量载入（剔完 START）。

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

    # 压缩事务悬挂检测：有 COMPACT_START 但其后没有更晚的边界标记
    # → 上次压缩被中断（崩溃/断电）的证据。无需修复（按最后有效边界
    # 裁剪、完全无边界才全量载入，本就是保守正确行为），但必须可检测。
    _last_start = -1
    for i in range(len(msgs) - 1, -1, -1):
        content = msgs[i].get("content", "")
        if isinstance(content, str) and content.startswith("[COMPACT_START]"):
            _last_start = i
            break
    if _last_start > last_idx:
        # 两种场景共用一条告警：有更早 boundary（仍按它裁剪）或完全无
        # boundary（全量载入），所以措辞说"按最后有效边界（无则全量）"。
        logger.warning(
            "检测到未完成的上下文压缩（有 COMPACT_START 无其后的 "
            "COMPACT_BOUNDARY）——上次压缩可能被中断；本次按最后有效"
            "边界保守载入（无边界则全量）"
        )

    # 孤立的 [COMPACT_START]（压缩被中断的事务痕迹）不该作为一条
    # user 消息发给模型——场上的 START 要么在边界之前（成功压缩的
    # 正常形态，被下面的边界切片自然带走），要么就是上面的悬挂证据。
    # ⚠️ 两个顺序坑：
    # 1) 剔除必须放在上面两轮扫描之后——先滤再扫会让悬挂检测永远
    #    看不到 START，等于把告警弄死；
    # 2) 不能先滤掉 START 再按 last_idx 切——last_idx 是按原始列表
    #    算的下标，边界前的 START 被滤掉后边界左移一位，
    #    msgs[last_idx:] 会从摘要后面那条切起，等于把整份摘要弄丢。
    # 所以切片始终用原始 msgs，START 只在两个出口的返回值里剔
    # （判断逻辑集中在 _is_orphan_compact_start，两处共用）。
    if last_idx < 0:
        # 完全无边界：全量载入，孤立 START 依然不能带给模型
        return [m for m in msgs if not _is_orphan_compact_start(m)]
    kept = [dict(m) for m in msgs[last_idx:]]
    # 剥掉首条的标记行（摘要正文保留）
    first = kept[0]
    content = first.get("content", "")
    first["content"] = content.replace("[COMPACT_BOUNDARY]\n", "", 1)
    # 边界之后混进来的孤立 START（悬挂证据）在这里剔除
    kept = [m for m in kept if not _is_orphan_compact_start(m)]
    logger.info(
        "resume：按 compact 边界裁剪（丢弃 %d 条 pre-compact 消息）",
        last_idx,
    )
    return kept


def _resume_warmup(rt, conv: list) -> list:
    """恢复预热：盖回 _timestamp（时间清理层复活）+ 跑一遍压缩无损层。

    大白话：会话从盘上读回来时，内存消息没有 _timestamp（时间清理层
    全哑——「距最后一条 assistant 超 60 分钟就清旧工具结果」没有时间
    依据）、也没跑过压缩（大工具结果都是原文）——首轮对话要么慢、
    要么中途才压。这里在用户看到提示符之前把这些活先干了（fail-open，
    预热失败照常进 REPL，首轮会自然补上）。

    timestamp/pinned 两个键必须剥掉（审查裁定）：store 透传的是非下划线
    键，strip_internal_fields 是黑名单制（只认 _timestamp/_ephemeral）
    剥不掉它们，带进 LLM 请求体在严格的 OpenAI 兼容端可能 400。pin
    保护不受影响——靠 [pinned] 文本前缀的机制仍在，字典标志只做
    store 往返（压缩层的 pinned 判定两者都认）。

    参数：
        rt：RuntimeContext（取 config/agent/home/session_id/session_store）
        conv：恢复裁剪后的对话消息（不含 system）
    返回：预热后的消息列表；timestamp 已消费成 _timestamp、双键已剥、
        无损压缩（落盘/时间清理/折叠）已跑。任何异常 fail-open 返回
        原列表（键可能已剥、时间戳可能已盖——都是无害的半成品）。
    """
    try:
        from datetime import datetime as _dt
        for m in conv:
            # 先消费/剥离两个 store 往返键（无论开关开不开都剥——协议
            # 安全不属于「预热可关」的范畴）
            ts = m.pop("timestamp", None)
            m.pop("pinned", None)
            if ts and not m.get("_timestamp"):
                try:
                    m["_timestamp"] = _dt.fromisoformat(str(ts)).timestamp()
                except (ValueError, TypeError):
                    pass  # 解析失败跳过该条（fail-open，时间清理层少一条依据）

        cfg = (rt.config or {}).get("context", {})
        if not cfg.get("resume_warmup_enabled", True):
            return conv
        from agent.context_pipeline import compress_if_needed, CompressionSessionState
        from agent.loop_host import loop_host
        # compress_if_needed 按「system 在场」的完整消息列表工作（内部
        # _split_system 摘出 system 保护、_reassemble 放回头部）——这里
        # 垫一条空 system 占位，跑完再用防御版剥头（与 agent 主循环压缩
        # 入口的消息形状一致）。llm_client 用主 agent 的：真到 L4 阈值就
        # 现场做摘要（比首轮对话中途压更好）；session_state 也用 agent
        # 的记账簿（预热的触发次数计入冷却/熔断，不白拿额度）
        msgs = [{"role": "system", "content": ""}] + conv
        # 压缩前原貌留一份（浅拷贝）：真触发 L4 时给记忆提取当「遗照」——
        # 管线就地改 dict，直接存别名会被污染成占位
        pre_msgs = [dict(m) for m in msgs]
        msgs, _changed, _compacted = loop_host.run_async(compress_if_needed(
            msgs,
            llm_client=getattr(rt.agent, "llm_client", None),
            model=getattr(rt.agent, "model", None),
            config=cfg,
            session_state=getattr(rt.agent, "_compress_session_state", None)
            or CompressionSessionState(),
            agent_home=rt.home,
            session_id=rt.session_id or "",
            session_store=rt.session_store,
        ))
        # L4 真压缩触发时，主循环 compacted 分支的三件「外围仪式」在这里
        # 补齐（形状照抄 agent/__init__.py::_run_context_compression）：
        # 1) on_pre_compress：趁被摘要掉的旧消息还在（用压缩前 pre_msgs），
        #    先让记忆管理器捞一把稳定事实——不然这段对话的长期信息随摘要
        #    蒸发，再也找不回来；
        # 2) 清 _surfaced_memory_ids：已注入记忆的跨轮去重集合不再挡着，
        #    压缩后恰恰最需要它们补上下文，重新放行注入；
        # 3) 往 _pending_ephemeral_messages 塞一条「醒来简报」：恢复后的
        #    第一次对话组装时模型看到「你刚被压缩过」，不会一脸懵（主循环
        #    是当场 append 进工作列表——这里在用户开口之前，只能走临时
        #    队列，首轮被消费、不进正式历史）。三件各自 fail-open：外围
        #    失败不撤压缩（压缩本身已经省下的 token 不能吐回去）。
        if _compacted:
            try:
                mm = getattr(rt.agent, "memory_manager", None)
                if mm is not None:
                    mm.on_pre_compress(None, pre_msgs)
            except Exception as e:
                logger.warning("warmup on_pre_compress 失败（fail-open）: %s", e)
            try:
                getattr(rt.agent, "_surfaced_memory_ids", set()).clear()
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)
            try:
                rt.agent._pending_ephemeral_messages.append({
                    "role": "user",
                    "content": (
                        "<post_compress_brief>\n你刚经历了上下文压缩（恢复会话"
                        "预热触发），更早的历史已被总结。身份和 system prompt "
                        "不变，继续正常对话。\n</post_compress_brief>"
                    ),
                    "_ephemeral": True,
                })
            except Exception as e:
                logger.warning("warmup 简报注入失败（fail-open）: %s", e)
        from agent.ephemeral_inject import drop_leading_system
        return [m for m in drop_leading_system(msgs) if not m.get("_ephemeral")]
    except Exception as e:
        logger.warning("resume 预热失败（fail-open 直接进 REPL）: %s", e)
        return conv


def _inject_progress_recovery(rt, session_id: str) -> None:
    """resume 后把 PROGRESS.md 进度外存注入 ephemeral 队列（fail-open）。

    进度外存此前只在压缩醒来时被回读；跨进程 /resume（隔天继续长任务）
    时主动补一次，恢复的模型立刻知道干到哪了。消息走 _pending_ephemeral
    通道：恢复后的第一次对话组装时被模型看到，不进历史不碰缓存。
    """
    try:
        agent = getattr(rt, "agent", None)
        queue = getattr(agent, "_pending_ephemeral_messages", None)
        if queue is None:
            return
        from agent.scratchpad import read_progress_file, scratchpad_dir
        ptext = read_progress_file(session_id, rt.home)
        if not ptext:
            return
        progress_path = scratchpad_dir(session_id, rt.home) / "PROGRESS.md"
        queue.append({
            "role": "user",
            "content": (
                "<progress_recovery>\n"
                "（以下是你在本会话较早阶段写入的任务进度外存，"
                "原样回读帮你恢复上下文）\n"
                f"{ptext[:20000]}\n"
                f"（该文件路径：{progress_path}。任务有新进展时请追加更新，"
                "一行一条、最新在前）\n"
                "</progress_recovery>"
            ),
            "_ephemeral": True,
        })
        logger.info("resume: 已注入 PROGRESS.md 进度外存（%d 字符）", len(ptext))
    except Exception as e:
        logger.warning("resume PROGRESS.md 回读失败（fail-open）: %s", e)


def _adopt_legacy_tasks(rt, session_id: str, archive_msgs: list) -> None:
    """把无归属的老任务/老 bg 条目认领给当前恢复的会话（fail-open）。

    背景：task/bg 的 session_id 归属是后来加的——在那之前建的任务没有
    这个字段，恢复时按会话过滤会把它们全部藏掉（用户视角：任务列表
    消失了）。认领依据是**会话档案**：恢复的历史消息里出现过的
    task_xxx / bg_xxx 编号就是这个会话建的（别会话的档案里不会有），
    给这些条目回填 session_id 后，面板/注入/续接全部恢复可见——
    别会话的老任务仍然不认领，不串台。
    """
    if not session_id or not archive_msgs:
        return
    try:
        import re as _re
        blob = "\n".join(
            str(m.get("content") or "") for m in archive_msgs
            if isinstance(m, dict)
        )
        task_ids = set(_re.findall(r"task_[0-9a-f]{12}", blob))
        bg_ids = set(_re.findall(r"bg_[0-9a-f]{8}", blob))
        if not task_ids and not bg_ids:
            return
        # 1) 任务库回填：只动"没有归属字段"的老任务（有归属的别覆盖）
        adopted = 0
        from agent.task_store import get_task_store
        store = get_task_store(rt.home)
        for t in store.list_all():
            if t.get("id") in task_ids and not t.get("session_id"):
                try:
                    store.update(t["id"], session_id=session_id)
                    adopted += 1
                except Exception:
                    pass
                    logger.warning("异常被吞(fail-open)", exc_info=True)
        # 2) bg 注册表回填：同理只补空归属
        if bg_ids:
            try:
                from agent.background import BackgroundManager
                from agent.atomic_io import atomic_write_text_lite
                reg_dir = Path(rt.home) / ".task_outputs"
                entries = BackgroundManager.load_registry_entries(reg_dir)
                changed = False
                for e in entries:
                    if e.get("task_id") in bg_ids and not e.get("session_id"):
                        e["session_id"] = session_id
                        changed = True
                if changed:
                    atomic_write_text_lite(
                        reg_dir / "bg_registry.json",
                        json.dumps(entries, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)
        if adopted:
            logger.info(
                "resume: 已认领 %d 条无归属老任务归属本会话", adopted,
            )
    except Exception as e:
        logger.warning("老任务认领失败（fail-open）: %s", e)


def _inject_task_recovery(rt) -> list:
    """resume 后把未完成任务清单注入 ephemeral 队列 + 重拉面板（fail-open）。

    为什么要注入：任务存在全局 task_store 里跨会话存活，但恢复的模型
    上下文里**没有**这份清单——续接执行全靠模型自觉调 task_list 碰运气。
    这里主动把 pending/in_progress/blocked/triage 的任务列出来（完成
    的不列），恢复后的第一次对话组装时模型就能看到「哪些没做完、从哪
    续接」，不丢任务上下文。

    返回：未完成任务行列表（回显块复用同一份数据；全完成返回空表）。
    """
    # 面板黑板重拉（用户可见的一半）：任务工具事件没来之前黑板是空的
    try:
        import cli_live
        cli_live.refresh_tasks()
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)
    # 模型侧注入（续接执行的一半）
    lines = []
    try:
        from agent.task_store import get_task_store
        store = get_task_store(rt.home)
        _sid = getattr(rt, "session_id", "") or ""
        rows = [
            t for t in store.list_all()
            if t.get("status") not in ("completed", "deleted")
            # 只认本会话的任务：任务库全局共享，别的会话/项目的残留
            # 任务注入进来就是"冒出第二份任务清单"（无归属的老任务
            # 说不清属于谁，同样不注入，仍可用 task_list 工具查全局）
            and t.get("session_id", "") == _sid
        ]
        for t in rows[:30]:
            mark = {"pending": "□", "in_progress": "■",
                    "blocked": "⊘", "triage": "⚕"}.get(t.get("status"), "·")
            lines.append(f"{mark} {t.get('subject', '')}（{t.get('id', '')}，{t.get('status', '')}）")
        if not rows:
            return []
        agent = getattr(rt, "agent", None)
        queue = getattr(agent, "_pending_ephemeral_messages", None)
        if queue is not None:
            queue.append({
                "role": "user",
                "content": (
                    "<task_recovery>\n"
                    "（会话恢复：以下是你之前规划但尚未全部完成的任务清单，"
                    "来自跨会话持久化的任务库。请从未完成的任务续接执行——"
                    "已完成的不要重做；做完调 task_complete，新发现用 "
                    "task_create 追加。in_progress 的任务上次可能被中断，"
                    "先核实其当前实际状态再继续。）\n"
                    + "\n".join(lines)
                    + "\n</task_recovery>"
                ),
                "_ephemeral": True,
            })
            logger.info("resume: 已注入待办任务清单（%d 条未完成）", len(rows))
    except Exception as e:
        logger.warning("resume 任务清单注入失败（fail-open）: %s", e)
    return lines


def _inject_bg_recovery(rt, session_id: str) -> None:
    """resume 后注入「后台任务遗态 + 可续跑子代理」（ephemeral，fail-open）。

    两块内容：
    1. bg 任务注册表里本会话还没跑完的条目——任务表/通知队列纯内存，
       进程一退就蒸发；不注入的话恢复的模型根本不知道曾有这些任务
       （registry 里的 running 是崩溃瞬间的遗照，进程已随程序退出终止；
       detach 的可能仍在独立跑但无人监管，监视文件可 read_file 查增量）。
    2. 上次被中断的异步子代理（subagent_persistence 里 interrupted 且
       属于本会话）——它们带已落盘的对话轨迹，可经 /resumable 或
       subagent_resume 续跑，不丢上下文。不列出来模型就只会重派从 0 跑。
    """
    parts = []
    try:
        from agent.background import BackgroundManager
        entries = BackgroundManager.load_registry_entries(
            Path(rt.home) / ".task_outputs")
        _sid = session_id or ""
        unfinished = [
            e for e in entries
            if e.get("status") in ("running", "stopping")
            and e.get("session_id", "") == _sid
        ]
        if unfinished:
            lines = []
            for e in unfinished[:5]:
                cmd = " ".join(str(c) for c in (e.get("command") or []))[:80]
                extra = ""
                if e.get("detach"):
                    extra = "；detach 任务，进程可能仍在独立运行但无人监管"
                if e.get("output_file"):
                    extra += f"；增量输出文件：{e['output_file']}"
                lines.append(f"- [{e.get('task_id')}] {cmd}{extra}")
            parts.append(
                "<bg_task_recovery>\n"
                "（会话恢复：以下后台任务在上次会话仍在运行。默认情况下"
                "程序退出后其进程已终止、结果不可再取——需要结果请重跑；"
                "带增量输出文件的先 read_file 看已产出的部分再决定。）\n"
                + "\n".join(lines)
                + "\n</bg_task_recovery>"
            )
        # 崩溃前已完成但通知大概率没送达的（本会话最近 3 条终态），
        # 带输出摘要兜底——模型的续接判断有据可依
        done_recent = [
            e for e in entries
            if e.get("status") in ("completed", "failed", "stopped")
            and e.get("session_id", "") == _sid
        ][:3]
        if done_recent:
            lines = []
            for e in done_recent:
                cmd = " ".join(str(c) for c in (e.get("command") or []))[:60]
                out = (e.get("stdout") or e.get("stderr") or "")[:200]
                lines.append(
                    f"- [{e.get('task_id')}] {cmd} → {e.get('status')}"
                    f"(exit={e.get('exit_code')}) 输出摘要：{out}")
            parts.append(
                "<bg_task_result_recovery>\n"
                "（会话恢复：以下后台任务在上次会话已结束，完成通知可能"
                "未送达。输出摘要见各行，如需完整结果可重跑该命令。）\n"
                + "\n".join(lines)
                + "\n</bg_task_result_recovery>"
            )
    except Exception as e:
        logger.warning("resume bg 任务注入失败（fail-open）: %s", e)

    # 中断的异步子代理：可续跑、带上下文
    try:
        from agent.subagent_persistence import _sessions_dir
        resumable = []
        for meta_path in _sessions_dir().glob("*.meta.json"):
            try:
                import json as _json
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                if (meta.get("status") == "interrupted"
                        and meta.get("parent_session_id", "") == (session_id or "")):
                    resumable.append(meta)
            except Exception:
                continue
        if resumable:
            lines = []
            for m in resumable[:5]:
                goal = (m.get("description") or m.get("goal") or "")[:80]
                # 进度摘要：从轨迹尾部抽最后一条 assistant 消息的前 80 字
                # + 总轮数——模型看到"已读完文件正在写报告"就不用自己去
                # cat 轨迹文件了（cat 会把子代理全部上下文灌进主代理，
                # 打穿隔离架构）
                _progress = ""
                try:
                    from agent import subagent_persistence as _sp
                    _aid = m.get("agent_id", "")
                    _tr = _sp.load_transcript(_aid)
                    _turns = len(_tr)
                    _last_asst = ""
                    for _tm in reversed(_tr):
                        if _tm.get("role") == "assistant" and _tm.get("content"):
                            _last_asst = str(_tm["content"])[:80]
                            break
                    _progress = f"（{_turns} 轮，最后动作：{_last_asst}…）"
                except Exception:
                    _progress = "（轨迹不可读）"
                lines.append(
                    f"- {m.get('agent_id')}：{goal}\n  {_progress}")
            parts.append(
                "<subagent_recovery>\n"
                "（会话恢复：以下子代理上次被中断，各自的对话轨迹已落盘"
                "——可带着已做部分续跑，不必从 0 重来。每条的进度摘要"
                "已列在下方，**不需要自己去 cat/ls 查**。恢复姿势（严格"
                "遵守）：\n"
                "1. **禁止先跑 tail/ls/cat 等任何侦查命令**——本注入已含"
                "每个子代理的状态、任务和进度，你自己查纯属浪费时间和"
                "token，还会把子代理的原始上下文灌进你的窗口；\n"
                "2. **本条回复里同时发出全部 subagent_resume 调用**——"
                "多个调用一起发才不被串行等待拖慢；一个一个发每轮要等"
                "上一个跑完才轮到下一个；\n"
                "3. 续跑子代理的旧上下文可能与磁盘不符——但这只影响"
                "**要修改文件**的场景（写文件有指纹校验兜底）。如果上次"
                "已经读完全部文件、正在写报告或输出结论，直接基于已有"
                "分析继续即可，**不要重读文件**；\n"
                "4. 不想续跑的（任务已被用户手工完成）可以跳过，在最终"
                "汇报里说明即可。）\n"
                + "\n".join(lines)
                + "\n</subagent_recovery>"
            )
    except Exception as e:
        logger.warning("resume 子代理注入失败（fail-open）: %s", e)

    if not parts:
        return
    try:
        agent = getattr(rt, "agent", None)
        queue = getattr(agent, "_pending_ephemeral_messages", None)
        if queue is None:
            return
        for p in parts:
            queue.append({"role": "user", "content": p, "_ephemeral": True})
        logger.info("resume: 已注入后台任务/子代理恢复信息（%d 块）", len(parts))
    except Exception as e:
        logger.warning("resume bg/subagent 注入投递失败（fail-open）: %s", e)


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
            # 现行占位以 [COMPACT_BOUNDARY]\n 开头（老格式前缀也留着兼容：
            # boundary 落库失败/旧会话的占位还是裸前缀形态）
            ("[COMPACT_BOUNDARY]", "[之前的对话已自动总结]", "[紧急上下文压缩")
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
        from agent.loop_host import loop_host
        # _summarize_conversation 是 async（因为底层 LLM 调用是 async）。
        # 本函数是同步的（被同步的命令处理函数调用），交常驻循环宿主跑
        # ——主 client 绑死宿主循环，不再各搭各的临时循环。
        summary = loop_host.run_async(
            _summarize_conversation(after, rt.agent.llm_client))
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


# ---------------------------------------------------------------------------
# goal / poor / output-style / trace / history / mailbox 等命令的处理函数
# ---------------------------------------------------------------------------

def _goal_state_path(rt) -> Path:
    """goal（目标驱动状态）的持久化文件路径：~/.codeAgent/.goal/current.json。

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
            # 拆解函数是异步的：交常驻循环宿主跑——斜杠命令都在 cli-worker
            # 线程、无运行循环，run_async 恒安全（旧 deprecated
            # get_event_loop() 判嵌套的补丁随迁移删除）
            from agent.loop_host import loop_host
            task_ids = loop_host.run_async(
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

    输出风格 = 一段预设提示词。目录：`<项目>/.codeAgent/output-styles/`
    （优先，覆盖同名）+ `~/.codeAgent/output-styles/`。切换时写
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
                "[dim]暂无输出风格。创建：~/.codeAgent/output-styles/<名>.md"
                " 或 .codeAgent/output-styles/<名>.md（正文即提示词）[/dim]"
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
        logger.warning("异常被吞(fail-open)", exc_info=True)


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
    """/init——给当前项目生成一份"项目说明书"CODEAGENT.md。

    流程：收集项目信息（目录树/关键文件/类型统计，缺哪跳哪）→
    主 LLM 按四段式生成（项目本质/常用命令/架构/约定）→
    写到 cwd/CODEAGENT.md（下次会话自动注入 system prompt，AI 进门就懂项目）。

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
    target = cwd / "CODEAGENT.md"
    if target.exists() and not force:
        console.print(
            "[yellow]CODEAGENT.md 已存在[/yellow] "
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
        logger.warning("收集目录结构失败（跳过）: %s", e)

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
        logger.warning("文件类型统计失败（跳过）: %s", e)

    info = "\n\n".join(parts) or "（空项目，无可用信息）"

    # ── 拼好提示词 → 主 LLM 按四段式生成 ──
    prompt = (
        "根据以下项目信息生成 CODEAGENT.md（项目指导文件，给 AI 编程助手看）。"
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

    # 生成交常驻循环宿主跑——主 client 绑死宿主循环；斜杠命令都在
    # cli-worker 线程、无运行循环，run_async 恒安全（旧的「已在事件
    # 循环」RuntimeError 兜底补丁随迁移删除）
    text = ""
    try:
        from agent.loop_host import loop_host
        text = loop_host.run_async(_gen())
    except Exception as e:
        logger.warning("init 生成失败（LLM 调用异常）: %s", e)
        text = ""

    if not text.strip():
        console.print("[red]CODEAGENT.md 生成失败（LLM 返回空）[/red]")
        return True

    try:
        target.write_text(text.strip() + "\n", encoding="utf-8")
    except Exception as e:
        console.print(f"[red]写入失败：[/red]{e}")
        return True

    console.print(
        f"[green]✓ CODEAGENT.md 已生成[/green] "
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
        logger.warning("异常被吞(fail-open)", exc_info=True)
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

    - 无参数：列出当前白名单（默认的 cwd + ~/.codeAgent，加上追加过的）
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
        from agent.llm_client import create_llm_client, aclose_llm_client
        model_cfg = get_current_model_config()
        if not (model_cfg.get("api_key") or model_cfg.get("auth_token")):
            console.print(f"[red]{target} 未配置 api_key 或 auth_token[/red]")
            return
        # 旧 client 留着最后关（先建新的再关旧的，中间零空窗）
        _old_client = rt.agent.llm_client
        _old_fb = getattr(rt.agent, "fallback_llm_client", None)
        rt.agent.llm_client = create_llm_client(model_cfg)
        rt.agent.model = model_cfg.get("model", target)
        rt.agent.model_format = model_cfg.get("format", "openai")
        # fallback client 同步换新：不换的话主 client 挂了会静默切回旧模型
        if getattr(rt.agent, "fallback_model", None):
            _fb_cfg = dict(model_cfg)
            _fb_cfg["model"] = rt.agent.fallback_model
            try:
                rt.agent.fallback_llm_client = create_llm_client(_fb_cfg)
            except Exception as e:
                logger.warning("重建 fallback client 失败（沿用旧的）: %s", e)
        # aux router 的兜底 main_client 同步换新（顺带关掉归它所有的旧池）
        _aux = getattr(rt.agent, "aux_llm_router", None)
        if _aux is not None:
            try:
                _aux.swap_main_client(create_llm_client(model_cfg))
            except Exception as e:
                logger.warning("aux router 兜底 client 换新失败（沿用旧的）: %s", e)
        # 最后关旧池：不关的话每次 /model 都漏一个绑着连接池的 client
        for _oc in (_old_client, _old_fb):
            if _oc is not None and _oc is not rt.agent.llm_client:
                try:
                    from agent.loop_host import loop_host
                    loop_host.run_async(aclose_llm_client(_oc), timeout=10)
                except Exception as e:
                    logger.warning("关闭旧模型 client 失败（忽略）: %s", e)
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












def _statusline_segments(rt, agent) -> list[str]:
    """回合末状态行的段拼装（底部工具栏是独立轻量拼装，见
    cli_layout.status_bar_segments，不走这里）。

    段的顺序：模型 │ 本会话 token 用量 │ goal 状态 │ 项目名。
    不吞异常——兜底交给调用方（_render_statusline 的 try）。

    参数：
        rt: RuntimeContext
        agent: AIAgent 实例
    返回：段字符串列表（可能为空）。
    """
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

    return segs


def _fmt_turn_duration(seconds: float) -> str:
    """秒数 → 回合耗时文案（claude code 的 ✻ 行同款节奏）。

    不足 1 分钟给秒（45s）；满 1 分钟给「1m 57s」。
    """
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def _render_statusline(rt, agent, elapsed_s=None) -> str:
    """拼一行回合收尾摘要（claude code 的 ✻ 行同款），每轮回答后打。

    形如：✻ Cooked for 1m 57s · ↓ 19.1K tok（goal 进行中再补一段）。
    动词从 cli_live 的动词表随机抽（Cooked/Brewed/Baked…——纯装饰）。
    返回值约定：非空字符串 → 主循环打印它；空串 → 什么都不打。
    任何异常都吞掉返回空串（状态行绝不能把主流程搞挂）。

    参数：
        rt: RuntimeContext
        agent: AIAgent 实例
        elapsed_s: 本回合耗时秒数（None=不显示耗时段）
    返回：状态行字符串（可能为空）。
    """
    try:
        # 1. 先查开关（默认开启）
        cfg = (getattr(rt, "config", None) or {})
        sl_cfg = cfg.get("statusline", {}) if isinstance(cfg, dict) else {}
        if not sl_cfg.get("enabled", True):
            return ""

        segs = []
        if elapsed_s is not None:
            try:
                import cli_live
                verb = cli_live.pick_finish_verb()
                dur = cli_live.fmt_elapsed(elapsed_s)
                segs.append(f"✻ {verb} for {dur}")
            except Exception:
                segs.append(f"✻ 用时 {_fmt_turn_duration(elapsed_s)}")

        # token 段：优先用实时累加的用量统计（含缓存部分）；
        # 没有就退回 session_total_tokens 老字段（兼容 mock/旧实例）。
        usage_stats = getattr(agent, "_llm_usage_stats", None)
        if usage_stats and isinstance(usage_stats, dict):
            tokens = (
                int(usage_stats.get("total_prompt_tokens", 0) or 0)
                + int(usage_stats.get("total_completion_tokens", 0) or 0)
            )
        else:
            tokens = int(getattr(agent, "session_total_tokens", 0) or 0)
        segs.append(f"会话 {_format_tokens(tokens)} tok")

        # goal 段：进行中/暂停才显示（完成/失败也提一嘴，取消的不提）
        goal = getattr(agent, "_goal_state", None)
        if goal is not None:
            gstatus = getattr(goal, "status", "") or ""
            if gstatus == "active":
                segs.append(
                    f"goal:进行中#{getattr(goal, 'iteration_count', 0) or 0}")
            elif gstatus == "paused":
                segs.append("goal:已暂停")

        return " · ".join(segs)
    except Exception:
        return ""


def _execute_turn(rt, agent_input: str) -> None:
    """跑一个完整回合的公共部分：调 agent、显示回答、落会话库、打收尾行。

    普通回合和后台唤醒回合原本是两段几乎逐行相同的代码（已经出现细微
    漂移），修 bug 极易漏一边——抽成一个函数。只抽快乐路径：中断/
    异常处理器留在调用方（两边的取消文案和日志标签不同），保证语义
    与原两段逐行等价。

    参数：
        rt：RuntimeContext（拿 agent/session_store/session_id）
        agent_input：本轮输入（已展开粘贴引用）
    返回：无。response 通过流式回调或兜底打印呈现。
    """
    from agent.loop_host import loop_host
    cli_events.reset_pending(rt)  # 新回合清 ◐ 黑板（防幻影残留）
    # 新回合清「本回合已中断」标志：上个回合的 Ctrl+C 残留会把本回合
    # 第一击误判成强退（单击退出的根因）
    try:
        import cli_layout
        cli_layout.reset_interrupt_press()
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)
    _turn_t0 = time.monotonic()
    rt.turn_active = True
    try:
        # 常驻循环宿主跑回合（替代 asyncio.run 现建现拆）：回合结束时
        # run_turn 的栅栏会取消本回合遗留 task，语义与旧关循环清场对齐
        response = loop_host.run_turn(rt.agent.run_conversation(agent_input))
    finally:
        rt.turn_active = False
    # 显示：非流式打 ● 块（markdown 照常渲染、首行贴橙色圆点）；兜底
    # 文案（中断/失败/空响应）保持黄字提醒，不走 ● 块
    if not getattr(rt.agent, "_stream_callback", None):
        if response and response.startswith(
            ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数",
             "[模型只产出了思考过程", "[LLM 返回了空响应")
        ):
            console.print(f"[yellow]{response}[/yellow]")
        else:
            cli_events.print_assistant_block(response)
    elif response and response.startswith(
        ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数",
         "[模型只产出了思考过程", "[LLM 返回了空响应")
    ):
        console.print(f"[yellow]{response}[/yellow]")
    if rt.session_store and rt.session_id:
        rt.session_store.append_message(
            rt.session_id, "assistant", response,
        )
    # 本回合任务清单动过 → 收尾在滚动历史里落一份静态快照
    # （claude code 同款：live 面板回合结束就收起，终态留档在这）
    try:
        import cli_live
        if cli_live.take_tasks_touched():
            cli_events.print_style_lines(
                cli_events.format_tasks_static_block())
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)
    # 每轮回答完打印 ✻ 收尾行（动词/用时/token/goal）。中断/异常路径不打
    # （用户主动断开就别再追加信息了）——所以本函数由调用方的
    # try/except 包着，异常根本走不到这里。
    try:
        _sl = _render_statusline(rt, rt.agent, time.monotonic() - _turn_t0)
        if _sl:
            console.print(f"[dim]{_sl}[/dim]")
    except Exception as _e:
        logger.warning("statusline 渲染失败（不阻塞）: %s", _e)


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

    try:
        rt = RuntimeContext()
        rt.initialize()
    except SystemExit:
        return
    except Exception as e:
        console.print(f"[red]初始化失败: {e}[/red]")
        logger.exception("初始化失败")
        return

    # === 启动横幅（自设计小猫咪 logo + 版本 + 模型 + 目录）===
    # 大白话：像出租车顶灯——上车先报「哪家公司、什么车型、跑哪条道」。
    # logo 是本项目自己的小猫（经典 /\_/\ 猫脸：● 眼睛 ▽ 鼻子），猫身
    # 配色跟输入提示符同款青绿，眼睛和鼻子琥珀点缀。字符全挑 GBK 可
    # 编码的——万一启动时 UTF-8 切换失败、落回 GBK 控制台也不变问号。
    try:
        from rich.text import Text as _BannerText
        _model_name = ((rt.config.get("model") or {}).get("name")
                       or (rt.config.get("model") or {}).get("model") or "?")
        _cwd = str(getattr(rt, "workspace_cwd", "") or Path.cwd())
        _cat = "bold fg:#00aa88"    # 猫身青绿（提示符同款）
        _face = "bold fg:#d97706"   # 眼睛/鼻子琥珀（点亮一点生气）
        _b = _BannerText()
        _b.append(" /\\_/\\    ", style=_cat)
        _b.append(f"CodeAgent v{APP_VERSION}\n", style="bold")
        _b.append("( ", style=_cat)
        _b.append("● ●", style=_face)
        _b.append(" )   ", style=_cat)
        _b.append(f"{_model_name} · 自学习 AI Agent\n")
        _b.append(" > ", style=_cat)
        _b.append("▽", style=_face)
        _b.append(" <    ", style=_cat)
        _b.append(_cwd, style="dim")
        console.print(_b)
        console.print()   # 横幅和第一轮对话之间留一行呼吸
    except Exception:
        pass   # 横幅挂了不挡启动

    # 启动时恢复历史会话：只有 -c/--continue 才恢复最近一个；
    # 不带参数就静默开新会话（不再询问、不展示历史清单）
    if resume_last:
        _auto_resume_last(rt)

    # 输入队列——模型干活时用户敲的字先排队，不打断当前回答；等工具批
    # 结束后由 agent 的排队输入回流机制以"临时消息"消化。
    # 生产端是操作台的 Enter 键位回调（cli_layout.submit_input），不再有
    # 独立输入线程。
    import queue as _queue_mod
    import threading as _threading_mod
    _input_q = _queue_mod.Queue()
    _input_stop = _threading_mod.Event()
    # 用"对象"当信号而不是字符串——防止用户真的输入 "__EOF__"
    # 这几个字导致误退出（对象地址唯一，用户敲不出来）
    _EOF_SENTINEL = object()

    # === 输入层装配：常驻操作台（cli_layout），唯一界面 ===
    import cli_layout

    # === Ctrl+C 回合中中断：pt 原始模式吃掉系统信号，靠 c-c 键位回调补通道 ===
    rt.turn_active = False

    def _do_turn_interrupt():
        """一击 = 全部中断（用户定的规则）。

        三板斧按序：标志位（子代理主循环下轮自查）→ 取消旗（线程池里的
        子代理任务不再继续）→ cancel_current_turn（正在烧的 LLM 调用直接
        断——不硬断的话用户要干等长响应落地，屏幕没变化只会继续按）。
        CancelledError 沿 worker 的 except 安静出回合，不刷栈。
        """
        rt.agent.interrupt()
        try:
            from tools.delegate_tool import cancel_all_subagents
            cancel_all_subagents("用户中断（Ctrl+C）")
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        try:
            from agent.loop_host import cancel_current_turn
            cancel_current_turn()
        except Exception:
            pass  # 没有在跑的回合时是安全空操作

    _interrupt_fn = cli_layout.build_interrupt_fn(
        lambda: getattr(rt, "turn_active", False),
        _do_turn_interrupt,
    )

    # === OS 级 SIGINT 兜底：不管 pt 键位有没有接到 Ctrl+C，信号层永远接住 ===
    # 根因：pt 的 c-c 键位只在 raw 模式下有效——事件循环忙渲染/接管间隙里
    # 信号穿透到 Python 默认处理器 → raise KeyboardInterrupt → asyncio
    # runner 崩 → Executor shutdown 满屏 → 程序闪退。装信号处理器后：
    #   回合进行中 → 中断（程序活着可继续对话）
    #   空闲 → EOF 哨兵（正常退出，再见+shutdown）
    #   绝不 raise KeyboardInterrupt
    # 防抖 0.5s：pt 键位和 OS 信号可能同一击都到，去抖防双处理
    import signal as _signal_mod
    _last_sigint = [0.0]

    def _os_sigint_handler(signum, frame):
        """OS 级 Ctrl+C：信号穿透 pt 键位时的兜底。"""
        import time as _sig_t
        _now = _sig_t.monotonic()
        if _now - _last_sigint[0] < 0.5:
            return
        _last_sigint[0] = _now
        if getattr(rt, "turn_active", False):
            _do_turn_interrupt()
        else:
            _input_q.put(_EOF_SENTINEL)

    try:
        _signal_mod.signal(_signal_mod.SIGINT, _os_sigint_handler)
    except (ValueError, OSError):
        pass

    def _force_exit():
        """Ctrl+C 双击：关闭所有进程和线程，立刻走人（不走优雅收尾）。

        顺序有讲究：先按中断+取消旗（让子代理/后台任务自己断）→
        硬取消当前回合（cancel_current_turn，正在烧 LLM 的回合立刻断，
        worker 线程以 CancelledError 安静出回合）→ EOF 请工作线程离场 →
        app.exit() 收界面。app.run() 返回后终端恢复正常模式，主线程的
        join(2s) 兜底才会用 os._exit——那时终端已不在 raw 模式，强杀
        不会把用户的终端搞坏。
        """
        try:
            console.print("[red]⚡ 强制退出——正在取消所有子代理和后台任务…[/red]")
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        # 面板遗照：强退不走 _execute_turn 收尾，子代理树/任务清单
        # 不落静态行就永远消失了——趁终端还在，先把快照打进滚动区
        try:
            import cli_live
            cli_live.dump_panel_snapshot()
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        rt._force_exiting = True
        # 强退窗口里还在收尾的线程（worker/子代理摘要/线程池关停）会
        # 互相踩出 RuntimeError 噪声——进程马上就没了，日志全静音
        #（只留这条红字提示，不再刷 traceback 吓人）
        logging.disable(logging.ERROR)
        _do_turn_interrupt()            # agent.interrupt + 全部取消旗
        # 常驻循环后回合挂在宿主循环上——强退前主动取消，别让它在
        # os._exit 兜底窗口里继续烧 LLM（双击强退语义保持）
        try:
            from agent.loop_host import cancel_current_turn
            cancel_current_turn()
        except Exception:
            pass  # 兜底还有 os._exit
        _input_q.put(_EOF_SENTINEL)     # 请工作线程离场
        _the_app = getattr(rt, "prompt_session", None)
        if _the_app is not None:
            try:
                _the_app.exit()         # UI 收摊（终端恢复交给 app.run 返回）
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)

    # 皮肤激活：settings.json 的 display.skin（失败回退 default，不挡启动）
    try:
        cli_skin.init_skin_from_config(rt.config)
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)

    # === 常驻操作台装配：cli_layout 的 Application（唯一界面，没有降级） ===
    _app = cli_layout.build_application(
        rt,
        completer=cli_layout.build_completer(rt),
        interrupt_fn=_interrupt_fn,
        input_queue=_input_q,
        eof_sentinel=_EOF_SENTINEL,
        force_exit_fn=_force_exit,
        history_path=get_codeagent_home() / ".input_history",
    )
    rt.prompt_session = _app
    if _app is None:
        # 老降级界面已删：操作台建不起来就是致命错误，明说后收摊
        console.print(
            "[red]界面构建失败：请确认在交互式终端"
            "（cmd / Windows Terminal / PowerShell / git-bash）里运行，"
            "且输出没有被重定向。[/red]"
        )
        rt.shutdown()
        return

    # 跨线程输入桥：审批/确认类提问（工作线程）经 pt 的 run_in_terminal
    # 执行——否则 stdin 被 pt 独占，提问挂着永远没人能答
    cli_layout.install_input_bridge(_app)

    # 把 app 登记给打印漏斗（cli_ui.emit_ansi 靠它把工作线程的打印搬进
    # UI 事件循环——不登记的话 spinner/状态栏会被冻进滚动历史）
    from cli_ui import set_active_app as _set_active_app
    _set_active_app(_app)

    # === 事件行渲染器：工具/子代理/任务 全走这里（完成行 + 工具栏黑板） ===
    cli_events.install_event_lines(rt)

    # spinner 线程接管底部条的节奏（0.1s 一拍：翻帧/计时/重绘）
    cli_layout.start_spinner_thread(rt, _app, _input_stop)
    # 把队列交给 agent（排队输入的回流通道）
    try:
        rt.agent.set_input_queue(_input_q)
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)

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

    # 唤醒预算：防"唤醒→起新任务→完成→又唤醒"的自激励循环。
    # 纯内存计数；用户真实输入即回血（见下方 reset 调用）。
    _wake_budget = WakeBudget(
        max_wakes=int(
            (rt.config.get("bg_task") or {}).get("max_consecutive_wakes", 3)
        )
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
            logger.warning("异常被吞(fail-open)", exc_info=True)

    # 老主循环（一行没改，只是搬了个家）：读输入 → 处理 → 调 agent → 显示，
    # 循环往复。跑在工作线程（cli-worker），主线程被 app.run() 占着管屏幕。
    def _worker_loop():
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
                break   # 再见与收尾由启动段统一处理（UI 退出后打）

            # idle wake（后台唤醒）：后台任务/异步子代理完成时塞进来的哨兵。
            # 预检没货（通知已被正在跑的回合消费掉了）就静默跳过——防哨兵
            # 风暴空烧 LLM。这不是用户输入：跳过粘贴转存/输入历史/slash/
            # 技能等所有用户输入分支，直接走对话执行。
            if user_input is _BG_WAKE_SENTINEL:
                if not rt.agent.has_pending_wake_payload():
                    continue
                if not _wake_budget.consume():
                    console.print(
                        "[dim][连续自动唤醒已达上限，后台通知暂存，"
                        "等用户下次输入时一并处理][/dim]"
                    )
                    continue
                console.print("[dim][后台任务完成，自动继续][/dim]")
                # 唤醒消息按普通 user 消息入会话库（审计可见、恢复后上下文连贯）
                if rt.session_store and rt.session_id:
                    rt.session_store.append_message(
                        rt.session_id, "user", _BG_WAKE_MESSAGE,
                    )
                try:
                    _execute_turn(rt, _BG_WAKE_MESSAGE)
                except KeyboardInterrupt:
                    rt.agent.interrupt()
                    try:
                        from tools.delegate_tool import cancel_all_subagents
                        cancel_all_subagents("用户中断（后台唤醒轮）")
                    except Exception:
                        pass
                        logger.warning("异常被吞(fail-open)", exc_info=True)
                    console.print("[yellow]\n[已中断][/yellow]")
                except (asyncio.CancelledError, concurrent.futures.CancelledError):
                    # 强退主动取消回合（cancel_current_turn）会走到这里——
                    # 这是用户要走的路不是崩溃，安静收场（强退横幅由 UI 打）。
                    # CancelledError 是 BaseException，不接住会打穿 worker
                    # 线程带出满屏 traceback。concurrent 版是 fut.result()
                    # 把取消搬运到 worker 线程后的实际类型，两个都接
                    pass
                except Exception as e:
                    if getattr(rt, "_force_exiting", False):
                        break   # 强退中：线程池已关停的噪声不刷屏，直接离场
                    console.print(f"[red]错误: {e}[/red]")
                    logger.exception("agent 运行错误（后台唤醒轮）")
                continue

            if not user_input:
                continue

            # 用户真实输入：唤醒预算回血（自动唤醒不许自回血，只有真人说话算数）
            _wake_budget.reset()

            # 输入回显（claude code 同款 "> 消息"）——输入框提交后就清空了，
            # 对话历史里留个影子，翻记录时才知道哪句是用户说的。
            # 用 rich Text（不解析 markup）：用户输入里带 [red] 之类不会被误上色。
            from rich.text import Text as _EchoText
            console.print(_EchoText("> " + user_input.replace("\n", "\n> ")))

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
                logger.warning("异常被吞(fail-open)", exc_info=True)

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
                        import cli_commands as _cc
                        _near = _cc.suggest(cmd_name.lower())
                        _hint = f"；你是不是想敲 {' 或 '.join(_near)}" if _near else ""
                        console.print(
                            f"[yellow]未知命令 {cmd_name}（/help 查看命令列表{_hint}；"
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
                cli_events.print_style_lines([
                    ("", f"● Skill({bundle_info['name']})"),
                    ("dim", f"  ⎿  Successfully loaded skill bundle "
                            f"({len(bundle_info['skills'])} skills)"),
                ])
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
                # 触发行走 claude code 风格：● Skill(名字) + ⎿ 加载成功
                cli_events.print_style_lines(
                    cli_events.format_skill_lines(skill_info["name"]))

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
                # 流式模式：回答框头（╭─ ⚕ CodeAgent ─╮）就是回合起始标记，
                # 不再打 "AI:" 前缀；纯工具轮/兜底文案有事件行和黄字兜底
                # run_conversation 是 async，但 run_interactive 保持同步签名
                # （run_skill_in_fork 等下游依赖同步上下文），所以每轮由
                # _execute_turn 经常驻循环宿主（loop_host.run_turn）同步
                # 驱动一次完整的异步对话。
                _execute_turn(rt, agent_input)
            except KeyboardInterrupt:
                rt.agent.interrupt()
                # 批量/异步子代理跑在线程池里收不到 Ctrl+C 信号，
                # 必须在这里显式按下它们的取消旗，否则进程退出被吊死
                try:
                    from tools.delegate_tool import cancel_all_subagents
                    cancel_all_subagents("用户中断")
                except Exception:
                    pass
                    logger.warning("异常被吞(fail-open)", exc_info=True)
                console.print("[yellow]\n[已中断][/yellow]")
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                # 强退主动取消回合（cancel_current_turn）会走到这里——
                # 这是用户要走的路不是崩溃，安静收场（强退横幅由 UI 打）。
                # CancelledError 是 BaseException，不接住会打穿 worker
                # 线程带出满屏 traceback。concurrent 版是 fut.result()
                # 把取消搬运到 worker 线程后的实际类型，两个都接
                pass
            except Exception as e:
                if getattr(rt, "_force_exiting", False):
                    break   # 强退中：线程池已关停的噪声不刷屏，直接离场
                console.print(f"[red]错误: {e}[/red]")
                logger.exception("agent 运行错误")

        # 无论从哪条路退出（EOF / /quit），都请 UI 收摊
        cli_layout.request_app_exit(_app)

    # 主线程跑 UI（app.run 阻塞），工作线程跑老主循环；
    # patch_stdout 包 app.run()——工作线程的 console.print 会被
    # 代理到 UI 线程、排在操作台上方（跟 hermes 同款）
    _worker_thread = _threading_mod.Thread(
        target=_worker_loop, daemon=True, name="cli-worker")
    _worker_thread.start()
    try:
        from prompt_toolkit.patch_stdout import patch_stdout
        _patch_ctx = patch_stdout()
    except Exception as _pe:
        logger.warning("patch_stdout 不可用（输出可能与输入框偶发交错）: %s", _pe)
        _patch_ctx = None
    try:
        if _patch_ctx is not None:
            with _patch_ctx:
                _app.run()
        else:
            _app.run()
    except (EOFError, KeyboardInterrupt, BrokenPipeError) as _ee:
        logger.info("界面退出: %s", _ee)
    finally:
        # 工作线程收到 EOF → request_app_exit → app.run 返回；
        # 这里反向兜底：UI 先退了（异常），也让工作线程尽快收工
        _set_active_app(None)   # 注销 app：之后的打印走直写（不再搬事件循环）
        # UI 退了但回合可能还在烧（Ctrl+C 从没被键位拦住的缝漏进来时，
        # app.run 是被 KeyboardInterrupt 打断的）——先按"全部中断"
        # （标志+子代理取消旗+cancel_current_turn 硬断在飞调用），worker
        # 才有机会在宽限期内从回合里出来走正常收尾。不按的话 worker
        # 卡在回合里等 LLM，join 超时 → 直接闪退（单击 Ctrl+C 整程序
        # 退出的执行点就是这）
        try:
            _do_turn_interrupt()
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)
        _input_stop.set()
        _input_q.put(_EOF_SENTINEL)
        try:
            _worker_thread.join(timeout=5.0)
        except KeyboardInterrupt:
            # 收尾窗口里又按 Ctrl+C：此刻 pt 已退、终端回到经典模式，
            # 信号以裸 KeyboardInterrupt 打进 join——不接住它会掀翻整个
            # 收尾段（跳过 os._exit 兜底，atexit/线程池/在跑的回合互相
            # 踩踏，满屏 traceback）。用户想走，就痛快放行：直接硬退。
            try:
                console.print("[dim][再次中断，立即强制退出…][/dim]")
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)
            import os as _os_mod
            _os_mod._exit(0)
        if _worker_thread.is_alive():
            # worker 没在宽限期内收工（回合卡死/子代理不退）：
            # 收尾段随时可能被用户再按 Ctrl+C 打断、daemon 线程残留
            # 还会撞 atexit 崩栈——这里强退是唯一干净的出路
            console.print("[dim][后台任务仍在收尾，强制退出…][/dim]")
            import os as _os_mod
            _os_mod._exit(0)

    # 「再见」也可能撞上收尾窗口里的 Ctrl+C——掀不出去了也犯不着炸栈
    try:
        console.print("\n再见！")
    except KeyboardInterrupt:
        import os as _os_mod
        _os_mod._exit(0)

    # === 退出前清理后台任务 ===
    # 再按一轮所有子代理的取消旗（中断分支已按过；正常退出路径在这里兜底）。
    # 收尾段必须防打断：用户狂按 Ctrl+C / 子代理死活不退时，这里任何
    # 裸异常都会变成满屏 traceback（收尾失败 ≠ 崩溃）
    try:
        from tools.delegate_tool import cancel_all_subagents
        if cancel_all_subagents("退出清理") > 0:
            console.print("[dim]正在停止后台子代理…[/dim]")
    except (Exception, KeyboardInterrupt):
        pass
    try:
        rt.shutdown()
    except (Exception, KeyboardInterrupt) as e:
        logger.warning("收尾清理异常（忽略）: %s", e)

    # === 终极自毁保底 ===
    # 收尾全走完了，但进程可能仍被卡死的 daemon 线程拖住不退（实测：
    # 子代理线程在已停机的循环上收尾、Ctrl+C 打不进去——"反复按没反应"
    # 的僵死）。8 秒后无论如何结束进程：正常路径此刻本来就要退出，多等
    # 8 秒无感；僵死路径保证必死，不会留在屏幕上装死。
    import os as _os_final
    import threading as _th_final

    def _final_exit():
        _os_final._exit(0)

    _t = _th_final.Timer(8.0, _final_exit)
    _t.daemon = True   # Timer 不收 daemon 关键字——退出码路径上别再炸
    _t.start()


# ---------------------------------------------------------------------------
# 主入口函数
# ---------------------------------------------------------------------------
# 设计决策：参数解析 + 分发逻辑放在 cli.main，
# main.py 只负责 stdout 编码 + MCP 初始化 + 调 cli.main，职责干净。
#
# 为什么不在 cli.main 外面再套一层 asyncio.run：
#   run_interactive 保持同步签名，内部经 loop_host.run_turn（常驻
#   循环宿主）同步驱动异步的 run_conversation（避免破坏
#   run_skill_in_fork 等下游同步调用链）。如果 cli.main 再套一层
#   asyncio.run，阻塞在 fut.result() 上等回合结果的其实是 cli-worker
#   线程（它等的是宿主循环线程，这本身没事）；真正的问题是主线程会被
#   同步的 run_interactive 整场占死——外层事件循环一次都转不起来，
#   纯属白搭。所以异步驱动只出现在 run_interactive 内部（紧贴异步
#   调用点），cli.main 本身只是个同步分发器。

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
