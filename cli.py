"""CLI 交互层（完整版）。

启动时初始化的组件：
  - MemoryStore（多文件 JSONL 记忆 + MEMORY.md 索引；会话内检索式 ephemeral 注入）
  - MemoryManager（编排器，可选外部 provider）
  - SessionStore（JSONL 文件会话持久化，接口兼容旧 SQLite）
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
# 启动期校验/容错（Bug #2/#3 fix）
# ---------------------------------------------------------------------------

def _validate_model_config(config: dict) -> None:
    """Bug #2 fix: 校验 config['model'] 必填字段，缺失时 SystemExit(2) + 友好提示。

    之前 cli.py:254/417/474 直接索引 config["model"]["name"]/["provider"]，
    缺字段时抛 KeyError 不友好（对比 api_key 缺失有友好提示，不一致）。

    本函数补齐：缺 name/provider 或值为空 → SystemExit(2) + 红字提示。
    完整 config 不抛。
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
    """Bug #3 fix: memory curator 主体，try/finally 保证 state 写盘。

    之前 cli.py:300-338 的 daemon 线程函数中途崩溃时 state 未保存，
    导致下次启动重复跑（浪费 LLM tokens）。提取为模块级 + try/finally，
    每次完成或失败都 save_memory_curator_state。

    X1 fix: 加 store 参数（共享主 agent 实例避免跨实例 race）。

    失败信息写入 last_run_summary，便于排查。
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
        # 第 1 阶段就崩——记录失败信息到 state
        logger.warning("Memory Curator 第 1 阶段失败: %s", e)
        review_summary = f"第 1 阶段失败: {e}"

    finally:
        # Bug #3 核心修复：无论中途是否崩，state 都写盘
        state["last_run_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        state["last_run_summary"] = review_summary
        try:
            save_memory_curator_state(memory_dir, state)
            logger.info("Memory Curator state 已保存: %s", review_summary)
        except Exception as save_err:
            # save 自己失败只能 log（不能阻塞主流程）
            logger.error("Memory Curator state 保存失败: %s", save_err)


# ---------------------------------------------------------------------------
# 运行时初始化
# ---------------------------------------------------------------------------

def _thread_llm_client(config: dict):
    """线程上下文专用 LLM client（R26 终审 follow-up）：per-call 独立连接，跨 loop 安全。
    改主 client 构造链（RuntimeContext）时须同步本镜像字段。"""
    from agent.llm_client import ThreadedLLMClient
    mc = (config or {}).get("model", {}) or {}
    # api_key 推导与 RuntimeContext 构造主 client 同源（见 RuntimeContext._derive_api_key）
    api_key = RuntimeContext._derive_api_key(mc)
    return ThreadedLLMClient({
        "format": mc.get("format", "openai"),
        "base_url": mc.get("base_url"),
        "model": mc.get("name"),
        "api_key": api_key,
        "auth_token": mc.get("auth_token") or "",
    })


class RuntimeContext:
    """聚合 agent 运行时的所有组件。"""

    def __init__(self):
        self.config = load_config()
        self.home = get_omnimate_home()
        # === CCAR11 Task 4 NEW: /add-dir 持久化白名单启动加载 ===
        # settings.json（load_config 读出的 config）的 security.extra_allowed_roots
        # → 运行时 safe_path 白名单（fail-open：单条失败跳过，不阻塞启动）
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
        self.bundle_commands = {}  # batch1-T3: 技能束斜杠命令
        self.quit_requested = False  # /quit 请求退出标志（走正常 shutdown）
        self.checkpoint_mgr = None  # Checkpoint（文件快照/回滚，对齐 Claude Code）
        # === P2-T8 NEW: Hooks 系统 ===
        from agent.hooks import HookRegistry
        self.hooks_registry = HookRegistry()

        # === P2b-T6 NEW: 后台任务管理器 ===
        from agent.background import BackgroundManager
        bg_cfg = self.config.get("bg_task", {})
        self.bg_manager = BackgroundManager(
            max_concurrent=bg_cfg.get("max_concurrent", 5),
            notification_stdout_cap=bg_cfg.get("notification_stdout_cap", 500),
            result_stdout_cap=bg_cfg.get("result_stdout_cap", 5000),
            default_timeout=bg_cfg.get("default_timeout", 600),
            stall_timeout=bg_cfg.get("stall_timeout", 45.0),  # P1-3: 停滞看门狗（45s 无 stdout 新增 → 通知）
        )

        # === P2c-T5 NEW: Cron 调度器 ===
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
                    max_age_days=cron_cfg.get("max_age_days", 7),  # === CronRecurringExpiry NEW ===
                )
                self.cron_scheduler.start()
            except Exception as e:
                logger.error("CronScheduler 启动失败: %s", e)
                self.cron_scheduler = None
        else:
            self.cron_scheduler = None

        # === P4a-T7 NEW: Agent Teams ===
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
                # 主 agent 自注册为 lead（仅当 registry 还没有 main running）
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

        # === ⑮ NEW: Handoff bundle 存储 ===
        self.handoff_store = None  # 在 initialize() 中真正初始化

        # === CCAR8 Task 12 NEW: mailbox + agent_name + trace_sink ===
        # mailbox：teammate 异步邮箱（Task 8 遗留接线）
        # agent_name：当前 agent 名（mailbox 收件人，默认 "main"）
        # trace_sink：本地 trace sink（Task 5 遗留接线，/trace 命令读这个）
        # 字段都先 None，在 initialize() 中真正填充
        # 注：aux_llm_client / aux_model 已删（CCAR8 final fix）——死字段，
        # 工具 dispatch 和 goal decompose 都走 agent_ref.aux_llm_router
        self.mailbox = None
        self.agent_name = "main"
        self.trace_sink = None
        self._poor_mode_on = False  # /poor 命令状态
        # CCAR10 Task 3: statusline 项目分区键（initialize 里赋值一次，fail-open None）
        self._statusline_project_key = None

        # X2 fix: atexit 兜底 shutdown（即使主循环异常/SystemExit 也会清理 SQLite 锁等）
        import atexit
        atexit.register(self.shutdown)

    def _make_memory_review_agent_factory(self):
        """构造 Memory Curator 第 2 阶段的后台 review agent 工厂。

        返回的 factory 调用时创建一个独立 AIAgent 实例:
        - 用主模型(opus/deepseek-v4-pro)
        - 无工具集(纯文本交互,LLM 输出 YAML 由 Python 执行)
        - 独立 messages 历史(不污染主对话)
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
                    enabled_toolsets=[],  # 不给工具,纯文本交互
                    omnimate_home=self.home,
                    config=self.config,
                )
            except Exception as e:
                logger.warning("构造 memory review agent 失败: %s", e)
                raise

        return factory

    def initialize(self):
        """初始化所有组件。"""
        # Bug #2 fix: 启动最早校验 model.name/provider（缺字段友好提示）
        _validate_model_config(self.config)

        # 0. 设置权限检查器（注入破坏性命令审批 callback + 持久化白名单）
        # B2: perm_mode 在 try 外定义，AIAgent 构造处（行 460+）也要用
        perm_mode = self.config.get("security", {}).get("permission_mode", "default")
        try:
            from agent.permission import set_default_checker, PermissionChecker
            from agent.settings import approved_commands_path, approved_paths_path
            set_default_checker(PermissionChecker(
                approval_callback=_make_approval_callback(
                    # R21 #45：审批 e 选项的 aux 解释器（延迟取——router 此后构造）
                    aux_provider=lambda: getattr(self, "aux_llm_router", None),
                ),
                whitelist_file=str(approved_commands_path()),
                paths_whitelist_file=str(approved_paths_path()),
                mode=perm_mode,
                hooks_registry=self.hooks_registry,  # round3 D2 NEW: 权限审计 hook
            ))
            # Task 7: 灌入 config["security"]["sandbox_mode"]
            from agent.permission import get_default_checker
            _checker = get_default_checker()
            if _checker is not None:
                sandbox_mode = self.config.get("security", {}).get("sandbox_mode", "off")
                if sandbox_mode in ("off", "on"):
                    _checker.set_sandbox_mode(sandbox_mode)
        except Exception as e:
            logger.debug("权限检查器初始化失败（用默认）: %s", e)

        # 0.5 加载声明式 hooks（如果启用）
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
        # 1a. 先建 SessionStore（memories/tasks 双写需要它注入）
        if self.config.get("sessions", {}).get("auto_save", True):
            db = self.config.get("sessions", {}).get("db_path")
            db_path = Path(db) if db else sessions_db_path()
            self.session_store = SessionStore(db_path)

        # 1b. MemoryStore（纯文件存储，无 SQLite 双写）
        if self.config.get("memory", {}).get("enabled", True):
            self.memory_store = MemoryStore(
                omnimate_home=self.home,
            )
            # batch2-T3: memory_manager 的 LLM client 在 agent 创建后注入
            # （因为需要和 aux_llm_router 共享）
            self.memory_manager = MemoryManager(self.memory_store)

        # 1c. TaskStore 全局单例（纯文件存储）
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

        # Checkpoint：文件快照/回滚（对齐 Claude Code，会话级）
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

        # 4. 创建 agent
        self.agent = self._create_agent()
        # R30f-H8：per-model 用量追踪（注入 agent；/usage 按 model 展示）
        try:
            from agent.usage_tracker import UsageTracker
            self.usage_tracker = UsageTracker(
                self.home, self.session_id or "default",
                default_provider=(self.config.get("model") or {}).get("provider", ""),
            )
            self.agent.set_usage_tracker(self.usage_tracker)
        except Exception as e:
            logger.warning("UsageTracker 初始化失败（fail-open）: %s", e)
            self.usage_tracker = None

        # 5. 扫描技能命令(内置 + 用户两个目录,用户优先)
        self.skill_commands = scan_skill_commands(all_skills_dirs())
        # batch1-T3: 扫描技能束命令(只在用户目录,不扫内置)
        self.bundle_commands = scan_bundle_commands(skills_dir())

        # 7. Handoff 存储
        try:
            handoff_dir = Path(self.home) / ".handoff"
            self.handoff_store = HandoffStore(handoff_dir)
        except Exception as e:
            logger.warning("HandoffStore 初始化失败: %s", e)
            self.handoff_store = None

        # === CCAR8 Task 12 NEW: 初始化 mailbox + trace_sink，注入 agent ===
        # mailbox 接线（Task 8 遗留）：用 team 目录，跟 team_bus 共享一个 mailbox 根
        try:
            from agent.team.mailbox import Mailbox
            mb_dir = Path(self.home) / ".team"
            mb_dir.mkdir(parents=True, exist_ok=True)
            self.mailbox = Mailbox(mb_dir)
            self.agent_name = "main"
            # 把 mailbox + agent_name 挂到 AIAgent（mailbox_tool 通过 agent_ref 读这两个）
            self.agent.set_mailbox(self.mailbox, self.agent_name)
            logger.info("mailbox 已注入 AIAgent（agent_name=%s）", self.agent_name)
        except Exception as e:
            logger.warning("mailbox 初始化失败（fail-open）: %s", e)
            self.mailbox = None

        # trace_sink 接线（Task 5 遗留）：从 config.trace.enabled 读开关
        trace_cfg = self.config.get("trace", {})
        if trace_cfg.get("enabled", True):
            try:
                from agent.trace import TraceSink
                self.trace_sink = TraceSink(base_dir=Path(self.home))
                # 回填到 agent（让 _register_trace_hooks 能拿到）
                if self.agent is not None:
                    self.agent._trace_sink = self.trace_sink
                    from agent.trace import _register_trace_hooks
                    if self.agent.hooks_registry is not None:
                        _register_trace_hooks(self.agent.hooks_registry, self.trace_sink)
            except Exception as e:
                logger.warning("TraceSink 初始化失败（fail-open）: %s", e)
                self.trace_sink = None

        # 6. 后台触发 curator（不阻塞启动）
        self._maybe_trigger_curator()

        # === Memory Curator 后台触发(照搬 skill curator 模式) ===
        try:
            from constants import get_omnimate_home
            from agent.memory_curator import should_run_now_memory
            memory_dir = get_omnimate_home() / ".memory"
            if memory_dir.exists() and should_run_now_memory(memory_dir, config=self.config):
                import threading

                def _run_memory_curator():
                    # Bug #3 fix: 调提取出的 _run_memory_curator_once，
                    # try/finally 保证 state 写盘（即使中途崩溃）。
                    # X1 fix: 传主 store 实例避免跨实例 race（threading.Lock 同实例生效）。
                    # factory 通过 config 字典临时透传（避免破坏函数签名）。
                    cfg_copy = dict(self.config) if isinstance(self.config, dict) else {}
                    try:
                        cfg_copy["_curator_factory"] = self._make_memory_review_agent_factory()
                    except Exception:
                        pass  # factory 创建失败也能跑（只跳过第 2 阶段）
                    try:
                        _run_memory_curator_once(
                            memory_dir, config=cfg_copy, store=self.memory_store,
                        )
                    except Exception as e:
                        logger.warning("Memory Curator 后台运行失败: %s", e)

                # CCAR9 final review Important：daemon 线程不自动继承主线程
                # contextvars（threading.Thread 在 3.12- 不 copy context）。
                # 项目分区键依赖 workspace_cwd ContextVar，不传 → fallback
                # os.getcwd()，多项目场景下他项目的 project/reference
                # 记忆永远不会被 curate。
                # 修法：在主线程里 copy_context()，target 用 ctx.run 包一层。
                _curator_ctx = contextvars.copy_context()
                threading.Thread(
                    target=lambda: _curator_ctx.run(_run_memory_curator),
                    daemon=True,
                    name="memory-curator",
                ).start()
        except Exception as e:
            logger.debug("Memory Curator 触发检查失败(不阻塞): %s", e)

        # === Task I: 清理 stale 子代理记录 + 过期 retention 清理 ===
        # 启动时把 status=running 但进程已退出的（上次崩溃残留）标记为 interrupted，
        # 并清理超过 retention_days 的已完成记录。
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

        # === CCAR10 Task 3 NEW: statusline 项目分区键（赋值一次，fail-open） ===
        # 放在 initialize 末尾（所有依赖就绪后），失败不影响主流程
        try:
            from agent.project_scope import get_project_memory_key
            self._statusline_project_key = get_project_memory_key()
        except Exception as e:
            logger.debug("statusline 项目键获取失败（不阻塞）: %s", e)
            self._statusline_project_key = ""

        # === Hooks: SESSION_START（会话已建立，声明式 hooks 已加载） ===
        self._fire_session_start()

    def _fire_session_start(self) -> None:
        """触发 SESSION_START hook（会话建立后）。"""
        if not getattr(self, "hooks_registry", None):
            return
        try:
            self.hooks_registry.run_session_start({
                "session_id": self.session_id or "",
            })
        except Exception as e:
            logger.warning("SESSION_START hook 触发异常: %s", e)

    def _fire_session_end(self) -> None:
        """触发 SESSION_END hook（会话关闭前，需在资源清理之前）。"""
        if not getattr(self, "hooks_registry", None):
            return
        try:
            self.hooks_registry.run_session_end({
                "session_id": self.session_id or "",
            })
        except Exception as e:
            logger.warning("SESSION_END hook 触发异常: %s", e)

    def _create_agent(self) -> AIAgent:
        """根据配置创建 agent。

        优先用 settings.json 里的 api_key 字段；为空时 fallback 到环境变量。
        """
        model_cfg = self.config.get("model", {})
        # api_key 和 auth_token 分别取(不混用)
        # DeepSeek Anthropic 端点用 auth_token(Bearer),用 api_key(x-api-key)会被拒
        api_key = model_cfg.get("api_key") or ""
        auth_token = model_cfg.get("auth_token") or ""

        # Fallback：JSON 里没填 key 时，尝试 provider 专属环境变量
        if not api_key and not auth_token:
            provider = (model_cfg.get("provider") or "").upper()
            api_key_env = model_cfg.get("api_key_env") or ""
            candidates = [
                api_key_env,
                f"{provider}_API_KEY" if provider else None,
            ]
            # 新模式 llm 扁平配置下 provider 是档位名（opus/haiku/sonnet），
            # <档位>_API_KEY 通常不存在；再按 base_url 域名 + 常见 provider
            # 环境变量兜底（如默认 DeepSeek 端点 → DEEPSEEK_API_KEY）
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

        # === batch2-T3 + 07: 创建 AuxLLMRouter ===
        aux_llm_router = None
        aux_cfg = self.config.get("aux_model")
        # 07 NEW: 优先读 aux_llm.endpoints 列表
        aux_llm_cfg = self.config.get("aux_llm", {})
        endpoints_cfg = aux_llm_cfg.get("endpoints", []) if aux_llm_cfg else []

        if endpoints_cfg or (aux_cfg and isinstance(aux_cfg, dict) and aux_cfg.get("model")):
            try:
                from agent.aux_llm import AuxLLMRouter, LLMEndpoint
                # R26 终审 follow-up：router 的降级 fallback 也可能从线程调
                # （分类器/curator）——用 per-call 独立 client，别建绑主循环的临时池
                from agent.llm_client import ThreadedLLMClient
                main_client = ThreadedLLMClient({
                    "format": model_cfg.get("format", "openai"),
                    "base_url": model_cfg.get("base_url"),
                    "api_key": api_key,
                    "model": model_cfg["name"],
                })
                # 07 NEW: endpoints 列表优先
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
                # R26 终审 follow-up：ThreadedLLMClient 无持久连接池可泄，无需 close
                main_client = None

        # === 04 NEW: 流式输出 callback ===
        # config["streaming"]["enabled"] 默认 True（CLI 边生成边打印）
        streaming_cfg = self.config.get("streaming", {})
        stream_callback = None
        if streaming_cfg.get("enabled", True):
            stream_callback = _make_cli_stream_callback()

        # batch2-T3: 给 memory_manager 注入 LLM client（优先用 aux）
        if self.memory_manager:
            if aux_llm_router:
                self.memory_manager._llm_client = aux_llm_router
                self.memory_manager._llm_model = (
                    aux_cfg.get("model") if aux_cfg else None
                )
            else:
                # 没配置 aux 时用主 client（延迟到 agent 创建后注入）
                pass

        # === F2 NEW: 注入 aux_llm_router provider 给 hook_exec ===
        # 声明式 hook 的 prompt/agent 类型 handler 需要通过 aux_llm 跑评估。
        # hook_exec 模块级 provider 注入点，未配置 aux 时 lambda 返回 None（fail-open）。
        from agent.hook_exec import set_aux_router_provider, set_config_provider
        set_aux_router_provider(lambda: aux_llm_router)
        # P3.2: 注入 config provider，让 dispatch_hook 能读 feature flag 门控
        # http / mcp_tool / agent 三种 handler 类型
        set_config_provider(lambda: self.config)

        # 注：aux_llm_client / aux_model 回填已删（CCAR8 final fix）——
        # 死字段，agent_ref.aux_llm_router 是单一来源。

        # === P4.1 NEW: 给 PermissionChecker 注入 aux_llm + config provider ===
        # 闸门 4（aux_llm 分类）需要这两个 provider。PermissionChecker 比 aux_llm_router
        # 先构造（行 294），所以这里回填（参考 hook_exec 同款 pattern）。
        # 未配置 aux_llm_router 时 lambda 返回 None → 闸门 4 fail-open 跳过。
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
            hooks_registry=self.hooks_registry,  # === P2-T8 NEW ===
            bg_manager=self.bg_manager,  # === P2b-T6 NEW ===
            cron_scheduler=self.cron_scheduler,  # === P2c-T5 NEW ===
            team_bus=self.team_bus,  # === P4a-T7 NEW ===
            team_coordinator=self.team_coordinator,  # === P4a-T7 NEW ===
            team_name="main",  # === P4a-T7 NEW ===
            aux_llm_router=aux_llm_router,  # === batch2-T3 NEW ===
            plan_approval_callback=cli_plan_approval_callback,  # === PlanMode NEW ===
            stream_callback=stream_callback,  # === 04 NEW: 流式输出 ===
            ask_user_bridge=_make_ask_user_bridge(),  # ask_user CLI 桥接
            checkpoint_manager=self.checkpoint_mgr,  # === Checkpoint NEW ===
            permission_mode=self.config.get("security", {}).get("permission_mode", "default"),  # === B2 NEW: 透传给 AIAgent ===
            # R17 #12：流空闲看门狗（llm.stream_idle_timeout_seconds，默认 90s，<=0 禁用）
            stream_idle_timeout=(self.config.get("llm") or {}).get("stream_idle_timeout_seconds"),
        )

        # batch2-T3: 如果 memory_manager 还没 LLM client，用 agent 的主 client
        if self.memory_manager and self.memory_manager._llm_client is None:
            self.memory_manager._llm_client = agent.llm_client
            self.memory_manager._llm_model = agent.model

        # === B1 NEW: 初始化 vision_client（image_analyze / image_ocr 共用） ===
        vision_cfg = self.config.get("vision", {}) or {}
        if vision_cfg.get("enabled", True):
            try:
                from agent.llm_client import create_llm_client
                vision_provider = vision_cfg.get("provider") or ""
                vision_model = vision_cfg.get("model") or ""
                # 如果 vision.model 为空，不创建独立 client（让工具回退到 llm_client）
                if vision_model:
                    agent._vision_client = create_llm_client({
                        "format": model_cfg.get("format", "openai"),
                        "base_url": model_cfg.get("base_url"),
                        "api_key": api_key,
                        "model": vision_model,
                    })
                    logger.info("vision_client 已初始化（model=%s）", vision_model)
                # 否则 agent._vision_client 保持 None，工具回退 llm_client
            except Exception as e:
                logger.warning("vision_client 初始化失败（用主 client 回退）: %s", e)
                agent._vision_client = None

        # === CCAR8 final fix NEW: 接线 MCP notifications → ChannelInbox ===
        # 背景：MCPTransport.set_notification_handler + StdioTransport._reader_loop
        # 都实现了，但没人把 ChannelInbox.push 注册成 handler，导致 MCP server 推
        # notification 时 handler=None 被丢弃，/inbox 永远空。这里补上接线。
        # fail-open：任何异常只 log warning，不影响 agent/MCP 功能。
        try:
            from agent.channel_inbox import ChannelInbox
            from agent.mcp_client import get_mcp_manager
            channel_inbox = ChannelInbox(base_dir=self.home)
            mgr = get_mcp_manager()
            # 遍历所有已连接的 MCP client，给底层 transport 注册 handler
            # 闭包变量捕获：server_name 用默认参数绑定（避免循环变量漂移）
            with mgr._lock:
                clients_snapshot = list(mgr._clients.items())
            for _server_name, _client in clients_snapshot:
                try:
                    _transport = getattr(_client, "_transport", None)
                    if _transport is None:
                        continue
                    # handler 收 (method, params)，把 server + payload 推入 inbox
                    # 不做 method 过滤：所有 notifications/* 都进 inbox（ChannelInbox
                    # 本身不过滤，由 LLM 通过 format_digest 自己判断）
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
            # 把 inbox 挂到 agent（主循环 _build_channel_injection 会读这个）
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
        """api_key 推导：settings 值优先，为空时按 provider/base_url 环境变量兜底。

        与 _create_agent 主推导链同款（供 _thread_llm_client 复用，保证
        ThreadedLLMClient 拿到的凭证与主 client 一致）。
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
        # 与主 client 构造一致：auth_token 场景下用 auth_token 作凭证
        return api_key or auth_token

    def _maybe_trigger_curator(self):
        """后台检查 curator 是否该运行（非阻塞）。"""
        if not self.config.get("curator", {}).get("enabled", True):
            return

        def _check():
            try:
                if should_run_now(skills_dir()):
                    console.print("[dim]后台 curator 触发：整理技能库...[/dim]")
                    # R26 #14：consolidate 组件现场取——session/memory store 是
                    # initialize 前段建好的实例（同实例防跨实例 race）；
                    # llm 优先 aux（便宜模型），fallback 主 client
                    # （与 reflection 的 llm_for_reflection 同模式）。
                    # 缺任一组件时 run_curator_review 内部跳过并 log。
                    _agent = getattr(self, "agent", None)
                    _aux = getattr(_agent, "aux_llm_router", None)
                    if _aux is not None and _aux.is_aux_configured:
                        _llm = _aux
                    else:
                        # R26 终审 follow-up：线程上下文绝不借用主循环绑定的
                        # 主 client——ThreadedLLMClient per-call 独立连接
                        _llm = _thread_llm_client(self.config)
                    run_curator_review(
                        skills_dir(),
                        session_store=self.session_store,
                        memory_store=self.memory_store,
                        llm=_llm,
                    )
            except Exception as e:
                logger.debug("curator 触发失败: %s", e)

        # 守护线程，不阻塞主循环
        t = threading.Thread(target=_check, daemon=True)
        t.start()

    def new_session(self):
        """开始新会话。

        R30 审计 Medium-7：补齐会话级状态清理——
        - checkpoint_mgr 重建绑定新会话（旧代码指向旧会话，此后快照全写进
          旧目录、/rewind 回滚错对象；对照 resume_session 会重建，漏项）
        - 压缩状态 / auto_extract 游标 / 记忆注入去重 / context_tip /
          中断残留 / ephemeral 队列等全部重置
        裁决：审批缓存（PermissionChecker._approved）**不清**——docstring
        称会话内缓存，但用户批过的命令跨 /new 反复询问弊大于利
        （用户意图优先于算法，见 CLAUDE.md 设计原则 4）。
        """
        if self.session_store:
            self.session_id = self.session_store.create_session(
                model=self.config["model"]["name"],
                provider=self.config["model"]["provider"],
            )
            self.agent.session_id = self.session_id
        # checkpoint 重建（对齐 resume_session 的做法）
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
        conv = [m for m in msgs if m.get("role") != "system"]
        # R22 #40：按最后 compact 边界裁剪 pre-compact 旧消息
        # （session append-only，不裁会载入全量旧历史；旧会话无标记保守全量）
        conv = _truncate_at_last_compact_boundary(conv)
        # 清理冗余摘要占位（保留最近一个）——压缩频率修复前的会话可能有几十个
        # "[之前的对话已自动总结]" 占位，全注入上下文会撑爆且混乱
        conv = _cleanup_redundant_summaries(conv)
        # R21 #41：孤儿并行工具结果修复（对齐 CC recoverOrphanedParallelToolResults）
        # 会话保存中断可能留下「部分批次」的悬空 tool_result（assistant 有
        # tool_calls 但 result 缺失，或 result 无对应 tool_calls）。主循环的
        # _fix_tool_call_pairs 只在发送前修（不落盘）；这里加载时立即修复并
        # 回写，让持久化状态和后续轮次都干净。
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

        # 重建 checkpoint（对齐恢复的会话 id）
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

        # 显示最近几条消息让用户看到上下文（不显示 tool 消息，太碎）
        recent = [m for m in msgs[-6:]
                  if (m.get("content") or "").strip() and m.get("role") != "tool"]
        if recent:
            _print_message_list(recent, char_limit=300, header=f"最近 {len(recent)} 条历史消息：")

        return True

    def shutdown(self):
        """清理资源：终止后台任务等（P2b-T6 + P2c-T5）。"""
        # === Hooks: SESSION_END（在资源清理前触发，保留会话上下文） ===
        self._fire_session_end()

        # flush 技能使用统计(内存缓存 → 磁盘)
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

        # 关闭 session_store 持久连接（Windows 上不关会锁文件）
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

        # === P4a-T7 NEW: 主 agent 标记 stopped + 清理子进程 ===
        if hasattr(self, "team_coordinator") and self.team_coordinator:
            try:
                self.team_coordinator.shutdown_all()
            except Exception as e:
                logger.warning("team_coordinator shutdown_all 失败: %s", e)
            try:
                self.team_coordinator.update_status("main", "completed")
            except Exception as e:
                logger.warning("team_coordinator shutdown 失败: %s", e)

        # === ⑪c NEW: agent 资源清理 ===
        if hasattr(self, "agent") and self.agent:
            try:
                self.agent.cleanup()
            except Exception as e:
                logger.warning("agent.cleanup 失败: %s", e)


# ---------------------------------------------------------------------------
# 回调
# ---------------------------------------------------------------------------

def _make_approval_callback(aux_provider=None):
    """创建审批 callback（破坏性命令 + 写入路径都用这个）。

    callback 接收字符串,根据内容自动判断是命令还是路径,显示不同 prompt。
    审批结果的作用域(如实说明,勿夸大):
    - 命令 → 同意后加入 ~/.OmniMate/approved_commands.json,跨会话不再询问相同命令
    - 路径 → 三档(T5):y=本次允许(父目录进会话缓存,同目录后续写入不再问);
      a=总是允许(父目录持久化到 settings.json security.extra_allowed_roots,
      跨会话生效,与 /add-dir 同通道);N=拒绝

    R21 #45：命令审批加 e 选项——aux LLM 解释这条命令的用途 + LOW/MEDIUM/HIGH
    风险（aux_provider 注入，None 时选项隐藏）。fail-open。
    """
    def _explain(command: str):
        """R21 #45：aux LLM 解释命令（用途 + 风险等级）。fail-open。"""
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
        # R30 审计 Medium-8：命令/路径判定提取为 _is_path_item
        #（旧启发式把 del /s /q tmp 误判成路径审批；check_path 恒带
        # "文件写入审批: "前缀，优先识别该契约）
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
                # 返回哨兵值,由 PermissionChecker.check_path 统一做持久化
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
                    continue  # 解释后重新问
                return answer in ("y", "yes")
    return callback


def _make_ask_user_bridge():
    """构造 ask_user 的 CLI 桥接：渲染问题面板 + 读取用户选择。

    bridge(qdata) -> list[str]（选中的 label 列表）。
    异常（EOFError/KeyboardInterrupt）由 ask_user handler 统一捕获。
    """
    def bridge(qdata):
        question = qdata.get("question", "")
        options = qdata.get("options") or []
        multi = qdata.get("multi", False)

        lines = [f"[bold]{question}[/bold]", ""]
        for i, opt in enumerate(options):
            label = opt.get("label", "")
            desc = opt.get("description", "")
            lines.append(
                f"[cyan]{i + 1}[/cyan]. {label}" + (f" — {desc}" if desc else "")
            )
        console.print(Panel.fit(
            "\n".join(lines),
            title="❓ 需要你选择",
            border_style="cyan",
        ))

        if multi:
            raw = console.input(
                "[bold]选择序号（逗号分隔，可多选）> [/bold] ",
            ).strip()
            idxs = [
                int(part) - 1
                for part in raw.replace("，", ",").split(",")
                if part.strip().isdigit()
            ]
        else:
            raw = console.input("[bold]选择序号 > [/bold] ").strip()
            idxs = [int(raw) - 1] if raw.strip().isdigit() else []

        return [options[i]["label"] for i in idxs if 0 <= i < len(options)]
    return bridge


def _ts() -> str:
    """当前时间戳前缀（[HH:MM:SS]），用于工具调用进度输出。"""
    from datetime import datetime
    return f"[{datetime.now().strftime('%H:%M:%S')}]"


def _make_tool_call_callback(config: dict):
    """构造工具调用回调，闭包缓存 show_tool_progress，避免每次工具调用都重读 settings.json。"""
    show = (config or {}).get("display", {}).get("show_tool_progress", True)

    def callback(name: str, args: dict):
        """工具调用时的回调（打印进度）。"""
        if not show:
            return
        short_args = {}
        for k, v in (args or {}).items():
            s = str(v)
            short_args[k] = s if len(s) <= 80 else s[:77] + "..."
        console.print(f"[dim]{_ts()} → 调用工具: {name} {short_args}[/dim]")

    return callback


# 向后兼容：模块级函数仍可用，但每次调用都重读 config（不推荐）
def _on_tool_call(name: str, args: dict):
    """工具调用时的回调（打印进度）。每次重读 config（保留向后兼容，新代码请用 _make_tool_call_callback）。"""
    if not load_config().get("display", {}).get("show_tool_progress", True):
        return
    short_args = {}
    for k, v in (args or {}).items():
        s = str(v)
        short_args[k] = s if len(s) <= 80 else s[:77] + "..."
    console.print(f"[dim]{_ts()} → 调用工具: {name} {short_args}[/dim]")


def _make_cli_stream_callback():
    """构造 CLI 流式输出回调（04）。

    每收到 LLM 内容增量就即时打印（end="", flush=True），
    让用户看到边生成边显示，首字延迟 < 500ms。
    工具调用开始时打印一行简短提示。
    流结束时不打印（done 后由主流程打印换行）。
    """
    import sys

    def cb(event: dict) -> None:
        etype = event.get("type")
        if etype == "content":
            delta = event.get("delta") or ""
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        elif etype == "tool_call_start":
            # 内容流式过程中若切到工具调用，先换行收尾
            print()  # noqa: T201
            name = event.get("name", "?")
            console.print(f"[dim]{_ts()} ⟳ 准备调用 {name}...[/dim]")
        elif etype == "progress":
            # 子代理运行中：周期进度（让用户知道还在工作，不是卡死）
            msg = (event.get("message") or "").strip()
            elapsed = int(event.get("elapsed_seconds") or 0)
            if msg:
                print()  # noqa: T201
                console.print(
                    f"[dim]{_ts()} ⟳ 子代理[{elapsed}s] {msg}[/dim]"
                )
        # "done" 不打印：留给主流程处理换行
    return cb


# ---------------------------------------------------------------------------
# Slash 命令处理
# ---------------------------------------------------------------------------

def _handle_handoff_command(args: str, rt) -> bool:
    """处理 /handoff <sub> [args]。

    子命令：save / list / load / show / delete / export / import / help
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
        # 显示最近 6 条
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

        # 当前会话非空时确认
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

        # 替换历史 + 新建 session 留痕
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
            # 精确检查每个 memory pointer 是否在本机存在
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
    """Plan Mode 审批回调：打印计划 + 询问 y/N/edit/c。

    返回 (approved: bool, feedback: str, clear_context: bool)。
    - y/yes → (True, "", False)  批准，保留上下文继续执行
    - c/clear → (True, "", True) 批准并清空上下文执行（T9：调研过程的消息
      全部丢弃只留计划指令，执行阶段不烧调研 token；完整历史仍在
      transcripts/会话库可查）
    - edit → 收集一行 feedback → (False, feedback, False)
    - 其他（n/空/任意）→ (False, "用户拒绝", False)
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
    """处理 slash 命令。返回 True 表示已处理。"""
    parts = cmd.split(None, 1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit"):
        # 不 raise SystemExit（会跳过 rt.shutdown() 的清理 + SESSION_END hook），
        # 改为设置标志，主循环检测后 break 走正常退出。
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
        # B2 NEW: /permission [default|bypass|acceptEdits]
        # 不带参数 → 显示当前模式；带参数 → 切换（同步改 checker.mode + rt.agent.permission_mode）
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
        # OS 沙箱（对齐 Claude Code /sandbox）
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
            # I4 fix: 对不支持 set_sandbox_mode 的退化 checker 防御
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
        # 阶段 4 NEW：展示会话启动时锁定的 hook 快照 + 磁盘 diff 检测
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
        # 磁盘 diff 检测
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
        # E2 NEW: 列出自定义子代理定义（~/.OmniMate/agents + ./.omnimate/agents）
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
                # CCAR4 Task A：展示 diff 文件路径
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

    # === CCAR8 Task 12 NEW: 6 个新命令 ===
    if name == "/goal":
        return _handle_goal_command(args, rt)
    if name == "/poor":
        return _handle_poor_command(args, rt)
    if name == "/output-style":
        return _handle_output_style_command(args, rt)
    if name == "/trace":
        return _handle_trace_command(args, rt)
    if name == "/history":
        # R21 #42：全局输入历史（/history 列最近 20 条；/history N 打印第 N 条原文）
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
        # /resume 已被会话级 _resume_session_interactive 占用，跨项目 bundle
        # 恢复用 /resume_bundle 区分
        return _handle_resume_command(args, rt)

    # === CCAR9 Task 4: /init 生成 OMNIMATE.md（对标 Claude Code /init）===
    if name == "/init":
        return _handle_init_command(rt, args)

    # === CCAR10 Task 5: /resumable 列出/恢复可续跑子代理 ===
    if name == "/resumable":
        return _handle_resumable_command(args, rt)

    # === CCAR11 Task 2: /compact 手动 L4 压缩 + /context token 分布 ===
    if name == "/compact":
        return _handle_compact_cli(args, rt)
    if name == "/context":
        return _handle_context_cli(args, rt)

    # === CCAR11 Task 3: /status 状态一览 + /doctor 自诊断 + /diff 会话改动 ===
    if name == "/status":
        return _handle_status_cli(args, rt)
    if name == "/doctor":
        return _handle_doctor_cli(args, rt)
    if name == "/diff":
        return _handle_diff_cli(args, rt)

    # === CCAR11 Task 4: /add-dir 追加 safe_path 写白名单（运行时 + 持久化） ===
    if name == "/add-dir":
        return _handle_add_dir_cli(args, rt)

    # === CCAR11 Task 5: /paste 读剪贴板图片存 .paste/ ===
    if name == "/paste":
        return _handle_paste_command(args, rt)

    # === CCAR15 Task 4: /skill-learning instinct 学习链路管理 ===
    if name == "/skill-learning":
        return _handle_skill_learning_command(args, rt)

    return False


def _truncate_at_last_compact_boundary(msgs: list) -> list:
    """R22 #40：从最后一条 [COMPACT_BOUNDARY] 标记起截断（对齐 CC 边界重链）。

    压缩时摘要占位带标记入库（agent 侧）；resume 载入时标记之前的
    pre-compact 旧消息全部裁掉（它们已被总结进占位，再载入既撑上下文
    又与摘要重复）。标记行剥掉、摘要正文保留。找不到标记返回原列表
    （保守——旧会话/未压缩会话行为不变）。
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
    # 剥首条标记行（保留摘要正文）
    first = kept[0]
    content = first.get("content", "")
    first["content"] = content.replace("[COMPACT_BOUNDARY]\n", "", 1)
    logger.info(
        "resume：按 compact 边界裁剪（丢弃 %d 条 pre-compact 消息）",
        last_idx,
    )
    return kept


def _cleanup_redundant_summaries(msgs: list) -> list:
    """清理历史里多余的摘要占位（保留最近一个）。

    压缩频率修复前，长会话可能被压几十次，DB 里存了多个"[之前的对话已自动总结]"
    占位。恢复时全部注入会让上下文被占位符撑爆且混乱——保留最近一个摘要，
    删掉更早的（其内容已被新摘要覆盖）。
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
    drop = set(summary_idx[:-1])  # 保留最后一个摘要
    return [m for i, m in enumerate(msgs) if i not in drop]


def _handle_rewind_command(rt: RuntimeContext, args: str) -> None:
    """/rewind：列出 checkpoint 快照，选择回滚（恢复文件 + 可选对话）。

    对齐 Claude Code：每个用户 prompt 前自动快照 agent 修改过的文件，
    这里列出里程碑并回滚到某个时点。
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

    # 摘要模式：s / s3 / s 0 —— 把该 checkpoint 之后的对话压成摘要
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

    # 4 模式菜单（对齐 Claude Code /rewind）
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
        # 恢复代码
        restored = mgr.restore_files(sid)
        if restored:
            console.print(f"[green]已恢复 {len(restored)} 个文件[/green]")
            for p in restored:
                console.print(f"  [dim]{p}[/dim]")
        else:
            console.print("[yellow]该快照没有可恢复的文件[/yellow]")

    if mode == "1" or mode == "2":
        # 恢复对话
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
    """把选中 checkpoint 之后的对话压成摘要（对齐 Claude Code Summarize from here）。

    保留该 checkpoint 时的对话，把其后追加的消息交给 LLM 压成摘要，
    conversation_history = checkpoint 对话 + [摘要 user 消息]。
    """
    mgr = getattr(rt, "checkpoint_mgr", None)
    if not mgr or not rt.agent:
        console.print("[yellow]Checkpoint 不可用[/yellow]")
        return

    ckpt_conv = mgr.get_conversation(sid)
    current = rt.agent.conversation_history

    # 该点之后 = 当前 history 中 checkpoint 对话之后追加的部分（前缀匹配）
    after = current
    if (ckpt_conv and len(current) > len(ckpt_conv)
            and current[:len(ckpt_conv)] == ckpt_conv):
        after = current[len(ckpt_conv):]
    if not after:
        console.print("[yellow]该 checkpoint 之后没有新对话[/yellow]")
        return

    try:
        from agent.context_compressor import _summarize_conversation
        # _summarize_conversation 在 Plan 2A 改为 async（LLMClient.chat_completions
        # 已 async）。_summarize_rewind 是 sync 函数（被 sync _handle_rewind_command
        # 调用），用 asyncio.run 桥接（同 reflection.py:150 的处理方式）。
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
# CCAR8 Task 12 NEW: 6 个新命令的 handler
# ---------------------------------------------------------------------------

def _goal_state_path(rt) -> Path:
    """goal 持久化路径：~/.OmniMate/.goal/current.json。"""
    return Path(rt.home) / ".goal" / "current.json"


def _goal_status_output(rt, capsys_safe: bool = True) -> None:
    """打印 goal 状态（直接 console.print）。"""
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
    """启动新 goal。

    核心三步（pause 旧 goal / 建新 GoalState / 持久化 + 挂 agent）走
    agent/goal.start_goal_agent 共享函数（CCAR12 Task 4，与 LLM goal_start
    工具同源）；CLI 层只保留 console 输出 / aux_llm 拆解 / history 注入。
    """
    from agent.goal import start_goal_agent

    # 0. 先记旧 goal（共享函数会 pause 它，这里只为打印）
    old_gs = getattr(rt.agent, "_goal_state", None)
    will_pause_old = old_gs is not None and old_gs.status == "active"

    # 1-2. 核心三步：pause 旧 goal + 建新 GoalState + 持久化 + 挂 agent
    goal_cfg = rt.config.get("goal", {}) if rt.config else {}
    budget_limit = goal_cfg.get("default_token_budget", 200_000)
    gs = start_goal_agent(
        rt.agent, objective, token_budget=budget_limit,
        persist_path=_goal_state_path(rt),
    )

    if will_pause_old:
        console.print(f"[dim]已自动 pause 旧 goal: {old_gs.objective}[/dim]")

    # 3. 尝试用 aux_llm 拆解为子 task（fail-open，无 aux 跳过）
    # CCAR8 final fix: 直接走 agent.aux_llm_router（不再走 RuntimeContext.aux_llm_client 死字段，
    # 工具 dispatch 路径早已统一到 agent_ref.aux_llm_router）
    aux_client = getattr(rt.agent, "aux_llm_router", None)
    if aux_client is not None:
        try:
            # 异步函数：用 asyncio.run 包装（CLI sync 路径）
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # 已在事件循环里（如测试环境），跳过同步调
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

    # 4. 把 objective 作为下一轮 user 输入（让 agent 开始追目标）
    # 设计：直接塞 conversation_history 末尾，主循环下次跑就看到。
    # ⚠️ 只能在 CLI 层做（会话循环外）；工具路径（goal_start tool）在
    # assistant(tool_calls) 与 tool result 之间插 user 消息会破坏严格交替。
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
    """安全包装 decompose_with_llm（任何异常返回空列表）。"""
    from agent.goal import decompose_with_llm
    try:
        return await decompose_with_llm(gs, objective, aux_client)
    except Exception as e:
        logger.warning("goal decompose 异常（fail-open）: %s", e)
        return []


def _handle_goal_command(args: str, rt) -> bool:
    """处理 /goal 命令。

    /goal <objective>       启动新 goal
    /goal status            查看状态
    /goal pause [reason]    手动 pause
    /goal resume            resume
    /goal continue          立即触发下一轮（pause → active）
    /goal clear             取消
    /goal tasks             列出关联 task
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
        # 删持久化文件
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

    # 否则当作 objective 启动新 goal
    objective = args.strip()
    if not objective:
        console.print("[yellow]用法: /goal <objective> | status | pause | resume | clear | tasks[/yellow]")
        return True
    _start_new_goal(rt, objective)
    return True


def _handle_output_style_command(args: str, rt) -> bool:
    """C6（CCB outputStyles）：/output-style 列表 / 切换 / off。

    风格目录：`<项目>/.omnimate/output-styles/`（覆盖）+ `~/.OmniMate/output-styles/`。
    切换写 settings.json 顶层 output_style + invalidate prompt（下一条消息生效）。
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
    """写 settings.json + 运行时 config + invalidate prompt。fail-open 持久化。"""
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
    """处理 /poor on|off|status。"""
    arg = args.strip().lower() if args else ""
    if not arg or arg == "status":
        is_on = getattr(rt, "_poor_mode_on", False)
        console.print(f"Poor Mode: [cyan]{'ON' if is_on else 'OFF'}[/cyan]")
        return True
    if arg == "on":
        import copy as _copy_mod
        from agent.poor_mode import apply_poor_preset
        # R30d-C11：开启前快照 config，off 时完整回滚（此前 off 只清标志，
        # 被关掉的 reflection/摘要等要等重启才恢复——on/off 不对称）
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
    """/trace today|yesterday|<YYYY-MM-DD>|tail [N]。"""
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

    # summary 路径：today / yesterday / 具体日期
    import datetime as _dt
    if when == "today":
        date_str = _dt.datetime.now().strftime("%Y-%m-%d")
    elif when == "yesterday":
        date_str = (_dt.datetime.now() - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        date_str = when  # 假设用户给了 YYYY-MM-DD

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
    """生成 <cwd>/OMNIMATE.md（CCAR9 Task 4，对标 Claude Code /init）。

    收集项目信息（目录树/关键文件/类型统计，fail-open 缺哪跳哪）
    → 主 LLM 生成四段式（项目本质/常用命令/架构/约定）
    → 写 cwd/OMNIMATE.md（下次会话由 prompt_builder 自动注入 system prompt）。

    - /init            已存在不覆盖
    - /init --force    覆盖重新生成
    """
    # 局部 import：与 codebase 一致（agent_defs/prompt_builder 等都这么干）
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

    # ── 收集项目信息（fail-open，缺哪跳哪）──
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

    # 2. 关键配置文件（前 4KB）
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

    # ── 构造 prompt → 主 LLM 四段式生成 ──
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

    # asyncio.run 在已有事件循环时会 RuntimeError（测试环境/REPL）
    # —— 走 try/except 兜底，对齐 _start_new_goal 的同款模式
    text = ""
    try:
        text = asyncio.run(_gen())
    except RuntimeError:
        # 已在事件循环内（如 pytest-asyncio 管理时）→ 放弃生成
        # （对齐 _start_new_goal：loop 已跑时 run_until_complete 会冲突，
        #   同步 CLI 路径不会进这分支；测试环境走 mock 不依赖真 LLM）
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
    """/inbox 显示 ChannelInbox 未消费消息。

    ChannelInbox 是 MCP server notifications 的落地 inbox。
    本命令只读展示，消费（mark_consumed）由主循环 _assemble_turn_messages 完成。
    """
    # channel_inbox 在 AIAgent._channel_inbox（Task 11 setter 注入）
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
    """/mailbox send|check|clear（teammate 邮箱 CLI 包装）。"""
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
        # CCAR8 final fix: 默认读 check_all（含已读）。
        # 之前默认 unread_only=True，但主循环 _build_mail_injection 每轮 check_unread
        # 后立即 mark_read，导致用户敲 /mailbox check 时邮件已被注入路径清空 → 永远空。
        # 修法：默认显示全部，加 --unread / -u 才过滤未读；"all" 关键字向后兼容。
        rest_lower = rest.lower()
        if "--unread" in rest_lower or "-u" in rest_lower.split():
            unread_only = True
        elif "all" in rest_lower:
            unread_only = False
        else:
            unread_only = False  # 默认显示全部（关键修复）
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
    """/resumable [agent_id]：列出/恢复可续跑的子代理（CCAR10 Task 5）。

    - 无参：列出 status=running 的子代理（含 agent_id/status/消息数/时间），
            并提示语义边界——只有存了 transcript 的子代理才能真正恢复
    - 有参：<agent_id> 调 _run_resume 续命，成功绿色显示结果前 2000 字符，
            失败红色提示

    实现要点：
    - subagent_persistence 的 list_resumable() **不接 base_dir**（走模块级
      _sessions_dir()），所以本 handler 也不传 base_dir；测试通过 monkeypatch
      _sessions_dir 来隔离
    - LLM 配置和 agent_ref 从 rt 取（与 _handle_trace/_handle_poor 同款模式）
    """
    import json as _json
    from agent import subagent_persistence as sp

    parts = (args or "").split()

    # ── 无参：列表分支 ──
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
            # 消息数：现场读 transcript 算（meta 里没存）
            try:
                n_msgs = len(sp.load_transcript(aid))
            except Exception:
                n_msgs = "?"
            # 时间：优先 updated_at，其次 created_at
            ts = it.get("updated_at") or it.get("created_at") or ""
            if isinstance(ts, (int, float)):
                import time as _time
                ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(ts))
            # agent_type 只读着好看
            atype = it.get("agent_type", "")
            atype_tag = f"[dim]({atype})[/dim] " if atype else ""
            # 关键：agent_id 用 \[ 转义——避免被 Rich 当成 style tag 吃掉
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
    # 支持用户在 agent_id 后附带自定义续命指令
    if len(parts) > 1:
        instruction = " ".join(parts[1:])

    from tools.subagent_resume_tool import _run_resume

    # 从 rt 取上下文（LLM 配置 + agent_ref），对齐 _handle_poor/_handle_trace 模式
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
        # 非 JSON 直接原样显示（理论上不会，_run_resume 一定返 JSON）
        console.print(f"[yellow]{result_json[:2000]}[/yellow]")
        return True

    if "error" in data:
        err = data.get("error", "")
        console.print(f"[red]恢复失败：[/red]{err}")
        return True

    # 成功：绿色显示前 2000 字符
    text = data.get("result", "")
    truncated_tag = " [dim](已截断到 2000 字符)[/dim]" if len(text) > 2000 else ""
    console.print(f"[green]恢复完成：[/green]{text[:2000]}{truncated_tag}")
    return True
















def _handle_diff_cli(args: str, rt) -> bool:
    """/diff 本会话文件改动（CCAR11 Task 3）。

    基于 CheckpointManager 的实际能力：列 tracked_files（编辑工具改过的
    文件）+ 快照数。无 checkpoint manager / 无追踪记录时提示。
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
# CCAR11 Task 4 NEW: /add-dir 运行时白名单 + 持久化
# ---------------------------------------------------------------------------

def _persist_extra_root(root: str) -> bool:
    """把额外白名单根目录持久化到 settings.json 的 security.extra_allowed_roots。

    读-改-写：load_settings() 已有内容（含默认值深合并）→ append（去重）→
    save_settings() 原子写回。走 load_config 同源的真实配置轨
    （config.yaml 首次启动会被迁走改名 .bak，写 yaml 是断轨的）。

    返回 True 表示新写入，False 表示已存在（幂等，不重复写）。
    fail-open：读/写失败抛异常给调用方（命令层捕获提示，不影响运行时白名单）。
    """
    from agent.settings import persist_extra_allowed_root

    # T5：逻辑下沉到 agent/settings.py（与写路径审批"总是允许"档共用同一通道）
    return persist_extra_allowed_root(root)


def _load_persisted_extra_roots(config: dict) -> int:
    """启动时把 settings.json 的 security.extra_allowed_roots 灌进运行时白名单。

    RuntimeContext.__init__ 调用（config 来自 load_config()，默认读
    settings.json——与 _persist_extra_root 写入同一文件，闭环不断轨）。
    返回成功加载数（fail-open：单条失败跳过）。
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
    """/add-dir <path>：追加 safe_path 写白名单（CCAR11 Task 4）。

    - 无参数：列出当前白名单（cwd + ~/.OmniMate + 额外追加）
    - 带目录：resolve + is_dir 校验 → 运行时追加（去重幂等）→ 持久化到
      settings.json 的 security.extra_allowed_roots（下次启动自动加载）
    - 安全底线不变：受保护路径（~/.ssh 等）和项目代码写保护在 safe_path
      里先于白名单检查，加白名单不能绕过。
    """
    from agent.permission import (
        default_allowed_roots, list_extra_allowed_roots,
        add_extra_allowed_root,
    )

    parts = (args or "").split()
    if not parts:
        # 列出当前白名单
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

    # 1) 运行时生效：追加进 default_allowed_roots（safe_path 默认路径）
    added = add_extra_allowed_root(target)
    # 2) 持久化：settings.json security.extra_allowed_roots（读-改-写）
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
# CCAR11 Task 5 NEW: /paste 剪贴板图片保存
# ---------------------------------------------------------------------------

def _handle_paste_command(args: str, rt) -> bool:
    """/paste：读剪贴板图片保存到 <workspace>/.paste/（CCAR11 Task 5）。

    小而美的 Windows 快捷路径：PowerShell Clipboard API 读图 → 存 PNG →
    打印路径。只保存不自动分析（用户可能想配文字再发给 AI）。

    fail-open 约束（设计如此，不是 bug）：
    - 非 Windows 平台不启动子进程，直接提示手动保存
    - PowerShell 失败/超时/剪贴板无图片 → 提示手动给路径，不崩
    - 保存路径走 get_workspace_cwd()（worktree 子代理 context 隔离）
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
            # 非 Windows 没有 PowerShell Clipboard，直接走 fail-open 提示
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
            "[dim]在消息中引用该路径即可让 AI 分析（image_analyze）[/dim]"
        )
    except Exception as e:
        console.print(
            f"[yellow]粘贴失败（{e}）。可手动保存图片后在消息中给路径。[/yellow]"
        )
    return True


























def _manage_whitelist(rt: RuntimeContext, args: str):
    """管理审批白名单（/approved）。

    /approved                    列出已批准命令 + 前缀规则 + 持久化写入根目录
    /approved remove <n|命令>    按序号/命令移除已批准命令
    /approved remove-root <n|路径>  移除持久化写入根目录（T5，settings.json
                                   security.extra_allowed_roots + 运行时白名单）
    /approved remove-prefix <pN|前缀>  移除前缀规则（R25 #2，curated 表派生）
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
                # 截断长命令
                display = cmd if len(cmd) <= 80 else cmd[:77] + "..."
                console.print(f"  [{i}] {display}")
        # R25 #2：前缀规则展示
        prefixes = sorted(getattr(checker, "_persistent_prefixes", set()) or set())
        if prefixes:
            console.print("\n[bold]前缀规则（同前缀命令免审批）：[/bold]")
            for i, p in enumerate(prefixes, 1):
                console.print(f"  p{i}. {p}")
        # T5：持久化写入根目录（"总是允许"档落盘的条目）
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

    # R25 #2：前缀规则移除（支持 p序号或完整前缀字符串）
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
    """切换当前激活模型。

    /model          列出所有模型
    /model <name>   切换到指定模型
    """
    from agent.settings import list_models, set_default_model, get_current_model_config

    models = list_models()
    if not models:
        console.print("[yellow]未配置任何模型。在 settings.json 的 llm 段或 models 段添加。[/yellow]")
        return

    current = get_current_model_config().get("name")

    if not args.strip():
        # 列出所有模型
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

    # 持久化切换
    if not set_default_model(target):
        console.print(f"[red]切换失败[/red]")
        return

    # 重建 agent 的 llm_client
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

        # === Hooks: CONFIG_CHANGE（模型切换） ===
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
    """构造状态行内容（CCAR10 Task 3）。

    返回值：非空字符串 → 主循环 console.print("[dim]...[/dim]")；
            空串      → 不打印。

    任何异常都吞掉返回 ""（statusline 不能影响主流程）。
    段顺序：model │ 会话 token │ goal 状态 │ 项目名
    """
    try:
        # 1. 开关检查（默认 enabled=True）
        cfg = (getattr(rt, "config", None) or {})
        sl_cfg = cfg.get("statusline", {}) if isinstance(cfg, dict) else {}
        if not sl_cfg.get("enabled", True):
            return ""

        segs = []

        # 2. model 段（agent.model 是实例字段，不带 ⚡ 也行——这里前缀 emoji 做视觉锚）
        model = getattr(agent, "model", "") or ""
        if model:
            segs.append(f"⚡{model}")

        # 3. token 段：优先用 _llm_usage_stats（实时累加，含 cache）；
        #    fallback 到 session_total_tokens（兼容旧字段 / mock）。
        usage_stats = getattr(agent, "_llm_usage_stats", None)
        if usage_stats and isinstance(usage_stats, dict):
            tokens = (
                int(usage_stats.get("total_prompt_tokens", 0) or 0)
                + int(usage_stats.get("total_completion_tokens", 0) or 0)
            )
        else:
            tokens = int(getattr(agent, "session_total_tokens", 0) or 0)
        segs.append(f"会话 {_format_tokens(tokens)} tok")

        # 4. goal 段：active/paused/completed/failed 显示；cancelled 隐藏（视为废弃）
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
            # cancelled / 未知 → 不显示

        # 5. 项目段：取 project_key 末段（canonical-git-root 后缀）
        proj_key = getattr(rt, "_statusline_project_key", "") or ""
        if proj_key:
            tail = proj_key.rsplit("-", 1)[-1]
            if tail:
                segs.append(f"项目:{tail}")

        return " │ ".join(segs)
    except Exception:
        return ""


def _is_path_item(item: str) -> bool:
    """R30 审计 Medium-8：审批回调的命令/路径判定（保守——误判成路径的
    命令会拿到"即将写入路径"文案和"总是允许并记住"的错误语义）。

    规则：多 token（含空格）一律命令形态（del /s /q tmp、rm -rf build/）；
    ~ 开头 / Windows 盘符（C:\\ 或 C:/）/ 单 token 含分隔符（src/lib、/tmp）
    才算路径形态。旧启发式 `"/" in item` + 三元优先级 bug（len<3 整体 False）
    已废弃。
    """
    s = (item or "").strip()
    if not s:
        return False
    # permission.check_path 的既有契约：路径审批恒带"文件写入审批: "前缀
    # （permission.py:1740）——直接识别，不再猜
    if s.startswith("文件写入审批"):
        return True
    if s.startswith("~"):
        return True
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        return True  # Windows 盘符
    if " " in s:
        return False  # 命令形态（含 flag / 路径参数——都不是"路径审批"）
    # 单 token：含分隔符即路径形态（命令不可能是单 token 含 / 或 \）
    return "/" in s or "\\" in s


def _should_exit_on_interrupt_sentinel(
    last_interrupt_ts: float, now: float, window: float = 1.0,
) -> bool:
    """R30 审计 High-3：输入线程的 Ctrl+C 哨兵是否应退出 REPL。

    同一次 Ctrl+C 可能同时被两个消费者收到：主线程（asyncio.run 内
    KeyboardInterrupt → agent.interrupt()，记录时间戳）与输入线程
    （console.input 抛 KeyboardInterrupt → _INTERRUPT_SENTINEL 入队）。
    若哨兵到达时窗口内刚发生过回合内中断，视为同一次按键的重复消费
    （中断本轮，不退出）→ 返回 False；空闲提示符下的 Ctrl+C（无近期
    回合内中断）保持原语义（退出）→ 返回 True。
    """
    return (now - last_interrupt_ts) > window


def run_interactive(resume_last: bool = False, cli_agents: dict = None):
    """启动交互式 CLI。

    参数：
        resume_last: True 时自动恢复最近会话（-c/--continue 触发）；
                     False 时提示用户选择。
        cli_agents: 阶段 6 NEW —— `--agents '{json}'` 传入的子代理定义。
                    优先级介于用户级和项目级之间（对齐 Claude Code `--agents`）。
    """
    # 阶段 6 NEW: 注入 CLI 子代理到 scan_agent_defs 的来源链
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

    # R22 #11：输入线程 + 队列（模型跑时用户输入排队，不打断当前响应；
    # 工具批结束后由 agent._drain_queued_input 回流 ephemeral）
    import queue as _queue_mod
    import threading as _threading_mod
    _input_q = _queue_mod.Queue()
    _input_stop = _threading_mod.Event()
    # R30c-C6：对象哨兵替代字符串哨兵——用户字面输入 "__EOF__" 不再误退出
    _EOF_SENTINEL = object()
    _INTERRUPT_SENTINEL = object()
    # R30 审计 High-3：最近一次"回合内中断"的时间戳（主线程 Ctrl+C 处理器
    # 记录；输入线程的 _INTERRUPT_SENTINEL 消费时用来去重同一次按键）
    _last_ctrl_c = 0.0

    def _input_reader():
        """daemon 线程：持续读控制台输入入队（EOF/异常即停）。"""
        while not _input_stop.is_set():
            try:
                line = console.input("[bold cyan]你:[/bold cyan] ")
                _input_q.put(line.strip())
            except EOFError:
                _input_q.put(_EOF_SENTINEL)
                return
            except KeyboardInterrupt:
                # 对齐原语义：输入等待时 Ctrl+C = 退出（原 except 里打"再见"）
                _input_q.put(_INTERRUPT_SENTINEL)
                return
            except Exception:
                return
    _input_thread = _threading_mod.Thread(target=_input_reader, daemon=True)
    _input_thread.start()
    # agent 接队列（回流通道）
    try:
        rt.agent.set_input_queue(_input_q)
    except Exception:
        pass

    # 主循环
    while True:
        # R30d-C8：优先消费模型运行期间排队的 slash 命令（agent drain 分流
        # 进 _queued_cli_commands；对话结束后在此按正常命令处理执行）
        _deferred = getattr(rt.agent, "_queued_cli_commands", None)
        if _deferred:
            user_input = _deferred.pop(0)
        else:
            user_input = _input_q.get()
        if user_input is _EOF_SENTINEL:
            console.print("\n再见！")
            break
        if user_input is _INTERRUPT_SENTINEL:
            # R30 审计 High-3：同一次 Ctrl+C 可能同时被输入线程（本哨兵）与
            # 主线程（回合内中断，记录 _last_ctrl_c）消费——窗口内到达的哨兵
            # 是重复消费，吞掉不退出；空闲提示符下的 Ctrl+C 保持退出语义
            if _should_exit_on_interrupt_sentinel(_last_ctrl_c, time.monotonic()):
                console.print("\n再见！")
                break
            continue

        if not user_input:
            continue

        # R21 #39：大段粘贴外存 + 占位符（session 存占位符省空间）
        from agent.input_history import store_paste_if_large
        user_input, _pasted_to = store_paste_if_large(user_input, rt.home)

        # R21 #42：全局输入历史（跨会话召回，fail-open）
        # R30d-C10：先外存再记历史——历史里存占位符而非全文大原文
        #（此前顺序反了，history.jsonl 存的是未替换的大原文，越滚越大）
        try:
            from agent.input_history import GlobalHistory
            GlobalHistory(rt.home).append(user_input)
        except Exception:
            pass

        # 0. `#` 快捷写记忆（对齐 Claude Code 体验）
        if user_input.startswith("#"):
            text = user_input[1:].strip()
            if text:
                _quick_save_memory(rt, text)
            else:
                console.print("[yellow]用法：# <记忆内容>（如 # 项目用 pytest）[/yellow]")
            continue

        # 1. 处理 slash 命令
        if user_input.startswith("/"):
            # 先检查是否是技能束命令（R30c-C6：解析统一为 split()[0]，
            # 与下方技能束/技能触发用同一规则）
            cmd_name = user_input.split()[0]
            if cmd_name in rt.bundle_commands:
                pass  # 走技能束触发逻辑
            elif cmd_name in rt.skill_commands:
                pass  # 走技能触发逻辑
            elif _handle_command(user_input, rt):
                if rt.quit_requested:
                    break  # /quit 请求：走正常退出（rt.shutdown()）
                continue
            else:
                # R30c-C6：未知 slash 命令——命令名形态（/word）时报错不发模型
                # （此前 /sesion 这类敲错会整条静默发给 LLM）。路径形态
                # （如 "/etc/passwd 是什么"）不拦，正常作为消息发送。
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

        # 2. 检查是否触发技能束（cmd_name 解析与步骤 1 同规则——R30c-C6 统一）
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
            # round3: context:fork 技能在隔离子代理跑
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
            # R21 #39：发送 agent 前展开粘贴引用（session 里存的是占位符）
            from agent.input_history import expand_paste_references
            agent_input = expand_paste_references(user_input, rt.home)
            # Checkpoint：每个用户 prompt 前快照（对齐 Claude Code，/rewind 可回滚）
            if rt.checkpoint_mgr:
                try:
                    rt.checkpoint_mgr.create_snapshot(
                        conversation=list(rt.agent.conversation_history),
                    )
                except Exception as e:
                    logger.warning("checkpoint 快照失败: %s", e)
            # 先打 AI: 前缀,让流式输出在这个前缀之后显示
            console.print("[bold green]AI:[/bold green]")
            # Task E1: run_conversation 已改 async（T_D4）。
            # 保持 run_interactive 同步签名（run_skill_in_fork 等下游依赖同步上下文），
            # 每次调用用 asyncio.run 驱动一次完整的 async run_conversation。
            # 全量 async 改造留到 Plan 2B（届时 skill_fork 也改 async，可消除嵌套 asyncio.run）。
            response = asyncio.run(rt.agent.run_conversation(agent_input))
            # 流式模式(stream_callback 已设)的内容已经在 run_conversation 过程中显示,
            # 不再重复 print。非流式模式(无 callback)才 print response。
            # 但 LLM 失败/预算耗尽等兜底文案不走流式（没有内容增量），必须显示，
            # 否则用户会看到"没反应就断了"（之前莫名断开的根因）。
            if not getattr(rt.agent, "_stream_callback", None):
                console.print(response)
            elif response and response.startswith(
                ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数",
                 "[模型只产出了思考过程", "[LLM 返回了空响应")
            ):
                # 流式模式下兜底消息（空响应/思考模型/grace exit/预算耗尽/LLM失败）
                # 不走 stream_callback，必须主动 print，否则用户看到"AI:"后空白
                console.print(f"[yellow]{response}[/yellow]")

            # 5. 保存助手响应到 session
            if rt.session_store and rt.session_id:
                rt.session_store.append_message(
                    rt.session_id, "assistant", response,
                )

            # CCAR10 Task 3 NEW: 每轮响应完打印 statusline（model/token/goal/项目）
            # 放在 response 完整输出之后、不接 Live（Windows + input() 冲突）；
            # 中断/异常路径都不打 statusline（用户主动断开就不该再追加信息）。
            try:
                _sl = _render_statusline(rt, rt.agent)
                if _sl:
                    console.print(f"[dim]{_sl}[/dim]")
            except Exception as _e:
                logger.debug("statusline 渲染失败（不阻塞）: %s", _e)
        except KeyboardInterrupt:
            rt.agent.interrupt()
            _last_ctrl_c = time.monotonic()  # High-3：哨兵去重窗口锚点
            console.print("[yellow]\n[已中断][/yellow]")
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            logger.exception("agent 运行错误")

    # === P2b-T6 NEW: 退出前清理后台任务 ===
    rt.shutdown()


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
        # user 消息必须在 run_conversation 之前入库，保证会话历史顺序：
        # user → assistant(tool_calls) → tool → ...（否则工具轮次会排在 user 前，
        # 恢复时 assistant 的 tool_calls 前面没有 user 消息，违反 API 消息协议）
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "user", message)
        # Task E1: run_conversation 已改 async（T_D4），同步入口用 asyncio.run 驱动。
        response = asyncio.run(rt.agent.run_conversation(message))
        # 流式输出（streaming.enabled 默认 True）已经实时打印过内容；
        # 这里只换行收尾，避免重复打印完整响应。非流式模式才 print(response)。
        streaming_enabled = rt.config.get("streaming", {}).get("enabled", True)
        if streaming_enabled:
            print()  # 流式结束换行
        else:
            print(response)
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "assistant", response)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        logger.exception("one-shot 运行错误")
    finally:
        # === P2b-T6 NEW: 退出前清理后台任务 ===
        rt.shutdown()


# ---------------------------------------------------------------------------
# Task E1: 主入口函数
# ---------------------------------------------------------------------------
# 设计决策：把原 main.py 内联的参数解析 + 分发逻辑抽到 cli.main，
# 这样 main.py 只负责 stdout 编码 + MCP 初始化 + 调 cli.main。
#
# 不在 cli.main 外层套 asyncio.run 包装的原因：
#   run_interactive / run_one_shot 保持同步签名，内部用 asyncio.run 驱动
#   async run_conversation（避免破坏 run_skill_in_fork 等下游同步调用链）。
#   若 cli.main 再套一层 asyncio.run，会与内部的 asyncio.run 嵌套报错：
#       "asyncio.run() cannot be called from a running event loop"
#   所以本 task 的 asyncio.run 包装发生在 run_one_shot / run_interactive 内部
#   （紧贴 async run_conversation 调用点），cli.main 只是同步分发器。
#
# Plan 2B 会把 run_interactive / run_one_shot / run_skill_in_fork 全改 async，
# 届时 cli.main 才真正需要 asyncio.run 包装（且不嵌套）。

def main(argv: list = None) -> None:
    """CLI 主入口（Task E1 抽出，供 main.py 调用）。

    参数：
        argv: 命令行参数列表（None 时用 sys.argv，便于测试）

    支持的调用形式：
        python main.py                         # 交互模式
        python main.py -c / --continue         # 自动恢复最近会话
        python main.py chat <msg>              # 非交互一次性问答
        python main.py --agents '{json}'       # CLI 注入子代理（阶段 6 NEW）
        python main.py --agents '{json}' chat <msg>
    """
    if argv is None:
        argv = sys.argv
    args = argv[1:]
    cli_agents_raw = None

    # 提取 --agents 参数（不破坏旧的 chat/-c/--continue 逻辑）
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
        # 从 args 里移除 --agents 及其值，让旧逻辑正常工作
        args = args[:idx] + args[idx + 2:]

    # 非交互模式：python main.py chat "你好"
    if args and args[0] == "chat":
        if cli_agents_raw:
            from agent.agent_defs import inject_cli_agents
            inject_cli_agents(cli_agents_raw)
        run_one_shot(" ".join(args[1:]))
        return

    # 交互模式：检查 -c / --continue 标志
    resume_last = "-c" in args or "--continue" in args
    run_interactive(resume_last=resume_last, cli_agents=cli_agents_raw)
