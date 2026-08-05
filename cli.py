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

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 运行时初始化
# ---------------------------------------------------------------------------

class RuntimeContext:
    """聚合 agent 运行时的所有组件。"""

    def __init__(self):
        self.config = load_config()
        self.home = get_omnimate_home()
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
        # 0. 设置权限检查器（注入破坏性命令审批 callback + 持久化白名单）
        # B2: perm_mode 在 try 外定义，AIAgent 构造处（行 460+）也要用
        perm_mode = self.config.get("security", {}).get("permission_mode", "default")
        try:
            from agent.permission import set_default_checker, PermissionChecker
            from agent.settings import approved_commands_path, approved_paths_path
            set_default_checker(PermissionChecker(
                approval_callback=_make_approval_callback(),
                whitelist_file=str(approved_commands_path()),
                paths_whitelist_file=str(approved_paths_path()),
                mode=perm_mode,
                hooks_registry=self.hooks_registry,  # round3 D2 NEW: 权限审计 hook
            ))
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

        # === Mem-T7 NEW: memory retriever 装配（多文件检索） ===
        self.memory_retriever = None
        mem_cfg = self.config.get("memory", {})
        if (mem_cfg.get("enabled", True) and
                mem_cfg.get("retrieval_enabled", True)):
            from agent.memory_retriever import retrieve_relevant
            self.memory_retriever = retrieve_relevant  # 函数引用

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

        # 6. 后台触发 curator（不阻塞启动）
        self._maybe_trigger_curator()

        # === Memory Curator 后台触发(照搬 skill curator 模式) ===
        try:
            from constants import get_omnimate_home
            from agent.memory_curator import (
                should_run_now_memory,
                apply_automatic_transitions,
                run_memory_review,
            )
            memory_dir = get_omnimate_home() / ".memory"
            if memory_dir.exists() and should_run_now_memory(memory_dir, config=self.config):
                import threading, datetime

                def _run_memory_curator():
                    try:
                        counts = apply_automatic_transitions(memory_dir)
                        # 第 2 阶段(主模型 review)
                        review_summary = f"第 1 阶段: {counts}"
                        try:
                            factory = self._make_memory_review_agent_factory()
                            review_report = run_memory_review(
                                memory_dir,
                                agent_factory=factory,
                                config=self.config,
                            )
                            review_summary = (
                                f"第 1 阶段: {counts}; "
                                f"第 2 阶段: reviewed={review_report['buckets_reviewed']}, "
                                f"actions={review_report['executed_actions']}, "
                                f"errors={review_report['errors']}"
                            )
                        except Exception as e:
                            logger.warning(
                                "第 2 阶段失败(保留第 1 阶段结果): %s", e,
                            )
                            review_summary = (
                                f"第 1 阶段: {counts}; 第 2 阶段失败: {e}"
                            )
                        # 写状态
                        from agent.memory_curator import (
                            load_memory_curator_state,
                            save_memory_curator_state,
                        )
                        state = load_memory_curator_state(memory_dir)
                        state["last_run_at"] = datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        state["last_run_summary"] = review_summary
                        save_memory_curator_state(memory_dir, state)
                        logger.info("Memory Curator 完成: %s", review_summary)
                    except Exception as e:
                        logger.warning("Memory Curator 后台运行失败: %s", e)

                threading.Thread(target=_run_memory_curator, daemon=True).start()
        except Exception as e:
            logger.debug("Memory Curator 触发检查失败(不阻塞): %s", e)

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
                # 先创建一个临时的主 client 供 router 用
                from agent.llm_client import create_llm_client
                main_client = create_llm_client({
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
        from agent.hook_exec import set_aux_router_provider
        set_aux_router_provider(lambda: aux_llm_router)

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
            memory_retriever=self.memory_retriever,  # === Mem-T7 NEW ===
            team_bus=self.team_bus,  # === P4a-T7 NEW ===
            team_coordinator=self.team_coordinator,  # === P4a-T7 NEW ===
            team_name="main",  # === P4a-T7 NEW ===
            aux_llm_router=aux_llm_router,  # === batch2-T3 NEW ===
            plan_approval_callback=cli_plan_approval_callback,  # === PlanMode NEW ===
            stream_callback=stream_callback,  # === 04 NEW: 流式输出 ===
            ask_user_bridge=_make_ask_user_bridge(),  # ask_user CLI 桥接
            checkpoint_manager=self.checkpoint_mgr,  # === Checkpoint NEW ===
            permission_mode=self.config.get("security", {}).get("permission_mode", "default"),  # === B2 NEW: 透传给 AIAgent ===
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

        return agent

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
        conv = [m for m in msgs if m.get("role") != "system"]
        # 清理冗余摘要占位（保留最近一个）——压缩频率修复前的会话可能有几十个
        # "[之前的对话已自动总结]" 占位，全注入上下文会撑爆且混乱
        conv = _cleanup_redundant_summaries(conv)
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

def _make_approval_callback():
    """创建审批 callback（破坏性命令 + 写入路径都用这个）。

    callback 接收字符串,根据内容自动判断是命令还是路径,显示不同 prompt。
    同意后:
    - 命令 → 加入 ~/.OmniMate/approved_commands.json
    - 路径 → 加入 ~/.OmniMate/approved_paths.json
    跨会话不再询问相同项。
    """
    def callback(item: str) -> bool:
        # 启发式判断:含路径分隔符或 ~ 开头 → 路径,否则 → 命令
        is_path = (
            "/" in item or "\\" in item or item.startswith("~")
            or item[1:3] == ":\\" if len(item) >= 3 else False
        )
        if is_path:
            console.print(f"[yellow]⚠️ 即将写入路径(白名单外)：[/yellow]")
            console.print(f"[bold]{item}[/bold]")
            try:
                answer = console.input(
                    "[bold]允许？(y/N):[/bold] [dim]（同意后整个父目录不再询问）[/dim] ",
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return False
            return answer in ("y", "yes")
        else:
            console.print(f"[yellow]⚠️ 即将执行破坏性命令：[/yellow]")
            console.print(f"[bold]{item}[/bold]")
            try:
                answer = console.input(
                    "[bold]允许执行？(y/N):[/bold] [dim]（同意后此命令不再询问）[/dim] ",
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return False
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
    """Plan Mode 审批回调：打印计划 + 询问 y/N/edit。

    返回 (approved: bool, feedback: str)。
    - y/yes → (True, "")
    - edit → 收集一行 feedback → (False, feedback)
    - 其他（n/空/任意）→ (False, "用户拒绝")
    """
    print("\n" + "=" * 60)
    print("Agent 提交了以下计划，请审批：")
    print("=" * 60)
    print(plan)
    print("=" * 60)
    print("\n批准？[y/N/edit]")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False, "用户中断输入"
    if choice in ("y", "yes"):
        return True, ""
    if choice == "edit":
        print("请输入修订建议（单行）：")
        try:
            feedback = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return False, "用户中断输入"
        return False, feedback or "用户未输入修订建议"
    return False, "用户拒绝（未提供原因）"


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

    return False


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
        summary = _summarize_conversation(after, rt.agent.llm_client)
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
        "[cyan]/memory[/cyan]    查看记忆\n"
        "[cyan]/sessions[/cyan]  列出历史会话\n"
        "[cyan]/resume[/cyan]    恢复历史会话（/resume [序号]）\n"
        "[cyan]/search[/cyan]    搜索历史对话（/search <关键词>）\n"
        "[cyan]/usage[/cyan]     显示工具用量\n"
        "[cyan]/stats[/cyan]     会话统计（跨会话聚合）\n"
        "[cyan]/model[/cyan]     切换模型（/model [name]）\n"
        "[cyan]/plan[/cyan]      进入计划模式（/plan off 强制退出）\n"
        "[cyan]/permission[/cyan]  查看或切换权限模式（/permission [default|bypass|acceptEdits]）\n"
        "[cyan]/agents[/cyan]   列出自定义子代理（来自 ~/.OmniMate/agents/*.md）\n"
        "[cyan]/approved[/cyan]  管理审批白名单\n"
        "[cyan]/rewind[/cyan]    回滚到某个 checkpoint（恢复文件 + 可选对话）\n"
        "[cyan]/handoff[/cyan]   会话移交（save/load/list/show/delete/export/import）\n"
        "[cyan]/help[/cyan]      显示本帮助\n"
        "[cyan]/quit[/cyan]      退出\n\n"
        "[dim]输入 /技能名 触发对应技能[/dim]",
        border_style="blue",
    ))


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
            # 先检查是否是技能束命令
            cmd_name = user_input.split()[0]
            if cmd_name in rt.bundle_commands:
                pass  # 走技能束触发逻辑
            elif cmd_name in rt.skill_commands:
                pass  # 走技能触发逻辑
            elif _handle_command(user_input, rt):
                if rt.quit_requested:
                    break  # /quit 请求：走正常退出（rt.shutdown()）
                continue

        # 2. 检查是否触发技能束
        cmd_name = user_input.split()[0] if " " in user_input else user_input
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
            response = rt.agent.run_conversation(user_input)
            # 流式模式(stream_callback 已设)的内容已经在 run_conversation 过程中显示,
            # 不再重复 print。非流式模式(无 callback)才 print response。
            # 但 LLM 失败/预算耗尽等兜底文案不走流式（没有内容增量），必须显示，
            # 否则用户会看到"没反应就断了"（之前莫名断开的根因）。
            if not getattr(rt.agent, "_stream_callback", None):
                console.print(response)
            elif response and response.startswith(
                ("[已被用户中断", "[LLM 调用失败", "[已达最大迭代次数")
            ):
                console.print(f"[yellow]{response}[/yellow]")

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
        response = rt.agent.run_conversation(message)
        print(response)
        if rt.session_store and rt.session_id:
            rt.session_store.append_message(rt.session_id, "assistant", response)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        logger.exception("one-shot 运行错误")
    finally:
        # === P2b-T6 NEW: 退出前清理后台任务 ===
        rt.shutdown()
