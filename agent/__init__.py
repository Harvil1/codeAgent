"""AIAgent 主类：整个系统的心脏。

一个 agent 实例 = 一个会话（session）。
会话之间通过 session_id 持久化到 SQLite。

核心循环（run_conversation）：
    while 预算还有 and 没中断:
        组装 messages（system + history）
        调用 LLM
        如果有 tool_calls：
            执行每个工具，结果追加到历史
            continue
        否则：
            返回最终响应

关键设计：
- 同步 while 循环（不异步，便于推理调试）
- system prompt 构建一次后缓存（保护 prompt cache）
- 中断是协作式的（设置标志，循环自己检查）
- 预算耗尽后给一次 grace call 让模型说最后一句话
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from agent.budget import IterationBudget
from agent.context_pipeline import CompressionSessionState, strip_internal_fields, reset_offload_decisions
from agent.context_compressor import reset_compact_circuit_breaker
from agent.prompt_builder import build_system_prompt
from tools.registry import registry

logger = logging.getLogger(__name__)


# ============================================================================
# CCAR8 Task 11：ephemeral 注入纯函数（可独立测试）
# ============================================================================
# 设计原则：
# - 所有 ephemeral 消息走 user 角色，**永不修改 system prompt**（保护 cache）
# - helper 返回带 `_ephemeral=True` 的 dict，调用方 append 到 messages（发给 LLM）
#   但**不追加到 conversation_history**（不进持久化）
# - fail-open：channel/mailbox 任一异常只 log warning，不影响主循环

def _build_goal_continue_message(goal_state) -> Optional[dict]:
    """构造 `<continue_goal>` ephemeral user 消息驱动下一轮。

    Args:
        goal_state: GoalState 实例（active 时才注入）；None 返回 None

    Returns:
        带 `_ephemeral=True` 的 user 消息 dict，或 None（不注入）
    """
    if goal_state is None:
        return None
    if goal_state.status != "active":
        return None
    return {
        "role": "user",
        "content": (
            f'<continue_goal objective="{goal_state.objective}" '
            f'iteration="{goal_state.iteration_count}" />'
        ),
        "_ephemeral": True,
    }


def _build_channel_injection(inbox) -> Optional[dict]:
    """构造 `<channel_push>` ephemeral user 消息（MCP notifications）。

    从 inbox 取 unconsumed，注入后 mark_consumed（fail-open）。

    Args:
        inbox: ChannelInbox 实例；None 返回 None

    Returns:
        ephemeral user 消息 dict，或 None（无消息/无 inbox）
    """
    if inbox is None:
        return None
    try:
        unconsumed = inbox.unconsumed()
        if not unconsumed:
            return None
        digest = inbox.format_digest(unconsumed)
        msg = {
            "role": "user",
            "content": (
                f'<channel_push count="{len(unconsumed)}">\n'
                f'{digest}\n</channel_push>'
            ),
            "_ephemeral": True,
        }
        inbox.mark_consumed([m["id"] for m in unconsumed])
        return msg
    except Exception as e:
        logger.warning("channel 注入 fail-open: %s", e)
        return None


def _build_mail_injection(mailbox, agent_name: str) -> Optional[dict]:
    """构造 `<mail>` ephemeral user 消息（teammate 邮件）。

    从 mailbox 取 unread，注入后 mark_read（fail-open）。

    Args:
        mailbox: Mailbox 实例；None 返回 None
        agent_name: 当前 agent 名（查收件人）；空串返回 None

    Returns:
        ephemeral user 消息 dict，或 None（无邮件/无 mailbox）
    """
    if mailbox is None or not agent_name:
        return None
    try:
        unread = mailbox.check_unread(agent_name)
        if not unread:
            return None
        digest_lines = []
        for m in unread:
            sender = m.get("from", "?")
            ts = m.get("ts", "")
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            kind = m.get("kind", "message")
            digest_lines.append(f"[{ts}] from={sender} kind={kind}\n{content}")
        digest = "\n\n".join(digest_lines)
        msg = {
            "role": "user",
            "content": (
                f'<mail unread="{len(unread)}">\n'
                f'{digest}\n</mail>'
            ),
            "_ephemeral": True,
        }
        mailbox.mark_read(agent_name, [m["id"] for m in unread])
        return msg
    except Exception as e:
        logger.warning("mailbox 注入 fail-open: %s", e)
        return None


class AIAgent:
    """核心 Agent 类。一个实例对应一个会话。"""

    # sentinel：_call_llm_with_escalation 触发了 reactive_compact，主循环需重试本轮
    _REACTIVE_RETRY = object()

    def __init__(
        self,
        *,
        base_url: str = None,
        api_key: str = None,
        auth_token: str = None,            # DeepSeek Anthropic 端点用 Bearer 认证
        effort_level: str = None,          # 思考强度: max / high / medium / low
        model: str = "deepseek-chat",
        model_format: str = "openai",      # openai / anthropic
        fallback_model: str = None,
        max_iterations: int = 200,
        enabled_toolsets: list = None,
        session_id: str = None,
        system_prompt_override: str = None,
        memory_store=None,
        memory_manager=None,
        session_store=None,
        omnimate_home=None,
        on_tool_call=None,
        on_response=None,
        config: dict = None,
        hooks_registry=None,   # === P2-T6 NEW ===
        bg_manager=None,       # === P2b-T5 NEW ===
        cron_scheduler=None,   # === P2c-T4 NEW ===
        memory_retriever=None,  # === Mem-T5 NEW ===
        team_bus=None,           # === P4a-T6 NEW ===
        team_coordinator=None,   # === P4a-T6 NEW ===
        team_name=None,          # === P4a-T6 NEW ===
        spawn_depth: int = 0,    # === P4b-T2 NEW ===
        aux_llm_router=None,     # === batch2-T3 NEW ===
        plan_approval_callback=None,  # === PlanMode NEW ===
        stream_callback=None,    # === 04 NEW: 流式输出回调 ===
        ask_user_bridge=None,    # ask_user CLI 桥接（渲染问题+读选择）
        checkpoint_manager=None, # === Checkpoint NEW: 文件快照/回滚（对齐 Claude Code）===
        permission_mode: str = "default",  # === B2 NEW: default | bypassPermissions ===
        initial_messages: list = None,  # === Task H NEW: fork 子代理初始 messages ===
        omit_project_memory: bool = False,  # === Task N NEW: 子代理跳过项目 OMNIMATE.md ===
        trace_sink=None,  # === CCAR8 Task 5 NEW: 本地 trace sink ===
        goal_state=None,  # === CCAR8 Task 11 NEW: 目标驱动状态机 ===
        channel_inbox=None,  # === CCAR8 Task 11 NEW: MCP notification 收件箱 ===
        mailbox=None,  # === CCAR8 Task 11 NEW: teammate 异步邮箱 ===
        agent_name: str = "main",  # === CCAR8 Task 11 NEW: 当前 agent 名（mailbox 收件人）===
    ):
        """
        参数：
            base_url: LLM API 的 base URL（兼容 OpenAI 格式）
            api_key: API 密钥
            model: 模型名（如 "deepseek-chat"）
            max_iterations: 工具调用迭代上限（防止跑飞）
            enabled_toolsets: 启用的工具集（如 ["core"]）
            session_id: 会话 ID（用于持久化）
            system_prompt_override: 跳过默认 prompt 构建（子代理场景）
            memory_store: 记忆存储（MEMORY.md + USER.md）
            on_tool_call: 工具调用回调（CLI 用它打印进度）
            on_response: 最终响应回调
            initial_messages: fork 子代理初始 conversation_history（Task H），
                None 时从空开始（默认）；非 None 时用传入的 messages 初始化
        """
        # 创建 LLM client（根据 model_format 选 OpenAI 兼容或 Anthropic 原生）
        from agent.llm_client import create_llm_client
        model_config = {
            "format": model_format,
            "base_url": base_url,
            "api_key": api_key,
            "auth_token": auth_token,
            "effort_level": effort_level,
            "model": model,
        }
        self.llm_client = create_llm_client(model_config)
        self.base_url = base_url
        self.api_key = api_key
        self.auth_token = auth_token
        self.model = model
        self.model_format = model_format
        self.effort_level = effort_level
        self.fallback_model = fallback_model

        # 备用 LLM client（主 client 重试耗尽时切换）
        self.fallback_llm_client = None
        if fallback_model:
            fb_config = dict(model_config)
            fb_config["model"] = fallback_model
            try:
                self.fallback_llm_client = create_llm_client(fb_config)
            except Exception as e:
                logger.warning("创建 fallback LLM client 失败: %s", e)

        self.max_iterations = max_iterations
        self.enabled_toolsets = enabled_toolsets or ["core"]
        self.session_id = session_id
        # 阶段 5 NEW: 创建会话级 env 文件并暴露路径给 hook（对齐 Claude Code CLAUDE_ENV_FILE）
        self._session_env_path = None
        self._setup_session_env_file()
        self.memory_store = memory_store
        self.memory_manager = memory_manager
        self.session_store = session_store
        # C3 修复：omnimate_home=None 时解析为默认 ~/.OmniMate，避免下游 TypeError
        if omnimate_home is not None:
            self.omnimate_home = omnimate_home
        else:
            from constants import get_omnimate_home
            self.omnimate_home = get_omnimate_home()
        self.on_tool_call = on_tool_call
        self.on_response = on_response

        # 预算（每个会话独立）
        self.iteration_budget = IterationBudget(max_iterations)

        # 中断标志（Ctrl+C 时被置为 True）
        self._interrupt_requested = False

        # 预算耗尽后的"最后一次机会"
        self._budget_grace_call = False
        # S8 fix 配套：grace 只能触发一次（防 dispatch→grace→dispatch 无限循环）
        self._grace_triggered = False

        # 系统提示：会话开始时构建一次，后续缓存
        self._system_prompt_built = system_prompt_override is not None
        # 05 NEW: 三层结构缓存
        self._stable_prompt: Optional[str] = system_prompt_override
        self._context_prompt: Optional[str] = ""
        # Task N NEW: 自定义子代理可跳过项目 OMNIMATE.md 注入（omitClaudeMd）
        self.omit_project_memory = bool(omit_project_memory)

        # 对话历史（不包含 system prompt，system 单独传）
        # Task H NEW: initial_messages 支持（fork 子代理继承父前缀）
        self.conversation_history: list = list(initial_messages) if initial_messages else []

        # 上下文压缩配置
        self.compression_enabled = True
        self._compression_attempts = 0
        # 完整 config（用于 context 阈值、hooks 等开关）
        self.config: dict = config or {}

        # === P2-T6 NEW: hooks 系统 ===
        self.hooks_registry = hooks_registry
        self._stop_fire_count = 0
        self._stop_hook_forced = False  # STOP hook 触发后让主循环继续（替代内联 continue）

        # === P2b-T5 NEW: 后台任务管理器 ===
        self.bg_manager = bg_manager

        # === P2c-T4 NEW: cron 调度器 ===
        self.cron_scheduler = cron_scheduler

        # === Mem-T5 NEW: memory 检索器 ===
        self.memory_retriever = memory_retriever
        # 缓存 memory 索引（会话内 frozen，保护 prompt cache）
        self._cached_memory_index = ""
        if self.memory_store:
            try:
                # 检索用完整索引（不受注入截断影响，能按需检索全部记忆）
                self._cached_memory_index = self.memory_store.full_index_text()
            except Exception as e:
                logger.warning("缓存 memory 索引失败: %s", e)

        # === P4a-T6 NEW: team 消息总线 + 协调器 ===
        self.team_bus = team_bus
        self.team_coordinator = team_coordinator
        self.team_name = team_name

        # === P4b-T2 NEW: idle 标志 + spawn 深度 ===
        self._idle_requested = False
        self.spawn_depth = spawn_depth

        # 上下文压缩会话状态（每实例一份，跨轮次追踪 L4 cooldown/计数）
        self._compress_session_state = CompressionSessionState()
        # 改造点 ①：新会话清空落盘决策（避免跨会话泄漏，保护 prompt cache）
        reset_offload_decisions()
        # 改造点 ②：新会话重置摘要熔断器（避免跨会话污染失败计数）
        reset_compact_circuit_breaker()
        # 改造点 ③：新会话重置 cache 监控状态（避免跨会话污染 baseline）
        try:
            from agent.cache_monitor import reset_cache_monitor
            reset_cache_monitor()
        except Exception as e:
            logger.debug("reset_cache_monitor 失败（fail-open）: %s", e)
        # CCAR4 Task A：从 config 读 diff 文件 LRU 上限，传给 cache_monitor
        try:
            from agent.cache_monitor import set_diff_limit
            _diff_limit = (
                (config or {}).get("context", {}).get(
                    "max_cache_break_diff_files", 100,
                )
            )
            set_diff_limit(_diff_limit)
        except Exception as e:
            logger.debug("set_diff_limit 失败（fail-open）: %s", e)

        # === batch2-T3 NEW: 辅助 LLM 路由器 ===
        self.aux_llm_router = aux_llm_router

        # === CCAR8 Task 5 NEW: trace sink hook 接入（fail-open）===
        # 把 sink 接到 6 个 hook 点（pre/post_llm_call + post_tool_use/failure
        # + subagent_start/stop）。hook 内部已 try/except，写盘失败只 log。
        # 防 Silent-Dead-Code：AIAgent 构造时必须真调 _register_trace_hooks，
        # 否则单元测试过但生产路径不 emit trace（CLAUDE.md 教训）。
        self._trace_sink = trace_sink
        if trace_sink is not None and hooks_registry is not None:
            try:
                from agent.trace import _register_trace_hooks
                _register_trace_hooks(hooks_registry, trace_sink)
            except Exception as e:
                logger.warning("trace hook 注册失败（不影响主流程）: %s", e)

        # === PlanMode NEW: 计划模式状态 + 审批回调 ===
        # plan_mode=True 时下一轮起切换到 ["plan"] 工具集（只读）
        # plan_approval_callback(plan: str) -> (approved: bool, feedback: str)
        # None 表示自动批准（测试/库用法）
        self.plan_mode: bool = False
        self.plan_approval_callback = plan_approval_callback
        # ask_user 桥接（CLI 渲染问题+读选择 / GUI HTTP）；None = 无桥接（fail-fast）
        self.ask_user_bridge = ask_user_bridge
        # Checkpoint：文件快照/回滚（编辑工具通过 _checkpoint_track 追踪修改文件）
        self.checkpoint_manager = checkpoint_manager
        # === B2 NEW: 权限模式（default | bypassPermissions），由 CLI 启动参数或 /permission 切换 ===
        self.permission_mode = permission_mode
        # 压缩后重注入：最近读过的文件 + 加载的技能（对齐 Claude Code）
        self._recent_read_files: list = []
        self._recent_skills: list = []
        # 上下文管理提示：接近上限时建议主动 /compact /new（对齐 Claude Code context rot）
        self._context_tip_shown = False

        # === B1 NEW: vision client（image_analyze / image_ocr 共用） ===
        # 默认 None；由 RuntimeContext 根据 config 注入，或测试时手工注入。
        self._vision_client = None

        # === batch1-T2 NEW: LLM 用量统计（prompt cache 记账）===
        self._llm_usage_stats = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
        }

        # === batch1-T4 NEW: 子 agent 追踪（中断传播）===
        self._children: list = []

        # === ⑪b NEW: 自动心跳桥 ===
        # spawned worker 每次工具调用后自动 bump task.last_heartbeat_at
        # 主 agent 无 OMNIMATE_KANBAN_TASK env，no-op
        try:
            from agent.team.auto_heartbeat import register as _register_auto_heartbeat
            _register_auto_heartbeat(self.hooks_registry)
        except Exception as e:
            logger.warning("注册 auto_heartbeat hook 失败（不影响主流程）: %s", e)

        # === 04 NEW: 流式输出回调 ===
        # None 时走非流式（向后兼容老测试）；非 None 时每收到一个 LLM chunk 就调用。
        # 回调签名：callback(event: dict) -> None
        # event["type"]: "content" | "tool_call_start" | "done"
        self._stream_callback = stream_callback

        # === P0-3 NEW: max_tokens 升级机制 ===
        # finish_reason=length 时先升 max_tokens 重试，避免直接续写打断思路。
        # 整个会话复用；升级是幂等的（最多升一次）。
        from agent.llm_retry import MaxTokensEscalator
        self._max_tokens_escalator = MaxTokensEscalator()

        # === 韧性状态：reactive_compact 已触发标记 ===
        # Task D：reactive_compact 改多次触发（冷却 60s + 上限 5 次/会话），
        # gate 逻辑下沉到 reactive_compact 内部（session_state 字段），
        # 这里不再维护独立的 _reacted flag。保留字段做向后兼容（测试可能读）。
        self._reacted: bool = False

        # === 失败重试检测 ===
        # 连续 N 次工具调用失败 → 注入提醒,防止死循环
        self._tool_failure_streak: int = 0
        self._last_tool_error: str = ""
        self._failure_threshold: int = 3

        # === CCALS-P0-2 NEW: 任务级反思引擎 ===
        # 每次任务正常结束时异步触发：用 aux_llm 从轨迹提炼 3 类经验写入 memory_store。
        # aux_llm 不可用时降级跳过；config["reflection"]["enabled"]=False 可关闭。
        self._reflection_enabled = (
            (config or {}).get("reflection", {}).get("enabled", True)
        )
        # CCALS-P0-2 节流：避免连续对话起 N 个反思线程烧 token
        # - 任意时刻最多 1 个反思在跑（_active_reflections）
        # - 距上次启动不足 N 轮时跳过（_last_reflection_turn + cooldown_turns）
        self._reflection_lock = __import__("threading").Lock()
        self._active_reflections = 0
        self._last_reflection_turn = -1
        self._reflection_cooldown_turns = int(
            (config or {}).get("reflection", {}).get("cooldown_turns", 3)
        )

        # === CCAR8 Task 11 NEW: Goal/Channel/Mailbox 三件套 ===
        # goal_state：goal-driven 自动多轮的状态机（None=未启用）
        # channel_inbox：MCP server notifications 落地 inbox
        # mailbox：teammate 异步邮箱（_agent_name 是收件人）
        # 设计：所有三个都用 setter（构造参数也支持，最灵活）
        self._goal_state = goal_state
        self._channel_inbox = channel_inbox
        self._mailbox = mailbox
        self._agent_name = agent_name
        # 待注入 ephemeral 消息队列：goal continue 跨轮注入用
        # 设计：主循环里 goal continue 时把 ephemeral 消息塞这里（不进 history），
        # 下一轮 _assemble_turn_messages 末尾消费并清空。这样既让 LLM 看到，
        # 又不污染 conversation_history（保护持久化 + prompt cache）
        self._pending_ephemeral_messages: list = []
        # CCAR10 Task 2: 降级 snapshot 一次性注入标志
        # 无 aux_llm_router 时主循环降级回 snapshot 索引注入；对齐旧"会话级 frozen"
        # 语义，注入一次后本会话不再重复注入（避免每轮重复塞同一索引）
        self._snapshot_injected: bool = False

    def cleanup(self):
        """清理 agent 持有的资源（调用方：RuntimeContext.shutdown）。

        幂等：多次调用安全。每个子清理都包 try/except，互不影响。
        """
        # 阶段 5 NEW: 清理 session env 文件 + unset 环境变量
        try:
            if getattr(self, "_session_env_path", None):
                self._session_env_path.unlink(missing_ok=True)
                self._session_env_path = None
            if "OMNIMATE_ENV_FILE" in os.environ:
                del os.environ["OMNIMATE_ENV_FILE"]
        except Exception as e:
            logger.warning("清理 session env 文件失败: %s", e)

        # X3 fix: 关闭 LLM client（HTTP 连接池），避免进程退出前泄漏
        for client_attr in ("llm_client", "fallback_llm_client", "_vision_client"):
            client = getattr(self, client_attr, None)
            if client is not None:
                try:
                    close_fn = getattr(client, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as e:
                    logger.warning("关闭 %s 失败（忽略）: %s", client_attr, e)

    def _setup_session_env_file(self):
        """会话启动时创建 .session/{session_id}.env 并设 OMNIMATE_ENV_FILE 环境变量。

        SessionStart hook 执行时能从 os.environ 读到这个路径，
        往里写 `export K=V` 行。terminal_tool 后续会 merge。
        """
        try:
            from constants import session_env_file
            env_path = session_env_file(self.session_id)
            env_path.parent.mkdir(parents=True, exist_ok=True)
            if not env_path.exists():
                env_path.touch()
            os.environ["OMNIMATE_ENV_FILE"] = str(env_path)
            self._session_env_path = env_path
        except Exception as e:
            logger.warning("创建 session env file 失败: %s", e)
    def interrupt(self):
        """请求中断（由 CLI 的 Ctrl+C 处理器调用）。

        协作式中断：不直接杀线程（可能损坏消息历史），而是设置标志。
        中断会传播到所有活跃子 agent。
        """
        self._interrupt_requested = True
        # 传播到子 agent
        for child in self._children:
            try:
                child.interrupt()
            except Exception as e:
                logger.warning("子 agent 中断失败: %s", e)

    def _record_llm_usage(self, response) -> None:
        """记录一次 LLM 调用的 token 用量（batch1-T2）。"""
        self._llm_usage_stats["total_calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        try:
            self._llm_usage_stats["total_prompt_tokens"] += (
                getattr(usage, "prompt_tokens", 0) or 0
            )
            self._llm_usage_stats["total_completion_tokens"] += (
                getattr(usage, "completion_tokens", 0) or 0
            )
            # prompt cache 相关（DeepSeek / OpenAI / Anthropic 都可能有）
            self._llm_usage_stats["total_cache_read_tokens"] += (
                getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cache_read_input_tokens", 0)
                or 0
            )
            self._llm_usage_stats["total_cache_creation_tokens"] += (
                getattr(usage, "prompt_cache_miss_tokens", 0)
                or getattr(usage, "cache_creation_input_tokens", 0)
                or 0
            )
        except Exception as e:
            logger.debug("记录 LLM usage 失败（fail-open）: %s", e)

    # ------------------------------------------------------------------
    # CCAR8 Task 11：Goal/Channel/Mailbox setter + goal 持久化路径
    # ------------------------------------------------------------------
    # 设计：构造参数 + setter 都支持（构造参数用于子代理场景，
    # setter 用于 CLI 启动后按需注入——如 /goal 命令触发后才创建 GoalState）

    def set_goal_state(self, goal_state) -> None:
        """注入 GoalState（None=清除）。"""
        self._goal_state = goal_state

    def set_channel_inbox(self, inbox) -> None:
        """注入 ChannelInbox（None=清除）。"""
        self._channel_inbox = inbox

    def set_mailbox(self, mailbox, agent_name: str = None) -> None:
        """注入 Mailbox。agent_name 为空时保留原值。"""
        self._mailbox = mailbox
        if agent_name:
            self._agent_name = agent_name

    def _goal_state_path(self):
        """goal 持久化路径：~/.OmniMate/.goal/current.json。"""
        from pathlib import Path
        return Path(self.omnimate_home) / ".goal" / "current.json"

    def _check_all_goal_tasks_done(self) -> bool:
        """goal 的所有 task_ids 是否全部 completed。

        Task 12 替换原占位（恒 False）为真实 TaskStore 查询：
        - 无 goal_state → False
        - 调 agent.goal.check_all_tasks_done(goal_state)
        """
        if self._goal_state is None:
            return False
        from agent.goal import check_all_tasks_done
        try:
            return check_all_tasks_done(self._goal_state)
        except Exception as e:
            logger.warning("_check_all_goal_tasks_done 查询失败（fail-open False）: %s", e)
            return False

    def _extract_turn_tokens(self, response) -> int:
        """从 response.usage 提取本轮总 token 数（prompt + completion）。

        用于 goal_state.evaluate_after_turn 累加 token_budget。
        fail-open：无 usage 字段返回 0。
        """
        if response is None:
            return 0
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        try:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            return int(prompt) + int(completion)
        except Exception:
            return 0

    @staticmethod
    def _extract_cache_read(usage) -> int:
        """从 usage 提取 cache read tokens（兼容 dict/对象 + DeepSeek/Anthropic 字段名）。

        DeepSeek 用 prompt_cache_hit_tokens，Anthropic 用 cache_read_input_tokens。
        流式路径的 SimpleNamespace.usage 两个字段都塞（见 _call_llm_streaming 末尾），
        所以这里 or 短路兜底任一非零即可。
        """
        if not usage:
            return 0
        try:
            if isinstance(usage, dict):
                return (
                    usage.get("prompt_cache_hit_tokens", 0)
                    or usage.get("cache_read_input_tokens", 0)
                    or 0
                )
            return (
                getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cache_read_input_tokens", 0)
                or 0
            )
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 04 NEW: 流式调用 LLM
    # ------------------------------------------------------------------

    async def _call_llm_streaming(self, *, messages, tools):
        """async 流式调用 LLM，每收到一个 chunk 调用 stream_callback。

        流式失败时 fallback 到非流式重试（带备用 client）。
        返回值与非流式路径完全兼容（SimpleNamespace 包装的 OpenAI 响应结构），
        让 _record_llm_usage / hook / tool_calls 处理代码不用改。

        注意：本方法是 **async function 返回 response 对象**（不是 async generator）。
        流式事件通过 stream_callback 回调报告，最终结果用 return 返回。

        stream_callback 事件类型：
            {"type": "content", "delta": str, "accumulated": str}  # 文本增量
            {"type": "tool_call_start", "name": str, "id": str}    # 工具调用开始
            {"type": "done", "finish_reason": str}                  # 流结束
        """
        from types import SimpleNamespace
        full_content = ""
        tool_call_buffers: dict[int, dict] = {}  # idx → {id, name, arguments}
        final_usage = None
        finish_reason = "stop"
        reasoning_content = None   # DeepSeek thinking(工具调用回传需要)
        thinking_signature = None

        try:
            # 从 config 读 max_tokens（用户在 settings.json llm 块配 "max_tokens": 8192）
            # 不配就不传，让 API 用默认值（换模型不用改代码）
            _extra = {}
            _cfg_mt = (
                (self.config or {}).get("model", {}).get("max_tokens")
                or (self.config or {}).get("llm", {}).get("max_tokens")
            )
            if _cfg_mt:
                _extra["max_tokens"] = _cfg_mt
            async for delta in self.llm_client.chat_completions_stream(
                messages, tools=tools, **_extra,
            ):
                # 内容流式
                delta_text = delta.get("content") or ""
                if delta_text:
                    full_content += delta_text
                    if self._stream_callback is not None:
                        try:
                            self._stream_callback({
                                "type": "content",
                                "delta": delta_text,
                                "accumulated": full_content,
                            })
                        except Exception as cb_err:
                            logger.warning(
                                "stream_callback(content) 异常（忽略）: %s", cb_err
                            )

                # 工具调用增量累积
                for tc in delta.get("tool_calls") or []:
                    idx = getattr(tc, "index", 0)
                    buf = tool_call_buffers.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        buf["id"] = tc_id
                    func = getattr(tc, "function", None)
                    if func is not None:
                        fname = getattr(func, "name", None)
                        if fname:
                            buf["name"] = fname
                        fargs = getattr(func, "arguments", None)
                        if fargs:
                            buf["arguments"] += fargs
                    # 第一次拿到 name 时通知 callback
                    if buf["name"] and not buf.get("_notified"):
                        buf["_notified"] = True
                        if self._stream_callback is not None:
                            try:
                                self._stream_callback({
                                    "type": "tool_call_start",
                                    "name": buf["name"],
                                    "id": buf["id"],
                                })
                            except Exception as cb_err:
                                logger.warning(
                                    "stream_callback(tool_call_start) 异常: %s",
                                    cb_err,
                                )

                # 最后一个 chunk 的 finish_reason / usage / thinking
                if delta.get("finish_reason"):
                    finish_reason = delta["finish_reason"]
                if delta.get("usage"):
                    final_usage = delta["usage"]
                # DeepSeek thinking 提取(工具调用时后续请求需回传)
                if delta.get("reasoning_content"):
                    reasoning_content = delta["reasoning_content"]
                if delta.get("thinking_signature"):
                    thinking_signature = delta["thinking_signature"]
        except Exception as stream_err:
            # 流式失败：先通知 callback，再 fallback 到非流式重试
            logger.warning(
                "流式调用失败，fallback 到非流式重试: %s", stream_err
            )
            from agent.llm_retry import call_with_retry
            response = await call_with_retry(
                self.llm_client,
                messages,
                tools=tools,
                fallback_llm_client=self.fallback_llm_client,
                config=self.config,
            )
            # 流式回调已经错过，但至少把完整内容回放给 callback
            choice_msg = response.choices[0].message
            if choice_msg.content and self._stream_callback is not None:
                try:
                    self._stream_callback({
                        "type": "content",
                        "delta": choice_msg.content,
                        "accumulated": choice_msg.content,
                    })
                except Exception:
                    pass
            return response

        # 合成 tool_calls 列表（按 idx 排序，过滤掉没 name 的）
        tool_calls_out = []
        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            if not buf["name"]:
                continue
            tool_calls_out.append(SimpleNamespace(
                id=buf["id"],
                type="function",
                function=SimpleNamespace(
                    name=buf["name"],
                    arguments=buf["arguments"] or "{}",
                ),
            ))

        # === P0-3 NEW: max_tokens 升级重试 ===
        # finish_reason=length 表示输出被 max_tokens 截断。
        # DeepSeek-reasoner 纯 thinking（content 空 + reasoning 有值）也算截断——
        # thinking 用完 max_tokens，content 没空间输出，但 finish_reason 可能是 "stop"。
        # 策略：先升级 max_tokens 重试（非流式，避免重复发 partial content），
        # 升级后仍空才放弃，让主循环处理。
        is_pure_thinking = (
            not full_content and not tool_calls_out and bool(reasoning_content)
        )
        if (
            (finish_reason == "length" or is_pure_thinking)
            and self._max_tokens_escalator is not None
            and not self._max_tokens_escalator.has_escalated
        ):
            new_max = self._max_tokens_escalator.escalate()
            trigger_reason = "纯 thinking（content 空）" if is_pure_thinking else "finish_reason=length"
            logger.info(
                "max_tokens 截断（%s），升级到 %d 重试", trigger_reason, new_max
            )
            try:
                from agent.llm_retry import call_with_retry
                retried = await call_with_retry(
                    self.llm_client,
                    messages,
                    tools=tools,
                    fallback_llm_client=self.fallback_llm_client,
                    max_tokens=new_max,
                    config=self.config,
                )
                retried_choice = retried.choices[0]
                retried_msg = retried_choice.message
                # 用重试结果覆盖（重试是完整响应）
                finish_reason = (
                    getattr(retried_choice, "finish_reason", None) or "stop"
                )
                full_content = retried_msg.content or ""
                if getattr(retried_msg, "tool_calls", None):
                    tool_calls_out = list(retried_msg.tool_calls)
                # 重试结果回放给 callback（同 fallback 路径模式）
                if retried_msg.content and self._stream_callback is not None:
                    try:
                        self._stream_callback({
                            "type": "content",
                            "delta": retried_msg.content,
                            "accumulated": retried_msg.content,
                        })
                    except Exception:
                        pass
                # 更新 usage
                if getattr(retried, "usage", None) is not None:
                    u = retried.usage
                    final_usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                        "cache_read": (
                            getattr(u, "cache_read_input_tokens", 0)
                            or getattr(u, "prompt_cache_hit_tokens", 0)
                        ),
                        "cache_creation": (
                            getattr(u, "cache_creation_input_tokens", 0)
                            or getattr(u, "prompt_cache_miss_tokens", 0)
                        ),
                    }
            except Exception as esc_err:
                logger.warning(
                    "max_tokens 升级重试失败（沿用截断响应）: %s", esc_err
                )

        # 通知 done
        if self._stream_callback is not None:
            try:
                self._stream_callback({
                    "type": "done",
                    "finish_reason": finish_reason,
                })
            except Exception:
                pass

        # 合成 OpenAI 兼容 response（让 _record_llm_usage / hook 等不用改）
        message = SimpleNamespace(
            content=full_content if full_content else None,
            tool_calls=tool_calls_out if tool_calls_out else None,
            reasoning_content=reasoning_content,
            thinking_signature=thinking_signature,
        )
        usage_ns = None
        if final_usage is not None:
            usage_ns = SimpleNamespace(
                prompt_tokens=final_usage.get("prompt_tokens", 0),
                completion_tokens=final_usage.get("completion_tokens", 0),
                prompt_cache_hit_tokens=final_usage.get("cache_read", 0),
                cache_read_input_tokens=final_usage.get("cache_read", 0),
                prompt_cache_miss_tokens=final_usage.get("cache_creation", 0),
                cache_creation_input_tokens=final_usage.get("cache_creation", 0),
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )],
            usage=usage_ns,
        )

    @property
    def llm_usage_stats(self) -> dict:
        """只读视图（副本）用于 /usage 展示。"""
        return dict(self._llm_usage_stats)

    def _get_system_prompt(self) -> str:
        """获取系统提示。第一次调用时构建，后续返回缓存。

        缓存是为了保护 LLM provider 的 prompt cache。

        05 升级：分 stable/context 两层缓存，volatile 每次取最新。
        - stable：跨会话不变（身份、指导），几乎 100% 命中 prompt cache
        - context：单会话内不变（记忆/技能/OMNIMATE.md）
        - volatile：每轮可变（reminder），不入缓存
        """
        if not self._system_prompt_built:
            from agent.prompt_builder import build_system_prompt_layers
            layers = build_system_prompt_layers(
                memory_store=self.memory_store,
                memory_manager=self.memory_manager,
                enabled_toolsets=self.enabled_toolsets,
                omit_project_memory=self.omit_project_memory,
            )
            # stable + context 缓存，volatile 即时取
            self._stable_prompt = layers.stable
            self._context_prompt = layers.context
            self._system_prompt_built = True
        return "\n\n".join(p for p in (self._stable_prompt, self._context_prompt) if p)

    def _get_volatile_prompt(self) -> str:
        """每轮重建的 volatile 部分（05）。

        当前为空（task 状态走 task 工具自取，无系统级 reminder）。
        保留入口以便未来扩展，不入 stable/context 缓存——直接拼到 system prompt 末尾。
        """
        return ""

    def invalidate_system_prompt(self):
        """使缓存的 system prompt 失效。

        警告：这会让 prompt cache 失效，增加成本。
        只在上下文压缩等极端场景使用。

        05 优化：压缩后只重建 context 层（stable 不变，prompt cache 仍命中 stable 段）。
        """
        self._context_prompt = None
        self._system_prompt_built = False
        # _stable_prompt 保留（理论上整次会话内 stable 永不变）

    async def run_conversation(
        self, user_message: str, cancel_event=None,
    ) -> str:
        """处理一条用户消息，返回助手最终响应。

        这是整个系统的核心循环。Task D4: 改 async（核心循环 async 化）。
        Task K: 新增 cancel_event 参数（threading.Event）——被父代理 set 时，
        本循环在下一轮迭代开头立即退出，并返回 _extract_partial_result() 保留
        已完成的中间结果（借鉴 Claude Code extractPartialResult）。

        参数：
            user_message: 用户消息文本
            cancel_event: 可选的 threading.Event；None 时无 cancel 检查（向后兼容）

        重构后主循环结构（自顶向下阅读）：
            循环前：hook → drain 外部消息 → 开场记忆检索 → 追加 user history
            循环内：组装 messages → 压缩 → 工具集/警告/hook → 调 LLM → 分发 tool_calls
            循环后：兜底响应（预算耗尽/中断）

        具体逻辑见各 _xxx 辅助方法。
        """
        # === P4b final-fix C1: 每个 run_conversation 调用重置 idle 标志 ===
        # 同一 agent 实例在 autonomous lifecycle 多个 WORK 周期复用时，
        # 上一次 idle 请求不应泄漏到下一次调用。
        self._idle_requested = False
        # 每条用户消息重置迭代预算：预算只限制"这条消息"的循环轮数，
        # 避免长会话（多轮用户消息累计消耗）中途耗尽后静默断开
        # （曾导致：tool 调用后 consume() 返回 False → break → 无提示回"你:"）。
        self.iteration_budget.reset()
        # S8 配套：每条用户消息独立 grace 机会（_grace_triggered 防同一消息内循环）
        self._grace_triggered = False
        self._budget_grace_call = False

        # ---------- 循环前准备 ----------
        user_message = self._run_prompt_submit_hook(user_message)
        # S9 fix: 不在此 drain（每轮 while 内重新 drain，避免多轮 tool_calls 中途消息收不到）

        # Task 2.5: 旧 _initial_memory_recall 已删除——记忆注入统一走 CCAR10
        # ephemeral（见下方 _pending_ephemeral_messages 块），user 消息原样入 history
        # （保护 prompt cache + 不污染持久化）
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # === CCAR10 Task 2: 检索式记忆注入（仅主代理 spawn_depth==0）===
        # snapshot 已从 system prompt 退役——改走 ephemeral 注入（保护 prompt cache）。
        # 每轮一次：放在 user 输入刚进主循环处（不是工具循环里）。
        # 降级链：aux_llm_router 为 None → snapshot 索引（实例级 flag 只注入一次，
        # 对齐旧"会话级 frozen"语义防每轮重复注入同一索引）。
        if (self.spawn_depth == 0 and self.memory_store is not None
                and self._pending_ephemeral_messages is not None):
            from agent.memory_injection import (
                build_relevant_memories_message, reset_injection_cache,
                _fallback_snapshot_message,
            )
            # 每轮开头清缓存（防跨轮 LRU 串）
            reset_injection_cache()
            try:
                msg = None
                if self.aux_llm_router is not None:
                    # 检索路径：每轮按 query 用 aux_llm 检索 Top N
                    msg = await build_relevant_memories_message(
                        query=user_message, memory_store=self.memory_store,
                        aux_llm_router=self.aux_llm_router,
                    )
                elif not self._snapshot_injected:
                    # 降级路径：无 aux → snapshot 一次性注入（本会话仅一次）
                    msg = _fallback_snapshot_message(self.memory_store)
                    if msg is not None:
                        self._snapshot_injected = True
                if msg is not None:
                    self._pending_ephemeral_messages.append(msg)
            except Exception as e:
                logger.debug("记忆注入 fail-open: %s", e)

        system_prompt = self._get_system_prompt()

        # 延迟导入避免循环依赖
        from model_tools import get_tool_definitions, handle_function_call

        # ---------- 主循环 ----------
        api_call_count = 0
        turn_exit_reason = "normal"

        while (
            api_call_count < self.max_iterations
            and self.iteration_budget.remaining > 0
        ) or self._budget_grace_call:
            # 中断检查（协作式）
            if self._interrupt_requested:
                turn_exit_reason = "interrupted_by_user"
                self._interrupt_requested = False  # 清除标志
                break

            # === Task K: cancel_event 检查（父代理触发）===
            # 每轮开头检查（不是每条 message）——性能损耗小，且足够及时
            # 借鉴 Claude Code AbortController：父代理 set 时本子代理优雅退出
            # 返回 _extract_partial_result() 保留已完成的 assistant 消息
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "Task K: 子代理被 cancel_event 中断，返回 partial result"
                )
                return self._extract_partial_result()

            # 消耗预算（grace call 不消耗）
            if not self._budget_grace_call:
                if not self.iteration_budget.consume():
                    break
            else:
                # S8 fix: grace call 跑完后清标志（防无限循环）
                # 之前 _budget_grace_call 永远是 False（死代码），现在 dispatch 后会置 True，
                # 这里在 grace 进入时立刻清，保证 grace 只触发一次
                self._budget_grace_call = False

            # 组装 messages + 注入 bg/cron/team/plan_mode 等临时消息
            # S9 fix: 每轮重新 drain（之前只循环前 drain 一次，
            # 多轮 tool_calls 中途新到的 bg/cron/team 消息进不去 LLM 上下文）
            injected = self._drain_injected_messages()
            messages = self._assemble_turn_messages(system_prompt, injected)

            # 上下文压缩（接近 token 上限时触发，可能重建 system_prompt）
            messages, system_prompt, _ = await self._run_context_compression(
                messages, system_prompt,
            )

            # strip 内部字段（_timestamp 等）——必须在 compress 之后、发 LLM 之前。
            # 改造点 ④ Critical fix：之前 strip 在 _assemble_turn_messages 末尾（compress 之前），
            # 导致 time_based_clear_old_tool_results 拿不到 _timestamp，整个功能在生产里是死代码。
            # 现在挪到这里：compress_if_needed 能读到 _timestamp（time-based MC 真生效），
            # 同时发给 LLM 的 messages 仍不含 _timestamp（保护 prompt cache）。
            # 双重 strip 幂等：strip_internal_fields 创建新 dict，已 stripped 的再 strip 无副作用。
            messages = strip_internal_fields(messages)

            # 工具集刷新（plan_mode 切换）+ retry warning + 动态记忆 + PRE_LLM_CALL hook
            tool_schemas = await self._prepare_toolset_and_injections(messages)

            # 防孤儿兜底：发送前修复 tool_call 配对。任何来源的孤儿 tool_result
            # （压缩边界 / 流式断连 / 工具异常）都会让 Anthropic API 报 400：
            # "tool_result must have corresponding tool_use in the previous message"。
            # _fix_tool_call_pairs 补缺失的假 result 或删除无主的 tool_result。
            try:
                from agent.context_compressor import _fix_tool_call_pairs
                messages = _fix_tool_call_pairs(messages)
            except Exception as pair_err:
                logger.warning("发送前配对修复失败（忽略）: %s", pair_err)

            # 调 LLM（含 max_tokens 升级 + reactive_compact）
            # Task D4: _call_llm_with_escalation 已改 async
            response = await self._call_llm_with_escalation(
                messages, tool_schemas, system_prompt,
            )
            if response is self._REACTIVE_RETRY:
                system_prompt = self._get_system_prompt()
                continue  # reactive_compact 已修改 history，重试本轮
            if response is None:
                turn_exit_reason = "llm_failed"  # LLM 错误已作为 assistant 消息塞回 history
                break

            api_call_count += 1
            # batch1-T2: 记录 LLM 用量（prompt cache 记账）
            self._record_llm_usage(response)

            # === batch2-T2: POST_LLM_CALL hook（LLM 返回后、处理 tool_calls 前）===
            response = self._run_post_llm_call_hook(response)

            # C1 修复：同步递增压缩会话状态轮次，L4 cooldown 依赖此值
            self._compress_session_state.increment_turn()

            assistant_msg = response.choices[0].message

            # 分支：有 tool_calls → 分发；无 tool_calls → 最终响应
            if assistant_msg.tool_calls:
                # Task D4: _dispatch_tool_calls 已改 async（T_D3）
                should_continue = await self._dispatch_tool_calls(
                    assistant_msg, handle_function_call,
                )
                if not should_continue:
                    break  # P4b-T2: idle 已请求
                # S8 fix: dispatch 后如果预算耗尽，置 grace 让下轮 LLM 看到工具结果
                # 之前 _budget_grace_call 是死代码（__init__ 设 False 后永不置 True），
                # 导致 LLM 刚调工具还没消化结果就因预算耗尽退出，体验差
                # 配套：_grace_triggered 保证整个会话只触发一次（防 dispatch→grace→dispatch 无限循环）
                if self.iteration_budget.remaining <= 0 and not self._grace_triggered:
                    self._budget_grace_call = True
                    self._grace_triggered = True
                    logger.info(
                        "迭代预算耗尽，触发 grace call 让 LLM 看到本轮工具结果再结束"
                    )
                # 继续循环，让 LLM 看到工具结果
                continue

            # 无 tool_calls = 最终响应
            final_content = self._finalize_response(assistant_msg, user_message)
            if self._stop_hook_forced:
                # STOP hook 注入了 force_msg，跳回 while 让 LLM 再跑一轮
                continue

            # === CCAR8 Task 11：goal continue（goal-driven 自动多轮）===
            # 设计：
            # - 只在 goal_state active 且本轮有 LLM response 时评估
            # - 决策 continue → 把 <continue_goal> ephemeral 消息塞 _pending_ephemeral_messages
            #   （**不进 conversation_history**，保护持久化 + prompt cache）
            # - 下一轮 _assemble_turn_messages 末尾会消费 _pending_ephemeral_messages
            #   并 append 到 messages（让 LLM 看到）
            # - 决策 pause/complete → break 正常返回
            # - prompt cache 保护：不动 system prompt，只加 user 消息
            if self._goal_state is not None and self._goal_state.status == "active":
                # 提取本轮 token 用量（从 response.usage）
                turn_tokens = self._extract_turn_tokens(response)
                decision = self._goal_state.evaluate_after_turn(
                    tokens_used=turn_tokens,
                    all_tasks_done=self._check_all_goal_tasks_done(),
                )
                # 持久化 goal 状态
                self._goal_state.save(self._goal_state_path())

                if decision == "continue":
                    # 注入 ephemeral user 消息驱动下一轮（不进 history）
                    cont_msg = _build_goal_continue_message(self._goal_state)
                    if cont_msg is not None:
                        self._pending_ephemeral_messages.append(cont_msg)
                    logger.info(
                        "goal continue: iteration=%d, tokens=%d",
                        self._goal_state.iteration_count,
                        self._goal_state.token_budget,
                    )
                    continue
                # pause / complete / fail → break
                logger.info(
                    "goal %s: reason=%s",
                    decision, self._goal_state.pause_reason,
                )

            return final_content

        # ---------- 循环结束（预算耗尽或中断）----------
        return self._handle_loop_exit(turn_exit_reason, user_message)

    # ------------------------------------------------------------------
    # run_conversation 辅助方法（按主循环调用顺序排列）
    # 纯重构：从原 run_conversation 抽出，行为完全等价。
    # ------------------------------------------------------------------

    def _run_prompt_submit_hook(self, user_message: str) -> str:
        """USER_PROMPT_SUBMIT hook 编排（可能改写 user_message）。"""
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                user_message = self.hooks_registry.run_user_prompt_submit(
                    user_message, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("USER_PROMPT_SUBMIT 编排异常: %s", e)
        return user_message

    def _drain_injected_messages(self) -> dict:
        """收集外部异步消息（后台任务/cron/team inbox/async 子代理），本轮注入后清空。

        返回 dict 含四个 key（任一可能为空）：
            bg_notifications: list, cron_messages: list,
            team_messages_text: str, delegation_results: list
        """
        bg_notifications = []
        if self.bg_manager:
            try:
                bg_notifications = self.bg_manager.drain_notifications()
            except Exception as e:
                logger.warning("drain_notifications 异常: %s", e)

        cron_messages = []
        if self.cron_scheduler:
            try:
                cron_messages = self.cron_scheduler.drain_due()
            except Exception as e:
                logger.warning("cron drain_due 异常: %s", e)

        team_messages_text = ""
        if self.team_bus and self.team_name:
            try:
                msgs = self.team_bus.read_inbox(self.team_name)
                if msgs:
                    team_messages_text = "\n".join(
                        f"[from {m.from_} ({m.type})] {m.content}"
                        for m in msgs
                    )
            except Exception as e:
                logger.warning("team inbox drain 异常: %s", e)

        # async 子代理完成通知（CCAR5 留的漏点 fix：原代码只 push 不 drain）
        # fail-open：queue 异常不崩主循环
        delegation_results = []
        try:
            from tools.delegate_tool import get_delegation_queue
            queue = get_delegation_queue()
            if queue.has_pending():
                delegation_results = queue.drain()
        except Exception as e:
            logger.warning("delegation_queue drain 异常: %s", e)

        return {
            "bg_notifications": bg_notifications,
            "cron_messages": cron_messages,
            "team_messages_text": team_messages_text,
            "delegation_results": delegation_results,
        }

    def _assemble_turn_messages(self, system_prompt: str, injected: dict) -> list:
        """组装本轮 messages：system + history + 注入临时消息 + reminder。

        injected 里 bg/cron/team 注入后会被原地清空（避免下轮重复）。
        plan_mode reminder 每轮重算（不消费）。

        CCAR8 Task 11：channel/mailbox 走 ephemeral 注入（fail-open），
        不进 conversation_history（保护 prompt cache + 持久化）。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
        ]

        # === CCAR8 Task 11：channel/mailbox ephemeral 注入（fail-open）===
        # 用模块级 helper：失败只 log warning，不影响主流程
        # 注入位置在 history 之后、bg/cron/team 之前——channel/mailbox 优先级更高
        # （外部协作消息比内部任务通知更紧急）
        channel_msg = _build_channel_injection(self._channel_inbox)
        if channel_msg is not None:
            messages.append(channel_msg)

        mail_msg = _build_mail_injection(self._mailbox, self._agent_name)
        if mail_msg is not None:
            messages.append(mail_msg)

        # 后台任务通知（消费型）
        if injected.get("bg_notifications"):
            bg = injected["bg_notifications"]
            notif_text = "\n".join(
                f"[task {n['task_id']} {n['status']}] "
                f"exit={n.get('exit_code')} "
                f"stdout_tail={(n.get('stdout') or '')[-200:]}"
                for n in bg
            )
            messages.append({
                "role": "user",
                "content": f"<task_notification>\n{notif_text}\n</task_notification>",
            })
            injected["bg_notifications"] = []

        # cron 定时消息（消费型）
        if injected.get("cron_messages"):
            cron = injected["cron_messages"]
            sched_text = "\n".join(
                f"[Scheduled: {m['job_id']}] {m['message']}"
                for m in cron
            )
            messages.append({
                "role": "user",
                "content": f"<scheduled_message>\n{sched_text}\n</scheduled_message>",
            })
            injected["cron_messages"] = []

        # team inbox（消费型）
        if injected.get("team_messages_text"):
            team_text = injected["team_messages_text"]
            messages.append({
                "role": "user",
                "content": f"<team_messages>\n{team_text}\n</team_messages>",
            })
            injected["team_messages_text"] = ""

        # async 子代理完成通知（消费型，每条转一条 ephemeral user 消息）
        # fail-open + result 截断（防 context 爆炸）
        delegation_results = injected.get("delegation_results") or []
        for r in delegation_results:
            try:
                success = r.get("success", False)
                delegation_id = r.get("delegation_id", "?")
                goal = r.get("goal", "")
                if success:
                    result_text = r.get("result", "")
                    if len(result_text) > 2000:
                        result_text = result_text[:2000] + (
                            f"...[truncated {len(result_text)} chars]"
                        )
                    text = (
                        f"[后台子代理完成] task_id={delegation_id}\n"
                        f"任务: {goal}\n"
                        f"结果: {result_text}"
                    )
                else:
                    error_text = r.get("error", "")
                    text = (
                        f"[后台子代理失败] task_id={delegation_id}\n"
                        f"任务: {goal}\n"
                        f"错误: {error_text}"
                    )
                messages.append({
                    "role": "user",
                    "content": f"<delegation_completion>\n{text}\n</delegation_completion>",
                })
            except Exception as e:
                logger.warning("delegation_results 注入异常: %s", e)
        if delegation_results:
            injected["delegation_results"] = []

        # Plan mode reminder（每轮重算）
        if self.plan_mode:
            messages.append({
                "role": "user",
                "content": (
                    "<plan_mode_reminder>\n"
                    "你处于【计划模式】，只能调研，不能修改任何东西。\n"
                    "完成调研后必须调 exit_plan_mode(plan=...) 提交计划等待用户审批。\n"
                    "计划要包含：要改什么文件、为什么、步骤、风险点。\n"
                    "</plan_mode_reminder>"
                ),
            })

        # 上下文管理提示（接近上限时建议主动 /compact /new）
        self._maybe_inject_context_tip(messages)

        # === CCAR8 Task 11：消费 _pending_ephemeral_messages（goal continue 注入点）===
        # 主循环里 goal continue 时把 ephemeral 消息塞这里（不进 history），
        # 本轮组装时消费并清空——让 LLM 看到但不污染持久化
        if self._pending_ephemeral_messages:
            messages.extend(self._pending_ephemeral_messages)
            self._pending_ephemeral_messages = []

        # 注意：不在这里 strip _timestamp——time-based MC 需要读 _timestamp
        # strip 移到主循环 _run_context_compression 之后、发 LLM 之前
        return messages

    def _maybe_inject_context_tip(self, messages: list) -> None:
        """上下文接近上限时注入管理提示（对齐 Claude Code context rot 建议）。

        官方明确：自动压缩发生在模型"最不聪明"的时刻，建议主动 /compact 并带方向；
        真正的新任务用 /new；大文件读取委托 subagent 只带摘要。
        只提示一次/会话，避免每轮刷屏。
        """
        if self._context_tip_shown:
            return
        try:
            from agent.context_compressor import estimate_message_tokens
            est = estimate_message_tokens(messages)
            token_threshold = self.config.get("context", {}).get(
                "llm_compact_token_threshold", 100000,
            )
            if self.model and "[1m]" in str(self.model):
                token_threshold = max(token_threshold, 700000)
            if est >= token_threshold * 0.7:
                self._context_tip_shown = True
                pct = int(est / token_threshold * 100) if token_threshold else 0
                messages.append({
                    "role": "user",
                    "content": (
                        "<context_management_tip>\n上下文接近上限（约 "
                        f"{pct}%）。为避免自动压缩发生在效果最差时：\n"
                        "1. 继续当前任务 → 建议主动 /compact 并说明保留哪些重点\n"
                        "2. 换新任务 → 建议 /new 新开对话（避免 context rot）\n"
                        "3. 大文件读取 → 用 subagent 委托子代理，只带摘要回主上下文\n"
                        "</context_management_tip>"
                    ),
                })
        except Exception as e:
            logger.debug("上下文管理提示注入失败（忽略）: %s", e)

    async def _run_context_compression(self, messages: list, system_prompt: str) -> tuple:
        """接近 token 上限时压缩上下文（async：compress_if_needed 已改 async）。

        返回 (messages, system_prompt, compressed: bool)。
        压缩后会重建 system prompt 并注入 <post_compress_brief>。

        Task D4 fix: 改 async + await compress_if_needed。
        """
        if not self.compression_enabled:
            return messages, system_prompt, False

        from agent.context_pipeline import compress_if_needed
        ctx_cfg = self.config.get("context", {})
        messages, compressed = await compress_if_needed(
            messages,
            llm_client=self.llm_client,
            model=self.model,
            config=ctx_cfg,
            session_state=self._compress_session_state,
            agent_home=self.omnimate_home,
            session_id=self.session_id,
            hooks_registry=self.hooks_registry,
        )
        if not compressed:
            return messages, system_prompt, False

        # batch2-T1: 压缩前调 memory_manager.on_pre_compress 提取事实
        if self.memory_manager:
            try:
                self.memory_manager.on_pre_compress(None, messages)
            except Exception as e:
                logger.warning("on_pre_compress 编排异常: %s", e)

        # 压缩修改了历史，同步并重建 system prompt
        # CCAR8 Task 11：strip ephemeral 消息（保护持久化——ephemeral 不应进 history）
        self.conversation_history = [
            m for m in messages[1:] if not m.get("_ephemeral")
        ]
        self.invalidate_system_prompt()
        system_prompt = self._get_system_prompt()
        self._compression_attempts += 1

        # 对齐 Claude Code compact_boundary：压缩摘要占位也入库，
        # 恢复时模型能知道"此处之前的旧消息已被总结"（避免困惑/重复总结）
        if self.conversation_history:
            first_msg = self.conversation_history[0]
            if first_msg.get("role") == "user":
                first_content = first_msg.get("content", "")
                if first_content.startswith(
                    ("[之前的对话已自动总结]", "[紧急上下文压缩")
                ):
                    self._persist_session_message("user", first_content)

        # PostCompressReanchor：注入"刚醒来"brief
        brief_parts = [
            "你刚经历了上下文压缩，历史已被总结。"
            "身份和 system prompt 不变。"
        ]
        mode_text = (
            "计划模式（只能调研，不能修改）"
            if self.plan_mode
            else "正常执行模式"
        )
        brief_parts.append(f"当前模式：{mode_text}")

        # CCAR4 Task B：对齐 Claude Code，压缩后重注入最近加载的技能正文 + 读过的文件，
        # 让 agent 压缩后不"失忆"（避免反复手动读文件/重新 load_skill）。
        # 委托到 post_compact_recovery 模块（fail-open，走 safe_path 白名单）。
        try:
            from agent.post_compact_recovery import build_post_compact_brief
            reinject = build_post_compact_brief(self)
        except Exception as e:
            logger.debug("post_compact_recovery fail-open: %s", e)
            reinject = ""
        if reinject:
            brief_parts.append(
                "以下是你最近加载的技能和读过的文件（压缩后重注入，帮助恢复上下文）：\n"
                f"{reinject}"
            )

        brief_parts.append("请继续之前的工作。")
        messages.append({
            "role": "user",
            "content": (
                "<post_compress_brief>\n"
                + "\n".join(brief_parts)
                + "\n</post_compress_brief>"
            ),
        })

        return messages, system_prompt, True

    async def _prepare_toolset_and_injections(self, messages: list) -> list:
        """刷新工具集（plan_mode 切换）+ 注入 retry_warning + PRE_LLM_CALL hook。

        会原地修改 messages（追加 reminder）。返回 tool_schemas（可能被 hook 修改）。

        Task 2.5: 动态记忆注入块已删除——记忆上下文统一由 CCAR10 ephemeral
        在 run_conversation 开场注入（每 user 轮一次，不重复、不污染 history）。

        Task D4 fix: 方法本身仍是 async（PRE_LLM_CALL hook + 其他异步依赖保留）。
        """
        from model_tools import get_tool_definitions

        # PlanMode: plan_mode 下强制切到 plan 工具集（只读）
        effective_toolsets = ["plan"] if self.plan_mode else self.enabled_toolsets
        # E2 NEW: 从 config 透传 disabled_tools（子代理自定义 .md 定义的 disallowedTools）
        _disabled = (self.config or {}).get("disabled_tools")
        tool_schemas = get_tool_definitions(
            effective_toolsets, disabled_tools=_disabled, agent=self)

        # 失败重试检测：连续 N 次工具失败 → 注入提醒
        if self._tool_failure_streak >= self._failure_threshold:
            messages.append({
                "role": "user",
                "content": (
                    f"<retry_warning>\n"
                    f"你已连续 {self._tool_failure_streak} 次工具调用失败。\n"
                    f"最近错误: {self._last_tool_error}\n\n"
                    f"**不要用完全相同的方式重试**。建议:\n"
                    f"1. 分析错误根因(看 stderr / error 字段)\n"
                    f"2. 换一种方法(改命令 / 改路径 / 改参数)\n"
                    f"3. 如果是环境问题(路径冲突 / 权限 / 版本不兼容),"
                    f"**停下来告诉用户**具体问题和解决建议\n"
                    f"</retry_warning>"
                ),
            })
            self._tool_failure_streak = 0  # 重置（提醒一次就够）

        # PRE_LLM_CALL hook
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                messages, tool_schemas = self.hooks_registry.run_pre_llm_call(
                    messages, tool_schemas, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("PRE_LLM_CALL hook 编排异常: %s", e)

        return tool_schemas

    async def _call_llm_with_escalation(
        self, messages: list, tool_schemas: list, system_prompt: str,
    ):
        """调 LLM，处理 max_tokens 升级 + reactive_compact。

        返回值：
            response 对象         - 正常返回
            self._REACTIVE_RETRY  - 触发了 reactive_compact，主循环应重试本轮
            None                  - LLM 错误（已写入 history），主循环应 break

        Task D4: 改 async（_call_llm_streaming + call_with_retry 均已 async）。
        改造点 ③：pre/post 调用 cache_monitor hook（fail-open，绝不影响主流程）。
        """
        # === 改造点 ③ pre-call：cache_monitor 快照 12 维 prompt 状态（fail-open）===
        # CCAR4 Task A：扩展到 12 维度 + per-tool hash（对齐 claude-code-main）
        cache_state = None
        try:
            from agent.cache_monitor import record_prompt_state
            # 从 config 提取 LLM 调用参数（对齐 _call_llm_streaming 的 max_tokens 提取逻辑）
            _cfg = self.config or {}
            _mt = (
                _cfg.get("model", {}).get("max_tokens")
                or _cfg.get("llm", {}).get("max_tokens")
            ) or 0
            _temp = _cfg.get("model", {}).get("temperature")
            # user 首条消息前缀（捕捉 user 消息变化）
            _ucp = ""
            if messages:
                for m in messages:
                    if m.get("role") == "user":
                        _ucp = str(m.get("content", ""))[:500]
                        break
            cache_state = record_prompt_state(
                system_prompt=system_prompt or "",
                tools=tool_schemas or [],
                model=self.model or "",
                max_tokens=_mt,
                temperature=_temp,
                stream_mode=self._stream_callback is not None,
                tool_choice=_cfg.get("model", {}).get("tool_choice"),
                betas=_cfg.get("model", {}).get("betas"),
                user_content_prefix=_ucp,
                messages_count=len(messages),
            )
        except Exception as e:
            logger.debug("cache_monitor pre-call fail-open: %s", e)

        try:
            if self._stream_callback is not None:
                response = await self._call_llm_streaming(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )
            else:
                from agent.llm_retry import call_with_retry, detect_length_finish
                response = await call_with_retry(
                    self.llm_client,
                    messages,
                    tools=tool_schemas if tool_schemas else None,
                    fallback_llm_client=self.fallback_llm_client,
                    config=self.config,
                )
                # P0-3: 非流式路径也支持 max_tokens 升级
                if (detect_length_finish(response)
                        and self._max_tokens_escalator is not None
                        and not self._max_tokens_escalator.has_escalated):
                    new_max = self._max_tokens_escalator.escalate()
                    logger.info("max_tokens 截断（非流式），升级到 %d 重试", new_max)
                    try:
                        response = await call_with_retry(
                            self.llm_client,
                            messages,
                            tools=tool_schemas if tool_schemas else None,
                            fallback_llm_client=self.fallback_llm_client,
                            max_tokens=new_max,
                            config=self.config,
                        )
                    except Exception as esc_err:
                        logger.warning(
                            "max_tokens 升级重试失败（沿用截断响应）: %s", esc_err,
                        )

            # === 改造点 ③ post-call：check_cache_break（fail-open）===
            if cache_state is not None:
                try:
                    from agent.cache_monitor import check_cache_break
                    cache_read = self._extract_cache_read(
                        getattr(response, "usage", None)
                    )
                    check_cache_break(
                        current_state=cache_state,
                        cache_read_tokens=cache_read,
                        query_source="main",
                    )
                except Exception as e:
                    logger.debug("cache_monitor post-call fail-open: %s", e)

            return response

        except Exception as e:
            # reactive_compact：API 报 prompt_too_long 时紧急压缩并重试
            # Task D：改多次触发（冷却 60s + 上限 5 次/会话），
            # gate 逻辑下沉到 reactive_compact 内部（session_state.reactive_count / reactive_last_at）
            # Task P1.2：加 feature flag 开关（默认 OFF，避免无意启用）
            err_str = str(e).lower()
            is_prompt_too_long = (
                "prompt_too_long" in err_str
                or "context_length" in err_str
                or "maximum context" in err_str
            )
            from agent.feature_flags import is_feature_enabled
            reactive_enabled = is_feature_enabled(
                self.config, "reactive_compact",
            )
            if reactive_enabled and is_prompt_too_long:
                from agent.context_pipeline import reactive_compact
                ctx_cfg = self.config.get("context", {})
                messages, changed = reactive_compact(
                    messages,
                    session_state=self._compress_session_state,
                    keep_recent=ctx_cfg.get("reactive_keep_recent", 5),
                    cooldown_seconds=ctx_cfg.get(
                        "reactive_compact_cooldown_seconds", 60),
                    max_per_session=ctx_cfg.get(
                        "reactive_compact_max_per_session", 5),
                )
                if changed:
                    self._reacted = True  # 向后兼容标记
                    # CCAR8 Task 11：strip ephemeral（保护持久化）
                    self.conversation_history = [
                        m for m in messages[1:] if not m.get("_ephemeral")
                    ]
                    self.invalidate_system_prompt()
                    logger.warning("reactive_compact 后重试本轮")
                    return self._REACTIVE_RETRY
                # changed=False：冷却中或达到上限，走正常错误路径

            logger.error("LLM API 调用失败（重试后）: %s", e)

            # === CCAR8 Task 11：goal 网络异常自动 pause ===
            # 关键词触发：529 / overloaded / timeout / connection / network
            # 只在 goal active 时 pause（避免无 goal 时副作用）
            # fail-open：pause 本身失败只 log
            if self._goal_state is not None and self._goal_state.status == "active":
                network_keywords = (
                    "529", "overloaded", "timeout",
                    "connection", "network", "timed out",
                    "connectionerror", "connectionreseterror",
                )
                if any(kw in err_str for kw in network_keywords):
                    try:
                        self._goal_state.pause(reason="network")
                        self._goal_state.save(self._goal_state_path())
                        logger.warning(
                            "goal 自动 pause（网络异常）: %s", self._goal_state.pause_reason,
                        )
                    except Exception as pause_err:
                        logger.warning("goal pause 失败（fail-open）: %s", pause_err)

            # 错误作为助手消息塞回，让模型有机会自我修正
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[API 错误: {e}]",
                "_timestamp": time.time(),
            })
            return None

    def _run_post_llm_call_hook(self, response):
        """POST_LLM_CALL hook 编排（可能修改 response）。"""
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                response = self.hooks_registry.run_post_llm_call(
                    response, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("POST_LLM_CALL hook 编排异常: %s", e)
        return response

    def _checkpoint_track(self, path: str) -> None:
        """编辑工具成功改文件后调用：记录该文件供 checkpoint 快照/回滚。"""
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.track_file(path)
            except Exception as e:
                logger.debug("checkpoint track 失败: %s", e)

    def _record_recent(self, kind: str, key: str) -> None:
        """记录最近读过的文件 / 加载的技能（去重保序，保留最近 10 个）。

        压缩后把这些重注入上下文，让 agent 不"失忆"（对齐 Claude Code）。
        """
        bucket = self._recent_read_files if kind == "read" else self._recent_skills
        if key in bucket:
            bucket.remove(key)
        bucket.append(key)
        del bucket[:-10]

    def _load_skill_body(self, name: str) -> str:
        """跨目录加载技能正文（去 frontmatter），失败返回空串。"""
        try:
            from tools.skill_tools import _find_skill_md, _get_skills_dirs
            from agent.skill_commands import parse_frontmatter
            md = _find_skill_md(
                name,
                _get_skills_dirs({"omnimate_home": str(self.omnimate_home)}),
            )
            if md is None:
                return ""
            content = md.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)
            return body.strip()
        except Exception as e:
            logger.debug("加载技能正文失败 %s: %s", name, e)
            return ""

    def _persist_session_message(self, role, content, *, tool_calls=None,
                                 tool_call_id=None, name=None) -> None:
        """把消息持久化到会话库（对齐 Claude Code：工具轮次完整入库）。

        当前会话内 conversation_history 在内存里已含工具轮次；这里把
        assistant(tool_calls) 和 tool 结果同步写入 sessions.db，让重开会话
        恢复时能全量重放。user 输入和最终 assistant 回复由 cli.py 持久化，
        这里只补工具轮次，避免重复。失败不阻塞主流程。
        """
        if not self.session_store or not self.session_id:
            return
        try:
            self.session_store.append_message(
                self.session_id, role, content or "",
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            )
        except Exception as e:
            logger.warning("持久化 %s 消息失败: %s", role, e)

    async def _dispatch_tool_calls(self, assistant_msg, handle_function_call) -> bool:
        """执行 assistant_msg.tool_calls，处理 plan_approval + 失败统计 + idle 检查。

        assistant 消息（含 tool_calls + thinking 字段）会先追加到 history。
        返回 True 表示继续主循环，False 表示 idle 已请求需退出。

        Task F2：safe/unsafe 分组 + asyncio.gather。
        - safe 工具（isConcurrencySafe=True，read_file/list_dir 等只读）并发跑
        - unsafe 工具（write_file/terminal/memory_save 等有副作用）串行 await
        - 结果按原 tool_call 顺序回填 history（_merge_results_in_order），
          保证 tool_call.id 与 tool_result 严格配对，不破坏消息历史严格交替
        - 单个 safe 失败不阻塞其他（return_exceptions=True → 转 JSON error）
        """
        # 追加 assistant 消息（DeepSeek 工具调用回传要求 thinking 字段）
        assistant_entry = {
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ],
        }
        rc = getattr(assistant_msg, "reasoning_content", None)
        sig = getattr(assistant_msg, "thinking_signature", None)
        if rc:
            assistant_entry["reasoning_content"] = rc
        if sig:
            assistant_entry["thinking_signature"] = sig
        assistant_entry["_timestamp"] = time.time()
        self.conversation_history.append(assistant_entry)
        # 对齐 Claude Code：assistant(tool_calls) 消息持久化，恢复时可重放
        self._persist_session_message(
            "assistant", assistant_entry.get("content") or "",
            tool_calls=assistant_entry.get("tool_calls"),
        )

        # ---- Task F2: safe/unsafe 分组 ----
        tool_calls = list(assistant_msg.tool_calls)
        safe_calls = []
        unsafe_calls = []
        for tc in tool_calls:
            entry = registry.get(tc.function.name)
            is_safe = bool(entry.isConcurrencySafe) if entry else False
            if is_safe:
                safe_calls.append(tc)
            else:
                unsafe_calls.append(tc)

        # ---- safe 组：先按顺序跑 pre-callback（_record_recent / on_tool_call），
        # 再 asyncio.gather 并发 handle_function_call（return_exceptions=True）。
        # 失败的 result 转 JSON error，单个失败不阻塞其他。
        safe_results_raw = await self._run_safe_group_concurrently(
            safe_calls, handle_function_call,
        )

        # ---- unsafe 组：串行 await，保留 plan_approval / 失败统计等完整逻辑 ----
        unsafe_results = []
        for tc in unsafe_calls:
            tool_content = await self._run_unsafe_tool_call(tc, handle_function_call)
            unsafe_results.append(tool_content)

        # ---- 按原 tool_call 顺序回填 history ----
        # safe 组结果同样需要做 plan_approval 检查和失败统计（极少触发，但
        # read_file 返回 error 仍应计入 streak；exit_plan_mode 是 unsafe，
        # 不会在 safe 组出现）。
        safe_processed = []
        for tc, raw in zip(safe_calls, safe_results_raw):
            tool_name = tc.function.name
            if isinstance(raw, Exception):
                content = json.dumps({
                    "error": f"concurrent dispatch failed: {raw}",
                    "error_type": "concurrent_dispatch_error",
                }, ensure_ascii=False)
            else:
                content = raw
            # plan_approval 检查（safe 组理论上不会触发，但保持对称以防 registry
            # 分类变化——比如未来某只读工具也可能产 plan_approval_required）
            content = self._maybe_handle_plan_approval(tc, content)
            # 失败统计
            self._update_failure_streak(content)
            safe_processed.append((tc, content))

        unsafe_processed = [(tc, c) for tc, c in zip(unsafe_calls, unsafe_results)]

        # 按原 tool_call 顺序合并并回填 history
        self._merge_results_in_order(
            tool_calls, safe_processed, unsafe_processed,
        )

        # P4b-T2: idle 标志检查
        if self._idle_requested:
            logger.info("idle 已请求，退出 run_conversation")
            return False
        return True

    async def _run_safe_group_concurrently(self, safe_calls, handle_function_call):
        """safe 组并发执行 handle_function_call。

        pre-callback（_record_recent / on_tool_call）按原顺序同步跑一遍
        （它们是廉价的记录/通知副作用，不进并发），再用 asyncio.gather 并发。
        return_exceptions=True 保证单个失败不阻塞其他。
        """
        # 按顺序跑 pre-callback（保持 _record_recent 的语义：先记录再执行）
        for tc in safe_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            if tool_name == "read_file" and tool_args.get("path"):
                self._record_recent("read", str(tool_args["path"]))
            elif tool_name == "load_skill" and tool_args.get("name"):
                self._record_recent("skill", str(tool_args["name"]))
            if self.on_tool_call:
                try:
                    self.on_tool_call(tool_name, tool_args)
                except Exception:
                    pass

        # 并发跑 handle_function_call
        async def _one(tc):
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            return await handle_function_call(
                tc.function.name, tool_args,
                session_id=self.session_id,
                memory_store=self.memory_store,
                session_store=self.session_store,
                omnimate_home=self.omnimate_home,
                tool_call_id=tc.id,
                config=self.config,
                hooks_registry=self.hooks_registry,
                bg_manager=self.bg_manager,
                team_bus=self.team_bus,
                team_coordinator=self.team_coordinator,
                team_name=self.team_name,
                agent_ref=self,
            )

        if not safe_calls:
            return []
        return await asyncio.gather(*[_one(tc) for tc in safe_calls],
                                    return_exceptions=True)

    async def _run_unsafe_tool_call(self, tc, handle_function_call):
        """unsafe 组单个工具调用（串行）。

        保留原完整逻辑：pre-callback + handle_function_call + plan_approval
        + 失败统计。返回最终要回填 history 的 tool_content（已含 plan_approval
        覆盖后的内容）。
        """
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            tool_args = {}

        # 压缩后重注入：记录最近读过的文件 / 加载的技能
        if tool_name == "read_file" and tool_args.get("path"):
            self._record_recent("read", str(tool_args["path"]))
        elif tool_name == "load_skill" and tool_args.get("name"):
            self._record_recent("skill", str(tool_args["name"]))

        if self.on_tool_call:
            try:
                self.on_tool_call(tool_name, tool_args)
            except Exception:
                pass

        result = await handle_function_call(
            tool_name, tool_args,
            session_id=self.session_id,
            memory_store=self.memory_store,
            session_store=self.session_store,
            omnimate_home=self.omnimate_home,
            tool_call_id=tc.id,
            config=self.config,
            hooks_registry=self.hooks_registry,
            bg_manager=self.bg_manager,
            team_bus=self.team_bus,
            team_coordinator=self.team_coordinator,
            team_name=self.team_name,
            agent_ref=self,
        )

        # plan_approval 处理（exit_plan_mode 等；unsafe 组独有路径）
        tool_content = self._maybe_handle_plan_approval(tc, result)
        # 失败统计
        self._update_failure_streak(tool_content)
        return tool_content

    def _maybe_handle_plan_approval(self, tc, result):
        """如果 tool 结果是 plan_approval_required，跑审批回调并返回最终 content。

        不是审批请求则原样返回。保留原 plan_approval 语义（默认 approved=True
        当 callback 为 None；plan_handled 后写 plan_approved/rejected 内容）。
        """
        try:
            result_data = json.loads(result) if isinstance(result, str) else {}
        except (json.JSONDecodeError, ValueError):
            result_data = {}

        if result_data.get("error_type") != "plan_approval_required":
            return result

        plan_text = result_data.get("plan", "")
        try:
            if self.plan_approval_callback is not None:
                approved, feedback = self.plan_approval_callback(plan_text)
            else:
                approved, feedback = True, ""
        except Exception as cb_exc:
            logger.warning("plan_approval_callback 异常: %s", cb_exc)
            approved = False
            feedback = f"审批回调异常: {cb_exc}"

        if approved:
            self.plan_mode = False
            return json.dumps({
                "plan_approved": True,
                "message": "用户已批准计划。现在可以开始执行：用 task_create 列出步骤，每步完成调 task_complete，依赖关系用 blocked_by。",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "plan_rejected": True,
                "feedback": feedback or "用户未提供拒绝原因",
                "message": "用户拒绝了计划。请根据 feedback 修订后重新调 exit_plan_mode。",
            }, ensure_ascii=False)

    def _update_failure_streak(self, tool_content):
        """根据单个 tool 结果更新 _tool_failure_streak / _last_tool_error。

        保留原语义：error 字段或 exit_code != 0 视为失败，streak +1；
        成功则清零。
        """
        try:
            rd = json.loads(tool_content) if isinstance(tool_content, str) else {}
            is_error = bool(rd.get("error")) or rd.get("exit_code", 0) != 0
        except (json.JSONDecodeError, TypeError):
            is_error = False
        if is_error:
            self._tool_failure_streak += 1
            err_snippet = (
                rd.get("error", "") or str(rd.get("stderr", ""))[:200]
                if isinstance(rd, dict) else ""
            )
            self._last_tool_error = str(err_snippet)[:200]
        else:
            self._tool_failure_streak = 0

    def _merge_results_in_order(self, all_calls, safe_processed, unsafe_processed):
        """按原 tool_call 顺序合并 safe/unsafe 结果并回填 history + 持久化。

        safe_processed / unsafe_processed: List[(tool_call, content)]。
        保证 tool_call.id 与 tool_result 严格配对（不破坏消息历史严格交替）。
        """
        safe_map = {tc.id: content for tc, content in safe_processed}
        unsafe_map = {tc.id: content for tc, content in unsafe_processed}
        for tc in all_calls:
            if tc.id in safe_map:
                content = safe_map[tc.id]
            elif tc.id in unsafe_map:
                content = unsafe_map[tc.id]
            else:
                # 不应发生：缺结果兜底（避免空 tool_result 破坏 LLM API 配对）
                content = json.dumps({
                    "error": "missing result for tool_call_id={}".format(tc.id),
                    "error_type": "missing_tool_result",
                }, ensure_ascii=False)
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": content,
                "_timestamp": time.time(),
            })
            self._persist_session_message(
                "tool", content, tool_call_id=tc.id, name=tc.function.name,
            )

    def _finalize_response(self, assistant_msg, user_message: str) -> str:
        """处理无 tool_calls 的最终响应：保存历史 + STOP hook + reflection。

        如果 STOP hook 触发 force_msg，会设置 self._stop_hook_forced = True，
        主循环看到这个标志应 continue（而非 return）。

        空响应处理（防"突然断开"bug）：
        - content 空 + reasoning_content 有值 → 用 reasoning 作为回复（思考模型纯 thinking）
        - 完全空（content + reasoning 都空）→ 友好兜底消息（不静默返回空串）
        """
        final_content = assistant_msg.content or ""

        # 空响应处理：reasoning_content fallback + 完全空兜底
        if not final_content:
            reasoning = getattr(assistant_msg, "reasoning_content", None)
            if reasoning:
                # 思考模型纯 thinking 响应：reasoning 对用户有价值，作为回复
                final_content = (
                    "[模型只产出了思考过程，未给最终回复。以下是思考内容：]\n\n"
                    f"<thinking>\n{reasoning}\n</thinking>"
                )
                logger.info(
                    "LLM 返回纯 thinking 响应（content 空 + reasoning_content 有值），"
                    "用 reasoning 作为回复"
                )
            else:
                # 完全空响应（content + reasoning 都空）：友好兜底
                # 之前这里静默返回空串，用户看到"突然断开"以为 agent 崩了
                final_content = (
                    "[LLM 返回了空响应（content 和 reasoning_content 都为空）。"
                    "可能是网络抖动、流式断连或 provider bug。请重试。]"
                )
                logger.warning(
                    "LLM 返回完全空响应（content + reasoning_content 都空），"
                    "可能是网络问题或模型 bug；返回友好兜底消息（不静默返回空串）"
                )

        self.conversation_history.append({
            "role": "assistant",
            "content": final_content,
            "_timestamp": time.time(),
        })

        if self.on_response:
            try:
                self.on_response(final_content)
            except Exception:
                pass

        # 异步写入外部记忆 provider（不阻塞）
        self._sync_memory(user_message, final_content)

        # STOP hook（可能触发 force_msg 让循环继续）
        self._stop_hook_forced = False
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)
                and self._stop_fire_count < self.config.get(
                    "hooks", {}).get("stop_hook_max_fires", 3)):
            try:
                force_msg = self.hooks_registry.run_stop(
                    session_id=self.session_id or "",
                    max_fires=self.config.get("hooks", {}).get(
                        "stop_hook_max_fires", 3),
                )
            except Exception as e:
                logger.warning("STOP hook 编排异常: %s", e)
                force_msg = None

            if force_msg:
                self._stop_fire_count += 1
                self.conversation_history.append({
                    "role": "user",
                    "content": f"[stop_hook]: {force_msg}",
                })
                self._stop_hook_forced = True
                return final_content  # 主循环看到标志会 continue

        # CCALS-P0-2: 任务级反思（异步，不阻塞返回）
        if self._reflection_enabled:
            try:
                self._trigger_reflection_async()
            except Exception as e:
                logger.warning("触发反思失败（不影响主流程）: %s", e)

        return final_content

    def _extract_partial_result(self) -> str:
        """Task K: 被中断时提取已完成部分（借鉴 Claude Code extractPartialResult）。

        返回最后一条 assistant 消息的 content（如有），加 [PARTIAL] 前缀
        让父代理知道这是中断而非完整结果；无 assistant 消息时返回空串。

        fail-open：conversation_history 异常时返回空串（不抛错），保证
        父代理 cancel 路径不因子代理 history 状态异常而崩。

        使用场景：
            1. cancel_event 被 set → run_conversation 主循环退出时调此方法
            2. 父代理 _delegate_sync 的 cancel 路径返回 partial result
        """
        try:
            history = getattr(self, "conversation_history", None) or []
            for msg in reversed(history):
                if (isinstance(msg, dict)
                        and msg.get("role") == "assistant"
                        and msg.get("content")):
                    content = msg["content"]
                    if isinstance(content, str) and content.strip():
                        return f"[PARTIAL] {content}"
            return ""
        except Exception as e:
            logger.warning("_extract_partial_result 异常（fail-open）: %s", e)
            return ""

    def _handle_loop_exit(self, turn_exit_reason: str, user_message: str) -> str:
        """循环结束（预算耗尽或中断）的兜底响应。"""
        if turn_exit_reason == "interrupted_by_user":
            fallback = "[已被用户中断]"
        elif turn_exit_reason == "llm_failed":
            fallback = (
                "[LLM 调用失败，本轮已中断] 重试或检查模型连接。"
                "详见日志（LLM API 调用失败）。"
            )
            # P3.3: 触发 STOP_FAILURE hook（与正常 STOP 区分，通知审计/告警 hook）
            self._trigger_stop_failure_hook(
                error="LLM 调用失败", error_type="LLMError",
            )
        else:
            fallback = "[已达最大迭代次数，强制停止]"

        self.conversation_history.append({
            "role": "assistant",
            "content": fallback,
            "_timestamp": time.time(),
        })
        # 异步写入外部记忆 provider（即使被打断也保留部分上下文）
        self._sync_memory(user_message, fallback)
        return fallback

    def _trigger_stop_failure_hook(self, *, error: str, error_type: str) -> None:
        """P3.3: 触发 STOP_FAILURE hook（fail-open，异常不影响主流程）。"""
        if not self.hooks_registry:
            return
        if not self.config.get("hooks", {}).get("enabled", True):
            return
        try:
            self.hooks_registry.run_stop_failure({
                "session_id": self.session_id or "",
                "error": error,
                "error_type": error_type,
            })
        except Exception as e:
            logger.warning("STOP_FAILURE hook 编排异常: %s", e)

    async def chat(self, message: str, cancel_event=None) -> str:
        """简单接口：发一条消息，返回响应。

        Task D4: 改 async（run_conversation 已 async）。
        Task K: 新增 cancel_event 参数，透传给 run_conversation
        （子代理场景用，让父代理能 cancel 子代理）。
        """
        if cancel_event is not None:
            return await self.run_conversation(message, cancel_event=cancel_event)
        return await self.run_conversation(message)

    def _sync_memory(self, user_message: str, assistant_message: str) -> None:
        """异步同步一轮到外部记忆 provider（如有）。

        内置文件记忆（MemoryStore）由 memory 工具主动写入，
        这里只触发外部 provider 的 sync_turn（如 SimpleProvider）。
        """
        if self.memory_manager is None:
            return
        try:
            self.memory_manager.sync_all(user_message, assistant_message)
        except Exception as e:
            logger.debug("memory_manager.sync_all 失败: %s", e)

    def _trigger_reflection_async(self) -> None:
        """CCALS-P0-2: 异步触发任务级反思（不阻塞返回 final_content）。

        策略：
        - 启动 daemon thread 跑 apply_reflection
        - aux_llm_router 优先用（便宜模型），fallback 到主 llm_client
        - memory_store 不可用时 no-op
        - 反思针对当前会话最近 N 条消息（含本轮用户消息+最终响应+中间过程）
        - 节流：任意时刻最多 1 个反思在跑 + 距上次启动不足 cooldown_turns 轮时跳过
        """
        import contextvars
        import threading
        # 拷贝引用（thread 启动后 conversation_history 可能继续变化）
        store = self.memory_store
        if store is None:
            return

        # 节流 1：已经有反思在跑 → 跳过（防连问烧 token）
        # 节流 2：距上次启动不足 cooldown_turns 轮 → 跳过
        with self._reflection_lock:
            current_turn = self._last_reflection_turn + 1  # 本轮的"逻辑序号"
            if self._active_reflections >= 1:
                return
            if (self._last_reflection_turn >= 0
                    and current_turn - self._last_reflection_turn < self._reflection_cooldown_turns):
                return
            self._active_reflections += 1
            self._last_reflection_turn = current_turn

        # 优先 aux_llm（便宜模型）
        llm_for_reflection = self.aux_llm_router or self.llm_client
        # 快照最近 20 条消息（避免 thread 启动后被改）
        messages_snapshot = list(self.conversation_history[-20:])

        def _bg():
            try:
                from agent.reflection import apply_reflection
                apply_reflection(
                    messages=messages_snapshot,
                    memory_store=store,
                    llm_client=llm_for_reflection,
                    session_id=self.session_id or "",
                )

                # 批次 C: 用户画像更新(每 5 次反思后)
                try:
                    from agent.user_profile import should_update_profile, build_and_save_profile
                    if should_update_profile() and self.aux_llm_router:
                        build_and_save_profile(
                            memory_store=store,
                            aux_llm=self.aux_llm_router,
                            agent_home=self.omnimate_home,
                        )
                except Exception as e:
                    logger.debug("用户画像更新失败(fail-open): %s", e)
            except Exception as e:
                logger.debug("反思后台任务异常: %s", e)
            finally:
                with self._reflection_lock:
                    self._active_reflections -= 1

        # CCAR9 final review Minor：daemon 线程不自动继承主线程 contextvars。
        # 会话内切 cwd 后，reflection 的 project 记忆会落错项目区（fallback
        # os.getcwd()）。修法：主线程里 copy_context()，target 用 ctx.run 包一层。
        _reflection_ctx = contextvars.copy_context()
        t = threading.Thread(
            target=lambda: _reflection_ctx.run(_bg),
            daemon=True,
            name="reflection",
        )
        t.start()
