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

import json
import logging
from typing import Optional

from openai import OpenAI

from agent.budget import IterationBudget
from agent.prompt_builder import build_system_prompt

logger = logging.getLogger(__name__)


class AIAgent:
    """核心 Agent 类。一个实例对应一个会话。"""

    def __init__(
        self,
        *,
        base_url: str = None,
        api_key: str = None,
        model: str = "deepseek-chat",
        model_format: str = "openai",      # openai / anthropic
        fallback_model: str = None,
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        session_id: str = None,
        system_prompt_override: str = None,
        memory_store=None,
        memory_manager=None,
        session_store=None,
        harvil_home=None,
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
        """
        # 创建 LLM client（根据 model_format 选 OpenAI 兼容或 Anthropic 原生）
        from agent.llm_client import create_llm_client
        model_config = {
            "format": model_format,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
        self.llm_client = create_llm_client(model_config)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.model_format = model_format
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
        self.memory_store = memory_store
        self.memory_manager = memory_manager
        self.session_store = session_store
        # C3 修复：harvil_home=None 时解析为默认 ~/.agent，避免下游 TypeError
        if harvil_home is not None:
            self.harvil_home = harvil_home
        else:
            from constants import get_agent_home
            self.harvil_home = get_agent_home()
        # 任务清单（TodoWrite 机制，P1 借鉴 Claude Code）
        try:
            from agent.todo import get_todo_manager
            self.todo_manager = get_todo_manager()
        except Exception:
            self.todo_manager = None
        self.on_tool_call = on_tool_call
        self.on_response = on_response

        # 预算（每个会话独立）
        self.iteration_budget = IterationBudget(max_iterations)

        # 中断标志（Ctrl+C 时被置为 True）
        self._interrupt_requested = False

        # 预算耗尽后的"最后一次机会"
        self._budget_grace_call = False

        # 系统提示：会话开始时构建一次，后续缓存
        self._cached_system_prompt: Optional[str] = system_prompt_override
        self._system_prompt_built = system_prompt_override is not None
        # 05 NEW: 三层结构缓存
        self._stable_prompt: Optional[str] = system_prompt_override
        self._context_prompt: Optional[str] = ""

        # 对话历史（不包含 system prompt，system 单独传）
        self.conversation_history: list = []

        # 上下文压缩配置
        self.compression_enabled = True
        self._compression_attempts = 0
        # 完整 config（用于 context 阈值、hooks 等开关）
        self.config: dict = config or {}

        # === P2-T6 NEW: hooks 系统 ===
        self.hooks_registry = hooks_registry
        self._stop_fire_count = 0

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
                self._cached_memory_index = self.memory_store.snapshot_for_prompt()
            except Exception as e:
                logger.warning("缓存 memory 索引失败: %s", e)

        # === P4a-T6 NEW: team 消息总线 + 协调器 ===
        self.team_bus = team_bus
        self.team_coordinator = team_coordinator
        self.team_name = team_name

        # === P4b-T2 NEW: idle 标志 + spawn 深度 ===
        self._idle_requested = False
        self.spawn_depth = spawn_depth

        # === batch2-T3 NEW: 辅助 LLM 路由器 ===
        self.aux_llm_router = aux_llm_router

        # === PlanMode NEW: 计划模式状态 + 审批回调 ===
        # plan_mode=True 时下一轮起切换到 ["plan"] 工具集（只读）
        # plan_approval_callback(plan: str) -> (approved: bool, feedback: str)
        # None 表示自动批准（测试/库用法）
        self.plan_mode: bool = False
        self.plan_approval_callback = plan_approval_callback

        # === B1 NEW: vision client（image_analyze / image_ocr / browser_vision 共用） ===
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
        # 主 agent 无 HARVIL_KANBAN_TASK env，no-op
        try:
            from agent.team.auto_heartbeat import register as _register_auto_heartbeat
            _register_auto_heartbeat(self.hooks_registry)
        except Exception as e:
            logger.warning("注册 auto_heartbeat hook 失败（不影响主流程）: %s", e)

        # === ⑪c NEW: 浏览器会话 ===
        # 仅在 browser toolset 启用时创建（lazy init，不立即启 Chromium）
        self.browser_session = None
        if "browser" in (enabled_toolsets or []):
            try:
                from agent.browser_session import BrowserSession
                self.browser_session = BrowserSession(headless=True)
                logger.info("BrowserSession 已创建（lazy，未启动）")
            except Exception as e:
                logger.warning(
                    "BrowserSession 初始化失败（browser 工具将不可用）: %s", e,
                )
                self.browser_session = None

        # === 04 NEW: 流式输出回调 ===
        # None 时走非流式（向后兼容老测试）；非 None 时每收到一个 LLM chunk 就调用。
        # 回调签名：callback(event: dict) -> None
        # event["type"]: "content" | "tool_call_start" | "done"
        self._stream_callback = stream_callback

    def cleanup(self):
        """清理 agent 持有的资源（调用方：RuntimeContext.shutdown）。

        幂等：多次调用安全。每个子清理都包 try/except，互不影响。
        """
        # 关闭浏览器
        if self.browser_session:
            try:
                self.browser_session.cleanup()
            except Exception as e:
                logger.warning("关闭 browser_session 失败: %s", e)
            finally:
                self.browser_session = None

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
    # 04 NEW: 流式调用 LLM
    # ------------------------------------------------------------------

    def _call_llm_streaming(self, *, messages, tools):
        """流式调用 LLM，每收到一个 chunk 调用 stream_callback。

        流式失败时 fallback 到非流式重试（带备用 client）。
        返回值与非流式路径完全兼容（SimpleNamespace 包装的 OpenAI 响应结构），
        让 _record_llm_usage / hook / tool_calls 处理代码不用改。

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

        try:
            for delta in self.llm_client.chat_completions_stream(
                messages, tools=tools,
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

                # 最后一个 chunk 的 finish_reason / usage
                if delta.get("finish_reason"):
                    finish_reason = delta["finish_reason"]
                if delta.get("usage"):
                    final_usage = delta["usage"]
        except Exception as stream_err:
            # 流式失败：先通知 callback，再 fallback 到非流式重试
            logger.warning(
                "流式调用失败，fallback 到非流式重试: %s", stream_err
            )
            from agent.llm_retry import call_with_retry
            response = call_with_retry(
                self.llm_client,
                messages,
                tools=tools,
                fallback_llm_client=self.fallback_llm_client,
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
        from types import SimpleNamespace
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
        - context：单会话内不变（记忆/技能/CLAUDE.md）
        - volatile：每轮可变（todo/reminder），不入缓存
        """
        if not self._system_prompt_built:
            # 从 config 读 language（默认 "zh"）
            language = (self.config or {}).get("language", "zh") if self.config else "zh"
            from agent.prompt_builder import build_system_prompt_layers
            layers = build_system_prompt_layers(
                memory_store=self.memory_store,
                memory_manager=self.memory_manager,
                enabled_toolsets=self.enabled_toolsets,
                language=language,
            )
            # stable + context 缓存，volatile 即时取
            self._stable_prompt = layers.stable
            self._context_prompt = layers.context
            # 兼容字段：合并结果（让旧代码用 _cached_system_prompt 的地方仍能读）
            self._cached_system_prompt = layers.render_flat()
            self._system_prompt_built = True
        return self._cached_system_prompt

    def _get_volatile_prompt(self) -> str:
        """每轮重建的 volatile 部分（05）。

        当前含：todo reminder（3 轮未更新时）、task 状态。
        不入 stable/context 缓存——直接拼到 system prompt 末尾。
        """
        parts = []
        if self.todo_manager and self.todo_manager.should_remind():
            parts.append("<todo_reminder>3 轮未更新 todo</todo_reminder>")
        # task_state 等可按需扩展
        return "\n\n".join(parts)

    def invalidate_system_prompt(self):
        """使缓存的 system prompt 失效。

        警告：这会让 prompt cache 失效，增加成本。
        只在上下文压缩等极端场景使用。

        05 优化：压缩后只重建 context 层（stable 不变，prompt cache 仍命中 stable 段）。
        """
        self._context_prompt = None
        self._cached_system_prompt = None
        self._system_prompt_built = False
        # _stable_prompt 保留（理论上整次会话内 stable 永不变）

    def run_conversation(self, user_message: str) -> str:
        """处理一条用户消息，返回助手最终响应。

        这是整个系统的核心循环。同步执行，不异步。
        """
        # === P4b final-fix C1: 每个 run_conversation 调用重置 idle 标志 ===
        # 同一 agent 实例在 autonomous lifecycle 多个 WORK 周期复用时，
        # 上一次 idle 请求不应泄漏到下一次调用。
        self._idle_requested = False

        # === P2-T6 NEW: USER_PROMPT_SUBMIT hook ===
        if (self.hooks_registry
                and self.config.get("hooks", {}).get("enabled", True)):
            try:
                user_message = self.hooks_registry.run_user_prompt_submit(
                    user_message, session_id=self.session_id or "",
                )
            except Exception as e:
                logger.warning("USER_PROMPT_SUBMIT 编排异常: %s", e)

        # === P2b-T5 NEW: drain 后台任务通知（临时，不进 history）===
        bg_notifications = []
        if self.bg_manager:
            try:
                bg_notifications = self.bg_manager.drain_notifications()
            except Exception as e:
                logger.warning("drain_notifications 异常: %s", e)
                bg_notifications = []

        # === P2c-T4 NEW: drain cron 定时消息（临时，不进 history）===
        cron_messages = []
        if self.cron_scheduler:
            try:
                cron_messages = self.cron_scheduler.drain_due()
            except Exception as e:
                logger.warning("cron drain_due 异常: %s", e)
                cron_messages = []

        # === P4a-T6 NEW: drain team inbox（临时，不进 history）===
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
                team_messages_text = ""

        # === Mem-T5 NEW: memory 检索 + 注入 ===
        relevant_memories_text = ""
        if (self.memory_retriever and self.memory_store
                and self._cached_memory_index):
            try:
                mem_cfg = (self.config or {}).get("memory", {})
                retrieval_model = mem_cfg.get("retrieval_model") or self.model
                max_results = mem_cfg.get("retrieval_max_results", 5)
                # batch2-T3: 优先用 aux_llm_router 做检索（便宜模型）
                retrieval_client = self.llm_client
                if self.aux_llm_router:
                    retrieval_client = self.aux_llm_router
                relevant_ids = self.memory_retriever(
                    query=user_message,
                    index_text=self._cached_memory_index,
                    llm_client=retrieval_client,
                    model=retrieval_model,
                    max_results=max_results,
                )
                if relevant_ids:
                    bodies = []
                    for mid in relevant_ids:
                        body = self.memory_store.load_body(mid)
                        if body:
                            bodies.append(f"[memory:{mid}]\n{body}")
                    if bodies:
                        relevant_memories_text = "\n\n".join(bodies)
            except Exception as e:
                logger.warning("memory retrieval 失败（fail-open）: %s", e)
                relevant_memories_text = ""

        # 组装实际入 history 的 user_content
        if relevant_memories_text:
            user_message_for_history = (
                f"<relevant_memories>\n{relevant_memories_text}\n</relevant_memories>\n\n"
                f"{user_message}"
            )
        else:
            user_message_for_history = user_message

        # 1. 追加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message_for_history,
        })

        # 2. 获取系统提示（第一次构建，后续缓存）
        system_prompt = self._get_system_prompt()

        # 3. 延迟导入避免循环依赖
        from model_tools import get_tool_definitions, handle_function_call

        # 4. 主循环
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

            # 消耗预算（grace call 不消耗）
            if not self._budget_grace_call:
                if not self.iteration_budget.consume():
                    break

            # 组装完整消息列表（system 临时加在最前面）
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
            ]

            # === P2b-T5 NEW: 注入后台任务通知（临时，不进 history）===
            if bg_notifications:
                notif_text = "\n".join(
                    f"[task {n['task_id']} {n['status']}] "
                    f"exit={n.get('exit_code')} "
                    f"stdout_tail={(n.get('stdout') or '')[-200:]}"
                    for n in bg_notifications
                )
                messages.append({
                    "role": "user",
                    "content": f"<task_notification>\n{notif_text}\n</task_notification>",
                })
                # 本轮通知已注入，清空避免后续轮次重复
                bg_notifications = []

            # === P2c-T4 NEW: 注入 cron 定时消息（临时，不进 history）===
            if cron_messages:
                sched_text = "\n".join(
                    f"[Scheduled: {m['job_id']}] {m['message']}"
                    for m in cron_messages
                )
                messages.append({
                    "role": "user",
                    "content": f"<scheduled_message>\n{sched_text}\n</scheduled_message>",
                })
                # 本轮注入后清空，避免后续轮次重复
                cron_messages = []

            # === P4a-T6 NEW: 注入 team messages（临时，不进 history）===
            if team_messages_text:
                messages.append({
                    "role": "user",
                    "content": f"<team_messages>\n{team_messages_text}\n</team_messages>",
                })
                # 本轮注入后清空，避免后续轮次重复
                team_messages_text = ""

            # TodoWrite 提醒：3 轮未更新时注入 reminder（临时，不进 history）
            if self.todo_manager and self.todo_manager.should_remind():
                reminder = self.todo_manager.format_for_reminder()
                if reminder:
                    messages.append({"role": "user", "content": reminder})

            # === PlanMode NEW: 计划模式提醒（临时，不进 history）===
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

            # 上下文压缩（接近 token 上限时触发）
            # Phase 1 Commit 7：双轨期结束，直接走新管线
            if self.compression_enabled:
                if not hasattr(self, "_compress_session_state"):
                    from agent.context_pipeline import CompressionSessionState
                    self._compress_session_state = CompressionSessionState()
                from agent.context_pipeline import compress_if_needed
                ctx_cfg = self.config.get("context", {})
                messages, compressed = compress_if_needed(
                    messages,
                    llm_client=self.llm_client,
                    model=self.model,
                    config=ctx_cfg,
                    session_state=self._compress_session_state,
                    agent_home=self.harvil_home,
                    session_id=self.session_id,
                )
                if compressed:
                    # === batch2-T1: 压缩前调 memory_manager.on_pre_compress 提取事实 ===
                    if self.memory_manager:
                        try:
                            self.memory_manager.on_pre_compress(None, messages)
                        except Exception as e:
                            logger.warning("on_pre_compress 编排异常: %s", e)
                    # 压缩会修改历史，需要同步并重建 system prompt
                    self.conversation_history = messages[1:]  # 跳过 system
                    self.invalidate_system_prompt()
                    system_prompt = self._get_system_prompt()
                    self._compression_attempts += 1
                    # === PostCompressReanchor NEW: 重新锚定当前工作上下文 ===
                    # 压缩后 LLM 看到的是"刚醒来"状态，可能丢失：
                    # - 当前 todo 进度（在内存但不在 conversation_history）
                    # - plan_mode 状态（虽然有每轮 reminder 但压缩打断节奏）
                    # 注入一条临时消息到当轮 messages 末尾，不进 conversation_history。
                    brief_parts = [
                        "你刚经历了上下文压缩，历史已被总结。"
                        "身份和 system prompt 不变。"
                    ]
                    if self.todo_manager:
                        try:
                            todo_brief = self.todo_manager.format_for_reminder()
                            if todo_brief:
                                brief_parts.append(f"当前任务清单：\n{todo_brief}")
                        except Exception as e:
                            logger.warning("读取 todo 摘要失败（brief 跳过 todo 行）: %s", e)
                    mode_text = (
                        "计划模式（只能调研，不能修改）"
                        if self.plan_mode
                        else "正常执行模式"
                    )
                    brief_parts.append(f"当前模式：{mode_text}")
                    brief_parts.append("请继续之前的工作。")
                    messages.append({
                        "role": "user",
                        "content": (
                            "<post_compress_brief>\n"
                            + "\n".join(brief_parts)
                            + "\n</post_compress_brief>"
                        ),
                    })

            # === PlanMode: plan_mode 下强制切到 plan 工具集（只读）===
            # 注意：必须在循环内每轮重算，让 plan_mode 中途切换（如审批通过）
            # 后能立即刷新工具集，否则 LLM 看到的还是 ['plan']，无法执行计划。
            effective_toolsets = ["plan"] if self.plan_mode else self.enabled_toolsets
            tool_schemas = get_tool_definitions(effective_toolsets, agent=self)

            # === batch2-T2: PRE_LLM_CALL hook（压缩后、调 LLM 前）===
            if (self.hooks_registry
                    and self.config.get("hooks", {}).get("enabled", True)):
                try:
                    messages, tool_schemas = self.hooks_registry.run_pre_llm_call(
                        messages, tool_schemas, session_id=self.session_id or "",
                    )
                except Exception as e:
                    logger.warning("PRE_LLM_CALL hook 编排异常: %s", e)

            # 调用 LLM（带重试和备用 client）
            try:
                # 04 NEW: 流式分支（stream_callback 设置时启用）
                if self._stream_callback is not None:
                    response = self._call_llm_streaming(
                        messages=messages,
                        tools=tool_schemas if tool_schemas else None,
                    )
                else:
                    from agent.llm_retry import call_with_retry
                    response = call_with_retry(
                        self.llm_client,
                        messages,
                        tools=tool_schemas if tool_schemas else None,
                        fallback_llm_client=self.fallback_llm_client,
                    )
            except Exception as e:
                # reactive_compact：API 报 prompt_too_long 时紧急压缩并重试（每会话一次）
                err_str = str(e).lower()
                is_prompt_too_long = (
                    "prompt_too_long" in err_str
                    or "context_length" in err_str
                    or "maximum context" in err_str
                )
                if (is_prompt_too_long
                        and not getattr(self, "_reacted", False)):
                    from agent.context_pipeline import reactive_compact
                    if not hasattr(self, "_compress_session_state"):
                        from agent.context_pipeline import CompressionSessionState
                        self._compress_session_state = CompressionSessionState()
                    messages, _ = reactive_compact(
                        messages,
                        session_state=self._compress_session_state,
                        keep_recent=self.config.get("context", {}).get(
                            "reactive_keep_recent", 5),
                    )
                    self._reacted = True
                    self.conversation_history = messages[1:]  # 跳过 system
                    self.invalidate_system_prompt()
                    system_prompt = self._get_system_prompt()
                    logger.warning("reactive_compact 后重试本轮")
                    continue  # 重试本轮
                logger.error("LLM API 调用失败（重试后）: %s", e)
                # 错误也作为助手消息塞回，让模型有机会自我修正
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[API 错误: {e}]",
                })
                break

            api_call_count += 1
            # batch1-T2: 记录 LLM 用量（prompt cache 记账）
            self._record_llm_usage(response)

            # === batch2-T2: POST_LLM_CALL hook（LLM 返回后、处理 tool_calls 前）===
            if (self.hooks_registry
                    and self.config.get("hooks", {}).get("enabled", True)):
                try:
                    response = self.hooks_registry.run_post_llm_call(
                        response, session_id=self.session_id or "",
                    )
                except Exception as e:
                    logger.warning("POST_LLM_CALL hook 编排异常: %s", e)

            # 每轮 LLM 调用后递增 todo 计数
            if self.todo_manager:
                self.todo_manager.increment_round()
            # C1 修复：同步递增压缩会话状态轮次，L4 cooldown 依赖此值
            if not hasattr(self, "_compress_session_state"):
                from agent.context_pipeline import CompressionSessionState
                self._compress_session_state = CompressionSessionState()
            self._compress_session_state.increment_turn()
            assistant_msg = response.choices[0].message

            # 处理工具调用
            if assistant_msg.tool_calls:
                # 先把 assistant 消息（带 tool_calls）追加到历史
                self.conversation_history.append({
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
                })

                # 执行每个工具调用
                for tc in assistant_msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # 通知 CLI 打印进度
                    if self.on_tool_call:
                        try:
                            self.on_tool_call(tool_name, tool_args)
                        except Exception:
                            pass

                    # 分发到工具注册表
                    result = handle_function_call(
                        tool_name, tool_args,
                        session_id=self.session_id,
                        memory_store=self.memory_store,
                        session_store=self.session_store,
                        harvil_home=self.harvil_home,
                        tool_call_id=tc.id,
                        config=self.config,
                        hooks_registry=self.hooks_registry,  # === P2-T7 NEW ===
                        bg_manager=self.bg_manager,          # === P2b-T7 NEW ===
                        team_bus=self.team_bus,              # === P4a-T6 NEW ===
                        team_coordinator=self.team_coordinator,  # === P4a-T6 NEW ===
                        team_name=self.team_name,            # === P4a-T6 NEW ===
                        agent_ref=self,                     # === P4b-T2 NEW ===
                    )

                    # === PlanMode NEW: 捕获 exit_plan_mode 审批请求 ===
                    plan_handled = False
                    try:
                        result_data = json.loads(result) if isinstance(result, str) else {}
                    except (json.JSONDecodeError, ValueError):
                        result_data = {}

                    if result_data.get("error_type") == "plan_approval_required":
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
                            tool_content = json.dumps({
                                "plan_approved": True,
                                "message": "用户已批准计划。现在可以开始执行：用 todo_write 列步骤然后执行。",
                            }, ensure_ascii=False)
                        else:
                            tool_content = json.dumps({
                                "plan_rejected": True,
                                "feedback": feedback or "用户未提供拒绝原因",
                                "message": "用户拒绝了计划。请根据 feedback 修订后重新调 exit_plan_mode。",
                            }, ensure_ascii=False)

                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": tool_content,
                        })
                        plan_handled = True

                    if not plan_handled:
                        # 工具结果追加到历史（必须配对 tool_call_id）
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": result,  # JSON 字符串
                        })

                # === P4b-T2 NEW: idle 标志检查 ===
                if self._idle_requested:
                    logger.info("idle 已请求，退出 run_conversation")
                    break

                # 继续循环，让 LLM 看到工具结果
                continue
            else:
                # 没有工具调用 = 最终响应
                final_content = assistant_msg.content or ""

                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content,
                })

                if self.on_response:
                    try:
                        self.on_response(final_content)
                    except Exception:
                        pass

                # 异步写入外部记忆 provider（不阻塞）
                self._sync_memory(user_message, final_content)

                # === P2-T6 NEW: STOP hook ===
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
                        continue  # 跳回 while，不 return

                return final_content

        # 循环结束（预算耗尽或中断）
        if turn_exit_reason == "interrupted_by_user":
            fallback = "[已被用户中断]"
        else:
            fallback = "[已达最大迭代次数，强制停止]"

        self.conversation_history.append({
            "role": "assistant",
            "content": fallback,
        })
        # 异步写入外部记忆 provider（即使被打断也保留部分上下文）
        self._sync_memory(user_message, fallback)
        return fallback

    def chat(self, message: str) -> str:
        """简单接口：发一条消息，返回响应。"""
        return self.run_conversation(message)

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
