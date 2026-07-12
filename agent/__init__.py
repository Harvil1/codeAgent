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
from agent.context_compressor import maybe_compress

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
        self.harvil_home = harvil_home
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

        # 对话历史（不包含 system prompt，system 单独传）
        self.conversation_history: list = []

        # 上下文压缩配置
        self.compression_enabled = True
        self._compression_attempts = 0
        # 完整 config（用于 context.use_new_pipeline 等开关）
        self.config: dict = config or {}

    def interrupt(self):
        """请求中断（由 CLI 的 Ctrl+C 处理器调用）。

        协作式中断：不直接杀线程（可能损坏消息历史），而是设置标志。
        """
        self._interrupt_requested = True

    def _get_system_prompt(self) -> str:
        """获取系统提示。第一次调用时构建，后续返回缓存。

        缓存是为了保护 LLM provider 的 prompt cache。
        """
        if not self._system_prompt_built:
            self._cached_system_prompt = build_system_prompt(
                memory_store=self.memory_store,
                memory_manager=self.memory_manager,
                enabled_toolsets=self.enabled_toolsets,
            )
            self._system_prompt_built = True
        return self._cached_system_prompt

    def invalidate_system_prompt(self):
        """使缓存的 system prompt 失效。

        警告：这会让 prompt cache 失效，增加成本。
        只在上下文压缩等极端场景使用。
        """
        self._cached_system_prompt = None
        self._system_prompt_built = False

    def run_conversation(self, user_message: str) -> str:
        """处理一条用户消息，返回助手最终响应。

        这是整个系统的核心循环。同步执行，不异步。
        """
        # 1. 追加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # 2. 获取系统提示（第一次构建，后续缓存）
        system_prompt = self._get_system_prompt()

        # 3. 获取工具定义（过滤启用的工具集 + check_fn）
        # 延迟导入避免循环依赖
        from model_tools import get_tool_definitions, handle_function_call
        tool_schemas = get_tool_definitions(self.enabled_toolsets)

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

            # TodoWrite 提醒：3 轮未更新时注入 reminder（临时，不进 history）
            if self.todo_manager and self.todo_manager.should_remind():
                reminder = self.todo_manager.format_for_reminder()
                if reminder:
                    messages.append({"role": "user", "content": reminder})

            # 上下文压缩（接近 token 上限时触发）
            if self.compression_enabled:
                use_new = self.config.get("context", {}).get(
                    "use_new_pipeline", False,
                )
                if use_new:
                    # 新管线：L1/L2/L4 + transcript 快照
                    if not hasattr(self, "_compress_session_state"):
                        from agent.context_pipeline import CompressionSessionState
                        self._compress_session_state = CompressionSessionState()
                    from agent.context_pipeline import compress_if_needed
                    ctx_cfg = self.config.get("context", {})
                    messages, compressed = compress_if_needed(
                        messages,
                        attempt_count=self._compression_attempts,
                        llm_client=self.llm_client,
                        model=self.model,
                        config=ctx_cfg,
                        session_state=self._compress_session_state,
                        agent_home=self.harvil_home,
                        session_id=self.session_id,
                    )
                else:
                    # 旧路径（双轨期保留，已废弃）
                    messages, compressed = maybe_compress(
                        messages,
                        attempt_count=self._compression_attempts,
                        model=self.model,
                        llm_client=self.llm_client,
                    )
                if compressed:
                    # 压缩会修改历史，需要同步并重建 system prompt
                    self.conversation_history = messages[1:]  # 跳过 system
                    self.invalidate_system_prompt()
                    system_prompt = self._get_system_prompt()
                    self._compression_attempts += 1

            # 调用 LLM（带重试和备用 client）
            try:
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
                use_new = self.config.get("context", {}).get(
                    "use_new_pipeline", False,
                )
                if (is_prompt_too_long and use_new
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
            # 每轮 LLM 调用后递增 todo 计数
            if self.todo_manager:
                self.todo_manager.increment_round()
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
                    )

                    # 工具结果追加到历史（必须配对 tool_call_id）
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": result,  # JSON 字符串
                    })

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
