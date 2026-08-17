"""LLM 调用抽象层：统一 OpenAI 兼容格式和 Anthropic 原生格式。

两种 client 都实现 chat_completions()，返回 OpenAI 兼容的响应结构
（response.choices[0].message.content / tool_calls），让 AIAgent 循环
不用关心底层 SDK 差异。

格式选择由 model_config["format"] 决定：
  - "openai"    → OpenAICompatClient（DeepSeek/OpenAI/OpenRouter 等）
  - "anthropic" → AnthropicClient（Claude 原生 API）
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

# 模块级导入 AsyncOpenAI，便于测试用 patch("agent.llm_client.AsyncOpenAI") 替换。
# （局部 import 无法被 unittest.mock.patch 定位到模块属性）
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# R17 #12：流空闲看门狗默认值（对齐 CCB STREAM_IDLE_TIMEOUT_MS=90s）
DEFAULT_STREAM_IDLE_TIMEOUT = 90.0


class LLMStreamIdleTimeout(Exception):
    """流式空闲超时（看门狗触发）：idle_timeout 秒内没有任何 chunk/event。

    上层（扣留-恢复，R17 #9）捕获后转非流式重试，而不是直接报错。
    """


async def _iterate_with_watchdog(
    stream: AsyncIterator,
    *,
    idle_timeout: float,
    describe: str = "",
):
    """带空闲看门狗的 async 迭代器包装（R17 #12）。

    idle_timeout 秒内没等到下一个 item → 尽力关闭流 → 抛 LLMStreamIdleTimeout。
    idle_timeout <= 0 表示禁用（原样迭代，向后兼容）。

    实现：async for 无法直接加超时，改手工 __anext__ + asyncio.wait_for。
    超时时 wait_for 会取消挂起的 __anext__（取消底层接收协程，安全）。
    与 CC 的差异：CC 另有「半超时 warning + >30s 事件间隔停顿计数」遥测，
    OmniMate 无对应遥测通道，只做超时 abort（记录差异）。
    """
    if idle_timeout <= 0:
        async for item in stream:
            yield item
        return
    it = stream.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=idle_timeout)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            closer = getattr(stream, "close", None)
            if closer is not None:
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass  # 关流失败无所谓：__anext__ 已被取消
            raise LLMStreamIdleTimeout(
                f"流式空闲超时（{idle_timeout:.0f}s 无数据{describe}），已中止流"
            )
        yield item


# ---------------------------------------------------------------------------
# usage 抽取（OpenAI / Anthropic SDK 字段名不同，统一为标准化 dict）
# ---------------------------------------------------------------------------

def _extract_openai_usage(usage_obj) -> Optional[dict]:
    """从 OpenAI 兼容 SDK 的 usage 对象提取标准化 token dict。

    返回 None 表示无 usage 对象。返回 dict 含 5 字段：
    prompt_tokens / completion_tokens / total_tokens /
    cache_read / cache_creation（DeepSeek/OpenAI 各自命名都做兜底）。

    被 LLMClient/OpenAICompatClient 的流式路径共用，避免字段提取逻辑重复。
    """
    if usage_obj is None:
        return None
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        "cache_read": getattr(usage_obj, "prompt_cache_hit_tokens", 0)
            or getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage_obj, "prompt_cache_miss_tokens", 0)
            or getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    }


def _extract_anthropic_usage(usage_obj) -> Optional[dict]:
    """从 Anthropic SDK 的 usage 对象提取标准化 token dict。

    Anthropic 用 input_tokens/output_tokens 命名；映射回 OpenAI 标准。
    total_tokens 由 input+output 计算（Anthropic 不直接给 total）。
    """
    if usage_obj is None:
        return None
    input_t = getattr(usage_obj, "input_tokens", 0) or 0
    output_t = getattr(usage_obj, "output_tokens", 0) or 0
    return {
        "prompt_tokens": input_t,
        "completion_tokens": output_t,
        "total_tokens": input_t + output_t,
        "cache_read": getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    }


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class LLMClient:
    """LLM 调用抽象基类。"""

    async def chat_completions(
        self,
        messages: List[dict],
        *,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """async 调用 LLM，返回 OpenAI 兼容的响应结构。

        返回对象必须有：
            response.choices[0].message.content  (str or None)
            response.choices[0].message.tool_calls  (list or None)
        """
        raise NotImplementedError

    async def chat_completions_stream(
        self,
        messages: List[dict],
        *,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """async 流式调用 LLM，yield dict chunk。

        每个 chunk 是 dict（不是 SDK 对象，避免上层处理多种 SDK 差异）：
            {
                "content": str,           # 本 chunk 增量文本（可空字符串）
                "tool_calls": list,       # 本 chunk 增量工具调用（可空列表）
                "finish_reason": str?,    # 仅最后一个 chunk 有
                "usage": dict?,           # 仅最后一个 chunk 有（可选）
            }

        默认实现：调非流式接口后模拟一次性 yield（让无流式能力的 client 也能用）。
        子类重写真流式。
        """
        resp = await self.chat_completions(messages, tools=tools, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        usage_dict = _extract_openai_usage(getattr(resp, "usage", None))
        yield {
            "content": msg.content or "",
            "tool_calls": list(msg.tool_calls or []),
            "finish_reason": getattr(choice, "finish_reason", "stop"),
            "usage": usage_dict,
        }

    def reset_client(self) -> None:
        """R26 #10：重建底层 HTTP client（连接重置后弃用旧连接池）。

        基类 no-op；子类按需重写。调用方保证线程安全（重试路径串行）。
        """


# ---------------------------------------------------------------------------
# OpenAI 兼容 client（DeepSeek/OpenAI/OpenRouter/本地 Ollama 等）
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    """OpenAI 兼容格式的 LLM client（async）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        # R26 #10：留存认证凭据，reset_client 重建时用（不依赖 SDK 暴露读取）
        self._api_key = api_key
        # R17 #12：流空闲看门狗（秒；<=0 禁用）
        self.stream_idle_timeout = stream_idle_timeout

    def reset_client(self) -> None:
        """R26 #10：丢弃可能坏死的连接池，重建 AsyncOpenAI。"""
        try:
            # best-effort 关旧 client（已坏也无妨）
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                if loop is not None:
                    loop.create_task(self.client.close())  # noqa
            except RuntimeError:
                pass
        except Exception:
            pass
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self._api_key)

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        """async 直接转发到 AsyncOpenAI SDK。返回原生的 OpenAI 响应对象。"""
        # 防御：上层（memory_manager / reflection / retriever）习惯在 kwargs 里
        # 传 model=xxx，但 self.model 已经是 client 的属性，重复传会让 OpenAI SDK
        # 报 "got multiple values for keyword argument 'model'"。这里 pop 掉。
        kwargs.pop("model", None)
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            **kwargs,
        )

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        """async 流式：stream=True，async for chunk。

        每个 chunk 是规范化后的 dict（见 LLMClient.chat_completions_stream 文档）。
        tool_calls 的 delta 保留 SDK 原对象（含 index/id/function 字段），
        上层负责按 index 累积。
        """
        # 同 chat_completions，防御 kwargs 里的 model 重复
        kwargs.pop("model", None)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        # R17 #12：空闲看门狗（90s 无 chunk → 中止流）
        async for chunk in _iterate_with_watchdog(
            stream,
            idle_timeout=self.stream_idle_timeout,
            describe=f"，model={self.model}",
        ):
            usage_dict = _extract_openai_usage(getattr(chunk, "usage", None))
            if not chunk.choices:
                # 最后一个 chunk 可能只有 usage
                if usage_dict:
                    yield {
                        "content": "",
                        "tool_calls": [],
                        "finish_reason": None,
                        "usage": usage_dict,
                    }
                continue
            delta = chunk.choices[0].delta
            yield {
                "content": delta.content or "",
                "tool_calls": list(delta.tool_calls or []),
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": usage_dict,
            }


# ---------------------------------------------------------------------------
# Anthropic 原生 client（Claude 系列）
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Anthropic 原生格式的 LLM client（async）。

    Claude API 和 OpenAI 格式的关键差异：
    - system 是顶层参数（不在 messages 数组里）
    - 工具 schema 格式不同（input_schema vs parameters）
    - 响应 content 是 block 数组（text/tool_use），不是单一字符串

    这里做双向转换，对外暴露 OpenAI 兼容接口。
    底层使用 AsyncAnthropic，所有 LLM 调用走 async/await。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "",
        base_url: str = None,
        *,
        auth_token: str = None,
        effort_level: str = None,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
    ):
        """创建 Anthropic async client。

        认证方式(二选一):
        - api_key:用 x-api-key header(Anthropic 官方)
        - auth_token:用 Authorization: Bearer header(DeepSeek Anthropic 端点)

        effort_level(思考强度,类 业界 的 CLAUDE_CODE_EFFORT_LEVEL):
        - "max": 85% 的 max_tokens 给思考预算(最强推理)
        - "high": 50% 给思考
        - "medium": 25% 给思考
        - "low" / None: 不思考(直接回答)
        """
        from anthropic import AsyncAnthropic
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        # auth_token 优先(DeepSeek 等 Anthropic 兼容端点用 Bearer)
        if auth_token:
            kwargs["auth_token"] = auth_token
        elif api_key:
            kwargs["api_key"] = api_key
        else:
            raise ValueError("AnthropicClient 需要 api_key 或 auth_token")
        self.client = AsyncAnthropic(**kwargs)
        self.model = model
        self.effort_level = (effort_level or "").lower() or None
        # R26 #10：留存认证/端点凭据，reset_client 重建时用（同名重建 AsyncAnthropic）
        self._api_key = api_key
        self._auth_token = auth_token
        self._base_url = base_url
        # R17 #12：流空闲看门狗（秒；<=0 禁用）
        self.stream_idle_timeout = stream_idle_timeout

    def reset_client(self) -> None:
        """R26 #10：丢弃可能坏死的连接池，重建 AsyncAnthropic（同名重建）。"""
        try:
            # best-effort 关旧 client（已坏也无妨）
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                if loop is not None:
                    loop.create_task(self.client.close())  # noqa
            except RuntimeError:
                pass
        except Exception:
            pass
        from anthropic import AsyncAnthropic
        kwargs = {}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # auth_token 优先(与 __init__ 同序)，兜底 api_key
        if self._auth_token:
            kwargs["auth_token"] = self._auth_token
        elif self._api_key:
            kwargs["api_key"] = self._api_key
        self.client = AsyncAnthropic(**kwargs)

    # effort_level → 思考参数(DeepSeek 格式)
    # 参考: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
    # DeepSeek 的 Anthropic 端点不用 Anthropic 原生的 budget_tokens,
    # 而是用 output_config.effort 控制思考强度。

    def _build_thinking_config(self):
        """返回 DeepSeek 思考模式开关参数。

        DeepSeek 默认思考模式 enabled。
        effort_level=low 时显式关闭,其余都 enabled。
        """
        if not self.effort_level or self.effort_level == "low":
            return None
        return {"type": "enabled"}

    def _build_output_config(self):
        """返回 DeepSeek 思考强度(output_config.effort)。

        DeepSeek Anthropic 格式:output_config={"effort": "max"/"high"}。
        medium 映射为 high(DeepSeek 兼容策略)。
        """
        if not self.effort_level or self.effort_level == "low":
            return None
        effort_map = {"max": "max", "high": "high", "medium": "high"}
        return {"effort": effort_map.get(self.effort_level, "high")}

    def _build_anthropic_kwargs(
        self,
        messages: list,
        tools: Optional[List[dict]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """构造 Anthropic API 调用参数（chat_completions 和 stream 共用）。

        含：system 拼接、消息转换、工具转换、思考模式 + output_config。
        抽出这个 helper 消除两个 chat_completions* 方法的重复（~25 行）。
        """
        system_parts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        conversation = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": conversation,
            "max_tokens": max_tokens,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # effort_level:思考模式(DeepSeek 格式:thinking 开关 + output_config 强度)
        thinking = self._build_thinking_config()
        if thinking:
            kwargs["thinking"] = thinking
        # output_config 是 DeepSeek 扩展参数,用 extra_body 传(不在 Anthropic SDK 标准字段里)
        output_config = self._build_output_config()
        if output_config:
            kwargs["extra_body"] = {"output_config": output_config}

        return kwargs

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        """async 调 anthropic.messages.create。"""
        max_tokens = kwargs.get("max_tokens", 4096)
        create_kwargs = self._build_anthropic_kwargs(messages, tools, max_tokens)
        response = await self.client.messages.create(**create_kwargs)
        return self._wrap_response(response)

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        """Anthropic 原生流式：async with messages.stream。"""
        max_tokens = kwargs.get("max_tokens", 4096)
        stream_kwargs = self._build_anthropic_kwargs(messages, tools, max_tokens)

        # 累积 tool_use（Anthropic 流式按 block 增量，需要聚合 id+name+完整 input）
        tool_buffers: Dict[int, Dict[str, Any]] = {}
        current_tool_idx: Optional[int] = None

        async with self.client.messages.stream(**stream_kwargs) as stream:
            # R17 #12：空闲看门狗（90s 无 event → 中止流；async with 兜底清理）
            async for event in _iterate_with_watchdog(
                stream,
                idle_timeout=self.stream_idle_timeout,
                describe=f"，model={self.model}",
            ):
                evt_type = getattr(event, "type", "")
                if evt_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is not None and getattr(block, "type", "") == "tool_use":
                        # 新 tool_use 块开始
                        idx = len(tool_buffers)
                        tool_buffers[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input_json": "",
                        }
                        current_tool_idx = idx
                elif evt_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            yield {
                                "content": text,
                                "tool_calls": [],
                                "finish_reason": None,
                                "usage": None,
                            }
                    elif delta_type == "input_json_delta":
                        # 工具参数 JSON 分片累积
                        partial = getattr(delta, "partial_json", "") or ""
                        if current_tool_idx is not None and partial:
                            tool_buffers[current_tool_idx]["input_json"] += partial
                elif evt_type == "content_block_stop":
                    current_tool_idx = None

            # 流结束：聚合 tool_calls 一次性 yield
            final_message = await stream.get_final_message()
            tool_calls_out = []
            for idx in sorted(tool_buffers.keys()):
                buf = tool_buffers[idx]
                tool_calls_out.append(SimpleNamespace(
                    id=buf["id"],
                    index=idx,
                    type="function",
                    function=SimpleNamespace(
                        name=buf["name"],
                        arguments=buf["input_json"] or "{}",
                    ),
                ))
            usage_dict = _extract_anthropic_usage(getattr(final_message, "usage", None))
            # 从 final_message 提取 thinking(DeepSeek 工具调用回传需要)
            thinking_text = ""
            thinking_sig = ""
            for block in getattr(final_message, "content", []):
                if getattr(block, "type", "") == "thinking":
                    thinking_text += getattr(block, "thinking", "")
                    thinking_sig = getattr(block, "signature", "") or thinking_sig
            yield {
                "content": "",
                "tool_calls": tool_calls_out,
                "finish_reason": "tool_calls" if tool_calls_out else "stop",
                "usage": usage_dict,
                "reasoning_content": thinking_text or None,
                "thinking_signature": thinking_sig or None,
            }

    def _convert_messages_to_anthropic(self, messages: list) -> list:
        """把 OpenAI 格式的消息列表转成 Anthropic 格式。

        关键:合并连续的 tool 消息成一个 user(tool_result × N)。
        Anthropic 协议要求所有 tool_results 在同一个 user 消息里,
        不允许连续多个 user 消息(OpenAI 允许每个 tool_result 独立)。
        """
        conversation = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m.get("role")

            if role == "system":
                i += 1
                continue

            if role == "tool":
                # 合并连续的 tool 消息成一个 user(tool_result × N)
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i].get("tool_call_id", ""),
                        "content": messages[i].get("content") or "",
                    })
                    i += 1
                conversation.append({"role": "user", "content": tool_results})
            else:
                conversation.append(self._convert_message(m))
                i += 1

        return conversation

    def _convert_message(self, msg: dict) -> dict:
        """OpenAI 消息 → Anthropic 消息。

        工具调用轮次回传 thinking block(DeepSeek 要求)。
        """
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            blocks = []
            # 工具调用时必须回传 thinking(DeepSeek 思考模式要求)
            rc = msg.get("reasoning_content")
            sig = msg.get("thinking_signature")
            if rc and sig:
                blocks.append({"type": "thinking", "thinking": rc, "signature": sig})

            has_tool_calls = bool(msg.get("tool_calls"))
            if has_tool_calls or blocks:
                # 有 tool_calls 或 thinking → 用 blocks 格式
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in (msg.get("tool_calls") or []):
                    try:
                        args = tc.get("function", {}).get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": args,
                    })
                return {"role": "assistant", "content": blocks}

        if role == "tool":
            # tool 结果：Anthropic 用 user 角色 + tool_result block
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content or "",
                }],
            }

        # 普通 user/assistant
        return {"role": role, "content": content or ""}

    def _convert_tools(self, openai_tools: List[dict]) -> List[dict]:
        """OpenAI 工具 schema → Anthropic 工具 schema。"""
        anthropic_tools = []
        for t in openai_tools:
            if t.get("type") == "function":
                func = t["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    # OmniMate 内部 schema 用 input_schema（不是 OpenAI 的 parameters）
                    # 两者兼容：先 parameters 后 input_schema 再 fallback 空
                    "input_schema": func.get("parameters") or func.get("input_schema") or {
                        "type": "object",
                        "properties": {},
                    },
                })
        return anthropic_tools

    def _wrap_response(self, anthropic_response):
        """把 Anthropic 响应包装成 OpenAI 兼容格式(SimpleNamespace)。

        保留 thinking 块的 reasoning_content + signature(DeepSeek 工具调用回传需要)。
        """
        content_text = ""
        tool_calls = None
        reasoning_content = ""
        thinking_signature = ""

        for block in anthropic_response.content:
            if block.type == "thinking":
                # DeepSeek 思考模式:thinking 块含思维链 + signature
                reasoning_content += getattr(block, "thinking", "")
                thinking_signature = getattr(block, "signature", "") or thinking_signature
            elif block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    ),
                ))

        message = SimpleNamespace(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            # 工具调用时,后续请求必须回传 thinking(DeepSeek 要求)
            reasoning_content=reasoning_content or None,
            thinking_signature=thinking_signature or None,
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=getattr(anthropic_response.usage, "input_tokens", 0),
                completion_tokens=getattr(anthropic_response.usage, "output_tokens", 0),
            ) if hasattr(anthropic_response, "usage") else None,
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_llm_client(model_config: Dict[str, Any]) -> LLMClient:
    """根据 model_config 的 format 字段创建对应 client。

    model_config 结构：
        {
            "format": "openai" | "anthropic",
            "base_url": "...",
            "api_key": "...",
            "model": "...",
        }
    """
    fmt = (model_config.get("format") or "openai").lower()
    api_key = model_config.get("api_key") or ""
    model = model_config.get("model") or ""
    base_url = model_config.get("base_url")
    # R17 #12：流空闲看门狗（config llm.stream_idle_timeout_seconds，默认 90s，<=0 禁用）
    try:
        idle_timeout = float(model_config.get("stream_idle_timeout", DEFAULT_STREAM_IDLE_TIMEOUT))
    except (TypeError, ValueError):
        idle_timeout = DEFAULT_STREAM_IDLE_TIMEOUT

    if fmt == "anthropic":
        # auth_token 用于 DeepSeek 等 Anthropic 兼容端点(Bearer 认证)
        auth_token = model_config.get("auth_token") or ""
        effort_level = model_config.get("effort_level") or ""
        return AnthropicClient(
            api_key=api_key or None,
            auth_token=auth_token or None,
            model=model,
            base_url=base_url,
            effort_level=effort_level or None,
            stream_idle_timeout=idle_timeout,
        )

    # 默认 openai 兼容
    return OpenAICompatClient(
        base_url=base_url, api_key=api_key, model=model,
        stream_idle_timeout=idle_timeout,
    )


async def aclose_llm_client(client) -> None:
    """best-effort 关闭底层 SDK client（AsyncOpenAI/AsyncAnthropic 都暴露 .client.close 协程；fail-open）。"""
    try:
        inner = getattr(client, "client", None)
        if inner is not None and hasattr(inner, "close"):
            result = inner.close()
            if asyncio.iscoroutine(result):
                await result
    except Exception:
        pass


class ThreadedLLMClient(LLMClient):
    """线程上下文专用 LLM client（R26 终审 follow-up 修复）。

    问题：httpx 连接池绑定首次使用时的事件循环。daemon 线程（curator/
    progress/ticker）里 asyncio.run 每次新建循环——复用主循环绑定的
    主 client 会报 "Event loop is closed"，甚至污染主对话的连接池。

    方案：每次调用在**当前循环内**新建底层 client，用完即关（aclose）。
    成本 = 每次一次 TCP 握手（后台任务低频，可忽略）；换来彻底的跨循环
    安全（在主循环调用同样安全——loop 无关）。reset_client 继承基类
    no-op（无持久池可重建）。
    """

    def __init__(self, model_config: Dict[str, Any]):
        self._model_config = dict(model_config or {})

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        client = create_llm_client(self._model_config)
        try:
            return await client.chat_completions(messages, tools=tools, **kwargs)
        finally:
            await aclose_llm_client(client)
