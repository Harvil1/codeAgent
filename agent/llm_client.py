"""LLM 调用抽象层：统一 OpenAI 兼容格式和 Anthropic 原生格式。

两种 client 都实现 chat_completions()，返回 OpenAI 兼容的响应结构
（response.choices[0].message.content / tool_calls），让 AIAgent 循环
不用关心底层 SDK 差异。

格式选择由 model_config["format"] 决定：
  - "openai"    → OpenAICompatClient（DeepSeek/OpenAI/OpenRouter 等）
  - "anthropic" → AnthropicClient（Claude 原生 API）
"""

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class LLMClient:
    """LLM 调用抽象基类。"""

    def chat_completions(
        self,
        messages: List[dict],
        *,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """调用 LLM，返回 OpenAI 兼容的响应结构。

        返回对象必须有：
            response.choices[0].message.content  (str or None)
            response.choices[0].message.tool_calls  (list or None)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI 兼容 client（DeepSeek/OpenAI/OpenRouter/本地 Ollama 等）
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    """OpenAI 兼容格式的 LLM client。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url

    def chat_completions(self, messages, *, tools=None, **kwargs):
        """直接转发到 OpenAI SDK。返回原生的 OpenAI 响应对象。"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Anthropic 原生 client（Claude 系列）
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Anthropic 原生格式的 LLM client。

    Claude API 和 OpenAI 格式的关键差异：
    - system 是顶层参数（不在 messages 数组里）
    - 工具 schema 格式不同（input_schema vs parameters）
    - 响应 content 是 block 数组（text/tool_use），不是单一字符串

    这里做双向转换，对外暴露 OpenAI 兼容接口。
    """

    def __init__(self, api_key: str, model: str, base_url: str = None):
        import anthropic
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat_completions(self, messages, *, tools=None, **kwargs):
        """把 OpenAI 格式的输入转成 Anthropic 格式，调用后包装返回。"""
        # 1. 分离 system 消息
        system_parts = []
        conversation = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                content = m.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                conversation.append(self._convert_message(m))

        system = "\n\n".join(system_parts) if system_parts else None

        # 2. 转换工具格式
        anthropic_tools = self._convert_tools(tools) if tools else None

        # 3. 调用 Anthropic
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=conversation,
            tools=anthropic_tools,
            max_tokens=kwargs.get("max_tokens", 4096),
        )

        # 4. 包装成 OpenAI 兼容响应
        return self._wrap_response(response)

    def _convert_message(self, msg: dict) -> dict:
        """OpenAI 消息 → Anthropic 消息。"""
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and msg.get("tool_calls"):
            # assistant 带工具调用：转成 content blocks
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                try:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                except (json.JSONDecodeError, KeyError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc["function"]["name"],
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
                    "input_schema": func.get("parameters") or {
                        "type": "object",
                        "properties": {},
                    },
                })
        return anthropic_tools

    def _wrap_response(self, anthropic_response):
        """把 Anthropic 响应包装成 OpenAI 兼容格式（SimpleNamespace）。"""
        content_text = ""
        tool_calls = None

        for block in anthropic_response.content:
            if block.type == "text":
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

    if fmt == "anthropic":
        return AnthropicClient(api_key=api_key, model=model, base_url=base_url)

    # 默认 openai 兼容
    return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model)
