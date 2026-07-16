"""辅助 LLM 统一路由器（batch2-T3）。

用途：为记忆检索、上下文压缩、自动记忆提取等辅助任务提供独立的 LLM 配置。
优先用 config 中配置的便宜模型（aux_model）；未配置时 fallback 到主 client。

设计原则：
- 对外暴露和主 LLMClient 相同的 chat_completions() 接口
- aux 失败时自动降级到主 client（fail-open）
- 零配置时完全透明（直接代理主 client）
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuxLLMRouter:
    """辅助 LLM 统一路由器。

    优先用 config.aux_model 配置的便宜模型；
    未配置时 fallback 到主 client。
    """

    def __init__(
        self,
        main_client: Any,
        main_model: Optional[str] = None,
        aux_config: Optional[Dict[str, Any]] = None,
    ):
        """
        参数：
            main_client: 主 LLM client（必须有 chat_completions 方法）
            main_model: 主模型名（用于传递给 chat_completions 的 model 参数）
            aux_config: 辅助 LLM 配置字典，结构：
                {
                    "format": "openai" | "anthropic",
                    "base_url": "...",
                    "api_key": "...",
                    "model": "...",
                }
                None 时直接用主 client。
        """
        self._main_client = main_client
        self._main_model = main_model
        self._aux_client = None
        self._aux_model = None

        if aux_config and aux_config.get("model"):
            try:
                from agent.llm_client import create_llm_client
                self._aux_client = create_llm_client(aux_config)
                self._aux_model = aux_config.get("model")
                logger.info(
                    "AuxLLMRouter 已启用辅助模型: %s",
                    self._aux_model,
                )
            except Exception as e:
                logger.warning(
                    "创建 aux LLM client 失败，fallback 到主 client: %s", e
                )

    @property
    def is_aux_configured(self) -> bool:
        """是否配置了独立的辅助模型。"""
        return self._aux_client is not None

    def chat_completions(self, messages: list, **kwargs):
        """优先用 aux，fallback 到 main。

        保持与 LLMClient.chat_completions 相同的接口。
        """
        if self._aux_client is not None:
            try:
                return self._aux_client.chat_completions(messages, **kwargs)
            except Exception as e:
                logger.warning(
                    "aux LLM 调用失败，fallback 到主 client: %s", e
                )
        return self._main_client.chat_completions(messages, **kwargs)
