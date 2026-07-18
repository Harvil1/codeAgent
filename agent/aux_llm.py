"""辅助 LLM 统一路由器（batch2-T3 + 07 升级：N 级熔断链）。

用途：为记忆检索、上下文压缩、自动记忆提取等辅助任务提供独立的 LLM 配置。
- 07 升级：从单 aux → 多 endpoint 链 + 熔断（连续失败自动跳过）
- 兼容老配置：单个 aux_config 自动转成 1 个 endpoint
- 主对话仍走主 client（通过 llm_retry 的 fallback_llm_client 配置）

熔断规则：连续 N 次失败 → 开 X 秒熔断 → 跳过此 endpoint。
所有 endpoint 失败 → 降级到主 client。
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpoint:
    """一个 LLM 端点（07）。"""
    name: str
    base_url: str
    api_key_env: str = ""          # 环境变量名（空表示用 api_key_default）
    api_key_default: str = ""      # 占位字符串（Ollama 用 "ollama"）
    model: str = ""
    priority: int = 100            # 数字越小优先级越高
    enabled: bool = True
    format: str = "openai"         # openai / anthropic


class AuxLLMRouter:
    """辅助 LLM 统一路由器（多 endpoint + 熔断）。

    用于 curator / memory retrieval / title 等后台任务。
    主对话仍走主 client。

    熔断规则：连续 CIRCUIT_FAILURE_THRESHOLD 次失败 → 熔断 CIRCUIT_OPEN_SECONDS 秒。
    """

    CIRCUIT_FAILURE_THRESHOLD = 3
    CIRCUIT_OPEN_SECONDS = 300  # 5 分钟

    def __init__(
        self,
        main_client: Any,
        main_model: Optional[str] = None,
        aux_config: Optional[Dict[str, Any]] = None,
        endpoints: Optional[List[LLMEndpoint]] = None,
    ):
        """
        参数：
            main_client: 主 LLM client（必须有 chat_completions 方法）
            main_model: 主模型名（保留参数，传给 chat_completions）
            aux_config: 老的单 aux 配置（向后兼容，自动转成 1 个 endpoint）
            endpoints: 新的多 endpoint 链（优先于 aux_config）
        """
        self._main_client = main_client
        self._main_model = main_model

        # 兼容老配置：aux_config 里有 model 字段时转成 1 个 endpoint
        raw_endpoints: List[LLMEndpoint] = []
        self._legacy_aux_api_key: Optional[str] = None  # 老配置直接传 api_key

        if endpoints:
            raw_endpoints = [e for e in endpoints if e.enabled]
        elif aux_config and aux_config.get("model"):
            raw_endpoints.append(LLMEndpoint(
                name="legacy_aux",
                base_url=aux_config.get("base_url", ""),
                api_key_env="",  # 老配置直接传 api_key
                model=aux_config["model"],
                priority=1,
                format=aux_config.get("format", "openai"),
            ))
            # 老 aux_config 直接给了 api_key，缓存起来
            self._legacy_aux_api_key = aux_config.get("api_key", "")

        # 按 priority 排序（小→大）
        raw_endpoints.sort(key=lambda e: e.priority)

        # eager 创建 client：失败的 endpoint 不进 _endpoints（让 is_aux_configured 反映真实状态）
        self._endpoints: List[LLMEndpoint] = []
        self._client_cache: Dict[str, Any] = {}
        for ep in raw_endpoints:
            client = self._create_client_for(ep)
            if client is not None:
                self._endpoints.append(ep)
                self._client_cache[ep.name] = client
            else:
                logger.warning("endpoint %s client 创建失败，已跳过", ep.name)

        # 熔断状态
        self._failure_counts: Dict[str, int] = {e.name: 0 for e in self._endpoints}
        self._circuit_open_until: Dict[str, float] = {e.name: 0 for e in self._endpoints}

        if self._endpoints:
            logger.info(
                "AuxLLMRouter 配置了 %d 个 endpoint: %s",
                len(self._endpoints),
                [e.name for e in self._endpoints],
            )

    def _create_client_for(self, ep: LLMEndpoint) -> Optional[Any]:
        """为单个 endpoint 创建 client（__init__ 时调用）。"""
        # 解析 api_key
        if ep.name == "legacy_aux" and self._legacy_aux_api_key:
            api_key = self._legacy_aux_api_key
        elif ep.api_key_env:
            api_key = os.environ.get(ep.api_key_env, "") or ep.api_key_default
            if not api_key:
                logger.debug(
                    "endpoint %s 未配置 api_key env %s", ep.name, ep.api_key_env,
                )
                return None
        else:
            api_key = ep.api_key_default

        try:
            from agent.llm_client import create_llm_client
            return create_llm_client({
                "format": ep.format,
                "base_url": ep.base_url,
                "api_key": api_key,
                "model": ep.model,
            })
        except Exception as e:
            logger.warning("创建 endpoint %s 的 client 失败: %s", ep.name, e)
            return None

    @property
    def is_aux_configured(self) -> bool:
        """是否配置了至少一个辅助 endpoint。"""
        return bool(self._endpoints)

    def _get_client(self, ep: LLMEndpoint) -> Optional[Any]:
        """懒创建 client，并缓存。返回 None 表示此 endpoint 不可用。"""
        if ep.name in self._client_cache:
            return self._client_cache[ep.name]

        # 解析 api_key
        if ep.name == "legacy_aux" and self._legacy_aux_api_key:
            api_key = self._legacy_aux_api_key
        elif ep.api_key_env:
            api_key = os.environ.get(ep.api_key_env, "") or ep.api_key_default
            if not api_key:
                logger.debug(
                    "endpoint %s 未配置 api_key env %s", ep.name, ep.api_key_env,
                )
                return None
        else:
            api_key = ep.api_key_default

        try:
            from agent.llm_client import create_llm_client
            client = create_llm_client({
                "format": ep.format,
                "base_url": ep.base_url,
                "api_key": api_key,
                "model": ep.model,
            })
            self._client_cache[ep.name] = client
            return client
        except Exception as e:
            logger.warning("创建 endpoint %s 的 client 失败: %s", ep.name, e)
            return None

    def chat_completions(self, messages: list, **kwargs):
        """依次尝试 endpoints，第一个成功就返回。

        熔断中的 endpoint 跳过。所有 endpoint 失败 → 降级到主 client。
        """
        now = time.time()
        tried: List[tuple] = []

        for ep in self._endpoints:
            # 熔断检查
            if self._circuit_open_until.get(ep.name, 0) > now:
                tried.append((ep.name, "circuit-open"))
                continue

            # client 已在 __init__ 时 eager 创建（cache 里直接读）
            client = self._client_cache.get(ep.name)
            if client is None:
                tried.append((ep.name, "no-client"))
                continue

            try:
                resp = client.chat_completions(messages, **kwargs)
                # 成功，重置失败计数
                self._failure_counts[ep.name] = 0
                return resp
            except Exception as e:
                logger.warning("endpoint %s 调用失败: %s", ep.name, e)
                self._failure_counts[ep.name] = (
                    self._failure_counts.get(ep.name, 0) + 1
                )
                tried.append((ep.name, str(e)[:100]))

                # 触发熔断
                if self._failure_counts[ep.name] >= self.CIRCUIT_FAILURE_THRESHOLD:
                    self._circuit_open_until[ep.name] = (
                        now + self.CIRCUIT_OPEN_SECONDS
                    )
                    logger.warning(
                        "endpoint %s 连续 %d 次失败，熔断 %ds",
                        ep.name, self.CIRCUIT_FAILURE_THRESHOLD,
                        self.CIRCUIT_OPEN_SECONDS,
                    )

        # 全部失败 → fallback 到主 client
        if tried:
            logger.info(
                "所有 aux endpoints 失败，降级到主 client。尝试: %s",
                tried,
            )
        return self._main_client.chat_completions(messages, **kwargs)

    # ------------------------------------------------------------------
    # 07 NEW: 熔断状态查询（测试 / 监控用）
    # ------------------------------------------------------------------

    def get_circuit_status(self) -> Dict[str, Dict]:
        """返回每个 endpoint 的熔断状态（用于监控 / 测试）。"""
        now = time.time()
        result = {}
        for ep in self._endpoints:
            open_until = self._circuit_open_until.get(ep.name, 0)
            result[ep.name] = {
                "failures": self._failure_counts.get(ep.name, 0),
                "circuit_open": open_until > now,
                "circuit_remaining_seconds": max(0, open_until - now),
            }
        return result

    def reset_circuit(self, name: Optional[str] = None) -> None:
        """手动重置熔断状态（测试 / 运维用）。

        name=None 时重置所有 endpoint。
        """
        targets = [name] if name else list(self._failure_counts.keys())
        for n in targets:
            self._failure_counts[n] = 0
            self._circuit_open_until[n] = 0
