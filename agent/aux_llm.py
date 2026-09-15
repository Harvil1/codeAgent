"""辅助 LLM 的统一调度台（多端点 + 熔断保护）。

主对话之外的一批「打杂」活（记忆检索、上下文压缩、自动提取记忆、
起标题）不值得用主模型，由便宜的小模型来干，本模块统一调度：
- 可以配一串备选端点（endpoint，即一组「API 地址 + 模型 + 密钥」），
  按优先级挨个试，谁先成功用谁
- 带熔断（保险丝）机制：某个端点连续失败就先「拉闸」跳过它一会儿，
  免得每次都白撞一遍
- 兼容旧式 aux_config 单端点配置（自动转成一个端点）
- 主对话不走这里，仍走主 client（由 llm_retry 的备胎机制负责）
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpoint:
    """一个 LLM 端点：一组「API 地址 + 模型 + 密钥」的组合。

    参数（字段）：
        name：这个端点的名字（起个别名好认）
        base_url：API 地址
        api_key_env：密钥放在哪个环境变量里（空 = 不从环境变量拿，
                    直接用 api_key_default）
        api_key_default：直接写死的占位密钥（本地 Ollama 常填 "ollama"）
        model：模型名
        priority：优先级，数字越小越先被尝试
        enabled：开关，False 就不启用
        format：API 格式，"openai" 或 "anthropic"
    """
    name: str
    base_url: str
    api_key_env: str = ""          # 环境变量名（空表示用 api_key_default）
    api_key_default: str = ""      # 占位字符串（Ollama 用 "ollama"）
    model: str = ""
    priority: int = 100            # 数字越小优先级越高
    enabled: bool = True
    format: str = "openai"         # openai / anthropic


class AuxLLMRouter:
    """辅助 LLM 的调度台：多个端点排队试，配熔断保险丝。

    服侍对象是后台杂活（curator 维护工/记忆检索/起标题等），
    主对话不归它管。

    熔断规则：某端点连续失败够 CIRCUIT_FAILURE_THRESHOLD 次 →
    拉闸 CIRCUIT_OPEN_SECONDS 秒，期间跳过它。
    """

    CIRCUIT_FAILURE_THRESHOLD = 3   # 连续失败几次拉闸
    CIRCUIT_OPEN_SECONDS = 300      # 拉闸多久（5 分钟）

    def __init__(
        self,
        main_client: Any,
        aux_config: Optional[Dict[str, Any]] = None,
        endpoints: Optional[List[LLMEndpoint]] = None,
        owns_main: bool = False,
    ):
        """把端点列表准备好、造好 client、初始化熔断账本。

        参数：
            main_client：主 LLM client（兜底用，必须有 chat_completions 方法）
            aux_config：旧式单 aux 配置（自动转成一个端点）
            endpoints：新的多端点列表（给了它就无视 aux_config）
            owns_main：main_client 的所有权声明——外部传入的东西谁传谁
                       负责，默认 False（共享引用，close() 绝不越权关它）；
                       只有调用方明确说「这是我专为你造的」（True），
                       close() 才会连兜底 client 一起关
        """
        self._main_client = main_client
        self._owns_main = owns_main

        # 兼容旧配置：aux_config 里有 model 字段时，转成 1 个端点
        raw_endpoints: List[LLMEndpoint] = []
        self._legacy_aux_api_key: Optional[str] = None  # 旧配置是直接给密钥的，先存着

        if endpoints:
            raw_endpoints = [e for e in endpoints if e.enabled]
        elif aux_config and aux_config.get("model"):
            raw_endpoints.append(LLMEndpoint(
                name="legacy_aux",
                base_url=aux_config.get("base_url", ""),
                api_key_env="",  # 旧配置不走环境变量，直接给密钥
                model=aux_config["model"],
                priority=1,
                format=aux_config.get("format", "openai"),
            ))
            # 旧配置里直接写了 api_key，存起来备用
            self._legacy_aux_api_key = aux_config.get("api_key", "")

        # 按优先级排序（数字小的排前面，先被尝试）
        raw_endpoints.sort(key=lambda e: e.priority)

        # 启动时就造好 client（而不是等第一次调用才造）：造不出来的
        # 端点直接不进名单——这样「是否配了辅助模型」的查询才真实
        self._endpoints: List[LLMEndpoint] = []
        self._client_cache: Dict[str, Any] = {}
        for ep in raw_endpoints:
            client = self._create_client_for(ep)
            if client is not None:
                self._endpoints.append(ep)
                self._client_cache[ep.name] = client
            else:
                logger.warning("endpoint %s client 创建失败，已跳过", ep.name)

        # 熔断（保险丝）账本：每个端点记两笔——连败次数、拉闸到几点
        self._failure_counts: Dict[str, int] = {e.name: 0 for e in self._endpoints}
        self._circuit_open_until: Dict[str, float] = {e.name: 0 for e in self._endpoints}

        if self._endpoints:
            logger.info(
                "AuxLLMRouter 配置了 %d 个 endpoint: %s",
                len(self._endpoints),
                [e.name for e in self._endpoints],
            )

    def _create_client_for(self, ep: LLMEndpoint) -> Optional[Any]:
        """给一个端点造出它的 LLM client（初始化时逐个调用）。

        参数：
            ep：要接线的端点

        返回：造好的 client；密钥没配或创建出错则返回 None（跳过该端点）。
        """
        # 先把密钥弄到手
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

    async def close(self) -> None:
        """关掉 router 自己造/拥有的全部 LLM client（进程收尾用，fail-open）。

        - endpoint 的缓存 client：router 构造时造的，一律关
        - 兜底 main_client：外部传入——owns_main=True（调用方声明「专用」）
          才关，共享引用（比如直接传了 agent 主 client）绝不越权关闭
        """
        from agent.llm_client import aclose_llm_client
        for name, client in list(self._client_cache.items()):
            try:
                await aclose_llm_client(client)
            except Exception as e:
                logger.warning("关闭 endpoint %s 的 client 失败（fail-open）: %s", name, e)
        self._client_cache.clear()
        if self._owns_main and self._main_client is not None:
            try:
                await aclose_llm_client(self._main_client)
            except Exception as e:
                logger.warning("关闭兜底 main client 失败（fail-open）: %s", e)
            self._main_client = None

    @property
    def is_aux_configured(self) -> bool:
        """问一句：到底配没配上辅助模型？（至少有一个能用的端点才算配了）"""
        return bool(self._endpoints)

    async def chat_completions(self, messages: list, **kwargs):
        """挨个试辅助端点，谁先成功就用谁的结果；全砸了就找主 client 兜底。

        正被拉闸（熔断中）的端点直接跳过。

        本方法是 async；跨线程同步等结果用 loop_host.run_async（外部
        线程专用——宿主循环线程里禁止同步等，自己等自己死锁）。

        参数：
            messages：对话历史（消息列表）
            **kwargs：其余参数原样传给底层 client

        返回：成功端点的 LLM 响应；辅助端点全失败时返回主 client 的结果。
        """
        now = time.time()
        tried: List[tuple] = []

        for ep in self._endpoints:
            # 先看保险丝：拉闸中就跳过
            if self._circuit_open_until.get(ep.name, 0) > now:
                tried.append((ep.name, "circuit-open"))
                continue

            # client 在初始化时就造好了，这里直接取
            client = self._client_cache.get(ep.name)
            if client is None:
                tried.append((ep.name, "no-client"))
                continue

            try:
                resp = await client.chat_completions(messages, **kwargs)
                # 成功了，连败计数清零
                self._failure_counts[ep.name] = 0
                return resp
            except Exception as e:
                logger.warning("endpoint %s 调用失败: %s", ep.name, e)
                self._failure_counts[ep.name] = (
                    self._failure_counts.get(ep.name, 0) + 1
                )
                tried.append((ep.name, str(e)[:100]))

                # 连败够了，拉闸
                if self._failure_counts[ep.name] >= self.CIRCUIT_FAILURE_THRESHOLD:
                    self._circuit_open_until[ep.name] = (
                        now + self.CIRCUIT_OPEN_SECONDS
                    )
                    logger.warning(
                        "endpoint %s 连续 %d 次失败，熔断 %ds",
                        ep.name, self.CIRCUIT_FAILURE_THRESHOLD,
                        self.CIRCUIT_OPEN_SECONDS,
                    )

        if self._main_client is None:
            # close() 之后 straggler 调用会走到这——死得明白，别让人看
            # NoneType.chat_completions 的莫名报错（对齐 loop_host 停机风格）
            raise RuntimeError("AuxLLMRouter 已关闭")

        # 辅助端点全军覆没 → 请主 client 出面兜底
        if tried:
            logger.info(
                "所有 aux endpoints 失败，降级到主 client。尝试: %s",
                tried,
            )
        return await self._main_client.chat_completions(messages, **kwargs)

    # ------------------------------------------------------------------
    # 熔断状态查询（测试 / 监控用）
    # ------------------------------------------------------------------
