"""LLM API 调用的重试与错误恢复。

策略：
  - 可重试错误（429 限流、5xx 服务器错误、连接错误）：指数退避重试
  - 不可重试错误（400 参数错、401 认证错）：立即抛出
  - 主模型重试耗尽后切换备用模型（如有配置）

借鉴 Claude Code 的韧性机制。
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # 秒，指数退避起点


def is_retryable(error: Exception) -> bool:
    """判断异常是否可重试。

    可重试：限流（429）、服务器错误（5xx）、连接错误、超时。
    不可重试：参数错误（400）、认证错误（401）、权限错误（403）。
    """
    try:
        from openai import (
            RateLimitError, APIConnectionError, APITimeoutError, APIStatusError,
        )
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            # 5xx 和 429 可重试
            return error.status_code >= 500 or error.status_code == 429
    except ImportError:
        pass

    # 兜底：按异常类名判断
    error_type = type(error).__name__.lower()
    if any(kw in error_type for kw in ("timeout", "connection", "temporary")):
        return True
    return False


def get_retry_after(error: Exception) -> Optional[float]:
    """从错误中提取 Retry-After（秒）。"""
    try:
        retry_after = getattr(error, "retry_after", None)
        if retry_after:
            return float(retry_after)
    except (TypeError, ValueError):
        pass
    # openai 库的 response headers
    try:
        headers = getattr(error, "response_headers", None) or {}
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            return float(ra)
    except Exception:
        pass
    return None


def call_with_retry(
    llm_client,
    messages: list,
    *,
    tools=None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    fallback_llm_client=None,
):
    """带重试和备用 client 的 LLM 调用。

    流程：
      1. 主 client 重试 max_retries 次（指数退避）
      2. 全部失败后，如果有 fallback_llm_client，用备用 client 再试 1 次
      3. 都失败则抛最后错误

    参数：
        llm_client: LLMClient 实例（实现 chat_completions 方法）
        messages: 消息列表
        tools: 工具 schema 列表（OpenAI 格式）
        max_retries: 最大重试次数
        initial_backoff: 首次退避秒数
        fallback_llm_client: 备用 LLMClient（主 client 失败时切换）
    """
    last_error: Optional[Exception] = None

    # 主 client 重试
    for attempt in range(max_retries):
        try:
            return llm_client.chat_completions(messages, tools=tools)
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                raise
            retry_after = get_retry_after(e)
            backoff = retry_after if retry_after else initial_backoff * (2 ** attempt)
            logger.warning(
                "LLM 调用失败（尝试 %d/%d），%.1fs 后重试: %s",
                attempt + 1, max_retries, backoff, e,
            )
            time.sleep(backoff)

    # 主 client 重试耗尽，尝试备用 client
    if fallback_llm_client is not None:
        logger.warning("主 LLM client 重试耗尽，切换备用 client")
        try:
            return fallback_llm_client.chat_completions(messages, tools=tools)
        except Exception as e:
            last_error = e
            logger.error("备用 client 也失败: %s", e)

    raise last_error
