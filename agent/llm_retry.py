"""LLM API 调用的重试与错误恢复。

策略：
  - 可重试错误（429 限流、5xx 服务器错误、连接错误）：指数退避重试
  - 不可重试错误（400 参数错、401 认证错）：立即抛出
  - 主模型重试耗尽后切换备用模型（如有配置）
  - finish_reason=length（max_tokens 截断）：先升 max_tokens 重试，再发续写提示

借鉴 业界 的韧性机制。
"""

import asyncio
import logging
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # 秒，指数退避起点
DEFAULT_JITTER_RATIO = 0.25    # 抖动比例：sleep = base + uniform(0, base*ratio)
DEFAULT_MAX_BACKOFF = 60.0     # 单次退避上限（普通模式；X7）
UNATTENDED_MAX_BACKOFF = 300.0  # R17 #44：unattended 长跑模式退避帽 5min（对齐 CCB）

# 持久重试（unattended）模式默认值（Task P2.1）
DEFAULT_UNATTENDED_MAX_HOURS = 24  # 持续重试最长 24 小时

# max_tokens 升级默认值（P0-3 / R17 #10）
# initial=None 表示不显式传 max_tokens，让 provider SDK 用模型默认值
# escalated=64000 对齐 CCB ESCALATED_MAX_TOKENS=64k（原 32768）。
# 模型输出上限更小的 provider（如 DeepSeek 8K）会报 400 溢出——
# 由 R17 #13 的 400 溢出自适应（parse_context_overflow）动态下调兜底，
# 升级调用失败本身也 fail-open 沿用截断响应。
DEFAULT_INITIAL_MAX_TOKENS: Optional[int] = None
DEFAULT_ESCALATED_MAX_TOKENS = 64000

# R17 #10：升级后仍截断的「续写恢复」上限（对齐 CC MAX_OUTPUT_TOKENS_RECOVERY_LIMIT=3）
DEFAULT_OUTPUT_RECOVERY_LIMIT = 3

# 529 连续失败阈值（P1-1）
# Anthropic 过载（status 529）通常持续一段时间，达到阈值立即切 fallback，
# 不在已知过载的 endpoint 上浪费重试次数（避免占用限流配额）。
DEFAULT_CONSECUTIVE_529_THRESHOLD = 3

# R25 #5：退避心跳分片（对齐 CCB unattended 30s 心跳——长退避期间保持可观察）
HEARTBEAT_CHUNK_SECONDS = 30.0


async def _sleep_with_heartbeat(total: float, heartbeat_cb=None) -> None:
    """分片 sleep：每 ≤30s 一个片段，片段间调 heartbeat_cb(elapsed, total)。

    - 无 callback 或 total <= 30s：直接 sleep（零开销，向后兼容）
    - callback 抛异常：吞掉（fail-open，心跳不能影响重试本身）
    """
    if heartbeat_cb is None or total <= HEARTBEAT_CHUNK_SECONDS:
        await asyncio.sleep(total)
        return
    elapsed = 0.0
    while elapsed < total:
        chunk = min(HEARTBEAT_CHUNK_SECONDS, total - elapsed)
        await asyncio.sleep(chunk)
        elapsed += chunk
        if elapsed < total:
            try:
                heartbeat_cb(elapsed, total)
            except Exception:
                pass


def _error_status_code(error: Exception) -> Optional[int]:
    """从异常提取 HTTP 状态码（兼容多种 SDK 形态）。"""
    code = getattr(error, "status_code", None)
    if code is not None:
        return code
    code = getattr(error, "status", None)
    if code is not None:
        return code
    return None


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

    # R26 #10：httpx 传输层错误（连接重置/断管/网络错误基类）一律可重试。
    # openai SDK 会把 httpx 错误包装成 APIConnectionError 再抛（上面已命中），
    # 但 httpx 错误也可能裸透出（其它 SDK/自定义路径），类名 "transporterror"
    # 不含 "connection"/"timeout" 关键字，按 isinstance 兜底。
    try:
        import httpx
        if isinstance(error, httpx.TransportError):
            return True
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


# R26 #10：连接重置类错误——重试前重建 client（弃用坏死连接池）
_RESET_NAMES = ("connectionreset", "brokenpipe", "remoteprotocol", "readerror", "writeerror")


def _is_connection_reset(error: Exception) -> bool:
    """是否连接重置/断管类错误（按异常类名与 __cause__ 链判断）。

    对齐 CCB withRetry 的 ECONNRESET/EPIPE 禁 keep-alive 重连：
    这类错误说明底层连接坏了，在同一连接池上重试大概率复现。
    """
    name = type(error).__name__.lower()
    if any(k in name for k in _RESET_NAMES):
        return True
    cause = getattr(error, "__cause__", None)
    while cause is not None:
        cname = type(cause).__name__.lower()
        if any(k in cname for k in _RESET_NAMES):
            return True
        cause = getattr(cause, "__cause__", None)
    return False


# ---------------------------------------------------------------------------
# R17 #13：400 溢出自适应（input + max_tokens > context limit 的数值解析）
# ---------------------------------------------------------------------------

# CCB 精确格式："input length and `max_tokens` exceed context limit: 188059 + 20000 > 200000"
_OVERFLOW_EXACT_RE = re.compile(
    r"input length and `max_tokens` exceed context limit:\s*(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)"
)
# OpenAI 经典格式："This model's maximum context length is 131072 tokens.
# However, you requested 140000 tokens (120000 input tokens and 20000 max_tokens)"
_OVERFLOW_LOOSE_RE = re.compile(
    r"maximum context length is (\d+) tokens?.*?(\d+) input tokens",
    re.DOTALL,
)

# 恢复缓冲与下限（对齐 CCB：安全缓冲 1k / 下限 3000）
_OVERFLOW_SAFETY_BUFFER = 1000
_OVERFLOW_MIN_MAX_TOKENS = 3000


def parse_context_overflow(error: Exception) -> Optional[Tuple[int, int]]:
    """从 400 溢出报文解析 (input_tokens, context_limit)。

    命中两种格式其一即返回；解析失败返回 None（调用方按普通 400 处理）。
    """
    msg = str(error)
    m = _OVERFLOW_EXACT_RE.search(msg)
    if m:
        return int(m.group(1)), int(m.group(3))
    m2 = _OVERFLOW_LOOSE_RE.search(msg)
    if m2:
        return int(m2.group(2)), int(m2.group(1))
    return None


def compute_overflow_max_tokens(input_tokens: int, context_limit: int) -> Optional[int]:
    """计算恢复用 max_tokens：context_limit - input - 1000，下限 3000。

    返回 None 表示输入本身就太大（剩余空间 < 3000），无恢复价值。
    """
    new_max = context_limit - input_tokens - _OVERFLOW_SAFETY_BUFFER
    if new_max < _OVERFLOW_MIN_MAX_TOKENS:
        return None
    return new_max


async def call_with_retry(
    llm_client,
    messages: list,
    *,
    tools=None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    fallback_llm_client=None,
    jitter_ratio: float = DEFAULT_JITTER_RATIO,
    max_tokens: Optional[int] = None,
    consecutive_529_threshold: int = DEFAULT_CONSECUTIVE_529_THRESHOLD,
    config: Optional[Dict[str, Any]] = None,
    background: bool = False,  # R25 #4：后台调用（子代理摘要等）遇 529 立即放弃
    heartbeat_cb=None,  # R25 #5：长退避分片心跳（fn(elapsed, total)）
):
    """带重试和备用 client 的 async LLM 调用。

    流程：
      1. 主 client 重试 max_retries 次（指数退避 + 抖动）
      2. 连续 N 次 529（Anthropic 过载）→ 立即切 fallback（不等耗尽）
      3. 全部失败后，如果有 fallback_llm_client，用备用 client 再试 1 次
      4. 都失败则抛最后错误

    改造说明（Plan 2A Task D1）：原同步 `def` 改 `async def`；
    `llm_client.chat_completions(...)` / `fallback_llm_client.chat_completions(...)`
    加 await（LLMClient 已改 async，Task B1/B2/B3）；
    `time.sleep(...)` 改 `await asyncio.sleep(...)`（不阻塞事件循环）。
    退避/抖动/529 早切/max_tokens 升级逻辑不变。

    持久重试模式（Task P2.1）：当 config 中 `bash_unattended_retry` flag 开启时，
      - max_retries 视为无限（持续重试不因计数耗尽退出）
      - 加 deadline = time.monotonic() + max_hours * 3600
      - 循环改为 while（计数 < effective_max_retries）+ deadline 检查
      - 普通 5xx/429 重试 / 529 早切 / fallback 逻辑保留
      - flag OFF 或 config=None → 完全走原 max_retries 逻辑（向后兼容）

    参数：
        llm_client: LLMClient 实例（async chat_completions 方法）
        messages: 消息列表
        tools: 工具 schema 列表（OpenAI 格式）
        max_retries: 最大重试次数
        initial_backoff: 首次退避秒数
        fallback_llm_client: 备用 LLMClient（主 client 失败时切换）
        jitter_ratio: 抖动比例（默认 0.25），sleep = base + uniform(0, base*ratio)。
                      多实例并发遇到 429 时避免雷击；设 0 关闭抖动（向后兼容）。
        max_tokens: 透传给 chat_completions 的 max_tokens（None 时不传，让 SDK 用默认）。
                    主循环检测 finish_reason=length 后用 MaxTokensEscalator 升级此值。
        consecutive_529_threshold: 连续 529 次数达阈值即切 fallback（默认 3）。
                                   0 表示禁用提前切换，走完所有重试。
        config: 配置字典（可选）。传入时检查 bash_unattended_retry flag：
                开启则启用持久重试模式，max_retries 被视为无限，
                加 max_hours（默认 24h）deadline 守护。
        background: True 表示后台任务调用。遇 529（服务过载）直接抛出不重试——
                    后台重试只会火上浇油，下个周期天然重跑（对齐 CCB 防放大）。
        heartbeat_cb: 长退避分片心跳回调 fn(elapsed, total)。退避 >30s 时
                      每 30s 调一次（异常吞掉，fail-open）——数分钟退避期间
                      用户/宿主不至于以为 agent 挂了。
    """
    last_error: Optional[Exception] = None
    # X6 fix: max_retries<=0 直接抛友好错误（否则下面 for 循环不进，最后 raise None → TypeError）
    # 注意：unattended 模式下 max_retries 被改写为 inf，此校验仅对原模式生效。
    # unattended 模式下原 max_retries 值被忽略，不走此分支。

    # ── Task P2.1: 持久重试（unattended）模式接入 ──
    # flag ON：max_retries=inf，加 deadline（time.monotonic 起算）
    # flag OFF / config=None：原 max_retries 逻辑不变
    unattended_enabled = False
    deadline: Optional[float] = None
    if config is not None:
        from agent.feature_flags import is_feature_enabled, get_feature_config
        if is_feature_enabled(config, "bash_unattended_retry"):
            unattended_enabled = True
            feature_cfg = get_feature_config(config, "bash_unattended_retry")
            max_hours = feature_cfg.get("max_hours", DEFAULT_UNATTENDED_MAX_HOURS)
            try:
                max_hours = float(max_hours)
            except (TypeError, ValueError):
                logger.warning(
                    "bash_unattended_retry.max_hours 类型异常（%s），用默认 %dh",
                    type(max_hours).__name__, DEFAULT_UNATTENDED_MAX_HOURS,
                )
                max_hours = float(DEFAULT_UNATTENDED_MAX_HOURS)
            deadline = time.monotonic() + max_hours * 3600.0
            logger.info(
                "持久重试（unattended）模式已开启：max_retries=∞，deadline=%d 小时后",
                int(max_hours),
            )

    # 计算生效的重试上限：unattended 模式下无限（用 float('inf') 比较）
    # 用 float('inf') 而非改写 max_retries 变量类型（保持 int 语义清晰）。
    if unattended_enabled:
        effective_max_retries: float = float('inf')
        # unattended 模式下跳过 max_retries<=0 的 ValueError 校验
        # （因为原值可能任意，被忽略）
    else:
        if max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {max_retries}")
        effective_max_retries = max_retries

    # max_tokens=None 时不传该参数，避免某些 provider 把 None 当 0 处理
    call_kwargs = {"tools": tools}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    consecutive_529 = 0  # 连续 529 计数器
    overflow_adjusts = 0  # R17 #13：400 溢出下调次数（独立计数，不耗正常重试）

    # 主 client 重试（while 循环兼容 finite max_retries 和 unattended 无限模式）
    attempt = 0
    while attempt < effective_max_retries:
        # unattended 模式：每次循环检查 deadline，超时退出
        if unattended_enabled and deadline is not None:
            if time.monotonic() >= deadline:
                logger.warning(
                    "持久重试 deadline 到期（已重试 %d 次），退出主 client 重试",
                    attempt,
                )
                break

        try:
            return await llm_client.chat_completions(messages, **call_kwargs)
        except Exception as e:
            last_error = e
            # === R17 #13：400 溢出自适应（input + max_tokens > context limit）===
            # 报文含可解析的溢出数值 → 动态下调 max_tokens 立即重试（最多 2 次，
            # 不耗正常重试计数）。解析不出 / 输入本身太大 / 已到下限 → 按普通
            # 400 不可重试抛出（透出给上层 reactive_compact / PTL 恢复路径）。
            if (
                _error_status_code(e) == 400
                and "max_tokens" in call_kwargs
                and overflow_adjusts < 2
            ):
                parsed = parse_context_overflow(e)
                if parsed is not None:
                    input_t, limit = parsed
                    new_max = compute_overflow_max_tokens(input_t, limit)
                    if new_max is not None and new_max < call_kwargs["max_tokens"]:
                        logger.warning(
                            "400 溢出（input %d + max_tokens > %d），下调 max_tokens "
                            "%d → %d 重试",
                            input_t, limit,
                            call_kwargs["max_tokens"], new_max,
                        )
                        call_kwargs["max_tokens"] = new_max
                        overflow_adjusts += 1
                        await asyncio.sleep(0.5)
                        continue
            if not is_retryable(e):
                raise

            # R25 #4：后台调用遇 529 立即放弃（防放大）——非前台 querySource
            # 不该在已知过载的 endpoint 上排队重试
            if background and _error_status_code(e) == 529:
                logger.warning("后台 LLM 调用遇 529（过载），放弃重试（防放大）")
                raise

            # R26 #10：连接重置 → 重建 client 再重试（不额外耗重试计数）
            if _is_connection_reset(e):
                reset = getattr(llm_client, "reset_client", None)
                if reset is not None:
                    try:
                        reset()
                        logger.warning("连接重置类错误，已重建 LLM client 重试: %s", e)
                    except Exception as re:
                        logger.debug("reset_client 失败（按原样重试）: %s", re)

            # P1-1: 529 连续失败精确切换
            # 过载往往持续一段时间，期间可能反复抛 529 / 5xx。把 5xx 都计入过载计数，
            # 避免出现"529→500→529"导致计数清零、永远到不了阈值的场景。
            # 仅当遇到非过载错误（429 限流、连接错误、超时）时才清零。
            status = _error_status_code(e)
            if status is not None and status >= 500:
                consecutive_529 += 1
            elif status is None:
                # 无状态码（连接/超时类）→ 视为本轮过载无关，重置
                consecutive_529 = 0
            # status < 500（如 429）→ 重置
            else:
                consecutive_529 = 0
            if (
                status == 529
                and consecutive_529_threshold > 0
                and consecutive_529 >= consecutive_529_threshold
                and fallback_llm_client is not None
            ):
                logger.warning(
                    "连续 %d 次 529（过载），立即切备用 client（不耗尽重试）",
                    consecutive_529,
                )
                break  # 跳出主 client 重试，进入 fallback 路径

            backoff = _compute_backoff(
                attempt=attempt,
                initial_backoff=initial_backoff,
                retry_after=get_retry_after(e),
                jitter_ratio=jitter_ratio,
                # R17 #44：unattended 模式退避帽放宽到 5min（对齐 CCB）——
                # 普通模式 60s 帽（防用户等死），长跑模式太频繁反而挤占限流配额
                max_backoff=(
                    UNATTENDED_MAX_BACKOFF
                    if unattended_enabled else DEFAULT_MAX_BACKOFF
                ),
            )
            if unattended_enabled:
                logger.warning(
                    "LLM 调用失败（unattended 尝试 %d，无重试上限），%.1fs 后重试: %s",
                    attempt + 1, backoff, e,
                )
            else:
                logger.warning(
                    "LLM 调用失败（尝试 %d/%d），%.1fs 后重试: %s",
                    attempt + 1, max_retries, backoff, e,
                )
            await _sleep_with_heartbeat(backoff, heartbeat_cb)
            attempt += 1

    # 主 client 重试耗尽（或被 529 阈值打断 / deadline 到期），尝试备用 client
    if fallback_llm_client is not None:
        logger.warning("切换备用 LLM client")
        try:
            return await fallback_llm_client.chat_completions(messages, **call_kwargs)
        except Exception as e:
            last_error = e
            logger.error("备用 client 也失败: %s", e)

    raise last_error


class MaxTokensEscalator:
    """跟踪 max_tokens 升级状态（P0-3）。

    策略：finish_reason=length（max_tokens 截断）时，先升级 max_tokens 重试一次，
    升级后仍不够才发"请继续"续写消息。升级重试不打断思路，续写容易接歪。

    用法：
        esc = MaxTokensEscalator()
        # 第一次调用：用 get_next_max_tokens()，None 表示用 SDK 默认
        response = client.chat_completions(messages, max_tokens=esc.get_next_max_tokens())
        if detect_length_finish(response) and not esc.has_escalated:
            esc.escalate()
            # 重试同一请求（不追加消息）
            response = client.chat_completions(messages, max_tokens=esc.get_next_max_tokens())
        # 升级后仍 length → 调用方发"请继续"

    一个 AIAgent 实例持有一个 escalator，整个会话生命周期复用。
    会话间 reset() 一次。
    """

    def __init__(
        self,
        *,
        initial: Optional[int] = DEFAULT_INITIAL_MAX_TOKENS,
        escalated: int = DEFAULT_ESCALATED_MAX_TOKENS,
    ):
        self._initial = initial
        self._escalated = escalated
        self.has_escalated = False

    def get_next_max_tokens(self) -> Optional[int]:
        """返回当前应使用的 max_tokens。

        - 未升级：返回 initial（默认 None，表示不传给 SDK）
        - 已升级：返回 escalated 值
        """
        return self._escalated if self.has_escalated else self._initial

    def escalate(self) -> int:
        """升级到 escalated 值。幂等：多次调用结果相同。

        返回升级后的 max_tokens。
        """
        self.has_escalated = True
        return self._escalated

    def reset(self) -> None:
        """重置到未升级状态（新会话用）。"""
        self.has_escalated = False


def detect_length_finish(response) -> bool:
    """检测 LLM 响应是否因 max_tokens 截断。

    finish_reason == "length" → True
    其他（"stop" / "tool_calls" / None / 结构异常）→ False

    fail-open：response 结构异常返回 False（不当截断处理，避免误升级）。
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return False
        first = choices[0]
        return getattr(first, "finish_reason", None) == "length"
    except (AttributeError, IndexError, TypeError):
        return False


def _compute_backoff(
    *,
    attempt: int,
    initial_backoff: float,
    retry_after: Optional[float],
    jitter_ratio: float = DEFAULT_JITTER_RATIO,
    max_backoff: float = None,
) -> float:
    """计算退避秒数：base + jitter。

    - base = retry_after（若有）或 initial_backoff * 2^attempt
    - jitter = uniform(0, base * jitter_ratio)
    - 返回 base + jitter

    X7 fix: 加上限，防止大 retry_after 或大 attempt 卡死主循环。
    R17 #44：上限参数化——普通模式 60s（防数小时 sleep 让用户以为 agent 挂了），
    unattended 长跑模式 5min（对齐 CCB；None 时用 DEFAULT_MAX_BACKOFF）。
    jitter_ratio=0 时返回纯 base（向后兼容）。
    """
    cap = DEFAULT_MAX_BACKOFF if max_backoff is None else max_backoff
    base = retry_after if retry_after else initial_backoff * (2 ** attempt)
    base = min(base, cap)
    if jitter_ratio <= 0:
        return base
    jitter = random.uniform(0, base * jitter_ratio)
    return base + jitter
