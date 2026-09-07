"""LLM 调用的「防摔垫」：失败了怎么办的整套预案。

调 LLM API 会遇到「再打一次也许就通了」的错（429 限流、5xx 服务器错误、
连接断/超时——值得自动重试）和「重打一百次也没用」的错（400 参数错、
401 认证错——得立刻报出来）。

本模块的策略：
  - 可重试错误（429/5xx/连接问题）：按「越等越久」的节奏自动重试
    （指数退避：1s、2s、4s……就像别人占线时你隔越来越久再拨）
  - 不可重试错误（400/401）：立即抛出，不浪费时间
  - 主模型实在打不通：切换备用模型再试一次（如果配了）
  - 回答被输出上限拦腰截断：先把上限调大重试，还不行就让模型「接着说」

处于 agent/__init__.py（主循环）之下、llm_client.py（真正的接线层）
之上——每次 LLM 请求都先经过这里的「防摔」包装。
"""

import asyncio
import logging
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # 首次等待的秒数（指数退避的起点，之后翻倍）
DEFAULT_JITTER_RATIO = 0.25    # 「抖动」比例：实际等待 = 基础值 + 随机(0, 基础值*比例)
DEFAULT_MAX_BACKOFF = 60.0     # 单次最长等多久（普通模式；防止一等几小时）
UNATTENDED_MAX_BACKOFF = 300.0  # 无人值守长跑模式的最长等待 5 分钟（对齐业界做法）

# 无人值守（unattended）持久重试模式的默认值：连续重试最多撑 24 小时
DEFAULT_UNATTENDED_MAX_HOURS = 24

# 输出上限（max_tokens）「升级」的默认值：
# - 初始 None = 不主动传这个参数，让 SDK 用模型自己的默认值
# - 升级到 64000（业界常用 64k）
# 设计取舍：输出上限较小的服务商（如 DeepSeek 只有 8K）收到 64k 会报
# 400 溢出——由下面的「400 溢出自适应」（parse_context_overflow）动态
# 下调兜底；万一升级调用本身失败，也沿用截断的回答（fail-open 不硬抛）。
DEFAULT_INITIAL_MAX_TOKENS: Optional[int] = None
DEFAULT_ESCALATED_MAX_TOKENS = 64000

# 升级后仍被截断时，「请模型接着说」的续写恢复最多做几轮（对齐业界，3 次）
DEFAULT_OUTPUT_RECOVERY_LIMIT = 3

# 529（服务过载）连续失败多少次就换备胎：
# 过载通常要持续一阵子，与其在已知过载的端点上一遍遍撞墙（还白占
# 限流配额），不如攒够 3 次立刻切备用模型。
DEFAULT_CONSECUTIVE_529_THRESHOLD = 3

# 退避「心跳」分片长度：等待超过 30 秒时切成小段，段间报个进度
# （不然几分钟没动静，用户还以为程序挂了）
HEARTBEAT_CHUNK_SECONDS = 30.0


async def _sleep_with_heartbeat(total: float, heartbeat_cb=None) -> None:
    """把漫长的等待切成最长 30 秒的小段睡，每段之间报一次「还活着」的心跳（无人值守模式一等好几分钟，全程无动静会让人以为程序挂了）。

    参数：
        total：总共要等的秒数
        heartbeat_cb：心跳回调函数 fn(已等秒数, 总秒数)；没传或总时长
                      不到 30 秒就直接睡（零开销，兼容旧用法）

    返回：无。回调抛异常会被吞掉——汇报进度的事不能拖垮重试正事。
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
    """从异常对象里抠出 HTTP 状态码（不同 SDK 存的字段名不一样，都试试）。

    参数：
        error：任意异常

    返回：状态码数字；异常里没带就返回 None。
    """
    code = getattr(error, "status_code", None)
    if code is not None:
        return code
    code = getattr(error, "status", None)
    if code is not None:
        return code
    return None


def is_retryable(error: Exception) -> bool:
    """判断一个错误值不值得「再试一次」。

    判断标准（像判断电话没打通的原因）：
    - 可重试：占线（429 限流）、对面机房出事（5xx）、信号断、超时
    - 不可重试：号码拨错（400 参数错）、门禁卡无效（401 认证错）、
      没权限（403）——这些重试一万次结果也一样

    参数：
        error：要判断的异常

    返回：True = 值得重试；False = 立刻报错别浪费时间。
    """
    try:
        from openai import (
            RateLimitError, APIConnectionError, APITimeoutError, APIStatusError,
        )
        if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            # 5xx 和 429 才值得重试，其余（4xx）都是「自己的问题」
            return error.status_code >= 500 or error.status_code == 429
    except ImportError:
        pass

    # httpx 的传输层错误（连接被重置/管道断裂/各种网络毛病的总类）
    # 一律可重试。openai SDK 通常会把它包装成 APIConnectionError
    # （上面已经命中），但其它路径可能裸着抛出来——这种异常的类名里
    # 不含 "connection"/"timeout" 字样，光靠名字兜底会漏判，
    # 所以这里按类型兜底。
    try:
        import httpx
        if isinstance(error, httpx.TransportError):
            return True
    except ImportError:
        pass

    # 最后的兜底：看异常类名里有没有「超时/连接/临时」这类字眼
    error_type = type(error).__name__.lower()
    if any(kw in error_type for kw in ("timeout", "connection", "temporary")):
        return True
    return False


def get_retry_after(error: Exception) -> Optional[float]:
    """从错误里读出服务器建议的「多久后再来」（Retry-After，单位秒）——429 限流时服务器会在响应头里写明，照着等比自己瞎猜高效。

    参数：
        error：要检查的异常

    返回：建议等待的秒数；没给就返回 None。
    """
    try:
        retry_after = getattr(error, "retry_after", None)
        if retry_after:
            return float(retry_after)
    except (TypeError, ValueError):
        pass
    # openai 库把响应头放在 response_headers 属性里
    try:
        headers = getattr(error, "response_headers", None) or {}
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            return float(ra)
    except Exception:
        pass
    return None


# 连接被掐断类错误的名字特征：遇到它们，重试前要先重建
# client 扔掉坏掉的连接池——在坏池子上重试大概率还是同样的错
_RESET_NAMES = ("connectionreset", "brokenpipe", "remoteprotocol", "readerror", "writeerror")


def _is_connection_reset(error: Exception) -> bool:
    """判断是不是「连接被掐断」类错误（看异常名和它引发错误的整条链）——
    这类错误在坏掉的连接池上重试大概率还是同样的错，重试前应先重建 client。

    参数：
        error：要判断的异常

    返回：True = 连接断了，重试前该重建 client。
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
# 400 溢出自适应：请求的「输入 + 输出上限」超过了模型的
# 上下文总容量时，服务器会报 400 并在报错文字里写出具体数字。这里用
# 两个正则把数字抠出来，好算出一个塞得下的输出上限再重试
# ---------------------------------------------------------------------------

# 第一种报错格式（数字最全）："input length and `max_tokens` exceed
# context limit: 188059 + 20000 > 200000"（输入 + 输出 > 总容量）
_OVERFLOW_EXACT_RE = re.compile(
    r"input length and `max_tokens` exceed context limit:\s*(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)"
)
# 第二种（OpenAI 老格式）："This model's maximum context length is 131072
# tokens. However, you requested 140000 tokens (120000 input tokens and
# 20000 max_tokens)"
_OVERFLOW_LOOSE_RE = re.compile(
    r"maximum context length is (\d+) tokens?.*?(\d+) input tokens",
    re.DOTALL,
)

# 重算输出上限时的安全余量（留 1000 token 缓冲）和最低保底（3000）
_OVERFLOW_SAFETY_BUFFER = 1000
_OVERFLOW_MIN_MAX_TOKENS = 3000


def parse_context_overflow(error: Exception) -> Optional[Tuple[int, int]]:
    """从 400 溢出的报错文字里抠出「输入量」和「总容量」两个数字。

    参数：
        error：服务器抛的 400 异常（看它的报错文字）

    返回：(输入 token 数, 上下文总容量)；两种报错格式都没匹配上则
    返回 None——调用方就当普通 400 处理。
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
    """算一个塞得下的输出上限：总容量 - 输入 - 1000 缓冲，最低保底 3000。

    参数：
        input_tokens：这回请求的输入 token 数
        context_limit：模型的上下文总容量

    返回：新的输出上限；返回 None 表示输入自己就快把容量占满了
    （剩余不到 3000），调小输出也救不了，没有恢复价值。
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
    background: bool = False,  # 后台调用（子代理摘要等）遇 529 立即放弃
    heartbeat_cb=None,  # 长退避分片心跳（fn(elapsed, total)）
):
    """带着全套「防摔预案」调一次 LLM：自动重试、自动换备胎、自动调参。

    整个流程像打电话的过程：
      1. 用主号码（主 client）重拨 max_retries 次，每次比上次多等一会
         （指数退避 + 随机抖动）
      2. 如果连续 N 次听到「对面过载」（529），不等拨完次数直接换备胎
      3. 主号码彻底打不通，而配了备胎（fallback_llm_client），就再试一次
      4. 备胎也不行，只好把最后一个错误抛出去

    本函数是 async：所有等待都用 asyncio 版 sleep（不让等待卡住整个事件循环）。

    无人值守持久重试模式：当配置里的 `bash_unattended_retry` 开关打开——
      - 重试次数视为无限（不会数满退出，只受时间限制）
      - 设一个「截止时刻」（默认 24 小时后），到点收手
      - 其余逻辑（普通重试/529 早切/备胎）照旧
      - 开关没开或不传 config，完全走按次数重试

    参数：
        llm_client：主 LLM client（需要有 async 的 chat_completions 方法）
        messages：对话历史（消息列表）
        tools：可用工具清单（OpenAI 格式，可不传）
        max_retries：最多重试几次
        initial_backoff：第一次失败后等多久（秒），之后逐次翻倍
        fallback_llm_client：备用 client（主 client 失败时切换，可不传）
        jitter_ratio：抖动比例（默认 0.25），实际等待 = 基础值 + 随机量。
                      作用：很多实例同时撞上限流时，加 randomness 错开
                      各自的重试时刻，避免「集体再撞」；设 0 关闭（兼容旧用法）。
        max_tokens：输出上限，透传给 chat_completions（None 就不传，
                    让 SDK 用默认值）。主循环发现回答被截断后，会用
                    MaxTokensEscalator 把这个值调大再试。
        consecutive_529_threshold：连续多少次 529 就换备胎（默认 3）。
                      0 表示不提前切，老老实实拨完全部次数。
        config：配置字典（可不传）。传了就会检查 bash_unattended_retry
                      开关，开了进入上面说的持久重试模式。
        background：True 表示这是后台任务的调用。遇到 529（过载）直接
                      放弃不重试——后台任务下个周期本来就会重跑，在过载
                      的端点上硬挤只会火上浇油。
        heartbeat_cb：长等待的心跳回调 fn(已等秒数, 总秒数)。等待超过
                      30 秒时每 30 秒报一次（回调出错会被吞掉）——不然
                      一等好几分钟，用户还以为程序挂了。

    返回：成功的话返回 LLM 的响应对象。
    抛错：所有尝试都失败时，抛最后一次遇到的那个异常。
    """
    last_error: Optional[Exception] = None
    # max_retries<=0 时下面的循环一次都不进，最后会变成「raise None」的
    # 莫名 TypeError——提前拦下给个说得清的错误。
    # 注意：无人值守模式下重试次数视为无限，不走这个校验（原值被忽略）。

    # ── 无人值守持久重试模式的接入 ──
    # 开关开：重试次数无限，只受截止时刻约束
    # 开关没开 / 没传 config：完全走按次数逻辑
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

    # 算出实际生效的重试上限：无人值守模式给无限（拿正无穷比较）。
    # 之所以不改 max_retries 本身、另开一个变量，是为了让它保持
    # 「int 类型的次数」这个清晰含义，不搞类型混淆。
    if unattended_enabled:
        effective_max_retries: float = float('inf')
        # 无人值守模式下跳过 max_retries<=0 的校验（原值随便填都被忽略）
    else:
        if max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {max_retries}")
        effective_max_retries = max_retries

    # max_tokens 为 None 时干脆不传这个参数——有的服务商会把 None
    # 当成 0 处理（一个字都不让说）
    call_kwargs = {"tools": tools}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    consecutive_529 = 0  # 连续 529（过载）的计数器
    overflow_adjusts = 0  # 400 溢出导致下调输出上限的次数（单独计数，不占用正常重试名额）

    # 主 client 的重试循环（while 写法同时兼容「按次数」和「无限+截止时刻」两种模式）
    attempt = 0
    while attempt < effective_max_retries:
        # 无人值守模式：每圈都看一眼表，到截止时刻就收手
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
            # === 400 溢出自适应 ===
            # 报错文字里带了具体数字 → 现场把输出上限调小、立刻重试
            # （最多调 2 次，不占正常重试的名额）。解析不出数字 / 输入
            # 本身太大 / 已调到底 → 当普通 400 抛出去，交给上层的
            # 上下文压缩等恢复机制处理。
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

            # 后台调用遇 529（过载）直接放弃（防「火上浇油」）——
            # 后台任务下个周期自然会重跑，没必要在过载端点上排队硬挤
            if background and _error_status_code(e) == 529:
                logger.warning("后台 LLM 调用遇 529（过载），放弃重试（防放大）")
                raise

            # 连接被掐断 → 先重建 client（换新连接池）再重试，不额外耗重试名额
            if _is_connection_reset(e):
                reset = getattr(llm_client, "reset_client", None)
                if reset is not None:
                    try:
                        reset()
                        logger.warning("连接重置类错误，已重建 LLM client 重试: %s", e)
                    except Exception as re:
                        logger.debug("reset_client 失败（按原样重试）: %s", re)

            # 529 连续失败计数：过载期间服务器可能 529/500 交替着抛，
            # 所以所有 5xx 都算「过载嫌疑」往计数器上加——不然出现
            # "529→500→529" 的交替就把计数清零了，永远凑不满阈值。
            # 只有遇到跟过载无关的错（429 限流/连接问题/超时）才清零。
            status = _error_status_code(e)
            if status is not None and status >= 500:
                consecutive_529 += 1
            elif status is None:
                # 没有状态码（连接/超时类问题）→ 跟过载无关，计数清零
                consecutive_529 = 0
            # 状态码小于 500（比如 429）→ 也清零
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
                break  # 跳出主 client 的重试循环，进入下面的备胎路径

            backoff = _compute_backoff(
                attempt=attempt,
                initial_backoff=initial_backoff,
                retry_after=get_retry_after(e),
                jitter_ratio=jitter_ratio,
                # 等待上限分两档：普通模式 60 秒（不能让干等的用户以为死机了），
                # 无人值守长跑模式放宽到 5 分钟（重试太勤反而白占限流配额）
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

    # 主 client 的重试名额用完（或被 529 阈值打断/到了截止时刻），换备胎试试
    if fallback_llm_client is not None:
        logger.warning("切换备用 LLM client")
        try:
            return await fallback_llm_client.chat_completions(messages, **call_kwargs)
        except Exception as e:
            last_error = e
            logger.error("备用 client 也失败: %s", e)

    raise last_error


class MaxTokensEscalator:
    """「输出上限调节器」：记住当前该用多大的 max_tokens。

    回答被输出上限截断时先调大上限重发原请求（一口气说完思路连贯）；
    调大后仍不够，才由调用方发「请继续」续写（接续处容易接歪）。
    本类就是记住「现在到哪一步了」的小账本。

    用法示例：
        esc = MaxTokensEscalator()
        # 第一次调用：问账本要当前上限（None 表示不传，让 SDK 用默认）
        response = client.chat_completions(messages, max_tokens=esc.get_next_max_tokens())
        if detect_length_finish(response) and not esc.has_escalated:
            esc.escalate()
            # 原请求原样重发（不追加消息），只是上限变大了
            response = client.chat_completions(messages, max_tokens=esc.get_next_max_tokens())
        # 调大后仍被截断 → 由调用方发"请继续"做续写恢复

    一个 AIAgent 实例配一个账本，整个会话反复用；换新会话时 reset 一次。
    """

    def __init__(
        self,
        *,
        initial: Optional[int] = DEFAULT_INITIAL_MAX_TOKENS,
        escalated: int = DEFAULT_ESCALATED_MAX_TOKENS,
    ):
        # 参数：initial = 初始上限（默认 None，不传给 SDK 用它默认值）；
        # escalated = 升级后的大上限（默认 64000）
        self._initial = initial
        self._escalated = escalated
        self.has_escalated = False

    def get_next_max_tokens(self) -> Optional[int]:
        """问账本：这次该用多大的输出上限。

        - 还没升级：返回初始值（默认 None，表示不传给 SDK）
        - 已升级：返回升级后的大值

        返回：本次调用应使用的 max_tokens（或 None）。
        """
        return self._escalated if self.has_escalated else self._initial

    def escalate(self) -> int:
        """把上限调到升级档。重复调也一样（账已记过就不再变）。

        返回：升级后的 max_tokens。
        """
        self.has_escalated = True
        return self._escalated

    def reset(self) -> None:
        """把账本翻回「还没升级」那一页（新会话开头的复位用）。"""
        self.has_escalated = False


def detect_length_finish(response) -> bool:
    """判断回答是不是被输出上限拦腰截断的。

    看一个标志位：结束原因（finish_reason）为 "length" 就是被截断；
    其他情况（"stop" 正常说完 / "tool_calls" 要调工具 / 没写 / 响应
    结构不对劲）都算没截断。

    参数：
        response：LLM 响应对象

    返回：True = 被截断。响应结构异常时也返回 False（宁可漏判也不
    误判——误判会白白升级上限）。
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
    """算出这次该等多久再重试。

    公式大白话：
    - 基础等待 = 服务器建议的 Retry-After（有就用它），否则
      首次等待 × 2 的「第几次重试」次方（1s→2s→4s→8s……）
    - 再加一点随机量（抖动），错开同时撞限流的难兄难弟们
    - 最后封顶，不让一次等待等出天荒地老

    参数：
        attempt：这是第几次重试（从 0 起）
        initial_backoff：首次等待的秒数
        retry_after：服务器建议的等待秒数（可不传）
        jitter_ratio：抖动比例（0 = 不要抖动，兼容旧用法）
        max_backoff：等待上限（不传用默认 60 秒）

    等待必须封顶且分两档：普通模式 60 秒、无人值守长跑模式 5 分钟
    （不封顶时 Retry-After 很大或重试次数多会一等几小时）。

    返回：实际该睡的秒数。
    """
    cap = DEFAULT_MAX_BACKOFF if max_backoff is None else max_backoff
    base = retry_after if retry_after else initial_backoff * (2 ** attempt)
    base = min(base, cap)
    if jitter_ratio <= 0:
        return base
    jitter = random.uniform(0, base * jitter_ratio)
    return base + jitter


# ---------------------------------------------------------------------------
# 拆分二期块 A：max_tokens 升级判定 / usage 加总 / 续写恢复两件 + 长退避
# 心跳——五件从 agent/__init__.py 逐字节平移而来（self→agent，属性仍全
# 在 AIAgent 实例上；两个纯函数原为 staticmethod，去壳直落模块级）
# ---------------------------------------------------------------------------

def try_escalate_max_tokens(agent, trigger_desc):
    """max_tokens 截断后的「升级判定」：看能不能调大上限，能就调并打日志。

    大白话：先过两道门——没装升级器（agent._max_tokens_escalator 为
    None）、或本轮已经升过级（防无限套娃）——任一道挡住就返回 None，
    调用方沿用截断响应。两道门都过了才真正调 escalate() 拿新上限
    （这步有副作用：标记「已升级」），并打一条 info 日志。

    参数：
        trigger_desc: 触发原因文案，原样进日志（如「非流式」、
            「finish_reason=length」、「纯 thinking（content 空）」）

    返回：新的 max_tokens 上限；不该/不能升级时返回 None。
    """
    if (agent._max_tokens_escalator is None
            or agent._max_tokens_escalator.has_escalated):
        return None
    new_max = agent._max_tokens_escalator.escalate()
    logger.info(
        "max_tokens 截断（%s），升级到 %d 重试", trigger_desc, new_max
    )
    return new_max


def merge_usage_tokens(final_usage, retried):
    """把升级重试响应的 usage 逐字段加总进旧账本。

    大白话：截断那次和升级重试这次是两笔真实花费，直接拿新 usage
    覆盖会漏记前一笔，所以四个字段（prompt/completion/两类缓存）
    逐项相加——Anthropic 字段优先、DeepSeek 字段兜底，加总顺序与
    字段名和原内联实现逐字节一致。重试响应没带 usage 就原样返回
    旧账本（新调用一分钱没记）。

    参数：
        final_usage: 已有用量 dict（不是 dict 时按空账本处理）
        retried: 升级重试拿到的响应对象

    返回：加总后的新用量 dict（不原地改旧 dict）。
    """
    if getattr(retried, "usage", None) is None:
        return final_usage
    u = retried.usage
    retry_usage = {
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
    prev_usage = final_usage if isinstance(final_usage, dict) else {}
    return {
        k: (prev_usage.get(k, 0) or 0) + (retry_usage.get(k, 0) or 0)
        for k in retry_usage
    }


async def recover_output_truncation(agent, response, messages, tool_schemas):
    """调大上限后仍被截断 → 让模型「从断点接着写」的续写恢复。

    做法：把截断的半截回答 +
    一条「从中断处直接继续、不道歉不复述」的指令，追加到**只在本次请求
    里用的局部消息**上再调 LLM，把续写拼上去；最多重试
    llm.output_recovery_limit 次（默认 3）。

    设计要点：
    - 局部 messages 不进正式历史——恢复成功后以「拼接好的完整回答」
      一条消息返回（主循环正常入史），不污染会话记录
    - 只处理纯文本截断；工具调用被截断（罕见）或没有可续内容就原样返回
    - 还没做过上限升级的截断不接手（那是上游升级路径的事）
    - 失败放行：恢复调用挂了就返回已拼接的部分（保留截断标记，
      主循环当最终响应处理）

    参数：
        response: 被截断的响应对象
        messages: 本轮消息列表
        tool_schemas: 工具 schema 列表

    返回：恢复后的响应对象（或原样返回）。
    """
    from agent.llm_retry import (
        DEFAULT_OUTPUT_RECOVERY_LIMIT,
        call_with_retry,
        detect_length_finish,
    )
    if not detect_length_finish(response):
        return response
    if (agent._max_tokens_escalator is not None
            and not agent._max_tokens_escalator.has_escalated):
        return response
    msg = response.choices[0].message
    if getattr(msg, "tool_calls", None):
        return response
    accumulated = msg.content or ""
    if not accumulated.strip():
        return response

    try:
        limit = int(
            (agent.config or {}).get("llm", {}).get(
                "output_recovery_limit", DEFAULT_OUTPUT_RECOVERY_LIMIT,
            )
        )
    except (TypeError, ValueError):
        limit = DEFAULT_OUTPUT_RECOVERY_LIMIT
    limit = max(0, limit)

    recovery_meta = (
        "你的上一条回复因输出 token 上限被截断。"
        "从中断处直接继续——不要道歉、不要复述已写内容，"
        "从被切断的那个位置接着写。把剩余工作拆成小块完成。"
    )
    recovery_max_tokens = (
        agent._max_tokens_escalator.get_next_max_tokens()
        if agent._max_tokens_escalator is not None else None
    )
    local_messages = list(messages) + [
        {"role": "assistant", "content": accumulated},
        {"role": "user", "content": recovery_meta},
    ]
    for attempt in range(1, limit + 1):
        try:
            resp = await call_with_retry(
                agent.llm_client,
                local_messages,
                tools=tool_schemas if tool_schemas else None,
                fallback_llm_client=agent.fallback_llm_client,
                max_tokens=recovery_max_tokens,
                config=agent.config,
                heartbeat_cb=llm_retry_heartbeat,  # 长退避心跳
            )
        except Exception as e:
            logger.warning("续写恢复调用失败（返回已拼接内容）: %s", e)
            break
        piece = (resp.choices[0].message.content or "")
        if piece:
            accumulated += piece
        if not detect_length_finish(resp):
            logger.info("续写恢复成功（第 %d 次），拼接 %d 字符", attempt, len(accumulated))
            return merge_continuation_response(
                response, resp, accumulated, finished=True,
            )
        local_messages = list(local_messages) + [
            {"role": "assistant", "content": piece},
            {"role": "user", "content": recovery_meta},
        ]
    if limit > 0:
        logger.warning("续写恢复 %d 次后仍截断，返回已拼接内容", limit)
    return merge_continuation_response(
        response, None, accumulated, finished=False,
    )


def merge_continuation_response(
    truncated_response, last_response, content, *, finished: bool,
):
    """续写恢复的最后一步：把截断响应和续写响应合并成一个。

    拼上内容、取最后一轮的用量，形状对齐截断前的响应结构。
    最后一轮续写如果带工具调用，必须保留并把
    结束原因标成 "tool_calls"（主循环按它分发工具；硬编码成
    「无工具调用 + stop」会把模型明确要调工具的意图吞掉还伪装成正常
    完成）。还没写完（仍截断）就维持 "length"。

    参数：
        truncated_response: 最初被截断的响应
        last_response: 最后一轮续写的响应（可能为 None）
        content: 拼接好的完整文本
        finished: 关键字参数，True = 恢复完成不再截断

    返回：合并后的响应对象。
    """
    from types import SimpleNamespace
    src_msg = truncated_response.choices[0].message
    last_msg = None
    if last_response is not None:
        try:
            last_msg = last_response.choices[0].message
        except (IndexError, AttributeError):
            last_msg = None
    tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None
    merged = SimpleNamespace(
        content=content if content else None,
        tool_calls=tool_calls,
        reasoning_content=getattr(src_msg, "reasoning_content", None),
        thinking_signature=getattr(src_msg, "thinking_signature", None),
    )
    usage = (
        getattr(last_response, "usage", None)
        if last_response is not None
        else getattr(truncated_response, "usage", None)
    )
    if not finished:
        finish_reason = "length"
    else:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=merged,
            finish_reason=finish_reason,
        )],
        usage=usage,
    )


def llm_retry_heartbeat(elapsed: float, total: float) -> None:
    """LLM 长退避期间的心跳（超过 30 秒的退避才触发，每 30 秒一跳）。

    目的：等几分钟重试的时候，用户别以为 agent 死了。
    只记 info 日志（进日志文件和轨迹），不弹通知（30 秒一弹会刷屏）。

    参数：
        elapsed: 已等待秒数
        total: 预计总共要等多久

    返回：无。
    """
    logger.info("LLM 重试退避中：已等待 %.0fs / 预计共 %.0fs", elapsed, total)
