# -*- coding: utf-8 -*-
"""LLM 用量记账三件（从 agent/__init__.py 平移而来）。

大白话：这三个函数原本是 AIAgent 的三个方法，拆出来只为给
agent/__init__.py 减负——行为零变化，搬的是同一份代码：

- record_llm_usage：每轮 LLM 调用完记一笔 token 账（/usage 命令、
  缓存命中分析都要有账可查），顺手按消息条数记「权威锚点」——
  输入 token 的口径按 usage 字段名判语义（DeepSeek 单值、
  Anthropic 三项相加），压缩阈值的混合计数全靠它。
- extract_turn_tokens：从响应的 usage 字段里取本轮花了多少 token
  （输入 + 输出），goal 的 token 预算累加从这取数。
- extract_cache_read：从 usage 里取「前缀缓存命中读的 token 数」，
  DeepSeek / Anthropic 各叫各的名，or 短路哪个非零用哪个。

本模块三条铁律（跟拆分一期约定一致）：
1. 属性全留 AIAgent——函数不自己存状态，一律读写 ``agent._xxx``
   （账本 ``agent._llm_usage_stats``、锚点 ``agent._last_usage_anchor``、
   可选的 ``agent._usage_tracker``），记账函数第一参固定收 agent 实例
   （原 ``self``）；两个纯提取函数连 agent 都不用收。
2. 禁止模块级 import agent root（防循环导入）——本模块只依赖 logging。
3. 函数体与原方法逐字节平移，唯一改写是 ``self`` → ``agent``。

外部契约（verify.py 的「token 估算与阈值」检查直调
``AIAgent._record_llm_usage`` 并断言锚点口径）由 AIAgent 上的薄委托
保住——签名/行为都不动，委托进本模块的 record_llm_usage。
"""

import logging

logger = logging.getLogger(__name__)


def record_llm_usage(agent, response, sent_message_count: int = None) -> None:
    """记一笔 LLM 调用的 token 用量账（/usage 命令、缓存命中分析都要有账可查）。

    sent_message_count 不为 None 时，顺手记一个「权威锚点」
    ``_last_usage_anchor = (消息条数, 输入 token 数)``——输入 token 的
    口径按 usage 字段名判语义（DeepSeek 命名 → prompt_tokens 已含缓存
    直接用；Anthropic 命名 → 三项相加；详见下方注释）。压缩阈值判定
    拿它做「权威值 + 新消息粗估」的混合计数，比全程粗估准。

    参数：
        response: LLM 返回的响应对象（从它的 usage 字段取数）
        sent_message_count: 本次实际发送的消息条数；None 时不记锚点

    返回：无（直接累加到 agent._llm_usage_stats）。全程 fail-open。
    """
    agent._llm_usage_stats["total_calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    try:
        prompt_t = getattr(usage, "prompt_tokens", 0) or 0
        agent._llm_usage_stats["total_prompt_tokens"] += prompt_t
        agent._llm_usage_stats["total_completion_tokens"] += (
            getattr(usage, "completion_tokens", 0) or 0
        )
        # 前缀缓存相关字段（DeepSeek / OpenAI / Anthropic 各叫各的名，都试一遍）
        cache_read = (
            getattr(usage, "prompt_cache_hit_tokens", 0)
            or getattr(usage, "cache_read_input_tokens", 0)
            or 0
        )
        cache_creation = (
            getattr(usage, "prompt_cache_miss_tokens", 0)
            or getattr(usage, "cache_creation_input_tokens", 0)
            or 0
        )
        agent._llm_usage_stats["total_cache_read_tokens"] += cache_read
        agent._llm_usage_stats["total_cache_creation_tokens"] += cache_creation
        # 记权威锚点（压缩阈值混合计数用）。
        # 口径按 usage 字段名判语义——不同服务商的 prompt_tokens 含义不同：
        # - DeepSeek 命名（prompt_cache_hit_tokens）：prompt_tokens 本身
        #   已含缓存命中+未命中，直接用它（旧版三项相加 = 双倍，5 万真实
        #   token 会谎报成 10 万，过早触发有损压缩）
        # - Anthropic 命名（cache_read_input_tokens）：prompt_tokens 不含
        #   缓存部分，三项相加才是真实输入
        # - OpenAI 官方（两者皆无）：prompt_tokens 即全量
        if sent_message_count:
            if hasattr(usage, "prompt_cache_hit_tokens"):
                anchor_tokens = prompt_t
            else:
                anchor_tokens = prompt_t + cache_read + cache_creation
            agent._last_usage_anchor = (sent_message_count, anchor_tokens)
        # 按模型分四项累计（有 tracker 才记；失败不炸）
        if getattr(agent, "_usage_tracker", None) is not None:
            try:
                agent._usage_tracker.record(
                    model=str(
                        getattr(response, "model", None)
                        or agent.model or "unknown"
                    ),
                    prompt=prompt_t,
                    completion=getattr(usage, "completion_tokens", 0) or 0,
                    cache_read=cache_read,
                    cache_creation=cache_creation,
                )
            except Exception as te:
                logger.debug("usage_tracker 记录失败（fail-open）: %s", te)
    except Exception as e:
        logger.debug("记录 LLM usage 失败（fail-open）: %s", e)


def extract_turn_tokens(response) -> int:
    """从响应的 usage 字段里取本轮花了多少 token（输入 + 输出）。

    goal 的 token 预算要在每轮结束后累加，数据就从这取。
    响应没有 usage 字段就返回 0（不炸）。

    参数：
        response: LLM 响应对象（可以是 None）

    返回：本轮 prompt + completion 的 token 总数（整数）。
    """
    if response is None:
        return 0
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    try:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        return int(prompt) + int(completion)
    except Exception:
        return 0


def extract_cache_read(usage) -> int:
    """从 usage 里取「前缀缓存命中读的 token 数」。

    各家服务商字段名不一样——DeepSeek 叫 prompt_cache_hit_tokens，
    Anthropic 叫 cache_read_input_tokens。流式路径合成的 usage 对象两个
    字段都塞了值（见 _call_llm_streaming 末尾），所以这里用 or 短路，
    哪个非零用哪个。dict 和对象两种形态都兼容，出错返回 0。

    参数：
        usage: usage 对象或 dict（可能是 None/空）

    返回：缓存读 token 数（整数，出错为 0）。
    """
    if not usage:
        return 0
    try:
        if isinstance(usage, dict):
            return (
                usage.get("prompt_cache_hit_tokens", 0)
                or usage.get("cache_read_input_tokens", 0)
                or 0
            )
        return (
            getattr(usage, "prompt_cache_hit_tokens", 0)
            or getattr(usage, "cache_read_input_tokens", 0)
            or 0
        )
    except Exception:
        return 0
