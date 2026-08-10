# agent/context_pipeline.py
"""分层压缩管线：L1 snip / L2 micro / L3.5 contextCollapse / L4 llm + reactive。

替代 context_compressor.maybe_compress 的单层 LLM 摘要。
设计详见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3。
Task P1.1（spec §7.1）补 L3.5 contextCollapse：按 token 占用比折叠早期段，不动 system prompt。
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from agent.context_compressor import (
    _summarize_conversation, _fix_tool_call_pairs, estimate_message_tokens,
)
from agent.transcript import snapshot_if_needed

logger = logging.getLogger(__name__)


def _split_system(messages: list) -> Tuple[Optional[dict], list]:
    """分离 system 消息（如果有）。返回 (system_msg_or_None, rest)。"""
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages


def _reassemble(system: Optional[dict], conv: list) -> list:
    """重新组装：system（若有）+ conv。"""
    return [system, *conv] if system else conv


def _has_tool_calls(msg: dict) -> bool:
    """assistant 消息是否含 tool_calls。"""
    tcs = msg.get("tool_calls")
    return bool(tcs)


def _is_tool_result(msg: dict) -> bool:
    """是否为 tool 结果消息（role=='tool'）。"""
    return msg.get("role") == "tool"


def time_based_clear_old_tool_results(messages: list, config: dict) -> Tuple[list, bool]:
    """改造点 ④：基于时间的微压缩——距最后一条 assistant > gap_minutes 时清旧 tool result。

    对齐 claude-code-main microCompact:evaluateTimeBasedTrigger。
    在 compress_if_needed 编排里最早跑（无 token 检查），L1 snip 之前。

    返回 (messages, changed)：changed=True 表示本次确实清了内容，
    让 compress_if_needed 的 changed flag 能正确反映 time-based MC 的贡献。

    行为：
      1. enabled=False → 直接返回 (messages, False)
      2. 找最后一条 role=="assistant" 的消息 index
      3. 读该消息的 _timestamp；没有就 fail-open 返回
      4. elapsed = (now - last_ts) / 60；< gap_minutes 就返回
      5. 超时：把 last_assistant_idx 之前的所有 tool result 清除内容，
         保留最后 keep_recent 个（不清）

    fail-open：异常不影响主流程，返回 (messages, False)。
    幂等：已清除的 tool result content 等于 CLEARED_MARK，再清也不变（changed=False）。
    """
    CLEARED_MARK = "[Old tool result content cleared]"
    try:
        # config 是 context 子字典（由 compress_if_needed 从 self.config.get("context", {}) 传入），
        # flat key 读法跟 snip_compact / offload_large_tool_results 一致
        ctx_cfg = config if isinstance(config, dict) else {}
        enabled = ctx_cfg.get("time_based_mc_enabled", True)
        if not enabled:
            return messages, False

        gap_minutes = ctx_cfg.get("time_based_mc_gap_minutes", 60)
        keep_recent = ctx_cfg.get("time_based_mc_keep_recent", 5)

        # 找最后一条 assistant 消息
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx < 0:
            return messages, False

        last_ts = messages[last_assistant_idx].get("_timestamp")
        if not last_ts:
            return messages, False

        elapsed_min = (time.time() - last_ts) / 60
        if elapsed_min < gap_minutes:
            return messages, False

        # 超时：找 last_assistant_idx 之前的所有 tool result
        tool_indices = [
            i for i in range(last_assistant_idx)
            if messages[i].get("role") == "tool"
        ]
        if len(tool_indices) <= keep_recent:
            return messages, False

        # 保留最后 keep_recent 个，前面的清内容
        to_clear = tool_indices[:-keep_recent] if keep_recent > 0 else tool_indices
        cleared_count = 0
        for i in to_clear:
            # 幂等：content 已经是 CLEARED_MARK 的不再计 cleared_count
            if messages[i].get("content") and messages[i].get("content") != CLEARED_MARK:
                messages[i]["content"] = CLEARED_MARK
                cleared_count += 1

        if cleared_count:
            logger.info(
                "time-based MC：清除了 %d 条旧工具结果（距上次 assistant %d 分钟）",
                cleared_count, int(elapsed_min),
            )
            return messages, True
        return messages, False
    except Exception as e:
        logger.warning("time-based MC 异常（fail-open）: %s", e)
        return messages, False


def strip_internal_fields(messages: list) -> list:
    """strip 消息列表里的内部字段（如 _timestamp），不污染发给 LLM 的 prompt。

    prompt cache 神圣不可侵犯：_timestamp 等内部字段绝不能进 LLM messages。
    在 _assemble_turn_messages 组装发给 LLM 的 messages 时调用。
    """
    INTERNAL_KEYS = ("_timestamp",)
    out = []
    for m in messages:
        if any(k in m for k in INTERNAL_KEYS):
            new_m = {k: v for k, v in m.items() if k not in INTERNAL_KEYS}
            out.append(new_m)
        else:
            out.append(m)
    return out


def snip_compact(
    messages: list,
    *,
    keep_first: int = 3,
    keep_last: int = 47,
    threshold: int = 50,
) -> Tuple[list, bool]:
    """L1：消息数 > threshold 时裁中间，保留首 N + 尾 M + 占位。

    无损：占位消息提示 LLM 去 .transcripts/latest.jsonl 读回完整内容。
    成对保护：head 边界遇到 assistant(tool_calls) 时往后扩到 tool result 结束，
              避免把 tool_call 留在 head、tool result 切走，导致 OpenAI 协议报孤儿。
    返回 (新消息, 是否裁剪)。
    """
    system, conv = _split_system(messages)
    # 已有占位 → 不二次裁（幂等）
    placeholders = [m for m in conv if "snip_compact" in str(m.get("content", ""))]
    if placeholders:
        return messages, False
    if len(conv) <= threshold:
        return messages, False
    if len(conv) <= keep_first + keep_last:
        return messages, False

    # head 边界成对保护：head 末尾是 assistant(tool_calls) 或 tool_result 时，
    # 把后续的连续 tool_result 都带上（防止把同一 assistant(tc) 的多 result 拆散）。
    # - 末尾是 assistant(tc)：纳入它所有 result
    # - 末尾是 tool_result：说明 head 已装下某 assistant(tc) 的部分 result，
    #   纳入剩余的连续 result（同序列）
    head_end = keep_first
    needs_extend = (
        head_end > 0
        and head_end < len(conv)
        and (_has_tool_calls(conv[head_end - 1]) or _is_tool_result(conv[head_end - 1]))
    )
    if needs_extend:
        while head_end < len(conv) and _is_tool_result(conv[head_end]):
            head_end += 1

    # tail 边界成对保护：tail 开头是 tool_result 时，它的 assistant(tc) 必然在
    # tail 之外（tail 第一条是 tool_result 意味着 prev 是 assistant 或更早的 result，
    # 都在 tail 外）。无条件 tail_start += 1 跳过这些孤儿 result（L1 无损，可读 transcript 找回）。
    # 注意：原逻辑误把"prev 是 assistant(tc)"当成"配对完整"——但 prev 在 tail 外，
    # assistant(tc) 也不在 tail 内，仍是孤儿。
    tail_start = len(conv) - keep_last
    while tail_start < len(conv) and _is_tool_result(conv[tail_start]) and tail_start > head_end:
        tail_start += 1

    head = conv[:head_end]
    tail = conv[tail_start:]
    omitted = tail_start - head_end
    placeholder = {
        "role": "user",
        "content": (
            f"[snip_compact: 中间 {omitted} 条已省略，"
            f"完整记录见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = head + [placeholder] + tail
    new_messages = _reassemble(system, new_conv)
    logger.info("L1 snip_compact: conv %d → %d (omitted %d, head_end=%d, tail_start=%d)",
                len(conv), len(new_conv), omitted, head_end, tail_start)
    return new_messages, True


def micro_compact(
    messages: list,
    *,
    threshold: int = 10000,
    preview_chars: int = 200,
    keep_recent: int = 3,
    agent_home=None,
) -> Tuple[list, bool]:
    """L2：对齐 Claude Code microCompact——按单条大小折叠笨重的 tool 结果。

    触发：单条 tool 结果 content 超过 threshold 才折叠（不是按数量）。
    折叠：原文落盘到 .task_outputs/，占位留 full_at 指针（agent 可 read_file 读回）。
    保护：最近 keep_recent（默认 3）条 tool 结果永远不折叠。
    安全：只换 content，保留 role/tool_call_id/name（不破 tool_call 配对）。
    幂等：已是占位/已 offload 的不再动。
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return messages, False
    protected = set(tool_indices[-keep_recent:])  # 最近 keep_recent 条保护

    folded = 0
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool" or i in protected:
            out.append(m)
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= threshold:
            out.append(m)
            continue
        if _already_micro_placeheld(m) or _already_offloaded(m):
            out.append(m)
            continue

        # 落盘原文 + 留 full_at 指针（可读回，对齐 Claude Code microCompact）
        if agent_home:
            try:
                from agent.output_offload import maybe_offload
                new_content = maybe_offload(
                    content,
                    tool_call_id=m.get("tool_call_id") or f"micro_{i}",
                    agent_home=agent_home,
                    threshold=0,  # 强制落盘
                    preview_chars=preview_chars,
                )
                if new_content != content:
                    new_m = dict(m)
                    new_m["content"] = new_content
                    out.append(new_m)
                    folded += 1
                    continue
            except Exception:
                pass

        # fallback（无 agent_home / 落盘失败）：hint 占位
        new_m = dict(m)
        new_m["content"] = json.dumps({
            "micro_compacted": True,
            "orig_chars": len(content),
            "hint": (
                f"Tool {m.get('name', '?')} 结果已折叠，"
                f"完整内容见 .transcripts/latest.jsonl 或重跑工具"
            ),
        }, ensure_ascii=False)
        out.append(new_m)
        folded += 1

    if folded == 0:
        return messages, False
    logger.info("L2 micro_compact: folded %d oversized tool results", folded)
    return out, True


def _already_micro_placeheld(msg: dict) -> bool:
    """检测 tool 消息 content 是否已是 micro_compacted 占位。"""
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
        return bool(parsed.get("micro_compacted"))
    except (json.JSONDecodeError, TypeError):
        return False


def _already_offloaded(msg: dict) -> bool:
    """检测 tool 消息 content 是否已是 output_offload 占位（P1-2）。

    output_offload 的占位 JSON 含 "truncated": true 和 "full_at" 字段，
    不要再二次落盘。
    """
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
        return bool(parsed.get("truncated")) and "full_at" in parsed
    except (json.JSONDecodeError, TypeError):
        return False


def offload_large_tool_results(
    messages: list,
    *,
    agent_home,
    threshold: int = 30000,
    preview_chars: int = 2000,
) -> Tuple[list, bool]:
    """L2.5：主动扫描所有 role=tool 消息，超阈值的落盘（P1-2）。

    背景：terminal/file_ops 等工具自己调 maybe_offload（被动），
    但 read_file/search_files/execute_code/bg_result 等没接。
    如果一条 tool 消息 content 超 30K 字符就直接进 messages，爆 context。

    本函数在 compress_if_needed 编排里跑，统一兜底：扫所有 tool 消息，
    超阈值且不是占位的，主动调 maybe_offload 落盘。

    返回 (新消息, 是否有变化)。消息结构除 content 外不变（保 tool_call_id/name 配对）。
    """
    from agent.output_offload import maybe_offload

    changed = False
    out = []
    for m in messages:
        if m.get("role") != "tool":
            out.append(m)
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= threshold:
            out.append(m)
            continue
        if _already_offloaded(m):
            out.append(m)  # 已是占位，不二次落盘
            continue

        tool_call_id = m.get("tool_call_id") or f"orphan_{id(m)}"
        new_content = maybe_offload(
            content,
            tool_call_id=tool_call_id,
            agent_home=agent_home,
            threshold=threshold,
            preview_chars=preview_chars,
        )
        if new_content != content:
            new_m = dict(m)
            new_m["content"] = new_content
            out.append(new_m)
            changed = True
        else:
            out.append(m)

    if changed:
        logger.info("L2.5 offload_large_tool_results: 至少 1 条 tool 消息已落盘")
    return out, changed


def apply_context_collapse(
    messages: list,
    *,
    threshold_ratio: float = 0.8,
    context_window: int = 128_000,
    keep_recent_turns: int = 3,
) -> Tuple[list, bool]:
    """L3.5 contextCollapse：基于 token 占用比折叠早期对话段。

    Task P1.1（spec §7.1）。**轻量、无损、可逆**——比 L4 llm_compact（调 LLM 有损摘要）
    便宜得多，放在 L1 snip + L2 micro 之后、L4 llm_compact 之前做兜底。

    触发：``estimate_message_tokens(messages) / context_window > threshold_ratio``
    动作：
        1. 拆 system（**不碰**，保护 prompt cache key）
        2. 找出所有 pinned 消息（``content`` 以 ``[pinned]`` 开头，或显式 ``pinned: True``）
           → 保护不折叠
        3. 保留最近 ``keep_recent_turns`` 轮（1 轮 = user + assistant = 2 条）
        4. 其余的"中间段"折叠成一个 ``{"role": "user", "content": "[context_collapse: ...]"}`` 占位
        5. 最终顺序：system + pinned + 占位 + 最近 N 轮
        6. 调用 ``_fix_tool_call_pairs`` 兜底（避免孤儿 tool result）

    可逆性：原文由 ``snapshot_if_needed``（compress_if_needed 编排在 L4 前调用）
    或 ``.transcripts/latest.jsonl`` 保留，占位本身提示用户/LLM 去那里找。

    幂等：已含 ``[context_collapse:`` 占位 → 直接返回 ``(messages, False)``。

    Args:
        messages: 完整消息列表（含开头的 system）
        threshold_ratio: 0-1，估算 token / context_window 超过该比例才触发
        context_window: 模型上下文窗口大小（tokens）；默认 128K（DeepSeek/OpenAI 常见值）
        keep_recent_turns: 保留最近多少**轮**对话（1 轮 = user + assistant = 2 条）

    Returns:
        (新消息列表, 是否发生变化)。新列表是浅拷贝；原列表不被修改。
    """
    # 幂等：已是折叠态 → 不二次折叠
    if any(
        "[context_collapse:" in str(m.get("content", ""))
        for m in messages
    ):
        return messages, False

    # 估算 token；没超阈值直接 noop
    est_tokens = estimate_message_tokens(messages)
    if est_tokens / max(context_window, 1) <= threshold_ratio:
        return messages, False

    system, conv = _split_system(messages)
    if len(conv) < (keep_recent_turns * 2 + 2):
        # 对话太少，没什么可折叠
        return messages, False

    # 拆 pinned + 中间段 + 最近 N 轮
    # 最近 N 轮 = conv 末尾 keep_recent_turns*2 条（允许放宽边界以保 tool_call 成对：
    # 如果 tail 开头是 tool_result，往前扩到对应的 assistant(tool_calls)）
    tail_len = keep_recent_turns * 2
    tail_start = len(conv) - tail_len
    while tail_start > 0 and _is_tool_result(conv[tail_start]) and tail_start > 1:
        tail_start -= 1  # 往前找配对的 assistant(tool_calls)

    head_region = conv[:tail_start]  # 可折叠区域
    tail_region = conv[tail_start:]  # 最近 N 轮（保护）

    # 从 head_region 中挑出 pinned 消息（保留原位序）
    pinned = []
    collapsible_indices = []
    for idx, m in enumerate(head_region):
        content = m.get("content", "")
        is_pinned = (
            (isinstance(content, str) and content.startswith("[pinned]"))
            or bool(m.get("pinned"))
        )
        if is_pinned:
            pinned.append((idx, m))
        else:
            collapsible_indices.append(idx)

    if not collapsible_indices:
        # 全是 pinned，没什么可折叠
        return messages, False

    folded_turns = len(collapsible_indices) // 2  # 粗略：2 条 = 1 轮
    placeholder = {
        "role": "user",
        "content": (
            f"[context_collapse: 已折叠 {folded_turns} 轮早期对话"
            f"（{len(collapsible_indices)} 条消息），"
            "完整记录见 .transcripts/latest.jsonl]"
        ),
    }

    # 重组：system + pinned 段（按原序）+ 占位 + tail
    new_conv = [m for _, m in pinned] + [placeholder] + tail_region
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    saved_tokens = est_tokens - estimate_message_tokens(new_messages)
    logger.info(
        "L3.5 context_collapse: conv %d → %d (folded %d msgs, saved ~%d tokens)",
        len(conv), len(new_conv), len(collapsible_indices), saved_tokens,
    )
    return new_messages, True


async def llm_compact(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    keep_recent: int = 10,
    token_threshold: int = 100000,
    msg_threshold: int = 100,
    precomputed_tokens: Optional[int] = None,
) -> Tuple[list, bool]:
    """L4：L1+L2 后仍超阈值时，调 LLM 总结早期对话（async：_summarize_conversation 已改 async）。

    precomputed_tokens: 调用方预算的 token 数(避免重复遍历)。None 时内部算。

    Task D4 fix: 改 async + await _summarize_conversation。
    """
    system, conv = _split_system(messages)
    if precomputed_tokens is not None:
        over_token = precomputed_tokens > token_threshold
    else:
        over_token = estimate_message_tokens(messages) > token_threshold
    if not over_token:  # 对齐 Claude Code：压缩由 token 驱动，不按消息数
        return messages, False
    if len(conv) <= keep_recent:
        return messages, False

    to_summarize = conv[:-keep_recent]
    keep = conv[-keep_recent:]

    summary = await _summarize_conversation(to_summarize, llm_client, model=model)
    if not summary:
        return messages, False

    placeholder = {
        "role": "user",
        "content": (
            "[之前的对话已自动总结]\n\n"
            f"{summary}\n\n"
            "[以下是最近的对话，请继续]"
        ),
    }
    new_conv = [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    logger.info(
        "L4 llm_compact: %d msgs summarized, %d chars → %d chars summary",
        len(to_summarize),
        sum(len(str(m.get("content", ""))) for m in to_summarize),
        len(summary),
    )
    return new_messages, True


@dataclass
class CompressionSessionState:
    """单会话的压缩状态。

    - reacted: 本会话是否已触发过 reactive_compact（once-per-session）
    - llm_compact_count: L4 触发次数
    - last_llm_compact_turn: 上次 L4 触发时的 current_turn（用于 cooldown）
    - current_turn: 当前 LLM 轮次（由 agent 主循环 increment）
    """
    reacted: bool = False
    llm_compact_count: int = 0
    last_llm_compact_turn: int = -10**6
    current_turn: int = 0

    def record_llm_compact(self) -> None:
        self.llm_compact_count += 1
        self.last_llm_compact_turn = self.current_turn

    def cooldown_ok(self, cooldown_turns: int) -> bool:
        return self.current_turn - self.last_llm_compact_turn >= cooldown_turns

    def increment_turn(self) -> None:
        self.current_turn += 1


def reactive_compact(
    messages: list,
    *,
    session_state: CompressionSessionState,
    keep_recent: int = 5,
) -> Tuple[list, bool]:
    """紧急通道：API 报 prompt_too_long 时调用。

    只留 system + 占位 + 最后 keep_recent 条。
    会话级 once-per-session：session_state.reacted=True 后不再触发。
    """
    if session_state.reacted:
        return messages, False

    system, conv = _split_system(messages)
    keep = conv[-keep_recent:] if len(conv) > keep_recent else conv[:]
    placeholder = {
        "role": "user",
        "content": (
            "[紧急上下文压缩：API 返回 prompt_too_long，"
            f"已只保留最近 {len(keep)} 条消息。"
            "完整历史见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    session_state.reacted = True
    logger.warning("reactive_compact triggered: kept last %d", len(keep))
    return new_messages, True


async def compress_if_needed(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    config: dict,
    session_state: CompressionSessionState,
    agent_home,
    session_id: str,
    hooks_registry=None,
) -> Tuple[list, bool]:
    """分层压缩编排器。返回 (新消息, 是否发生变化)（async：L4 llm_compact 已改 async）。

    顺序：L1 snip → L2 micro → L2.6 总量预算 → **L3.5 contextCollapse** → L4 llm。
    每层独立判定是否触发，最终统一过 _fix_tool_call_pairs。

    L3.5（Task P1.1，spec §7.1）：``features.context_collapse.enabled=True`` 时，
    按 ``est_tokens / context_window > threshold_ratio`` 触发，折叠早期段为占位
    （无损、可逆；保护 system prompt + pinned）。
    放在 L4 llm_compact 前做兜底——便宜得多，能少调 LLM 摘要。

    L4 预算用 session_state.llm_compact_count，避免 L1+L2 循环误耗 L4 配额。

    hooks_registry：可选。非 None 时在压缩前后触发 PRE_COMPACT/POST_COMPACT
    事件；PRE_COMPACT 任一 hook 返回 abort 则跳过本次压缩。

    Task D4 fix: 改 async + await llm_compact。
    """
    # PRE_COMPACT hook（可 abort）
    if hooks_registry is not None:
        try:
            abort = hooks_registry.run_pre_compact({
                "session_id": session_id,
                "layer": "orchestrator",
            })
            if abort.get("abort"):
                logger.info("PRE_COMPACT hook 请求 abort，跳过压缩")
                return messages, False
        except Exception as e:
            logger.warning("PRE_COMPACT hook 触发异常（视为允许）: %s", e)

    # ── 改造点 ④：time-based MC（最早跑，无 token 检查）──
    # 距最后一条 assistant > 60min 时，把旧 tool result 内容替换为清除标记
    # 对齐 claude-code-main microCompact:evaluateTimeBasedTrigger
    # 返回 (messages, c0)：c0=True 时表示 time-based MC 清了内容，
    # 必须参与最终 changed flag（否则 compress=False 会导致 conversation_history
    # 不同步，下一轮又把原始 content 塞回去——time-based MC 效果只活一轮）
    messages, c0 = time_based_clear_old_tool_results(messages, config)

    # L1 snip（对齐 Claude Code：减少频繁裁中间，由 L4 token 主导）
    messages, c1 = snip_compact(
        messages,
        keep_first=config.get("snip_keep_first", 3),
        keep_last=config.get("snip_keep_last", 47),
        threshold=config.get("snip_message_threshold", 200),
    )

    # L2 micro（对齐 Claude Code microCompact：按单条大小折叠 + 落盘留指针 + 保最近3条）
    # 替代原 L2.5 单条 offload——micro_compact 内部按大小触发 + 落盘 + 可读回
    offload_threshold = config.get("output_offload_threshold", 10000)
    offload_preview = config.get("output_offload_preview", 2000)
    from agent.output_offload import maybe_offload
    messages, c2 = micro_compact(
        messages,
        threshold=offload_threshold,
        preview_chars=offload_preview,
        keep_recent=config.get("micro_keep_recent_results", 3),
        agent_home=agent_home,
    )

    # L2.6 总量预算：全部 tool 结果合计仍超预算 → 最大的再落盘
    c26 = False
    TOTAL_TOOL_BUDGET = config.get("tool_result_total_budget", 200_000)
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if tool_indices:
        tool_total = sum(
            len(str(messages[i].get("content", ""))) for i in tool_indices
        )
        if tool_total > TOTAL_TOOL_BUDGET:
            # 按当前长度排序,最大的先 offload
            sorted_indices = sorted(
                tool_indices,
                key=lambda i: len(str(messages[i].get("content", ""))),
                reverse=True,
            )
            for i in sorted_indices:
                if tool_total <= TOTAL_TOOL_BUDGET:
                    break
                content = messages[i].get("content", "")
                if not isinstance(content, str) or len(content) <= offload_threshold:
                    continue
                if _already_offloaded(messages[i]):
                    continue
                new_content = maybe_offload(
                    content,
                    tool_call_id=messages[i].get("tool_call_id") or f"budget_{i}",
                    agent_home=agent_home,
                    threshold=offload_threshold,
                    preview_chars=offload_preview,
                )
                if new_content != content:
                    messages[i] = dict(messages[i])
                    messages[i]["content"] = new_content
                    tool_total -= len(content) - len(new_content)
                    c26 = True
                    logger.info(
                        "L2.6 总量预算 offload: tool 消息 %d %d→%d",
                        i, len(content), len(new_content),
                    )

    # L3.5 contextCollapse（Task P1.1，spec §7.1）
    # 触发条件：flag 开 + est_tokens / context_window > threshold_ratio（默认 0.8）
    # 无损、可逆（折叠段在 .transcripts/latest.jsonl），保护 system prompt + pinned
    # 放在 L4 前——便宜得多，能挡掉很多 L4 调用
    c35 = False
    from agent.feature_flags import is_feature_enabled, get_feature_config
    if is_feature_enabled(config, "context_collapse"):
        cc_cfg = get_feature_config(config, "context_collapse")
        cc_threshold = cc_cfg.get("threshold_ratio", 0.8)
        # context_window：优先 config 显式声明的值；否则按模型推断
        context_window = config.get("context_collapse_context_window")
        if not context_window:
            if model and "[1m]" in str(model):
                context_window = 1_000_000
            else:
                context_window = 128_000  # DeepSeek/OpenAI 常见值
        # 折叠前先把 transcript 快照（L3.5 虽无损但好习惯，保持可读回）
        # 注意：L3.5 不像 L4 那样有损，这里不强制 force
        messages, c35 = apply_context_collapse(
            messages,
            threshold_ratio=cc_threshold,
            context_window=context_window,
            keep_recent_turns=config.get("context_collapse_keep_recent_turns", 3),
        )
        if c35:
            logger.info(
                "L3.5 contextCollapse 触发（ratio=%.2f, window=%d）",
                cc_threshold, context_window,
            )

    # L4 llm（条件：未超 max_attempts + cooldown 已过 + 超阈值）
    c4 = False
    max_attempts = config.get("max_compress_attempts", 3)
    cooldown = config.get("llm_compact_cooldown_turns", 5)
    llm_compact_count = session_state.llm_compact_count
    conv_len = len(_split_system(messages)[1])
    est_tokens = estimate_message_tokens(messages)

    # 方向 1: 自适应压缩阈值(1M 上下文模型放宽到 700K)
    # 1M 窗口留 30% 给输出(300K),70% 给输入(700K)
    # 对齐 Claude Code:压缩完全由 token 驱动(接近窗口才压缩),
    # 不按消息数触发(曾因"消息数 > 100 就压"导致长会话被压 51 次、agent 反复失忆)。
    token_threshold = config.get("llm_compact_token_threshold", 100000)
    if model and "[1m]" in str(model):
        token_threshold = max(token_threshold, 700000)

    over_threshold = est_tokens > token_threshold
    cooldown_ok = session_state.cooldown_ok(cooldown)
    logger.info(
        "L4 trigger check: over_threshold=%s, est_tokens=%d, conv_msgs=%d, "
        "llm_compact_count=%d/%d, cooldown_ok=%s",
        over_threshold, est_tokens, conv_len,
        llm_compact_count, max_attempts, cooldown_ok,
    )
    if over_threshold and llm_compact_count < max_attempts and cooldown_ok:
        logger.info("L4 triggered")
        # L4 前落盘 transcript（force=True，因为 L4 是有损的）
        if config.get("transcript_enabled", True):
            try:
                snapshot_if_needed(
                    messages,
                    agent_home=agent_home,
                    session_id=session_id,
                    force=True,
                    enabled=True,
                    retention=config.get("transcript_retention", 20),
                )
            except Exception as e:
                logger.warning("transcript snapshot 失败（不阻塞 L4）: %s", e)

        messages, c4 = await llm_compact(
            messages,
            llm_client=llm_client,
            model=model,
            keep_recent=config.get("llm_compact_keep_recent", 30),
            token_threshold=token_threshold,  # 自适应阈值
            precomputed_tokens=est_tokens,
        )
        if c4:
            session_state.record_llm_compact()
    elif over_threshold:
        if llm_compact_count >= max_attempts:
            logger.info("L4 skipped: max_attempts reached (%d/%d)", llm_compact_count, max_attempts)
        else:
            logger.info("L4 skipped: cooldown active (last=%d, current=%d, need=%d)",
                        session_state.last_llm_compact_turn, session_state.current_turn, cooldown)
    else:
        logger.info("L4 skipped: below threshold (est_tokens=%d, conv_msgs=%d)", est_tokens, conv_len)

    changed = c0 or c1 or c2 or c26 or c35 or c4
    if changed:
        # 终极保险：再过一遍 _fix_tool_call_pairs
        system, conv = _split_system(messages)
        messages = _reassemble(system, _fix_tool_call_pairs(conv))

    # POST_COMPACT hook（通知压缩完成）
    if hooks_registry is not None:
        try:
            hooks_registry.run_post_compact({
                "session_id": session_id,
                "layer": "orchestrator",
            })
        except Exception as e:
            logger.warning("POST_COMPACT hook 触发异常（忽略）: %s", e)

    return messages, changed
