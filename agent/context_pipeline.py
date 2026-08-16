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
from datetime import datetime
from typing import Optional, Tuple

from agent.context_compressor import (
    _summarize_conversation, _fix_tool_call_pairs, estimate_message_tokens,
    reset_compact_circuit_breaker,
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

    CCAR8 Task 11：新增 `_ephemeral` 标记，标记的消息 strip 后保留 content/role
    但去除标记本身（_ephemeral 是 AIAgent 内部追踪用的，LLM 不需要看到）。
    """
    INTERNAL_KEYS = ("_timestamp", "_ephemeral")
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
    # 用占位前缀 "[snip_compact:" + role=="user" + startswith 三重限定
    # 不能用裸子串 "snip_compact"——用户消息提到这字样会误判（Bug 5）
    # 不能只看子串 "[snip_compact:"——tool 消息读源码/输出含该子串也误判（Bug 6）
    # 真实占位格式见 line 195-199，role 是 user，content 以 "[snip_compact:" 开头
    placeholders = [
        m for m in conv
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("[snip_compact:")
    ]
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
    threshold: int = 50000,
    preview_chars: int = 2000,
    message_threshold: int = 200000,
    freeze: bool = True,
) -> Tuple[list, bool]:
    """L2.5：主动扫描所有 role=tool 消息，超阈值落盘（改造点 ① 精细化）。

    三层触发逻辑：
      1. **per-tool 阈值**（threshold，默认 50K）：单条 tool result 超阈值 → 落盘
      2. **per-message 聚合阈值**（message_threshold，默认 200K）：一段连续 tool result
         （不跨 user/assistant 边界）总和超阈值 → 按大小降序逐个落盘直到总和 < 阈值
      3. **跨轮次决策冻结**（freeze=True）：已落盘的 tool_call_id 直接从 _offload_decisions
         重放预览内容，不重新评估（保护 prompt cache，保证 byte-identical）

    返回 (新消息, 是否有变化)。消息结构除 content 外不变（保 tool_call_id/name 配对）。
    """
    from agent.output_offload import maybe_offload

    changed = False
    out = []
    for m in messages:
        if m.get("role") != "tool":
            out.append(m)
            continue

        tc_id = m.get("tool_call_id") or ""

        # ── 决策冻结：已落盘的直接重放（byte-identical，保护 prompt cache）──
        if freeze and tc_id and tc_id in _offload_decisions:
            decision = _offload_decisions[tc_id]
            # 直接用记录的预览内容替换（不重新评估）
            if m.get("content") != decision["preview"]:
                new_m = dict(m)
                new_m["content"] = decision["preview"]
                out.append(new_m)
                # 冻结重放不算 changed（没新落盘，只是保持一致）
            else:
                out.append(m)
            continue

        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= threshold:
            out.append(m)
            continue
        if _already_offloaded(m):
            out.append(m)  # 已是占位（来自其他路径），不二次落盘
            continue

        # per-tool 阈值触发
        effective_tc_id = tc_id or f"orphan_{id(m)}"
        new_content = maybe_offload(
            content,
            tool_call_id=effective_tc_id,
            agent_home=agent_home,
            threshold=threshold,
            preview_chars=preview_chars,
        )
        if new_content != content:
            new_m = dict(m)
            new_m["content"] = new_content
            out.append(new_m)
            changed = True
            if freeze and tc_id:
                _record_decision(tc_id, new_content)
        else:
            out.append(m)

    # ── per-message 聚合检查 ──
    if message_threshold > 0:
        agg_changed = _enforce_per_message_budget(
            out, message_threshold, agent_home, preview_chars, freeze,
        )
        if agg_changed:
            changed = True

    if changed:
        logger.info("L2.5 offload_large_tool_results: 至少 1 条 tool 消息已落盘（精细化）")
    return out, changed


# ---------------------------------------------------------------------------
# 改造点 ①：决策冻结 + per-message 聚合
# ---------------------------------------------------------------------------

_offload_decisions: dict = {}  # tool_call_id -> {"preview": str, "file_path": str|None}
_OFFLOAD_DECISIONS_LIMIT = 1000  # LRU 上限，防长会话内存膨胀


def _record_decision(tc_id: str, preview: str, file_path: str = None) -> None:
    """记录落盘决策到 _offload_decisions，超 _OFFLOAD_DECISIONS_LIMIT 时 LRU 淘汰。

    dict 在 Py3.7+ 保序（插入顺序），简化版 LRU：超限时删最早的（next(iter)）。
    """
    if len(_offload_decisions) >= _OFFLOAD_DECISIONS_LIMIT:
        # 淘汰最早的一个（dict 在 Py3.7+ 保序）
        oldest = next(iter(_offload_decisions))
        del _offload_decisions[oldest]
    _offload_decisions[tc_id] = {"preview": preview, "file_path": file_path}


def reset_offload_decisions() -> None:
    """会话开始时清空决策（避免跨会话泄漏）。

    在 AIAgent.__init__ 调用，保证新会话不复用上一会话的落盘决策。
    """
    _offload_decisions.clear()


def _enforce_per_message_budget(
    messages: list,
    limit: int,
    agent_home,
    preview_chars: int,
    freeze: bool,
) -> bool:
    """per-message 聚合检查：一段连续 tool result 总和 > limit 时选最大的几个落盘。

    分组规则（对齐 spec）：按 **user** 消息边界分组——一段连续的 tool result
    （中间可以有 assistant(tool_calls)，但不能跨 user 消息）算一组。
    这反映了"一次用户输入触发的所有工具调用"是一个逻辑单元。

    超限的组：按 size 降序，逐个落盘直到总和 < limit。

    注意：已在 _offload_decisions 里（决策冻结命中）的不重复处理；
    已是占位（_already_offloaded）的跳过。
    """
    from agent.output_offload import maybe_offload

    changed = False

    # 1. 按 user 消息边界分组：收集所有 tool result 索引，
    #    遇到新 user 消息就开新段
    segments = []  # list of list of indices
    current_seg = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            current_seg.append(i)
        elif m.get("role") == "user":
            # user 消息是分组边界——user 之后的 tool result 属于新段
            if current_seg:
                segments.append(current_seg)
                current_seg = []
        # assistant / system 消息不打断段（tool result 中间可以有 assistant(tool_calls)）
    if current_seg:
        segments.append(current_seg)

    # 2. 对每段算总和，超 limit 的按 size 降序逐个落盘
    for seg in segments:
        # 过滤掉已是占位或决策冻结命中的（它们 content 已经很小）
        candidates = []
        seg_total = 0
        for idx in seg:
            m = messages[idx]
            content = m.get("content", "")
            seg_total += len(content) if isinstance(content, str) else 0
            tc_id = m.get("tool_call_id") or ""
            # 决策冻结命中的或已是占位的不候选
            if freeze and tc_id and tc_id in _offload_decisions:
                continue
            if _already_offloaded(m):
                continue
            if not isinstance(content, str):
                continue
            candidates.append((idx, len(content)))

        if seg_total <= limit:
            continue
        if not candidates:
            continue

        # 按 size 降序，逐个落盘直到总和 < limit
        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, size in candidates:
            if seg_total <= limit:
                break
            m = messages[idx]
            content = m.get("content", "")
            tc_id = m.get("tool_call_id") or f"agg_{idx}"
            new_content = maybe_offload(
                content,
                tool_call_id=tc_id,
                agent_home=agent_home,
                threshold=0,  # 强制落盘（聚合触发）
                preview_chars=preview_chars,
            )
            if new_content != content:
                new_m = dict(m)
                new_m["content"] = new_content
                messages[idx] = new_m
                seg_total -= size - len(new_content)
                changed = True
                if freeze and tc_id:
                    _record_decision(tc_id, new_content)
                logger.info(
                    "per-message 聚合 offload: tool 消息 idx=%d %d→%d",
                    idx, size, len(new_content),
                )

    return changed


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


def _build_compact_boundary(coverage: str, preserved: str) -> str:
    """构造 compact boundary 标注（T8，对齐 CCB annotateBoundaryWithPreservedSegment）。

    三要素：压缩时间 / 摘要覆盖范围 / 保留段范围，
    外加"保留段精确 vs 摘要转述"提示——帮模型区分哪些内容是原文、
    哪些是转述（引用具体数据/路径/命令输出时以保留段为准）。
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "[compact_boundary]\n"
        f"- 压缩时间：{ts}\n"
        f"- 摘要覆盖范围：{coverage}（由 LLM 转述，细节可能有省略）\n"
        f"- 保留段范围：{preserved}（原样保留，含工具结果原文）\n"
        "- 注意：保留段内容是精确的，摘要内容是转述；"
        "引用具体数据/路径/命令输出以保留段为准。\n"
    )


async def llm_compact(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    keep_recent: int = 10,
    token_threshold: int = 100000,
    msg_threshold: int = 100,
    precomputed_tokens: Optional[int] = None,
    session_memory: Optional[str] = None,
    from_idx: int = 0,
    up_to_idx: int = -1,
) -> Tuple[list, bool]:
    """L4：L1+L2 后仍超阈值时，调 LLM 总结早期对话（async：_summarize_conversation 已改 async）。

    precomputed_tokens: 调用方预算的 token 数(避免重复遍历)。None 时内部算。

    session_memory: 预提取的 session memory（改造点 ② 软目标）。
    有值时传给 _summarize_conversation 替代 LLM 摘要。
    SessionStore.get_memory_extract 尚未实现，目前永远 None（Phase 2 再接入）。

    Task C（partial compact）：
    - **from_idx/up_to_idx**：只压 conv[from_idx:up_to_idx] 段，保留 head + tail 原文
    - 默认 0/-1 = 全量（向后兼容，走原 keep_recent 逻辑）
    - partial 模式时 keep_recent 被忽略（from/up_to 完全决定切片）

    Task D4 fix: 改 async + await _summarize_conversation。
    """
    system, conv = _split_system(messages)
    if precomputed_tokens is not None:
        over_token = precomputed_tokens > token_threshold
    else:
        over_token = estimate_message_tokens(messages) > token_threshold
    if not over_token:  # 对齐 Claude Code：压缩由 token 驱动，不按消息数
        return messages, False

    # Task C：partial 模式 vs 全量模式
    is_partial = from_idx != 0 or up_to_idx != -1

    if is_partial:
        # partial 模式：head + summary + tail 拼装
        effective_up_to = len(conv) if up_to_idx < 0 else up_to_idx
        head = conv[:from_idx]
        tail = conv[effective_up_to:] if effective_up_to < len(conv) else []

        summary = await _summarize_conversation(
            conv,  # 传完整 conv，由 _summarize_conversation 内部切片
            llm_client, model=model,
            session_memory=session_memory,
            from_idx=from_idx,
            up_to_idx=effective_up_to,
        )
        if not summary:
            return messages, False

        placeholder = {
            "role": "user",
            "content": (
                _build_compact_boundary(
                    f"消息 {from_idx}-{effective_up_to}",
                    f"head（消息 0~{from_idx}，{len(head)} 条原文）"
                    f"+ tail（消息 {effective_up_to}~，{len(tail)} 条原文）",
                )
                + f"\n[对话摘要（{from_idx}-{effective_up_to}）]\n\n"
                f"{summary}\n\n"
                "[以下是压缩段之后的对话，请继续]"
            ),
        }
        new_conv = head + [placeholder] + tail
        new_conv = _fix_tool_call_pairs(new_conv)
        new_messages = _reassemble(system, new_conv)

        summarized_count = effective_up_to - from_idx
        logger.info(
            "L4 llm_compact (partial %d-%d): %d msgs summarized, head=%d tail=%d",
            from_idx, effective_up_to, summarized_count,
            len(head), len(tail),
        )
        try:
            from agent.cache_monitor import notify_compaction
            notify_compaction()
        except Exception as e:
            logger.debug("notify_compaction fail-open: %s", e)
        return new_messages, True

    # 全量模式（原逻辑，向后兼容）
    if len(conv) <= keep_recent:
        return messages, False

    to_summarize = conv[:-keep_recent]
    keep = conv[-keep_recent:]

    summary = await _summarize_conversation(
        to_summarize, llm_client, model=model,
        session_memory=session_memory,
    )
    if not summary:
        return messages, False

    placeholder = {
        "role": "user",
        "content": (
            _build_compact_boundary(
                f"conv 第 1~{len(to_summarize)} 条消息（共 {len(to_summarize)} 条）",
                f"最近 {len(keep)} 条消息",
            )
            + "\n[之前的对话已自动总结]\n\n"
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
    # 改造点 ③：通知 cache_monitor 下次 cache 下降是预期的（compact 压缩了 messages）
    # 放在 return 前，确保只在实际发生压缩时通知
    try:
        from agent.cache_monitor import notify_compaction
        notify_compaction()
    except Exception as e:
        logger.debug("notify_compaction fail-open: %s", e)
    return new_messages, True


@dataclass
class CompressionSessionState:
    """单会话的压缩状态。

    - reactive_last_at: 上次 reactive_compact 触发的 time.time()（0=从未触发）
    - reactive_count: 本会话 reactive_compact 已触发次数
    - llm_compact_count: L4 触发次数
    - last_llm_compact_turn: 上次 L4 触发时的 current_turn（用于 cooldown）
    - current_turn: 当前 LLM 轮次（由 agent 主循环 increment）
    - llm_compact_failures: R18 #18 L4 连续失败计数（触发熔断用；
      摘要生成层的熔断在 context_compressor 的模块级状态里，这里管的是
      「触发」层——失败后本会话不再触发 L4，省无效的摘要调用）

    向后兼容：``reacted`` 属性保留为只读代理（``reactive_count > 0``），
    旧代码读 ``state.reacted`` 不破坏。
    """
    reactive_last_at: float = 0.0
    reactive_count: int = 0
    llm_compact_count: int = 0
    last_llm_compact_turn: int = -10**6
    current_turn: int = 0
    llm_compact_failures: int = 0

    @property
    def reacted(self) -> bool:
        """向后兼容：reacted 等价于 reactive_count > 0。"""
        return self.reactive_count > 0

    def record_llm_compact(self) -> None:
        self.llm_compact_count += 1
        self.last_llm_compact_turn = self.current_turn

    def cooldown_ok(self, cooldown_turns: int) -> bool:
        return self.current_turn - self.last_llm_compact_turn >= cooldown_turns

    def increment_turn(self) -> None:
        self.current_turn += 1


# R18 #18：L4 触发熔断阈值（对齐 CCB MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES=3）
MAX_CONSECUTIVE_L4_FAILURES = 3


# 冷却窗口 + 上限的默认值（可被 config 覆盖）
REACTIVE_COOLDOWN_SECONDS = 60
REACTIVE_MAX_PER_SESSION = 5


def reactive_compact(
    messages: list,
    *,
    session_state: CompressionSessionState,
    keep_recent: int = 5,
    cooldown_seconds: float = REACTIVE_COOLDOWN_SECONDS,
    max_per_session: int = REACTIVE_MAX_PER_SESSION,
    now_fn=time.time,
) -> Tuple[list, bool]:
    """紧急通道：API 报 prompt_too_long 时调用。

    只留 system + 占位 + 最后 keep_recent 条。
    **多次触发**（Task D 改造）：每次 PTL 都可触发，受两层保护：
      1. **冷却窗口**：距上次触发 < ``cooldown_seconds`` 则跳过（默认 60s）
      2. **单会话上限**：已触发 ``max_per_session`` 次则跳过（默认 5）

    ``now_fn`` 参数仅为测试注入用（生产代码不传）。
    """
    # 保护 1：冷却窗口
    now = now_fn()
    elapsed = now - session_state.reactive_last_at
    if session_state.reactive_count > 0 and elapsed < cooldown_seconds:
        logger.info(
            "reactive_compact 冷却中（距上次 %ds < %ds），跳过",
            int(elapsed), int(cooldown_seconds),
        )
        return messages, False

    # 保护 2：单会话上限
    if session_state.reactive_count >= max_per_session:
        logger.warning(
            "reactive_compact 达到单会话上限 %d 次，跳过",
            session_state.reactive_count,
        )
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

    session_state.reactive_last_at = now
    session_state.reactive_count += 1
    logger.warning(
        "reactive_compact triggered (#%d): kept last %d",
        session_state.reactive_count, len(keep),
    )
    # 通知 cache_monitor：下次 cache 下降是预期压缩（对齐 llm_compact 的 pattern）
    # fail-open：异常只 debug log，不影响压缩结果
    try:
        from agent.cache_monitor import notify_compaction
        notify_compaction()
    except Exception as e:
        logger.debug("notify_compaction fail-open: %s", e)
    return new_messages, True


def estimate_turn_growth(messages: list, *, window: int = 3, default: int = 8000) -> int:
    """预估"下一轮还要烧多少 token"（T1，防压缩震荡）。

    对齐 CCB autoCompact 的 estimateMaxTurnGrowth：阈值 = 有效窗口 − buffer −
    单轮增长预估。这里取最近 window 轮（user 边界分组）的**单轮 token
    大小最大值**作为增长预估——一轮大工具结果进来会直接把下一轮顶过线，
    提前压缩避免"压完→下一轮又到线→再压"的震荡。

    Args:
        messages: 完整消息列表（含 system，会被跳过）
        window: 观察窗口（最近几轮），config context.llm_compact_growth_window
        default: 历史不足 window 轮时的保守默认，config llm_compact_growth_default

    Returns:
        预估增量（tokens）。空/历史不足 → default。
    """
    try:
        _, conv = _split_system(messages)
        # 按 user 边界分轮：一条 user 消息 + 后续 assistant/tool 直到下一条 user
        turns: list = []
        current: list = []
        for m in conv:
            if m.get("role") == "user":
                if current:
                    turns.append(current)
                current = [m]
            elif current:
                current.append(m)
        if current:
            turns.append(current)

        if len(turns) < max(1, window):
            return default
        recent = turns[-window:]
        return max(estimate_message_tokens(t) for t in recent)
    except Exception:
        return default


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

    顺序：L1 snip → L2 micro（per-tool）→ L2.5 per-message 聚合 → L2.6 总量预算
    → **L3.5 contextCollapse** → L4 llm。
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
    # 改造点 ①：threshold 默认从 10K 提到 50K（精细化，避免小结果也落盘）
    offload_threshold = config.get("output_offload_threshold", 50000)
    offload_preview = config.get("output_offload_preview", 2000)
    offload_freeze = config.get("offload_decision_freeze", True)
    from agent.output_offload import maybe_offload

    # ── 改造点 ①：决策冻结预处理 ──
    # 已落盘的 tool_call_id 直接从 _offload_decisions 重放预览内容（byte-identical）
    # 放在 L2 之前——冻结重放让 L2 看到的 content 已经是预览（不会重复落盘）
    c_freeze = False
    if offload_freeze and _offload_decisions:
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            tc_id = m.get("tool_call_id") or ""
            if not tc_id or tc_id not in _offload_decisions:
                continue
            decision = _offload_decisions[tc_id]
            if m.get("content") != decision["preview"]:
                messages[i] = dict(m)
                messages[i]["content"] = decision["preview"]
                c_freeze = True

    messages, c2 = micro_compact(
        messages,
        threshold=offload_threshold,
        preview_chars=offload_preview,
        keep_recent=config.get("micro_keep_recent_results", 3),
        agent_home=agent_home,
    )
    # 记录 micro_compact 产生的新落盘决策
    if offload_freeze and c2:
        for m in messages:
            if m.get("role") != "tool":
                continue
            tc_id = m.get("tool_call_id") or ""
            if not tc_id or tc_id in _offload_decisions:
                continue
            content = m.get("content", "")
            if isinstance(content, str) and _already_offloaded(m):
                _record_decision(tc_id, content)

    # ── L2.5per_msg：per-message 聚合预算（改造点 ① 接入生产路径）──
    # 按 user 消息边界分组，一段连续 tool result 总和 > message_offload_threshold
    # 时按大小降序逐个落盘。这比 L2.6 全局预算更精细——L2.6 只看全局总和，
    # 不区分哪个 user turn 的工具结果。per-message 先按段处理，L2.6 做最后兜底。
    # 顺序：L2 micro（per-tool）→ L2.5per_msg（per-message 聚合）→ L2.6（全局预算）
    c_per_msg = False
    msg_threshold = config.get("message_offload_threshold", 200_000)
    if msg_threshold > 0:
        c_per_msg = _enforce_per_message_budget(
            messages,
            limit=msg_threshold,
            agent_home=agent_home,
            preview_chars=offload_preview,
            freeze=offload_freeze,
        )
        if c_per_msg:
            logger.info("L2.5 per-message 聚合 offload: 按 user 边界分组落盘")

    # L2.6 总量预算：全部 tool 结果合计仍超预算 → 最大的再落盘（全局兜底）
    # 改造点 ① Round 1 fix：解耦 L2.6 budget 与 message_offload_threshold
    #   L2.6 读 tool_result_total_budget（默认 200K），不再 aliasing message_offload_threshold
    #   ——两者语义不同：message_offload_threshold 是 per-segment 阈值，
    #   tool_result_total_budget 是全局 tool 结果总量上限
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
                    # 改造点 ①：记录决策（跨轮次 byte-identical 重放）
                    if offload_freeze:
                        tc_id = messages[i].get("tool_call_id") or f"budget_{i}"
                        _record_decision(tc_id, new_content)
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

    # T1（核心机制对齐第 1 项）：单轮增长预估——est + growth >= threshold 提前触发。
    # 一次大工具结果进来会直接把下一轮顶过线，等真到线再压就是
    # "压完→下一轮又到线→再压"的震荡；提前量 = 最近几轮的最大单轮增速。
    growth = estimate_turn_growth(
        messages,
        window=config.get("llm_compact_growth_window", 3),
        default=config.get("llm_compact_growth_default", 8000),
    )
    over_threshold = est_tokens + growth >= token_threshold
    cooldown_ok = session_state.cooldown_ok(cooldown)
    # R18 #18：L4 触发熔断——连续失败达阈值本会话不再触发（触发层熔断，
    # 与摘要生成层熔断互补：前者省无效调用，后者降级规则总结）
    tripped = session_state.llm_compact_failures >= MAX_CONSECUTIVE_L4_FAILURES
    logger.info(
        "L4 trigger check: over_threshold=%s, est_tokens=%d, growth=%d, conv_msgs=%d, "
        "llm_compact_count=%d/%d, cooldown_ok=%s, failures=%d%s",
        over_threshold, est_tokens, growth, conv_len,
        llm_compact_count, max_attempts, cooldown_ok,
        session_state.llm_compact_failures,
        " (TRIPPED)" if tripped else "",
    )
    if over_threshold and llm_compact_count < max_attempts and cooldown_ok and not tripped:
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
            # T1：传入 est+growth（提前触发时 est 可能未到 threshold，
            # llm_compact 内部门槛用同一个"下一轮预期水位"判定，避免二次拦截）
            precomputed_tokens=est_tokens + growth,
        )
        if c4:
            session_state.record_llm_compact()
            session_state.llm_compact_failures = 0  # 成功清零（R18 #18）
        else:
            # R18 #18：触发后摘要失败（返回未压缩）→ 连续失败计数 +1
            session_state.llm_compact_failures += 1
            logger.warning(
                "L4 触发但压缩未生效（连续失败 %d/%d）",
                session_state.llm_compact_failures, MAX_CONSECUTIVE_L4_FAILURES,
            )
    elif over_threshold:
        if tripped:
            logger.info(
                "L4 skipped: 触发熔断（连续失败 %d 次）",
                session_state.llm_compact_failures,
            )
        elif llm_compact_count >= max_attempts:
            logger.info("L4 skipped: max_attempts reached (%d/%d)", llm_compact_count, max_attempts)
        else:
            logger.info("L4 skipped: cooldown active (last=%d, current=%d, need=%d)",
                        session_state.last_llm_compact_turn, session_state.current_turn, cooldown)
    else:
        logger.info("L4 skipped: below threshold (est_tokens=%d, conv_msgs=%d)", est_tokens, conv_len)

    changed = c0 or c1 or c_freeze or c2 or c_per_msg or c26 or c35 or c4
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
