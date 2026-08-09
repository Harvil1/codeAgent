# agent/context_pipeline.py
"""分层压缩管线：L1 snip / L2 micro / L4 llm + reactive。

替代 context_compressor.maybe_compress 的单层 LLM 摘要。
设计详见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3。
"""
import json
import logging
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

    顺序：L1 snip → L2 micro → (条件) transcript 快照 → L4 llm。
    每层独立判定是否触发，最终统一过 _fix_tool_call_pairs。

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

    changed = c1 or c2 or c26 or c4
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
