# agent/context_pipeline.py
"""分层压缩管线：L1 snip / L2 micro / L4 llm + reactive。

替代 context_compressor.maybe_compress 的单层 LLM 摘要。
设计详见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3。
"""
import json
import logging
from typing import Optional, Tuple

from agent.context_compressor import (
    _summarize_conversation, _fix_tool_call_pairs, estimate_message_tokens,
)

logger = logging.getLogger(__name__)


def _split_system(messages: list) -> Tuple[Optional[dict], list]:
    """分离 system 消息（如果有）。返回 (system_msg_or_None, rest)。"""
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages


def _reassemble(system: Optional[dict], conv: list) -> list:
    """重新组装：system（若有）+ conv。"""
    return [system, *conv] if system else conv


def snip_compact(
    messages: list,
    *,
    keep_first: int = 3,
    keep_last: int = 47,
    threshold: int = 50,
) -> Tuple[list, bool]:
    """L1：消息数 > threshold 时裁中间，保留首 N + 尾 M + 占位。

    无损：占位消息提示 LLM 去 .transcripts/latest.jsonl 读回完整内容。
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

    head = conv[:keep_first]
    tail = conv[-keep_last:]
    omitted = len(conv) - keep_first - keep_last
    placeholder = {
        "role": "user",
        "content": (
            f"[snip_compact: 中间 {omitted} 条已省略，"
            f"完整记录见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = head + [placeholder] + tail
    new_messages = _reassemble(system, new_conv)
    logger.info("L1 snip_compact: conv %d → %d (omitted %d)",
                len(conv), len(new_conv), omitted)
    return new_messages, True


def micro_compact(
    messages: list,
    *,
    keep_recent: int = 3,
) -> Tuple[list, bool]:
    """L2：把较旧的 tool 消息 content 替换为占位 JSON。

    无损：占位提示去 .transcripts/latest.jsonl 或重跑工具。
    安全：只换 content，保留 role/tool_call_id/name（不破 tool_call 配对）。
    幂等：已是占位的不再动。
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return messages, False

    to_compact = set(tool_indices[:-keep_recent])  # 除最后 keep_recent 个外
    folded = 0
    out = []
    for i, m in enumerate(messages):
        if i in to_compact and not _already_micro_placeheld(m):
            new_m = dict(m)
            orig_len = len(str(m.get("content", "")))
            new_m["content"] = json.dumps({
                "micro_compacted": True,
                "orig_chars": orig_len,
                "hint": (
                    f"Tool {m.get('name', '?')} 结果已折叠，"
                    f"完整内容见 .transcripts/latest.jsonl 或重跑工具"
                ),
            }, ensure_ascii=False)
            out.append(new_m)
            folded += 1
        else:
            out.append(m)

    if folded == 0:
        return messages, False
    logger.info("L2 micro_compact: folded %d old tool results", folded)
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


def llm_compact(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    keep_recent: int = 10,
    token_threshold: int = 100000,
    msg_threshold: int = 100,
) -> Tuple[list, bool]:
    """L4：L1+L2 后仍超阈值时，调 LLM 总结早期对话。

    有损：用 1 次 API 调用换上下文空间。调用方应先 transcript.snapshot_if_needed(force=True)。
    沿用现有 _summarize_conversation（含 _rule_based_summary 降级）和 _fix_tool_call_pairs。
    """
    system, conv = _split_system(messages)
    over_token = estimate_message_tokens(messages) > token_threshold
    over_msg = len(conv) > msg_threshold
    if not (over_token or over_msg):
        return messages, False
    if len(conv) <= keep_recent:
        return messages, False

    to_summarize = conv[:-keep_recent]
    keep = conv[-keep_recent:]

    summary = _summarize_conversation(to_summarize, llm_client, model=model)
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
