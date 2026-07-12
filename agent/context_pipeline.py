# agent/context_pipeline.py
"""分层压缩管线：L1 snip / L2 micro / L4 llm + reactive。

替代 context_compressor.maybe_compress 的单层 LLM 摘要。
设计详见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3。
"""
import logging
from typing import Optional, Tuple

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
