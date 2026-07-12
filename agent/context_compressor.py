"""上下文压缩：当历史接近 token 上限时，总结早期对话。

这是唯一允许调用 agent.invalidate_system_prompt() 的场景。

策略：
1. 监控请求的 token 数（消息数作为近似）
2. 超过阈值时触发
3. 把早期消息用轻量 LLM 总结成一条
4. 保留最近几轮完整对话
5. 修复可能被压缩边界破坏的 tool_call 配对
6. 重建 system prompt
"""

import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# 压缩配置
MAX_COMPRESS_ATTEMPTS = 3          # 一个会话最多压缩 3 次
COMPRESS_COOLDOWN_TURNS = 5         # 压缩后 5 轮内不再压缩（未使用，预留）
MESSAGES_BEFORE_COMPRESS = 40       # 消息数阈值
KEEP_RECENT_MESSAGES = 10           # 保留最近 10 条消息（约 5 轮）


def maybe_compress(
    messages: list,
    *,
    attempt_count: int = 0,
    model: str = None,
    llm_client=None,
    context_window_tokens: int = 128000,
) -> Tuple[list, bool]:
    """检查并执行压缩。

    返回 (新消息列表, 是否压缩了)。

    messages: 完整消息列表（含 system 在最前）
    llm_client: OpenAI 兼容客户端（用于调用轻量模型总结）

    .. deprecated::
        单层 LLM 摘要压缩。新代码请用 ``agent.context_pipeline.compress_if_needed``。
        保留是为了双轨期回退（``config.context.use_new_pipeline=False`` 时仍调用）。
        下个 minor 版本完全移除。
    """
    import warnings
    warnings.warn(
        "maybe_compress 已废弃，请改用 agent.context_pipeline.compress_if_needed",
        DeprecationWarning,
        stacklevel=2,
    )
    # 消息数不够，不压缩
    if len(messages) < MESSAGES_BEFORE_COMPRESS:
        return messages, False

    # 压缩次数用完
    if attempt_count >= MAX_COMPRESS_ATTEMPTS:
        return messages, False

    # 分离 system prompt
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    conversation = messages[1:] if system_msg else messages

    # 要压缩的部分（跳过最近几条）
    if len(conversation) <= KEEP_RECENT_MESSAGES:
        return messages, False

    to_compress = conversation[:-KEEP_RECENT_MESSAGES]
    keep_recent = conversation[-KEEP_RECENT_MESSAGES:]

    # 调用 LLM 总结
    summary = _summarize_conversation(to_compress, llm_client, model=model)

    if not summary:
        return messages, False  # 总结失败

    # 重组
    new_messages = []
    if system_msg:
        new_messages.append(system_msg)

    # 总结作为 user 消息注入
    new_messages.append({
        "role": "user",
        "content": (
            "[之前的对话已自动总结]\n\n"
            f"{summary}\n\n"
            "[以下是最近的对话，请继续]"
        ),
    })

    # 最近的消息保留（修复可能被压缩边界破坏的 tool_call 配对）
    new_messages.extend(_fix_tool_call_pairs(keep_recent))

    logger.info(
        "上下文已压缩: %d 条消息 → %d 条（总结: %d 字符）",
        len(messages), len(new_messages), len(summary),
    )

    return new_messages, True


def _summarize_conversation(
    messages: list,
    llm_client=None,
    *,
    model: str = None,
    summary_model: str = None,
) -> str:
    """调用 LLM 总结对话历史。

    使用轻量模型（如果客户端可用），否则返回占位总结。
    """
    # 格式化对话
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "tool":
            # 工具结果截断
            content = content[:200] + "..." if len(content) > 200 else content
            formatted.append(f"[工具结果] {content}")
        elif role == "assistant" and msg.get("tool_calls"):
            tool_names = [
                tc.get("function", {}).get("name", "?")
                for tc in msg["tool_calls"]
            ]
            formatted.append(f"[助手调用了工具: {', '.join(tool_names)}]")
            if content:
                formatted.append(f"[助手补充] {content[:200]}")
        else:
            formatted.append(f"[{role}] {content}")

    dialog = "\n\n".join(formatted)

    prompt = (
        "请把以下对话总结成关键信息，保留：\n"
        "1. 用户的核心需求\n"
        "2. 已完成的工作\n"
        "3. 关键的决策和发现\n"
        "4. 待办的事项\n"
        "5. 重要的文件路径、命令、配置\n\n"
        "用简洁的要点格式，不要超过 800 字。\n\n"
        f"对话内容:\n{dialog}"
    )

    if llm_client is None:
        # 无客户端时返回占位总结（避免完全丢失上下文）
        return _rule_based_summary(messages)

    try:
        response = llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning("LLM 压缩总结失败，使用规则提取: %s", e)
        return _rule_based_summary(messages)


def _rule_based_summary(messages: list) -> str:
    """无 LLM 时的降级规则提取。

    保留所有 user 消息 + 关键 assistant 消息。
    """
    lines = ["[规则提取（LLM 不可用）]"]
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:150]
        if role == "user" and content:
            lines.append(f"- 用户: {content}")
        elif role == "assistant" and content and not msg.get("tool_calls"):
            lines.append(f"- 助手: {content}")
    return "\n".join(lines)[:1500]


def _fix_tool_call_pairs(messages: list) -> list:
    """修复压缩边界可能破坏的 tool_call 配对。

    如果压缩点恰好在 assistant(tool_calls) 和 tool 结果之间，
    需要补一条假的 tool 结果，否则 API 会报错。
    """
    fixed = []
    pending_tool_calls = {}

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                pending_tool_calls[tc["id"]] = tc["function"]["name"]
            fixed.append(msg)
        elif msg.get("role") == "tool":
            pending_tool_calls.pop(msg.get("tool_call_id"), None)
            fixed.append(msg)
        else:
            fixed.append(msg)

    # 如果有未配对的 tool_call，补一条 tool 结果
    if pending_tool_calls:
        for call_id, name in pending_tool_calls.items():
            fixed.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps({
                    "note": "此工具调用的结果因上下文压缩而丢失",
                }, ensure_ascii=False),
            })

    return fixed


def estimate_message_tokens(messages: list) -> int:
    """粗略估算消息列表的 token 数。

    简化公式：字符数 / 3（中英文混合的经验值）。
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total_chars += len(str(content))
        for tc in msg.get("tool_calls", []) or []:
            total_chars += len(str(tc.get("function", {}).get("arguments", "")))
    return total_chars // 3
