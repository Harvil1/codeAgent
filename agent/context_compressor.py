"""上下文压缩工具函数。

原 ``maybe_compress`` 单层 LLM 摘要路径已在 Phase 1 Commit 7 移除，
由 ``agent.context_pipeline.compress_if_needed``（4 层管线）取代。

本模块保留下列被 pipeline 复用的工具函数：
    - ``_summarize_conversation``：调用 LLM 总结对话
    - ``_rule_based_summary``：无 LLM 时的降级规则提取
    - ``_fix_tool_call_pairs``：修复压缩边界破坏的 tool_call 配对
    - ``estimate_message_tokens``：粗略估算 token 数
"""

import json
import logging

logger = logging.getLogger(__name__)


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
        "请把以下对话总结成关键信息。\n\n"
        "**必须逐字保留(不能省略/改写)**:\n"
        "- 文件路径(如 D:/project/xxx.py、~/.agent/workspace/xxx)\n"
        "- 命令(如 uv add、pip install、taskkill、netstat、browser-use)\n"
        "- 用户明确的要求/约束(如\"不要加注释\"\"用 uv 不用 pip\")\n"
        "- 关键决策(如\"选方案 A 因为 B\"\"Chrome 需要 taskkill 后重启\")\n"
        "- 错误关键词(如\"PYTHONHOME 冲突\"\"9222 端口未监听\"\"API key 错误\")\n"
        "- 端口号、IP 地址、版本号(如 9222、3.12、0.13.6)\n\n"
        "另外保留:\n"
        "1. 用户的核心需求\n"
        "2. 已完成的工作\n"
        "3. 待办的事项\n\n"
        "用简洁的要点格式,不要超过 800 字。\n\n"
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

    处理两类协议违反（OpenAI 要求 tool result 必须紧跟 tool_calls 配对）：
    - 正向孤儿：assistant(tool_calls) 后没 tool result → 补一条假 tool result
    - 反向孤儿：tool 消息的 tool_call_id 不在前面任何 assistant(tool_calls) 里 → 删掉
      （压缩边界或上游 bug 可能产生这种孤儿，会让 API 报 400:
       "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"）
    """
    # 第一遍：收集所有合法 tool_call_ids（来自 assistant(tool_calls)）
    seen_tool_call_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                seen_tool_call_ids.add(tc.get("id"))

    # 第二遍：构建修复后的 messages
    fixed = []
    pending_tool_calls = {}  # 正向孤儿的待补列表
    dropped_orphans = 0

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                pending_tool_calls[tc["id"]] = tc["function"]["name"]
            fixed.append(msg)
        elif msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id not in seen_tool_call_ids:
                # 反向孤儿：tool result 但前面没对应 tool_calls → 删掉
                dropped_orphans += 1
                continue
            pending_tool_calls.pop(call_id, None)
            fixed.append(msg)
        else:
            fixed.append(msg)

    if dropped_orphans:
        logger.warning(
            "_fix_tool_call_pairs: 删除 %d 条孤儿 tool result（无对应 tool_calls）",
            dropped_orphans,
        )

    # 如果有未配对的 tool_call，补一条 tool 结果
    # 上限保护：恶意/异常输入可能导致大量未配对 tool_call，全量补全会让 messages 暴涨。
    # 超过阈值截断 + 记日志，便于排查。
    MAX_PENDING_REPAIR = 100
    if pending_tool_calls:
        if len(pending_tool_calls) > MAX_PENDING_REPAIR:
            logger.warning(
                "_fix_tool_call_pairs: 未配对 tool_call 数 %d 超过上限 %d，仅补全前 %d 条",
                len(pending_tool_calls), MAX_PENDING_REPAIR, MAX_PENDING_REPAIR,
            )
            pending_tool_calls = dict(list(pending_tool_calls.items())[:MAX_PENDING_REPAIR])
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
        # 高频路径优化：content 多为 str，直接 len() 避免 str() 转换开销
        if isinstance(content, str):
            total_chars += len(content)
        else:
            # list/其他结构（多模态消息）降级处理
            total_chars += len(str(content))
        for tc in msg.get("tool_calls", []) or []:
            args = tc.get("function", {}).get("arguments", "")
            if isinstance(args, str):
                total_chars += len(args)
            else:
                total_chars += len(str(args))
    return total_chars // 3
