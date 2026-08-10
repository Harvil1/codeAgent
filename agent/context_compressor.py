"""上下文压缩工具函数。

原 ``maybe_compress`` 单层 LLM 摘要路径已在 Phase 1 Commit 7 移除，
由 ``agent.context_pipeline.compress_if_needed``（4 层管线）取代。

本模块保留下列被 pipeline 复用的工具函数：
    - ``_summarize_conversation``：调用 LLM 总结对话（9 段式 + PTL 重试 + 熔断器）
    - ``_rule_based_summary``：无 LLM 时的降级规则提取
    - ``_fix_tool_call_pairs``：修复压缩边界破坏的 tool_call 配对
    - ``estimate_message_tokens``：粗略估算 token 数
    - ``_strip_analysis_draft``：剥离 LLM <analysis> 草稿区
    - ``reset_compact_circuit_breaker``：会话开始时重置熔断器
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 改造点 ②：9 段式结构化 prompt 常量
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT_9SECTION = """请把以下对话总结成 9 段结构化摘要。

**必须按以下 9 段输出，每段不能省略**：

1. **Primary Request and Intent**：用户的核心请求和意图
2. **Key Technical Concepts**：涉及的关键技术概念、库名、API
3. **Files and Code Sections**：涉及的文件路径（**逐字保留**）+ 关键代码段
4. **Errors and fixes**：遇到的错误（**逐字保留错误消息**）+ 修复方法
5. **Problem Solving**：问题解决过程、调试思路
6. **All user messages**：所有用户消息原文（**逐字保留**，不能改写）
7. **Pending Tasks**：待办任务、未完成的工作
8. **Current Work**：当前正在做什么
9. **Optional Next Step**：可选的下一步

**铁律**：
- 文件路径、命令、错误消息、用户原话必须**逐字保留**（不能省略/改写）
- 用 markdown 格式
- 每段不超过 200 字（除了 user messages 段保留原文）

对话内容：
{dialog}
"""


# ---------------------------------------------------------------------------
# 改造点 ②：熔断器模块级状态
# ---------------------------------------------------------------------------

_consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 3
_compact_circuit_open = False

# PTL（prompt_too_long）重试上限
MAX_PTL_RETRIES = 3


async def _summarize_conversation(
    messages: list,
    llm_client=None,
    *,
    model: str = None,
    summary_model: str = None,
    session_memory: str = None,
) -> str:
    """调用 LLM 用 9 段式 prompt 总结对话历史（async：对齐 batch2 async 改造）。

    改造点 ②：
    - **9 段式结构化 prompt**：强制保留文件路径/错误消息/用户原话
    - **PTL 重试**：prompt_too_long 时丢 20% 旧消息重试（最多 MAX_PTL_RETRIES 次）
    - **熔断器**：连续 MAX_CONSECUTIVE_FAILURES 次失败后不再调 LLM
    - **session_memory 替代**：有预提取 memory 时直接用，不调 LLM

    Args:
        messages: 对话历史
        llm_client: LLM 客户端（None 时走规则总结）
        model: 主模型名
        summary_model: 摘要专用模型（优先于 model）
        session_memory: 预提取的 session memory（有则替代 LLM 摘要）
    """
    global _consecutive_failures, _compact_circuit_open

    # 1. 熔断器检查（连续失败达上限 → 直接走规则总结，不调 LLM）
    if _compact_circuit_open:
        logger.warning(
            "摘要熔断器开启（连续 %d 次失败），跳过 LLM 摘要",
            _consecutive_failures,
        )
        return _rule_based_summary(messages)

    # 2. session memory 优先（有预提取就直接用，省一次 LLM 调用）
    if session_memory and session_memory.strip():
        logger.info("用 session memory 替代 LLM 摘要")
        return session_memory

    # 3. 无客户端时走规则总结
    if llm_client is None:
        return _rule_based_summary(messages)

    # 4. 格式化对话 + 9 段式 prompt
    working_messages = list(messages)  # 不污染入参（PTL 重试会修改）
    dialog = _format_dialog_for_summary(working_messages)
    prompt = SUMMARIZE_PROMPT_9SECTION.format(dialog=dialog)

    # 5. PTL 重试（最多 MAX_PTL_RETRIES 次，每次丢 20% 旧消息）
    for retry in range(MAX_PTL_RETRIES + 1):
        try:
            response = await llm_client.chat_completions(
                [{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content or ""
            # 成功：重置熔断器
            _consecutive_failures = 0
            _compact_circuit_open = False
            # 剥离 <analysis> 草稿（LLM 内部推理，不存入最终摘要）
            return _strip_analysis_draft(summary)
        except Exception as e:
            err_str = str(e).lower()
            is_ptl = "prompt_too_long" in err_str or "context_length" in err_str
            if is_ptl and retry < MAX_PTL_RETRIES:
                # PTL：丢 20% 旧消息重试
                drop_count = max(1, len(working_messages) // 5)
                working_messages = working_messages[drop_count:]
                dialog = _format_dialog_for_summary(working_messages)
                prompt = SUMMARIZE_PROMPT_9SECTION.format(dialog=dialog)
                logger.warning(
                    "PTL 重试 %d/%d：丢弃 %d 条旧消息",
                    retry + 1, MAX_PTL_RETRIES, drop_count,
                )
                continue
            # 其他错误或 PTL 重试耗尽 → 走规则总结 + 累加熔断器
            _consecutive_failures += 1
            if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _compact_circuit_open = True
                logger.error(
                    "摘要熔断器开启（连续 %d 次失败）",
                    _consecutive_failures,
                )
            logger.warning("LLM 摘要失败（降级规则总结）: %s", e)
            return _rule_based_summary(working_messages)

    return _rule_based_summary(working_messages)


def _format_dialog_for_summary(messages: list) -> str:
    """格式化消息列表为摘要 prompt 用的对话文本。

    从原 _summarize_conversation 的内联格式化逻辑抽取：
    - tool 消息：content 截断到 200 字符
    - assistant(tool_calls)：显示工具名列表
    - 其他：原样显示 role + content
    """
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
    return "\n\n".join(formatted)


def _strip_analysis_draft(summary: str) -> str:
    """剥离 <analysis> 草稿区（LLM 内部推理用，不存入最终摘要）。

    某些模型（如 Claude）可能在回复前加 <analysis>thinking...</analysis>
    做内部推理。这部分不是最终摘要内容，需要剥离。
    """
    return re.sub(r'<analysis>.*?</analysis>\s*', '', summary, flags=re.DOTALL)


def reset_compact_circuit_breaker() -> None:
    """会话开始时重置熔断器状态（避免跨会话污染）。

    在 AIAgent.__init__ 调用，保证新会话不从上一会话继承失败计数。
    """
    global _consecutive_failures, _compact_circuit_open
    _consecutive_failures = 0
    _compact_circuit_open = False


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

    处理两类协议违反：
    - 正向孤儿：assistant(tool_calls) 后没立刻是所有 id 的 tool_result
      → 在 result 序列末尾补假 tool_result（紧跟 assistant(tc) 后面，
        满足 Anthropic "tool_use ids found without tool_result blocks immediately after"）
    - 反向孤儿：tool 消息的 tool_call_id 不在前面任何 assistant(tool_calls) 里 → 删掉
      （压缩边界或上游 bug 可能产生这种孤儿，会让 API 报 400:
       "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"）

    ⚠️ Anthropic 严格要求 tool_use 后 immediately 跟所有 id 的 tool_result。
    补漏位置必须在 assistant(tc) 后的连续 result 序列末尾，**不能** append 到
    messages 末尾——否则中间隔了其他消息，仍违反 immediately after → 仍 400。
    """
    # 第一遍：原辅助（保留兼容性，但 X13 fix 后实际用 seen_so_far）
    # 收集全部 tool_call_ids 仍用于"已知 id 集合"，但反向孤儿检查用 seen_so_far
    seen_tool_call_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                seen_tool_call_ids.add(tc.get("id"))

    # X13 fix: 反向孤儿按"截至当前位置"判断（seen_so_far 逐步累加）。
    # 之前用全局 seen_tool_call_ids 漏判错序：tool(B) 在 assistant(tc B) 之前
    # 也通过检查（因为 B 在全局集合里），导致 API 400。
    seen_so_far = set()

    # 第二遍：单遍构建，pending 跟踪当前 assistant(tc) 缺的 result id。
    # 遇到非 tool 消息（user/system/新 assistant）时立刻把 pending 补完，
    # 保证假 result 紧跟在前一个 result 序列末尾（即 assistant(tc) 后面）。
    fixed = []
    pending_tool_calls = {}  # {id: name}：当前 assistant(tc) 待补 result 的 id
    dropped_orphans = 0
    MAX_PENDING_REPAIR = 100

    def _flush_pending():
        nonlocal pending_tool_calls
        if not pending_tool_calls:
            return
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
        pending_tool_calls = {}

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # 新 assistant(tc) → 前一个 result 序列结束，先补完旧 pending
            _flush_pending()
            for tc in msg["tool_calls"]:
                tid = tc.get("id")
                if tid:
                    pending_tool_calls[tid] = tc["function"]["name"]
                    # X13 fix: 累加到 seen_so_far（之后的位置才能引用这个 id）
                    seen_so_far.add(tid)
            fixed.append(msg)
        elif msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            # X13 fix: 反向孤儿按"截至当前位置"判断（seen_so_far），不用全局集合
            if call_id not in seen_so_far:
                # 反向孤儿：tool result 但前面没对应 tool_calls → 删掉
                dropped_orphans += 1
                continue
            pending_tool_calls.pop(call_id, None)
            fixed.append(msg)
        else:
            # 非工具消息：result 序列结束 → 立刻补完 pending
            _flush_pending()
            fixed.append(msg)

    # 末尾还可能剩 pending（assistant(tc) 是最后一条，后续没消息）
    _flush_pending()

    if dropped_orphans:
        logger.warning(
            "_fix_tool_call_pairs: 删除 %d 条孤儿 tool result（无对应 tool_calls）",
            dropped_orphans,
        )

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
