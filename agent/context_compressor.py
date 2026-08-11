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
    from_idx: int = 0,
    up_to_idx: int = -1,
) -> str:
    """调用 LLM 用 9 段式 prompt 总结对话历史（async：对齐 batch2 async 改造）。

    改造点 ②：
    - **9 段式结构化 prompt**：强制保留文件路径/错误消息/用户原话
    - **PTL 重试**：prompt_too_long 时丢 20% 旧消息重试（最多 MAX_PTL_RETRIES 次）
    - **熔断器**：连续 MAX_CONSECUTIVE_FAILURES 次失败后不再调 LLM
    - **session_memory 替代**：有预提取 memory 时直接用，不调 LLM

    Task C（partial compact）：
    - **from_idx/up_to_idx**：只压 messages[from_idx:up_to_idx] 段（默认 0/-1 = 全量）
    - 段消息数 < 2 时返回空串（不压）

    Args:
        messages: 对话历史
        llm_client: LLM 客户端（None 时走规则总结）
        model: 主模型名
        summary_model: 摘要专用模型（优先于 model）
        session_memory: 预提取的 session memory（有则替代 LLM 摘要）
        from_idx: 从第 N 条开始压（默认 0 = 从头）
        up_to_idx: 压到第 N 条为止（默认 -1 = 压到末尾）
    """
    global _consecutive_failures, _compact_circuit_open

    # Task C：partial 提取（from_idx/up_to_idx）
    is_partial = from_idx != 0 or up_to_idx != -1
    effective_up_to = len(messages) if up_to_idx < 0 else up_to_idx
    to_summarize = messages[from_idx:effective_up_to]
    if is_partial and len(to_summarize) < 2:
        logger.info(
            "partial compact: 段消息数 %d < 2，跳过压缩",
            len(to_summarize),
        )
        return ""

    # 1. 熔断器检查（连续失败达上限 → 直接走规则总结，不调 LLM）
    if _compact_circuit_open:
        logger.warning(
            "摘要熔断器开启（连续 %d 次失败），跳过 LLM 摘要",
            _consecutive_failures,
        )
        return _rule_based_summary(to_summarize)

    # 2. session memory 优先（有预提取就直接用，省一次 LLM 调用）
    if session_memory and session_memory.strip():
        logger.info("用 session memory 替代 LLM 摘要")
        return session_memory

    # 3. 无客户端时走规则总结
    if llm_client is None:
        return _rule_based_summary(to_summarize)

    # 4. 格式化对话 + 9 段式 prompt（用 to_summarize 而不是全量 messages）
    working_messages = list(to_summarize)  # 不污染入参（PTL 重试会修改）
    dialog = _format_dialog_for_summary(working_messages)
    prompt = SUMMARIZE_PROMPT_9SECTION.format(dialog=dialog)

    # 5. PTL 重试（最多 MAX_PTL_RETRIES 次）
    # Task E：用 tokenGap 精确算法替旧 20% 粗丢
    # 改造点 ② review fix：按 spec 伪代码传 system message（你是技术对话摘要助手）
    # + model 参数（summary_model 优先于 model）。OpenAICompatClient.chat_completions
    # 会 pop 掉 model kwarg 用 self.model（客户端构造时绑定），但 aux_llm_router /
    # 未来其他 client 实现可能用外部传入的 model，保持 spec 一致性。
    summary_system_prompt = "你是技术对话摘要助手。"
    effective_model = model or summary_model
    for retry in range(MAX_PTL_RETRIES + 1):
        try:
            response = await llm_client.chat_completions(
                [
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=effective_model,
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
                # Task E：精确算法替旧 20% 粗丢
                # _compute_ptl_drop_count 内部会 fallback 到 20% 当错误消息无法解析
                drop_count = _compute_ptl_drop_count(
                    working_messages, str(e),
                    model_max_tokens=_get_model_max_tokens(effective_model or ""),
                )
                old_drop = max(1, len(working_messages) // 5)
                working_messages = working_messages[drop_count:]
                dialog = _format_dialog_for_summary(working_messages)
                prompt = SUMMARIZE_PROMPT_9SECTION.format(dialog=dialog)
                logger.warning(
                    "PTL 重试 %d/%d：tokenGap 精确算法丢 %d 条（旧 20%% 会丢 %d 条）",
                    retry + 1, MAX_PTL_RETRIES, drop_count, old_drop,
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


# ---------------------------------------------------------------------------
# Task E：PTL tokenGap 精确算法
# ---------------------------------------------------------------------------


def _get_model_max_tokens(model_name: str) -> int:
    """根据模型名查 max_tokens（简化版查表）。

    覆盖常见 provider：
      - ``[1m]`` / ``1m`` 后缀 → 1M（Claude 扩展上下文）
      - ``v4`` / ``deepseek`` → 65536（DeepSeek）
      - ``claude-3-5`` / ``sonnet`` / ``haiku`` → 200000（Anthropic）
      - 默认 → 64000（OpenAI 常见值）
    """
    if not model_name:
        return 64000
    name = model_name.lower()
    if "[1m]" in name or "1m" in name:
        return 1_000_000
    if "v4" in name or "deepseek" in name:
        return 65536
    if "claude-3-5" in name or "sonnet" in name or "haiku" in name:
        return 200000
    return 64000


def _compute_ptl_drop_count(
    messages: list,
    error_msg: str,
    model_max_tokens: int = 64000,
    safety_margin: float = 0.85,
) -> int:
    """根据 PTL 错误和当前 messages 大小，计算精确该丢多少条。

    tokenGap 算法（借鉴 claude-code-main）：
      1. 从错误消息提取 token 上限和实际 token 数（如果可解析）
      2. 算预算（``model_max_tokens * safety_margin``）
      3. 算超了多少
      4. 把超的量换算成要丢的消息条数（按平均大小，多丢 10% 保险）

    **fallback**：错误消息无法解析时走旧 20% 算法（``max(1, len // 5)``），
    保证不比现状差。

    **保护**：至少留 2 条（``min(drop, len-2)``），不能丢光。
    """
    # fallback：错误消息无法解析（空 / 不含 token 数）走旧 20% 算法
    if not error_msg:
        return max(1, len(messages) // 5)

    err_lower = error_msg.lower()

    # 检测是否是 PTL 类错误（如果不是，走 fallback）
    is_ptl = (
        "prompt_too_long" in err_lower
        or "context_length" in err_lower
        or "too long" in err_lower
        or "maximum context" in err_lower
    )
    if not is_ptl:
        return max(1, len(messages) // 5)

    # 尝试提取 token 数——三种 provider 格式：
    # DeepSeek: "prompt_too_long: input 75000 > 65536"
    # Anthropic: "prompt is too long: 75000 > 64000"
    # OpenAI: "maximum context length is 65536 tokens, however you requested 75000"
    #
    # 策略：
    #   actual = "(\d+) >"  或  "requested (\d+)"
    #   limit  = "> (\d+)"  或  "max... is (\d+)" 或 "max...[:=]\s*(\d+)"
    actual_tokens = None
    limit_tokens = None

    # actual: "75000 >" 格式（DeepSeek / Anthropic）
    m = re.search(r"(\d{3,7})\s*>", error_msg)
    if m:
        actual_tokens = int(m.group(1))

    # actual: "requested 75000" 格式（OpenAI）
    if actual_tokens is None:
        m = re.search(r"requested\s+(\d{3,7})", err_lower)
        if m:
            actual_tokens = int(m.group(1))

    # limit: "> 65536" 格式（DeepSeek / Anthropic）
    m = re.search(r">\s*(\d{3,7})", error_msg)
    if m:
        limit_tokens = int(m.group(1))

    # limit: "is 65536 tokens" 或 "max... 65536" 格式（OpenAI）
    if limit_tokens is None:
        m = re.search(r"(?:is|max[a-z_]*)\s*[:=]?\s*(\d{3,7})", err_lower)
        if m:
            limit_tokens = int(m.group(1))

    # 无法提取 → fallback 到 20%
    if actual_tokens is None and limit_tokens is None:
        return max(1, len(messages) // 5)

    # 用提取到的值覆盖默认
    if limit_tokens:
        model_max_tokens = limit_tokens
    if actual_tokens is None:
        actual_tokens = estimate_message_tokens(messages)

    # 算预算（留 15% 给输出 + prompt overhead）
    budget = int(model_max_tokens * safety_margin)
    overflow = max(0, actual_tokens - budget)

    if overflow == 0:
        return 1  # 兜底丢 1 条（PTL 报错但算出来没超——可能是 margin 太严）

    # 算每条消息平均大小
    msg_count = max(1, len(messages))
    avg_tokens_per_msg = actual_tokens / msg_count

    # 算要丢多少条（多丢 10% 保险，避免连续 PTL）
    drop_count = int(overflow / avg_tokens_per_msg * 1.1) + 1

    # 保护：至少留 2 条
    return min(drop_count, max(1, len(messages) - 2))
