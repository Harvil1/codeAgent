"""compact 工具:让 LLM 主动触发上下文压缩。

LLM 觉得对话太长、之前查的信息已经不需要细节时,调这个工具强制 L4 压缩。
压缩后:之前的对话被 LLM 总结成摘要,保留最近 N 条消息。

借鉴 learn-claude-code s08 的 compact 工具设计。
适合长任务场景(做 PPT、写代码、多轮调试)——LLM 自己管理 context,
不用等被动阈值。

⚠️ 调这个工具时不要同时调其他工具(压缩会丢弃大部分历史,
其他工具的结果可能变孤儿被自动清理)。
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


COMPACT_SCHEMA = {
    "name": "compact",
    "description": (
        "主动压缩对话历史。当觉得对话太长、之前查的信息已经不需要细节时调用。"
        "压缩后:之前的对话被总结成摘要,保留最近 10 条消息。"
        "**适合场景**:做了多轮查结构/读文件后准备动手写、长任务中段节省 context。"
        "**注意**:调这个工具时不要同时调其他工具。"
        "**partial 压缩**:可传 from_idx/up_to_idx 只压中段,保留头尾原文。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": (
                    "可选,告诉压缩引擎重点关注什么。"
                    "如 'PPT 结构已查清,保留封面/目录设计思路'。"
                    "不填则通用总结。"
                ),
            },
            "from_idx": {
                "type": "integer",
                "description": (
                    "可选,partial 压缩起始位置（从第 N 条消息开始压，默认 0=从头）。"
                    "配合 up_to_idx 使用可只压中段，保留头尾原文。"
                    "不填=全量压缩。"
                ),
            },
            "up_to_idx": {
                "type": "integer",
                "description": (
                    "可选,partial 压缩结束位置（压到第 N 条为止，默认 -1=压到末尾）。"
                    "配合 from_idx 使用可只压中段。"
                    "不填=全量压缩。"
                ),
            },
        },
    },
}


async def _handle_compact(args: dict, **kwargs) -> str:
    """主动触发 L4 上下文压缩（Task C: 支持 partial from_idx/up_to_idx）。

    通过 agent_ref 拿到 agent 实例,调 llm_compact 替换 conversation_history。
    主循环下一轮会自动用新的(已压缩)history 调 LLM。

    Task C: partial 模式（from_idx/up_to_idx 非默认值时启用）
    - 只压 conv[from_idx:up_to_idx] 段，保留 head + tail 原文
    - 适合"只压中段"场景（如早期查询已完成、中间段可丢弃细节）
    """
    agent = kwargs.get("agent_ref")
    if agent is None:
        return json.dumps(
            {"error": "compact 工具需要 agent_ref(运行时注入)"},
            ensure_ascii=False,
        )

    focus = args.get("focus", "") or ""
    from_idx = args.get("from_idx", 0)
    up_to_idx = args.get("up_to_idx", -1)
    is_partial = from_idx != 0 or up_to_idx != -1

    # 太短不值得压（partial 模式放宽到 2 条）
    history_len = len(agent.conversation_history)
    min_threshold = 2 if is_partial else 10
    if history_len < min_threshold:
        return json.dumps({
            "success": False,
            "reason": f"对话太短({history_len} 条 < {min_threshold}),不值得压缩",
        }, ensure_ascii=False)

    # 构造完整 messages(system + conv)
    try:
        system_prompt = agent._get_system_prompt()
    except Exception:
        system_prompt = ""
    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + list(agent.conversation_history)
    )

    # 强制 L4 压缩(阈值设 0 让它必触发,绕过 over_threshold 检查)
    from agent.context_pipeline import llm_compact
    keep_recent = 30  # 对齐 L4：主动 compact 后保留更多最近，减少失忆
    new_messages, changed = await llm_compact(
        full_messages,
        llm_client=agent.llm_client,
        model=getattr(agent, "model", None),
        keep_recent=keep_recent,
        token_threshold=0,   # 0 = 强制触发(任何 > 0 都超 0)
        msg_threshold=0,
        from_idx=from_idx,
        up_to_idx=up_to_idx,
    )

    if not changed:
        return json.dumps({
            "success": False,
            "reason": "压缩未生效(可能 keep_recent >= 对话长度或 partial 段 < 2)",
        }, ensure_ascii=False)

    # 替换 agent 状态:new_messages[0] 是 system,后面是 conv
    agent.conversation_history = new_messages[1:]
    # 让下轮重建 system prompt(虽然内容可能一样,但保险)
    try:
        agent.invalidate_system_prompt()
    except Exception:
        pass

    before_len = len(full_messages)
    after_len = len(new_messages)
    mode_desc = f"partial {from_idx}-{up_to_idx}" if is_partial else "full"
    logger.info(
        "compact 工具触发压缩(%s): messages %d → %d (focus=%r)",
        mode_desc, before_len, after_len, focus[:50],
    )

    return json.dumps({
        "success": True,
        "mode": mode_desc,
        "before_messages": before_len,
        "after_messages": after_len,
        "kept_recent": keep_recent if not is_partial else None,
        "from_idx": from_idx if is_partial else None,
        "up_to_idx": up_to_idx if is_partial else None,
        "message": (
            f"对话历史已{'部分' if is_partial else ''}压缩({before_len} → {after_len} 条, "
            f"mode={mode_desc})。"
            + ("之前的对话被 LLM 总结成摘要,保留了最近 "
               f"{keep_recent} 条消息。继续基于摘要 + 最近消息工作。"
               if not is_partial else
               "中段已压缩为摘要,头尾原文保留。")
            + (f" 关注点: {focus}" if focus else "")
        ),
    }, ensure_ascii=False)


# 模块级注册(import 时自动)
registry.register(
    name="compact",
    toolset="core",
    schema=COMPACT_SCHEMA,
    handler=_handle_compact,
    emoji="🗜️",
    isConcurrencySafe=False,  # 副作用：触发 LLM 压缩上下文（改消息历史），必须串行
)
