"""「压缩对话」工具：让 AI 自己判断"旧对话不需要细节了"并主动压缩——
不等系统自动触发，把旧对话交给一次 LLM 摘要总结，只保留最近 N 条原文。
（本文件属于工具层，被 tools/registry.py 自动发现注册，底层压缩复用
agent/context_pipeline.py 的 llm_compact。）

⚠️ 调它时别同时调其他工具：压缩会扔掉大部分历史，同批其他工具的
结果消息会变成"没有对应的提问"的孤儿，被自动清理掉。
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
    """把（可选指定范围的）旧对话压缩成 LLM 摘要，返回结果 JSON。

    支持两种模式：全量压缩（旧对话全部变摘要，保留最近 30 条）和
    部分压缩（只压指定下标之间的中段，开头结尾保留原文）。

    参数：
    - args：工具参数字典。focus 是可选的"摘要时重点保什么"提示；
      from_idx/up_to_idx 可选，指定只压第 from_idx 条到第 up_to_idx 条
      （默认 0/-1 表示全量）。
    - kwargs：运行时注入的命名上下文，本函数只用到 agent_ref
      （AIAgent 主实例，借它拿对话历史、模型客户端和配置）。

    返回：JSON 字符串。success=True 带压缩前后条数；太短/没生效/出错
    时 success=False 或 error 字段。
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

    # 太短的对话压了没意义；只压中段时门槛放宽到 2 条。
    # 全量门槛 = keep_recent + 1：引擎全量分支要求 len(conv) > keep_recent
    # 才肯压，门口就用同一把尺（10~30 条的旧门槛放进去了也是必然被
    # 引擎拒绝，白跑一趟还报个含糊的"压缩未生效"）
    keep_recent = 30  # 比系统自动压缩保留更多近期消息，主动压缩后少失忆
    history_len = len(agent.conversation_history)
    min_threshold = 2 if is_partial else keep_recent + 1
    if history_len < min_threshold:
        return json.dumps({
            "success": False,
            "reason": f"对话太短({history_len} 条 < {min_threshold}),不值得压缩",
        }, ensure_ascii=False)

    # 压缩引擎要看到完整对话（系统提示 + 对话本体），这里拼一下
    try:
        system_prompt = agent._get_system_prompt()
    except Exception:
        system_prompt = ""
    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + list(agent.conversation_history)
    )

    # 压缩引擎自带"超过阈值才压"的检查，这里把阈值设成 0，
    # 相当于"我让你压你就必须压"，跳过它的犹豫。
    from agent.context_pipeline import llm_compact
    new_messages, changed = await llm_compact(
        full_messages,
        llm_client=agent.llm_client,
        model=getattr(agent, "model", None),
        keep_recent=keep_recent,
        token_threshold=0,   # 0 = 见上，逼它无条件触发
        from_idx=from_idx,
        up_to_idx=up_to_idx,
        focus_hint=focus,  # schema 承诺的"重点保什么"真接进摘要 prompt
    )

    if not changed:
        # 三个拒绝原因都列出来：keep_recent >= 对话长度（全量）/ partial 段 < 2 /
        # 摘要不小于被替换段（L4 收敛检查不过，见 context_pipeline._summary_shrinks）。
        # 这是给 LLM 的自诊断信号——列不全它会拿旧解释瞎调参重试，白烧摘要调用。
        return json.dumps({
            "success": False,
            "reason": (
                "压缩未生效(可能 keep_recent >= 对话长度/partial 段 < 2/"
                "摘要不小于被替换段)"
            ),
        }, ensure_ascii=False)

    # 换上新历史：返回的第一条是系统提示，后面才是对话本体
    agent.conversation_history = new_messages[1:]
    # 让下轮重建 system prompt（内容大概率没变，但保险起见走一遍失效流程）
    try:
        agent.invalidate_system_prompt()
    except Exception:
        logger.warning("异常被吞(fail-open)", exc_info=True)

    # 冷却记账（与主循环 L4 成功后的收尾对齐）：不记账的话，紧接着的
    # 自动压缩看不到"刚压过"，冷却期判定失真、可能连着再压一次
    _state = getattr(agent, "_compress_session_state", None)
    if _state is not None:
        try:
            _state.record_llm_compact()
        except Exception as e:
            logger.warning("compact 工具冷却记账失败（不阻塞）: %s", e)

    # 边界占位落库（与主循环/紧急压缩的收尾一致）：不落 [COMPACT_BOUNDARY]
    # 的话，压完重启 = 会话库没有裁剪锚点，恢复全量载入旧历史，白压了。
    # agent_ref 身上有 session_store/session_id（主循环同款取法），fail-open。
    try:
        from agent.context_pipeline import (
            _persist_compact_marker, take_last_compact_placeholder,
        )
        _ph = take_last_compact_placeholder()
        if _ph:
            _persist_compact_marker(
                getattr(agent, "session_store", None),
                getattr(agent, "session_id", None) or "",
                _ph,
            )
    except Exception as e:
        logger.warning("compact 工具边界落库失败（fail-open）: %s", e)

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


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="compact",
    toolset="core",
    schema=COMPACT_SCHEMA,
    handler=_handle_compact,
    emoji="🗜️",
    isConcurrencySafe=False,  # 会重写整个对话历史，并发跑会互相踩，只能排队执行
)
