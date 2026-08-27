"""fork（分叉——复制一份现有会话去另开分支，不影响原会话）子代理的初始消息构造。

核心思想：
1. 父代理的 system prompt 一字不改地复用 → 得到 cache-identical（缓存一致——
   前缀字节完全相同，服务端的 prompt cache 才会命中）的开头
2. 父代理最近 N 轮 assistant 回复也原样复用，配上占位的 tool_result → 对话前缀也能共享缓存
3. 每个子代理自己的任务说明（directive）放在最后一条

为什么死磕"字节原样保留"：prompt cache 只有前缀字节一模一样才命中，
命中后这一大段就按缓存价计费，能省一半以上的 token 开销。
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def build_forked_messages(
    parent_messages: list,
    parent_system_prompt: str,
    child_directive: str,
    max_parent_turns: int = 3,
    full_history: bool = False,
    full_history_max_turns: int = 50,
) -> List[dict]:
    """给 fork 出来的子代理拼一份初始 messages（对话消息列表）。

    背景：fork 子代理要继承父代理的对话上下文来共享 prompt cache，
    但又不能让它看到父代理工具调用的真实结果（fork 语义：只继承"说过什么"，
    不继承"看到过什么"），所以工具结果统一换成占位符。

    默认结构（full_history=False）：
    [父最近 N 轮 assistant 回复 + 各自的占位 tool_result] + [child_directive]

    全量结构（full_history=True，T10 引入）：
    [父完整的 user/assistant 对话流（tool result 换占位符）] + [child_directive]
    —— 复杂任务需要完整上下文时用（调用方写 subagent fork: "full"），
    assistant 轮数超过 full_history_max_turns 会截断（防上下文失控），
    且截断起点回退到 user 消息边界（不让 assistant 消息开头变成没头没脑的孤儿）。

    两种模式都保证 cache-identical：继承自父的消息字节一字不动。

    参数：
        parent_messages：父代理的对话历史（conversation_history，不含 system prompt）
        parent_system_prompt：父代理的 system prompt（这个参数实际不进 messages，只留给日志用）
        child_directive：交给子代理的任务说明
        max_parent_turns：默认模式继承父代理最近 N 轮 assistant 回复（默认 3）
        full_history：True = 走全量模式（对应 subagent fork: "full"）
        full_history_max_turns：全量模式下 assistant 轮数上限（默认 50）

    返回：
        拼好的 messages 列表（不含 system prompt——system 走 system_prompt_override 单独传）。
    """
    forked = []

    if not parent_messages:
        # 父代理还没说过话，子代理只能拿到任务说明
        forked.append(_make_directive(child_directive))
        return forked

    # === T10：全量模式 ===
    if full_history:
        try:
            return _build_full_history_fork(
                parent_messages, child_directive, full_history_max_turns,
            )
        except Exception as e:
            logger.warning(
                "build_forked_messages 全量构造失败（fallback 最近 N turn）: %s", e,
            )

    # 挑出父代理的 assistant 轮（有正文内容的）
    try:
        parent_assistant_turns = [
            m for m in parent_messages
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
    except TypeError:
        # 历史踩坑防御：parent_messages 不是正常的 dict 列表时不硬撑，降级成只给任务说明
        logger.warning("build_forked_messages: parent_messages 类型异常，返回纯 directive")
        forked.append(_make_directive(child_directive))
        return forked

    # 只取最近 N 轮
    recent = parent_assistant_turns[-max_parent_turns:] if max_parent_turns > 0 else []

    for turn in recent:
        # 字节原样入列——改一个字都会让 prompt cache 失效（前缀对不上）
        forked.append(turn)
        # 这轮如果带了 tool_calls，必须补配对的占位 tool_result（不然消息序列不合法，API 会 400）
        tool_calls = turn.get("tool_calls") or []
        for tc in tool_calls:
            tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
            forked.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "[fork placeholder — 父代理的实际结果子代理看不到]",
            })

    # 最后垫上任务说明（子代理具体要干的事，作为最后一条 user 消息）
    forked.append(_make_directive(child_directive))

    return forked


def _build_full_history_fork(
    parent_messages: list,
    child_directive: str,
    max_turns: int,
) -> List[dict]:
    """全量 fork（T10 引入）：父代理完整 user/assistant 对话流照搬，工具真实结果换占位符。

    参数：
        parent_messages：父代理对话历史
        child_directive：子代理任务说明（垫在末尾）
        max_turns：assistant 消息数上限

    返回：
        拼好的 messages 列表。

    规则：
    - 只保留 user / assistant / tool 三种角色（别的跳过）
    - tool 消息（真实结果）一律替换成占位符——fork 语义就是真实结果不可见
    - assistant 消息超过 max_turns 时只留最近 max_turns 条，
      且截断起点回退到最近的 user 消息处（避免开头就是没头没脑的 assistant）
    """
    stream = []
    for m in parent_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            stream.append(m)
        elif role == "assistant":
            stream.append(m)
        elif role == "tool":
            stream.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": "[fork placeholder — 父代理的实际结果子代理看不到]",
            })

    asst_idx = [i for i, m in enumerate(stream) if m.get("role") == "assistant"]
    if len(asst_idx) > max_turns > 0:
        cut = asst_idx[-max_turns]
        start = 0
        for i in range(cut, -1, -1):
            if stream[i].get("role") == "user":
                start = i
                break
        stream = stream[start:]

    stream.append(_make_directive(child_directive))
    return stream


def _make_directive(child_directive: str) -> dict:
    """把任务说明包装成一条 user 消息（附上"你是 fork 出来的"身份说明）。

    参数：
        child_directive：任务说明文本

    返回：
        {"role": "user", "content": ...} 消息 dict。
    """
    return {
        "role": "user",
        "content": (
            "[FORK DIRECTIVE]\n"
            f"{child_directive}\n\n"
            "你是从父代理 fork 出来的子代理。前面的对话历史来自父代理"
            "（cache 共享），tool_result 是占位符（真实结果不可见）。"
            "请基于 directive 完成任务。"
        ),
    }


def build_forked_system_prompt(
    parent_system_prompt: str,
    child_role: str = "leaf",
) -> str:
    """构造 fork 子代理的 system prompt。

    做法：父 system prompt 一字不动 + 末尾追加一段 fork 说明。前缀字节保持一致，
    prompt cache 才能命中（这是 fork 模式省钱的根基）。

    参数：
        parent_system_prompt：父代理的完整 system prompt
        child_role：子代理角色——leaf（叶子，只干活不再派活）或 orchestrator（还能再派活）

    返回：
        父 system prompt 原文 + 追加的 fork 标记段。
    """
    fork_marker = (
        "\n\n## FORK MODE\n"
        "你是从父代理 fork 出来的子代理。前面的对话历史来自父代理（cache 共享）。\n"
        f"角色：{child_role}\n"
    )
    return parent_system_prompt + fork_marker
