"""fork 子代理消息构造（Task H / CCAR5）。

核心思想（借鉴 Claude Code forkSubagent.ts）：
1. 复用父 system prompt 字节 → cache-identical 前缀
2. 复用父最近 N 个 assistant turn + placeholder tool_result → 对话前缀共享
3. per-child directive（任务说明）放在末尾

cache-identical 保证：父 prompt 字节 + 父 assistant turn 字节必须原样保留，
prompt cache 才会命中，省 token 50%+。
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
    """构造 fork 子代理的初始 messages。

    结构（full_history=False，默认，现状语义）：
    [父最近 N 个 assistant turn + placeholder tool_result] + [child_directive]

    结构（full_history=True，T10）：
    [父完整 user/assistant 流（tool result 换 placeholder）] + [child_directive]
    —— 复杂任务需要完整上下文时用（subagent fork: "full"），
    assistant turn 截到 full_history_max_turns（防失控），截断从 user 边界起。

    保证 cache-identical：父前缀字节完全一致。

    参数：
        parent_messages: 父 agent 的 conversation_history（不含 system prompt）
        parent_system_prompt: 父的 system prompt（保留参数用于日志，实际不影响 messages）
        child_directive: 子代理的任务说明
        max_parent_turns: 继承父最近 N 个 assistant turn（默认 3）
        full_history: True = 全量模式（subagent fork: "full"）
        full_history_max_turns: 全量模式的 assistant turn 上限（默认 50）

    返回：
        forked messages list（不含 system prompt，system 走 system_prompt_override）
    """
    forked = []

    if not parent_messages:
        # 父历史为空，forked 只有 directive
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

    # 提取父 assistant turn（有 content 的）
    try:
        parent_assistant_turns = [
            m for m in parent_messages
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
    except TypeError:
        # parent_messages 不是 iterable of dict，fail-safe
        logger.warning("build_forked_messages: parent_messages 类型异常，返回纯 directive")
        forked.append(_make_directive(child_directive))
        return forked

    # 取最近 N 个
    recent = parent_assistant_turns[-max_parent_turns:] if max_parent_turns > 0 else []

    for turn in recent:
        # 父 assistant turn 字节原样保留（cache-identical 前提）
        forked.append(turn)
        # 如果 turn 含 tool_calls，补 placeholder tool_result
        tool_calls = turn.get("tool_calls") or []
        for tc in tool_calls:
            tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
            forked.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "[fork placeholder — 父代理的实际结果子代理看不到]",
            })

    # 加 child directive（具体任务，最后一条 user 消息）
    forked.append(_make_directive(child_directive))

    return forked


def _build_full_history_fork(
    parent_messages: list,
    child_directive: str,
    max_turns: int,
) -> List[dict]:
    """全量 fork（T10）：完整 user/assistant 流，tool result 换 placeholder。

    - 只保留 user / assistant / tool 三类消息（其他 role 跳过）
    - tool 消息（真实结果）替换为 fork placeholder（fork 语义：真实结果不可见）
    - assistant 消息数超过 max_turns 时截到最近 max_turns 个，
      截断起点回退到最近的 user 边界（不造 assistant 孤儿开头）
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
    """构造 per-child directive 消息。"""
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

    保持父 system prompt 字节（cache-identical）+ 追加 fork 标记。

    参数：
        parent_system_prompt: 父 agent 的完整 system prompt
        child_role: 子代理角色（leaf / orchestrator）

    返回：
        父 system prompt 字节 + fork marker（cache-identical 前缀保证）
    """
    fork_marker = (
        "\n\n## FORK MODE\n"
        "你是从父代理 fork 出来的子代理。前面的对话历史来自父代理（cache 共享）。\n"
        f"角色：{child_role}\n"
    )
    return parent_system_prompt + fork_marker
