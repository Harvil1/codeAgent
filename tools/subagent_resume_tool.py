"""subagent_resume 工具：恢复中断的子代理继续干（CCAR10 Task 4，补 CCAR5-I Phase 2）。

CCAR5-I 把子代理对话落盘（~/.OmniMate/.agent-sessions/<agent_id>.jsonl），
但只做了持久化半边；本 task 补恢复入口：
- load_transcript 读历史消息
- 用 initial_messages 重启 AIAgent 子代理（继承父深度+1、minimal 工具集、关闭摘要）
- 续写同一 transcript 文件（append_message 复用同一 agent_id）
- 完成后 mark_completed

恢复的是"对话历史"（role/content 消息），不是内部内存状态——
内存重建代价过高且 transcript 已经够用。

设计要点：
- **handler 签名 (args, **dispatch_kwargs)**（CCAR8 教训防 silent-dead-code）
- **fail-open**：load/append/mark 失败一律不崩，返错误 JSON
- **重资源串行**：spawn AIAgent 重启，isConcurrencySafe=False
- **接缝 _spawn_resumed_agent**：生产真跑子代理，测试 patch 隔离
"""
import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 接缝：构造子代理跑续命任务（被测试 patch 隔离）
# ---------------------------------------------------------------------------

def _spawn_resumed_agent(
    messages: list,
    instruction: str,
    *,
    agent_ref=None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    auth_token: Optional[str] = None,
    model: Optional[str] = None,
    model_format: Optional[str] = None,
    config: Optional[dict] = None,
    memory_store=None,
) -> str:
    """构造子代理跑续命任务。被测试 patch 的接缝。

    生产实现：参考 tools/delegate_tool.py:_run_child 的子代理构造模式——
    - spawn_depth+1（从 agent_ref 取父深度，防递归）
    - minimal 工具集（对齐 leaf 子代理，避免乱 spawn）
    - summary_only 关闭（resume 要完整结果，不要 300 字摘要）
    - 同步执行（asyncio.run 驱动 child.chat）
    - memory_store 透传（Task 5 follow-up：让续跑子代理复用父记忆库）

    Args:
        messages: 续跑历史（load_transcript 读出的 + 末尾追加的 user 指令）
        instruction: 本次续命指令
        agent_ref: 父 AIAgent 引用（取 spawn_depth + LLM 配置 fallback）
        base_url/api_key/auth_token/model/model_format: LLM 配置透传
        config: 配置 dict（参考 _run_child 的 LLM 配置 fallback 链）
        memory_store: 父记忆库（可选，None 则 child 自建默认 store）

    Returns:
        子代理最终响应文本
    """
    from agent import AIAgent

    # spawn_depth：父+1（防递归），fail-open 取不到就用 1
    parent_depth = getattr(agent_ref, "spawn_depth", 0) if agent_ref else 0
    child_spawn_depth = parent_depth + 1

    # LLM 配置：透传优先，其次从 config 取（对齐 _run_child fallback 链）
    if not (api_key or auth_token) or not model:
        try:
            from config import load_config
            _cfg = config or load_config()
            haiku_cfg = _cfg.get("haiku_model")
            sub_cfg = haiku_cfg or _cfg.get("model", {})
            base_url = base_url or sub_cfg.get("base_url")
            model = model or sub_cfg.get("model") or sub_cfg.get("name")
            if not api_key:
                api_key = sub_cfg.get("api_key") or ""
            if not auth_token:
                auth_token = sub_cfg.get("auth_token") or ""
            if not model_format:
                model_format = sub_cfg.get("format", "anthropic")
        except Exception as e:
            raise RuntimeError(f"subagent_resume 无法获取 LLM 配置: {e}")

    # system_prompt：明确告诉子代理这是续命场景，要参考前面的历史
    system_prompt = (
        "你是一个子代理，正在恢复执行之前未完成的任务。\n"
        "下面对话历史是你上次的工作记录，请基于它继续完成用户的新指令。\n"
        "约束：\n"
        "- 专注完成新指令，不重复已完成的工作\n"
        "- 完成后给出清晰的总结（关键发现、文件路径、命令、错误信息）\n"
        "- 不要做任务范围外的事\n"
    )

    child = AIAgent(
        base_url=base_url,
        api_key=api_key or None,
        auth_token=auth_token or None,
        model=model,
        model_format=model_format or "anthropic",
        max_iterations=50,
        enabled_toolsets=["minimal"],  # 最小工具集，对齐 leaf 子代理
        system_prompt_override=system_prompt,
        spawn_depth=child_spawn_depth,
        permission_mode="default",
        config=config,
        memory_store=memory_store,  # Task 5 follow-up：透传父记忆库
        initial_messages=messages,  # 复用 CCAR5 的 initial_messages 机制
        on_response=None,  # resume 不再递归落盘（主入口已 append）
    )

    # 同步跑（asyncio.run 驱动 async chat，跟 _run_child 同款桥接）
    import asyncio
    # 最后一条是续命指令（_run_resume 追加的 user 消息），
    # 用空 prompt 让 child 基于已有 history 继续跑
    # —— 但 AIAgent.chat(first_msg) 是发起对话，需要给一条触发消息
    # 这里用 instruction 触发（initial_messages 已含历史，这条 user 是新指令）
    # 注意：initial_messages 末尾已经是 {"role":"user","content":instruction}，
    # 所以 first_msg 给个最小确认即可（避免双重 user 消息）。
    # 实际上 AIAgent.chat 会把 first_msg 作为新 user turn 追加，所以
    # _run_resume 构造 messages 时不带末尾 instruction，由 chat 追加。
    result = asyncio.run(child.chat(instruction))
    return result


# ---------------------------------------------------------------------------
# 恢复入口（工具 + CLI 共用）
# ---------------------------------------------------------------------------

def _run_resume(agent_id: str, instruction: str, **dispatch_kwargs) -> str:
    """恢复入口（工具 handler 和 Task 5 CLI 共用）。

    流程：
    1. load_transcript(agent_id) → 历史消息
    2. 构造续跑 messages = 历史 + 末尾 user 指令
    3. _spawn_resumed_agent(messages, instruction) → 子代理跑完
    4. transcript 续写（append_message 用同一 agent_id）
    5. mark_completed

    Args:
        agent_id: 要恢复的子代理 ID
        instruction: 续跑指令
        **dispatch_kwargs: 工具 dispatch 透传的上下文（agent_ref / config 等）

    Returns:
        JSON 字符串：{"agent_id": ..., "result": ...} 或
        {"error": ..., "error_type": ...}
    """
    from agent import subagent_persistence as sp

    # ① 加载 transcript（fail-open：异常/空都返错误 JSON）
    try:
        messages = sp.load_transcript(agent_id)
    except Exception as e:
        logger.warning("subagent_resume: transcript 加载失败 [%s]: %s", agent_id, e)
        return json.dumps(
            {"error": f"transcript 加载失败: {e}",
             "error_type": "load_failed",
             "agent_id": agent_id},
            ensure_ascii=False,
        )

    if not messages:
        return json.dumps(
            {"error": f"agent {agent_id} 无历史消息（transcript 为空或不存在）",
             "error_type": "empty_transcript",
             "agent_id": agent_id},
            ensure_ascii=False,
        )

    # ② 过滤掉 _ts 等内部字段（只保留 role + content + tool_calls 等对话语义字段）
    clean_msgs = []
    for m in messages:
        clean = {k: v for k, v in m.items()
                 if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        if "role" in clean:
            clean_msgs.append(clean)

    if not clean_msgs:
        return json.dumps(
            {"error": f"agent {agent_id} transcript 含 0 条有效对话消息",
             "error_type": "empty_transcript",
             "agent_id": agent_id},
            ensure_ascii=False,
        )

    # ③ 续跑：把续命指令作为最后 user 消息追加到 initial_messages
    # _spawn_resumed_agent → AIAgent.chat(instruction) 会把它作为 first turn 发出，
    # 所以这里不在 initial_messages 里重复——_spawn_resumed_agent 内部用 chat(instruction)
    # 触发对话，initial_messages 是"之前的历史"
    # ④ 调 _spawn_resumed_agent（接缝，可能被测试 patch）
    # Task 5 follow-up：从 agent_ref 取 memory_store 透传给续跑子代理（复用父记忆库）
    _agent_ref = dispatch_kwargs.get("agent_ref")
    _memory_store = dispatch_kwargs.get("memory_store")
    if _memory_store is None and _agent_ref is not None:
        _memory_store = getattr(_agent_ref, "memory_store", None)
    try:
        result = _spawn_resumed_agent(
            clean_msgs,
            instruction,
            agent_ref=_agent_ref,
            base_url=dispatch_kwargs.get("base_url"),
            api_key=dispatch_kwargs.get("api_key"),
            auth_token=dispatch_kwargs.get("auth_token"),
            model=dispatch_kwargs.get("model"),
            model_format=dispatch_kwargs.get("model_format"),
            config=dispatch_kwargs.get("config"),
            memory_store=_memory_store,
        )
    except Exception as e:
        logger.warning("subagent_resume 跑失败 [%s]: %s", agent_id, e)
        return json.dumps(
            {"error": f"子代理恢复执行失败: {e}",
             "error_type": "resume_failed",
             "agent_id": agent_id},
            ensure_ascii=False,
        )

    # ⑤ transcript 续写（指令 + 结果），同一 agent_id 的 jsonl
    try:
        import time
        sp.append_message(agent_id, {
            "role": "user",
            "content": instruction,
            "_ts": time.time(),
        })
        sp.append_message(agent_id, {
            "role": "assistant",
            "content": result,
            "_ts": time.time(),
        })
    except Exception as e:
        logger.warning(
            "subagent_resume: transcript 续写失败（fail-open）[%s]: %s",
            agent_id, e,
        )

    # ⑥ mark_completed
    try:
        sp.mark_completed(agent_id, "completed")
    except Exception as e:
        logger.warning(
            "subagent_resume: mark_completed 失败（fail-open）[%s]: %s",
            agent_id, e,
        )

    return json.dumps(
        {"agent_id": agent_id, "result": result, "resumed": True},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 工具 handler（dispatch 契约：args, **kwargs）
# ---------------------------------------------------------------------------

def _handle_subagent_resume(args: dict, **dispatch_kwargs) -> str:
    """工具 handler：恢复中断的子代理。

    Handler 签名严格遵循 dispatch 契约（CCAR8 教训）：
    (args: dict, **dispatch_kwargs) → JSON 字符串

    Args 通过 args 取（LLM 传入的字段），命名上下文（agent_ref / config 等）
    通过 dispatch_kwargs 取。
    """
    agent_id = (args.get("agent_id") or "").strip()
    if not agent_id:
        return json.dumps(
            {"error": "agent_id 不能为空",
             "error_type": "invalid_args"},
            ensure_ascii=False,
        )
    instruction = args.get("instruction") or "继续完成剩余工作"
    return _run_resume(agent_id, instruction, **dispatch_kwargs)


# ---------------------------------------------------------------------------
# Schema + 注册
# ---------------------------------------------------------------------------

SUBAGENT_RESUME_SCHEMA = {
    "name": "subagent_resume",
    "description": (
        "恢复一个中断的子代理继续执行（CCAR10 Task 4）。"
        "transcript 已持久化到 ~/.OmniMate/.agent-sessions/<agent_id>.jsonl，"
        "传入 agent_id 加载历史对话并续跑，续写记录到同一文件。\n\n"
        "**适用场景**：\n"
        "- 上次子代理任务没跑完（interrupted / 进程重启）\n"
        "- 想基于之前的探索结果继续做下一步\n"
        "- 避免重复跑前置分析\n\n"
        "**不适用**：\n"
        "- agent_id 不存在（返 empty_transcript 错误）\n"
        "- transcript 为空\n"
        "- 想从某个具体 checkpoint 继续（本工具只支持从 transcript 末尾续）"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": (
                    "要恢复的子代理 ID（格式：sub-xxxxxxxx-YYYYMMDD-HHMMSS-xxxxxxxx）。"
                    "可用 list_resumable 或 /sessions 命令查。"
                ),
            },
            "instruction": {
                "type": "string",
                "description": (
                    "续跑指令（默认'继续完成剩余工作'）。"
                    "建议明确告诉子代理接下来要做什么，不要让它猜。"
                ),
            },
        },
        "required": ["agent_id"],
    },
}


registry.register(
    name="subagent_resume",
    schema=SUBAGENT_RESUME_SCHEMA,
    handler=_handle_subagent_resume,
    toolset="core",
    emoji="🔄",
    isConcurrencySafe=False,  # 重资源：spawn AIAgent 重启，必须串行
)
