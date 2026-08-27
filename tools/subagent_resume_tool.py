"""subagent_resume 工具：让中断的子代理（主对话派出去帮忙干活的分身）从断点继续干活。

背景：子代理的对话记录会写到磁盘
（~/.OmniMate/.agent-sessions/<agent_id>.jsonl）；本工具是"取"的入口，
整体流程是：
- load_transcript 把历史消息读回来
- 用 initial_messages 重新启动一个 AIAgent 子代理
  （深度 = 父深度+1、只给 minimal 最小工具集、关掉摘要压缩）
- 继续往同一个 transcript 文件追加记录（append_message 复用同一 agent_id）
- 跑完后 mark_completed 标记完成

注意：恢复的只是"对话历史"（一条条 role/content 消息），不是内存里的
运行时状态——重建内存代价太高，对话记录已经够用了。

设计要点（历史踩坑，别丢）：
- **handler 签名必须是 (args, **dispatch_kwargs)**——注意：
  签名不符时 dispatch 静默不调用，代码成了摆设（silent-dead-code）
- **fail-open**：读文件/追加/标记完成任何一步失败都不崩，返回错误 JSON
- **重资源要串行**：要重启一个 AIAgent，所以 isConcurrencySafe=False
- **留了 _spawn_resumed_agent 这个接缝**：生产代码跑真子代理，测试时
  patch 掉它做隔离
"""
import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 接缝：真正构造子代理跑续命任务的地方（测试会 patch 掉它来隔离）
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
    """构造一个子代理，把续命任务跑完，返回最终回答。专门留的测试接缝。

    背景：生产实现参考 tools/delegate_tool.py:_run_child 的子代理构造套路——
    - spawn_depth+1（从 agent_ref 拿父深度再 +1，防止子代理套子代理无限递归）
    - 只给 minimal 最小工具集（跟 leaf 叶子子代理对齐，免得它乱派活）
    - 关掉 summary_only（resume 要的是完整结果，不是 300 字摘要）
    - 同步执行（用 asyncio.run 驱动 child.chat 这个 async 方法）
    - memory_store 透传（漏传的话续跑子代理
      用不上父对话的记忆库，等于失忆）

    参数：
        messages: 续跑用的历史消息（load_transcript 读出并清洗过的）
        instruction: 这次让它继续干活的指令
        agent_ref: 父 AIAgent 的引用（用来取 spawn_depth 和 LLM 配置兜底）
        base_url: LLM 服务地址，能传就透传
        api_key: LLM 的 API key，能传就透传
        auth_token: LLM 的认证 token，能传就透传
        model: 模型名，能传就透传
        model_format: 模型消息格式，能传就透传
        config: 配置 dict（LLM 配置兜底链会用到，参考 _run_child）
        memory_store: 父对话的记忆库（可选；None 时子代理自己建默认的）

    返回：
        子代理的最终响应文本
    """
    from agent import AIAgent

    # spawn_depth 取父深度 +1（防递归）；万一取不到就保守用 1
    parent_depth = getattr(agent_ref, "spawn_depth", 0) if agent_ref else 0
    child_spawn_depth = parent_depth + 1

    # LLM 配置的取值顺序：显式传参优先，缺了再从 config 兜底（跟 _run_child 同款链）
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

    # system prompt 明说这是续命场景，让它参考前面的历史接着干，别重来一遍
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
        memory_store=memory_store,  # 必须透传父记忆库（漏了续跑子代理等于失忆）
        initial_messages=messages,  # 用 initial_messages 机制带入历史
        on_response=None,  # resume 不再递归落盘（主入口已经统一 append 了）
    )

    # 同步跑（asyncio.run 桥接 async chat，跟 _run_child 同款）
    import asyncio
    # 交接说明：AIAgent.chat(first_msg) 是"发起对话"，会把 first_msg 作为
    # 新的 user 轮追加到 initial_messages 后面。所以 _run_resume 构造
    # messages 时故意不带末尾的 instruction，由 chat 追加，避免出现
    # 两条重复的 user 消息。
    result = asyncio.run(child.chat(instruction))
    return result


# ---------------------------------------------------------------------------
# 恢复入口（工具 handler 和 CLI 共用）
# ---------------------------------------------------------------------------

def _run_resume(agent_id: str, instruction: str, **dispatch_kwargs) -> str:
    """恢复一个中断的子代理：读历史 → 起子代理续跑 → 续写记录 → 标记完成。

    背景：这是工具 handler 和 CLI 共用的入口，逻辑集中在一处免得
    两边跑偏。流程：
    1. load_transcript(agent_id) 读出历史消息
    2. 组装续跑用的 messages = 历史 + 末尾的 user 指令
    3. _spawn_resumed_agent 起子代理跑完
    4. 继续写 transcript（append_message 用同一 agent_id）
    5. mark_completed 标记完成

    参数：
        agent_id: 要恢复的子代理 ID
        instruction: 这次续跑的指令
        **dispatch_kwargs: 工具 dispatch 透传进来的上下文
            （agent_ref / config / memory_store 等）

    返回：
        JSON 字符串：成功是 {"agent_id": ..., "result": ...}，
        失败是 {"error": ..., "error_type": ...}
    """
    from agent import subagent_persistence as sp

    # ① 读 transcript（fail-open：异常或空都返回错误 JSON，不崩）
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

    # ② 把 _ts 这类内部记账字段滤掉，只留 role/content/tool_calls 等对话语义字段
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

    # ③④ 起子代理续跑（接缝函数，测试可能 patch 掉）。
    # instruction 不塞进 initial_messages：_spawn_resumed_agent 内部用
    # chat(instruction) 触发对话，它会作为新 user 轮追加，塞了会重复。
    # 从 agent_ref 拿 memory_store 透传，
    # 让续跑子代理复用父记忆库
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

    # ⑤ 续写 transcript：这次的指令 + 结果，写进同一 agent_id 的 jsonl
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

    # ⑥ 标记完成
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
    """工具 handler：LLM 调 subagent_resume 时进这里，校验参数后转 _run_resume。

    背景：handler 签名必须严格是 (args: dict, **dispatch_kwargs) → JSON 字符串
    （签名不符时 dispatch 静默不调用，代码等于白写）。
    LLM 传的字段从 args 取，命名上下文（agent_ref / config 等）从
    dispatch_kwargs 取。
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
        "恢复一个中断的子代理继续执行。"
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
    isConcurrencySafe=False,  # 重资源（要重启一个 AIAgent），必须串行跑
)
