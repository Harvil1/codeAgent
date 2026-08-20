"""memory_recall 工具：让 LLM 在对话中途主动去翻历史记忆。

背景：会话开始时会把记忆索引塞进 system prompt（系统提示词），但那是一次性的。
如果 LLM 中途突然想「用户上次是不是提过 X？」，就需要一个能随时查的工具——
这就是本工具。它是按需查询，不会改动 system prompt，所以不会破坏 prompt cache
（提示词前缀缓存——缓存一失效，同样内容的请求就要重新算一遍，费用翻倍）。

实现路径：调 agent/memory_retriever.py 的 retrieve_relevant（它内部用辅助小模型
挑出最相关的 N 条）→ 再用 MemoryStore.get 拿到每条的正文。

为什么必须写成 async（异步函数）：
- retrieve_relevant 内部要调辅助 LLM，本来就是 async
- handler（工具处理函数）如果写成普通函数又去拿它的结果，拿到的是个
  coroutine 对象（还没跑完的"欠条"），会被误当成字符串返回给 LLM，必错

为了不污染 prompt cache，本工具只做读操作：
- 只读 full_index_text（实时生成的索引全文）+ get(id)，绝不写记忆
- 绝不调用 snapshot_for_prompt——那是会话开始时冻结的快照，动它就毁缓存
"""
import json
import logging

from agent.memory_retriever import retrieve_relevant
from tools.registry import registry

logger = logging.getLogger(__name__)


MEMORY_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "主动召回与 query 相关的历史记忆。"
        "用于你中途想查'用户上次提过 X 吗'的场景。"
        "与会话开始时注入的索引正交——按需查，不污染 system prompt。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "查询关键词",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}


async def _handle_memory_recall(args: dict, **dispatch_kwargs) -> str:
    """memory_recall 的处理函数：查索引选记忆，再逐条取正文拼结果。

    背景：retrieve_relevant 是 async（异步），所以本 handler 也必须 async，
    否则拿到的是 coroutine「欠条」而不是真结果。

    签名要遵守 registry.dispatch 的统一契约：dispatch(args, **dispatch_kwargs)。
    - 工具参数（query / top_k）从 args 拿
    - 命名上下文从 dispatch_kwargs 拿：
        memory_store: 记忆仓库实例（registry 直接透传过来）
        agent_ref:    AIAgent 主对象，辅助模型路由器 aux_llm_router 挂在它属性上
                      （和 agent/__init__.py 里 _auto_recall_memory 的用法一致：
                       只传 llm_client=aux_llm_router、model 留空——由路由器自己挑模型）

    参数：
        args: LLM 传来的参数——query（查询关键词）和 top_k（最多返回几条）
        dispatch_kwargs: 命名上下文，用 memory_store 和 agent_ref 两项

    返回：JSON 字符串。
      成功 -> {"query": ..., "hits": [...], "count": N}
      失败 -> {"error": "...", "error_type": "not_configured" 或异常类名}

    注意：本工具绝不调用 snapshot_for_prompt——那是会话开始时冻结的快照，
    用来保护 prompt cache。这里只读 full_index_text（每次实时生成）+ get(id)。
    """
    # memory_store 由 dispatch_kwargs 直接透传
    memory_store = dispatch_kwargs.get("memory_store")
    if memory_store is None:
        return json.dumps(
            {"error": "memory_store not configured", "error_type": "not_configured"}
        )

    # 辅助 LLM 客户端从 agent_ref.aux_llm_router 上取
    # （和 _auto_recall_memory 的做法一致：路由器自己挑模型，所以 model 传 None）
    agent_ref = dispatch_kwargs.get("agent_ref")
    aux_client = getattr(agent_ref, "aux_llm_router", None) if agent_ref else None
    if aux_client is None:
        return json.dumps(
            {"error": "aux_llm_client not configured", "error_type": "not_configured"}
        )

    query = args["query"]
    # top_k 强制夹在 [1, 20] 之间，防 LLM 传来离谱的数字
    top_k = max(1, min(20, args.get("top_k", 5)))

    try:
        # full_index_text 是现场实时生成的（不走缓存快照，也不截断）；
        # 用带年龄标注的版本（每条标 [age: Nd]，并在提示里让模型优先看新记忆）——T4 轮引入
        index_text = memory_store.full_index_text_with_age()
        memory_ids = await retrieve_relevant(
            query=query,
            index_text=index_text,
            llm_client=aux_client,
            model=None,  # 模型交给辅助路由器自己挑（轻量档，如 haiku/flash 这类便宜快的）
            max_results=top_k,
        )
    except Exception as e:
        logger.warning("memory_recall 失败: %s", e)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__}
        )

    # 拿正文：逐个 id 去记忆仓库查，查不到（None）的直接跳过
    hits = []
    for mid in memory_ids:
        entry = memory_store.get(mid)
        if entry is None:
            continue
        hits.append(
            {
                "id": entry.id,
                "name": entry.name,
                "description": entry.description,
                "summary": entry.summary,
                "body": entry.body,
            }
        )

    return json.dumps(
        {"query": query, "hits": hits, "count": len(hits)}, ensure_ascii=False
    )


# 在模块顶部注册（这个文件一被 import，工具就自动登记生效）
# toolset="memory" 只是记忆族的分组标签；LLM 到底能不能看见它，由 toolsets._CORE_TOOLS 决定
# （历史踩坑：「登记了」不等于「可见」——早期 toolset="memory" 没列进任何 TOOLSETS 定义，
# 结果工具注册了却对 LLM 隐身；现已列入 _CORE_TOOLS，对齐 Claude Code 里
# LocalMemoryRecallTool 属于核心工具的定位）
# is_async=True：handler 是 async 函数，显式标注对齐 registry 语义（虽然 dispatch
# 实际会用 inspect 自动识别，标上更保险）
registry.register(
    name="memory_recall",
    schema=MEMORY_RECALL_SCHEMA,
    handler=_handle_memory_recall,
    toolset="memory",
    is_async=True,
    emoji="🔍",
    isConcurrencySafe=False,  # 历史踩坑（CCAR8 修复）：会调辅助 LLM 消耗配额，串行跑更稳，避免并发狂烧
)
