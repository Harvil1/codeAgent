"""memory_recall 工具：LLM 主动召回历史记忆。

与会话开始时注入 system prompt 的索引正交——按需查，不污染 prompt cache。
适用：LLM 中途想"用户上次提过 X 吗？"。

实现：调 retrieve_relevant（async，LLM 选 top-N）→ MemoryStore.get 拿正文。

为什么 async：
- retrieve_relevant 内部调 aux_llm_client.chat_completions，已经是 async
- handler 用 async def + await 才不会拿到 coroutine 误当字符串返回

不污染 snapshot_for_prompt（frozen 保护 prompt cache）：
- 只读 full_index_text + get，不写记忆
- snapshot_for_prompt 在本工具中绝不调用
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
    "inputSchema": {
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
    """async handler：retrieve_relevant 是 async。

    签名对齐 registry.dispatch 契约：dispatch(args, **dispatch_kwargs)。
    - 工具参数（query / top_k）从 args 取
    - 命名上下文从 dispatch_kwargs 取：
        memory_store: MemoryStore 实例（registry 直接透传）
        agent_ref:    AIAgent 实例，aux_llm_router 挂在其属性上
                      （对齐 agent/__init__.py:_auto_recall_memory 的用法：
                       llm_client=aux_llm_router, model=None——router 内部选模型）

    返回 JSON 字符串：
      成功 -> {"query": ..., "hits": [...], "count": N}
      失败 -> {"error": "...", "error_type": "not_configured" / 异常类名}

    注意：本工具绝不调用 snapshot_for_prompt——那是 frozen 的会话级快照，
    用来保护 prompt cache。这里只读 full_index_text（每次实时生成）+ get(id)。
    """
    # memory_store：dispatch_kwargs 直接透传
    memory_store = dispatch_kwargs.get("memory_store")
    if memory_store is None:
        return json.dumps(
            {"error": "memory_store not configured", "error_type": "not_configured"}
        )

    # aux_llm_client：从 agent_ref.aux_llm_router 取
    # （与 _auto_recall_memory 一致：router 内部自带模型选择，model=None）
    agent_ref = dispatch_kwargs.get("agent_ref")
    aux_client = getattr(agent_ref, "aux_llm_router", None) if agent_ref else None
    if aux_client is None:
        return json.dumps(
            {"error": "aux_llm_client not configured", "error_type": "not_configured"}
        )

    query = args["query"]
    # top_k 钳到 [1, 20]，防御 LLM 传野值
    top_k = max(1, min(20, args.get("top_k", 5)))

    try:
        # full_index_text 是实时生成的（_cached_snapshot 直接返回，无截断）
        index_text = memory_store.full_index_text()
        memory_ids = await retrieve_relevant(
            query=query,
            index_text=index_text,
            llm_client=aux_client,
            model=None,  # aux_llm_router 内部选模型（haiku/flash）
            max_results=top_k,
        )
    except Exception as e:
        logger.warning("memory_recall 失败: %s", e)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__}
        )

    # 拿正文：每个 id 查 MemoryStore.get（None 跳过）
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


# 模块顶部注册（import 即生效）
# toolset="memory"：和 memory_save / memory_update 等同族，不在 core
# is_async=True：handler 是 async def，对齐 registry 语义（虽然 dispatch 用 inspect 自动识别）
registry.register(
    name="memory_recall",
    schema=MEMORY_RECALL_SCHEMA,
    handler=_handle_memory_recall,
    toolset="memory",
    is_async=True,
    emoji="🔍",
    isConcurrencySafe=False,  # CCAR8 fix: 调 aux_llm（retrieve_relevant），消耗配额，串行更稳
)
