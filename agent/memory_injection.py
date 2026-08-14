"""检索式记忆注入（CCAR10，对标 CCB findRelevantMemories）。

每轮按用户 query 用 aux_llm 选 Top N 相关记忆，构造 ephemeral user
消息注入（不进 system prompt / history，保护 prompt cache）。
直接替代原 snapshot 全量索引注入（无 aux 时主循环降级回 snapshot）。
"""
import logging
from typing import Optional

from agent.memory_retriever import retrieve_relevant

logger = logging.getLogger(__name__)

# 同轮去重：上一个 (query, result) 缓存（LRU 1）
_last_query: Optional[str] = None
_last_result: Optional[Optional[dict]] = None


def reset_injection_cache() -> None:
    """测试用：清空同轮缓存。"""
    global _last_query, _last_result
    _last_query, _last_result = None, None


async def build_relevant_memories_message(
    *, query: str, memory_store, aux_llm_router, max_results: int = 5,
) -> Optional[dict]:
    """检索相关记忆并构造 ephemeral 注入消息。None = 不注入。fail-open。"""
    global _last_query, _last_result
    if not query or not query.strip():
        return None
    if memory_store is None or aux_llm_router is None:
        return None
    # 同轮去重（同 query 直接用上次结果）
    if query == _last_query and _last_result is not None:
        return _last_result

    try:
        index_text = memory_store.full_index_text()
        if not index_text or not index_text.strip():
            _last_query, _last_result = query, None
            return None
        memory_ids = await retrieve_relevant(
            query=query, index_text=index_text,
            llm_client=aux_llm_router, model=None,
            max_results=max_results,
        )
    except Exception as e:
        logger.warning("检索式记忆注入失败（fail-open 不注入）: %s", e)
        _last_query, _last_result = query, None
        return None

    lines = []
    for mid in memory_ids or []:
        try:
            entry = memory_store.get(mid)
        except Exception:
            entry = None
        if entry is None:
            continue
        body = (getattr(entry, "body", "") or "")[:500]
        lines.append(f"- [{entry.type}] {entry.name}: {body}")
    if not lines:
        _last_query, _last_result = query, None
        return None

    msg = {
        "role": "user",
        "content": (
            f'<relevant_memories count="{len(lines)}">\n'
            + "\n".join(lines)
            + "\n</relevant_memories>\n"
            "（以上是按当前问题检索的历史记忆，仅供参考）"
        ),
        "_ephemeral": True,
    }
    _last_query, _last_result = query, msg
    return msg
