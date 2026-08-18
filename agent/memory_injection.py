"""检索式记忆注入（CCAR10，对标 CCB findRelevantMemories）。

每轮按用户 query 用 aux_llm 选 Top N 相关记忆，构造 ephemeral user
消息注入（不进 system prompt / history，保护 prompt cache）。
直接替代原 snapshot 全量索引注入（无 aux 时主循环降级回 snapshot）。
"""
import logging
from contextvars import ContextVar
from typing import Optional

from agent.memory_retriever import retrieve_relevant

logger = logging.getLogger(__name__)

# 同轮去重：上一个 (query, result) 缓存（LRU 1）
# R30c-C1：模块级可变全局改 ContextVar——同进程并发 agent（asyncio task /
# to_thread 子代理各持 context 副本）互相看不到对方的缓存，消除串味；
# 主循环同 task 内顺序轮次语义不变。
_last_query_var: ContextVar[Optional[str]] = ContextVar(
    "memory_injection_last_query", default=None,
)
_last_result_var: ContextVar[Optional[Optional[dict]]] = ContextVar(
    "memory_injection_last_result", default=None,
)


def reset_injection_cache() -> None:
    """测试用：清空当前 context 的同轮缓存。"""
    _last_query_var.set(None)
    _last_result_var.set(None)


async def build_relevant_memories_message(
    *, query: str, memory_store, aux_llm_router, max_results: int = 5,
    active_tools=None, surfaced: set = None,
) -> Optional[dict]:
    """检索相关记忆并构造 ephemeral 注入消息。None = 不注入。fail-open。

    R30f-H9：
      - active_tools：当前对话正在使用的工具名（反噪音——用法文档类不召回）
      - surfaced：调用方持有的"已注入记忆 id"集合。双重作用：作为 exclude
        传入检索（跨轮去重，已注入的不占槽位），并把本轮新选中的 id 收进去
        （调用方跨轮持有）。
    """
    if not query or not query.strip():
        return None
    if memory_store is None or aux_llm_router is None:
        return None
    # 同轮去重（同 query 直接用上次结果）
    if query == _last_query_var.get() and _last_result_var.get() is not None:
        return _last_result_var.get()

    try:
        # T4：带年龄标注（[age: Nd] + prompt 新记忆优先规则，防召回过期记忆）
        index_text = memory_store.full_index_text_with_age()
        if not index_text or not index_text.strip():
            _last_query_var.set(query)
            _last_result_var.set(None)
            return None
        memory_ids = await retrieve_relevant(
            query=query, index_text=index_text,
            llm_client=aux_llm_router, model=None,
            max_results=max_results,
            active_tools=list(active_tools) if active_tools else None,
            exclude_ids=set(surfaced) if surfaced else None,
        )
    except Exception as e:
        logger.warning("检索式记忆注入失败（fail-open 不注入）: %s", e)
        _last_query_var.set(query)
        _last_result_var.set(None)
        return None

    from datetime import datetime, timezone as _tz
    now = datetime.now(_tz.utc)
    lines = []
    selected_ids = []
    for mid in memory_ids or []:
        try:
            entry = memory_store.get(mid)
        except Exception:
            entry = None
        if entry is None:
            continue
        body = (getattr(entry, "body", "") or "")[:500]
        # R30f-H9：过期警示（对齐 CCB staleness caveat）——老记忆里的
        # file:line 引用会让错误断言显得权威，>1 天的标注"须核对"
        stale_note = ""
        updated = getattr(entry, "updated_at", None)
        if isinstance(updated, datetime) and updated.tzinfo is not None:
            days = (now - updated).days
            if days > 1:
                stale_note = f" [age: {days}d 引用可能已过期，使用前核对当前代码]"
        lines.append(f"- [{entry.type}] {entry.name}: {body}{stale_note}")
        selected_ids.append(mid)
    if not lines:
        _last_query_var.set(query)
        _last_result_var.set(None)
        return None

    if surfaced is not None:
        try:
            surfaced.update(selected_ids)
        except Exception:
            pass

    msg = {
        "role": "user",
        "content": (
            f'<relevant_memories count="{len(lines)}">\n'
            + "\n".join(lines)
            + "\n</relevant_memories>\n"
            "（以上是按当前问题检索的历史记忆，仅供参考；带 age 标注的引用可能过期）"
        ),
        "_ephemeral": True,
    }
    _last_query_var.set(query)
    _last_result_var.set(msg)
    return msg


def _fallback_snapshot_message(memory_store) -> Optional[dict]:
    """无 aux_llm_router 时的降级：退回 snapshot 索引注入。

    直接替代决策的保底链——用户没配 aux 模型时记忆功能不丢。
    返回 ephemeral user 消息（同轮注入后即弃）。
    fail-open：任何异常返回 None。
    """
    try:
        snap = memory_store.snapshot_for_prompt()
    except Exception as e:
        logger.warning("snapshot 降级注入失败: %s", e)
        return None
    if not snap or not snap.strip():
        return None
    return {
        "role": "user",
        "content": (
            "<memory_index>\n" + snap + "\n</memory_index>\n"
            "（以上是记忆索引（降级模式），供参考）"
        ),
        "_ephemeral": True,
    }
