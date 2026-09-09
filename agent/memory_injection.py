"""检索式记忆注入。

记忆（AI 对用户/项目沉淀下来的事实条目，跨会话保留）不能中途塞进
system prompt——会让前缀缓存失效、成本翻倍。本文件的做法是：每一轮按
当前用户问题，用便宜的辅助模型（aux_llm）挑出最相关的几条记忆，拼成
一条"阅后即焚"的 user 消息注入（ephemeral：只在本次 API 请求出现，
不进 system prompt 也不进对话历史，缓存毫发无伤）。用户没配辅助
模型时降级回 snapshot 全量注入（见文件末尾的降级函数）。
"""
import logging
from contextvars import ContextVar
from typing import Optional

from agent.memory_retriever import retrieve_relevant

logger = logging.getLogger(__name__)

# 同一轮的去重缓存：只记住上一次 (query, 结果) 这一对（相当于容量为 1 的缓存）。
# 必须用 ContextVar 而不是模块级全局变量——同进程里并发的多个 agent
# （asyncio task / to_thread 里的子代理）会互相看到对方的缓存，串味；
# ContextVar 让各并发上下文各持一份副本，互不可见，
# 主循环自己在同一个 task 里顺序轮次，行为不变。
_last_query_var: ContextVar[Optional[str]] = ContextVar(
    "memory_injection_last_query", default=None,
)
_last_result_var: ContextVar[Optional[Optional[dict]]] = ContextVar(
    "memory_injection_last_result", default=None,
)

# 检索命中率遥测：4 个数看记忆系统是否在正常工作（旧版零统计，
# 「记忆有没有被注入」完全不可见）。每 20 次请求打一次 INFO 汇总。
_retrieval_stats = {"requests": 0, "llm_returned": 0, "resolved": 0, "injected": 0}


def reset_injection_cache() -> None:
    """清空当前上下文的同轮缓存（主要给测试用，避免用例间串味）。"""
    _last_query_var.set(None)
    _last_result_var.set(None)


def build_augmented_query(user_message: str, agent) -> str:
    """把裸用户消息增强成带上下文签名的检索 query（纯机械拼接，零 LLM）。

    长任务后期用户常说「继续/好/下一步」——光靠这几个字查记忆没有
    信号。把三类现成信号拼进 query：最近读的文件、进行中任务、最近
    一条 assistant 回复尾部。全部为空时原样返回（行为与未增强一致）。

    参数：
        user_message：当前用户消息原文
        agent：AIAgent 实例（读 _recent_read_files / conversation_history /
               codeagent_home）
    返回：增强后的 query（长度由下游 retrieve_relevant 的 query[:1000]
          统一截断，这里不另设上限）。
    """
    parts = []
    # 信号 1：最近读的文件（末尾 3 个）
    try:
        files = list(getattr(agent, "_recent_read_files", None) or [])[-3:]
        if files:
            parts.append("最近文件: " + ", ".join(str(f) for f in files))
    except Exception:
        pass
    # 信号 2：进行中任务（第 1 条；无 home 不碰全局 store）
    try:
        home = getattr(agent, "codeagent_home", None)
        if home:
            from agent.task_store import get_task_store
            in_progress = get_task_store(home).list_all(status="in_progress") or []
            if in_progress:
                t = in_progress[0]
                parts.append(f"进行中任务: {t.get('id', '')} {t.get('subject', '')}")
    except Exception:
        pass
    # 信号 3：最近一条有正文的 assistant 回复尾部 200 字
    try:
        history = list(getattr(agent, "conversation_history", None) or [])
        for m in reversed(history):
            if (isinstance(m, dict) and m.get("role") == "assistant"
                    and m.get("content") and not m.get("tool_calls")):
                tail = str(m["content"])[-200:].replace("\n", " ")
                parts.append(f"最近回复: {tail}")
                break
    except Exception:
        pass
    if not parts:
        return user_message
    return f"{user_message}\n\n[上下文签名]\n" + "\n".join(parts)


async def build_relevant_memories_message(
    *, query: str, memory_store, aux_llm_router, max_results: int = 5,
    active_tools=None, surfaced: set = None,
) -> Optional[dict]:
    """检索相关记忆并拼出一条 ephemeral 注入消息。返回 None 表示本轮不注入。

    检索式记忆注入的主入口，主循环每轮调用一次。

    参数：
    - query：当前用户消息（检索依据）
    - memory_store：记忆库（提供索引和按 ID 取条目）
    - aux_llm_router：辅助 LLM 路由（做检索挑选）
    - max_results：最多注入几条（默认 5）
    - active_tools：当前对话正在使用的工具名（反噪音——
      这些工具的"用法文档"类记忆不召回）
    - surfaced：调用方持有的"已注入记忆 ID"集合，一物两用：传给检索层
      做跨轮去重（已注入的不占名额），同时本轮新选中的 ID 也会收进去
      （调用方拿着它跨轮累积）

    返回：拼好的 ephemeral user 消息 dict；失败/无相关记忆返回 None
    （fail-open：任何异常都不注入，绝不影响主对话）。
    """
    if not query or not query.strip():
        return None
    if memory_store is None or aux_llm_router is None:
        return None
    # 同一个 query 刚检索过，直接复用上次结果（省一次 LLM 调用）
    if query == _last_query_var.get() and _last_result_var.get() is not None:
        return _last_result_var.get()

    try:
        # 索引带年龄标注（[age: Nd] + prompt 里"新记忆优先"规则），防召回过期信息
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
        # 口径：只统计真正发起过 retrieve_relevant 的请求——空索引早退、
        # 同轮缓存命中、检索前异常都不计（它们没花检索成本，计了会把
        # 「没记忆可查」混进「检索请求量」，命中率读数失真）
        _retrieval_stats["requests"] += 1
        _retrieval_stats["llm_returned"] += len(memory_ids or [])
        if _retrieval_stats["requests"] % 20 == 1 and _retrieval_stats["requests"] > 1:
            logger.info(
                "记忆检索遥测：%s（累计）",
                _retrieval_stats,
            )
        if not memory_ids:
            # 确定性兜底：LLM 空手/失败时关键词匹配顶上（防单点）
            from agent.memory_retriever import keyword_fallback_ids
            memory_ids = keyword_fallback_ids(
                query, index_text, max_results=max_results,
                exclude_ids=set(surfaced) if surfaced else None,
            )
            if memory_ids:
                logger.info(
                    "记忆检索走关键词兜底：%d 条", len(memory_ids),
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
            # ID 纠错：LLM 抄错一两位（大小写/截尾）时从索引里救回——
            # 旧版直接静默丢条，救不回才放弃
            try:
                from agent.memory_retriever import correct_memory_id
                fixed = correct_memory_id(mid, index_text)
            except Exception:
                fixed = None
            if fixed and fixed != mid:
                try:
                    entry = memory_store.get(fixed)
                except Exception:
                    entry = None
                if entry is not None:
                    logger.info("记忆 ID 纠错：%s → %s", mid, fixed)
                    mid = fixed
        if entry is None:
            continue
        _retrieval_stats["resolved"] += 1
        body = (getattr(entry, "body", "") or "")[:500]
        # 超过 1 天的记忆里 file:line
        # 引用很可能已过时——旧引用会让错误断言显得有凭有据，必须提示核对
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
    _retrieval_stats["injected"] += 1
    return msg
