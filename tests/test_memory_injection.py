"""CCAR10 Task 1：检索式记忆注入消息构造的单元测试。"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.memory_injection import build_relevant_memories_message


@pytest.fixture(autouse=True)
def _reset_cache():
    """每个测试前清同轮缓存，避免模块级 LRU 污染。"""
    from agent.memory_injection import reset_injection_cache
    reset_injection_cache()
    yield
    reset_injection_cache()


def _make_store(entries: dict):
    """entries: {memory_id: (name, type, body)}"""
    store = MagicMock()
    store.full_index_text.return_value = "idx"

    def _get(mid):
        if mid not in entries:
            return None
        name, mtype, body = entries[mid]
        e = MagicMock()
        e.name, e.type, e.body = name, mtype, body
        e.description = name
        return e

    store.get.side_effect = _get
    return store


def test_builds_message_with_hits():
    store = _make_store({
        "general#a": ("偏好A", "user", "用户偏好正文"),
        "general#b": ("事实B", "project", "项目事实正文"),
    })
    aux = MagicMock()
    with patch(
        "agent.memory_injection.retrieve_relevant",
        new=AsyncMock(return_value=["general#a", "general#b"]),
    ):
        msg = asyncio.run(build_relevant_memories_message(
            query="用户问什么", memory_store=store, aux_llm_router=aux,
        ))
    assert msg is not None
    assert msg["role"] == "user"
    assert msg["_ephemeral"] is True
    assert "<relevant_memories" in msg["content"]
    assert "偏好A" in msg["content"] and "事实B" in msg["content"]


def test_body_truncated_to_500():
    store = _make_store({"general#a": ("n", "user", "x" * 2000)})
    with patch(
        "agent.memory_injection.retrieve_relevant",
        new=AsyncMock(return_value=["general#a"]),
    ):
        msg = asyncio.run(build_relevant_memories_message(
            query="q", memory_store=store, aux_llm_router=MagicMock(),
        ))
    assert msg["content"].count("x") <= 510  # 截到 ~500


def test_empty_query_returns_none():
    msg = asyncio.run(build_relevant_memories_message(
        query="", memory_store=MagicMock(), aux_llm_router=MagicMock(),
    ))
    assert msg is None


def test_no_hits_returns_none():
    store = _make_store({})
    with patch(
        "agent.memory_injection.retrieve_relevant",
        new=AsyncMock(return_value=[]),
    ):
        msg = asyncio.run(build_relevant_memories_message(
            query="q", memory_store=store, aux_llm_router=MagicMock(),
        ))
    assert msg is None


def test_retrieval_failure_failopen():
    with patch(
        "agent.memory_injection.retrieve_relevant",
        new=AsyncMock(side_effect=RuntimeError("aux down")),
    ):
        msg = asyncio.run(build_relevant_memories_message(
            query="q", memory_store=MagicMock(), aux_llm_router=MagicMock(),
        ))
    assert msg is None  # fail-open 不崩


def test_same_query_cached_within_round():
    """同 query 第二次调用不再触发检索（一轮缓存）。

    R30c-C1：缓存改 ContextVar 后，两次调用须在同一 task/context 内
    （对齐生产形态——同一轮内）；asyncio.run 各自拷 context 测不到缓存。
    """
    store = _make_store({"general#a": ("n", "user", "b")})
    mock_rr = AsyncMock(return_value=["general#a"])

    async def _two_calls():
        await build_relevant_memories_message(
            query="same", memory_store=store, aux_llm_router=MagicMock())
        await build_relevant_memories_message(
            query="same", memory_store=store, aux_llm_router=MagicMock())

    with patch("agent.memory_injection.retrieve_relevant", new=mock_rr):
        asyncio.run(_two_calls())
    assert mock_rr.await_count == 1


# ============================================================================
# CCAR10 Task 2: snapshot 退役 + 降级链
# ============================================================================


def test_snapshot_removed_from_system_prompt(tmp_path):
    """system prompt 不再含记忆索引段（直接替代决策）。

    snapshot 改走 ephemeral 注入（_pending_ephemeral_messages），
    不进 system prompt（保护 prompt cache）。
    """
    from agent.memory_store import MemoryStore
    from agent.prompt_builder import build_system_prompt_layers
    from agent.workspace_context import workspace_cwd_context

    with workspace_cwd_context(str(tmp_path)):
        ms = MemoryStore(omnimate_home=tmp_path)
        ms.save(name="某记忆", description="描述", type="user")
        layers = build_system_prompt_layers(
            memory_store=ms, include_guidance=False,
        )
    assert "记忆索引" not in layers.context


def test_fallback_to_snapshot_without_aux():
    """无 aux_llm_router 时主循环降级回 snapshot 注入。

    验证降级函数：_fallback_snapshot_message(ms) 构造 snapshot 注入。
    """
    from agent.memory_injection import _fallback_snapshot_message
    ms = MagicMock()
    ms.snapshot_for_prompt.return_value = "索引内容"
    msg = _fallback_snapshot_message(ms)
    assert msg is not None and "索引内容" in msg["content"]
    assert msg["_ephemeral"] is True
    # 空 snapshot → None
    ms.snapshot_for_prompt.return_value = ""
    assert _fallback_snapshot_message(ms) is None
    # 异常 fail-open
    ms.snapshot_for_prompt.side_effect = RuntimeError("boom")
    assert _fallback_snapshot_message(ms) is None
