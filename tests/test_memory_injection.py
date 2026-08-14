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
    """同 query 第二次调用不再触发检索（一轮缓存）。"""
    store = _make_store({"general#a": ("n", "user", "b")})
    mock_rr = AsyncMock(return_value=["general#a"])
    with patch("agent.memory_injection.retrieve_relevant", new=mock_rr):
        asyncio.run(build_relevant_memories_message(
            query="same", memory_store=store, aux_llm_router=MagicMock()))
        asyncio.run(build_relevant_memories_message(
            query="same", memory_store=store, aux_llm_router=MagicMock()))
    assert mock_rr.await_count == 1
