import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.memory_recall_tool import _handle_memory_recall, MEMORY_RECALL_SCHEMA


def test_schema_basic():
    assert MEMORY_RECALL_SCHEMA["name"] == "memory_recall"
    props = MEMORY_RECALL_SCHEMA["inputSchema"]["properties"]
    assert "query" in props
    assert "top_k" in props
    assert props["top_k"]["default"] == 5
    assert "query" in MEMORY_RECALL_SCHEMA["inputSchema"]["required"]


def test_handle_no_ctx_returns_not_configured():
    """ctx 为 None 时返回 not_configured 错误。"""
    result = asyncio.run(_handle_memory_recall({}, {"query": "test"}, None))
    data = json.loads(result)
    assert data["error_type"] == "not_configured"


def test_handle_no_aux_llm_returns_not_configured():
    """ctx 有 memory_store 但 aux_llm_client 为 None 时返回 not_configured。"""
    ctx = MagicMock()
    ctx.memory_store.full_index_text.return_value = "index"
    ctx.aux_llm_client = None
    result = asyncio.run(_handle_memory_recall({}, {"query": "test"}, ctx))
    data = json.loads(result)
    assert data["error_type"] == "not_configured"


def test_handle_returns_hits():
    """正常路径：retrieve_relevant 返回 id 列表，再查 MemoryStore 拿正文。"""
    ctx = MagicMock()
    ctx.memory_store.full_index_text.return_value = "idx"
    ctx.aux_llm_client = MagicMock()
    ctx.aux_model = "deepseek-chat"

    # 模拟 memory
    entry = MagicMock()
    entry.id = "general#abc"
    entry.name = "user_role"
    entry.description = "用户角色"
    entry.summary = "L1 摘要"
    entry.body = "完整正文"
    ctx.memory_store.get.return_value = entry

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=["general#abc"]),
    ):
        result = asyncio.run(
            _handle_memory_recall({}, {"query": "用户角色", "top_k": 5}, ctx)
        )
    data = json.loads(result)
    assert data["count"] == 1
    assert data["hits"][0]["id"] == "general#abc"
    assert data["hits"][0]["summary"] == "L1 摘要"


def test_handle_top_k_clamped():
    """top_k > 20 截到 20，< 1 截到 1。"""
    ctx = MagicMock()
    ctx.memory_store.full_index_text.return_value = "idx"
    ctx.aux_llm_client = MagicMock()
    ctx.aux_model = "m"

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=[]),
    ) as mock_retrieve:
        asyncio.run(_handle_memory_recall({}, {"query": "x", "top_k": 100}, ctx))
        # max_results 应该被截到 20
        assert mock_retrieve.call_args.kwargs["max_results"] == 20

        asyncio.run(_handle_memory_recall({}, {"query": "x", "top_k": -5}, ctx))
        assert mock_retrieve.call_args.kwargs["max_results"] == 1


def test_handle_does_not_affect_snapshot():
    """调用 memory_recall 不影响 MemoryStore.snapshot_for_prompt（frozen）。"""
    ctx = MagicMock()
    ctx.memory_store.full_index_text.return_value = "idx"
    ctx.memory_store.snapshot_for_prompt.return_value = "FROZEN"
    ctx.aux_llm_client = MagicMock()
    ctx.aux_model = "m"
    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=[]),
    ):
        asyncio.run(_handle_memory_recall({}, {"query": "x"}, ctx))
    # snapshot_for_prompt 没被调用（frozen 保持不变）
    ctx.memory_store.snapshot_for_prompt.assert_not_called()


def test_registered_in_registry():
    """模块 import 后 memory_recall 应该已注册到 registry 的 memory toolset。"""
    import tools.memory_recall_tool  # noqa: 触发注册
    from tools.registry import registry
    entry = registry.get("memory_recall")
    assert entry is not None
    assert entry.toolset == "memory"
    assert entry.is_async is True


def test_handle_exception_returns_error():
    """retrieve_relevant 抛异常时返回 error JSON。"""
    ctx = MagicMock()
    ctx.memory_store.full_index_text.return_value = "idx"
    ctx.aux_llm_client = MagicMock()
    ctx.aux_model = "m"

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(side_effect=RuntimeError("LLM 挂了")),
    ):
        result = asyncio.run(
            _handle_memory_recall({}, {"query": "x"}, ctx)
        )
    data = json.loads(result)
    assert "error" in data
    assert data["error_type"] == "RuntimeError"
