import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.memory_recall_tool import _handle_memory_recall, MEMORY_RECALL_SCHEMA


def test_schema_basic():
    assert MEMORY_RECALL_SCHEMA["name"] == "memory_recall"
    props = MEMORY_RECALL_SCHEMA["parameters"]["properties"]
    assert "query" in props
    assert "top_k" in props
    assert props["top_k"]["default"] == 5
    assert "query" in MEMORY_RECALL_SCHEMA["parameters"]["required"]


def test_handle_no_memory_store_returns_not_configured():
    """memory_store 缺失（dispatch_kwargs 里没传）时返回 not_configured 错误。

    模拟 dispatch 真实调用：handler(args, **dispatch_kwargs)。
    不传 memory_store= → dispatch_kwargs.get("memory_store") 为 None。
    """
    result = asyncio.run(
        _handle_memory_recall({"query": "test"}, agent_ref=None)
    )
    data = json.loads(result)
    assert data["error_type"] == "not_configured"


def test_handle_no_aux_llm_returns_not_configured():
    """agent_ref 有但 aux_llm_router 为 None 时返回 not_configured。"""
    memory_store = MagicMock()
    memory_store.full_index_text.return_value = "index"
    agent_ref = MagicMock()
    agent_ref.aux_llm_router = None  # 未配置 aux_llm

    result = asyncio.run(
        _handle_memory_recall(
            {"query": "test"},
            memory_store=memory_store,
            agent_ref=agent_ref,
        )
    )
    data = json.loads(result)
    assert data["error_type"] == "not_configured"


def test_handle_returns_hits():
    """正常路径：retrieve_relevant 返回 id 列表，再查 MemoryStore 拿正文。"""
    memory_store = MagicMock()
    memory_store.full_index_text.return_value = "idx"

    # agent_ref 上挂 aux_llm_router（对齐 _auto_recall_memory 接线）
    agent_ref = MagicMock()
    agent_ref.aux_llm_router = MagicMock()

    # 模拟 memory
    entry = MagicMock()
    entry.id = "general#abc"
    entry.name = "user_role"
    entry.description = "用户角色"
    entry.summary = "L1 摘要"
    entry.body = "完整正文"
    memory_store.get.return_value = entry

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=["general#abc"]),
    ):
        result = asyncio.run(
            _handle_memory_recall(
                {"query": "用户角色", "top_k": 5},
                memory_store=memory_store,
                agent_ref=agent_ref,
            )
        )
    data = json.loads(result)
    assert data["count"] == 1
    assert data["hits"][0]["id"] == "general#abc"
    assert data["hits"][0]["summary"] == "L1 摘要"


def test_handle_top_k_clamped():
    """top_k > 20 截到 20，< 1 截到 1。"""
    memory_store = MagicMock()
    memory_store.full_index_text.return_value = "idx"
    agent_ref = MagicMock()
    agent_ref.aux_llm_router = MagicMock()

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=[]),
    ) as mock_retrieve:
        asyncio.run(
            _handle_memory_recall(
                {"query": "x", "top_k": 100},
                memory_store=memory_store,
                agent_ref=agent_ref,
            )
        )
        # max_results 应该被截到 20
        assert mock_retrieve.call_args.kwargs["max_results"] == 20

        asyncio.run(
            _handle_memory_recall(
                {"query": "x", "top_k": -5},
                memory_store=memory_store,
                agent_ref=agent_ref,
            )
        )
        assert mock_retrieve.call_args.kwargs["max_results"] == 1


def test_handle_does_not_affect_snapshot():
    """调用 memory_recall 不影响 MemoryStore.snapshot_for_prompt（frozen）。"""
    memory_store = MagicMock()
    memory_store.full_index_text.return_value = "idx"
    memory_store.snapshot_for_prompt.return_value = "FROZEN"
    agent_ref = MagicMock()
    agent_ref.aux_llm_router = MagicMock()

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(return_value=[]),
    ):
        asyncio.run(
            _handle_memory_recall(
                {"query": "x"},
                memory_store=memory_store,
                agent_ref=agent_ref,
            )
        )
    # snapshot_for_prompt 没被调用（frozen 保持不变）
    memory_store.snapshot_for_prompt.assert_not_called()


def test_registered_in_registry():
    """模块 import 后 memory_recall 应该已注册到 registry 的 memory toolset。"""
    import tools.memory_recall_tool  # noqa: 触发注册
    from tools.registry import registry
    entry = registry.get("memory_recall")
    assert entry is not None
    assert entry.toolset == "memory"
    assert entry.is_async is True


def test_memory_recall_in_core_tools():
    """memory_recall 必须进 _CORE_TOOLS 才对 LLM 可见（发现 ≠ 可见）。

    防回归点：注册 toolset="memory" 但 "memory" 不在 TOOLSETS 定义的话，
    工具实现了却从未暴露给 LLM（silent-dead-code）。
    """
    from toolsets import _CORE_TOOLS
    assert "memory_recall" in _CORE_TOOLS


def test_handle_exception_returns_error():
    """retrieve_relevant 抛异常时返回 error JSON。"""
    memory_store = MagicMock()
    memory_store.full_index_text.return_value = "idx"
    agent_ref = MagicMock()
    agent_ref.aux_llm_router = MagicMock()

    with patch(
        "tools.memory_recall_tool.retrieve_relevant",
        new=AsyncMock(side_effect=RuntimeError("LLM 挂了")),
    ):
        result = asyncio.run(
            _handle_memory_recall(
                {"query": "x"},
                memory_store=memory_store,
                agent_ref=agent_ref,
            )
        )
    data = json.loads(result)
    assert "error" in data
    assert data["error_type"] == "RuntimeError"


def test_handler_signature_matches_dispatch_contract():
    """dispatch 调 handler(args, **kwargs)，签名必须兼容（防 silent-dead-code）。

    历史教训：曾写成 (args, kwargs, ctx) 三位置参数，单元测试直调三参数
    漏检，生产 dispatch 调用 100% TypeError（silent-dead-code）。
    async handler 同样适用——registry.dispatch 用 inspect.iscoroutinefunction
    判定后 await handler(args, **kwargs)。
    """
    sig = inspect.signature(_handle_memory_recall)
    params = list(sig.parameters.values())
    # 第一个参数是位置参数（args）
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[0].name == "args"
    # 必须有 **kwargs 接收 dispatch 上下文
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    # 必须是 async（registry 据此走 await 分支）
    assert inspect.iscoroutinefunction(_handle_memory_recall)
