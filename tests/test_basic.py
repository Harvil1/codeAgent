"""基础测试：验证 budget、registry、toolsets、config 等核心模块。

运行：
    uv run pytest tests/ -v
"""

import json
from pathlib import Path

import pytest

from agent.budget import IterationBudget
from tools.registry import registry, discover_builtin_tools
from toolsets import resolve_toolset
from config import load_config, _deep_merge


@pytest.fixture(autouse=True)
def _isolate_global_registry():
    """每个测试自动隔离全局 registry 状态。

    test_basic.py 注册测试工具（test_tool_basic / test_tool_bad / test_tool_dict）
    到全局 registry 单例。如不清理，后续测试（如 test_all_tools_classified）
    会看到这些测试工具，误判为"未分类工具"而 fail。

    本 fixture 在每个测试前 snapshot registry._tools，测试后恢复。
    """
    snapshot = dict(registry._tools)
    yield
    registry._tools.clear()
    registry._tools.update(snapshot)


# ---------------------------------------------------------------------------
# IterationBudget
# ---------------------------------------------------------------------------

def test_budget_consume():
    budget = IterationBudget(5)
    assert budget.remaining == 5
    assert budget.consume() is True
    assert budget.remaining == 4


def test_budget_exhausted():
    budget = IterationBudget(2)
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.remaining == 0


def test_budget_refund():
    budget = IterationBudget(1)
    budget.consume()
    assert budget.remaining == 0
    budget.refund()
    assert budget.remaining == 1


def test_budget_reset():
    budget = IterationBudget(5)
    budget.consume()
    budget.consume()
    budget.reset()
    assert budget.remaining == 5
    budget.reset(total=10)
    assert budget.remaining == 10


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

async def test_registry_register_and_dispatch():
    """测试工具注册和分发。"""
    # 注意：name 加前缀避免和真实工具冲突
    registry.register(
        name="test_tool_basic",
        toolset="test",
        schema={"name": "test_tool_basic", "description": "测试", "parameters": {}},
        handler=lambda args, **kw: json.dumps({"result": "ok"}, ensure_ascii=False),
    )

    result = await registry.dispatch("test_tool_basic", {})
    data = json.loads(result)
    assert data["result"] == "ok"


async def test_registry_unknown_tool():
    """测试未知工具返回错误。"""
    result = await registry.dispatch("nonexistent_tool_xyz", {})
    data = json.loads(result)
    assert "error" in data
    assert data.get("error_type") == "unknown_tool"


async def test_registry_handler_exception():
    """测试 handler 抛异常时返回错误 JSON。"""
    def bad_handler(args, **kw):
        raise RuntimeError("boom")

    registry.register(
        name="test_tool_bad",
        toolset="test",
        schema={"name": "test_tool_bad", "description": "", "parameters": {}},
        handler=bad_handler,
    )

    result = await registry.dispatch("test_tool_bad", {})
    data = json.loads(result)
    assert data["error_type"] == "tool_exception"
    assert "boom" in data["error"]


async def test_registry_dict_result_normalized():
    """dict 返回值被规范化为 JSON 字符串。"""
    registry.register(
        name="test_tool_dict",
        toolset="test",
        schema={"name": "test_tool_dict", "description": "", "parameters": {}},
        handler=lambda args, **kw: {"k": "v"},
    )
    result = await registry.dispatch("test_tool_dict", {})
    assert isinstance(result, str)
    assert json.loads(result) == {"k": "v"}


# ---------------------------------------------------------------------------
# Toolsets
# ---------------------------------------------------------------------------

def test_resolve_core_toolset():
    tools = resolve_toolset("core")
    assert "terminal" in tools
    assert "read_file" in tools


def test_resolve_unknown_toolset():
    assert resolve_toolset("nonexistent") == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_config():
    config = load_config(Path("/nonexistent/path/config.yaml"))
    assert config["model"]["provider"] == "deepseek"
    assert config["model"]["name"] == "deepseek-chat"
    assert "core" in config["enabled_toolsets"]


def test_deep_merge():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 3, "z": 4}, "c": 5}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}


# ---------------------------------------------------------------------------
# 工具发现（集成）
# ---------------------------------------------------------------------------

def test_discover_builtin_tools():
    """验证自动发现能找到 terminal/file_operations。"""
    imported = discover_builtin_tools()
    # 至少 terminal_tool 和 file_operations 应被发现
    assert any("terminal_tool" in m for m in imported)
    assert any("file_operations" in m for m in imported)
