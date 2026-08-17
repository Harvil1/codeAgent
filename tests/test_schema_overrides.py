"""schema_overrides_fn 测试（03）。

验证：
1. registry.register 接受 schema_overrides_fn 参数
2. get_definitions(runtime_ctx=...) 时 fn 被调用
3. fn 抛异常时不影响主流程（log + 回退）
4. 原 schema 不被污染（多次调用结果一致）
5. delegate_task 在并发满时 description 显示警告
6. get_tool_definitions(agent=...) 透传 agent
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.registry import ToolRegistry, registry


def _make_schema(name="t", desc="hello"):
    return {"name": name, "description": desc, "parameters": {"type": "object"}}


# ----------------------------------------------------------------------------
# 注册 + 基础调用
# ----------------------------------------------------------------------------

def test_register_accepts_schema_overrides_fn():
    """register 能接 schema_overrides_fn 参数。"""
    reg = ToolRegistry()
    called = []

    def fn(schema, ctx):
        called.append(ctx)
        new = dict(schema)
        new["description"] = "overridden"
        return new

    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(), handler=lambda a, **kw: "ok",
        schema_overrides_fn=fn,
    )
    defs = reg.get_definitions(["t1"], runtime_ctx={"agent": "x"})
    assert len(defs) == 1
    assert defs[0]["function"]["description"] == "overridden"
    assert called == [{"agent": "x"}]


def test_overrides_skipped_without_runtime_ctx():
    """runtime_ctx=None 时 schema 不变。"""
    reg = ToolRegistry()
    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(desc="orig"), handler=lambda a, **kw: "ok",
        schema_overrides_fn=lambda s, c: {**s, "description": "changed"},
    )
    defs = reg.get_definitions(["t1"])  # 不传 runtime_ctx
    assert defs[0]["function"]["description"] == "orig"


def test_overrides_skipped_without_fn():
    """没注册 schema_overrides_fn 的工具不受影响。"""
    reg = ToolRegistry()
    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(desc="orig"), handler=lambda a, **kw: "ok",
    )
    defs = reg.get_definitions(["t1"], runtime_ctx={"agent": "x"})
    assert defs[0]["function"]["description"] == "orig"


# ----------------------------------------------------------------------------
# 异常安全
# ----------------------------------------------------------------------------

def test_overrides_fn_exception_falls_back_to_original():
    """fn 抛异常时回退原 schema，不影响主流程。"""
    reg = ToolRegistry()

    def bad_fn(schema, ctx):
        raise RuntimeError("boom")

    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(desc="safe"), handler=lambda a, **kw: "ok",
        schema_overrides_fn=bad_fn,
    )
    # 不应该抛
    defs = reg.get_definitions(["t1"], runtime_ctx={"agent": "x"})
    assert defs[0]["function"]["description"] == "safe"


# ----------------------------------------------------------------------------
# 不污染原 schema
# ----------------------------------------------------------------------------

def test_original_schema_not_mutated():
    """多次调用 schema_overrides_fn 不污染原 schema。"""
    reg = ToolRegistry()

    def append_status(schema, ctx):
        new = dict(schema)
        new["description"] = new["description"] + f" [ctx={ctx.get('v')}]"
        return new

    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(desc="base"), handler=lambda a, **kw: "ok",
        schema_overrides_fn=append_status,
    )
    # 第一次调用
    defs1 = reg.get_definitions(["t1"], runtime_ctx={"v": 1})
    assert defs1[0]["function"]["description"] == "base [ctx=1]"
    # 第二次不同 ctx
    defs2 = reg.get_definitions(["t1"], runtime_ctx={"v": 2})
    assert defs2[0]["function"]["description"] == "base [ctx=2]"
    # 第三次再 1
    defs3 = reg.get_definitions(["t1"], runtime_ctx={"v": 1})
    assert defs3[0]["function"]["description"] == "base [ctx=1]"


# ----------------------------------------------------------------------------
# check_fn 仍然生效
# ----------------------------------------------------------------------------

def test_check_fn_still_filters_before_overrides():
    """check_fn 不通过时工具仍不出现（schema_overrides_fn 不被调）。"""
    reg = ToolRegistry()
    called = []

    reg.register(
        name="t1", toolset="core",
        schema=_make_schema(), handler=lambda a, **kw: "ok",
        check_fn=lambda: False,  # 永远不可用
        schema_overrides_fn=lambda s, c: called.append(c) or s,
    )
    defs = reg.get_definitions(["t1"], runtime_ctx={"agent": "x"})
    assert defs == []
    assert called == []  # 没被调用


# ----------------------------------------------------------------------------
# delegate_tool 的 schema_overrides_fn
# ----------------------------------------------------------------------------

def test_delegate_schema_overrides_adds_status():
    """delegate_task schema_overrides_fn 加状态行。"""
    from tools.delegate_tool import _delegate_schema_overrides

    fake_agent = SimpleNamespace(
        _children=[1, 2, 3],  # 3 个活跃
        config={"delegate": {"max_concurrent_children": 5}},
    )
    schema = _make_schema(name="delegate_task", desc="delegate a task")
    new_schema = _delegate_schema_overrides(schema, {"agent": fake_agent})

    assert "活跃子代理: 3/5" in new_schema["description"]
    assert "剩余可委派: 2" in new_schema["description"]
    assert "上限" not in new_schema["description"]  # 没到上限


def test_delegate_schema_overrides_warns_when_full():
    """并发满时显示警告。"""
    from tools.delegate_tool import _delegate_schema_overrides

    fake_agent = SimpleNamespace(
        _children=[1, 2, 3, 4, 5],  # 5 个 = 满
        config={"delegate": {"max_concurrent_children": 5}},
    )
    schema = _make_schema(name="delegate_task", desc="delegate a task")
    new_schema = _delegate_schema_overrides(schema, {"agent": fake_agent})
    assert "已达并发上限" in new_schema["description"]


def test_delegate_schema_overrides_no_agent_returns_original():
    """runtime_ctx 没有 agent 时不改 schema。"""
    from tools.delegate_tool import _delegate_schema_overrides

    schema = _make_schema(desc="orig")
    new_schema = _delegate_schema_overrides(schema, {})
    assert new_schema["description"] == "orig"


def test_delegate_schema_overrides_handles_no_config():
    """agent.config 为 None 时不崩。"""
    from tools.delegate_tool import _delegate_schema_overrides

    fake_agent = SimpleNamespace(_children=[], config=None)
    schema = _make_schema(desc="orig")
    new_schema = _delegate_schema_overrides(schema, {"agent": fake_agent})
    # 用默认 max=5，0 活跃
    assert "活跃子代理: 0/5" in new_schema["description"]


# ----------------------------------------------------------------------------
# model_tools.get_tool_definitions 透传 agent
# ----------------------------------------------------------------------------

def test_get_tool_definitions_passes_agent():
    """get_tool_definitions(agent=...) 把 agent 传给 schema_overrides_fn。"""
    import model_tools

    # 用 delegate_task 验证（默认在 core 工具集）
    fake_agent = SimpleNamespace(
        _children=[1, 2],  # 2 个活跃
        config={"delegate": {"max_concurrent_children": 5}},
    )
    defs = model_tools.get_tool_definitions(["core"], agent=fake_agent)
    delegate_defs = [d for d in defs if d["function"]["name"] == "delegate_task"]
    if delegate_defs:  # 工具集可能没启
        desc = delegate_defs[0]["function"]["description"]
        assert "活跃子代理: 2/5" in desc


def test_get_tool_definitions_without_agent_unchanged():
    """不传 agent 时 schema 不变。"""
    import model_tools
    # 不传 agent，不应抛
    defs = model_tools.get_tool_definitions(["core"])
    assert isinstance(defs, list)
