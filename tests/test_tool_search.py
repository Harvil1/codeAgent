"""Task D1: tool_search 工具 + catalog + get_tool_definitions 拆分。

测试 ToolSearch 功能：
1. tool_search 按关键字模糊匹配 mcp__ 工具，返回完整 schema
2. 空 query 报错
3. 无匹配返回空 results
4. registry.get_catalog_entry 返回精简 schema（含 hint）
5. get_tool_definitions 对 mcp__ 工具走 catalog，built-in 走完整 schema
"""

import json
from unittest.mock import patch, MagicMock

import pytest


def test_tool_search_returns_matching_mcp_schemas():
    """tool_search 模糊匹配 mcp__ 工具名/描述，返回完整 schema。"""
    from tools.tool_search_tool import _handle_tool_search
    # 用真实 ToolEntry 注册到独立的 ToolRegistry 实例上，
    # 然后 patch tools.tool_search_tool.registry 指向它（避免污染全局单例）
    from tools.registry import ToolRegistry, ToolEntry

    fake_schemas = {
        "mcp__github__create_pull_request": {
            "name": "mcp__github__create_pull_request",
            "description": "创建 PR",
            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
        "mcp__github__list_issues": {
            "name": "mcp__github__list_issues",
            "description": "列 issue",
            "parameters": {"type": "object", "properties": {}},
        },
        "mcp__filesystem__read_file": {
            "name": "mcp__filesystem__read_file",
            "description": "读文件",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    fake_reg = ToolRegistry()
    for n, s in fake_schemas.items():
        fake_reg.register(
            name=n, toolset="mcp", schema=s, handler=lambda args, **kw: "{}",
        )
    with patch("tools.tool_search_tool.registry", fake_reg):
        out = json.loads(_handle_tool_search({"query": "github pr"}))
    assert "results" in out
    names = [r["name"] for r in out["results"]]
    assert "mcp__github__create_pull_request" in names
    assert "mcp__filesystem__read_file" not in names  # 不匹配


def test_tool_search_empty_query_returns_error():
    from tools.tool_search_tool import _handle_tool_search
    out = json.loads(_handle_tool_search({"query": ""}))
    assert "error" in out


def test_tool_search_no_match_returns_empty():
    from tools.tool_search_tool import _handle_tool_search
    from tools.registry import ToolRegistry
    fake_reg = ToolRegistry()
    fake_reg.register(
        name="mcp__x__y", toolset="mcp",
        schema={"name": "mcp__x__y", "description": "x", "parameters": {}},
        handler=lambda args, **kw: "{}",
    )
    with patch("tools.tool_search_tool.registry", fake_reg):
        out = json.loads(_handle_tool_search({"query": "zzz"}))
    assert out["results"] == []


def test_get_catalog_entry_minimal():
    """registry.get_catalog_entry 返回精简 schema（name + 短描述 + hint）。"""
    from tools.registry import ToolRegistry, ToolEntry
    reg = ToolRegistry()
    reg.register(
        name="mcp__srv__tool",
        toolset="mcp",
        schema={"name": "mcp__srv__tool", "description": "一个很长的描述" * 10,
                "parameters": {"type": "object", "properties": {"a": {}}}},
        handler=lambda args, **kw: "{}",
    )
    cat = reg.get_catalog_entry("mcp__srv__tool")
    assert cat["name"] == "mcp__srv__tool"
    assert "tool_search" in cat["description"]  # 含 hint
    assert "parameters" not in cat or cat["parameters"] == {"type": "object", "properties": {}}


def test_get_tool_definitions_uses_catalog_for_mcp():
    """get_tool_definitions 对 mcp__ 工具用 catalog，built-in 用完整 schema。"""
    from model_tools import get_tool_definitions, ensure_tools_discovered
    from tools.registry import registry
    ensure_tools_discovered()
    # 注册一个假 mcp 工具
    registry.register(
        name="mcp__test__probe",
        toolset="mcp",
        schema={"name": "mcp__test__probe", "description": "probe",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}},
        handler=lambda args, **kw: "{}",
        override=True,
    )
    defs = get_tool_definitions(["core", "mcp"])
    by_name = {d["function"]["name"]: d["function"] for d in defs}
    # built-in terminal 应有完整 parameters
    assert "terminal" in by_name
    assert "parameters" in by_name["terminal"]
    # mcp__ 工具应是 catalog（无详细 parameters 或空）
    assert "mcp__test__probe" in by_name
    params = by_name["mcp__test__probe"].get("parameters", {})
    # catalog 应缺详细 properties 或为空对象
    assert not params.get("properties", {}).get("x"), "catalog 不该含详细 parameters"


def test_tool_search_respects_mcp_server_filter():
    """child 的 mcp_server_filter 限制 tool_search 只搜可见子集。

    场景：自定义子代理 mcpServers: [github]，LLM schema 目录里只有 github 工具
    （C1 filter 生效），但调 tool_search(query='issue') 不应搜出 filesystem/jira 等
    其他连接 server 的工具——否则 LLM 拿到完整参数后调用，registry.dispatch 命中，
    实际执行被 filter 掉的工具，违背 spec 第 286 行承诺。
    """
    from tools.tool_search_tool import _handle_tool_search
    from tools.registry import ToolRegistry, ToolEntry

    fake_schemas = {
        "mcp__github__list_issues": {
            "name": "mcp__github__list_issues",
            "description": "列 issue",
            "parameters": {"type": "object", "properties": {}},
        },
        "mcp__filesystem__read_file": {
            "name": "mcp__filesystem__read_file",
            "description": "读文件 issue 相关",
            "parameters": {"type": "object", "properties": {}},
        },
        "mcp__jira__create_ticket": {
            "name": "mcp__jira__create_ticket",
            "description": "建 issue ticket",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    fake_reg = ToolRegistry()
    for n, s in fake_schemas.items():
        fake_reg.register(
            name=n, toolset="mcp", schema=s, handler=lambda args, **kw: "{}",
        )
    # child agent 只允许 github
    class FakeAgent:
        config = {"mcp_server_filter": ["github"]}

    with patch("tools.tool_search_tool.registry", fake_reg):
        out = json.loads(_handle_tool_search({"query": "issue"}, agent_ref=FakeAgent()))
    names = [r["name"] for r in out["results"]]
    # 应只返回 github 工具（即使 filesystem/jira 描述含 issue 也不该出现）
    assert all("github" in n for n in names), f"含非 github 工具: {names}"
    assert "mcp__github__list_issues" in names


def test_tool_search_no_filter_returns_all_servers():
    """无 mcp_server_filter（主代理场景）—— 所有 server 工具都搜得到。"""
    from tools.tool_search_tool import _handle_tool_search
    from tools.registry import ToolRegistry

    fake_reg = ToolRegistry()
    for n in ["mcp__github__list_issues", "mcp__jira__create_ticket"]:
        fake_reg.register(
            name=n, toolset="mcp",
            schema={"name": n, "description": "issue", "parameters": {}},
            handler=lambda args, **kw: "{}",
        )
    # 无 agent_ref → 不过滤
    with patch("tools.tool_search_tool.registry", fake_reg):
        out = json.loads(_handle_tool_search({"query": "issue"}))
    names = [r["name"] for r in out["results"]]
    assert "mcp__github__list_issues" in names
    assert "mcp__jira__create_ticket" in names
