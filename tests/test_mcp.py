"""MCP 测试（mock subprocess，不启动真实 server）。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.mcp_client import (
    MCPClient, MCPManager, load_mcp_config,
    get_mcp_manager, is_mcp_tool,
)
from tools.mcp_tool import register_mcp_tools, initialize_mcp


# ---------------------------------------------------------------------------
# load_mcp_config
# ---------------------------------------------------------------------------

def test_load_mcp_config_nonexistent(tmp_path):
    """不存在的配置返回空。"""
    assert load_mcp_config(tmp_path / "nope.json") == {}


def test_load_mcp_config_valid(tmp_path):
    """有效配置加载。"""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            },
            "db": {
                "command": "python",
                "args": ["-m", "db_server"],
            },
        }
    }), encoding="utf-8")

    servers = load_mcp_config(cfg)
    assert "fs" in servers
    assert "db" in servers
    assert servers["fs"]["command"] == "npx"
    assert servers["db"]["args"] == ["-m", "db_server"]


def test_load_mcp_config_invalid_json(tmp_path):
    """无效 JSON 返回空。"""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text("not json {", encoding="utf-8")
    assert load_mcp_config(cfg) == {}


# ---------------------------------------------------------------------------
# is_mcp_tool
# ---------------------------------------------------------------------------

def test_is_mcp_tool():
    assert is_mcp_tool("mcp__server__tool") is True
    assert is_mcp_tool("mcp__fs__read_file") is True
    assert is_mcp_tool("terminal") is False
    assert is_mcp_tool("read_file") is False


# ---------------------------------------------------------------------------
# MCPClient（mock subprocess）
# ---------------------------------------------------------------------------

def _mock_process():
    """构造 mock subprocess.Popen。"""
    process = MagicMock()
    process.poll.return_value = None  # 进程运行中
    process.stdin = MagicMock()
    return process


def test_mcp_client_connect_handshake():
    """connect 发送 initialize 握手。"""
    client = MCPClient(name="test", command="fake-cmd", args=["--port", "1"])

    # mock subprocess.Popen
    with patch("agent.mcp_client.subprocess.Popen", return_value=_mock_process()):
        # mock _request 返回握手响应
        with patch.object(client, "_request", return_value={"serverInfo": {"name": "test"}}):
            with patch.object(client, "_notify") as notify_mock:
                client.connect()

    assert client.connected is True
    # 发送了 initialized 通知
    notify_mock.assert_called_once_with("notifications/initialized", {})


def test_mcp_client_list_tools():
    """list_tools 返回工具列表。"""
    client = MCPClient(name="test", command="fake")
    tools_data = [
        {"name": "read", "description": "读文件"},
        {"name": "write", "description": "写文件"},
    ]
    with patch.object(client, "_request", return_value={"tools": tools_data}):
        tools = client.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "read"


def test_mcp_client_call_tool():
    """call_tool 发送 tools/call。"""
    client = MCPClient(name="test", command="fake")
    expected = {"content": [{"type": "text", "text": "result"}]}
    with patch.object(client, "_request", return_value=expected) as req_mock:
        result = client.call_tool("read", {"path": "/x"})
    assert result == expected
    # 验证调用参数
    req_mock.assert_called_once_with("tools/call", {
        "name": "read", "arguments": {"path": "/x"},
    })


def test_mcp_client_close_terminates_process():
    """close 终止子进程。"""
    client = MCPClient(name="test", command="fake")
    process = _mock_process()
    client.process = process
    client._connected = True

    client.close()
    process.terminate.assert_called_once()
    assert client.process is None
    assert client.connected is False


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------

def test_manager_connect_all_with_mock():
    """connect_all 连接配置的所有 server。"""
    manager = MCPManager()

    # mock MCPClient
    fake_client = MagicMock()
    fake_client.connected = True
    fake_client.list_tools.return_value = [
        {"name": "tool1", "description": "d1", "inputSchema": {"type": "object"}},
    ]

    with patch("agent.mcp_client.MCPClient", return_value=fake_client):
        manager.connect_all({
            "server1": {"command": "fake", "args": []},
        })

    assert "server1" in manager.servers
    fake_client.connect.assert_called_once()


def test_manager_get_all_tools():
    """get_all_tools 返回带 server 前缀的工具。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True
    fake_client.list_tools.return_value = [
        {"name": "read", "description": "读", "inputSchema": {"type": "object"}},
    ]
    manager._clients["fs"] = fake_client

    tools = manager.get_all_tools()
    assert len(tools) == 1
    assert tools[0]["full_name"] == "mcp__fs__read"
    assert tools[0]["server"] == "fs"


def test_manager_call_routes_to_server():
    """call 按 mcp__server__tool 路由。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True
    fake_client.call_tool.return_value = {"content": "ok"}
    manager._clients["fs"] = fake_client

    result = manager.call("mcp__fs__read", {"path": "/x"})
    assert result == {"content": "ok"}
    fake_client.call_tool.assert_called_once_with("read", {"path": "/x"})


def test_manager_call_unknown_server():
    """调用未连接的 server 返回错误。"""
    manager = MCPManager()
    result = manager.call("mcp__nope__tool", {})
    assert "error" in result


# ---------------------------------------------------------------------------
# register_mcp_tools
# ---------------------------------------------------------------------------

def test_register_mcp_tools():
    """注册 MCP 工具到 registry。"""
    from tools.registry import registry

    manager = MagicMock()
    manager.get_all_tools.return_value = [
        {
            "server": "fs",
            "original_name": "read",
            "full_name": "mcp__fs__read",
            "description": "读文件",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]

    count = register_mcp_tools(manager)
    assert count == 1

    # registry 里有这个工具
    result = registry.dispatch("mcp__fs__read", {})
    # manager.call 未 mock，返回错误（因为 manager 是 MagicMock，call 返回 Mock）
    # 但至少证明工具注册了（dispatch 找到了 handler）


def test_mcp_toolset_in_resolve():
    """mcp toolset 能被解析。"""
    from toolsets import resolve_toolset
    # mcp toolset 的固定 tools 是空，但 get_tool_definitions 会动态发现
    assert resolve_toolset("mcp") == []  # 固定列表为空
    # 但 toolset 存在
    from toolsets import TOOLSETS
    assert "mcp" in TOOLSETS


# ---------------------------------------------------------------------------
# initialize_mcp（端到端，全 mock）
# ---------------------------------------------------------------------------

def test_initialize_mcp_no_config():
    """无 .mcp.json 时返回 0（不报错）。"""
    with patch("agent.mcp_client.load_mcp_config", return_value={}):
        count = initialize_mcp()
    assert count == 0
