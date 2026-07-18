"""MCP 测试（mock subprocess，不启动真实 server）。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.mcp_client import (
    MCPClient, MCPManager, load_mcp_config,
    get_mcp_manager, is_mcp_tool,
    MCPTransport, StdioTransport, HTTPTransport,
)
from tools.mcp_tool import register_mcp_tools, initialize_mcp


# ---------------------------------------------------------------------------
# load_mcp_config
# ---------------------------------------------------------------------------

def test_load_mcp_config_nonexistent(tmp_path):
    """不存在的配置返回空。"""
    assert load_mcp_config(tmp_path / "nope.json") == {}


def test_load_mcp_config_valid(tmp_path):
    """有效配置加载（stdio 格式）。"""
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


def test_load_mcp_config_supports_http_format(tmp_path):
    """HTTP/OAuth 配置也能加载。"""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "github": {
                "url": "https://api.github-mcp.com/v1",
                "headers": {"X-Custom": "v"},
            },
            "notion": {
                "url": "https://mcp.notion.com/v1",
                "oauth": {"token_url": "x", "client_id": "y"},
            },
        }
    }), encoding="utf-8")
    servers = load_mcp_config(cfg)
    assert servers["github"]["url"] == "https://api.github-mcp.com/v1"
    assert servers["notion"]["oauth"]["client_id"] == "y"


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
# MCPClient transport 选择
# ---------------------------------------------------------------------------

def test_mcp_client_selects_stdio_transport():
    """配 command → StdioTransport。"""
    client = MCPClient(name="t", command="fake-cmd", args=["--port", "1"])
    assert isinstance(client._transport, StdioTransport)


def test_mcp_client_selects_http_transport():
    """配 url → HTTPTransport。"""
    client = MCPClient(name="t", url="https://example.com/mcp")
    assert isinstance(client._transport, HTTPTransport)


def test_mcp_client_requires_command_or_url():
    """没 command 也没 url → ValueError。"""
    with pytest.raises(ValueError):
        MCPClient(name="t")


def test_mcp_client_connect_handshake():
    """connect 触发 transport.connect（含 initialize 握手）。"""
    client = MCPClient(name="test", command="fake-cmd")
    fake_transport = MagicMock()
    fake_transport.is_connected = True
    client._transport = fake_transport

    client.connect()

    assert client.connected is True
    fake_transport.connect.assert_called_once()


def test_mcp_client_list_tools_delegates_to_transport():
    """list_tools 委托给 transport.send_request。"""
    client = MCPClient(name="t", command="fake")
    fake_transport = MagicMock()
    fake_transport.send_request.return_value = {
        "tools": [{"name": "read"}, {"name": "write"}],
    }
    client._transport = fake_transport

    tools = client.list_tools()
    assert len(tools) == 2
    fake_transport.send_request.assert_called_once_with("tools/list", {})


def test_mcp_client_call_tool_delegates_to_transport():
    """call_tool 委托给 transport.send_request。"""
    client = MCPClient(name="t", command="fake")
    fake_transport = MagicMock()
    fake_transport.send_request.return_value = {"content": "ok"}
    client._transport = fake_transport

    result = client.call_tool("read", {"path": "/x"})
    assert result == {"content": "ok"}
    fake_transport.send_request.assert_called_once_with(
        "tools/call", {"name": "read", "arguments": {"path": "/x"}},
    )


def test_mcp_client_close_calls_transport_close():
    """close 调用 transport.close。"""
    client = MCPClient(name="t", command="fake")
    fake_transport = MagicMock()
    client._transport = fake_transport
    client._connected = True

    client.close()
    fake_transport.close.assert_called_once()
    assert client.connected is False


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------

def test_manager_connect_all_with_mock():
    """connect_all 连接配置的所有 server。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True

    with patch("agent.mcp_client.MCPClient", return_value=fake_client):
        manager.connect_all({
            "server1": {"command": "fake", "args": []},
        })

    assert "server1" in manager.servers
    fake_client.connect.assert_called_once()


def test_manager_connect_all_passes_http_config():
    """connect_all 把 HTTP 字段透传给 MCPClient。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True

    with patch("agent.mcp_client.MCPClient", return_value=fake_client) as mc:
        manager.connect_all({
            "github": {
                "url": "https://api.x.com/v1",
                "headers": {"X-A": "b"},
                "oauth": {"token_url": "t"},
            },
        })

    # 验证 MCPClient 被以 HTTP 参数构造
    _, kwargs = mc.call_args
    assert kwargs["url"] == "https://api.x.com/v1"
    assert kwargs["headers"] == {"X-A": "b"}
    assert kwargs["oauth"] == {"token_url": "t"}


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
# HTTPTransport（preflight / OAuth 用 mock）
# ---------------------------------------------------------------------------

def test_http_transport_preflight_rejects_html():
    """普通 HTML 网页 URL 3s 内被拒绝。"""
    t = HTTPTransport(url="https://example.com/")
    # 模拟 session 返回 HTML
    fake_session = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    fake_response.status_code = 200
    fake_session.head.return_value = fake_response
    fake_session.get.return_value = fake_response
    t._session = fake_session

    ok, reason = t._preflight()
    assert ok is False
    assert "HTML" in reason or "MCP" in reason


def test_http_transport_preflight_accepts_json():
    """合法 MCP server（返回 JSON）通过预检。"""
    t = HTTPTransport(url="https://api.github-mcp.com/v1")
    fake_session = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "application/json"}
    fake_response.status_code = 200
    fake_session.head.return_value = fake_response
    t._session = fake_session

    ok, _ = t._preflight()
    assert ok is True


def test_http_transport_preflight_accepts_405():
    """405 也算端点存活。"""
    t = HTTPTransport(url="https://api.x.com/mcp")
    fake_session = MagicMock()
    head_resp = MagicMock()
    head_resp.headers = {"Content-Type": ""}
    head_resp.status_code = 405
    fake_session.head.return_value = head_resp
    t._session = fake_session

    get_resp = MagicMock()
    get_resp.headers = {"Content-Type": ""}
    get_resp.status_code = 405
    fake_session.get.return_value = get_resp

    ok, reason = t._preflight()
    assert ok is True
    assert "405" in reason


def test_http_transport_preflight_reports_401():
    """401 报告需要 OAuth。"""
    t = HTTPTransport(url="https://api.x.com/mcp")
    fake_session = MagicMock()
    head_resp = MagicMock()
    head_resp.headers = {"Content-Type": ""}
    head_resp.status_code = 401
    fake_session.head.return_value = head_resp
    t._session = fake_session

    get_resp = MagicMock()
    get_resp.headers = {"Content-Type": ""}
    get_resp.status_code = 401
    fake_session.get.return_value = get_resp

    ok, reason = t._preflight()
    assert ok is False
    assert "401" in reason


def test_http_transport_parses_sse_response():
    """SSE 响应能提取最后一个 result。"""
    t = HTTPTransport(url="https://x")
    sse_text = (
        "event: message\n"
        'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n'
        "\n"
    )
    result = t._parse_sse_response(sse_text)
    assert result == {"tools": []}


def test_http_transport_oauth_refresh(monkeypatch):
    """OAuth 用 refresh_token 换 access_token。"""
    t = HTTPTransport(
        url="https://x",
        oauth_config={
            "token_url": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "sec",
            "refresh_token": "rt",
        },
    )
    # mock requests.post
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "access_token": "new-token",
        "expires_in": 3600,
    }
    import sys
    fake_requests = MagicMock()
    fake_requests.post.return_value = fake_response
    with patch.dict(sys.modules, {"requests": fake_requests}):
        t._refresh_access_token()

    assert t._access_token == "new-token"
    assert t._token_expires_at > 0


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


def test_mcp_toolset_in_resolve():
    """mcp toolset 能被解析。"""
    from toolsets import resolve_toolset
    assert resolve_toolset("mcp") == []
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
