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
    SSETransport, WebSocketTransport,
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
    # 模拟 httpx.Client 返回 HTML
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    fake_response.status_code = 200
    fake_client.head.return_value = fake_response
    fake_client.get.return_value = fake_response
    t._client = fake_client

    ok, reason = t._preflight()
    assert ok is False
    assert "HTML" in reason or "MCP" in reason


def test_http_transport_preflight_accepts_json():
    """合法 MCP server（返回 JSON）通过预检。"""
    t = HTTPTransport(url="https://api.github-mcp.com/v1")
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "application/json"}
    fake_response.status_code = 200
    fake_client.head.return_value = fake_response
    t._client = fake_client

    ok, _ = t._preflight()
    assert ok is True


def test_http_transport_preflight_accepts_405():
    """405 也算端点存活。"""
    t = HTTPTransport(url="https://api.x.com/mcp")
    fake_client = MagicMock()
    head_resp = MagicMock()
    head_resp.headers = {"Content-Type": ""}
    head_resp.status_code = 405
    fake_client.head.return_value = head_resp
    t._client = fake_client

    get_resp = MagicMock()
    get_resp.headers = {"Content-Type": ""}
    get_resp.status_code = 405
    fake_client.get.return_value = get_resp

    ok, reason = t._preflight()
    assert ok is True
    assert "405" in reason


def test_http_transport_preflight_reports_401():
    """401 报告需要 OAuth。"""
    t = HTTPTransport(url="https://api.x.com/mcp")
    fake_client = MagicMock()
    head_resp = MagicMock()
    head_resp.headers = {"Content-Type": ""}
    head_resp.status_code = 401
    fake_client.head.return_value = head_resp
    t._client = fake_client

    get_resp = MagicMock()
    get_resp.headers = {"Content-Type": ""}
    get_resp.status_code = 401
    fake_client.get.return_value = get_resp

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
    """OAuth 用 refresh_token 换 access_token（httpx）。"""
    t = HTTPTransport(
        url="https://x",
        oauth_config={
            "token_url": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "sec",
            "refresh_token": "rt",
        },
    )
    # mock httpx.post
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "access_token": "new-token",
        "expires_in": 3600,
    }
    import sys
    fake_httpx = MagicMock()
    fake_httpx.post.return_value = fake_response
    # 保留 httpx.Client 等其他属性（避免 connect 时报错）
    real_httpx = __import__("httpx")
    fake_httpx.Client = real_httpx.Client
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
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


# ===========================================================================
# Phase 5: 新增 transport 测试
# ===========================================================================

# ---------------------------------------------------------------------------
# Transport 抽象基类
# ---------------------------------------------------------------------------

def test_mcp_transport_abc_cannot_instantiate():
    """MCPTransport 是抽象基类，不能直接实例化。"""
    with pytest.raises(TypeError):
        MCPTransport()


def test_all_transports_inherit_mcp_transport():
    """所有 transport 都继承 MCPTransport。"""
    assert issubclass(StdioTransport, MCPTransport)
    assert issubclass(HTTPTransport, MCPTransport)
    assert issubclass(SSETransport, MCPTransport)
    assert issubclass(WebSocketTransport, MCPTransport)


# ---------------------------------------------------------------------------
# MCPClient transport 解析（显式 transport 字段）
# ---------------------------------------------------------------------------

def test_mcp_client_explicit_stdio_transport():
    """显式 transport=stdio → StdioTransport。"""
    client = MCPClient(name="t", transport="stdio", command="fake-cmd")
    assert isinstance(client._transport, StdioTransport)


def test_mcp_client_explicit_http_transport():
    """显式 transport=http → HTTPTransport（无需 flag，向后兼容）。"""
    client = MCPClient(name="t", transport="http", url="https://x.com/mcp")
    assert isinstance(client._transport, HTTPTransport)


def test_mcp_client_explicit_sse_requires_flag():
    """显式 transport=sse 需要 mcp_http_transport flag。"""
    # flag 关 → ValueError
    with pytest.raises(ValueError, match="mcp_http_transport"):
        MCPClient(name="t", transport="sse", url="https://x.com/sse")

    # flag 开 → SSETransport
    config = {"features": {"mcp_http_transport": {"enabled": True}}}
    client = MCPClient(
        name="t", transport="sse", url="https://x.com/sse", config=config,
    )
    assert isinstance(client._transport, SSETransport)


def test_mcp_client_explicit_websocket_requires_flag():
    """显式 transport=websocket 需要 mcp_websocket_transport flag。"""
    # flag 关 → ValueError
    with pytest.raises(ValueError, match="mcp_websocket_transport"):
        MCPClient(name="t", transport="websocket", url="wss://x.com/ws")

    # flag 开 → WebSocketTransport
    config = {"features": {"mcp_websocket_transport": {"enabled": True}}}
    client = MCPClient(
        name="t", transport="websocket", url="wss://x.com/ws", config=config,
    )
    assert isinstance(client._transport, WebSocketTransport)


def test_mcp_client_ws_url_scheme_auto_detect():
    """wss:// URL → websocket transport（需 flag）。"""
    config = {"features": {"mcp_websocket_transport": {"enabled": True}}}
    client = MCPClient(name="t", url="wss://x.com/ws", config=config)
    assert isinstance(client._transport, WebSocketTransport)

    # ws:// 同理
    client2 = MCPClient(name="t2", url="ws://x.com/ws", config=config)
    assert isinstance(client2._transport, WebSocketTransport)


def test_mcp_client_ws_url_without_flag_raises():
    """wss:// URL 但 flag 关 → ValueError。"""
    with pytest.raises(ValueError, match="mcp_websocket_transport"):
        MCPClient(name="t", url="wss://x.com/ws")


def test_mcp_client_unknown_transport_raises():
    """未知 transport 类型 → ValueError。"""
    with pytest.raises(ValueError, match="未知 transport"):
        MCPClient(name="t", transport="ftp", url="ftp://x.com")


def test_mcp_client_no_transport_url_command_raises():
    """没 transport/url/command → ValueError。"""
    with pytest.raises(ValueError, match="必须配"):
        MCPClient(name="t")


# ---------------------------------------------------------------------------
# SSETransport 测试
# ---------------------------------------------------------------------------

def test_sse_transport_preflight_accepts_event_stream():
    """SSE 端点（返回 event-stream）通过预检。"""
    t = SSETransport(url="https://mcp.example.com/sse")
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "text/event-stream"}
    fake_response.status_code = 200
    fake_client.get.return_value = fake_response
    t._client = fake_client

    ok, _ = t._preflight()
    assert ok is True


def test_sse_transport_preflight_rejects_html():
    """SSE 预检拒绝 HTML。"""
    t = SSETransport(url="https://example.com/")
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.headers = {"Content-Type": "text/html"}
    fake_response.status_code = 200
    fake_client.get.return_value = fake_response
    t._client = fake_client

    ok, reason = t._preflight()
    assert ok is False


def test_sse_transport_parses_sse_response():
    """SSETransport 能解析 SSE 流。"""
    t = SSETransport(url="https://x")
    sse_text = (
        'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
    )
    result = t._parse_sse_response(sse_text)
    assert result == {"tools": []}


def test_sse_transport_oauth_refresh():
    """SSETransport OAuth refresh（httpx）。"""
    t = SSETransport(
        url="https://x",
        oauth_config={
            "token_url": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "sec",
            "refresh_token": "rt",
        },
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "access_token": "sse-token",
        "expires_in": 3600,
    }
    fake_httpx = MagicMock()
    fake_httpx.post.return_value = fake_response
    real_httpx = __import__("httpx")
    fake_httpx.Client = real_httpx.Client
    import sys
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        t._refresh_access_token()

    assert t._access_token == "sse-token"


# ---------------------------------------------------------------------------
# WebSocketTransport 测试（不连真实 server）
# ---------------------------------------------------------------------------

def test_websocket_transport_init():
    """WebSocketTransport 初始化。"""
    t = WebSocketTransport(url="wss://x.com/ws")
    assert t.url == "wss://x.com/ws"
    assert t._ws is None
    assert t._loop is None
    assert t.is_connected is False


def test_websocket_transport_send_request_not_connected():
    """未连接时 send_request 抛 RuntimeError。"""
    t = WebSocketTransport(url="wss://x.com/ws")
    with pytest.raises(RuntimeError, match="未建立"):
        t.send_request("tools/list", {})


def test_websocket_transport_close_when_not_connected():
    """未连接时 close 不报错。"""
    t = WebSocketTransport(url="wss://x.com/ws")
    t.close()  # 不抛
    assert t.is_connected is False


def test_websocket_transport_send_notification_when_not_connected():
    """未连接时 send_notification 不报错（静默跳过）。"""
    t = WebSocketTransport(url="wss://x.com/ws")
    t.send_notification("test/method", {})  # 不抛


# ---------------------------------------------------------------------------
# Feature flag 集成测试
# ---------------------------------------------------------------------------

def test_mcp_manager_connect_all_passes_config():
    """connect_all 把 app_config 透传给 MCPClient。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True

    with patch("agent.mcp_client.MCPClient", return_value=fake_client) as mc:
        manager.connect_all(
            {"server1": {"command": "fake", "args": []}},
            app_config={"features": {"mcp_http_transport": {"enabled": True}}},
        )

    _, kwargs = mc.call_args
    assert kwargs["config"] == {"features": {"mcp_http_transport": {"enabled": True}}}


def test_mcp_manager_connect_all_skips_sse_without_flag(caplog):
    """connect_all 遇到 SSE 但 flag 关 → 跳过 + log warning。"""
    manager = MCPManager()
    manager.connect_all({
        "sse-server": {
            "transport": "sse",
            "url": "https://mcp.example.com/sse",
        },
    })
    # 没连上
    assert "sse-server" not in manager.servers


def test_mcp_manager_connect_all_skips_websocket_without_flag():
    """connect_all 遇到 WebSocket 但 flag 关 → 跳过。"""
    manager = MCPManager()
    manager.connect_all({
        "ws-server": {
            "transport": "websocket",
            "url": "wss://mcp.example.com/ws",
        },
    })
    assert "ws-server" not in manager.servers


def test_mcp_manager_connect_all_connects_sse_with_flag():
    """connect_all 遇到 SSE 且 flag 开 → 连接（mock transport）。"""
    manager = MCPManager()
    fake_client = MagicMock()
    fake_client.connected = True

    config = {"features": {"mcp_http_transport": {"enabled": True}}}
    with patch("agent.mcp_client.MCPClient", return_value=fake_client):
        manager.connect_all(
            {"sse-server": {"transport": "sse", "url": "https://x.com/sse"}},
            app_config=config,
        )
    assert "sse-server" in manager.servers


# ---------------------------------------------------------------------------
# .mcp.json 配置加载（transport 字段）
# ---------------------------------------------------------------------------

def test_load_mcp_config_with_transport_field(tmp_path):
    """配置含 transport 字段时正确加载。"""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": ["server"],
            },
            "remote": {
                "transport": "sse",
                "url": "https://mcp.example.com/sse",
            },
            "realtime": {
                "transport": "websocket",
                "url": "wss://mcp.example.com/ws",
            },
        }
    }), encoding="utf-8")

    servers = load_mcp_config(cfg)
    assert servers["fs"]["transport"] == "stdio"
    assert servers["remote"]["transport"] == "sse"
    assert servers["realtime"]["transport"] == "websocket"


def test_load_mcp_config_backward_compat_no_transport(tmp_path):
    """旧配置（无 transport 字段）仍能加载。"""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["server"],
            },
            "github": {
                "url": "https://api.github-mcp.com/v1",
            },
        }
    }), encoding="utf-8")

    servers = load_mcp_config(cfg)
    assert "transport" not in servers["fs"]
    assert servers["fs"]["command"] == "npx"
    assert servers["github"]["url"] == "https://api.github-mcp.com/v1"


# ===== 项目级 .mcp.json 首连审批 =====

class TestProjectMcpApproval:
    def _setup(self, tmp_path, monkeypatch, approved_keys=None):
        """构造隔离环境：假 home + 项目目录（含 .mcp.json）。"""
        import json as _json
        home = tmp_path / "home"
        home.mkdir()
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / ".mcp.json").write_text(_json.dumps({
            "mcpServers": {"evil": {"command": "run-evil"}, "ok": {"command": "run-ok"}}
        }), encoding="utf-8")
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr("agent.workspace_context.get_workspace_cwd", lambda: str(proj))
        if approved_keys is None:
            approved_keys = []
        # settings.json 预置已批准
        settings = home / "settings.json"
        settings.write_text(_json.dumps({
            "mcp": {"approved_project_servers": approved_keys}
        }), encoding="utf-8")
        return proj

    def test_unapproved_project_server_skipped(self, tmp_path, monkeypatch):
        """未批准且 callback 拒绝 → 项目 server 不连接（fail-closed）。"""
        self._setup(tmp_path, monkeypatch)
        from tools import mcp_tool
        from agent.mcp_client import get_mcp_manager
        connected = []
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: connected.extend((config or {}).keys()),
        )
        count = mcp_tool.initialize_mcp(approval_callback=lambda n, d: False)
        assert "evil" not in connected
        assert "ok" not in connected

    def test_approved_via_callback_connects_and_persists(self, tmp_path, monkeypatch):
        """callback 同意 → 连接 + 持久化，第二次不再问。"""
        proj = self._setup(tmp_path, monkeypatch)
        from tools import mcp_tool
        from agent.mcp_client import get_mcp_manager
        connected = []
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: connected.extend((config or {}).keys()),
        )
        asked = []
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: asked.append(n) or True)
        assert set(asked) == {"evil", "ok"}
        assert set(connected) == {"evil", "ok"}
        # 持久化生效：再跑一次不再询问
        asked.clear()
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: None,
        )
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: asked.append(n) or True)
        assert asked == []

    def test_no_callback_fail_closed(self, tmp_path, monkeypatch):
        """无 callback（daemon/team worker 等非交互场景）→ 项目 server 全跳过。"""
        self._setup(tmp_path, monkeypatch)
        from tools import mcp_tool
        from agent.mcp_client import get_mcp_manager
        connected = []
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: connected.extend((config or {}).keys()),
        )
        mcp_tool.initialize_mcp()  # 不传 callback
        assert connected == []


# ===== 审批 key 内容指纹 =====

class TestApprovalKeyFingerprint:
    def test_fingerprint_stable_and_config_sensitive(self):
        from agent.settings import server_cfg_fingerprint
        a = server_cfg_fingerprint({"command": "run-a", "args": [1, 2]})
        b = server_cfg_fingerprint({"args": [1, 2], "command": "run-a"})  # 键序无关
        c = server_cfg_fingerprint({"command": "run-evil", "args": [1, 2]})
        assert a == b and a != c and len(a) == 8

    def test_approval_key_formats(self):
        from agent.settings import mcp_approval_key
        # 无 cfg：老格式（兼容内联场景的 server 名命名空间）
        assert mcp_approval_key("/p", "foo") == "/p::foo"
        # 有 cfg：带指纹
        k = mcp_approval_key("/p", "foo", {"command": "x"})
        assert k.startswith("/p::foo::") and len(k) == len("/p::foo::") + 8

    def test_project_server_key_changes_with_config(self):
        """项目 .mcp.json server 配置变了 → 审批 key 变 → 重新询问。"""
        from agent.settings import mcp_approval_key
        k1 = mcp_approval_key("/p", "evil", {"command": "run-ok"})
        k2 = mcp_approval_key("/p", "evil", {"command": "run-evil"})
        assert k1 != k2

    def test_initialize_mcp_uses_fingerprinted_key(self, tmp_path, monkeypatch):
        """端到端：批准时持久化的 key 带指纹；配置漂移后再启动会再问。"""
        import json as _json
        from tools import mcp_tool
        from agent.mcp_client import get_mcp_manager

        proj = self._setup_proj(tmp_path, monkeypatch)
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: None,
        )

        # ① 第一次 callback 拒绝 → settings 无 key
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: False)
        approved = self._load_approved(tmp_path)
        assert approved == []

        # ② callback 同意 → settings 里存在含 "::" 两次的指纹 key
        asked = []
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: asked.append(n) or True)
        approved = self._load_approved(tmp_path)
        assert len(approved) == 2
        for k in approved:
            assert k.count("::") == 2 and len(k.rsplit("::", 1)[1]) == 8

        # ③ 改 .mcp.json 的 command → approved key 不再匹配 → 又被问（asked 再 +1）
        (proj / ".mcp.json").write_text(_json.dumps({
            "mcpServers": {"evil": {"command": "run-evil-2"}, "ok": {"command": "run-ok-2"}}
        }), encoding="utf-8")
        asked.clear()
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: asked.append(n) or True)
        assert set(asked) == {"evil", "ok"}

    def _setup_proj(self, tmp_path, monkeypatch):
        """复用 TestProjectMcpApproval._setup 的隔离构造。"""
        return TestProjectMcpApproval._setup(self, tmp_path, monkeypatch)

    def _load_approved(self, tmp_path):
        import json as _json
        data = _json.loads(
            (tmp_path / "home" / "settings.json").read_text(encoding="utf-8")
        )
        return data.get("mcp", {}).get("approved_project_servers", [])


# ===== 项目级 agent 内联 MCP 首连审批 =====

class TestInlineMcpApproval:
    def _setup_inline(self, tmp_path, monkeypatch, approved_keys=None):
        """隔离环境：假 home + 项目目录（agent .md 带内联 MCP，无 .mcp.json）。"""
        import json as _json
        home = tmp_path / "home"
        home.mkdir()
        proj = tmp_path / "proj"
        agents_dir = proj / ".omnimate" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "helper.md").write_text(
            "---\nname: helper\ndescription: t\n"
            "inlineMcpServers:\n  evil: {command: run-evil}\n---\nbody\n",
            encoding="utf-8")
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr(
            "agent.workspace_context.get_workspace_cwd", lambda: str(proj))
        settings = home / "settings.json"
        settings.write_text(_json.dumps({
            "mcp": {"approved_project_servers": approved_keys or []}
        }), encoding="utf-8")
        return proj

    def _approved_in_settings(self, tmp_path):
        import json as _json
        data = _json.loads(
            (tmp_path / "home" / "settings.json").read_text(encoding="utf-8"))
        return data.get("mcp", {}).get("approved_project_servers", [])

    def test_startup_asks_and_persists(self, tmp_path, monkeypatch):
        """项目 agent .md 的内联 server 启动时过审批（同 .mcp.json 语义）。"""
        from tools import mcp_tool
        from agent.mcp_client import get_mcp_manager
        proj = self._setup_inline(tmp_path, monkeypatch)
        # ① callback 拒 → 持久化无 key
        monkeypatch.setattr(
            get_mcp_manager(), "connect_all",
            lambda config=None, **kw: None,
        )
        asked = []
        mcp_tool.initialize_mcp(
            approval_callback=lambda n, d: asked.append(n) or False)
        assert asked == ["(agent 内联) evil"]
        assert self._approved_in_settings(tmp_path) == []
        # ② callback 允 → settings 有指纹 key；再跑不再问
        mcp_tool.initialize_mcp(
            approval_callback=lambda n, d: asked.append(n) or True)
        approved = self._approved_in_settings(tmp_path)
        assert len(approved) == 1 and "evil" in approved[0] and "agent-mcp" in approved[0]
        asked.clear()
        mcp_tool.initialize_mcp(approval_callback=lambda n, d: asked.append(n) or True)
        assert asked == []

    def test_spawn_skips_unapproved(self, tmp_path, monkeypatch):
        """delegate spawn 时未批准的项目级内联 server → fail-closed 跳过。"""
        from agent.agent_defs import AgentDefinition
        from tools.delegate_tool import inline_mcp_spawn_allowed

        proj = self._setup_inline(tmp_path, monkeypatch)
        ad = AgentDefinition(
            name="helper", description="t",
            inline_mcp_servers={"evil": {"command": "run-evil"}},
        )
        ad.source = "project"
        cfg = {"command": "run-evil"}

        # ① 项目来源 + 无批准 → 拒（fail-closed）
        assert inline_mcp_spawn_allowed(ad, "evil", cfg) is False
        # ② 写入同源 key（cwd resolve lower + agent-mcp:: 前缀 + 指纹）→ 放行
        from agent.settings import mcp_approval_key, persist_project_mcp_approval
        _pk = str(Path(proj).resolve()).lower()
        persist_project_mcp_approval(
            mcp_approval_key(_pk, "agent-mcp::evil", cfg))
        assert inline_mcp_spawn_allowed(ad, "evil", cfg) is True
        # ③ 用户级/CLI 来源（缺省按 user）不拦
        ad2 = AgentDefinition(name="u", inline_mcp_servers={"evil": cfg})
        assert ad2.source == "user"
        assert inline_mcp_spawn_allowed(ad2, "evil", cfg) is True


# ---------------------------------------------------------------------------
# 真实子进程回归（2026-08-17 对话测试逮到 reader 线程 _connected 鸡蛋问题）
# ---------------------------------------------------------------------------

# 最小 stdio JSON-RPC server（纯 ASCII 输出，避免编码干扰）
_REAL_SERVER_SRC = r'''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid = req.get("id")
    if rid is None:
        continue
    m = req.get("method", "")
    if m == "initialize":
        out = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
               "serverInfo": {"name": "t", "version": "0"}}
    elif m == "tools/list":
        out = {"tools": [{"name": "echo", "description": "echo",
                          "inputSchema": {"type": "object", "properties": {}}}]}
    elif m == "tools/call":
        out = {"content": [{"type": "text", "text": "ECHO-REAL-OK"}], "isError": False}
    else:
        out = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": out}) + "\n")
    sys.stdout.flush()
'''


class TestStdioRealSubprocess:
    def test_handshake_list_call_with_real_subprocess(self, tmp_path):
        """真实 spawn：握手 + tools/list + tools/call 全链路。

        历史回归：_reader_loop 曾以 `while self._connected` 为条件，而
        _connected 在握手成功后才置 True → 握手期间 reader 立即退出 →
        60s 超时 → stdio MCP 生产全断（mock 测试看不见，必须真 spawn）。
        """
        import sys as _sys
        import time
        server = tmp_path / "srv.py"
        server.write_text(_REAL_SERVER_SRC, encoding="utf-8")
        client = MCPClient("t-real", command=_sys.executable, args=[str(server)])
        # 超时缩短：真坏时 60s 太慢
        client._transport._response_timeout = 10.0
        t0 = time.time()
        client.connect()
        assert time.time() - t0 < 10, "握手应在秒级完成（曾回归为 60s 超时）"
        tools = client.list_tools()
        assert tools and tools[0]["name"] == "echo"
        result = client.call_tool("echo", {})
        assert result["content"][0]["text"] == "ECHO-REAL-OK"
        client.close()

    def test_reader_survives_non_utf8_line(self, tmp_path):
        """server 输出坏字节行后连接仍可用（errors=replace 容错）。"""
        import sys as _sys
        # 全二进制读写（避免文本层/buffer 混用打乱顺序）；先吐一行坏字节
        # （模拟 server 往 stdout 打 GBK/二进制日志）
        src = (
            "import sys, json\n"
            "sys.stdout.buffer.write(b'\\xbb\\xdc bad bytes\\n')\n"
            "sys.stdout.buffer.flush()\n"
            "for line in sys.stdin.buffer:\n"
            "    line = line.strip()\n"
            "    if not line:\n"
            "        continue\n"
            "    req = json.loads(line.decode('utf-8'))\n"
            "    rid = req.get('id')\n"
            "    if rid is None:\n"
            "        continue\n"
            "    m = req.get('method', '')\n"
            "    if m == 'initialize':\n"
            "        out = {'protocolVersion': '2024-11-05', 'capabilities': {},\n"
            "               'serverInfo': {'name': 't', 'version': '0'}}\n"
            "    elif m == 'tools/list':\n"
            "        out = {'tools': [{'name': 'echo', 'description': 'echo',\n"
            "                          'inputSchema': {'type': 'object', 'properties': {}}}]}\n"
            "    else:\n"
            "        out = {}\n"
            "    data = json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': out})\n"
            "    sys.stdout.buffer.write(data.encode('utf-8') + b'\\n')\n"
            "    sys.stdout.buffer.flush()\n"
        )
        server = tmp_path / "srv_bad.py"
        server.write_text(src, encoding="utf-8")
        client = MCPClient("t-bad", command=_sys.executable, args=[str(server)])
        client._transport._response_timeout = 10.0
        client.connect()
        tools = client.list_tools()
        assert tools and tools[0]["name"] == "echo"
        client.close()
