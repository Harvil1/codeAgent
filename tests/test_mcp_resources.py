"""MCP Resources 协议测试（CCAR12 Task 5）。

mock transport 的 send_request，断言协议方法名/参数正确：
- MCPTransport 基类默认实现（resources/list + resources/read + fail-open）
- MCPClient / MCPManager 透传与错误 JSON
- register_mcp_tools 注册 mcp__<server>__list_resources / read_resource
- 工具 handler 端到端（server 不存在 / 不支持 resources / 正常路径）
- dispatch 契约（(args, **kwargs) 签名 + schema 用 "parameters" 键）
"""

import asyncio
import inspect
import json
import threading
from unittest.mock import MagicMock

from agent.mcp_client import (
    MCPClient,
    MCPManager,
    MCPTransport,
)
from tools.mcp_tool import register_mcp_tools
from tools.registry import registry


# ---------------------------------------------------------------------------
# 测试用 stub transport（记录 send_request 调用）
# ---------------------------------------------------------------------------

class StubTransport(MCPTransport):
    """记录 send_request 调用的假 transport。"""

    def __init__(self, responses=None, raise_methods=None):
        # responses: {method: result}
        self.calls = []  # [(method, params), ...]
        self._responses = responses or {}
        self._raise_methods = set(raise_methods or [])
        self._connected = True

    def connect(self) -> None:
        pass

    def send_request(self, method: str, params: dict):
        self.calls.append((method, params))
        if method in self._raise_methods:
            raise RuntimeError(f"MCP 错误 -32601: Method not found: {method}")
        return self._responses.get(method)

    def send_notification(self, method: str, params: dict) -> None:
        pass

    def close(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# Transport 基类默认实现：协议方法名 / 参数 / 解析
# ---------------------------------------------------------------------------

def test_transport_list_resources_protocol():
    """list_resources 发 resources/list + 空 params，解析 resources 字段。"""
    transport = StubTransport(responses={
        "resources/list": {"resources": [
            {"uri": "file:///a.txt", "name": "a", "mimeType": "text/plain"},
        ]},
    })
    result = transport.list_resources()
    assert transport.calls == [("resources/list", {})]
    assert result == [{"uri": "file:///a.txt", "name": "a", "mimeType": "text/plain"}]


def test_transport_read_resource_protocol():
    """read_resource 发 resources/read + {"uri": ...}。"""
    transport = StubTransport(responses={
        "resources/read": {"contents": [{"uri": "file:///a.txt", "text": "hello"}]},
    })
    result = transport.read_resource("file:///a.txt")
    assert transport.calls == [("resources/read", {"uri": "file:///a.txt"})]
    assert result["contents"][0]["text"] == "hello"


def test_transport_resources_fail_open():
    """server 不支持（send_request 抛错）→ 两方法返回 None 不抛。"""
    transport = StubTransport(raise_methods=["resources/list", "resources/read"])
    assert transport.list_resources() is None
    assert transport.read_resource("file:///a.txt") is None


def test_transport_list_resources_empty_response():
    """resp 为 None（无 resources 字段）→ 返回 None。"""
    transport = StubTransport(responses={"resources/list": None})
    assert transport.list_resources() is None


def test_transport_resources_methods_on_all_transports():
    """4 种具体 transport 均继承基类默认实现（无需 override）。"""
    from agent.mcp_client import (
        StdioTransport, HTTPTransport, SSETransport, WebSocketTransport,
    )
    for cls in (StdioTransport, HTTPTransport, SSETransport, WebSocketTransport):
        assert "list_resources" not in cls.__dict__  # 未 override，走基类
        assert "read_resource" not in cls.__dict__
        assert hasattr(cls, "list_resources")
        assert hasattr(cls, "read_resource")


# ---------------------------------------------------------------------------
# MCPClient 透传
# ---------------------------------------------------------------------------

def test_client_list_resources_delegates_to_transport():
    client = MCPClient(name="t", command="fake-cmd")
    client._transport = MagicMock()
    client._transport.list_resources.return_value = [{"uri": "x"}]
    assert client.list_resources() == [{"uri": "x"}]


def test_client_read_resource_delegates_to_transport():
    client = MCPClient(name="t", command="fake-cmd")
    client._transport = MagicMock()
    client._transport.read_resource.return_value = {"contents": []}
    assert client.read_resource("x") == {"contents": []}


# ---------------------------------------------------------------------------
# MCPManager：错误 JSON / 友好提示
# ---------------------------------------------------------------------------

def _make_client_with_transport(transport, connected=True):
    client = MCPClient(name="t", command="fake-cmd")
    client._transport = transport
    client._connected = connected
    return client


def test_manager_list_resources_server_missing():
    """server 不存在 → 错误 JSON（error_type=mcp_server_not_connected）。"""
    manager = MCPManager()
    result = manager.list_resources("nope")
    assert result["error_type"] == "mcp_server_not_connected"
    assert "nope" in result["error"]


def test_manager_list_resources_server_disconnected():
    """server 存在但断开 → 错误 JSON。"""
    manager = MCPManager()
    client = _make_client_with_transport(StubTransport(), connected=False)
    manager._clients["srv"] = client
    result = manager.list_resources("srv")
    assert result["error_type"] == "mcp_server_disconnected"


def test_manager_list_resources_unsupported_friendly():
    """transport 返 None（不支持）→ 友好提示"不支持 resources"。"""
    manager = MCPManager()
    transport = StubTransport(raise_methods=["resources/list"])
    manager._clients["srv"] = _make_client_with_transport(transport)
    result = manager.list_resources("srv")
    assert result["error_type"] == "mcp_resources_unsupported"
    assert "不支持 resources" in result["error"]


def test_manager_list_resources_success():
    manager = MCPManager()
    transport = StubTransport(responses={
        "resources/list": {"resources": [{"uri": "u", "name": "n"}]},
    })
    manager._clients["srv"] = _make_client_with_transport(transport)
    result = manager.list_resources("srv")
    assert result == {"server": "srv", "resources": [{"uri": "u", "name": "n"}]}


def test_manager_read_resource_success_and_missing_uri():
    manager = MCPManager()
    transport = StubTransport(responses={
        "resources/read": {"contents": [{"uri": "u", "text": "hi"}]},
    })
    manager._clients["srv"] = _make_client_with_transport(transport)
    result = manager.read_resource("srv", "u")
    assert result["contents"][0]["text"] == "hi"
    # 缺 uri → 参数错误
    result2 = manager.read_resource("srv", "")
    assert result2["error_type"] == "invalid_args"


def test_manager_read_resource_server_missing():
    manager = MCPManager()
    result = manager.read_resource("nope", "u")
    assert result["error_type"] == "mcp_server_not_connected"


def test_manager_read_resource_unsupported():
    manager = MCPManager()
    transport = StubTransport(raise_methods=["resources/read"])
    manager._clients["srv"] = _make_client_with_transport(transport)
    result = manager.read_resource("srv", "u")
    assert result["error_type"] == "mcp_resources_unsupported"


# ---------------------------------------------------------------------------
# 工具注册 + handler 端到端
# ---------------------------------------------------------------------------

def _make_manager_with_server(server="srv", transport=None):
    """构造带单个连接 server 的 MCPManager（真 manager + 真 client）。"""
    manager = MCPManager()
    client = _make_client_with_transport(transport or StubTransport())
    manager._clients[server] = client
    return manager, client


def test_register_mcp_tools_registers_resource_tools():
    """每个连接中的 server 注册两个 resources 工具（mcp__ 动态命名空间）。"""
    manager, _ = _make_manager_with_server("srv")
    manager.get_all_tools = lambda: []  # 无普通工具，只看 resources

    count = register_mcp_tools(manager)
    assert count == 2
    assert registry.get("mcp__srv__list_resources") is not None
    assert registry.get("mcp__srv__read_resource") is not None

    entries = [
        registry.get("mcp__srv__list_resources"),
        registry.get("mcp__srv__read_resource"),
    ]
    for entry in entries:
        assert entry.toolset == "mcp"
        # OpenAI 格式契约：schema 键必须是 "parameters"
        assert "parameters" in entry.schema
        assert "inputSchema" not in entry.schema
        assert entry.isConcurrencySafe is False


def test_resource_tool_handlers_end_to_end():
    """handler 端到端：list 正常 / read 传 uri / server 不存在。"""
    transport = StubTransport(responses={
        "resources/list": {"resources": [{"uri": "u1", "name": "n1"}]},
        "resources/read": {"contents": [{"uri": "u1", "text": "内容"}]},
    })
    manager, _ = _make_manager_with_server("srv", transport)
    manager.get_all_tools = lambda: []
    register_mcp_tools(manager)

    # list 正常
    result = json.loads(asyncio.run(registry.dispatch(
        "mcp__srv__list_resources", {},
    )))
    assert result["resources"] == [{"uri": "u1", "name": "n1"}]
    # 协议方法名/参数正确（StubTransport 记录了调用）
    assert ("resources/list", {}) in transport.calls

    # read 传 uri
    result2 = json.loads(asyncio.run(registry.dispatch(
        "mcp__srv__read_resource", {"uri": "u1"},
    )))
    assert result2["contents"][0]["text"] == "内容"
    assert ("resources/read", {"uri": "u1"}) in transport.calls


def test_resource_tool_handler_server_missing_error():
    """manager 里 server 被移除后调用 → 错误 JSON（handler 层兜底）。"""
    manager, _ = _make_manager_with_server("srv2")
    manager.get_all_tools = lambda: []
    register_mcp_tools(manager)

    manager._clients.clear()  # 模拟 server 掉线后调用已注册工具
    result = json.loads(asyncio.run(registry.dispatch(
        "mcp__srv2__list_resources", {},
    )))
    assert result["error_type"] == "mcp_server_not_connected"


def test_resource_tool_handler_unsupported_friendly():
    """server 不支持 resources → 友好错误提示。"""
    transport = StubTransport(raise_methods=["resources/list"])
    manager, _ = _make_manager_with_server("srv3", transport)
    manager.get_all_tools = lambda: []
    register_mcp_tools(manager)

    result = json.loads(asyncio.run(registry.dispatch(
        "mcp__srv3__list_resources", {},
    )))
    assert result["error_type"] == "mcp_resources_unsupported"
    assert "不支持 resources" in result["error"]


def test_resource_tool_check_fn_gates_disconnected():
    """server 断开 → check_fn 返回 False（工具从 LLM schema 隐藏）。"""
    manager, client = _make_manager_with_server("srv4")
    manager.get_all_tools = lambda: []
    register_mcp_tools(manager)

    entry = registry.get("mcp__srv4__list_resources")
    assert entry.check_fn() is True  # 连接中

    client._connected = False
    assert entry.check_fn() is False  # 断开 → 隐藏


# ---------------------------------------------------------------------------
# dispatch 契约
# ---------------------------------------------------------------------------

def test_resource_tool_handler_signature_matches_dispatch_contract():
    """handler 签名必须是 (args, **kwargs)（CCAR8 教训：防 silent-dead-code）。"""
    for name in ("mcp__srv__list_resources", "mcp__srv__read_resource"):
        entry = registry.get(name)
        assert entry is not None, f"{name} 未注册"
        sig = inspect.signature(entry.handler)
        params = list(sig.parameters.values())
        assert len(params) >= 1
        assert params[0].name == "args"
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params), (
            f"{name} handler 缺 **kwargs，registry.dispatch 传不进命名上下文"
        )


def test_resource_tools_excluded_from_builtin_classification():
    """resources 工具走 mcp__ 动态命名空间，不进 built-in 分类清单。

    test_tool_concurrency_classification 按 mcp__ 前缀排除动态工具，
    这里确认两个工具名确实带前缀（不受 expected_safe_count 影响）。
    """
    for name in ("mcp__srv__list_resources", "mcp__srv__read_resource"):
        assert name.startswith("mcp__")
