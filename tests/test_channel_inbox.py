"""ChannelInbox + MCP notification 接口的测试。

覆盖：
- ChannelInbox 落盘 + unconsumed + mark_consumed + format_digest
- fail-open（写盘异常不抛）
- MCPTransport.set_notification_handler 默认 no-op
- StdioTransport reader 线程 dispatch notification 到 handler
"""
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from agent.channel_inbox import ChannelInbox


# ---------------------------------------------------------------------------
# ChannelInbox 基础
# ---------------------------------------------------------------------------

def test_push_and_unconsumed(tmp_path: Path):
    inbox = ChannelInbox(tmp_path)
    mid = inbox.push("feishu", {"text": "hello from feishu"})
    unread = inbox.unconsumed()
    assert len(unread) == 1
    assert unread[0]["id"] == mid
    assert unread[0]["server"] == "feishu"
    assert unread[0]["payload"] == {"text": "hello from feishu"}


def test_unconsumed_ordering(tmp_path: Path):
    """按 ts 升序（先 push 的在前）。"""
    inbox = ChannelInbox(tmp_path)
    inbox.push("srv", {"n": 1})
    inbox.push("srv", {"n": 2})
    inbox.push("srv", {"n": 3})
    unread = inbox.unconsumed()
    assert [u["payload"]["n"] for u in unread] == [1, 2, 3]


def test_mark_consumed_deletes_files(tmp_path: Path):
    inbox = ChannelInbox(tmp_path)
    mid1 = inbox.push("srv", {"n": 1})
    mid2 = inbox.push("srv", {"n": 2})
    inbox.mark_consumed([mid1])
    unread = inbox.unconsumed()
    assert len(unread) == 1
    assert unread[0]["id"] == mid2


def test_push_failopen_on_disk_error(tmp_path: Path, caplog):
    inbox = ChannelInbox(tmp_path)
    inbox._inbox_dir = "/nonexistent"
    with caplog.at_level(logging.WARNING):
        inbox.push("srv", {"x": 1})  # 不抛
    assert any("channel" in r.message.lower() for r in caplog.records)


def test_unconsumed_limit(tmp_path: Path):
    inbox = ChannelInbox(tmp_path)
    for i in range(10):
        inbox.push("srv", {"n": i})
    unread = inbox.unconsumed(limit=3)
    assert len(unread) == 3


def test_format_digest(tmp_path: Path):
    """format_digest 把消息列表格式化为字符串。"""
    inbox = ChannelInbox(tmp_path)
    inbox.push("feishu", {"text": "msg1"})
    inbox.push("slack", {"text": "msg2"})
    unread = inbox.unconsumed()
    digest = inbox.format_digest(unread)
    assert "feishu" in digest
    assert "slack" in digest
    assert "msg1" in digest


def test_format_digest_empty(tmp_path: Path):
    """空列表返回空字符串。"""
    inbox = ChannelInbox(tmp_path)
    assert inbox.format_digest([]) == ""


def test_format_digest_truncates_long_payload(tmp_path: Path):
    """payload 超过 500 字符时截断。"""
    inbox = ChannelInbox(tmp_path)
    long_text = "x" * 800
    inbox.push("srv", {"big": long_text})
    unread = inbox.unconsumed()
    digest = inbox.format_digest(unread)
    assert "..." in digest
    assert len(digest) < 800 + 100  # 截断后总长度受限


# ---------------------------------------------------------------------------
# MCPTransport.set_notification_handler（接口层）
# ---------------------------------------------------------------------------

def test_set_notification_handler_default_noop():
    """默认 set_notification_handler 不抛（向后兼容）。"""
    from agent.mcp_client import HTTPTransport
    t = HTTPTransport(url="https://x.com/mcp")
    # 默认 handler 为 None
    assert t.notification_handler is None
    # 设置 handler 不抛
    t.set_notification_handler(lambda m, p: None)
    assert t.notification_handler is not None


def test_stdio_transport_set_notification_handler():
    """StdioTransport 支持设置 notification handler。"""
    from agent.mcp_client import StdioTransport
    t = StdioTransport("echo")
    assert t.notification_handler is None
    handler_called = []

    def handler(method, params):
        handler_called.append((method, params))

    t.set_notification_handler(handler)
    assert t.notification_handler is handler


# ---------------------------------------------------------------------------
# StdioTransport reader 线程 dispatch notification（集成层）
# ---------------------------------------------------------------------------

def test_stdio_transport_dispatches_notification(tmp_path: Path):
    """StdioTransport reader 循环：收到 notification 时调 handler。

    不真起 MCP server——mock process.stdout 模拟 reader 输入。
    """
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")  # 不会真连
    received = []
    transport.set_notification_handler(
        lambda method, params: received.append((method, params))
    )
    # 模拟 reader 收到 notification（method + 无 id）
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdout.readline.side_effect = [
        json.dumps({"method": "notifications/message", "params": {"data": "hi"}}),
        "",  # EOF → break
    ]
    transport._reader_loop()
    assert len(received) == 1
    assert received[0][0] == "notifications/message"
    assert received[0][1] == {"data": "hi"}


def test_stdio_transport_reader_dispatches_response_to_queue(tmp_path: Path):
    """reader 循环：response（带 id）走 queue，不调 handler。"""
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")
    received = []
    transport.set_notification_handler(
        lambda method, params: received.append((method, params))
    )
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 42, "result": {"tools": []}}),
        "",
    ]
    transport._reader_loop()
    # handler 没被调（response 不走 handler）
    assert received == []
    # response 进了 queue
    data = transport._response_queue.get_nowait()
    assert data["id"] == 42
    assert data["result"] == {"tools": []}


def test_stdio_transport_reader_skips_invalid_json(tmp_path: Path):
    """reader 遇到非 JSON 行（debug 输出）跳过，不抛。"""
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")
    received = []
    transport.set_notification_handler(
        lambda method, params: received.append((method, params))
    )
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdout.readline.side_effect = [
        "this is not json\n",
        json.dumps({"method": "notifications/progress", "params": {"p": 50}}),
        "",
    ]
    transport._reader_loop()
    assert len(received) == 1
    assert received[0][0] == "notifications/progress"


def test_stdio_transport_send_request_uses_queue(tmp_path: Path):
    """send_request 改造后从 response_queue 拿响应（不再直接 readline）。"""
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdin.write = MagicMock()
    transport.process.stdin.flush = MagicMock()
    # 预放一个 response 到 queue
    transport._response_queue.put({
        "jsonrpc": "2.0", "id": 1, "result": {"ok": True},
    })
    result = transport.send_request("tools/list", {})
    assert result == {"ok": True}


def test_stdio_transport_send_request_timeout(tmp_path: Path):
    """queue.get timeout 抛 RuntimeError。"""
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")
    transport._response_timeout = 0.05  # 加速测试
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdin.write = MagicMock()
    transport.process.stdin.flush = MagicMock()
    import pytest
    with pytest.raises(RuntimeError, match="超时"):
        transport.send_request("tools/list", {})


def test_stdio_transport_handler_exception_does_not_crash_reader(tmp_path: Path):
    """handler 抛异常时 reader 不崩，继续读后续消息。"""
    from agent.mcp_client import StdioTransport

    transport = StdioTransport("echo")

    def bad_handler(method, params):
        raise ValueError("handler bug")

    transport.set_notification_handler(bad_handler)
    received_later = []

    # 第二个 handler：handler 抛过后不能换（reader 持有引用），
    # 但我们可以验证 reader 没崩——通过后续消息能继续被尝试 dispatch
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdout.readline.side_effect = [
        json.dumps({"method": "notifications/x", "params": {}}),
        json.dumps({"method": "notifications/y", "params": {}}),
        "",
    ]
    # 不应抛
    transport._reader_loop()
    # 验证：reader 走完了（没卡死也没抛）


# ---------------------------------------------------------------------------
# 端到端：MCP notification → ChannelInbox.push → unconsumed（final fix）
# ---------------------------------------------------------------------------

def test_stdio_transport_to_channel_inbox_end_to_end(tmp_path: Path):
    """端到端：StdioTransport 收 notification → handler → ChannelInbox.push → unconsumed。

    这是 final review Critical bug 的回归测试：之前 handler 没注册导致
    notification 被丢弃、/inbox 永远空。
    """
    from agent.mcp_client import StdioTransport

    inbox = ChannelInbox(tmp_path)
    transport = StdioTransport("echo")
    # 模拟 cli._create_agent 接线：把 inbox.push 包成 handler
    # 注意闭包变量捕获（server_name 用默认参数绑定）
    def _make_handler(sname, inb):
        def _handler(method, params):
            inb.push(sname, {"method": method, **(params or {})})
        return _handler
    transport.set_notification_handler(_make_handler("feishu", inbox))

    # 模拟 reader 线程读 stdout
    transport._connected = True
    transport.process = MagicMock()
    transport.process.poll.return_value = None
    transport.process.stdout.readline.side_effect = [
        json.dumps({
            "method": "notifications/message",
            "params": {"level": "info", "text": "build done"},
        }),
        "",
    ]
    transport._reader_loop()

    # 验证：inbox 里有一条消息
    msgs = inbox.unconsumed()
    assert len(msgs) == 1
    assert msgs[0]["server"] == "feishu"
    payload = msgs[0]["payload"]
    assert payload["method"] == "notifications/message"
    assert payload["text"] == "build done"
    assert payload["level"] == "info"


def test_multiple_servers_route_to_shared_inbox(tmp_path: Path):
    """多 server 接同一个 inbox：handler 闭包捕获 server_name 正确。"""
    from agent.mcp_client import StdioTransport

    inbox = ChannelInbox(tmp_path)

    def _make_handler(sname, inb):
        def _handler(method, params):
            inb.push(sname, {"method": method, **(params or {})})
        return _handler

    # 两个 transport，分别绑不同 server_name
    for srv_name in ["feishu", "slack"]:
        t = StdioTransport("echo")
        t.set_notification_handler(_make_handler(srv_name, inbox))
        t._connected = True
        t.process = MagicMock()
        t.process.poll.return_value = None
        t.process.stdout.readline.side_effect = [
            json.dumps({"method": "notifications/x", "params": {"from": srv_name}}),
            "",
        ]
        t._reader_loop()

    msgs = inbox.unconsumed()
    assert len(msgs) == 2
    servers = sorted(m["server"] for m in msgs)
    assert servers == ["feishu", "slack"]
