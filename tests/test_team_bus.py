"""MessageBus 测试。"""
import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from agent.team.bus import MessageBus, TeamMessage


def test_send_creates_inbox_file(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    mid = bus.send(from_="alice", to="bob", type_="message", content="hi")
    assert mid  # non-empty
    inbox = tmp_path / "inbox" / "bob.jsonl"
    assert inbox.exists()
    lines = inbox.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["from"] == "alice"
    assert parsed["to"] == "bob"
    assert parsed["content"] == "hi"
    assert parsed["type"] == "message"
    assert parsed["id"] == mid


def test_send_returns_message_id(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    mid = bus.send(from_="a", to="b", type_="message", content="x")
    assert isinstance(mid, str) and len(mid) > 0


def test_send_invalid_type_raises(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    with pytest.raises(ValueError):
        bus.send(from_="a", to="b", type_="invalid_kind", content="x")


def test_read_inbox_returns_messages(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="m1")
    bus.send(from_="a", to="b", type_="message", content="m2")
    msgs = bus.read_inbox("b")
    assert len(msgs) == 2
    assert msgs[0].content == "m1"
    assert msgs[1].content == "m2"


def test_read_inbox_is_consumptive(tmp_path: Path):
    """读后清空。"""
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="x")
    first = bus.read_inbox("b")
    assert len(first) == 1
    second = bus.read_inbox("b")
    assert second == []


def test_read_inbox_empty_returns_empty_list(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    assert bus.read_inbox("nobody") == []


def test_read_inbox_unknown_name_returns_empty(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    # 即使 inbox 文件不存在也返回 []
    assert bus.read_inbox("nonexistent") == []


def test_list_inboxes_returns_names(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="x")
    bus.send(from_="a", to="c", type_="message", content="x")
    names = set(bus.list_inboxes())
    assert "b" in names
    assert "c" in names


def test_concurrent_send_safe(tmp_path: Path):
    """多线程并发 send 同一 inbox 不丢消息。"""
    bus = MessageBus(team_dir=tmp_path)
    errors = []

    def worker(n: int):
        try:
            for i in range(20):
                bus.send(from_="sender", to="target",
                         type_="message", content=f"msg-{n}-{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    msgs = bus.read_inbox("target")
    assert len(msgs) == 5 * 20  # 100


def test_concurrent_read_write_safe(tmp_path: Path):
    """读 + 写并发不抛异常。"""
    bus = MessageBus(team_dir=tmp_path)
    errors = []

    def writer():
        try:
            for i in range(20):
                bus.send(from_="a", to="b", type_="message", content=f"m{i}")
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(20):
                bus.read_inbox("b")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []


# ============ P1-8: request_id 协议化测试 ============

def test_send_request_auto_generates_request_id(tmp_path):
    """send_request 自动生成 request_id（req_ 前缀）。"""
    from agent.team.bus import MessageBus
    bus = MessageBus(team_dir=tmp_path / "team")
    req_id = bus.send_request(from_="alice", to="bob", content="查一下余额")

    assert req_id.startswith("req_")
    # bob 的 inbox 应该有一条 request 消息，request_id 匹配
    msgs = bus.read_inbox("bob")
    assert len(msgs) == 1
    assert msgs[0].type == "request"
    assert msgs[0].request_id == req_id
    assert msgs[0].from_ == "alice"
    assert msgs[0].content == "查一下余额"


def test_send_response_requires_request_id(tmp_path):
    """send_response 必须传 request_id（否则 raise）。"""
    from agent.team.bus import MessageBus
    bus = MessageBus(team_dir=tmp_path / "team")

    with pytest.raises(ValueError, match="request_id"):
        bus.send_response(from_="bob", to="alice", request_id=None, content="ok")


def test_send_response_pairs_with_request(tmp_path):
    """send_response 用 request_id 关联到原 request。"""
    from agent.team.bus import MessageBus
    bus = MessageBus(team_dir=tmp_path / "team")

    req_id = bus.send_request(from_="alice", to="bob", content="查余额")
    # bob 收到 request，处理后回复
    bus.send_response(from_="bob", to="alice", request_id=req_id, content="余额 100")

    # alice 收到 response
    msgs = bus.read_inbox("alice")
    assert len(msgs) == 1
    assert msgs[0].type == "response"
    assert msgs[0].request_id == req_id  # 配对
    assert msgs[0].content == "余额 100"


def test_send_response_type_validates_request_id_via_send(tmp_path):
    """直接 send(type_='response') 时也强制 request_id（类型校验）。"""
    from agent.team.bus import MessageBus
    bus = MessageBus(team_dir=tmp_path / "team")

    # 不传 request_id 应该 raise
    with pytest.raises(ValueError, match="response.*request_id"):
        bus.send(
            from_="bob", to="alice", type_="response",
            content="ok", request_id=None,
        )


def test_find_response_matches_request_id(tmp_path):
    """find_response 从一堆消息里按 request_id 找 response。"""
    from agent.team.bus import MessageBus, TeamMessage
    bus = MessageBus(team_dir=tmp_path / "team")

    target_req = "req_abc123"
    msgs = [
        TeamMessage(id="1", from_="x", to="me", type="message",
                    content="noise", ts="t"),
        TeamMessage(id="2", from_="y", to="me", type="response",
                    content="result", ts="t", request_id="req_other"),
        TeamMessage(id="3", from_="z", to="me", type="response",
                    content="匹配的响应", ts="t", request_id=target_req),
    ]
    found = bus.find_response(msgs, target_req)
    assert found is not None
    assert found.content == "匹配的响应"


def test_find_response_returns_none_when_not_found(tmp_path):
    """找不到匹配的 response 时返回 None。"""
    from agent.team.bus import MessageBus, TeamMessage
    bus = MessageBus(team_dir=tmp_path / "team")

    msgs = [
        TeamMessage(id="1", from_="x", to="me", type="message",
                    content="noise", ts="t"),
    ]
    found = bus.find_response(msgs, "req_missing")
    assert found is None


def test_request_response_full_roundtrip(tmp_path):
    """完整 request-response 往返：alice 发请求 → bob 收+回 → alice 收回复。"""
    from agent.team.bus import MessageBus
    bus = MessageBus(team_dir=tmp_path / "team")

    # alice 发请求
    req_id = bus.send_request(from_="alice", to="bob", content="任务 X 状态？")

    # bob 读 inbox，处理
    bob_msgs = bus.read_inbox("bob")
    assert len(bob_msgs) == 1
    request = bob_msgs[0]
    assert request.type == "request"
    assert request.request_id == req_id

    # bob 根据 request_id 回复
    bus.send_response(
        from_="bob", to="alice",
        request_id=request.request_id,
        content="任务 X 已完成 50%",
    )

    # alice 读回复，用 find_response 找到配对的
    alice_msgs = bus.read_inbox("alice")
    response = bus.find_response(alice_msgs, req_id)
    assert response is not None
    assert response.content == "任务 X 已完成 50%"
    assert response.from_ == "bob"
