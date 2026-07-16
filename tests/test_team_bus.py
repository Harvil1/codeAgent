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
