"""Mailbox 异步队列测试（CCAR8 Task 8）。

测试覆盖：
- send + check_unread 基本投递
- check_all 含已读
- mark_read 持久化（跨实例）
- clear
- 多收件人
- kind 字段
- send fail-open（写盘失败不抛）
"""
import logging
from pathlib import Path

from agent.team.mailbox import Mailbox


def test_send_and_check_unread(tmp_path: Path):
    mb = Mailbox(tmp_path)
    msg_id = mb.send(to="alice", from_="bob", content="hello")
    unread = mb.check_unread("alice")
    assert len(unread) == 1
    assert unread[0]["id"] == msg_id
    assert unread[0]["content"] == "hello"
    assert unread[0]["read"] is False


def test_check_all_includes_read(tmp_path: Path):
    mb = Mailbox(tmp_path)
    mid = mb.send(to="alice", from_="bob", content="hi")
    mb.mark_read("alice", [mid])
    all_msgs = mb.check_all("alice")
    assert len(all_msgs) == 1
    assert all_msgs[0]["read"] is True
    # unread 应该为空
    assert mb.check_unread("alice") == []


def test_mark_read_persists(tmp_path: Path):
    """mark_read 写盘后，新实例也能看到 read 状态。"""
    mb1 = Mailbox(tmp_path)
    mid = mb1.send(to="alice", from_="bob", content="hi")
    mb1.mark_read("alice", [mid])
    # 新实例
    mb2 = Mailbox(tmp_path)
    unread = mb2.check_unread("alice")
    assert unread == []
    all_msgs = mb2.check_all("alice")
    assert all_msgs[0]["read"] is True


def test_clear(tmp_path: Path):
    mb = Mailbox(tmp_path)
    mb.send(to="alice", from_="bob", content="1")
    mb.send(to="alice", from_="bob", content="2")
    count = mb.clear("alice")
    assert count == 2
    assert mb.check_all("alice") == []


def test_send_to_multiple_recipients(tmp_path: Path):
    mb = Mailbox(tmp_path)
    mb.send(to="alice", from_="bob", content="for alice")
    mb.send(to="charlie", from_="bob", content="for charlie")
    assert len(mb.check_unread("alice")) == 1
    assert len(mb.check_unread("charlie")) == 1
    assert mb.check_unread("alice")[0]["content"] == "for alice"


def test_kind_field(tmp_path: Path):
    mb = Mailbox(tmp_path)
    mb.send(to="alice", from_="bob", content="task", kind="task")
    msg = mb.check_unread("alice")[0]
    assert msg["kind"] == "task"


def test_send_failopen_on_disk_error(tmp_path: Path, caplog):
    """写盘失败不抛。"""
    mb = Mailbox(tmp_path)
    mb._base = "/nonexistent/path"
    with caplog.at_level(logging.WARNING):
        mb.send(to="x", from_="y", content="z")  # 不抛
    assert any("mailbox" in r.message.lower() for r in caplog.records)
