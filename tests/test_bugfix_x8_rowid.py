"""回归测试：get_messages(limit) 不报 no such column: rowid。

bug：子查询用 ORDER BY rowid DESC LIMIT 时，若内层 SELECT
显式列出字段（不含 rowid），外层 SELECT * 看不到 rowid，外层
ORDER BY rowid ASC 报 sqlite3.OperationalError。
"""
import pytest


def test_get_messages_with_limit_does_not_error_on_rowid(tmp_path):
    """get_messages(limit=N) 不应报 'no such column: rowid'。"""
    from agent.session_store import SessionStore

    store = SessionStore(tmp_path / "test.db")
    sid = store.create_session(model="test", provider="test")
    store.append_message(sid, role="user", content="hi")
    store.append_message(sid, role="assistant", content="hello")
    store.append_message(sid, role="user", content="how are you")

    # 调 get_messages(limit=2) —— 回归点：不应抛 sqlite3.OperationalError
    msgs = store.get_messages(sid, limit=2)
    assert len(msgs) == 2
    # 应是"最后 2 条"按时间正序（assistant + 最后 user）
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "how are you"


def test_get_messages_with_limit_one(tmp_path):
    """边界：limit=1 只返回最后一条。"""
    from agent.session_store import SessionStore

    store = SessionStore(tmp_path / "test.db")
    sid = store.create_session(model="test", provider="test")
    store.append_message(sid, role="user", content="first")
    store.append_message(sid, role="assistant", content="last")

    msgs = store.get_messages(sid, limit=1)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "last"


def test_get_messages_with_limit_larger_than_count(tmp_path):
    """边界：limit > 实际消息数 → 返回全部。"""
    from agent.session_store import SessionStore

    store = SessionStore(tmp_path / "test.db")
    sid = store.create_session(model="test", provider="test")
    store.append_message(sid, role="user", content="only")

    msgs = store.get_messages(sid, limit=10)
    assert len(msgs) == 1


def test_get_messages_without_limit_unchanged(tmp_path):
    """回归：无 limit 路径不受影响。"""
    from agent.session_store import SessionStore

    store = SessionStore(tmp_path / "test.db")
    sid = store.create_session(model="test", provider="test")
    store.append_message(sid, role="user", content="a")
    store.append_message(sid, role="assistant", content="b")

    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "a"
    assert msgs[1]["content"] == "b"
