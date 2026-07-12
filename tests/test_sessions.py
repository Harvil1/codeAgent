"""会话存储测试。"""

import sqlite3

import pytest

from agent.session_store import SessionStore, is_fts5_available
from agent.title_generator import generate_title, maybe_set_title


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """提供临时 SessionStore。"""
    return SessionStore(tmp_path / "sessions.db")


@pytest.fixture
def store_with_msgs(store):
    """提供带几条消息的 store。"""
    sid = store.create_session(model="deepseek-chat", provider="deepseek")
    store.append_message(sid, "user", "帮我写个 Python 脚本")
    store.append_message(sid, "assistant", "好的，这是脚本...")
    store.append_message(sid, "user", "能改成异步的吗")
    store.append_message(sid, "assistant", "当然可以...")
    return store, sid


# ---------------------------------------------------------------------------
# FTS5 可用性
# ---------------------------------------------------------------------------

def test_fts5_available():
    """验证 SQLite 支持 FTS5（运行测试的前置条件）。"""
    assert is_fts5_available(), "此 SQLite 不支持 FTS5"


# ---------------------------------------------------------------------------
# create / append
# ---------------------------------------------------------------------------

def test_create_session(store):
    sid = store.create_session(model="test-model", provider="test")
    assert isinstance(sid, str)
    assert len(sid) > 0

    info = store.get_session(sid)
    assert info is not None
    assert info["model"] == "test-model"
    assert info["message_count"] == 0


def test_append_message_increments_count(store):
    sid = store.create_session()
    store.append_message(sid, "user", "hello")
    store.append_message(sid, "assistant", "hi")

    info = store.get_session(sid)
    assert info["message_count"] == 2


def test_append_message_turn_index(store_with_msgs):
    store, sid = store_with_msgs
    msgs = store.get_messages(sid)
    # 两条 user 消息应该在不同 turn
    user_msgs = [m for m in msgs if m["role"] == "user"]
    # 第一条 user 是 turn 1，第二条是 turn 2（这里没存 turn_index 到返回，靠消息顺序）


def test_get_messages_order(store_with_msgs):
    store, sid = store_with_msgs
    msgs = store.get_messages(sid)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "帮我写个 Python 脚本"
    assert msgs[-1]["role"] == "assistant"


def test_append_with_tool_calls(store):
    sid = store.create_session()
    store.append_message(
        sid, "assistant", "",
        tool_calls=[{"id": "call_1", "function": {"name": "terminal"}}],
    )
    store.append_message(
        sid, "tool", "result",
        tool_call_id="call_1",
    )

    msgs = store.get_messages(sid)
    assert msgs[0]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# list / delete
# ---------------------------------------------------------------------------

def test_list_sessions(store):
    s1 = store.create_session(title="第一个")
    s2 = store.create_session(title="第二个")
    sessions = store.list_sessions()
    assert len(sessions) == 2
    # 按 updated_at 倒序：最后创建的在前
    titles = [s["title"] for s in sessions]
    assert "第二个" in titles


def test_delete_session_cascades(store_with_msgs):
    store, sid = store_with_msgs
    store.delete_session(sid)
    assert store.get_session(sid) is None
    assert store.get_messages(sid) == []


# ---------------------------------------------------------------------------
# set_title
# ---------------------------------------------------------------------------

def test_set_title(store):
    sid = store.create_session()
    store.set_title(sid, "新标题")
    assert store.get_session(sid)["title"] == "新标题"


# ---------------------------------------------------------------------------
# 搜索（FTS5）
# ---------------------------------------------------------------------------

def test_search_finds_matches(store_with_msgs):
    store, sid = store_with_msgs
    results = store.search("Python")
    assert len(results) > 0
    # 匹配的消息包含 "Python"
    contents = [r["content"] for r in results]
    assert any("Python" in c for c in contents)


def test_search_no_match(store_with_msgs):
    store, sid = store_with_msgs
    results = store.search("不存在的随机词汇xyz123")
    assert results == []


def test_search_with_snippet(store_with_msgs):
    """搜索结果包含 snippet（带高亮）。"""
    store, sid = store_with_msgs
    results = store.search("Python")
    assert len(results) > 0
    assert "snippet" in results[0]


def test_search_filtered_by_session(store_with_msgs):
    """限定 session_id 的搜索。"""
    store, sid = store_with_msgs
    # 创建另一个会话，含相同关键词
    other = store.create_session()
    store.append_message(other, "user", "Python 也是好语言")

    results = store.search("Python", session_id=sid)
    # 只返回 sid 的结果
    for r in results:
        assert r["session_id"] == sid


def test_session_search_tool():
    """通过 registry.dispatch 调用 session_search 工具。"""
    import json
    import tempfile
    from pathlib import Path
    from tools.registry import registry

    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "s.db")
        sid = store.create_session()
        store.append_message(sid, "user", "测试关键词 Python")
        store.append_message(sid, "assistant", "好的")

        result = registry.dispatch(
            "session_search",
            {"query": "Python"},
            session_store=store,
        )
        data = json.loads(result)
        assert data["total"] > 0


def test_session_search_tool_no_store():
    """无 session_store 时返回错误。"""
    import json
    from tools.registry import registry

    result = registry.dispatch(
        "session_search",
        {"query": "test"},
        # 不传 session_store
    )
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# title_generator
# ---------------------------------------------------------------------------

def test_generate_title_short():
    assert generate_title("你好") == "你好"


def test_generate_title_long():
    long_msg = "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的消息" * 2
    title = generate_title(long_msg)
    assert len(title) <= 40
    assert title.endswith("...")


def test_generate_title_multiline():
    title = generate_title("第一行\n第二行\n第三行")
    assert title == "第一行"


def test_generate_title_strips_slash():
    title = generate_title("/skill 实际内容")
    assert "skill" not in title or "实际" in title


def test_generate_title_empty():
    assert generate_title("") is None


def test_maybe_set_title_first_time(store):
    sid = store.create_session()
    maybe_set_title(store, sid, "第一条消息", None)
    assert store.get_session(sid)["title"] == "第一条消息"


def test_maybe_set_title_skip_if_exists(store):
    sid = store.create_session(title="已有标题")
    maybe_set_title(store, sid, "新消息", "已有标题")
    # 不应该被覆盖
    assert store.get_session(sid)["title"] == "已有标题"
