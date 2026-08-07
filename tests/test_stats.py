"""SessionStore.get_stats() 测试。"""

import pytest

from agent.session_store import SessionStore


@pytest.fixture
def store_with_data(tmp_path):
    """构造带 3 个会话 + 各种消息 + 工具调用的 store。"""
    s = SessionStore(tmp_path / "test.db")

    # 会话 1：长会话，多个工具调用
    sid1 = s.create_session(title="调试 bug", model="deepseek-chat", provider="deepseek")
    s.append_message(sid1, "user", "帮我查问题")
    s.append_message(sid1, "assistant", "", tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}},
        {"id": "c2", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
    ])
    s.append_message(sid1, "tool", '{"stdout":"file"}', tool_call_id="c1")
    s.append_message(sid1, "tool", "content", tool_call_id="c2")
    s.append_message(sid1, "assistant", "看到 file")
    s.append_message(sid1, "user", "再来一次")
    s.append_message(sid1, "assistant", "", tool_calls=[
        {"id": "c3", "type": "function",
         "function": {"name": "terminal", "arguments": '{"cmd":"pwd"}'}}
    ])

    # 会话 2：短会话
    sid2 = s.create_session(title="问天气", model="deepseek-chat", provider="deepseek")
    s.append_message(sid2, "user", "今天天气")
    s.append_message(sid2, "assistant", "晴")

    # 会话 3：纯聊天无工具
    sid3 = s.create_session(title="闲聊", model="claude-sonnet-4", provider="anthropic")
    s.append_message(sid3, "user", "你好")
    s.append_message(sid3, "assistant", "你好啊")

    return s, (sid1, sid2, sid3)


def test_get_stats_overview(store_with_data):
    s, _ = store_with_data
    stats = s.get_stats()
    assert stats["sessions"] == 3
    # 总消息数：会话1 (7) + 会话2 (2) + 会话3 (2) = 11
    assert stats["messages"] == 11
    assert stats["earliest"] is not None
    assert stats["latest"] is not None


def test_get_stats_empty_store(tmp_path):
    """空 store 返回零值。"""
    s = SessionStore(tmp_path / "empty.db")
    stats = s.get_stats()
    assert stats["sessions"] == 0
    assert stats["messages"] == 0
    assert stats["earliest"] is None
    assert stats["latest"] is None
    assert stats["top_sessions"] == []
    assert stats["tool_calls"] == []
    assert stats["role_distribution"] == {}


def test_get_stats_top_sessions_sorted(store_with_data):
    s, (sid1, sid2, sid3) = store_with_data
    stats = s.get_stats()
    # 会话 1 应该排第一（消息最多）
    assert stats["top_sessions"][0]["id"] == sid1
    assert stats["top_sessions"][0]["message_count"] == 7
    # 只取 Top 5
    assert len(stats["top_sessions"]) <= 5


def test_get_stats_tool_call_ranking(store_with_data):
    s, _ = store_with_data
    stats = s.get_stats()
    # terminal: 2 次（c1 + c3）, read_file: 1 次
    assert stats["tool_calls"][0] == {"name": "terminal", "count": 2}
    assert {"name": "read_file", "count": 1} in stats["tool_calls"]


def test_get_stats_tool_calls_top_10_limit(tmp_path):
    """超过 10 个工具也只返回 Top 10。"""
    s = SessionStore(tmp_path / "test.db")
    sid = s.create_session()
    # 构造 15 个不同的工具调用，每个 1 次
    calls = [
        {"id": f"c{i}", "type": "function",
         "function": {"name": f"tool_{i}", "arguments": "{}"}}
        for i in range(15)
    ]
    s.append_message(sid, "assistant", "", tool_calls=calls)
    stats = s.get_stats()
    assert len(stats["tool_calls"]) == 10


def test_get_stats_role_distribution(store_with_data):
    s, _ = store_with_data
    stats = s.get_stats()
    # 三个会话累计 user/assistant/tool
    # sid1: 2 user / 3 assistant / 2 tool
    # sid2: 1 user / 1 assistant
    # sid3: 1 user / 1 assistant
    assert stats["role_distribution"]["user"] == 4
    assert stats["role_distribution"]["assistant"] == 5
    assert stats["role_distribution"]["tool"] == 2


def test_get_stats_malformed_tool_calls_skipped(tmp_path):
    """损坏的 tool_calls 字段不应让 get_stats 崩。"""
    s = SessionStore(tmp_path / "test.db")
    sid = s.create_session()
    # JSONL 版：直接写异常 tool_calls 到 .jsonl（字符串而非 list）
    path = s._session_file(sid)
    path.write_text(
        '{"id":"m1","role":"assistant","content":"","tool_calls":"not-valid-json",'
        '"tool_call_id":null,"name":null,"timestamp":"2026-01-01T00:00:00Z","turn_index":1}\n',
        encoding="utf-8",
    )
    # 不应抛
    stats = s.get_stats()
    assert stats["sessions"] == 1


def test_get_stats_malformed_tool_calls_structure_skipped(tmp_path):
    """tool_calls 不是 list of dicts 也应跳过。"""
    s = SessionStore(tmp_path / "test.db")
    sid = s.create_session()
    path = s._session_file(sid)
    path.write_text(
        '{"id":"m1","role":"assistant","content":"","tool_calls":{"weird":"object"},'
        '"tool_call_id":null,"name":null,"timestamp":"2026-01-01T00:00:00Z","turn_index":1}\n'
        '{"id":"m2","role":"assistant","content":"","tool_calls":["not-a-dict"],'
        '"tool_call_id":null,"name":null,"timestamp":"2026-01-01T00:00:00Z","turn_index":2}\n',
        encoding="utf-8",
    )
    stats = s.get_stats()
    assert stats["tool_calls"] == []  # 无有效工具调用
