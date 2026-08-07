"""会话搜索增强（B2）测试：role/tool/since/until 过滤。"""

import pytest

from agent.session_store import SessionStore


@pytest.fixture
def store_with_msgs(tmp_path):
    """构造带过滤特征的 store。"""
    s = SessionStore(tmp_path / "test.db")

    sid1 = s.create_session(title="调试")
    # 2026-07-01 的消息
    s.append_message(sid1, "user", "python 报错 KeyError")
    s.append_message(sid1, "assistant", "", tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "terminal", "arguments": '{"cmd":"python a.py"}'}}
    ])

    sid2 = s.create_session(title="部署")
    # 2026-07-10 的消息
    s.append_message(sid2, "user", "部署 python 服务")
    s.append_message(sid2, "assistant", "已用 docker 部署")

    # JSONL 版：直接改 .jsonl 文件内容调 timestamp（让 since/until 过滤有意义）
    import json as _json
    for sid, target_role, new_ts in [
        (sid1, "user", "2026-07-01T10:00:00Z"),
        (sid2, "user", "2026-07-10T15:00:00Z"),
    ]:
        path = s._session_file(sid)
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = _json.loads(line)
                if obj.get("role") == target_role:
                    obj["timestamp"] = new_ts
                lines.append(_json.dumps(obj, ensure_ascii=False))
        from agent.atomic_io import atomic_write_text
        atomic_write_text(path, "\n".join(lines) + "\n")

    return s


def test_search_role_filter(store_with_msgs):
    """role=user 只返回 user 消息。"""
    results = store_with_msgs.search("python", role="user")
    assert len(results) >= 1
    assert all(r["role"] == "user" for r in results)


def test_search_role_filter_assistant(store_with_msgs):
    """role=assistant 只返回 assistant 消息。"""
    results = store_with_msgs.search("python", role="assistant")
    assert all(r["role"] == "assistant" for r in results)


def test_search_since_filter(store_with_msgs):
    """since=2026-07-05 只返回 7-10 的消息。"""
    results = store_with_msgs.search("python", since="2026-07-05T00:00:00")
    # 只应匹配 sid2 的 user 消息
    assert len(results) == 1
    assert "部署" in results[0]["content"] or "服务" in results[0]["content"]


def test_search_until_filter(store_with_msgs):
    """until=2026-07-05 只返回 7-01 的消息。"""
    results = store_with_msgs.search("python", until="2026-07-05T23:59:59")
    # 只应匹配 sid1 的 user 消息
    assert len(results) == 1
    assert "KeyError" in results[0]["content"] or "报错" in results[0]["content"]


def test_search_since_until_range(store_with_msgs):
    """since + until 组合。"""
    results = store_with_msgs.search(
        "python",
        since="2026-07-01T00:00:00",
        until="2026-07-01T23:59:59",
    )
    assert all("2026-07-01" in r["timestamp"] for r in results)


def test_search_tool_name_filter(store_with_msgs):
    """tool=terminal 只返回含 terminal 调用的消息。"""
    results = store_with_msgs.search("报错", tool_name="terminal")
    # assistant 消息含 terminal tool_call，但 FTS 只匹配 user 消息的关键词
    # 这里 query="报错" 只匹配 sid1 user 消息；user 消息没 tool_calls
    # 所以应该返回 0 条
    assert len(results) == 0

    # 不带 keyword 限制（全匹配）测 tool 过滤
    # FTS 必须有 query，所以换个思路：用 python 关键词 + tool 过滤
    # sid1 的 user 消息含 "python"，但 user 没 tool_calls → 不应匹配
    results2 = store_with_msgs.search("python", tool_name="terminal")
    assert all(r["role"] != "user" for r in results2)  # user 不应有 tool_calls


def test_search_combined_filters(store_with_msgs):
    """role + since 组合。"""
    results = store_with_msgs.search(
        "python", role="user", since="2026-07-05T00:00:00",
    )
    # 只 sid2 user 消息（2026-07-10）
    assert len(results) == 1
    assert results[0]["role"] == "user"


def test_search_invalid_role_no_crash(store_with_msgs):
    """无效 role 不应崩（FTS 应返回空）。"""
    results = store_with_msgs.search("python", role="invalid_role")
    assert results == []


def test_search_backward_compatible(store_with_msgs):
    """旧调用（不带 filter 参数）应该照常工作。"""
    results = store_with_msgs.search("python")
    assert len(results) >= 1
    # 不应包含 tool_calls 字段（保持向后兼容）
    for r in results:
        assert "tool_calls" not in r
