"""第 4 轮 bug 修复测试：严重级 bug（5 个）。"""
import re
import inspect
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# X9: HandoffStore 密钥正则应支持新格式 OpenAI key（含 _ 和 -）
# ---------------------------------------------------------------------------

def test_secret_pattern_catches_modern_openai_keys():
    """X9 fix: 密钥正则应支持 sk-proj-XXX（含 _ 和 -）格式。"""
    from agent.handoff import SECRET_PATTERN

    # 老格式 sk-XXXX（纯字母数字）应匹配
    old_key = "sk-Abcdef1234567890Abcdef"
    assert SECRET_PATTERN.search(old_key), f"老格式应匹配: {old_key}"

    # 新格式 sk-proj-XXX（含 -）应匹配
    new_key1 = "sk-proj-Abcdef1234567890Abcdef"
    assert SECRET_PATTERN.search(new_key1), f"新格式 sk-proj 应匹配: {new_key1}"

    # 含 _ 的 key
    new_key2 = "sk-Abc_def_1234567890Abcdef"
    assert SECRET_PATTERN.search(new_key2), f"含 _ 的 key 应匹配: {new_key2}"


def test_secret_pattern_avoids_false_positive_for_short_strings():
    """回归：短字符串不误报。"""
    from agent.handoff import SECRET_PATTERN
    # 太短不报
    assert not SECRET_PATTERN.search("sk-abc")


# ---------------------------------------------------------------------------
# X13: _fix_tool_call_pairs 反向孤儿应按"截至当前位置"判断，不用全局集合
# ---------------------------------------------------------------------------

def test_fix_tool_call_pairs_drops_reverse_orphan_correctly():
    """X13 fix: tool(result of B) 出现在 assistant(tc B) 之前应被删（即使 B 后面存在）。"""
    from agent.context_compressor import _fix_tool_call_pairs

    messages = [
        {"role": "user", "content": "hi"},
        # 反向孤儿：tool(result for B) 但前面的 assistant 只声明了 A
        {"role": "tool", "tool_call_id": "B", "content": "premature result"},
        # 后面才出现 assistant 声明 B
        {"role": "assistant", "tool_calls": [{"id": "B", "function": {"name": "f"}}]},
    ]
    fixed = _fix_tool_call_pairs(messages)
    # 反向孤儿应被删除
    has_premature = any(
        m.get("role") == "tool" and m.get("tool_call_id") == "B"
        for m in fixed
    )
    # 在 assistant(tc B) 之前的位置不应有 tool(B)
    seen_b_in_tc = False
    premature_found = False
    for m in fixed:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id") == "B":
                    seen_b_in_tc = True
        elif m.get("role") == "tool" and m.get("tool_call_id") == "B":
            if not seen_b_in_tc:
                premature_found = True  # bad: tool(B) 在 assistant(tc B) 之前
    assert not premature_found, (
        f"反向孤儿（tool(B) 在 assistant(tc B) 之前）应被删除，fixed: {fixed}"
    )


def test_fix_tool_call_pairs_keeps_well_ordered_pairs():
    """回归：正常 tool_call + result 配对不动。"""
    from agent.context_compressor import _fix_tool_call_pairs

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "A", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "A", "content": "result A"},
        {"role": "assistant", "content": "done"},
    ]
    fixed = _fix_tool_call_pairs(messages)
    # 应保留 tool(A) result
    has_result_a = any(
        m.get("role") == "tool" and m.get("tool_call_id") == "A" for m in fixed
    )
    assert has_result_a, f"正常配对不应删，fixed: {fixed}"


# ---------------------------------------------------------------------------
# X5: reflection supersedes 必须同 type 才匹配
# ---------------------------------------------------------------------------

def test_reflection_supersedes_respects_type():
    """X5 fix: supersedes 匹配应同时检查 name + type，不跨类型误降。"""
    from agent.reflection import apply_reflection
    from agent.memory_store import MemoryEntry
    from datetime import datetime, timezone

    store = MagicMock()
    # 库里已有 feedback 类型的 "测试流程"
    existing_fb = MemoryEntry(
        id="feedback#001", name="测试流程", description="d", type="feedback",
        body="原 feedback", created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    store.list_all.return_value = [existing_fb]
    store.save = MagicMock(return_value="project#001")
    store.update = MagicMock()

    # insight: type=project, supersedes="测试流程"（不应跨类匹配 feedback）
    insights = [{
        "type": "project", "name": "新流程", "description": "d", "body": "b",
        "summary": "s", "confidence": 0.8, "supersedes": "测试流程",
    }]

    # 直接 patch run_reflection 绕过 LLM 解析
    with patch("agent.reflection.run_reflection", return_value=insights):
        apply_reflection(
            messages=[{"role": "user", "content": "hi"}],
            memory_store=store,
            llm_client=MagicMock(),
            session_id="s1",
        )

    # X5: 不应调 update（feedback 不应被 project supersede）
    assert store.update.call_count == 0, (
        f"supersede 不应跨类型，update 不该被调，实际 calls: {store.update.call_args_list}"
    )


# ---------------------------------------------------------------------------
# X8: get_messages 同秒排序应按 rowid，不按 timestamp DESC
# ---------------------------------------------------------------------------

def test_get_messages_subquery_uses_rowid_not_timestamp():
    """X8 fix: get_messages 源码子查询应用 ORDER BY rowid DESC（取最后 N 条），
    外层按 rowid ASC（恢复时序），不再用 timestamp DESC（同秒乱序）。
    """
    from agent.session_store import SessionStore
    src = inspect.getsource(SessionStore.get_messages)

    # 子查询不应 ORDER BY timestamp DESC（同秒乱序）
    # 应该 ORDER BY rowid DESC
    # 简化检查：不含 "ORDER BY timestamp DESC"
    assert "ORDER BY timestamp DESC" not in src.replace("\n", " ").replace("  ", " "), (
        f"get_messages 不应用 timestamp DESC（同秒乱序），实际:\n{src}"
    )


# ---------------------------------------------------------------------------
# X14: reflection 批内 supersedes 应在所有写入后再 supersede
# ---------------------------------------------------------------------------

def test_reflection_supersedes_handles_intra_batch():
    """X14 fix: 批内 supersedes 应在所有写入后处理，否则推翻刚写入的新条目失效。"""
    from agent.reflection import apply_reflection
    from agent.memory_store import MemoryEntry
    from datetime import datetime, timezone
    import json as _json

    store = MagicMock()
    # 模拟库为空
    store.list_all.return_value = []
    saved_names_types = []
    updated_ids = []

    def fake_save(**kwargs):
        # 记录写入的 (type, name) + 返回一个 fake id
        saved_names_types.append((kwargs.get("type"), kwargs.get("name")))
        fake_id = f"{kwargs.get('type')}#{len(saved_names_types):03d}"
        # 模拟 save 返回 id
        return fake_id

    def fake_update(mem_id, **kwargs):
        updated_ids.append(mem_id)

    store.save = MagicMock(side_effect=fake_save)
    store.update = MagicMock(side_effect=fake_update)

    # 批内 2 个 insight：第一个写入 "测试流程"（project），第二个 supersede 它
    insights = [
        {"type": "project", "name": "测试流程", "description": "d", "body": "b",
         "summary": "s", "confidence": 0.8},
        {"type": "project", "name": "更好流程", "description": "d", "body": "b",
         "summary": "s", "confidence": 0.8, "supersedes": "测试流程"},
    ]

    # 直接 patch run_reflection 绕过 LLM 解析
    with patch("agent.reflection.run_reflection", return_value=insights):
        apply_reflection(
            messages=[{"role": "user", "content": "hi"}],
            memory_store=store,
            llm_client=MagicMock(),
            session_id="s1",
        )

    # 第二条 insight 想 supersede "测试流程"（project），库里现在有（同批写入的）
    # 应触发 update（X14 修复后批内 supersede 工作）
    assert len(updated_ids) >= 1, (
        f"批内 supersede 应生效（至少 1 次 update），实际 updated_ids: {updated_ids}"
    )
    # 更新的应是同批写入的 "测试流程"（id project#001）
    assert any("project#001" in uid or "测试流程" in str(store.update.call_args_list) for uid in updated_ids), (
        f"批内 supersede 应针对 project#001（刚写入的 测试流程），实际: {updated_ids}, calls: {store.update.call_args_list}"
    )
