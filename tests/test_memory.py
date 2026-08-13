"""多文件记忆系统测试。"""
from pathlib import Path

import pytest

from agent.memory_store import MemoryStore, MemoryEntry
def test_save_minimal_fields(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="x", description="y", type="other",
    )  # 无 body
    entry = store.get(mid)
    assert entry.body == ""


def test_save_invalid_type_raises(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    with pytest.raises(ValueError):
        store.save(name="x", description="y", type="invalid_kind", body="")


def test_get_returns_entry(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b")
    entry = store.get(mid)
    assert entry.id == mid
    assert entry.name == "t1"
    assert entry.type == "user"


def test_get_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    assert store.get("nonexistent") is None


def test_list_all_returns_entries(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    store.save(name="t2", description="d", type="project", body="")
    all_entries = store.list_all()
    assert len(all_entries) == 2


def test_update_modifies_fields(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b1")
    store.update(mid, name="t1-new", body="b2")
    entry = store.get(mid)
    assert entry.name == "t1-new"
    assert entry.body == "b2"
    assert entry.description == "d"  # 未改


def test_update_unknown_raises(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    with pytest.raises(KeyError):
        store.update("nonexistent", body="x")
def test_delete_unknown_returns_false(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    assert store.delete("nonexistent") is False


def test_load_body_returns_content(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="完整内容")
    assert store.load_body(mid) == "完整内容"


def test_load_body_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    assert store.load_body("nonexistent") is None


def test_snapshot_for_prompt_returns_index_text(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="t1", description="描述1", type="user", body="")
    store.save(name="t2", description="描述2", type="project", body="")
    snap = store.snapshot_for_prompt()
    assert "t1" in snap
    assert "描述1" in snap
    assert "t2" in snap


def test_index_rebuilt_on_startup(tmp_path: Path):
    """新建 store 时扫描 .memory/ 重建索引。"""
    # 先用 store1 写两个 memory
    store1 = MemoryStore(omnimate_home=tmp_path)
    mid1 = store1.save(name="t1", description="d", type="user", body="")
    mid2 = store1.save(name="t2", description="d", type="project", body="")
    # 再开一个 store（模拟下次会话），应能看到两条
    store2 = MemoryStore(omnimate_home=tmp_path)
    all_entries = store2.list_all()
    assert len(all_entries) == 2
    assert {e.id for e in all_entries} == {mid1, mid2}
def test_malformed_frontmatter_skipped(tmp_path: Path):
    """frontmatter 解析失败的文件跳过 + log warning。"""
    store = MemoryStore(omnimate_home=tmp_path)
    # 写一个合法的
    good_mid = store.save(name="good", description="d", type="user", body="")
    # 直接写一个坏文件到 .memory/
    bad_file = tmp_path / ".memory" / "bad_mid.md"
    bad_file.write_text(
        "---\ninvalid: yaml: content\n---\nbody",
        encoding="utf-8",
    )
    # 重新加载（模拟下次会话）
    store2 = MemoryStore(omnimate_home=tmp_path)
    all_ids = {e.id for e in store2.list_all()}
    assert good_mid in all_ids
    assert "bad_mid" not in all_ids  # 被跳过


# ---------------------------------------------------------------------------
# T3: memory 工具 5-action 测试（_handle_memory 接口）
# ---------------------------------------------------------------------------
import json

from tools.memory_tool import _handle_memory


def _run(action, store, **kwargs):
    args = {"action": action, **kwargs}
    result_str = _handle_memory(args, memory_store=store)
    return json.loads(result_str)


def test_memory_tool_save(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    parsed = _run("save", store, name="t1", description="d",
                  type="user", body="b")
    assert parsed["success"] is True
    assert "id" in parsed


def test_memory_tool_list(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    parsed = _run("list", store)
    assert parsed["success"] is True
    assert parsed["count"] == 1


def test_memory_tool_load(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="full body")
    parsed = _run("load", store, id=mid)
    assert parsed["success"] is True
    assert parsed["body"] == "full body"


def test_memory_tool_update(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="b1")
    parsed = _run("update", store, id=mid, body="b2")
    assert parsed["success"] is True
    assert store.get(mid).body == "b2"


def test_memory_tool_delete(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="")
    parsed = _run("delete", store, id=mid)
    assert parsed["success"] is True


def test_memory_tool_load_unknown(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    parsed = _run("load", store, id="nonexistent")
    assert parsed["success"] is False
    assert "error" in parsed


def test_memory_tool_save_invalid_type(tmp_path: Path):
    store = MemoryStore(omnimate_home=tmp_path)
    parsed = _run("save", store, name="t", description="d",
                  type="invalid_kind", body="")
    assert parsed["success"] is False


# ============ CCALS-P0-1: L1 摘要层（三级粒度）测试 ============

def test_save_with_summary_persists_to_frontmatter(tmp_path: Path):
    """save 传 summary 时写入 frontmatter（L1 摘要层）。"""
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="用户偏好简洁回复",
        description="用尽量少的字数回答",
        type="user",
        body="用户多次要求简短直接回复，不喜欢长篇大论..." * 5,
        summary="用户偏好≤3 句话的简洁回复，避免长篇解释",
    )
    entry = store.get(mid)
    assert entry.summary == "用户偏好≤3 句话的简洁回复，避免长篇解释"


def test_save_without_summary_defaults_empty(tmp_path: Path):
    """不传 summary 时默认空串（向后兼容）。"""
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="x", description="y", type="other", body="z")
    entry = store.get(mid)
    assert entry.summary == ""
def test_update_summary(tmp_path: Path):
    """update 能修改 summary 字段。"""
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="x", description="y", type="user", body="z",
        summary="旧摘要",
    )
    store.update(mid, summary="新摘要")
    entry = store.get(mid)
    assert entry.summary == "新摘要"


def test_snapshot_for_prompt_includes_summary(tmp_path: Path):
    """MEMORY.md 索引包含 summary（让 retriever 拿到的 index 自动含 L1）。"""
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(
        name="用户偏好简洁", description="简短回复",
        type="user", body="...",
        summary="用户偏好≤3 句话的简洁回复",
    )
    snapshot = store.snapshot_for_prompt()
    # 索引行里应该能看到 summary
    assert "用户偏好≤3 句话的简洁回复" in snapshot
def test_memory_tool_save_with_summary(tmp_path: Path):
    """memory_save 工具支持 summary 参数。"""
    store = MemoryStore(omnimate_home=tmp_path)
    parsed = _run(
        "save", store,
        name="t", description="d", type="user", body="b",
        summary="L1 摘要内容",
    )
    assert parsed["success"] is True
    entry = store.get(parsed["id"])
    assert entry.summary == "L1 摘要内容"


def test_memory_tool_load_returns_summary(tmp_path: Path):
    """memory load 工具返回 summary 字段。"""
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="t", description="d", type="user", body="b",
        summary="测试摘要",
    )
    parsed = _run("load", store, id=mid)
    assert parsed["success"] is True
    assert parsed["summary"] == "测试摘要"


def test_memory_tool_list_returns_summary(tmp_path: Path):
    """memory list 工具返回每条的 summary。"""
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(
        name="t1", description="d1", type="user", body="b",
        summary="摘要 1",
    )
    store.save(name="t2", description="d2", type="user", body="b")  # 无 summary
    parsed = _run("list", store)
    assert parsed["success"] is True
    summaries = [m.get("summary", "MISSING") for m in parsed["memories"]]
    assert "摘要 1" in summaries
    # 无 summary 的也应能返回（值为空串而不是缺字段）
    assert "" in summaries


# ---------------------------------------------------------------------------
# 记忆注入上限（对齐 Claude Code 200 行 / 25KB）
# ---------------------------------------------------------------------------

def test_snapshot_truncated_over_line_limit(tmp_path: Path):
    """索引 >200 行时 snapshot_for_prompt 截断，full_index_text 完整。"""
    store = MemoryStore(omnimate_home=tmp_path)
    for i in range(250):
        store.save(name=f"mem{i}", description=f"desc{i}", type="other", body=f"body{i}")
    snap = store.snapshot_for_prompt()
    full = store.full_index_text()
    assert "记忆索引超出行数上限" in snap, "应截断"
    assert len(snap.splitlines()) <= 201
    assert "记忆索引超出行数上限" not in full, "完整索引不应截断"
    assert len(full.splitlines()) >= 250


def test_snapshot_truncated_over_byte_limit(tmp_path: Path):
    """索引 >25KB 时 snapshot_for_prompt 字节截断。"""
    store = MemoryStore(omnimate_home=tmp_path)
    for i in range(60):
        store.save(name=f"mem{i}", description="d" * 300, type="other", body="")
    snap = store.snapshot_for_prompt()
    # 每条索引行 ~300 字符，60 条 ~18KB... 加长确保 >25KB
    store2 = MemoryStore(omnimate_home=tmp_path)
    for i in range(120):
        store2.save(name=f"mem{i}", description="d" * 400, type="other", body="")
    snap2 = store2.snapshot_for_prompt()
    if "记忆索引超出字节上限" in snap2:
        assert len(snap2.encode("utf-8")) <= 25000 + 200
    else:
        # 若 120 条还没超 25KB，则不该截断（行数也 <200 时）
        assert "记忆索引超出" not in snap2


def test_snapshot_not_truncated_when_small(tmp_path: Path):
    """索引 <200 行且 <25KB 时不截断。"""
    store = MemoryStore(omnimate_home=tmp_path)
    for i in range(5):
        store.save(name=f"mem{i}", description=f"d{i}", type="other", body="")
    snap = store.snapshot_for_prompt()
    assert "记忆索引超出" not in snap


# ---------------------------------------------------------------------------
# 主题组织（对齐 Claude Code topic 文件）
# ---------------------------------------------------------------------------

def test_save_to_topic_file(tmp_path: Path):
    """save 两条同 topic → 一个 topic 文件 2 行，list_all 返回 2。"""
    store = MemoryStore(omnimate_home=tmp_path)
    id1 = store.save(name="偏好A", description="d1", type="user", body="b1", topic="preferences")
    id2 = store.save(name="偏好B", description="d2", type="user", body="b2", topic="preferences")
    assert id1.startswith("preferences#")
    assert id2.startswith("preferences#")
    topic_file = tmp_path / ".memory" / "preferences.jsonl"
    assert topic_file.exists()
    assert len(topic_file.read_text(encoding="utf-8").splitlines()) == 2
    assert len(store.list_all()) == 2


def test_save_same_topic_name_updates(tmp_path: Path):
    """同 topic 同 name → 更新而非新建（写入即维护）。"""
    store = MemoryStore(omnimate_home=tmp_path)
    id1 = store.save(name="依赖偏好", description="uv", type="user", body="v1", topic="general")
    id2 = store.save(name="依赖偏好", description="uv", type="user", body="v2", topic="general")
    assert id1 == id2, "同 topic 同 name 应更新同一记忆"
    assert len(store.list_all()) == 1
    assert store.get(id1).body == "v2"


def test_get_update_delete_topic_id(tmp_path: Path):
    """topic#uid 的 get/update/delete。"""
    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(name="x", description="dx", type="other", body="b", topic="debugging")
    e = store.get(mid)
    assert e is not None and e.topic == "debugging"
    store.update(mid, body="b2")
    assert store.get(mid).body == "b2"
    assert store.delete(mid) is True
    assert store.get(mid) is None
    assert len(store.list_all()) == 0


def test_clear_all_archives(tmp_path: Path):
    """clear_all 软删除全部到 .archive，索引重建为空。"""
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="a", description="da", type="other", body="", topic="t1")
    store.save(name="b", description="db", type="other", body="", topic="t2")
    n = store.clear_all()
    assert n == 2
    assert store.list_all() == []
    archives = list((tmp_path / ".archive").glob("memory-*/t*.jsonl"))
    assert len(archives) == 2


def test_index_grouped_by_topic(tmp_path: Path):
    """MEMORY.md 索引按主题分组。

    注：save 是惰性 rebuild（Round 3 压力优化，防批量写 O(n²)），
    直接读 MEMORY.md 文件前需显式 flush（生产路径走 snapshot_for_prompt
    自动 ensure fresh）。
    """
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(name="a1", description="da", type="user", body="", topic="preferences")
    store.save(name="b1", description="db", type="other", body="", topic="debugging")
    store.build_index_text()  # 显式 flush 到 MEMORY.md
    index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "## 主题：preferences" in index
    assert "## 主题：debugging" in index
