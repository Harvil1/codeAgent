"""多文件记忆系统测试。"""
from pathlib import Path

import pytest

from agent.memory_store import MemoryStore, MemoryEntry


def test_save_creates_file_and_updates_index(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="用户偏好简洁回复",
        description="用尽量少的字数回答",
        type="user",
        body="用户多次要求简短直接回复",
    )
    assert mid  # non-empty string
    # 文件应存在
    mem_file = tmp_path / ".memory" / f"{mid}.md"
    assert mem_file.exists()
    # 索引文件 MEMORY.md 也应被更新
    index_text = tmp_path / "MEMORY.md"
    assert index_text.exists()
    assert mid in index_text.read_text(encoding="utf-8")


def test_save_minimal_fields(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="x", description="y", type="other",
    )  # 无 body
    entry = store.get(mid)
    assert entry.body == ""


def test_save_invalid_type_raises(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    with pytest.raises(ValueError):
        store.save(name="x", description="y", type="invalid_kind", body="")


def test_get_returns_entry(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b")
    entry = store.get(mid)
    assert entry.id == mid
    assert entry.name == "t1"
    assert entry.type == "user"


def test_get_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.get("nonexistent") is None


def test_list_all_returns_entries(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    store.save(name="t2", description="d", type="project", body="")
    all_entries = store.list_all()
    assert len(all_entries) == 2


def test_update_modifies_fields(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b1")
    store.update(mid, name="t1-new", body="b2")
    entry = store.get(mid)
    assert entry.name == "t1-new"
    assert entry.body == "b2"
    assert entry.description == "d"  # 未改


def test_update_unknown_raises(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    with pytest.raises(KeyError):
        store.update("nonexistent", body="x")


def test_delete_moves_to_archive(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b")
    ok = store.delete(mid)
    assert ok is True
    # 主目录文件不存在
    assert not (tmp_path / ".memory" / f"{mid}.md").exists()
    # archive 下能找到
    archives = list((tmp_path / ".archive").glob("memory-*/" + f"{mid}.md"))
    assert len(archives) >= 1


def test_delete_unknown_returns_false(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.delete("nonexistent") is False


def test_load_body_returns_content(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="完整内容")
    assert store.load_body(mid) == "完整内容"


def test_load_body_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.load_body("nonexistent") is None


def test_snapshot_for_prompt_returns_index_text(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="描述1", type="user", body="")
    store.save(name="t2", description="描述2", type="project", body="")
    snap = store.snapshot_for_prompt()
    assert "t1" in snap
    assert "描述1" in snap
    assert "t2" in snap


def test_index_rebuilt_on_startup(tmp_path: Path):
    """新建 store 时扫描 .memory/ 重建索引。"""
    # 先用 store1 写两个 memory
    store1 = MemoryStore(harvil_home=tmp_path)
    mid1 = store1.save(name="t1", description="d", type="user", body="")
    mid2 = store1.save(name="t2", description="d", type="project", body="")
    # 再开一个 store（模拟下次会话），应能看到两条
    store2 = MemoryStore(harvil_home=tmp_path)
    all_entries = store2.list_all()
    assert len(all_entries) == 2
    assert {e.id for e in all_entries} == {mid1, mid2}


def test_migrate_legacy_archives_old_files(tmp_path: Path):
    """启动时检测旧 MEMORY.md / USER.md 格式（无 frontmatter），备份到 .archive/。"""
    # 写一个旧格式 MEMORY.md
    (tmp_path / "MEMORY.md").write_text(
        "# Agent Memory\n\n- 老记忆 1\n- 老记忆 2\n",
        encoding="utf-8",
    )
    (tmp_path / "USER.md").write_text(
        "# User Profile\n\n- 老用户画像\n",
        encoding="utf-8",
    )

    # 启动 store 应备份老文件 + 新建空索引
    store = MemoryStore(harvil_home=tmp_path)
    # 老文件已被新（空）索引覆盖
    new_index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "老记忆" not in new_index
    # archive 里有备份
    archives = list((tmp_path / ".archive").glob("legacy-memory-*/MEMORY.md"))
    assert len(archives) >= 1
    assert "老记忆" in archives[0].read_text(encoding="utf-8")


def test_malformed_frontmatter_skipped(tmp_path: Path):
    """frontmatter 解析失败的文件跳过 + log warning。"""
    store = MemoryStore(harvil_home=tmp_path)
    # 写一个合法的
    good_mid = store.save(name="good", description="d", type="user", body="")
    # 直接写一个坏文件到 .memory/
    bad_file = tmp_path / ".memory" / "bad_mid.md"
    bad_file.write_text(
        "---\ninvalid: yaml: content\n---\nbody",
        encoding="utf-8",
    )
    # 重新加载（模拟下次会话）
    store2 = MemoryStore(harvil_home=tmp_path)
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
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("save", store, name="t1", description="d",
                  type="user", body="b")
    assert parsed["success"] is True
    assert "id" in parsed


def test_memory_tool_list(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    parsed = _run("list", store)
    assert parsed["success"] is True
    assert parsed["count"] == 1


def test_memory_tool_load(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="full body")
    parsed = _run("load", store, id=mid)
    assert parsed["success"] is True
    assert parsed["body"] == "full body"


def test_memory_tool_update(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="b1")
    parsed = _run("update", store, id=mid, body="b2")
    assert parsed["success"] is True
    assert store.get(mid).body == "b2"


def test_memory_tool_delete(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="")
    parsed = _run("delete", store, id=mid)
    assert parsed["success"] is True


def test_memory_tool_load_unknown(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("load", store, id="nonexistent")
    assert parsed["success"] is False
    assert "error" in parsed


def test_memory_tool_save_invalid_type(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("save", store, name="t", description="d",
                  type="invalid_kind", body="")
    assert parsed["success"] is False


# ============ CCALS-P0-1: L1 摘要层（三级粒度）测试 ============

def test_save_with_summary_persists_to_frontmatter(tmp_path: Path):
    """save 传 summary 时写入 frontmatter（L1 摘要层）。"""
    store = MemoryStore(harvil_home=tmp_path)
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
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="x", description="y", type="other", body="z")
    entry = store.get(mid)
    assert entry.summary == ""


def test_legacy_file_without_summary_loads_as_empty(tmp_path: Path):
    """老格式记忆文件（frontmatter 无 summary 字段）兼容读为空。"""
    store = MemoryStore(harvil_home=tmp_path)
    # 手工构造一个无 summary 的老文件
    legacy_file = tmp_path / ".memory" / "legacy.md"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(
        "---\n"
        "name: 老记忆\ndescription: 老格式无 summary\ntype: user\n"
        "created_at: 2026-01-01T00:00:00\nupdated_at: 2026-01-01T00:00:00\n"
        "---\n\n老 body 内容",
        encoding="utf-8",
    )
    # 触发一次扫描
    entries = store.list_all()
    legacy = [e for e in entries if e.id == "legacy"][0]
    assert legacy.summary == ""
    assert legacy.body == "老 body 内容"


def test_update_summary(tmp_path: Path):
    """update 能修改 summary 字段。"""
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="x", description="y", type="user", body="z",
        summary="旧摘要",
    )
    store.update(mid, summary="新摘要")
    entry = store.get(mid)
    assert entry.summary == "新摘要"


def test_snapshot_for_prompt_includes_summary(tmp_path: Path):
    """MEMORY.md 索引包含 summary（让 retriever 拿到的 index 自动含 L1）。"""
    store = MemoryStore(harvil_home=tmp_path)
    store.save(
        name="用户偏好简洁", description="简短回复",
        type="user", body="...",
        summary="用户偏好≤3 句话的简洁回复",
    )
    snapshot = store.snapshot_for_prompt()
    # 索引行里应该能看到 summary
    assert "用户偏好≤3 句话的简洁回复" in snapshot


def test_snapshot_for_prompt_legacy_without_summary_still_works(tmp_path: Path):
    """老文件无 summary 时索引不崩（snapshot 仍可读）。"""
    store = MemoryStore(harvil_home=tmp_path)
    legacy_file = tmp_path / ".memory" / "legacy.md"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text(
        "---\nname: 老\ndescription: 老描述\ntype: user\n"
        "created_at: 2026-01-01T00:00:00\nupdated_at: 2026-01-01T00:00:00\n---\n\nbody",
        encoding="utf-8",
    )
    # legacy 文件在 store init 后建，需手工重建索引
    store.build_index_text()
    snapshot = store.snapshot_for_prompt()
    assert "老描述" in snapshot


def test_memory_tool_save_with_summary(tmp_path: Path):
    """memory_save 工具支持 summary 参数。"""
    store = MemoryStore(harvil_home=tmp_path)
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
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="t", description="d", type="user", body="b",
        summary="测试摘要",
    )
    parsed = _run("load", store, id=mid)
    assert parsed["success"] is True
    assert parsed["summary"] == "测试摘要"


def test_memory_tool_list_returns_summary(tmp_path: Path):
    """memory list 工具返回每条的 summary。"""
    store = MemoryStore(harvil_home=tmp_path)
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
