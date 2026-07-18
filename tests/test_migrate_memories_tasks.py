"""迁移脚本测试：.memory/*.md 和 .tasks/*.json → SQLite。"""
import json
from pathlib import Path

import pytest

from agent.session_store import SessionStore
from agent.memory_store import _format_frontmatter
from scripts.migrate_memories_tasks_to_sqlite import (
    migrate_memories,
    migrate_tasks,
)


def _write_memory_md(path: Path, mid: str, name: str, description: str, body: str):
    """模拟 memory_store 的文件格式。"""
    import yaml
    from datetime import datetime
    meta = {
        "name": name,
        "description": description,
        "type": "user",
        "created_at": "2026-07-18T10:00:00",
        "updated_at": "2026-07-18T10:00:00",
    }
    content = _format_frontmatter(meta) + body
    path.write_text(content, encoding="utf-8")


def test_migrate_memories_copies_files(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    _write_memory_md(memory_dir / "m1.md", "m1", "用户偏好", "喜欢中文", "用中文回复。")
    _write_memory_md(memory_dir / "m2.md", "m2", "项目", "项目说明", "测试项目。")

    stats = migrate_memories(tmp_path, sqlite)
    assert stats["ok"] == 2
    assert stats["failed"] == 0

    # 校验
    m1 = sqlite.get_memory("m1")
    assert m1 is not None
    assert m1["name"] == "用户偏好"
    assert m1["description"] == "喜欢中文"


def test_migrate_memories_skips_no_frontmatter(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    # 无 frontmatter 的文件
    (memory_dir / "bad.md").write_text("just text, no yaml", encoding="utf-8")

    stats = migrate_memories(tmp_path, sqlite)
    assert stats["ok"] == 0
    assert stats["skipped"] == 1


def test_migrate_memories_missing_dir(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    stats = migrate_memories(tmp_path, sqlite)
    assert stats["ok"] == 0
    assert "reason" in stats


def test_migrate_tasks_copies_files(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    t1 = {
        "id": "task_abc123", "subject": "做某事",
        "description": "详细说明", "status": "pending",
        "owner": None, "created_at": "2026-07-18T10:00:00",
        "updated_at": "2026-07-18T10:00:00",
        "blocked_by": [], "comments": [],
    }
    (tasks_dir / "task_abc123.json").write_text(
        json.dumps(t1, ensure_ascii=False), encoding="utf-8"
    )

    stats = migrate_tasks(tmp_path, sqlite)
    assert stats["ok"] == 1
    assert stats["failed"] == 0

    loaded = sqlite.get_task("task_abc123")
    assert loaded is not None
    assert loaded["subject"] == "做某事"


def test_migrate_tasks_skips_invalid_json(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    (tasks_dir / "bad.json").write_text("not valid json {{{", encoding="utf-8")

    stats = migrate_tasks(tmp_path, sqlite)
    assert stats["ok"] == 0
    assert stats["failed"] == 1


def test_migrate_idempotent(tmp_path: Path):
    """重复迁移不重复插入（INSERT OR REPLACE）。"""
    sqlite = SessionStore(tmp_path / "test.db")
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    _write_memory_md(memory_dir / "m1.md", "m1", "A", "d", "")

    migrate_memories(tmp_path, sqlite)
    migrate_memories(tmp_path, sqlite)  # 再跑一次

    assert len(sqlite.list_memories()) == 1


def test_migrate_preserves_chinese_content(tmp_path: Path):
    """中文内容无损迁移。"""
    sqlite = SessionStore(tmp_path / "test.db")
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    _write_memory_md(
        memory_dir / "m1.md", "m1",
        "上海出差记录",
        "上周去上海见客户",
        "客户在上海中心，讨论了 AI 项目。",
    )
    migrate_memories(tmp_path, sqlite)

    m = sqlite.get_memory("m1")
    assert "上海" in m["name"]
    assert "客户" in m["body"]
    # trigram 搜索能命中
    hits = sqlite.search_memories("上海")
    assert any(h["id"] == "m1" for h in hits)
