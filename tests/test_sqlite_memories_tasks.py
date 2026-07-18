"""SQLite memories/tasks 入库 + trigram 搜索测试。

验证：
1. SessionStore 的 memories/tasks CRUD 正常
2. trigram 中文子串搜索能命中
3. MemoryStore + SessionStore 双写：写文件 + 写 SQLite
4. TaskStore + SessionStore 双写
5. search() 中文 fallback 走通
"""
import threading
from pathlib import Path

import pytest

from agent.session_store import (
    SessionStore,
    is_trigram_available,
    _contains_cjk,
)
from agent.memory_store import MemoryStore
from agent.task_store import TaskStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "test.db")


# ----------------------------------------------------------------------------
# memories CRUD
# ----------------------------------------------------------------------------

def test_save_and_get_memory(store: SessionStore):
    store.save_memory(
        id="m1", name="用户偏好", description="喜欢简洁回复",
        type="user", body="用户讨厌啰嗦的输出，喜欢直接给答案。",
        created_at="2026-07-18T10:00:00",
        updated_at="2026-07-18T10:00:00",
    )
    m = store.get_memory("m1")
    assert m is not None
    assert m["name"] == "用户偏好"
    assert m["type"] == "user"
    assert "讨厌啰嗦" in m["body"]


def test_list_memories_excludes_archived(store: SessionStore):
    store.save_memory(id="m1", name="A", description="d", type="other", body="",
                      created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00")
    store.save_memory(id="m2", name="B", description="d", type="other", body="",
                      created_at="2026-07-18T11:00:00", updated_at="2026-07-18T11:00:00")
    assert len(store.list_memories()) == 2
    assert store.archive_memory("m1") is True
    active = store.list_memories()
    assert len(active) == 1
    assert active[0]["id"] == "m2"
    # include_archived=True 看全部
    assert len(store.list_memories(include_archived=True)) == 2


def test_update_memory_partial(store: SessionStore):
    store.save_memory(id="m1", name="old", description="d", type="other", body="",
                      created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00")
    ok = store.update_memory("m1", name="new", updated_at="2026-07-18T12:00:00")
    assert ok is True
    m = store.get_memory("m1")
    assert m["name"] == "new"
    assert m["updated_at"] == "2026-07-18T12:00:00"


def test_update_memory_miss_returns_false(store: SessionStore):
    assert store.update_memory("nonexistent", name="x") is False


# ----------------------------------------------------------------------------
# memories trigram 搜索（CJK 关键）
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not is_trigram_available(), reason="FTS5 trigram 不可用")
def test_search_memories_chinese_substring(store: SessionStore):
    """trigram 应能搜中文子串（unicode61 不行）。"""
    store.save_memory(
        id="m1", name="系统配置", description="服务器的系统配置说明",
        type="project", body="配置文件在 /etc/app/config.yaml",
        created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00",
    )
    store.save_memory(
        id="m2", name="用户偏好", description="用户喜欢简洁",
        type="user", body="不要啰嗦",
        created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00",
    )
    # 搜 "配置" 应能命中 m1（description 和 body 都含）
    hits = store.search_memories("配置")
    assert len(hits) >= 1
    ids = [h["id"] for h in hits]
    assert "m1" in ids
    assert "m2" not in ids


@pytest.mark.skipif(not is_trigram_available(), reason="FTS5 trigram 不可用")
def test_search_memories_english(store: SessionStore):
    store.save_memory(
        id="m1", name="auth", description="authentication module",
        type="project", body="uses JWT tokens",
        created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00",
    )
    hits = store.search_memories("auth")
    assert any(h["id"] == "m1" for h in hits)


# ----------------------------------------------------------------------------
# tasks CRUD
# ----------------------------------------------------------------------------

def test_save_and_get_task(store: SessionStore):
    body = {
        "id": "t1", "subject": "修复 bug", "description": "登录页崩溃",
        "status": "pending", "owner": None,
        "created_at": "2026-07-18T10:00:00", "updated_at": "2026-07-18T10:00:00",
        "blocked_by": [], "comments": [],
    }
    import json
    store.save_task(
        id="t1", subject="修复 bug", description="登录页崩溃",
        status="pending", owner=None,
        created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00",
        blocked_by=[], body_json=json.dumps(body, ensure_ascii=False),
    )
    t = store.get_task("t1")
    assert t is not None
    assert t["subject"] == "修复 bug"
    assert t["status"] == "pending"


def test_list_tasks_status_filter(store: SessionStore):
    import json
    for i, status in enumerate(["pending", "completed", "deleted", "in_progress"]):
        body = {"id": f"t{i}", "subject": f"任务{i}", "status": status}
        store.save_task(
            id=f"t{i}", subject=f"任务{i}", description="",
            status=status, owner=None,
            created_at=f"2026-07-18T10:0{i}:00", updated_at=f"2026-07-18T10:0{i}:00",
            blocked_by=[], body_json=json.dumps(body, ensure_ascii=False),
        )
    # status=completed
    completed = store.list_tasks(status="completed")
    assert len(completed) == 1
    assert completed[0]["id"] == "t1"
    # 默认不含 deleted
    non_deleted = store.list_tasks()
    assert len(non_deleted) == 3
    # include_deleted=True
    all_tasks = store.list_tasks(include_deleted=True)
    assert len(all_tasks) == 4


def test_search_tasks_trigram(store: SessionStore):
    import json
    body = {
        "id": "t1", "subject": "数据库迁移",
        "description": "把 memories 表从 .md 搬到 SQLite",
        "status": "in_progress",
    }
    store.save_task(
        id="t1", subject="数据库迁移",
        description="把 memories 表从 .md 搬到 SQLite",
        status="in_progress", owner=None,
        created_at="2026-07-18T10:00:00", updated_at="2026-07-18T10:00:00",
        blocked_by=[], body_json=json.dumps(body, ensure_ascii=False),
    )
    hits = store.search_tasks("迁移")
    assert any(h["id"] == "t1" for h in hits)


# ----------------------------------------------------------------------------
# 双写集成：MemoryStore + SessionStore
# ----------------------------------------------------------------------------

def test_memory_store_dual_write(tmp_path: Path):
    """MemoryStore.save 同时写 .md 和 SQLite。"""
    sqlite = SessionStore(tmp_path / "test.db")
    ms = MemoryStore(harvil_home=tmp_path, sqlite_store=sqlite)

    mid = ms.save(
        name="用户偏好", description="喜欢中文回复",
        type="user", body="回复用中文，简洁。",
    )
    # 文件存在
    assert (tmp_path / ".memory" / f"{mid}.md").exists()
    # SQLite 里有
    row = sqlite.get_memory(mid)
    assert row is not None
    assert row["name"] == "用户偏好"
    assert row["type"] == "user"
    # trigram 搜得到
    hits = sqlite.search_memories("中文")
    assert any(h["id"] == mid for h in hits)


def test_memory_store_dual_delete(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    ms = MemoryStore(harvil_home=tmp_path, sqlite_store=sqlite)
    mid = ms.save(name="A", description="d", type="other", body="")
    assert sqlite.get_memory(mid) is not None
    ms.delete(mid)
    # SQLite 标记 archived
    row = sqlite.get_memory(mid)
    assert row is not None
    assert row["archived"] == 1
    # 不在 list_memories 默认结果里
    assert all(r["id"] != mid for r in sqlite.list_memories())


def test_memory_store_without_sqlite_still_works(tmp_path: Path):
    """纯文件模式（向后兼容）：不传 sqlite_store 也能跑。"""
    ms = MemoryStore(harvil_home=tmp_path)
    mid = ms.save(name="A", description="d", type="other", body="x")
    assert ms.get(mid) is not None
    assert ms.get(mid).body == "x"


# ----------------------------------------------------------------------------
# 双写集成：TaskStore + SessionStore
# ----------------------------------------------------------------------------

def test_task_store_dual_write(tmp_path: Path):
    sqlite = SessionStore(tmp_path / "test.db")
    ts = TaskStore(harvil_home=tmp_path, sqlite_store=sqlite)
    task = ts.create(subject="任务A", description="一个测试任务")
    # 文件存在
    assert (tmp_path / ".tasks" / f"{task['id']}.json").exists()
    # SQLite 里有
    sqlite_task = sqlite.get_task(task["id"])
    assert sqlite_task is not None
    assert sqlite_task["subject"] == "任务A"
    # trigram 搜得到
    hits = sqlite.search_tasks("任务")
    assert any(h["id"] == task["id"] for h in hits)


def test_task_store_attach_sqlite_after_init(tmp_path: Path):
    """attach_sqlite 运行时注入。"""
    ts = TaskStore(harvil_home=tmp_path)
    assert ts._sqlite is None
    sqlite = SessionStore(tmp_path / "test.db")
    ts.attach_sqlite(sqlite)
    assert ts._sqlite is sqlite
    # 创建任务后双写生效
    task = ts.create(subject="后注入的任务", description="")
    assert sqlite.get_task(task["id"]) is not None


def test_task_store_without_sqlite_still_works(tmp_path: Path):
    ts = TaskStore(harvil_home=tmp_path)
    task = ts.create(subject="任务", description="d")
    assert ts.get(task["id"]) is not None


# ----------------------------------------------------------------------------
# search() 中文 fallback
# ----------------------------------------------------------------------------

def test_contains_cjk():
    assert _contains_cjk("hello") is False
    assert _contains_cjk("你好") is True
    assert _contains_cjk("mix 中文") is True


def test_search_messages_cjk_fallback(tmp_path: Path):
    """session_store.search() 对中文子串的 trigram fallback。"""
    store = SessionStore(tmp_path / "test.db")
    sid = store.create_session(model="test", provider="test")
    store.append_message(
        session_id=sid, role="user",
        content="今天的系统配置说明在哪个文件？",
    )
    store.append_message(
        session_id=sid, role="assistant",
        content="配置文件在 /etc/app/config.yaml。",
    )
    # 搜 "配置"（unicode61 找不到，trigram fallback 走 LIKE）
    hits = store.search("配置")
    assert len(hits) >= 1
    # 至少一条命中含 "配置"
    assert any("配置" in h["content"] for h in hits)
