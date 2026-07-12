"""记忆系统测试。"""

import json
import tempfile
from pathlib import Path

import pytest

from agent.memory_store import MemoryStore
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from plugins.memory.simple_provider import SimpleProvider


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """提供临时目录的 MemoryStore。"""
    return MemoryStore(tmp_path)


def test_memory_store_add(tmp_store):
    assert tmp_store.add("memory", "用户偏好简洁回复")
    assert "用户偏好简洁回复" in tmp_store.memory_entries


def test_memory_store_add_idempotent(tmp_store):
    """重复 add 同一条记忆是幂等的。"""
    tmp_store.add("memory", "相同内容")
    tmp_store.add("memory", "相同内容")
    assert tmp_store.memory_entries.count("相同内容") == 1


def test_memory_store_add_user(tmp_store):
    assert tmp_store.add("user", "资深后端工程师")
    assert "资深后端工程师" in tmp_store.user_entries


def test_memory_store_add_invalid_target(tmp_store):
    assert tmp_store.add("invalid", "xxx") is False


def test_memory_store_char_limit(tmp_path):
    """超出字数限制时拒绝写入。"""
    store = MemoryStore(tmp_path, memory_char_limit=10)
    assert store.add("memory", "12345")  # 5 字符，OK
    assert store.add("memory", "1234567890") is False  # 超限


def test_memory_store_replace(tmp_store):
    tmp_store.add("memory", "旧内容")
    assert tmp_store.replace("memory", "旧内容", "新内容")
    assert "新内容" in tmp_store.memory_entries
    assert "旧内容" not in tmp_store.memory_entries


def test_memory_store_remove(tmp_store):
    tmp_store.add("memory", "要删除的")
    assert tmp_store.remove("memory", "要删除的")
    assert "要删除的" not in tmp_store.memory_entries


def test_memory_store_modify_dispatch(tmp_store):
    """modify() 统一入口能分发到 add/replace/remove。"""
    assert tmp_store.modify("add", "memory", "条目A")
    assert tmp_store.modify("replace", "memory", "条目B", old_content="条目A")
    assert "条目B" in tmp_store.memory_entries
    assert tmp_store.modify("remove", "memory", "条目B")
    assert "条目B" not in tmp_store.memory_entries


def test_memory_store_persist(tmp_path):
    """写入后落盘，重新加载能读到。"""
    store1 = MemoryStore(tmp_path)
    store1.add("memory", "持久化测试")

    store2 = MemoryStore(tmp_path)
    assert "持久化测试" in store2.memory_entries


def test_memory_store_snapshot(tmp_store):
    tmp_store.add("memory", "记忆条目")
    tmp_store.add("user", "用户条目")
    snapshot = tmp_store.snapshot_for_prompt()
    assert "记忆条目" in snapshot
    assert "用户条目" in snapshot


def test_memory_store_empty_snapshot(tmp_store):
    """空记忆的 snapshot 是空字符串。"""
    assert tmp_store.snapshot_for_prompt() == ""


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

def test_memory_manager_no_provider():
    """没有外部 provider 时，所有方法都是安全的 no-op。"""
    store = MemoryStore(tempfile.mkdtemp())
    mgr = MemoryManager(store)
    mgr.initialize("session-1")
    assert mgr.build_system_prompt() == ""  # 空记忆
    assert mgr.prefetch_all("query") == ""
    mgr.sync_all("u", "a")  # 不应该报错
    mgr.shutdown()


def test_memory_manager_with_simple_provider(tmp_path):
    """带 SimpleProvider 的 manager 能 sync_turn。"""
    store = MemoryStore(tmp_path)
    provider = SimpleProvider(tmp_path)
    mgr = MemoryManager(store, external_provider=provider)

    mgr.initialize("session-1")
    mgr.sync_all("用户问题", "助手回答")

    # 等待后台线程完成
    mgr._sync_executor.shutdown(wait=True)

    # 验证 provider 持久化了
    assert provider._store_path.exists()
    turns = json.loads(provider._store_path.read_text(encoding="utf-8"))
    assert len(turns) == 1
    assert turns[0]["user"] == "用户问题"
    assert turns[0]["assistant"] == "助手回答"


# ---------------------------------------------------------------------------
# memory 工具集成（通过 registry）
# ---------------------------------------------------------------------------

def test_memory_tool_with_store(tmp_path):
    """memory 工具能通过 kwargs 接收 memory_store 并调用 modify。"""
    from tools.registry import registry

    store = MemoryStore(tmp_path)
    result = registry.dispatch(
        "memory",
        {"action": "add", "target": "memory", "content": "通过工具写入"},
        memory_store=store,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert "通过工具写入" in store.memory_entries


def test_memory_tool_without_store():
    """没有 memory_store 时返回明确错误。"""
    from tools.registry import registry
    result = registry.dispatch(
        "memory",
        {"action": "add", "target": "memory", "content": "xxx"},
        # 不传 memory_store
    )
    data = json.loads(result)
    assert data["success"] is False
    assert "未初始化" in data["error"]
