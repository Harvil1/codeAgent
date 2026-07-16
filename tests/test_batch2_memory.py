"""batch2-T1: 自动记忆提取（on_pre_compress）测试。"""
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_mock_llm(returned_text: str):
    """构造 mock LLM client（chat_completions 返回指定 content）。"""
    m = MagicMock()
    m.chat_completions.return_value.choices = [
        MagicMock(message=MagicMock(content=returned_text))
    ]
    return m


def _make_messages(n: int = 8) -> list:
    """构造 n 条 messages。"""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"消息 {i}：用户喜欢深色主题。"})
    return msgs


# ---------------------------------------------------------------------------
# 无 LLM client 时仍为 no-op
# ---------------------------------------------------------------------------

def test_on_pre_compress_no_llm_client_is_noop(tmp_path):
    """没有 llm_client 时 on_pre_compress 是 no-op。"""
    store = MemoryStore(harvil_home=tmp_path)
    mm = MemoryManager(memory_store=store)
    mm.on_pre_compress(None, _make_messages(8))
    # 后台线程可能还在跑，等一下
    time.sleep(0.1)
    assert store.list_all() == []


def test_on_pre_compress_short_messages_skipped(tmp_path):
    """messages 少于 6 条时跳过。"""
    store = MemoryStore(harvil_home=tmp_path)
    llm = _make_mock_llm('[]')
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")
    mm.on_pre_compress(None, _make_messages(3))
    time.sleep(0.2)
    llm.chat_completions.assert_not_called()


def test_on_pre_compress_empty_messages_skipped(tmp_path):
    """空 messages 时跳过。"""
    store = MemoryStore(harvil_home=tmp_path)
    llm = _make_mock_llm('[]')
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")
    mm.on_pre_compress(None, [])
    time.sleep(0.2)
    llm.chat_completions.assert_not_called()


# ---------------------------------------------------------------------------
# LLM 提取 → memory_store.save
# ---------------------------------------------------------------------------

def test_on_pre_compress_extracts_and_saves(tmp_path):
    """LLM 返回事实 → memory_store.save 被调用。"""
    store = MemoryStore(harvil_home=tmp_path)
    facts = json.dumps([
        {"type": "user", "name": "dark-theme",
         "description": "用户喜欢深色主题", "body": "用户偏好深色 IDE 主题"},
        {"type": "project", "name": "python-version",
         "description": "项目用 Python 3.11", "body": "项目代码库用 Python 3.11"},
    ])
    llm = _make_mock_llm(facts)
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    # 等后台线程完成
    time.sleep(0.5)

    entries = store.list_all()
    assert len(entries) == 2
    names = {e.name for e in entries}
    assert "dark-theme" in names
    assert "python-version" in names


def test_on_pre_compress_empty_facts_no_save(tmp_path):
    """LLM 返回 [] → 不保存。"""
    store = MemoryStore(harvil_home=tmp_path)
    llm = _make_mock_llm('[]')
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.3)

    assert store.list_all() == []


def test_on_pre_compress_invalid_type_falls_back_to_other(tmp_path):
    """LLM 返回非法 type → fallback 到 other。"""
    store = MemoryStore(harvil_home=tmp_path)
    facts = json.dumps([
        {"type": "invalid_type", "name": "test", "description": "desc", "body": "body"},
    ])
    llm = _make_mock_llm(facts)
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.3)

    entries = store.list_all()
    assert len(entries) == 1
    assert entries[0].type == "other"


def test_on_pre_compress_llm_failure_is_fail_open(tmp_path):
    """LLM 抛异常 → 不影响压缩（fail-open）。"""
    store = MemoryStore(harvil_home=tmp_path)
    llm = MagicMock()
    llm.chat_completions.side_effect = RuntimeError("API down")
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    # 不应该抛异常
    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.3)
    assert store.list_all() == []


def test_on_pre_compress_malformed_json_no_save(tmp_path):
    """LLM 返回非 JSON → 不保存。"""
    store = MemoryStore(harvil_home=tmp_path)
    llm = _make_mock_llm("这不是 JSON")
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.3)
    assert store.list_all() == []


def test_on_pre_compress_max_5_facts(tmp_path):
    """LLM 返回超过 5 条 → 只保存 5 条。"""
    store = MemoryStore(harvil_home=tmp_path)
    facts = json.dumps([
        {"type": "other", "name": f"fact-{i}",
         "description": f"desc-{i}", "body": f"body-{i}"}
        for i in range(10)
    ])
    llm = _make_mock_llm(facts)
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.5)

    entries = store.list_all()
    assert len(entries) <= 5


def test_on_pre_compress_skip_facts_without_name(tmp_path):
    """缺少 name 的事实被跳过。"""
    store = MemoryStore(harvil_home=tmp_path)
    facts = json.dumps([
        {"type": "other", "description": "有 desc 无 name", "body": "body"},
        {"type": "other", "name": "ok", "description": "正常", "body": "ok body"},
    ])
    llm = _make_mock_llm(facts)
    mm = MemoryManager(memory_store=store, llm_client=llm, llm_model="test")

    mm.on_pre_compress(None, _make_messages(8))
    time.sleep(0.3)

    entries = store.list_all()
    assert len(entries) == 1
    assert entries[0].name == "ok"


# ---------------------------------------------------------------------------
# _serialize_messages / _parse_facts 单元测试
# ---------------------------------------------------------------------------

def test_serialize_messages_basic():
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    text = MemoryManager._serialize_messages(msgs)
    assert "[user]" in text
    assert "你好" in text


def test_serialize_messages_skips_empty():
    msgs = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": None},
    ]
    text = MemoryManager._serialize_messages(msgs)
    assert text.strip() == ""


def test_parse_facts_valid_json():
    content = '[{"type":"user","name":"x","description":"d"}]'
    result = MemoryManager._parse_facts(content)
    assert len(result) == 1
    assert result[0]["name"] == "x"


def test_parse_facts_json_in_text():
    content = '这是结果：[{"type":"other","name":"y","description":"d"}] 完成'
    result = MemoryManager._parse_facts(content)
    assert len(result) == 1


def test_parse_facts_invalid_returns_empty():
    assert MemoryManager._parse_facts("no json here") == []
    assert MemoryManager._parse_facts("") == []
