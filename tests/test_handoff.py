"""会话移交 bundle 测试。"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from agent.handoff import (
    HandoffStore,
    HandoffBundle,
    BundleNotFoundError,
    BundleCorruptedError,
    AmbiguousBundleIDError,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """提供临时 HandoffStore。"""
    return HandoffStore(tmp_path / ".handoff")


@pytest.fixture
def sample_transcript():
    """OpenAI Chat Completions 格式的 transcript 示例。"""
    return [
        {"role": "user", "content": "帮我写个 Python 脚本"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_abc", "type": "function",
             "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"stdout":"file1.txt"}'},
        {"role": "assistant", "content": "看到 file1.txt..."},
    ]


@pytest.fixture
def sample_model():
    return {"name": "deepseek-chat", "provider": "deepseek"}


# ---------------------------------------------------------------------------
# save / load 往返
# ---------------------------------------------------------------------------

def test_save_creates_valid_bundle(store, sample_transcript, sample_model):
    """保存后文件存在、format_version='1'、ID 非空、checksum 正确。"""
    bundle_id = store.save(
        transcript=sample_transcript,
        source_session_id="sess-123",
        model=sample_model,
        title="测试 bundle",
    )
    assert bundle_id  # 非空
    assert len(bundle_id) >= 16  # 时间戳+uuid 至少 19 字符

    # 文件存在
    bundle_path = Path(store._handoff_dir) / f"{bundle_id}.json"
    assert bundle_path.exists()

    # 解析内容
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert data["format_version"] == "1"
    assert data["bundle_id"] == bundle_id
    assert data["title"] == "测试 bundle"
    assert data["source_session_id"] == "sess-123"
    assert data["source_platform"] == "cli"
    assert data["model"] == sample_model
    assert data["transcript"] == sample_transcript
    assert data["handoff_state"] == "pending"
    assert data["schema_checksum"].startswith("sha256:")


def test_load_preserves_transcript_byte_for_byte(
    store, sample_transcript, sample_model
):
    """save 后立即 load，transcript 逐字段相等。"""
    bundle_id = store.save(
        transcript=sample_transcript,
        source_session_id=None,
        model=sample_model,
    )
    bundle = store.load(bundle_id)

    assert isinstance(bundle, HandoffBundle)
    assert bundle.bundle_id == bundle_id
    assert bundle.transcript == sample_transcript
    assert bundle.format_version == "1"
    assert bundle.source_platform == "cli"
    assert bundle.handoff_state == "pending"


def test_atomic_write_no_tmp_residue(store, sample_transcript, sample_model):
    """原子写入成功后无 .json.tmp 残留文件。"""
    store.save(transcript=sample_transcript, source_session_id=None, model=sample_model)

    tmp_files = list(Path(store._handoff_dir).glob("*.tmp"))
    assert tmp_files == []


def test_load_unknown_format_version_rejected(store, tmp_path):
    """format_version='2' 的 bundle 加载失败。"""
    # 手工写一个不兼容版本（文件名必须匹配 bundle_id 才能被 _resolve_id 找到）
    bad_path = tmp_path / ".handoff" / "0123456789012345abc.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text(json.dumps({
        "format_version": "2",
        "bundle_id": "0123456789012345abc",
        "created_at": "2026-07-17T12:00:00Z",
        "title": None,
        "source_session_id": None,
        "source_platform": "cli",
        "model": {"name": "x", "provider": "x"},
        "transcript": [],
        "memory_pointers": [],
        "skill_states": {},
        "todo_state": None,
        "task_pointers": [],
        "handoff_state": "pending",
        "notes": None,
        "schema_checksum": "sha256:0",
    }), encoding="utf-8")

    with pytest.raises(BundleCorruptedError):
        store.load("0123456789012345abc")


def test_load_checksum_mismatch_warns_but_loads(
    store, sample_transcript, sample_model, caplog
):
    """checksum 不匹配时记 warning 但仍加载。"""
    bundle_id = store.save(
        transcript=sample_transcript,
        source_session_id=None,
        model=sample_model,
    )
    # 篡改 transcript
    bundle_path = Path(store._handoff_dir) / f"{bundle_id}.json"
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    data["transcript"].append({"role": "user", "content": "篡改内容"})
    bundle_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    import logging
    with caplog.at_level(logging.WARNING):
        bundle = store.load(bundle_id)
    assert bundle is not None
    assert any("checksum" in rec.message.lower() for rec in caplog.records)


def test_load_nonexistent_raises(store):
    """加载不存在的 bundle 抛 BundleNotFoundError。"""
    with pytest.raises(BundleNotFoundError):
        store.load("nonexistent-id-xyz")


# ---------------------------------------------------------------------------
# list / resolve / delete
# ---------------------------------------------------------------------------

def test_list_bundles_sorted_by_created_at_desc(
    store, sample_transcript, sample_model
):
    """多个 bundle 按 created_at 倒序。"""
    # 顺序保存 3 个，时间戳应递增
    id_a = store.save(transcript=sample_transcript, source_session_id=None,
                      model=sample_model, title="A")
    id_b = store.save(transcript=sample_transcript, source_session_id=None,
                      model=sample_model, title="B")
    id_c = store.save(transcript=sample_transcript, source_session_id=None,
                      model=sample_model, title="C")

    bundles = store.list_bundles()
    assert len(bundles) == 3
    # 倒序：最新的在前
    assert bundles[0].title == "C"
    assert bundles[1].title == "B"
    assert bundles[2].title == "A"
    # 元信息完整
    assert bundles[0].message_count == len(sample_transcript)
    assert bundles[0].handoff_state == "pending"
    assert bundles[0].file_size > 0


def test_resolve_id_full_ulid(store, sample_transcript, sample_model):
    """完整 ID 精确匹配。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model)
    assert store.resolve_id(bundle_id) == bundle_id


def test_resolve_id_prefix_unique(store, sample_transcript, sample_model):
    """4+ 字符前缀唯一时正确解析。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model)
    prefix = bundle_id[:8]  # 取前 8 字符
    assert store.resolve_id(prefix) == bundle_id


def test_resolve_id_prefix_ambiguous(
    store, sample_transcript, sample_model, monkeypatch
):
    """前缀匹配多个 bundle 时抛 AmbiguousBundleIDError。"""
    # 强制两个 bundle 用相同前缀（mock _generate_id）
    call_count = [0]
    def fake_gen():
        call_count[0] += 1
        return f"20260717120000{call_count[0]:08d}"  # 前 14 字符相同
    monkeypatch.setattr("agent.handoff._generate_id", fake_gen)

    id1 = store.save(transcript=sample_transcript, source_session_id=None,
                     model=sample_model)
    id2 = store.save(transcript=sample_transcript, source_session_id=None,
                     model=sample_model)

    prefix = id1[:14]  # 两 ID 前 14 字符相同
    with pytest.raises(AmbiguousBundleIDError) as exc_info:
        store.resolve_id(prefix)
    # candidates 字段含两个 ID
    assert id1 in exc_info.value.candidates
    assert id2 in exc_info.value.candidates


def test_resolve_id_list_index(store, sample_transcript, sample_model):
    """序号 0 是最新（list 顺序）。"""
    store.save(transcript=sample_transcript, source_session_id=None,
               model=sample_model, title="old")
    newest_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model, title="new")

    assert store.resolve_id("0") == newest_id


def test_delete_is_soft_to_archive(store, sample_transcript, sample_model):
    """delete 把文件移到 .archive/，不硬删。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model)
    bundle_path = Path(store._handoff_dir) / f"{bundle_id}.json"
    assert bundle_path.exists()

    archived_path = store.delete(bundle_id)

    assert not bundle_path.exists()  # 原位置消失
    assert archived_path.exists()    # 归档存在
    assert archived_path.parent.name == ".archive"
    # list 不再包含
    assert all(b.bundle_id != bundle_id for b in store.list_bundles())
