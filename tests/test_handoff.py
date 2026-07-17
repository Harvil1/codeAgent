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


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------

def test_export_import_roundtrip(
    store, sample_transcript, sample_model, tmp_path
):
    """export 后 import 到新 store，内容一致。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None,
                           model=sample_model, title="原 bundle")

    dest = tmp_path / "exported.json"
    returned_path = store.export_to(bundle_id, dest)
    assert returned_path == dest
    assert dest.exists()

    # 用新 store 导入
    new_store = HandoffStore(tmp_path / "new_handoff")
    new_id = new_store.import_from(dest)

    # 加载对比内容
    orig = store.load(bundle_id)
    imported = new_store.load(new_id)

    assert imported.transcript == orig.transcript
    assert imported.title == orig.title
    assert imported.model == orig.model
    assert imported.source_platform == orig.source_platform  # 不改 source_platform


def test_import_with_existing_id_regenerates_ulid(
    store, sample_transcript, sample_model, tmp_path
):
    """同 bundle_id 二次导入得到新 ULID，但内容一致。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model)
    bundle_path = Path(store._handoff_dir) / f"{bundle_id}.json"

    # 再次导入相同文件
    new_id = store.import_from(bundle_path)
    assert new_id != bundle_id  # 重新生成

    # 内容一致
    a = store.load(bundle_id)
    b = store.load(new_id)
    assert a.transcript == b.transcript


# ---------------------------------------------------------------------------
# 密钥扫描
# ---------------------------------------------------------------------------

def test_secret_detection_rejects_sk_key(store, sample_model):
    """含 sk- 开头的 OpenAI/DeepSeek key 触发 SecretDetectedError。"""
    from agent.handoff import SecretDetectedError
    transcript = [
        {"role": "user", "content": "我的 API key 是 sk-abc123def456ghi789jkl012mno345pqr678"},
    ]
    with pytest.raises(SecretDetectedError) as exc_info:
        store.save(transcript=transcript, source_session_id=None, model=sample_model)
    # matches 含命中信息
    assert len(exc_info.value.matches) >= 1


def test_secret_detection_bearer_token(store, sample_model):
    from agent.handoff import SecretDetectedError
    transcript = [
        {"role": "tool", "tool_call_id": "x",
         "content": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturepart"},
    ]
    with pytest.raises(SecretDetectedError):
        store.save(transcript=transcript, source_session_id=None, model=sample_model)


def test_secret_detection_api_key_pattern(store, sample_model):
    from agent.handoff import SecretDetectedError
    transcript = [
        {"role": "assistant", "content": '配置：api_key="AKIAIOSFODNN7EXAMPLE123456"'},
    ]
    with pytest.raises(SecretDetectedError):
        store.save(transcript=transcript, source_session_id=None, model=sample_model)


def test_secret_detection_pem_private_key(store, sample_model):
    from agent.handoff import SecretDetectedError
    transcript = [
        {"role": "user", "content": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."},
    ]
    with pytest.raises(SecretDetectedError):
        store.save(transcript=transcript, source_session_id=None, model=sample_model)


def test_secret_detection_allow_secrets_flag(
    store, sample_model, caplog
):
    """allow_secrets=True 时跳过扫描（不推荐，API 层逃生口）。"""
    transcript = [
        {"role": "user", "content": "key: sk-abc123def456ghi789jkl012mno345pqr678"},
    ]
    bundle_id = store.save(
        transcript=transcript, source_session_id=None,
        model=sample_model, allow_secrets=True,
    )
    assert bundle_id  # 不抛


# ---------------------------------------------------------------------------
# 大小上限
# ---------------------------------------------------------------------------

def test_size_limit_rejects_large_bundle(store, sample_model):
    """构造 >10MB transcript 触发 BundleTooLargeError。"""
    from agent.handoff import BundleTooLargeError
    # 11MB 文本（约 1100 万字符）
    big_content = "x" * (11 * 1024 * 1024)
    transcript = [{"role": "user", "content": big_content}]
    with pytest.raises(BundleTooLargeError):
        store.save(transcript=transcript, source_session_id=None, model=sample_model)


# ---------------------------------------------------------------------------
# mark_completed
# ---------------------------------------------------------------------------

def test_mark_completed_updates_state(store, sample_transcript, sample_model):
    """mark_completed 把 handoff_state 改为 'completed'。"""
    bundle_id = store.save(transcript=sample_transcript,
                           source_session_id=None, model=sample_model)
    assert store.load(bundle_id).handoff_state == "pending"

    store.mark_completed(bundle_id)

    assert store.load(bundle_id).handoff_state == "completed"


# ---------------------------------------------------------------------------
# CLI 集成
# ---------------------------------------------------------------------------

def test_handle_handoff_save_command(tmp_path, monkeypatch):
    """mock RuntimeContext + AIAgent，验证 /handoff save 流程。"""
    from cli import RuntimeContext, _handle_command

    # 构造最小 rt mock
    handoff_dir = tmp_path / ".handoff"

    class FakeAgent:
        def __init__(self):
            self.conversation_history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            self.session_id = "fake-session-id"
            self.config = {"model": {"name": "deepseek-chat", "provider": "deepseek"}}
        def invalidate_system_prompt(self):
            pass

    class FakeRT:
        def __init__(self):
            self.home = tmp_path
            self.agent = FakeAgent()
            self.session_id = "fake-session-id"
            self.config = {"model": {"name": "deepseek-chat", "provider": "deepseek"}}
            from agent.handoff import HandoffStore
            self.handoff_store = HandoffStore(handoff_dir)

    rt = FakeRT()
    # 直接调用 _handle_command（绕过交互循环）
    handled = _handle_command("/handoff save 测试标题", rt)
    assert handled is True

    # bundle 已生成
    bundles = rt.handoff_store.list_bundles()
    assert len(bundles) == 1
    assert bundles[0].title == "测试标题"
    assert bundles[0].message_count == 2


def test_handle_handoff_list_command(tmp_path, capsys):
    """/handoff list 输出表格。"""
    from cli import RuntimeContext, _handle_command
    from agent.handoff import HandoffStore

    handoff_dir = tmp_path / ".handoff"
    store = HandoffStore(handoff_dir)
    store.save(
        transcript=[{"role": "user", "content": "hi"}],
        source_session_id=None,
        model={"name": "x", "provider": "x"},
        title="测试",
    )

    class FakeRT:
        def __init__(self):
            self.handoff_store = store

    rt = FakeRT()
    handled = _handle_command("/handoff list", rt)
    assert handled is True

    out = capsys.readouterr().out
    assert "测试" in out


def test_handle_handoff_no_subcommand_shows_help(tmp_path, capsys):
    """/handoff 无参数显示子命令帮助。"""
    from cli import _handle_command

    class FakeRT:
        pass

    handled = _handle_command("/handoff", FakeRT())
    assert handled is True
    out = capsys.readouterr().out
    assert "save" in out
    assert "list" in out
    assert "load" in out


def test_handle_handoff_load_overwrites_history(tmp_path, monkeypatch):
    """/handoff load <id> 替换 agent.conversation_history。"""
    from cli import _handle_command
    from agent.handoff import HandoffStore

    handoff_dir = tmp_path / ".handoff"
    store = HandoffStore(handoff_dir)
    new_bundle_transcript = [
        {"role": "user", "content": "loaded msg 1"},
        {"role": "assistant", "content": "loaded reply"},
    ]
    bundle_id = store.save(
        transcript=new_bundle_transcript,
        source_session_id=None,
        model={"name": "x", "provider": "x"},
    )

    class FakeAgent:
        def __init__(self):
            self.conversation_history = [{"role": "user", "content": "old"}]
            self.session_id = "old-session"
        def invalidate_system_prompt(self):
            pass

    class FakeRT:
        def __init__(self):
            self.agent = FakeAgent()
            self.handoff_store = store
            # mock SessionStore（create_session 返回新 id）
            class FakeSessionStore:
                def create_session(self, **kwargs):
                    return "new-session-id"
            self.session_store = FakeSessionStore()

    rt = FakeRT()
    # mock 用户确认（输入 y）
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "y")

    handled = _handle_command(f"/handoff load {bundle_id}", rt)
    assert handled is True

    # agent 历史已被替换
    assert rt.agent.conversation_history == new_bundle_transcript
    assert rt.agent.session_id == "new-session-id"
