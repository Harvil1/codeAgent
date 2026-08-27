"""跨项目会话恢复测试。"""
import json
from pathlib import Path
from unittest.mock import MagicMock

from agent.cross_project import (
    auto_save_current_session,
    list_recent_bundles_across_projects,
)
from agent.handoff import HandoffStore


def test_save_with_source_cwd(tmp_path: Path):
    """save 支持 source_cwd 和 auto_saved kwargs。"""
    store = HandoffStore(tmp_path)
    bid = store.save(
        transcript=[{"role": "user", "content": "hi"}],
        source_session_id="s1",
        model={"main": "deepseek-chat"},
        title="测试",
        source_cwd="/path/to/project",
        auto_saved=True,
    )
    # 读出来验证
    bundle = json.loads((tmp_path / f"{bid}.json").read_text(encoding="utf-8"))
    assert bundle["source_cwd"] == "/path/to/project"
    assert bundle["auto_saved"] is True


def test_list_bundles_includes_source_cwd(tmp_path: Path):
    """list_bundles 返回的 meta 含 source_cwd 字段。"""
    store = HandoffStore(tmp_path)
    store.save(
        transcript=[],
        source_session_id=None,
        model={},
        source_cwd="/proj/A",
    )
    metas = store.list_bundles()
    assert metas[0].source_cwd == "/proj/A"
    assert metas[0].auto_saved is False  # 默认


def test_old_bundle_backward_compatible(tmp_path: Path):
    """旧 bundle（缺 source_cwd / auto_saved）能正确读出（默认值）。"""
    bid = "oldbundle001"
    bundle_dict = {
        "format_version": "1",
        "bundle_id": bid,
        "created_at": "2026-01-01T00:00:00.000Z",
        "title": "old",
        "source_session_id": None,
        "source_platform": "cli",
        "model": {},
        "transcript": [],
        "memory_pointers": [],
        "skill_states": {},
        "todo_state": None,
        "task_pointers": [],
        "handoff_state": "pending",
        "notes": None,
        "schema_checksum": "sha256:fake",
    }
    (tmp_path / f"{bid}.json").write_text(
        json.dumps(bundle_dict), encoding="utf-8"
    )
    store = HandoffStore(tmp_path)
    metas = store.list_bundles()
    assert metas[0].source_cwd is None
    assert metas[0].auto_saved is False


def test_list_recent_bundles_across_projects(tmp_path: Path):
    """跨项目列出最近 bundle。"""
    store = HandoffStore(tmp_path)
    store.save(
        transcript=[], source_session_id=None, model={},
        source_cwd="/proj/A", title="A1",
    )
    store.save(
        transcript=[], source_session_id=None, model={},
        source_cwd="/proj/B", title="B1",
    )
    # 不过滤
    all_bundles = list_recent_bundles_across_projects(store, limit=10)
    assert len(all_bundles) == 2
    # 按 cwd 过滤
    a_only = list_recent_bundles_across_projects(store, cwd_filter="/proj/A")
    assert len(a_only) == 1
    assert a_only[0].title == "A1"


def test_auto_save_current_session(tmp_path: Path):
    """自动保存当前会话为 bundle。"""
    store = HandoffStore(tmp_path)
    agent = MagicMock()
    agent.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    agent.session_id = "test_session"
    agent.model = {"main": "deepseek-chat"}

    bid = auto_save_current_session(
        store, agent, title="auto test",
        source_cwd="/proj/X",
    )
    metas = store.list_bundles()
    assert metas[0].auto_saved is True
    assert metas[0].source_cwd == "/proj/X"
    assert metas[0].title == "auto test"
