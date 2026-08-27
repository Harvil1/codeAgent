"""Checkpoint / Rewind 测试（文件快照 + 回滚）。"""

import json
from pathlib import Path

from agent.checkpoint import CheckpointManager


def _make_mgr(tmp_path, session="s1", max_snapshots=100):
    return CheckpointManager(tmp_path / ".checkpoints", session, max_snapshots=max_snapshots)


def test_track_file_dedup(tmp_path):
    mgr = _make_mgr(tmp_path)
    mgr.track_file("D:/x/a.py")
    mgr.track_file("D:/x/a.py")  # 重复
    mgr.track_file("D:/x/b.py")
    assert len(mgr.tracked_files()) == 2


def test_create_snapshot_copies_files(tmp_path):
    target = tmp_path / "proj" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("v1", encoding="utf-8")
    mgr = _make_mgr(tmp_path)
    mgr.track_file(str(target))

    sid = mgr.create_snapshot(conversation=[{"role": "user", "content": "hi"}])

    snap_dir = mgr._session_dir / sid
    assert snap_dir.exists()
    assert (snap_dir / "snapshot.json").exists()
    meta = json.loads((snap_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert len(meta["files"]) == 1
    assert meta["files"][0]["path"] == str(target)
    assert len(meta["conversation"]) == 1
    # 文件副本存在
    store = snap_dir / "files" / meta["files"][0]["store"]
    assert store.exists()
    assert store.read_text(encoding="utf-8") == "v1"


def test_restore_files_rolls_back(tmp_path):
    target = tmp_path / "proj" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("v1", encoding="utf-8")
    mgr = _make_mgr(tmp_path)
    mgr.track_file(str(target))
    sid = mgr.create_snapshot()

    # 修改文件后再快照，再改
    target.write_text("v2", encoding="utf-8")
    mgr.create_snapshot()
    target.write_text("v3", encoding="utf-8")

    restored = mgr.restore_files(sid)  # 回滚到 v1 快照
    assert str(target) in restored
    assert target.read_text(encoding="utf-8") == "v1"


def test_get_conversation(tmp_path):
    mgr = _make_mgr(tmp_path)
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    sid = mgr.create_snapshot(conversation=conv)
    assert mgr.get_conversation(sid) == conv


def test_prune_old_snapshots(tmp_path):
    mgr = _make_mgr(tmp_path, max_snapshots=2)
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    mgr.track_file(str(target))
    ids = [mgr.create_snapshot() for _ in range(4)]

    snaps = mgr.list_snapshots()
    assert len(snaps) == 2  # 只保留最近 2 个
    assert snaps[0]["id"] == ids[-1]  # 最新在前
    assert snaps[1]["id"] == ids[-2]


def test_list_snapshots_order(tmp_path):
    mgr = _make_mgr(tmp_path)
    mgr.create_snapshot()
    mgr.create_snapshot()
    snaps = mgr.list_snapshots()
    assert len(snaps) == 2
    # 时间倒序（id 是时间戳字符串，字典序即时间序）
    assert snaps[0]["id"] > snaps[1]["id"]
    assert "ts" in snaps[0]
    assert "msg_count" in snaps[0]
