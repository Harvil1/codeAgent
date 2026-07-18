"""多 Agent 安全加固测试（06）。

验证：
1. 幻觉检测：声称的 task_id / 文件路径不存在时追加警告
2. 真实存在的 ID 不触发警告
3. URL / 版本号不被误判为文件路径
4. mark_blocked 累积 block_history
5. 同 kind 阻塞 3 次升级到 triage
6. triage 状态不在 find_ready
7. task_update 工具 status=blocked 走 mark_blocked
"""
import json
from pathlib import Path

import pytest

from agent.team.hallucination_check import (
    extract_claimed_ids,
    verify_claims,
    append_warning,
)
from agent.task_store import TaskStore, TRIAGE_THRESHOLD


# ----------------------------------------------------------------------------
# extract_claimed_ids
# ----------------------------------------------------------------------------

def test_extract_task_ids():
    text = "已创建 task_001 和 task_002，更新了 task-abc"
    tasks, _ = extract_claimed_ids(text)
    assert "task_001" in tasks
    assert "task_002" in tasks
    assert "task-abc" in tasks


def test_extract_file_paths_filters_urls():
    text = "see https://example.com/file.txt and local output.log"
    _, files = extract_claimed_ids(text)
    assert "output.log" in files
    # URL 不应被当本地文件
    assert not any("example.com" in f for f in files)


def test_extract_filters_version_numbers():
    text = "升级到 v1.2.3 或 v2.0"
    _, files = extract_claimed_ids(text)
    # 版本号不应被当文件路径
    assert not any("v1.2.3" in f for f in files)


def test_extract_dedup_preserves_order():
    text = "task_001 task_002 task_001"
    tasks, _ = extract_claimed_ids(text)
    assert tasks == ["task_001", "task_002"]


# ----------------------------------------------------------------------------
# verify_claims
# ----------------------------------------------------------------------------

def test_verify_claims_missing_task(tmp_path: Path):
    """声称的 task 不存在 → missing_tasks。"""
    store = TaskStore(harvil_home=tmp_path)
    store.create(subject="real", description="")
    # store 里有真实 task，但 text 声称另一个
    real_tasks = store.list_all()
    real_id = real_tasks[0]["id"]
    text = f"已创建 {real_id} 和 task_nonexistent_xyz"

    result = verify_claims(text, task_store=store)
    assert "task_nonexistent_xyz" in result["missing_tasks"]
    assert real_id not in result["missing_tasks"]
    assert result["hallucination_detected"] is True


def test_verify_claims_missing_file(tmp_path: Path):
    """声称的文件不存在 → missing_files。"""
    text = "已写入 result.txt（不存在的文件）"
    result = verify_claims(text, fs_cwd=tmp_path)
    assert "result.txt" in result["missing_files"]
    assert result["hallucination_detected"] is True


def test_verify_claims_real_file_no_warning(tmp_path: Path):
    """真实存在的文件不触发警告。"""
    (tmp_path / "exists.txt").write_text("ok", encoding="utf-8")
    text = "已写入 exists.txt"
    result = verify_claims(text, fs_cwd=tmp_path)
    assert result["hallucination_detected"] is False


def test_verify_claims_no_store_skips_task_check():
    """task_store=None 时跳过 task 校验。"""
    text = "已创建 task_xyz123"
    result = verify_claims(text, task_store=None)
    assert result["missing_tasks"] == []
    assert result["hallucination_detected"] is False


# ----------------------------------------------------------------------------
# append_warning
# ----------------------------------------------------------------------------

def test_append_warning_when_hallucination():
    text = "完成"
    verification = {
        "hallucination_detected": True,
        "missing_tasks": ["task_999"],
        "missing_files": [],
    }
    new_text = append_warning(text, verification)
    assert "完成" in new_text
    assert "task_999" in new_text
    assert "⚠️" in new_text


def test_append_warning_no_hallucination_unchanged():
    text = "完成"
    verification = {
        "hallucination_detected": False,
        "missing_tasks": [],
        "missing_files": [],
    }
    assert append_warning(text, verification) == "完成"


# ----------------------------------------------------------------------------
# TaskStore.mark_blocked
# ----------------------------------------------------------------------------

def test_mark_blocked_records_history(tmp_path: Path):
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    result = store.mark_blocked(task["id"], kind="dependency", reason="等 task_X")
    assert result["status"] == "blocked"
    assert result["block_count"] == 1

    updated = store.get(task["id"])
    assert updated["status"] == "blocked"
    assert updated["block_kind"] == "dependency"
    assert updated["block_reason"] == "等 task_X"
    assert len(updated["block_history"]) == 1
    assert updated["block_count_by_kind"]["dependency"] == 1


def test_mark_blocked_upgrades_to_triage(tmp_path: Path):
    """同 kind 阻塞 >=3 次升级 triage。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    tid = task["id"]

    r1 = store.mark_blocked(tid, kind="dependency", reason="等1")
    assert r1["status"] == "blocked"
    r2 = store.mark_blocked(tid, kind="dependency", reason="等2")
    assert r2["status"] == "blocked"
    r3 = store.mark_blocked(tid, kind="dependency", reason="等3")
    assert r3["status"] == "triage"
    assert r3["block_count"] == 3


def test_mark_blocked_different_kinds_dont_upgrade(tmp_path: Path):
    """不同 kind 不互相累积。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    tid = task["id"]

    store.mark_blocked(tid, kind="dependency", reason="d1")
    store.mark_blocked(tid, kind="dependency", reason="d2")
    # capability 2 次不应升级
    store.mark_blocked(tid, kind="capability", reason="c1")
    store.mark_blocked(tid, kind="capability", reason="c2")
    # 此时 dependency 还只 2 次（不到 3），不升级
    updated = store.get(tid)
    # mark_blocked 会重置 status 每次都 "blocked"（除非升级）
    assert updated["status"] == "blocked"


def test_mark_blocked_unknown_kind_falls_back_to_transient(tmp_path: Path):
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    result = store.mark_blocked(task["id"], kind="bogus_kind", reason="")
    assert result["kind"] == "transient"


def test_mark_blocked_missing_task_raises(tmp_path: Path):
    store = TaskStore(harvil_home=tmp_path)
    with pytest.raises(KeyError):
        store.mark_blocked("task_nonexistent", kind="transient")


def test_triage_not_in_find_ready(tmp_path: Path):
    """triage 状态不出现在 find_ready。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    tid = task["id"]
    # 升级到 triage
    for _ in range(TRIAGE_THRESHOLD):
        store.mark_blocked(tid, kind="dependency", reason="x")
    # 此时 status=triage，find_ready 只看 pending，自然排除
    ready = store.find_ready()
    assert all(t["id"] != tid for t in ready)


def test_block_history_capped_at_20(tmp_path: Path):
    """block_history 超过 20 条滚动覆盖。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="A", description="")
    tid = task["id"]
    # 阻塞 25 次（每次都会进入 history）
    for i in range(25):
        store.mark_blocked(tid, kind="transient", reason=f"r{i}")
    updated = store.get(tid)
    assert len(updated["block_history"]) <= 20


# ----------------------------------------------------------------------------
# task_update 工具 status=blocked 走 mark_blocked
# ----------------------------------------------------------------------------

def test_task_update_blocked_route(tmp_path: Path):
    """task_update(status=blocked, block_kind=...) 走 mark_blocked 路径。"""
    from tools.task_tools import _handle_task_update
    from agent.task_store import get_task_store

    # 全局单例（task_tools 内部用 get_task_store()）
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="A", description="")

    result_json = _handle_task_update({
        "id": task["id"],
        "status": "blocked",
        "block_kind": "dependency",
        "block_reason": "等 task_X",
    })
    result = json.loads(result_json)
    assert result["success"] is True
    assert result["action"] == "block"
    assert result["new_status"] == "blocked"
    assert result["block_count"] == 1
    assert result["upgraded_to_triage"] is False


def test_task_update_blocked_triage_after_three(tmp_path: Path):
    """3 次 blocked 后 task_update 返回 upgraded_to_triage=True。"""
    from tools.task_tools import _handle_task_update
    from agent.task_store import get_task_store

    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="A", description="")

    last = None
    for _ in range(TRIAGE_THRESHOLD):
        last = json.loads(_handle_task_update({
            "id": task["id"],
            "status": "blocked",
            "block_kind": "dependency",
            "block_reason": "等",
        }))
    assert last["upgraded_to_triage"] is True
    assert last["new_status"] == "triage"


def test_task_update_normal_status_unchanged(tmp_path: Path):
    """非 blocked 的 status 仍走原有 update 路径。"""
    from tools.task_tools import _handle_task_update
    from agent.task_store import get_task_store

    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="A", description="")

    result_json = _handle_task_update({
        "id": task["id"],
        "status": "in_progress",
        "owner": "tester",
    })
    result = json.loads(result_json)
    assert result["success"] is True
    assert "task" in result
    assert result["task"]["status"] == "in_progress"
    assert result["task"]["owner"] == "tester"
