"""CCAR9 Task 2: MemoryStore 按 type 路由写入（全局区 vs 项目区）。

分层隔离设计：
- user/feedback 类全局共享（跨项目可见）
- project/reference 类按项目分区（项目 A 的事实在项目 B 不可见）

测试用 workspace_cwd_context 控制「当前项目目录」（tmp_path 非 git →
项目键 = sanitize(该路径)），验证分区隔离。
"""
import pytest
from pathlib import Path

from agent.memory_store import MemoryStore
from agent.workspace_context import workspace_cwd_context


def _proj_a(tmp_path):
    d = tmp_path / "projA"
    d.mkdir(exist_ok=True)
    return d


def _proj_b(tmp_path):
    d = tmp_path / "projB"
    d.mkdir(exist_ok=True)
    return d


def test_save_routes_by_type(tmp_path):
    """user/feedback → 全局区；project/reference → 项目区。"""
    home = tmp_path / "home"
    proj = _proj_a(tmp_path)
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        ms.save(name="u1", description="用户偏好", type="user")
        ms.save(name="f1", description="反馈", type="feedback")
        ms.save(name="p1", description="项目事实", type="project")
        ms.save(name="r1", description="外部引用", type="reference")

    # 全局区有 user/feedback 的 topic 文件
    global_files = list((home / ".memory").glob("*.jsonl"))
    global_text = "".join(f.read_text(encoding="utf-8") for f in global_files)
    assert "u1" in global_text and "f1" in global_text
    assert "p1" not in global_text and "r1" not in global_text

    # 项目区有 project/reference（tmp_path 非 git → 键 = sanitize(proj)）
    from agent.project_scope import get_project_memory_key
    proj_dir = home / ".memory" / "projects" / get_project_memory_key(str(proj))
    proj_text = "".join(
        f.read_text(encoding="utf-8") for f in proj_dir.glob("*.jsonl")
    )
    assert "p1" in proj_text and "r1" in proj_text


def test_project_memory_isolated_across_projects(tmp_path):
    """【核心】项目 A 存的 project 记忆在项目 B 的 MemoryStore 里不可见。"""
    home = tmp_path / "home"
    a, b = _proj_a(tmp_path), _proj_b(tmp_path)

    with workspace_cwd_context(str(a)):
        ms_a = MemoryStore(omnimate_home=home)
        ms_a.save(name="tanke-fact", description="A 项目的事实",
                  type="project")

    with workspace_cwd_context(str(b)):
        ms_b = MemoryStore(omnimate_home=home)
        # user 类全局可见（分层隔离）
        ms_b.save(name="shared-pref", description="跨项目偏好", type="user")
        snap_b = ms_b.snapshot_for_prompt()

    assert "A 项目的事实" not in snap_b, "项目 A 的记忆泄漏到项目 B"
    assert "跨项目偏好" in snap_b, "user 类应全局共享"


def test_get_finds_entries_in_both_zones(tmp_path):
    """get() 按 memory_id 跨区查找（id 格式不变）。"""
    home = tmp_path / "home"
    proj = _proj_a(tmp_path)
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        uid_global = ms.save(name="u1", description="d", type="user")
        uid_project = ms.save(name="p1", description="d", type="project")

        assert ms.get(uid_global) is not None
        assert ms.get(uid_project) is not None
        assert ms.get(uid_project).name == "p1"


def test_update_delete_work_across_zones(tmp_path):
    """update/delete 按条目实际所在区操作。"""
    home = tmp_path / "home"
    proj = _proj_a(tmp_path)
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        pid = ms.save(name="p1", description="old", type="project")
        uid = ms.save(name="u1", description="old", type="user")

        ms.update(pid, description="new-proj")
        ms.update(uid, description="new-user")
        assert ms.get(pid).description == "new-proj"
        assert ms.get(uid).description == "new-user"

        assert ms.delete(pid) is True
        assert ms.get(pid) is None
        assert ms.get(uid) is not None  # 另一个区不受影响
