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


# ==================================================================
# CCAR9 Task 3: MEMORY.md 索引分节 + cwd 切换感知
# ==================================================================


def test_rebuild_index_two_sections(tmp_path):
    """MEMORY.md 分'全局记忆'和'当前项目记忆'两节。

    全局节在前，项目节在后；项目节标题含项目键；条目内容在对应节。
    同时验证项目条目的链接路径指向项目区子目录（Task 2 遗留问题 1）。
    """
    from agent.project_scope import get_project_memory_key
    home = tmp_path / "home"
    proj = _proj_a(tmp_path)
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        ms.save(name="u1", description="全局条目", type="user")
        ms.save(name="p1", description="项目条目", type="project")
        ms.build_index_text()

    index = (home / "MEMORY.md").read_text(encoding="utf-8")
    assert "全局" in index and "项目" in index
    assert "全局条目" in index and "项目条目" in index
    # 项目条目在项目节（顺序：全局节在前）
    assert index.index("全局条目") < index.index("项目条目")
    # 分节标题存在
    assert "## 全局记忆" in index
    proj_key = get_project_memory_key(str(proj))
    assert f"## 当前项目记忆（{proj_key}）" in index
    # Task 2 遗留问题 1：项目条目链接路径应含 projects/<key>/
    # 找到项目条目行，验证其链接路径
    proj_line = next(
        (ln for ln in index.splitlines()
         if "项目条目" in ln and ln.strip().startswith("-")),
        None,
    )
    assert proj_line is not None, "项目条目行未找到"
    assert f"projects/{proj_key}/" in proj_line, (
        f"项目条目链接路径应含 projects/{proj_key}/，实际: {proj_line}"
    )


def test_snapshot_switches_with_cwd(tmp_path):
    """切换项目目录后，快照只含新项目的项目区（当前项目约束）。

    同实例：项目 A 存 pa → 切到 B → 项目 B 存 pb → 快照应只含 pb
    不含 pa（当前项目键变了触发 rebuild，_index_built_key 感知）。
    """
    home = tmp_path / "home"
    a, b = _proj_a(tmp_path), _proj_b(tmp_path)
    with workspace_cwd_context(str(a)):
        ms = MemoryStore(omnimate_home=home)
        ms.save(name="pa", description="A 的事实", type="project")
    with workspace_cwd_context(str(b)):
        ms_b = MemoryStore(omnimate_home=home)
        ms_b.save(name="pb", description="B 的事实", type="project")
        snap = ms_b.snapshot_for_prompt()
    assert "A 的事实" not in snap
    assert "B 的事实" in snap


def test_list_all_merges_zones(tmp_path):
    """list_all 跨区合并：user(全局) + project(项目区) 都返回。

    Task 2 可能已覆盖（save_routes_by_type 间接测了）——验证即可。
    """
    home = tmp_path / "home"
    proj = _proj_a(tmp_path)
    with workspace_cwd_context(str(proj)):
        ms = MemoryStore(omnimate_home=home)
        ms.save(name="u1", description="d", type="user")
        ms.save(name="p1", description="d", type="project")
        entries = ms.list_all()
    names = {e.name for e in entries}
    assert {"u1", "p1"} <= names


def test_snapshot_same_instance_follows_cwd(tmp_path):
    """【核心】同实例切项目后，快照跟随新项目（_index_built_key 感知）。

    Task 3 关键约束：除了 dirty flag，_ensure_index_fresh 还要比较当前
    项目键——存 _index_built_key（rebuild 时记录），键变了也触发 rebuild。
    场景：同实例 A 下 build → 切 B（无新写入，只是切 cwd）→ snapshot
    应反映 B 的项目区（空，因为 B 没写过），不再含 A 的项目条目。
    """
    home = tmp_path / "home"
    a, b = _proj_a(tmp_path), _proj_b(tmp_path)
    with workspace_cwd_context(str(a)):
        ms = MemoryStore(omnimate_home=home)
        ms.save(name="pa", description="A 的事实", type="project")
        ms.build_index_text()  # 此时 _index_built_key = A 的键
        snap_a = ms.snapshot_for_prompt()
        assert "A 的事实" in snap_a
    # 不新建实例——直接切 cwd（模拟 CLI 在同会话里 cd 到另一个项目）
    with workspace_cwd_context(str(b)):
        # 没有新 save，_index_dirty 应为 False；但项目键变了，应触发 rebuild
        snap_b = ms.snapshot_for_prompt()
    assert "A 的事实" not in snap_b, "同实例切项目后旧项目条目应消失"
