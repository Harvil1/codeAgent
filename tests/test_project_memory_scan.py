"""阶段 2 测试：OMNIMATE.md 递归向上扫描。

覆盖：
- `_scan_project_memory_files`：基本递归（多层有 OMNIMATE.md）
- `.git` 停止语义（扫到 .git 那层就停，不再向上）
- 无 OMNIMATE.md 时返回空
- 返回顺序：从根到 cwd（外层先注入）
- build_system_prompt_layers 集成：项目记忆被注入 context 层
"""
from pathlib import Path
from unittest.mock import patch


def test_scan_basic_recursion(tmp_path):
    """三层目录都有 OMNIMATE.md → 全部扫到，顺序从根到 cwd。"""
    from agent.prompt_builder import _scan_project_memory_files
    # /tmp/a/OMNIMATE.md  (root)
    # /tmp/a/b/OMNIMATE.md
    # /tmp/a/b/c/        (cwd)
    root = tmp_path / "a"
    mid = root / "b"
    cwd = mid / "c"
    cwd.mkdir(parents=True)
    (root / "OMNIMATE.md").write_text("# root", encoding="utf-8")
    (mid / "OMNIMATE.md").write_text("# mid", encoding="utf-8")

    result = _scan_project_memory_files(cwd)
    assert result == [(root / "OMNIMATE.md"), (mid / "OMNIMATE.md")]


def test_scan_stops_at_git(tmp_path):
    """.git 之上不再扫：项目根 .git 之上的 OMNIMATE.md 不被收集。"""
    from agent.prompt_builder import _scan_project_memory_files
    # /tmp/outer/OMNIMATE.md  (不应该扫到)
    # /tmp/outer/proj/.git/
    # /tmp/outer/proj/OMNIMATE.md
    # /tmp/outer/proj/sub/   (cwd)
    outer = tmp_path / "outer"
    proj = outer / "proj"
    sub = proj / "sub"
    sub.mkdir(parents=True)
    (outer / "OMNIMATE.md").write_text("# outer", encoding="utf-8")
    (proj / "OMNIMATE.md").write_text("# proj", encoding="utf-8")
    (proj / ".git").mkdir()

    result = _scan_project_memory_files(sub)
    # 只扫到 proj 那一层（含 .git），outer 之上不被扫
    assert result == [proj / "OMNIMATE.md"]
    assert outer / "OMNIMATE.md" not in result


def test_scan_no_omnimate_md(tmp_path):
    """目录树里没 OMNIMATE.md → 返回空列表。"""
    from agent.prompt_builder import _scan_project_memory_files
    cwd = tmp_path / "x" / "y" / "z"
    cwd.mkdir(parents=True)
    result = _scan_project_memory_files(cwd)
    assert result == []


def test_scan_no_git_walks_to_root(tmp_path):
    """没有 .git，扫到磁盘根（实际 tmp_path 上层不应有 OMNIMATE.md）。"""
    from agent.prompt_builder import _scan_project_memory_files
    cwd = tmp_path / "deep" / "nested"
    cwd.mkdir(parents=True)
    result = _scan_project_memory_files(cwd)
    # tmp_path 之上一般无 OMNIMATE.md，但函数必须能正常返回不抛
    assert isinstance(result, list)


def test_scan_includes_cwd_level(tmp_path):
    """cwd 本身有 OMNIMATE.md → 被扫到。"""
    from agent.prompt_builder import _scan_project_memory_files
    cwd = tmp_path / "proj"
    cwd.mkdir()
    (cwd / "OMNIMATE.md").write_text("# cwd level", encoding="utf-8")
    result = _scan_project_memory_files(cwd)
    assert (cwd / "OMNIMATE.md") in result


def test_build_system_prompt_injects_project_memory(tmp_path, monkeypatch):
    """集成：build_system_prompt_layers 调用时注入 cwd 上的 OMNIMATE.md。"""
    from agent.prompt_builder import build_system_prompt_layers
    # 模拟 cwd = tmp_path，里面放一个 OMNIMATE.md
    (tmp_path / "OMNIMATE.md").write_text(
        "# Test Project\n本项目的约定", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    layers = build_system_prompt_layers()
    flat = layers.render_flat()
    assert "Test Project" in flat
    assert "本项目的约定" in flat
