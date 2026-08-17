"""阶段 3 测试：@import 语法。

覆盖：
- `@docs/arch.md` 正确展开
- `@~/xxx` 展开 home 路径
- code span 里的 `@anthropic-ai/sdk` 不展开
- code block 里的 `@path` 不展开
- 递归 max_depth=5（超过只 warn 不抛）
- 防环：A 引用 B，B 引用 A，只展开一次
- 文件不存在的 @path 原样保留
"""
from pathlib import Path


def test_expand_basic_import(tmp_path):
    """`@docs/arch.md` 相对路径正确展开。"""
    from agent.prompt_builder import _expand_imports
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "arch.md").write_text("# 架构\n三层", encoding="utf-8")
    content = "项目说明\n详见 @docs/arch.md\n"
    result = _expand_imports(content, tmp_path)
    assert "# 架构" in result
    assert "三层" in result


def test_expand_home_path(tmp_path):
    """`@~/file` 展开 home 目录。"""
    from agent.prompt_builder import _expand_imports
    import os
    home = Path.home()
    target = home / "_omnimate_test_import.md"
    try:
        target.write_text("home content", encoding="utf-8")
        content = "see @~/_omnimate_test_import.md"
        result = _expand_imports(content, tmp_path)
        assert "home content" in result
    finally:
        if target.exists():
            target.unlink()


def test_code_span_not_expanded(tmp_path):
    """code span 里的 `@anthropic-ai/sdk` 不展开。"""
    from agent.prompt_builder import _expand_imports
    (tmp_path / "anthropic-ai").mkdir()
    (tmp_path / "anthropic-ai" / "sdk").write_text("should-not-inject", encoding="utf-8")
    content = "install `@anthropic-ai/sdk` now"
    result = _expand_imports(content, tmp_path)
    # code span 里的应该原样保留
    assert "`@anthropic-ai/sdk`" in result
    assert "should-not-inject" not in result


def test_code_block_not_expanded(tmp_path):
    """fenced code block 里的 @path 不展开。"""
    from agent.prompt_builder import _expand_imports
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "bar.md").write_text("should-not-inject", encoding="utf-8")
    content = "前文\n\n```\nsee @foo/bar.md\n```\n\n后文"
    result = _expand_imports(content, tmp_path)
    assert "should-not-inject" not in result
    assert "@foo/bar.md" in result  # 原样保留


def test_nonexistent_path_preserved(tmp_path):
    """文件不存在的 @path 原样保留不抛。"""
    from agent.prompt_builder import _expand_imports
    content = "see @nonexistent/path.md"
    result = _expand_imports(content, tmp_path)
    assert "@nonexistent/path.md" in result


def test_no_slash_not_treated_as_import(tmp_path):
    """`@foo` 没 / 不算 import。"""
    from agent.prompt_builder import _expand_imports
    content = "email me @foo bar"
    result = _expand_imports(content, tmp_path)
    assert "@foo" in result  # 原样


def test_recursion_depth_limit(tmp_path):
    """递归深度超 5 → 只 warn 不抛。"""
    from agent.prompt_builder import _expand_imports
    # 构造 7 层嵌套：a 引用 b，b 引用 c...
    files = []
    for i in range(7):
        p = tmp_path / f"lvl{i}.md"
        if i < 6:
            p.write_text(f"level {i}\n@lvl{i + 1}.md", encoding="utf-8")
        else:
            p.write_text(f"level {i} (deepest)", encoding="utf-8")
        files.append(p)
    result = _expand_imports("@lvl0.md", tmp_path)
    # 应该至少展开若干层但不抛异常
    assert "level 0" in result


def test_cycle_protection(tmp_path):
    """A 引用 B，B 引用 A：防环，只展开一次。"""
    from agent.prompt_builder import _expand_imports
    (tmp_path / "a.md").write_text("A content\n@b.md", encoding="utf-8")
    (tmp_path / "b.md").write_text("B content\n@a.md", encoding="utf-8")
    result = _expand_imports("@a.md", tmp_path)
    # 都展开过一次
    assert "A content" in result
    assert "B content" in result
    # 第二次引用被替换为注释（防环）
    assert "@import 已展开过" in result
