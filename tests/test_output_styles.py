# tests/test_output_styles.py
"""C6（CCB 借鉴 outputStyles）：输出风格目录发现 + prompt 注入 + /output-style。

- 风格 = 一个 .md 文件：文件名即风格名，frontmatter（name/description 可选），
  正文即提示词。
- 目录：`<cwd 向上到 git root>/.omnimate/output-styles/`（项目级）+
  `~/.OmniMate/output-styles/`（用户级）；同名项目级覆盖用户级。
- 注入：system prompt 的 context 层（会话内不变，保 cache）。
- 切换：/output-style <name> 写 settings.json 顶层 output_style + 失效 prompt。
"""

from pathlib import Path

import pytest


def _mk_style(directory: Path, filename: str, body: str, fm: str = ""):
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath(filename).write_text(
        f"---\n{fm}---\n{body}" if fm else body, encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. 目录发现
# ---------------------------------------------------------------------------

def test_discover_user_and_project_styles(tmp_path, monkeypatch):
    """用户级 + 项目级并存；同名项目级覆盖用户级。"""
    from agent.output_styles import discover_output_styles

    monkeypatch.chdir(tmp_path)
    user = tmp_path / "home"
    _mk_style(user / "output-styles", "concise.md", "回答不超过三句话。")
    _mk_style(user / "output-styles", "verbose.md", "输出详细步骤。")
    proj = tmp_path / ".omnimate" / "output-styles"
    _mk_style(proj, "verbose.md", "项目版：输出极详细步骤与引用。", fm="name: verbose\ndescription: 项目的详细风格\n")

    styles = discover_output_styles(cwd=str(tmp_path), agent_home=user)
    assert set(styles) == {"concise", "verbose"}
    assert "项目版" in styles["verbose"].body  # 项目级覆盖
    assert styles["verbose"].description == "项目的详细风格"
    assert "三句话" in styles["concise"].body


def test_discover_walks_up_to_git_root(tmp_path, monkeypatch):
    """子目录启动也能发现 git root 的项目级风格（对齐 OMNIMATE.md 语义）。"""
    from agent.output_styles import discover_output_styles

    (tmp_path / ".git").mkdir()
    _mk_style(tmp_path / ".omnimate" / "output-styles", "terse.md", "只回结论。")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    styles = discover_output_styles(cwd=str(sub), agent_home=tmp_path / "home")
    assert "terse" in styles


def test_discover_empty_is_empty(tmp_path, monkeypatch):
    """无任何风格目录 → 空表（不抛）。"""
    from agent.output_styles import discover_output_styles

    monkeypatch.chdir(tmp_path)
    assert discover_output_styles(cwd=str(tmp_path), agent_home=tmp_path / "h") == {}


# ---------------------------------------------------------------------------
# 2. resolve + prompt 注入
# ---------------------------------------------------------------------------

def test_resolve_unknown_style_fails_open(tmp_path, monkeypatch):
    """config 指向不存在的风格 → None（fail-open，不注入任何东西）。"""
    from agent.output_styles import resolve_output_style

    monkeypatch.chdir(tmp_path)
    assert resolve_output_style(
        {"output_style": "nope"}, cwd=str(tmp_path), agent_home=tmp_path / "h",
    ) is None
    assert resolve_output_style(
        {}, cwd=str(tmp_path), agent_home=tmp_path / "h",
    ) is None


def test_build_system_prompt_injects_style(tmp_path, monkeypatch):
    """build_system_prompt(output_style=...) → context 层含风格正文。"""
    from agent.prompt_builder import build_system_prompt
    from agent.output_styles import discover_output_styles, render_style_section

    monkeypatch.chdir(tmp_path)
    user = tmp_path / "home"
    _mk_style(user / "output-styles", "concise.md", "回答不超过三句话。")
    style = discover_output_styles(cwd=str(tmp_path), agent_home=user)["concise"]

    prompt = build_system_prompt(
        enabled_toolsets=[],
        output_style_text=render_style_section(style),
    )
    assert "回答不超过三句话" in prompt
    assert "输出风格" in prompt
    # 未设置时不含该节
    prompt_off = build_system_prompt(enabled_toolsets=[])
    assert "输出风格" not in prompt_off


# ---------------------------------------------------------------------------
# 3. /output-style 命令
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self):
        self.config = {}
        self._invalidated = 0

    def invalidate_system_prompt(self):
        self._invalidated += 1


class _FakeRT:
    def __init__(self, home):
        self.home = home
        self.config = {}
        self.agent = _FakeAgent()


def test_output_style_command_list_and_set(tmp_path, monkeypatch, capsys):
    """/output-style 列表（标当前）；/output-style <name> 写 settings + 失效 prompt。"""
    import cli as cli_mod
    from agent.settings import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    user = tmp_path / "output-styles"
    _mk_style(user, "concise.md", "回答不超过三句话。")

    rt = _FakeRT(tmp_path)
    # 列表
    assert cli_mod._handle_command("/output-style", rt) is True
    out = capsys.readouterr().out
    assert "concise" in out

    # 切换
    assert cli_mod._handle_command("/output-style concise", rt) is True
    assert load_settings().get("output_style") == "concise"
    assert rt.agent._invalidated == 1
    assert rt.config.get("output_style") == "concise"  # 运行时同步

    # 未知风格：报错且不改设置
    capsys.readouterr()
    assert cli_mod._handle_command("/output-style nope", rt) is True
    assert "没有找到" in capsys.readouterr().out
    assert load_settings().get("output_style") == "concise"

    # off：清除
    assert cli_mod._handle_command("/output-style off", rt) is True
    assert load_settings().get("output_style") is None
