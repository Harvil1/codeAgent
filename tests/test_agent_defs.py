import tempfile
from pathlib import Path
from agent.agent_defs import scan_agent_defs, AgentDefinition


def test_scan_finds_user_and_project_defs(monkeypatch, tmp_path):
    """扫描 ~/.OmniMate/agents + <cwd>/.omnimate/agents，项目级覆盖用户级。"""
    user_dir = tmp_path / "user_agents"
    proj_dir = tmp_path / "proj_agents"
    user_dir.mkdir()
    proj_dir.mkdir()
    (user_dir / "explorer.md").write_text(
        "---\nname: explorer\ndescription: 探索代码\nmodel: deepseek-chat\n"
        "tools:\n  - core\nmaxTurns: 30\n---\n你是探索者。\n",
        encoding="utf-8",
    )
    (proj_dir / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: 审查\n---\n审代码。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: user_dir)
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: proj_dir)
    defs = scan_agent_defs()
    assert "explorer" in defs
    assert "reviewer" in defs
    assert defs["explorer"].model == "deepseek-chat"
    assert defs["explorer"].max_turns == 30
    assert "你是探索者" in defs["explorer"].system_prompt


def test_project_overrides_user(monkeypatch, tmp_path):
    """同名时项目级覆盖用户级。"""
    user_dir = tmp_path / "u"; user_dir.mkdir()
    proj_dir = tmp_path / "p"; proj_dir.mkdir()
    (user_dir / "x.md").write_text("---\nname: x\ndescription: 用户版\n---\nA\n", encoding="utf-8")
    (proj_dir / "x.md").write_text("---\nname: x\ndescription: 项目版\n---\nB\n", encoding="utf-8")
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: user_dir)
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: proj_dir)
    defs = scan_agent_defs()
    assert defs["x"].description == "项目版"


def test_unknown_definition_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: tmp_path)
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: tmp_path)
    defs = scan_agent_defs()
    assert "nope" not in defs


def test_builtin_agents_loaded():
    """scan_agent_defs 默认返回内置 explore/plan。"""
    from agent.agent_defs import scan_agent_defs
    defs = scan_agent_defs()
    assert "explore" in defs, "内置 explore 子代理未加载"
    assert "plan" in defs, "内置 plan 子代理未加载"
    assert defs["explore"].tools == ["explore"]
    assert defs["plan"].tools == ["plan"]


def test_builtin_overridable_by_user(monkeypatch, tmp_path):
    """用户目录同名定义覆盖内置。"""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    (user_dir / "explore.md").write_text(
        "---\nname: explore\ndescription: 我的探索\n---\n自定义版\n",
        encoding="utf-8",
    )
    import agent.agent_defs as ad
    monkeypatch.setattr(ad, "_user_agents_dir", lambda: user_dir)
    monkeypatch.setattr(ad, "_project_agents_dir", lambda: tmp_path / "noexist")
    defs = ad.scan_agent_defs()
    assert defs["explore"].description == "我的探索"


def test_explore_toolset_resolves():
    """explore 工具集解析为只读工具。"""
    from toolsets import resolve_toolset
    tools = resolve_toolset("explore")
    assert "read_file" in tools
    assert "search_files" in tools
    assert "write_file" not in tools  # 无修改类
    assert "terminal" not in tools


def test_parse_memory_skills_mcp_servers(tmp_path):
    """frontmatter 的 memory/skills/mcpServers 字段正确解析。"""
    from agent.agent_defs import _parse_one
    md = tmp_path / "rich.md"
    md.write_text(
        "---\n"
        "name: rich\n"
        "description: 富配置子代理\n"
        "memory: true\n"
        "skills:\n"
        "  - skill-a\n"
        "  - skill-b\n"
        "mcpServers:\n"
        "  - github\n"
        "  - filesystem\n"
        "---\n正文\n",
        encoding="utf-8",
    )
    ad = _parse_one(md)
    assert ad is not None
    assert ad.memory is True
    assert ad.skills == ["skill-a", "skill-b"]
    assert ad.mcp_servers == ["github", "filesystem"]


def test_parse_defaults_when_fields_absent(tmp_path):
    """无 memory/skills/mcpServers 字段时，默认值正确。"""
    from agent.agent_defs import _parse_one
    md = tmp_path / "simple.md"
    md.write_text("---\nname: simple\ndescription: 简单\n---\n正文\n", encoding="utf-8")
    ad = _parse_one(md)
    assert ad.memory is False
    assert ad.skills == []
    assert ad.mcp_servers == []
