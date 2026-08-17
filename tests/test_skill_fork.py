import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def test_scan_skill_commands_reads_context_field():
    """scan_skill_commands 解析 frontmatter context 字段。"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "fork-skill").mkdir()
        (Path(d) / "fork-skill" / "SKILL.md").write_text(
            "---\nname: fork-skill\ndescription: x\ncontext: fork\n---\nbody\n",
            encoding="utf-8",
        )
        (Path(d) / "normal-skill").mkdir()
        (Path(d) / "normal-skill" / "SKILL.md").write_text(
            "---\nname: normal-skill\ndescription: y\n---\nbody\n", encoding="utf-8",
        )
        from agent.skill_commands import scan_skill_commands
        cmds = scan_skill_commands(d)
    assert cmds["/fork-skill"]["context"] == "fork"
    assert cmds["/normal-skill"]["context"] is None


def test_run_skill_in_fork_returns_child_result():
    """run_skill_in_fork 用子代理跑技能，返回结果。"""
    from agent.skill_fork import run_skill_in_fork
    fake_parent = MagicMock()
    fake_parent.base_url = "http://x"
    fake_parent.api_key = "k"
    fake_parent.auth_token = None
    fake_parent.model = "m"
    fake_parent.model_format = "openai"
    fake_parent.omnimate_home = Path(tempfile.mkdtemp())
    fake_parent.spawn_depth = 0
    fake_child = MagicMock()
    fake_child.chat = AsyncMock(return_value="子代理结果")
    with patch("agent.skill_fork.AIAgent", return_value=fake_child) as mock_cls:
        out = run_skill_in_fork("skill-x", "正文", "用户问题", fake_parent)
    assert out == "子代理结果"
    mock_cls.assert_called_once()
    fake_child.chat.assert_called_once_with("用户问题")


def test_run_skill_in_fork_fail_open():
    """子代理构造失败时返回错误消息（不抛）。"""
    from agent.skill_fork import run_skill_in_fork
    fake_parent = MagicMock(spec=[])  # 缺属性 → 构造抛
    out = run_skill_in_fork("x", "b", "q", fake_parent)
    assert "失败" in out
