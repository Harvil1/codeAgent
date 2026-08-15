"""T3（核心机制对齐第 3 项）：技能 frontmatter files: 附件（触发时一并注入）。

- frontmatter files: 声明参考文件，slash 注入 + load_skill 工具两处附带
- 路径相对技能目录；单文件截 8K（config skills.file_attachment_max_chars）；总数 5
- 缺失 fail-open（跳过+提示）；skill_view 不注入附件
"""
import json

import pytest


def _make_skill(tmp_path, name="my-skill", files=None, body="技能正文"):
    d = tmp_path / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    fm_lines = [
        "---",
        f"name: {name}",
        "description: 测试技能",
    ]
    if files is not None:
        fm_lines.append("files:")
        for f in files:
            fm_lines.append(f"  - {f}")
    fm_lines.append("---")
    (d / "SKILL.md").write_text("\n".join(fm_lines) + f"\n\n{body}\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# execute_skill（slash 注入路径）
# ---------------------------------------------------------------------------

def test_execute_skill_includes_attachments(tmp_path):
    """带 files 的技能，slash 注入消息含附件内容。"""
    from agent.skill_commands import execute_skill

    d = _make_skill(tmp_path, files=["ref.md", "notes.txt"])
    (d / "ref.md").write_text("# 参考文档\n细节 A", encoding="utf-8")
    (d / "notes.txt").write_text("备注内容 B", encoding="utf-8")

    msg = execute_skill(str(d / "SKILL.md"), "用户输入")
    assert "技能正文" in msg
    assert "细节 A" in msg, "ref.md 内容应作为附件注入"
    assert "备注内容 B" in msg
    assert "用户消息: 用户输入" in msg  # 原格式保留


def test_execute_skill_missing_file_fail_open(tmp_path):
    """附件文件缺失 → 跳过+提示，不炸。"""
    from agent.skill_commands import execute_skill

    d = _make_skill(tmp_path, files=["gone.md"])
    msg = execute_skill(str(d / "SKILL.md"), "hi")
    assert "技能正文" in msg
    assert "缺失" in msg or "跳过" in msg


def test_execute_skill_no_files_unchanged(tmp_path):
    """无 files → 输出与现状等价（回归）。"""
    from agent.skill_commands import execute_skill

    d = _make_skill(tmp_path, files=None)
    msg = execute_skill(str(d / "SKILL.md"), "hi")
    assert "技能正文" in msg
    assert "附件" not in msg


def test_execute_skill_truncates_large_attachment(tmp_path):
    """单附件超 8K → 截断。"""
    from agent.skill_commands import execute_skill

    d = _make_skill(tmp_path, files=["big.txt"])
    (d / "big.txt").write_text("A" * 20000, encoding="utf-8")
    msg = execute_skill(str(d / "SKILL.md"), "hi")
    assert "A" * 9000 not in msg  # 8K 截断
    assert "截断" in msg


def test_attachment_count_capped_at_5(tmp_path):
    """附件总数上限 5 个。"""
    from agent.skill_commands import read_skill_attachment_files

    d = _make_skill(tmp_path, files=[f"f{i}.txt" for i in range(8)])
    for i in range(8):
        (d / f"f{i}.txt").write_text(f"content{i}", encoding="utf-8")
    items = read_skill_attachment_files(str(d), [f"f{i}.txt" for i in range(8)])
    assert len(items) == 5


def test_attachment_path_traversal_blocked(tmp_path):
    """附件路径逃出技能目录 → 跳过（防 ../ traversal）。"""
    from agent.skill_commands import read_skill_attachment_files

    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET", encoding="utf-8")
    d = _make_skill(tmp_path, files=["../secret.txt"])
    items = read_skill_attachment_files(str(d), ["../secret.txt"])
    assert len(items) == 0


# ---------------------------------------------------------------------------
# load_skill 工具
# ---------------------------------------------------------------------------

def test_load_skill_returns_attachments(tmp_path):
    """load_skill 返回 JSON 含 attachments 字段。"""
    from tools.skill_tools import _handle_load_skill

    d = _make_skill(tmp_path, files=["ref.md"])
    (d / "ref.md").write_text("附件内容 X", encoding="utf-8")

    result = json.loads(_handle_load_skill(
        {"name": "my-skill"}, omnimate_home=str(tmp_path),
    ))
    assert result.get("body") == "技能正文"
    atts = result.get("attachments")
    assert isinstance(atts, list) and len(atts) == 1
    assert atts[0]["path"] == "ref.md"
    assert "附件内容 X" in atts[0]["content"]


def test_skill_view_has_no_attachments(tmp_path):
    """skill_view（用户视角）不注入附件（保持现状）。"""
    import inspect
    import tools.skill_tools as st
    src = inspect.getsource(st._handle_skill_view)
    assert "attachments" not in src


def test_config_key_present():
    """config skills.file_attachment_max_chars 默认 8000（有真实读取点）。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["skills"]["file_attachment_max_chars"] == 8000
