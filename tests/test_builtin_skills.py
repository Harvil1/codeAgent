"""内置技能 frontmatter 校验：所有 SKILL.md 合法且可被扫描到。"""

from constants import builtin_skills_dir, all_skills_dirs
from agent.prompt_builder import _build_skill_index


def _skill_dirs():
    root = builtin_skills_dir()
    return sorted(
        (root / d.name for d in root.iterdir() if (root / d.name / "SKILL.md").is_file()),
        key=lambda p: p.name,
    )


def test_builtin_skills_have_valid_frontmatter():
    dirs = _skill_dirs()
    assert len(dirs) >= 14, f"内置技能应 >=14 个,实际 {len(dirs)}"
    for d in dirs:
        content = (d / "SKILL.md").read_text(encoding="utf-8")
        assert content.startswith("---"), f"{d.name}/SKILL.md 缺 frontmatter"
        parts = content.split("---", 2)
        assert len(parts) >= 3, f"{d.name}/SKILL.md frontmatter 不完整"
        fm = parts[1]
        assert "name:" in fm, f"{d.name}/SKILL.md 缺 name"
        assert "description:" in fm, f"{d.name}/SKILL.md 缺 description"


def test_builtin_skills_appear_in_index():
    index = _build_skill_index(all_skills_dirs())
    for d in _skill_dirs():
        assert f"/{d.name}" in index, f"{d.name} 未进技能索引"
