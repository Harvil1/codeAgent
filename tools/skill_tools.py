"""技能查看工具：skills_list + skill_view。

skills_list：列出所有可用技能
skill_view：查看某技能的完整内容
"""

import json
from pathlib import Path

from tools.registry import registry
from agent.skill_commands import parse_frontmatter
from tools.skill_usage import bump_view, load_usage


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "列出所有可用技能。",
    "parameters": {"type": "object", "properties": {}},
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "查看某技能的完整内容。",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "技能名"},
        },
        "required": ["name"],
    },
}


def _get_skills_dir_from_context(kwargs: dict) -> Path:
    """从工具调用上下文获取技能目录。"""
    home = kwargs.get("harvil_home")
    if home:
        return Path(home) / "skills"
    from constants import skills_dir as _skills_dir
    return _skills_dir()


def _handle_skills_list(args: dict, **kwargs) -> str:
    skills_dir = _get_skills_dir_from_context(kwargs)
    if not skills_dir.exists():
        return json.dumps({"skills": []}, ensure_ascii=False)

    usage = load_usage(skills_dir)
    skills = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        name = skill_md.parent.name
        rec = usage.get(name, {})
        # 跳过归档的
        if rec.get("state") == "archived":
            continue

        # 尝试从 frontmatter 读描述
        description = rec.get("description", "")
        if not description:
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(content)
                description = frontmatter.get("description", "")
            except Exception:
                pass

        skills.append({
            "name": name,
            "description": description,
            "use_count": rec.get("use_count", 0),
            "view_count": rec.get("view_count", 0),
            "state": rec.get("state", "active"),
        })

    return json.dumps({"skills": skills}, ensure_ascii=False)


def _handle_skill_view(args: dict, **kwargs) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skills_dir = _get_skills_dir_from_context(kwargs)
    skill_md = skills_dir / name / "SKILL.md"
    if not skill_md.exists():
        return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

    content = skill_md.read_text(encoding="utf-8")
    bump_view(skills_dir, name)  # 查看计数 +1

    return json.dumps({
        "name": name,
        "content": content,
        "path": str(skill_md),
    }, ensure_ascii=False)


registry.register(
    name="skills_list",
    toolset="core",
    schema=SKILLS_LIST_SCHEMA,
    handler=_handle_skills_list,
    emoji="📋",
)

registry.register(
    name="skill_view",
    toolset="core",
    schema=SKILL_VIEW_SCHEMA,
    handler=_handle_skill_view,
    emoji="👁️",
)


# ---------------------------------------------------------------------------
# load_skill：LLM 主动按需加载技能正文
# ---------------------------------------------------------------------------

LOAD_SKILL_SCHEMA = {
    "name": "load_skill",
    "description": (
        "按名字加载技能的完整指令正文。system prompt 里有技能索引"
        "（名字+描述，~100 tokens/技能），当你判断需要某个技能的详细流程时"
        "调用此工具获取完整内容（~2000 tokens/技能）。"
        "区别于 skill_view：load_skill 只返回指令正文（去 frontmatter），"
        "专门给 LLM 按需读取执行。"
        "\n\n支持技能束：传 name=\"bundle:<bundle_name>\" 一次性加载多个技能"
        "（在 ~/.agent/.skill-bundles.json 配置）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "技能名"},
        },
        "required": ["name"],
    },
}


def _handle_load_skill(args: dict, **kwargs) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skills_dir = _get_skills_dir_from_context(kwargs)

    # batch1-T3: 支持 bundle:<name> 加载技能束
    if name.startswith("bundle:"):
        bundle_name = name[len("bundle:"):]
        from agent.skill_bundle import load_bundle
        result = load_bundle(bundle_name, skills_dir)
        # 对成功加载的技能 bump view 计数
        for sname in result.get("skills_loaded", []):
            try:
                bump_view(skills_dir, sname)
            except Exception:
                pass
        return json.dumps(result, ensure_ascii=False)

    skill_md = skills_dir / name / "SKILL.md"
    if not skill_md.exists():
        return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

    content = skill_md.read_text(encoding="utf-8")
    # 去掉 frontmatter，只返回指令正文
    _, body = parse_frontmatter(content)

    bump_view(skills_dir, name)  # 加载也计入 view 计数

    return json.dumps({
        "name": name,
        "body": body.strip(),
        "path": str(skill_md),
    }, ensure_ascii=False)


registry.register(
    name="load_skill",
    toolset="core",
    schema=LOAD_SKILL_SCHEMA,
    handler=_handle_load_skill,
    emoji="📖",
)
