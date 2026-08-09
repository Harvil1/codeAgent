"""skill_manage 工具：agent 自己创建/修改/归档技能。

这是"自学习"的关键——agent 把学到的方法沉淀成文件。

action：
  create      - 创建新技能
  edit        - 重写整个 SKILL.md
  patch       - 部分修改（查找替换）
  delete      - 归档技能（永不真删除）
  write_file  - 写附属文件（scripts/references/templates）
  remove_file - 删附属文件
"""

import json
from pathlib import Path

from tools.registry import registry
from tools.skill_usage import bump_patch, mark_agent_created, archive_skill


SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    "description": (
        "管理技能文件。可以创建、修改、归档技能。\n"
        "完成复杂任务后发现可复用的方法时，用 action='create' 保存。\n"
        "使用技能发现过时内容时，用 action='patch' 修复。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit", "patch", "delete",
                         "write_file", "remove_file"],
                "description": (
                    "create: 创建新技能\n"
                    "edit: 重写整个 SKILL.md\n"
                    "patch: 部分修改（查找替换）\n"
                    "delete: 归档技能（永不真删除）\n"
                    "write_file: 写附属文件\n"
                    "remove_file: 删附属文件"
                ),
            },
            "name": {
                "type": "string",
                "description": "技能名",
            },
            "content": {
                "type": "string",
                "description": "SKILL.md 完整内容（create/edit 时）",
            },
            "old_string": {
                "type": "string",
                "description": "要查找的文本（patch 时）",
            },
            "new_string": {
                "type": "string",
                "description": "替换为的文本（patch 时）",
            },
            "file_path": {
                "type": "string",
                "description": "附属文件路径（write_file/remove_file）",
            },
            "file_content": {
                "type": "string",
                "description": "附属文件内容（write_file）",
            },
            "absorbed_into": {
                "type": "string",
                "description": "归档时声明的合并目标（delete 时）",
            },
        },
        "required": ["action", "name"],
    },
}


def _get_skills_dir_from_context(kwargs: dict) -> Path:
    """从工具调用上下文获取技能目录。

    优先从 kwargs 获取 omnimate_home；否则回退到 constants 默认值。
    """
    home = kwargs.get("omnimate_home")
    if home:
        return Path(home) / "skills"
    # 回退到 constants 的默认 skills 目录
    from constants import skills_dir as _skills_dir
    return _skills_dir()


def _handle_skill_manage(args: dict, **kwargs) -> str:
    action = args.get("action")
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skills_dir = _get_skills_dir_from_context(kwargs)
    skill_dir = skills_dir / name
    skill_md = skill_dir / "SKILL.md"

    if action == "create":
        content = args.get("content", "")
        if not content.strip():
            return json.dumps({"error": "content 不能为空"}, ensure_ascii=False)
        if skill_dir.exists():
            return json.dumps({"error": f"技能已存在: {name}"}, ensure_ascii=False)

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(content, encoding="utf-8")

        # 标记为 agent 创建（如果是后台 curator 创建的）
        is_background = kwargs.get("is_background_review", False)
        if is_background:
            mark_agent_created(skills_dir, name)

        return json.dumps({
            "success": True,
            "message": f"技能已创建: {skill_dir}",
        }, ensure_ascii=False)

    elif action == "edit":
        # 重写整个 SKILL.md
        content = args.get("content", "")
        if not content.strip():
            return json.dumps({"error": "content 不能为空"}, ensure_ascii=False)
        if not skill_dir.exists():
            return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(content, encoding="utf-8")
        bump_patch(skills_dir, name)

        return json.dumps({
            "success": True,
            "message": f"已重写技能: {name}",
        }, ensure_ascii=False)

    elif action == "patch":
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        if not old_string:
            return json.dumps({"error": "old_string 不能为空"}, ensure_ascii=False)
        if not skill_md.exists():
            return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

        content = skill_md.read_text(encoding="utf-8")
        if old_string not in content:
            return json.dumps({"error": "未找到 old_string"}, ensure_ascii=False)

        new_content = content.replace(old_string, new_string, 1)
        skill_md.write_text(new_content, encoding="utf-8")

        bump_patch(skills_dir, name)  # 修改计数 +1

        return json.dumps({
            "success": True,
            "message": f"已修改技能: {name}",
        }, ensure_ascii=False)

    elif action == "delete":
        # 永不真删除，只归档
        absorbed_into = args.get("absorbed_into", "")
        ok, msg = archive_skill(skills_dir, name)
        return json.dumps({
            "success": ok,
            "message": msg,
            "absorbed_into": absorbed_into,
        }, ensure_ascii=False)

    elif action == "write_file":
        file_path = args.get("file_path", "")
        file_content = args.get("file_content", "")
        if not file_path:
            return json.dumps({"error": "file_path 不能为空"}, ensure_ascii=False)
        if not skill_dir.exists():
            return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

        # 安全：防止路径遍历
        target = (skill_dir / file_path).resolve()
        if not str(target).startswith(str(skill_dir.resolve())):
            return json.dumps({"error": "路径越界"}, ensure_ascii=False)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_content, encoding="utf-8")

        return json.dumps({
            "success": True,
            "message": f"已写入: {target}",
        }, ensure_ascii=False)

    elif action == "remove_file":
        file_path = args.get("file_path", "")
        if not file_path:
            return json.dumps({"error": "file_path 不能为空"}, ensure_ascii=False)

        target = (skill_dir / file_path).resolve()
        if not str(target).startswith(str(skill_dir.resolve())):
            return json.dumps({"error": "路径越界"}, ensure_ascii=False)
        if not target.exists():
            return json.dumps({"error": f"文件不存在: {target}"}, ensure_ascii=False)

        target.unlink()
        return json.dumps({
            "success": True,
            "message": f"已删除: {target}",
        }, ensure_ascii=False)

    else:
        return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)


registry.register(
    name="skill_manage",
    toolset="core",
    schema=SKILL_MANAGE_SCHEMA,
    handler=_handle_skill_manage,
    emoji="📚",
    isConcurrencySafe=False,  # 副作用：创建/更新/归档/删除技能文件，必须串行
)
