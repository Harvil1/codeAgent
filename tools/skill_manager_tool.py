"""skill_manage 工具：agent 自己动手创建、修改、归档技能——把可复用的方法
沉淀成磁盘上的 Markdown 技能文件（skill），下次直接翻出来照着做。

一个工具七种用法（由 action 参数区分）：
  create      - 创建新技能
  edit        - 重写整个 SKILL.md
  patch       - 部分修改（查找替换）
  delete      - 归档技能（永不真删除，只挪到 .archive/ 目录，随时可恢复）
  write_file  - 写附属文件（脚本/参考资料/模板等）
  remove_file - 删附属文件

在项目里的位置：属于工具层（tools/），注册进中央工具注册表暴露给 LLM；
使用统计委托给 tools/skill_usage.py。
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
    """算出本次操作该把技能写到哪个目录。

    技能目录跟数据主目录走（主目录可自定义），所以每次都要现算。

    参数：
    - kwargs：工具调用上下文；优先取里面的 codeagent_home（自定义数据主目录）

    返回：技能目录路径（<主目录>/skills）；没传 codeagent_home 就用 constants 里的默认值。
    """
    home = kwargs.get("codeagent_home")
    if home:
        return Path(home) / "skills"
    # 没传自定义主目录，就用全局默认的技能目录
    from constants import skills_dir as _skills_dir
    return _skills_dir()


def _handle_skill_manage(args: dict, **kwargs) -> str:
    """skill_manage 的实际处理函数：按 action 分发到对应的新建/改写/小修/归档/附属文件操作。

    参数：
    - args：LLM 传的工具参数——action（要做什么）、name（技能名）、
      content（create/edit 时的全文）、old_string/new_string（patch 时的查找替换对）、
      file_path/file_content（附属文件操作）、absorbed_into（归档时声明的合并去向）
    - kwargs：运行时上下文；这里看 codeagent_home（定位技能目录）
      和 is_background_review（是否后台维护工人创建的）

    返回：JSON 字符串，成功带 success=True + 消息，失败带 error 说明原因。
    """
    action = args.get("action")
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skills_dir = _get_skills_dir_from_context(kwargs)
    # name 是外部输入，先验"它必须只是一个目录名"：带分隔符（a/b、
    # D:\x）会让拼接整体跳出技能根，纯点号（..、.）会指到技能根本身，
    # 都等于绕过文件写入的审批闸门在技能目录外动手
    if Path(name).name != name or name.strip(".") == "":
        return json.dumps({"error": "技能名不合法（只能是目录名，不允许路径分隔符或点号）"}, ensure_ascii=False)
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

        # 只有后台维护工人（curator，定期整理技能/记忆的后台程序）创建的才标记为 agent 创建；
        # 用户当面让 agent 建的不标记——区别在于前者会被 curator 自动管理生命周期
        is_background = kwargs.get("is_background_review", False)
        if is_background:
            mark_agent_created(skills_dir, name)

        return json.dumps({
            "success": True,
            "message": f"技能已创建: {skill_dir}",
        }, ensure_ascii=False)

    elif action == "edit":
        # 整篇重写（对比 patch 的小修小补）
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

        bump_patch(skills_dir, name)  # 修改计数加一，进统计

        return json.dumps({
            "success": True,
            "message": f"已修改技能: {name}",
        }, ensure_ascii=False)

    elif action == "delete":
        # 设计铁律「完全可逆」：删除只是挪进 .archive/ 目录，永不真删，随时可恢复
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

        # 安全检查：解析后的最终路径必须还在技能目录里面，防止用 ../ 之类的写法逃出去写别的文件
        # （用 is_relative_to 而不是字符串前缀——前缀比较放得过宽，
        # skills2 这类兄弟目录也长得像技能目录的前缀）
        target = (skill_dir / file_path).resolve()
        if not target.is_relative_to(skill_dir.resolve()):
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
        # 同款防路径逃逸检查：最终路径不许跑出技能目录
        if not target.is_relative_to(skill_dir.resolve()):
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
    isConcurrencySafe=False,  # 有副作用（创建/更新/归档/删文件），必须一个一个来，不能并发

)
