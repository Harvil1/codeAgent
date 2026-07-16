"""记忆工具：管理持久化多文件记忆。

action:
  - save: 创建新记忆（必需 name/description/type）
  - update: 更新已有记忆字段
  - delete: 软删除（移到 .archive/）
  - load: 读 body
  - list: 列出所有记忆
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "管理持久化记忆（跨会话保存）。每条记忆是一个独立文件，含 frontmatter + body。\n"
        "写入立即落盘，但索引下次会话才注入到 system prompt（保护 prompt cache）。\n\n"
        "action:\n"
        "  - save: 创建新记忆（必需 name/description/type）\n"
        "  - update: 更新字段（必需 id）\n"
        "  - delete: 软删除（必需 id）\n"
        "  - load: 读完整 body（必需 id）\n"
        "  - list: 列出所有记忆\n\n"
        "type 可选值: user / feedback / project / reference / other"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "update", "delete", "load", "list"],
            },
            "id": {"type": "string", "description": "update/delete/load 时必需"},
            "name": {"type": "string", "description": "save 时必需；update 可选"},
            "description": {"type": "string", "description": "save 时必需；update 可选"},
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference", "other"],
                "description": "save 时必需；update 可选",
            },
            "body": {"type": "string", "description": "save/update 时可选"},
        },
        "required": ["action"],
    },
}


def _handle_memory(args: dict, **kwargs) -> str:
    action = args.get("action")
    store = kwargs.get("memory_store")

    if store is None:
        return json.dumps({
            "success": False, "error": "记忆系统未初始化",
        }, ensure_ascii=False)

    try:
        if action == "save":
            mid = store.save(
                name=args.get("name", ""),
                description=args.get("description", ""),
                type=args.get("type", "other"),
                body=args.get("body", ""),
            )
            return json.dumps({
                "success": True, "action": "save", "id": mid,
                "message": "已保存（索引下次会话生效）",
            }, ensure_ascii=False)

        if action == "update":
            mid = args.get("id", "")
            entry = store.update(
                mid,
                name=args.get("name"),
                description=args.get("description"),
                type=args.get("type"),
                body=args.get("body"),
            )
            return json.dumps({
                "success": True, "action": "update", "id": mid,
                "entry": {
                    "name": entry.name, "description": entry.description,
                    "type": entry.type,
                },
            }, ensure_ascii=False)

        if action == "delete":
            mid = args.get("id", "")
            ok = store.delete(mid)
            if not ok:
                return json.dumps({
                    "success": False, "error": f"未找到: {mid}",
                }, ensure_ascii=False)
            return json.dumps({
                "success": True, "action": "delete", "id": mid,
                "message": "已软删除到 .archive/",
            }, ensure_ascii=False)

        if action == "load":
            mid = args.get("id", "")
            entry = store.get(mid)
            if entry is None:
                return json.dumps({
                    "success": False, "error": f"未找到: {mid}",
                }, ensure_ascii=False)
            return json.dumps({
                "success": True, "id": mid,
                "name": entry.name, "description": entry.description,
                "type": entry.type, "body": entry.body,
                "created_at": entry.created_at.isoformat(timespec="seconds"),
                "updated_at": entry.updated_at.isoformat(timespec="seconds"),
            }, ensure_ascii=False)

        if action == "list":
            entries = store.list_all()
            summaries = [
                {"id": e.id, "name": e.name, "description": e.description, "type": e.type}
                for e in entries
            ]
            return json.dumps({
                "success": True, "count": len(entries), "memories": summaries,
            }, ensure_ascii=False)

        return json.dumps({
            "success": False, "error": f"未知 action: {action}",
        }, ensure_ascii=False)

    except (ValueError, KeyError) as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.exception("memory 工具异常")
        return json.dumps({
            "success": False, "error": f"内部错误: {e}",
        }, ensure_ascii=False)


registry.register(
    name="memory",
    toolset="core",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    emoji="🧠",
)
