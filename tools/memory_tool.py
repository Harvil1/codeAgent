"""记忆工具：让 LLM 自己增删改查长期记忆（跨会话保存的那种）。

在项目里的位置：tools 层的 core 工具，真正的存取逻辑在 agent/memory_store.py 的
MemoryStore 里，本文件只负责把 LLM 的调用参数翻译成对 store 的操作。

支持 5 种 action（动作）：
  - save: 新建一条记忆（必填 name/description/type；同主题同名的会自动变成更新，不堆积）
  - update: 改已有记忆的字段（必填 id）
  - delete: 软删除——不真删，移到 .archive/ 归档目录，随时能找回
  - load: 读某条记忆的完整正文
  - list: 列出全部记忆的摘要
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
        "  - save: 创建或更新记忆（必需 name/description/type；同 topic 同 name 自动更新）\n"
        "  - update: 更新字段（必需 id）\n"
        "  - delete: 软删除（必需 id）\n"
        "  - load: 读完整 body（必需 id）\n"
        "  - list: 列出所有记忆\n\n"
        "type 可选值: user / feedback / project / reference / other\n"
        "topic: 可选主题，记忆按主题组织到 .memory/{topic}.jsonl；\n"
        "        默认 general。同一主题下建议用一致的 name，同 name 会自动更新而非堆积。\n\n"
        "⚠️ 写入即维护：保存前先用 action=list 查重，同主题同 name 用 update 更新，\n"
        "   避免记忆无限堆积（索引应保持精简）。\n\n"
        "记忆的详略分层（索引只进前两层，省 token）：\n"
        "  - name: 标题（索引定位用）\n"
        "  - description: 一句话钩子（索引行展示）\n"
        "  - summary: 摘要（80-100 字符，判断相关性用，避免读全文）\n"
        "  - body: 全文正文（完整内容，load 时返回）"
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
            "summary": {
                "type": "string",
                "description": "摘要（80-100 字符，判断相关性用）；save 可选，update 可选",
            },
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference", "other"],
                "description": "save 时必需；update 可选",
            },
            "body": {"type": "string", "description": "save/update 时可选"},
            "topic": {
                "type": "string",
                "description": "主题（save 可选，默认 general）",
            },
        },
        "required": ["action"],
    },
}


def _handle_memory(args: dict, **kwargs) -> str:
    """memory 工具的总入口：按 action 分发到 store 的对应操作。

    遵守统一契约：参数从 args 拿，命名上下文（这里是 memory_store）从 kwargs 拿，
    返回 JSON 字符串。

    参数：
        args: LLM 传来的工具参数，核心是 action（save/update/delete/load/list），
              其余键（id/name/description/type/body/summary/topic）按 action 需要取用
        kwargs: dispatch 透传的命名上下文，这里只用到 memory_store（记忆仓库实例）

    返回：JSON 字符串。成功带 success=True 和结果；失败带 success=False 和 error。
    """
    action = args.get("action")
    store = kwargs.get("memory_store")

    if store is None:
        return json.dumps({
            "success": False, "error": "记忆系统未初始化",
        }, ensure_ascii=False)

    try:
        if action == "save":
            topic = args.get("topic", "general")
            name = args.get("name", "")
            # 「保存前先查重」：同 topic 同 name 已有 → 本次自动算更新，不会越存越多
            # 查重必须限定在 save 实际会写入的那个分区，
            # 否则可能出现「回复说已更新、实际却在另一个分区新建了一条」的谎报
            existing = store.find_by_topic_name(
                topic, name, type=args.get("type", "other"),
            )
            mid = store.save(
                name=name,
                description=args.get("description", ""),
                type=args.get("type", "other"),
                body=args.get("body", ""),
                summary=args.get("summary", ""),
                topic=topic,
            )
            was_update = existing is not None
            return json.dumps({
                "success": True,
                "action": "update" if was_update else "save",
                "id": mid,
                "message": (
                    "已更新同名记忆（写入即维护，避免堆积）"
                    if was_update else "已保存（索引下次会话生效）"
                ),
            }, ensure_ascii=False)

        if action == "update":
            mid = args.get("id", "")
            entry = store.update(
                mid,
                name=args.get("name"),
                description=args.get("description"),
                type=args.get("type"),
                body=args.get("body"),
                summary=args.get("summary"),
            )
            return json.dumps({
                "success": True, "action": "update", "id": mid,
                "entry": {
                    "name": entry.name, "description": entry.description,
                    "type": entry.type, "summary": entry.summary,
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
                "summary": entry.summary,  # summary 字段方便先看摘要再决定是否读全文
                "created_at": entry.created_at.isoformat(timespec="seconds"),
                "updated_at": entry.updated_at.isoformat(timespec="seconds"),
            }, ensure_ascii=False)

        if action == "list":
            entries = store.list_all()
            summaries = [
                {
                    "id": e.id, "name": e.name, "description": e.description,
                    "type": e.type, "summary": e.summary,  # 列表也带 summary，一眼判断相关性
                }
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
    isConcurrencySafe=False,  # 一个工具混着读写动作：save/update/delete 会改数据，虽然 load/list 只读但没法拆开，所以整体按串行处理最安全
)
