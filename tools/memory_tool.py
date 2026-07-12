"""记忆工具：让 agent 主动保存/修改持久化记忆。

target：
  - 'memory': agent 的笔记（环境事实、项目约定）
  - 'user': 用户画像（偏好、沟通风格）

action：
  - 'add': 添加一条新记忆
  - 'replace': 替换已有记忆（按旧内容匹配）
  - 'remove': 删除一条记忆

注意：写入立即落盘，但下次会话才注入到 system prompt（保护 prompt cache）。

当前实现：占位版本，handler 通过 kwargs 接收 memory_store。
完整实现在 04-memory.md 中完善（让 handler 真正调用 memory_store.modify）。
"""

import json
from tools.registry import registry


MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "管理持久化记忆（跨会话保存）。用于保存用户偏好、"
        "环境细节、工具怪癖等稳定事实。\n"
        "写入会立即落盘，但下次会话才注入到 system prompt"
        "（保护 prompt cache）。\n\n"
        "target:\n"
        "  - 'memory': agent 的笔记（环境事实、项目约定）\n"
        "  - 'user': 用户画像（偏好、沟通风格）\n\n"
        "action:\n"
        "  - 'add': 添加一条新记忆\n"
        "  - 'replace': 替换已有记忆（按旧内容匹配）\n"
        "  - 'remove': 删除一条记忆"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "default": "memory",
            },
            "content": {
                "type": "string",
                "description": "记忆内容（声明式事实，不是指令）",
            },
            "old_content": {
                "type": "string",
                "description": "replace 时的旧内容",
            },
        },
        "required": ["action", "content"],
    },
}


def _handle_memory(args: dict, **kwargs) -> str:
    """处理记忆操作。

    通过 kwargs 接收 memory_store（由 agent 在 dispatch 时注入）。
    04 阶段会完善：真正调用 memory_store.modify()。
    """
    action = args.get("action")
    target = args.get("target", "memory")
    content = args.get("content", "")
    old_content = args.get("old_content", "")

    memory_store = kwargs.get("memory_store")

    if memory_store is None:
        # 记忆系统未初始化（例如子代理无记忆场景）
        return json.dumps({
            "success": False,
            "error": "记忆系统未初始化",
        }, ensure_ascii=False)

    try:
        memory_store.modify(action, target, content, old_content)
        return json.dumps({
            "success": True,
            "action": action,
            "target": target,
            "message": f"已{action}到 {target} 记忆（下次会话生效）",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        }, ensure_ascii=False)


registry.register(
    name="memory",
    toolset="core",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    emoji="🧠",
)
