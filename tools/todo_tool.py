"""todo_write 工具：让 LLM 维护任务清单。

LLM 主动调用以追踪多步任务进度。系统约束同时只能 1 个 in_progress。
"""

import json

from agent.todo import get_todo_manager
from tools.registry import registry


TODO_WRITE_SCHEMA = {
    "name": "todo_write",
    "description": (
        "更新任务清单（替换整个列表）。用于追踪多步任务进度。"
        "约束：同时只能 1 个 in_progress，强制顺序聚焦。"
        "开始多步任务时创建清单，每步完成时更新状态。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "任务 ID（可选，默认用序号）",
                        },
                        "text": {
                            "type": "string",
                            "description": "任务描述",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "任务状态",
                        },
                    },
                    "required": ["text", "status"],
                },
                "description": "完整的任务列表（替换式，不是增量）",
            },
        },
        "required": ["items"],
    },
}


def _handle_todo_write(args: dict, **kwargs) -> str:
    items = args.get("items") or []
    manager = kwargs.get("todo_manager") or get_todo_manager()
    result = manager.write(items)
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="todo_write",
    toolset="core",
    schema=TODO_WRITE_SCHEMA,
    handler=_handle_todo_write,
    emoji="📋",
)
