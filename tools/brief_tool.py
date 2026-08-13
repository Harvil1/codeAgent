"""brief 工具：让 LLM 输出结构化简报。

与 Plan Mode 互补：Plan 详细，Brief 一句话总结"我接下来要干啥"。
LLM 在重要操作前调用 brief 给用户预览。
"""
import json

from tools.registry import registry

BRIEF_SCHEMA = {
    "name": "brief",
    "description": (
        "输出结构化简报（一段话讲清'我接下来要干啥'）。"
        "用于 (1) 在重要操作前给用户快速预览；"
        "(2) 在 Plan Mode 之外提供更轻量的'我打算这么做'摘要。"
        "与 Plan Mode 互补：Plan 详细，Brief 一句话。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "一句话主标题",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "接下来的关键步骤（每项一句话）",
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已知风险或不确定（每项一句话）",
            },
            "audience": {
                "type": "string",
                "enum": ["user", "approval"],
                "default": "user",
                "description": "user=给用户看；approval=给审批流看（更正式）",
            },
        },
        "required": ["headline"],
    },
}


def _handle_brief(args: dict, kwargs: dict, ctx) -> str:
    """直接 echo args（格式化）——这是个输出格式约定工具。"""
    return json.dumps(
        {
            "headline": kwargs["headline"],
            "steps": kwargs.get("steps", []),
            "risks": kwargs.get("risks", []),
            "audience": kwargs.get("audience", "user"),
        },
        ensure_ascii=False,
    )


# 模块顶部注册（import 即生效）
registry.register(
    name="brief",
    schema=BRIEF_SCHEMA,
    handler=_handle_brief,
    toolset="core",
    emoji="📋",
    isConcurrencySafe=True,  # CCAR8 fix: 纯 echo 无副作用，可并发
)
