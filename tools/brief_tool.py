"""「简报」工具：让 AI 用固定格式说一句"我接下来要干啥"——与 Plan Mode 互补
（Plan 详细，Brief 一句话预告），给用户快速预览。

本文件属于工具层（tools/），被 tools/registry.py 自动发现注册。
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
    "parameters": {
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


def _handle_brief(args: dict, **kwargs) -> str:
    """把模型给的简报字段原样整理成 JSON 回显——一个"定格式"工具，不干活，
    只逼模型按统一字段（标题/步骤/风险/受众）输出，界面侧可稳定解析展示。

    参数：
    - args：工具参数字典。headline 是一句话标题；steps 是接下来的
      关键步骤列表；risks 是已知风险列表；audience 区分给用户看
      还是给审批流看。
    - kwargs：运行时注入的命名上下文（本工具用不到，占位满足统一签名——
      所有 handler 都必须是 (args, **kwargs) 形状，否则分发器叫不动它）。

    返回：JSON 字符串，即整理后的简报内容。
    """
    return json.dumps(
        {
            "headline": args["headline"],
            "steps": args.get("steps", []),
            "risks": args.get("risks", []),
            "audience": args.get("audience", "user"),
        },
        ensure_ascii=False,
    )


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="brief",
    schema=BRIEF_SCHEMA,
    handler=_handle_brief,
    toolset="core",
    emoji="📋",
    isConcurrencySafe=True,  # 纯回显没副作用，随便并发
)
