"""Plan Mode 工具：exit_plan_mode。

LLM 在计划模式下调研完成后，调此工具请求用户审批计划。
handler 不真正"完成"任务，而是返回特殊 error_type="plan_approval_required"，
由 agent 主循环捕获并调用 plan_approval_callback 走审批流程。

discover_builtin_tools() 通过 AST 扫描自动发现本模块（顶层有 registry.register）。
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


def handle_exit_plan_mode(args: dict, **kwargs) -> str:
    """exit_plan_mode handler。

    返回特殊 error_type 让主循环走审批分支：
    - 空 plan → invalid_args（LLM 重试）
    - 非空 plan → plan_approval_required（主循环捕获，调回调）
    """
    plan = (args or {}).get("plan", "")
    if not isinstance(plan, str):
        plan = str(plan) if plan else ""
    plan = plan.strip()
    if not plan:
        return json.dumps({
            "error": "plan 字段不能为空。请提供完整的实施计划摘要：要改什么文件、为什么、步骤、风险点。",
            "error_type": "invalid_args",
        }, ensure_ascii=False)

    return json.dumps({
        "error": "等待用户审批计划",
        "error_type": "plan_approval_required",
        "plan": plan,
    }, ensure_ascii=False)


# 模块顶层 register —— 被 discover_builtin_tools AST 扫描自动发现
registry.register(
    name="exit_plan_mode",
    toolset="plan",
    schema={
        "name": "exit_plan_mode",
        "description": (
            "Plan 写好后调此工具请求用户审批。批准后才能进入执行模式。"
            "调用此工具前必须已完成调研。"
            "plan 参数要包含完整的实施步骤：要改什么文件、为什么、步骤、风险点。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "完整的实施计划摘要。",
                }
            },
            "required": ["plan"],
        },
    },
    handler=handle_exit_plan_mode,
    emoji="📋",
    isConcurrencySafe=False,  # 状态变更：触发审批流程（切换 plan/execute 模式），必须串行
)
