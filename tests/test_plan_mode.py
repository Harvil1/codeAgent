"""Plan Mode 测试。

覆盖：
- plan 工具集定义
- exit_plan_mode handler
- AIAgent.plan_mode 字段
- 主循环工具集切换
- reminder 注入
- 审批分支（通过/拒绝/回调异常/无回调）
- CLI slash 命令
- 端到端集成
"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Task 1: plan 工具集
# ============================================================================

def test_plan_toolset_defined():
    """plan 工具集存在且包含 8 个只读工具。"""
    from toolsets import resolve_toolset, TOOLSETS

    assert "plan" in TOOLSETS, "TOOLSETS 缺少 'plan' 条目"

    tools = resolve_toolset("plan")
    expected = {
        "read_file", "search_files",
        "skills_list", "skill_view", "load_skill",
        "session_search",
        "todo_write",
        "exit_plan_mode",
    }
    assert set(tools) == expected, f"plan 工具集内容不符: {set(tools) ^ expected}"
    assert len(tools) == 8


def test_plan_toolset_excludes_destructive_tools():
    """plan 工具集不含 terminal/write_file/skill_manage 等修改类工具。"""
    from toolsets import resolve_toolset

    tools = resolve_toolset("plan")
    forbidden = {
        "terminal", "write_file", "skill_manage", "memory",
        "delegate_task", "execute_code",
        "task_create", "task_update", "task_complete",
    }
    assert not (set(tools) & forbidden), (
        f"plan 工具集不应包含修改类工具，发现: {set(tools) & forbidden}"
    )
