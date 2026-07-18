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


# ============================================================================
# Task 2: exit_plan_mode handler
# ============================================================================

def test_exit_plan_mode_registered():
    """exit_plan_mode 被注册到 registry 的 plan toolset。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()  # 触发 tools/*.py 自注册

    entry = registry._tools.get("exit_plan_mode")
    assert entry is not None, "exit_plan_mode 未注册"
    assert entry.toolset == "plan", f"toolset 应为 'plan'，实际 '{entry.toolset}'"


def test_exit_plan_mode_empty_plan_returns_invalid_args():
    """空 plan 返回 invalid_args 错误。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = registry.dispatch("exit_plan_mode", {"plan": ""})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"
    assert "plan" in data["error"]


def test_exit_plan_mode_missing_plan_returns_invalid_args():
    """缺 plan 参数返回 invalid_args 错误。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = registry.dispatch("exit_plan_mode", {})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"


def test_exit_plan_mode_valid_plan_returns_approval_required():
    """非空 plan 返回 plan_approval_required + 原文。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    plan_text = "## 步骤\n1. 改 foo.py\n2. 加测试"
    result_json = registry.dispatch("exit_plan_mode", {"plan": plan_text})
    data = json.loads(result_json)
    assert data["error_type"] == "plan_approval_required"
    assert data["plan"] == plan_text


def test_exit_plan_mode_whitespace_only_plan_returns_invalid_args():
    """纯空白 plan 视为空。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = registry.dispatch("exit_plan_mode", {"plan": "   \n\t  "})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"
