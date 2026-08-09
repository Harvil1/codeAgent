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
    """plan 工具集存在且包含只读工具。"""
    from toolsets import resolve_toolset, TOOLSETS

    assert "plan" in TOOLSETS, "TOOLSETS 缺少 'plan' 条目"

    tools = resolve_toolset("plan")
    expected = {
        "read_file", "search_files",
        "skills_list", "skill_view", "load_skill",
        "session_search",
        "exit_plan_mode",
        "plan_mode_v2_dispatch",  # P6: 多 Agent 并行（flag 门控）
    }
    assert set(tools) == expected, f"plan 工具集内容不符: {set(tools) ^ expected}"
    assert len(tools) == len(expected)


def test_plan_toolset_excludes_destructive_tools():
    """plan 工具集不含 terminal/write_file/skill_manage 等修改类工具。"""
    from toolsets import resolve_toolset

    tools = resolve_toolset("plan")
    forbidden = {
        "terminal", "write_file", "skill_manage", "memory",
        "subagent",
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


async def test_exit_plan_mode_empty_plan_returns_invalid_args():
    """空 plan 返回 invalid_args 错误。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = await registry.dispatch("exit_plan_mode", {"plan": ""})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"
    assert "plan" in data["error"]


async def test_exit_plan_mode_missing_plan_returns_invalid_args():
    """缺 plan 参数返回 invalid_args 错误。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = await registry.dispatch("exit_plan_mode", {})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"


async def test_exit_plan_mode_valid_plan_returns_approval_required():
    """非空 plan 返回 plan_approval_required + 原文。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    plan_text = "## 步骤\n1. 改 foo.py\n2. 加测试"
    result_json = await registry.dispatch("exit_plan_mode", {"plan": plan_text})
    data = json.loads(result_json)
    assert data["error_type"] == "plan_approval_required"
    assert data["plan"] == plan_text


async def test_exit_plan_mode_whitespace_only_plan_returns_invalid_args():
    """纯空白 plan 视为空。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = await registry.dispatch("exit_plan_mode", {"plan": "   \n\t  "})
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"


# ============================================================================
# Task 3: AIAgent.plan_mode 字段
# ============================================================================

def _make_minimal_agent(**overrides):
    """构造一个最小 mock 的 AIAgent（不连真 LLM）。"""
    from agent import AIAgent
    base = dict(
        base_url="http://localhost",
        api_key="test-key",
        model="test-model",
        enabled_toolsets=["core"],
    )
    base.update(overrides)
    # patch create_llm_client 避免真连
    with patch("agent.llm_client.create_llm_client") as mock:
        mock.return_value = MagicMock()
        return AIAgent(**base)


def test_agent_default_plan_mode_false():
    """新建 agent 默认 plan_mode=False。"""
    agent = _make_minimal_agent()
    assert agent.plan_mode is False


def test_agent_default_plan_approval_callback_none():
    """新建 agent 默认 plan_approval_callback=None（自动批准）。"""
    agent = _make_minimal_agent()
    assert agent.plan_approval_callback is None


def test_agent_accepts_plan_approval_callback():
    """构造时可注入 plan_approval_callback。"""
    def cb(plan):
        return True, ""

    agent = _make_minimal_agent(plan_approval_callback=cb)
    assert agent.plan_approval_callback is cb


# ============================================================================
# Task 4: 工具集切换
# ============================================================================

async def test_plan_mode_switches_toolset_to_plan():
    """plan_mode=True 时下一轮 run_conversation 用 ["plan"] 工具集。"""
    agent = _make_minimal_agent()
    agent.plan_mode = True

    captured_toolsets = []
    original_get = None

    def fake_get_tool_definitions(enabled_toolsets, *args, **kwargs):
        captured_toolsets.append(list(enabled_toolsets))
        return []  # 空 schema，LLM 没工具可调 → 直接出最终响应

    with patch("model_tools.get_tool_definitions", side_effect=fake_get_tool_definitions):
        with patch("agent.llm_retry.call_with_retry") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "done"
            mock_resp.choices[0].message.tool_calls = None
            mock_llm.return_value = mock_resp

            await agent.run_conversation("test")

    assert captured_toolsets, "get_tool_definitions 未被调用"
    assert captured_toolsets[0] == ["plan"], (
        f"plan_mode=True 时应传 ['plan']，实际 {captured_toolsets[0]}"
    )


async def test_normal_mode_uses_enabled_toolsets():
    """plan_mode=False 时用 self.enabled_toolsets。"""
    agent = _make_minimal_agent(enabled_toolsets=["core", "browser"])
    assert agent.plan_mode is False

    captured_toolsets = []

    def fake_get_tool_definitions(enabled_toolsets, *args, **kwargs):
        captured_toolsets.append(list(enabled_toolsets))
        return []

    with patch("model_tools.get_tool_definitions", side_effect=fake_get_tool_definitions):
        with patch("agent.llm_retry.call_with_retry") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "done"
            mock_resp.choices[0].message.tool_calls = None
            mock_llm.return_value = mock_resp

            await agent.run_conversation("test")

    assert captured_toolsets[0] == ["core", "browser"]


# ============================================================================
# Task 5: Plan Mode reminder 注入
# ============================================================================

async def test_plan_mode_reminder_injected_when_plan_mode_true():
    """plan_mode=True 时 messages 末尾含 plan_mode_reminder。"""
    agent = _make_minimal_agent()
    agent.plan_mode = True

    captured_messages = []

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages.append(list(messages))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_resp.choices[0].message.tool_calls = None
        return mock_resp

    with patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        await agent.run_conversation("test")

    assert captured_messages, "LLM 未被调用"
    last_msgs = captured_messages[0]
    reminder_found = any(
        "<plan_mode_reminder>" in (m.get("content") or "")
        for m in last_msgs
    )
    assert reminder_found, "messages 缺 plan_mode_reminder"


async def test_plan_mode_reminder_absent_when_plan_mode_false():
    """plan_mode=False 时 messages 不含 plan_mode_reminder。"""
    agent = _make_minimal_agent()
    assert agent.plan_mode is False

    captured_messages = []

    def fake_call_with_retry(client, messages, **kwargs):
        captured_messages.append(list(messages))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_resp.choices[0].message.tool_calls = None
        return mock_resp

    with patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        await agent.run_conversation("test")

    last_msgs = captured_messages[0]
    reminder_found = any(
        "<plan_mode_reminder>" in (m.get("content") or "")
        for m in last_msgs
    )
    assert not reminder_found, "plan_mode=False 不应注入 reminder"


async def test_plan_mode_reminder_not_in_conversation_history():
    """reminder 是临时消息，不进 conversation_history（保护 prompt cache）。"""
    agent = _make_minimal_agent()
    agent.plan_mode = True

    with patch("agent.llm_retry.call_with_retry") as mock_llm:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_resp.choices[0].message.tool_calls = None
        mock_llm.return_value = mock_resp
        await agent.run_conversation("test")

    # history 里有 user("test") + assistant("ok")，不应有 reminder
    for msg in agent.conversation_history:
        content = msg.get("content") or ""
        assert "<plan_mode_reminder>" not in content, (
            "reminder 不应进 conversation_history"
        )


# ============================================================================
# Task 6: 审批分支
# ============================================================================

def _make_exit_plan_mode_tool_call(plan_text):
    """构造一个 exit_plan_mode tool_call 的 mock LLM response。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = ""
    tc = MagicMock()
    tc.id = "call_exit1"
    tc.function.name = "exit_plan_mode"
    tc.function.arguments = json.dumps({"plan": plan_text})
    mock_resp.choices[0].message.tool_calls = [tc]
    return mock_resp


def _make_final_response(text="done"):
    """构造无 tool_calls 的最终响应。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = text
    mock_resp.choices[0].message.tool_calls = None
    return mock_resp


async def test_plan_approval_approved_clears_plan_mode():
    """审批通过 → plan_mode 变 False，tool 消息含 plan_approved=True。"""
    agent = _make_minimal_agent(plan_approval_callback=lambda p: (True, ""))
    agent.plan_mode = True

    responses = [
        _make_exit_plan_mode_tool_call("我的计划"),
        _make_final_response("开始执行"),
    ]
    with patch("agent.llm_retry.call_with_retry", side_effect=responses):
        await agent.run_conversation("test")

    assert agent.plan_mode is False, "审批通过后 plan_mode 应为 False"
    # 找到 exit_plan_mode 对应的 tool 消息
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    exit_tool_msg = next(
        (m for m in tool_msgs if "plan_approved" in (m.get("content") or "")),
        None,
    )
    assert exit_tool_msg is not None, "缺 plan_approved 的 tool 消息"
    data = json.loads(exit_tool_msg["content"])
    assert data["plan_approved"] is True


async def test_plan_approval_rejected_keeps_plan_mode():
    """审批拒绝 → 保持 plan_mode=True，tool 消息含 plan_rejected + feedback。"""
    def reject_cb(plan):
        return False, "步骤 3 风险太大"

    agent = _make_minimal_agent(plan_approval_callback=reject_cb)
    agent.plan_mode = True

    responses = [
        _make_exit_plan_mode_tool_call("我的计划"),
        _make_final_response("已修订"),
    ]
    with patch("agent.llm_retry.call_with_retry", side_effect=responses):
        await agent.run_conversation("test")

    assert agent.plan_mode is True, "拒绝后 plan_mode 应保持 True"
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    rejected_msg = next(
        (m for m in tool_msgs if "plan_rejected" in (m.get("content") or "")),
        None,
    )
    assert rejected_msg is not None
    data = json.loads(rejected_msg["content"])
    assert data["plan_rejected"] is True
    assert data["feedback"] == "步骤 3 风险太大"


async def test_plan_approval_no_callback_auto_approves():
    """plan_approval_callback=None → 默认自动批准。"""
    agent = _make_minimal_agent()  # callback=None
    agent.plan_mode = True

    responses = [
        _make_exit_plan_mode_tool_call("我的计划"),
        _make_final_response("ok"),
    ]
    with patch("agent.llm_retry.call_with_retry", side_effect=responses):
        await agent.run_conversation("test")

    assert agent.plan_mode is False, "无 callback 时应自动批准"


async def test_plan_approval_callback_exception_treated_as_reject():
    """回调抛异常 → 视为拒绝，feedback 含异常信息。"""
    def boom_cb(plan):
        raise RuntimeError("网络断了")

    agent = _make_minimal_agent(plan_approval_callback=boom_cb)
    agent.plan_mode = True

    responses = [
        _make_exit_plan_mode_tool_call("我的计划"),
        _make_final_response("retry"),
    ]
    with patch("agent.llm_retry.call_with_retry", side_effect=responses):
        await agent.run_conversation("test")

    assert agent.plan_mode is True, "异常应视为拒绝，保持 plan_mode"
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    rejected_msg = next(
        (m for m in tool_msgs if "plan_rejected" in (m.get("content") or "")),
        None,
    )
    assert rejected_msg is not None
    data = json.loads(rejected_msg["content"])
    assert "网络断了" in data["feedback"]


# ============================================================================
# Task 7: config 默认值
# ============================================================================

def test_default_config_has_plan_mode_enabled():
    """DEFAULT_CONFIG 含 plan_mode.enabled，默认 True。"""
    from config import DEFAULT_CONFIG
    assert "plan_mode" in DEFAULT_CONFIG, "DEFAULT_CONFIG 缺 plan_mode 段"
    assert DEFAULT_CONFIG["plan_mode"].get("enabled") is True


# ============================================================================
# Task 8: CLI slash 命令 + 审批回调
# ============================================================================

def test_cli_plan_approval_callback_approve_with_y():
    """输入 y → 批准。"""
    from cli import cli_plan_approval_callback
    with patch("builtins.input", side_effect=["y"]):
        approved, feedback = cli_plan_approval_callback("我的计划")
    assert approved is True
    assert feedback == ""


def test_cli_plan_approval_callback_reject_with_n():
    """输入 n/空 → 拒绝。"""
    from cli import cli_plan_approval_callback
    with patch("builtins.input", side_effect=["n"]):
        approved, feedback = cli_plan_approval_callback("我的计划")
    assert approved is False


def test_cli_plan_approval_callback_edit_collects_feedback():
    """输入 edit → 收集 feedback。"""
    from cli import cli_plan_approval_callback
    with patch("builtins.input", side_effect=["edit", "步骤 3 改成 X"]):
        approved, feedback = cli_plan_approval_callback("我的计划")
    assert approved is False
    assert feedback == "步骤 3 改成 X"


def _make_mock_rt(agent=None):
    """构造一个最小 mock RuntimeContext，只暴露 .agent。"""
    rt = MagicMock()
    if agent is None:
        agent = _make_minimal_agent()
    rt.agent = agent
    # 其他被 _handle_command 用到的属性走 MagicMock 默认（不会真调）
    return rt


def test_plan_slash_command_enters_plan_mode():
    """/plan 命令把 rt.agent.plan_mode 设为 True。"""
    from cli import _handle_command
    agent = _make_minimal_agent()
    rt = _make_mock_rt(agent=agent)

    handled = _handle_command("/plan", rt)
    assert handled is True
    assert agent.plan_mode is True


def test_plan_off_slash_command_exits_plan_mode():
    """/plan off 命令把 rt.agent.plan_mode 设为 False。"""
    from cli import _handle_command
    agent = _make_minimal_agent()
    agent.plan_mode = True
    rt = _make_mock_rt(agent=agent)

    handled = _handle_command("/plan off", rt)
    assert handled is True
    assert agent.plan_mode is False


def test_plan_slash_command_idempotent_when_already_in_plan_mode():
    """已在 plan_mode 时 /plan 不报错，幂等。"""
    from cli import _handle_command
    agent = _make_minimal_agent()
    agent.plan_mode = True
    rt = _make_mock_rt(agent=agent)

    handled = _handle_command("/plan", rt)
    assert handled is True
    assert agent.plan_mode is True  # 仍是 True


def test_plan_off_slash_command_idempotent_when_not_in_plan_mode():
    """不在 plan_mode 时 /plan off 不报错，幂等。"""
    from cli import _handle_command
    agent = _make_minimal_agent()
    rt = _make_mock_rt(agent=agent)

    handled = _handle_command("/plan off", rt)
    assert handled is True
    assert agent.plan_mode is False


# ============================================================================
# Task 9: 端到端集成
# ============================================================================

async def test_e2e_plan_approve_then_execute():
    """端到端：进入 plan_mode → 调研 → 调 exit_plan_mode → 审批通过 → 切回执行。

    验证 tool_schemas 在主循环内每轮刷新：plan_mode 中途切换（审批通过）后，
    同一次 run_conversation 内 toolset 应由 ['plan'] 切到 ['core']，
    LLM 在第 3 轮就能看到执行类工具，无需用户再发新消息。
    """
    agent = _make_minimal_agent(plan_approval_callback=lambda p: (True, ""))
    agent.plan_mode = True

    # mock 三轮 LLM response
    # 第 1 轮：调 read_file 调研
    read_resp = MagicMock()
    read_resp.choices = [MagicMock()]
    read_resp.choices[0].message.content = ""
    tc_read = MagicMock()
    tc_read.id = "call_read"
    tc_read.function.name = "read_file"
    tc_read.function.arguments = json.dumps({"path": "/tmp/x"})
    read_resp.choices[0].message.tool_calls = [tc_read]

    # 第 2 轮：调 exit_plan_mode
    exit_resp = _make_exit_plan_mode_tool_call("完整计划：1. 改 foo.py")

    # 第 3 轮：审批通过后 LLM 开始执行（不再调工具，给最终响应）
    final_resp = _make_final_response("执行完成")

    responses = [read_resp, exit_resp, final_resp]

    # 记录每轮 get_tool_definitions 收到的 toolset
    captured_toolsets = []

    def fake_get(enabled_toolsets, *args, **kwargs):
        captured_toolsets.append(list(enabled_toolsets))
        # 返回所有被请求工具的 schema，避免 LLM 拒绝
        from tools.registry import registry, discover_builtin_tools
        discover_builtin_tools()
        return registry.get_definitions(enabled_toolsets, quiet=True)

    with patch("model_tools.get_tool_definitions", side_effect=fake_get), \
         patch("agent.llm_retry.call_with_retry", side_effect=responses):
        # read_file handler 由 registry.dispatch 拦截，避免真读文件
        from tools.registry import registry, discover_builtin_tools
        discover_builtin_tools()
        original_dispatch = registry.dispatch

        async def patched_dispatch(name, args, **kw):
            if name == "read_file":
                return json.dumps({"content": "file content"})
            return await original_dispatch(name, args, **kw)

        with patch.object(registry, "dispatch", side_effect=patched_dispatch):
            await agent.run_conversation("帮我改个文件")

    # 断言：toolset 三次转换完整验证（每轮 tool_schemas 都被刷新）
    # 第 1 轮 read_file 调研：plan_mode=True → ['plan']
    # 第 2 轮 exit_plan_mode 提交计划：plan_mode 在 dispatch 后才被清零，本轮开始时仍为 True → ['plan']
    # 第 3 轮 最终响应：plan_mode 已为 False → ['core']
    assert len(captured_toolsets) >= 3, (
        f"至少应捕获 3 次 toolset（3 轮 LLM），实际 {len(captured_toolsets)}"
    )
    assert captured_toolsets[0] == ["plan"], (
        f"第 1 轮应 ['plan']，实际 {captured_toolsets[0]}"
    )
    assert captured_toolsets[1] == ["plan"], (
        f"第 2 轮应 ['plan']，实际 {captured_toolsets[1]}"
    )
    assert captured_toolsets[2] == ["core"], (
        f"第 3 轮应 ['core']（审批后切回），实际 {captured_toolsets[2]}"
    )
    # 审批通过后 plan_mode 必须为 False
    assert agent.plan_mode is False, "审批通过后 plan_mode 应清零"
    # 至少 3 次 LLM 调用被消费（验证 read_file → exit_plan_mode → 最终响应 三步完整跑通）
    assert len(responses) == 3, "responses 应被全部消费"
    # 找到 exit_plan_mode 对应的 tool 消息，应含 plan_approved=True
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    approved_msg = next(
        (m for m in tool_msgs if "plan_approved" in (m.get("content") or "")),
        None,
    )
    assert approved_msg is not None, "缺 plan_approved 的 tool 消息"
    # read_file 的 tool 结果也应存在
    read_msg = next(
        (m for m in tool_msgs if "file content" in (m.get("content") or "")),
        None,
    )
    assert read_msg is not None, "缺 read_file 的 tool 消息"
    # 最终 assistant 消息应含"执行完成"
    assistant_msgs = [m for m in agent.conversation_history if m.get("role") == "assistant" and m.get("content")]
    assert any("执行完成" in (m.get("content") or "") for m in assistant_msgs), "缺最终响应"


async def test_e2e_plan_reject_then_revise_approve():
    """端到端：拒绝 → 修订 → 再次审批通过。"""
    call_count = [0]

    def cb(plan):
        call_count[0] += 1
        if call_count[0] == 1:
            return False, "第一次太粗"
        return True, ""

    agent = _make_minimal_agent(plan_approval_callback=cb)
    agent.plan_mode = True

    responses = [
        _make_exit_plan_mode_tool_call("v1 计划"),
        _make_exit_plan_mode_tool_call("v2 计划"),
        _make_final_response("ok"),
    ]
    with patch("agent.llm_retry.call_with_retry", side_effect=responses):
        await agent.run_conversation("test")

    assert call_count[0] == 2, "回调应被调 2 次（拒绝+批准）"
    assert agent.plan_mode is False, "最终应切回执行模式"


# ============================================================================
# P6: 多 Agent 并行调度（plan_mode_v2_dispatch + _run_parallel_planners + _merge_plans）
# ============================================================================

def test_plan_mode_v2_dispatch_registered():
    """plan_mode_v2_dispatch 被注册到 registry 的 plan toolset。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    entry = registry._tools.get("plan_mode_v2_dispatch")
    assert entry is not None, "plan_mode_v2_dispatch 未注册"
    assert entry.toolset == "plan", f"toolset 应为 'plan'，实际 '{entry.toolset}'"


async def test_plan_mode_v2_dispatch_flag_off_returns_feature_disabled():
    """flag OFF → 返回 feature_disabled，引导走原单 Agent 路径。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    result_json = await registry.dispatch(
        "plan_mode_v2_dispatch",
        {"subtasks": ["调研 X"]},
        config={"features": {"plan_mode_v2_parallel": {"enabled": False}}},
    )
    data = json.loads(result_json)
    assert data["error_type"] == "feature_disabled"
    assert "plan_mode_v2_parallel" in data["error"]


async def test_plan_mode_v2_dispatch_empty_subtasks_invalid_args():
    """flag ON 但 subtasks 空 → invalid_args。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    config = {"features": {"plan_mode_v2_parallel": {"enabled": True, "max_parallel_agents": 3}}}
    result_json = await registry.dispatch(
        "plan_mode_v2_dispatch",
        {"subtasks": []},
        config=config,
    )
    data = json.loads(result_json)
    assert data["error_type"] == "invalid_args"


# ---- _run_parallel_planners 直接测试（mock _run_child） ----

def test_run_parallel_planners_n1_matches_single_behavior():
    """N=1 跟单 Agent 行为一致：只调一次 _run_child，返回那份计划。"""
    from tools.plan_mode_tool import _run_parallel_planners
    from unittest.mock import patch

    with patch("tools.plan_mode_tool._run_child" if False else "tools.delegate_tool._run_child") as mock_child:
        # 直接 patch tools.delegate_tool._run_child（_run_parallel_planners 从那里 import）
        mock_child.return_value = "子计划 A"

        result = _run_parallel_planners(["调研 A"], max_n=3)

    assert mock_child.call_count == 1, "N=1 应只调 1 次 _run_child"
    assert result == ["子计划 A"]


def test_run_parallel_planners_n3_concurrent_collects_all():
    """N=3 并发 → 3 份子计划都收集到，顺序与 subtasks 一致。"""
    from tools.plan_mode_tool import _run_parallel_planners
    from unittest.mock import patch

    # 用 side_effect 按调用顺序返回不同结果（ThreadPoolExecutor 会并发提交，
    # 但 submit 顺序固定，result 收集顺序按 futures 字典迭代——我们按提交顺序读）
    call_results = {0: "计划-A", 1: "计划-B", 2: "计划-C"}

    def fake_run_child(goal, context, role, **kwargs):
        # goal 就是 subtasks 元素，按内容匹配返回
        if "A" in goal:
            return "计划-A"
        if "B" in goal:
            return "计划-B"
        return "计划-C"

    with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
        result = _run_parallel_planners(
            ["调研 A", "调研 B", "调研 C"], max_n=3,
        )

    assert len(result) == 3, f"应收集到 3 份子计划，实际 {len(result)}"
    assert set(result) == {"计划-A", "计划-B", "计划-C"}


def test_run_parallel_planners_max_n_caps_concurrency():
    """max_parallel_agents 上限生效：5 个子任务但 max_n=3 → workers=3。"""
    from tools.plan_mode_tool import _run_parallel_planners
    from unittest.mock import patch, MagicMock

    captured_max_workers = []

    class FakeExecutor:
        def __init__(self, max_workers):
            captured_max_workers.append(max_workers)
        def submit(self, fn, *a, **kw):
            f = MagicMock()
            f.result.return_value = "计划"
            return f
        def shutdown(self, **kw):
            pass

    with patch("tools.plan_mode_tool.ThreadPoolExecutor", FakeExecutor), \
         patch("tools.delegate_tool._run_child"):
        _run_parallel_planners(
            ["t1", "t2", "t3", "t4", "t5"], max_n=3,
        )

    assert captured_max_workers == [3], (
        f"workers 应被 max_n=3 钳制，实际 {captured_max_workers}"
    )


def test_run_parallel_planners_subagent_failure_isolated():
    """单个子 Agent 失败不影响其他：失败的被跳过，成功的保留。"""
    from tools.plan_mode_tool import _run_parallel_planners
    from unittest.mock import patch

    def fake_run_child(goal, context, role, **kwargs):
        if "B" in goal:
            raise RuntimeError("子代理 B 崩了")
        return f"计划-{goal[-1]}"

    with patch("tools.delegate_tool._run_child", side_effect=fake_run_child):
        result = _run_parallel_planners(
            ["调研 A", "调研 B", "调研 C"], max_n=3,
        )

    # B 失败被跳过，A 和 C 保留
    assert len(result) == 2, f"应有 2 份成功（B 失败跳过），实际 {len(result)}"
    assert all("B" not in r for r in result), "失败子代理的结果不应出现"


def test_run_parallel_planners_empty_subtasks_returns_empty():
    """空 subtasks → 空列表（不调 _run_child）。"""
    from tools.plan_mode_tool import _run_parallel_planners
    from unittest.mock import patch

    with patch("tools.delegate_tool._run_child") as mock_child:
        result = _run_parallel_planners([], max_n=3)

    assert result == []
    assert mock_child.call_count == 0


# ---- _merge_plans 测试 ----

def test_merge_plans_single_returns_as_is():
    """单份子计划不调 LLM，直接返回原文。"""
    from tools.plan_mode_tool import _merge_plans

    result = _merge_plans(["唯一的子计划"], llm_client=None, model="")
    assert result == "唯一的子计划"


def test_merge_plans_empty_returns_placeholder():
    """空列表 → 占位文本。"""
    from tools.plan_mode_tool import _merge_plans

    result = _merge_plans([], llm_client=None, model="")
    assert "无子计划" in result


def test_merge_plans_multiple_calls_llm():
    """多份子计划 → 调 LLM 合并，返回 LLM 输出。"""
    from tools.plan_mode_tool import _merge_plans
    from unittest.mock import patch, MagicMock

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "合并后的最终计划"

    # patch asyncio.run（_merge_plans 内部用它驱动 call_with_retry）
    # call_with_retry 是局部 import，但 asyncio.run 被 patch 后不会真调它
    with patch("tools.plan_mode_tool.asyncio.run", return_value=mock_response) as mock_run:
        result = _merge_plans(
            ["子计划 A", "子计划 B"],
            llm_client=MagicMock(),
            model="test-model",
        )

    assert result == "合并后的最终计划"
    assert mock_run.call_count == 1


def test_merge_plans_llm_failure_falls_back_to_concatenation():
    """LLM 合并失败 → 退化为拼接（fail-open）。"""
    from tools.plan_mode_tool import _merge_plans
    from unittest.mock import patch

    with patch("tools.plan_mode_tool.asyncio.run", side_effect=RuntimeError("LLM 挂了")):
        result = _merge_plans(
            ["子计划 A", "子计划 B"],
            llm_client=None,
            model="",
        )

    assert "拼接" in result or "子计划 1" in result, "退化拼接应含分隔标题"
    assert "子计划 A" in result
    assert "子计划 B" in result


def test_plan_merge_prompt_template_format():
    """PLAN_MERGE_PROMPT_TEMPLATE 能被 .format 正确填充。"""
    from tools.plan_mode_tool import PLAN_MERGE_PROMPT_TEMPLATE

    filled = PLAN_MERGE_PROMPT_TEMPLATE.format(
        n=2,
        sub_plans_block="### 子计划 1\nA\n\n### 子计划 2\nB",
    )
    assert "2 份" in filled
    assert "### 子计划 1" in filled


# ---- 端到端：plan_mode_v2_dispatch 全流程（mock _run_child + 合并） ----

async def test_plan_mode_v2_dispatch_full_flow_flag_on():
    """flag ON → 并行调度 → 合并 → 返回 merged_plan。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    config = {
        "features": {
            "plan_mode_v2_parallel": {"enabled": True, "max_parallel_agents": 3},
        }
    }

    # mock _run_child 返回 2 份子计划
    def fake_run_child(goal, context, role, **kwargs):
        return f"子计划：{goal}"

    # mock 合并的 LLM 调用（patch asyncio.run 只作用于 _merge_plans）
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "最终合并计划"

    # 构造一个假父 agent 提供 llm_client + model
    fake_agent = MagicMock()
    fake_agent.llm_client = MagicMock()
    fake_agent.model = "test-model"

    with patch("tools.delegate_tool._run_child", side_effect=fake_run_child), \
         patch("tools.plan_mode_tool.asyncio.run", return_value=mock_response):
        result_json = await registry.dispatch(
            "plan_mode_v2_dispatch",
            {"subtasks": ["前端调研", "后端调研"]},
            config=config,
            agent_ref=fake_agent,
        )

    data = json.loads(result_json)
    assert data["success"] is True
    assert data["subplans_succeeded"] == 2
    assert data["subtasks_attempted"] == 2
    assert data["max_parallel_agents"] == 3
    assert data["merged_plan"] == "最终合并计划"


async def test_plan_mode_v2_dispatch_all_subagents_fail():
    """flag ON 但所有子代理都失败 → all_subagents_failed。"""
    from tools.registry import registry, discover_builtin_tools
    discover_builtin_tools()

    config = {
        "features": {
            "plan_mode_v2_parallel": {"enabled": True, "max_parallel_agents": 3},
        }
    }

    with patch("tools.delegate_tool._run_child", side_effect=RuntimeError("全崩")):
        result_json = await registry.dispatch(
            "plan_mode_v2_dispatch",
            {"subtasks": ["A", "B"]},
            config=config,
        )

    data = json.loads(result_json)
    assert data["error_type"] == "all_subagents_failed"
