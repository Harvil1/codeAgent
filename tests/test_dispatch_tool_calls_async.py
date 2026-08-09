"""_dispatch_tool_calls async 契约测试。

Task D3：把 _dispatch_tool_calls 从 sync 改 async。
契约：
- 必须是 coroutine function
- 内部必须 await handle_function_call（不是同步调用）
- 其他逻辑（plan_approval / 错误统计 / idle 检查 / 消息回填）不变

注意：本 task 不引入并发——仍是 for tc in tool_calls: 串行 await。
"""
import inspect
import json

import pytest

from agent import AIAgent


def test_dispatch_tool_calls_is_coroutine():
    """_dispatch_tool_calls 必须是 async def（Task D3 改造）。"""
    assert inspect.iscoroutinefunction(AIAgent._dispatch_tool_calls), \
        "_dispatch_tool_calls 必须是 async def（Task D3 改造）"


async def test_dispatch_awaits_handle_function_call():
    """handle_function_call 必须被 await（不是直接同步调用）。

    构造一个 fake handle_function_call：若是被 await 则正常返回，
    若被同步调用（coroutine 未 await）则不会执行其内部逻辑。
    用 spy 标志区分 "被 await" vs "只创建 coroutine"。
    """
    agent = _make_minimal_agent()
    assistant_msg = _make_assistant_msg([
        {"id": "call_1", "name": "echo", "arguments": '{"msg": "hi"}'},
    ])

    awaited = {"count": 0}

    async def fake_handle(tool_name, args, **kwargs):
        awaited["count"] += 1
        return json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)

    result = await agent._dispatch_tool_calls(assistant_msg, fake_handle)

    # _dispatch_tool_calls 返回 True（继续主循环）
    assert result is True
    # handle_function_call 被真实 await 了一次
    assert awaited["count"] == 1, \
        f"handle_function_call 应被 await 1 次，实际 {awaited['count']}（可能同步调用未 await）"


async def test_dispatch_serial_order_multiple_tool_calls():
    """多个 tool_call 串行 await，顺序保持（不引入并发）。"""
    agent = _make_minimal_agent()
    assistant_msg = _make_assistant_msg([
        {"id": "call_1", "name": "first", "arguments": '{}'},
        {"id": "call_2", "name": "second", "arguments": '{}'},
        {"id": "call_3", "name": "third", "arguments": '{}'},
    ])

    order = []

    async def fake_handle(tool_name, args, **kwargs):
        order.append(tool_name)
        return json.dumps({"ok": True}, ensure_ascii=False)

    result = await agent._dispatch_tool_calls(assistant_msg, fake_handle)

    assert result is True
    assert order == ["first", "second", "third"], \
        f"串行顺序必须保持，实际 {order}"
    # history 应有 3 个 tool 结果 + 1 个 assistant
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    assert len(tool_msgs) == 3


async def test_dispatch_plan_approval_branch_preserved():
    """plan_approval 分支（error_type=plan_approval_required）逻辑保留。"""
    agent = _make_minimal_agent()
    assistant_msg = _make_assistant_msg([
        {"id": "call_1", "name": "exit_plan_mode", "arguments": '{}'},
    ])

    async def fake_handle(tool_name, args, **kwargs):
        return json.dumps(
            {"error_type": "plan_approval_required", "plan": "step 1\nstep 2"},
            ensure_ascii=False,
        )

    # 默认无 callback → 自动 approved
    result = await agent._dispatch_tool_calls(assistant_msg, fake_handle)
    assert result is True
    # plan_handled 后写入 tool 消息应含 plan_approved
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    parsed = json.loads(tool_msgs[0]["content"])
    assert parsed.get("plan_approved") is True
    assert agent.plan_mode is False


async def test_dispatch_failure_streak_logic_preserved():
    """失败统计逻辑保留（工具返回 error → _tool_failure_streak 自增）。"""
    agent = _make_minimal_agent()
    assert agent._tool_failure_streak == 0

    assistant_msg = _make_assistant_msg([
        {"id": "call_1", "name": "fail_tool", "arguments": '{}'},
    ])

    async def fake_handle(tool_name, args, **kwargs):
        return json.dumps({"error": "boom"}, ensure_ascii=False)

    await agent._dispatch_tool_calls(assistant_msg, fake_handle)
    assert agent._tool_failure_streak == 1, "失败工具应让 streak 自增"

    # 成功调用应清零
    assistant_msg2 = _make_assistant_msg([
        {"id": "call_2", "name": "ok_tool", "arguments": '{}'},
    ])

    async def fake_handle_ok(tool_name, args, **kwargs):
        return json.dumps({"ok": True}, ensure_ascii=False)

    await agent._dispatch_tool_calls(assistant_msg2, fake_handle_ok)
    assert agent._tool_failure_streak == 0, "成功工具应清零 streak"


async def test_dispatch_idle_request_returns_false():
    """idle 标志检查保留：_idle_requested=True 时返回 False。"""
    agent = _make_minimal_agent()
    agent._idle_requested = True

    assistant_msg = _make_assistant_msg([
        {"id": "call_1", "name": "x", "arguments": '{}'},
    ])

    async def fake_handle(tool_name, args, **kwargs):
        return json.dumps({"ok": True}, ensure_ascii=False)

    result = await agent._dispatch_tool_calls(assistant_msg, fake_handle)
    assert result is False, "idle_requested 时 _dispatch_tool_calls 应返回 False"


# ---------- 辅助构造函数 ----------

class _FakeChoice:
    def __init__(self, tool_calls):
        self.message = _FakeMsg(tool_calls)


class _FakeMsg:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = [_FakeTC(tc) for tc in tool_calls] if tool_calls else None


class _FakeTC:
    def __init__(self, spec):
        self.id = spec["id"]
        self.function = _FakeFunc(spec["name"], spec.get("arguments", "{}"))


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def _make_assistant_msg(tool_calls_spec):
    """构造一个 assistant_msg（仿 OpenAI ChatCompletion.message）。"""
    msg = _FakeMsg(tool_calls_spec)
    return msg


def _make_minimal_agent():
    """构造一个最小可用的 AIAgent（不连真 LLM）。"""
    from unittest.mock import MagicMock
    agent = AIAgent.__new__(AIAgent)
    # 必要字段
    agent.conversation_history = []
    agent.session_id = "test-session"
    agent.omnimate_home = "/tmp"
    agent.config = {}
    agent.memory_store = None
    agent.session_store = None
    agent.hooks_registry = None
    agent.bg_manager = None
    agent.team_bus = None
    agent.team_coordinator = None
    agent.team_name = None
    agent.plan_mode = False
    agent.plan_approval_callback = None
    agent.on_tool_call = None
    agent._tool_failure_streak = 0
    agent._last_tool_error = ""
    agent._idle_requested = False
    agent._recent_files = []
    agent._recent_skills = []
    agent._persist_session_message = lambda *a, **kw: None
    agent._record_recent = lambda *a, **kw: None
    return agent
