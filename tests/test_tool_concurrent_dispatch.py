"""工具并发执行测试（Task F2）。

契约（plan 5 场景）：
1. 多个 safe 工具应并发执行（总耗时 ≈ 最慢的一个，不是相加）
2. unsafe 工具应串行（一个跑完才下一个）
3. 并发执行后结果按原 tool_call 顺序回填（保证 tool_call.id 配对）
4. 一个 safe 工具失败不影响其他 safe 工具（return_exceptions=True）
5. safe 组先并发跑完，再串行跑 unsafe 组

实现要点：
- 通过真实 registry.register 注册 fake 工具（标 isConcurrencySafe）
- 用 asyncio.sleep 制造可观测的时间差
- 用 monkeypatch 给 agent 注入 _record_recent / on_tool_call 等 noop
"""
import asyncio
import json
import time

import pytest

from agent import AIAgent
from tools.registry import registry


# ---------- 辅助构造（参考 test_dispatch_tool_calls_async.py）----------

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
    return _FakeMsg(tool_calls_spec)


def _make_minimal_agent():
    """构造最小可用 AIAgent（不连真 LLM）。"""
    agent = AIAgent.__new__(AIAgent)
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


def _register_fake_tool(name, is_safe):
    """在真实 registry 注册一个 fake 工具条目（仅元数据，handler 不被测）。"""
    registry.register(
        name=name,
        toolset="test",
        schema={"name": name, "description": "fake", "parameters": {}},
        handler=lambda args, **kw: json.dumps({"ok": True}, ensure_ascii=False),
        isConcurrencySafe=is_safe,
    )


def _cleanup_tools(names):
    """测试后从 registry 移除 fake 工具，避免污染其他测试。"""
    with registry._lock:
        for n in names:
            registry._tools.pop(n, None)


# ---------- 5 个 plan 场景 ----------

async def test_safe_tools_run_concurrently():
    """场景 1：多个 safe 工具应并发执行。

    3 个 safe 工具各 sleep 0.3s，并发跑应该 ≈ 0.3s（< 0.7s），
    串行会 ≈ 0.9s。
    """
    tool_names = ["f2_safe_a", "f2_safe_b", "f2_safe_c"]
    for n in tool_names:
        _register_fake_tool(n, is_safe=True)
    try:
        agent = _make_minimal_agent()
        assistant_msg = _make_assistant_msg([
            {"id": "call_1", "name": "f2_safe_a", "arguments": '{}'},
            {"id": "call_2", "name": "f2_safe_b", "arguments": '{}'},
            {"id": "call_3", "name": "f2_safe_c", "arguments": '{}'},
        ])

        async def fake_handle(tool_name, args, **kwargs):
            await asyncio.sleep(0.3)
            return json.dumps({"tool": tool_name}, ensure_ascii=False)

        start = time.monotonic()
        await agent._dispatch_tool_calls(assistant_msg, fake_handle)
        elapsed = time.monotonic() - start

        # 并发：3×0.3=0.9 应压到 ≈0.3。留宽裕，<0.7 表示确实并发了
        assert elapsed < 0.7, (
            f"safe 工具应并发（3×0.3s≈0.3s 实际 {elapsed:.2f}s），"
            "若 ≥0.7s 说明仍在串行"
        )
    finally:
        _cleanup_tools(tool_names)


async def test_unsafe_tools_run_sequentially():
    """场景 2：unsafe 工具应串行（一个跑完才下一个）。

    3 个 unsafe 工具各 sleep 0.2s，串行跑应 ≈ 0.6s（≥ 0.55s）。
    """
    tool_names = ["f2_unsafe_a", "f2_unsafe_b", "f2_unsafe_c"]
    for n in tool_names:
        _register_fake_tool(n, is_safe=False)
    try:
        agent = _make_minimal_agent()
        assistant_msg = _make_assistant_msg([
            {"id": "call_1", "name": "f2_unsafe_a", "arguments": '{}'},
            {"id": "call_2", "name": "f2_unsafe_b", "arguments": '{}'},
            {"id": "call_3", "name": "f2_unsafe_c", "arguments": '{}'},
        ])

        async def fake_handle(tool_name, args, **kwargs):
            await asyncio.sleep(0.2)
            return json.dumps({"tool": tool_name}, ensure_ascii=False)

        start = time.monotonic()
        await agent._dispatch_tool_calls(assistant_msg, fake_handle)
        elapsed = time.monotonic() - start

        # 串行：3×0.2=0.6s，应 ≥0.55（留点调度余量）
        assert elapsed >= 0.55, (
            f"unsafe 工具应串行（3×0.2s≈0.6s 实际 {elapsed:.2f}s），"
            "若 <0.55s 说明误并发了"
        )
    finally:
        _cleanup_tools(tool_names)


async def test_results_merged_in_tool_call_id_order():
    """场景 3：并发执行后结果按原 tool_call 顺序回填。

    构造 tool_call 顺序 [A, B, C]，但 fake_handle 让 B 先返回、A 最后返回。
    回填到 conversation_history 的 tool 消息必须按 [A, B, C] 顺序，
    否则 LLM API 会报 400（tool_call_id 与 tool_result 不配对）。
    """
    tool_names = ["f2_order_a", "f2_order_b", "f2_order_c"]
    for n in tool_names:
        _register_fake_tool(n, is_safe=True)
    try:
        agent = _make_minimal_agent()
        assistant_msg = _make_assistant_msg([
            {"id": "tc_A", "name": "f2_order_a", "arguments": '{}'},
            {"id": "tc_B", "name": "f2_order_b", "arguments": '{}'},
            {"id": "tc_C", "name": "f2_order_c", "arguments": '{}'},
        ])

        async def fake_handle(tool_name, args, **kwargs):
            # 故意错乱完成顺序：C 最快、A 最慢
            if tool_name == "f2_order_a":
                await asyncio.sleep(0.3)
            elif tool_name == "f2_order_b":
                await asyncio.sleep(0.15)
            else:
                await asyncio.sleep(0.05)
            return json.dumps({"who": tool_name}, ensure_ascii=False)

        await agent._dispatch_tool_calls(assistant_msg, fake_handle)

        tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
        assert len(tool_msgs) == 3, f"应有 3 个 tool 消息，实际 {len(tool_msgs)}"

        # tool_call_id 必须按原顺序：tc_A, tc_B, tc_C
        ids = [m["tool_call_id"] for m in tool_msgs]
        assert ids == ["tc_A", "tc_B", "tc_C"], (
            f"tool_result 必须按原 tool_call 顺序回填，实际 {ids}"
        )
        # 内容也必须各自配对（A → f2_order_a，不是别的）
        names = [json.loads(m["content"])["who"] for m in tool_msgs]
        assert names == ["f2_order_a", "f2_order_b", "f2_order_c"], (
            f"tool_result 内容必须与 tool_call.id 严格配对，实际 {names}"
        )
    finally:
        _cleanup_tools(tool_names)


async def test_failure_in_one_safe_tool_doesnt_block_others():
    """场景 4：一个 safe 工具失败不影响其他 safe 工具。

    3 个 safe 工具，中间那个抛异常，其他两个应正常返回。
    失败的工具结果应转成 JSON error（error_type=concurrent_dispatch_error）。
    """
    tool_names = ["f2_fail_a", "f2_fail_b", "f2_fail_c"]
    for n in tool_names:
        _register_fake_tool(n, is_safe=True)
    try:
        agent = _make_minimal_agent()
        assistant_msg = _make_assistant_msg([
            {"id": "tc_a", "name": "f2_fail_a", "arguments": '{}'},
            {"id": "tc_b", "name": "f2_fail_b", "arguments": '{}'},
            {"id": "tc_c", "name": "f2_fail_c", "arguments": '{}'},
        ])

        async def fake_handle(tool_name, args, **kwargs):
            if tool_name == "f2_fail_b":
                raise RuntimeError("boom from B")
            return json.dumps({"ok": tool_name}, ensure_ascii=False)

        await agent._dispatch_tool_calls(assistant_msg, fake_handle)

        tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
        assert len(tool_msgs) == 3, (
            f"B 失败不应让其他工具消失，应仍有 3 个 tool 消息，实际 {len(tool_msgs)}"
        )

        # A 和 C 应正常
        a_data = json.loads(tool_msgs[0]["content"])
        c_data = json.loads(tool_msgs[2]["content"])
        assert a_data.get("ok") == "f2_fail_a", f"A 应正常，实际 {a_data}"
        assert c_data.get("ok") == "f2_fail_c", f"C 应正常，实际 {c_data}"

        # B 应是 JSON error
        b_data = json.loads(tool_msgs[1]["content"])
        assert "error" in b_data, f"B 失败应转成 JSON error，实际 {b_data}"
        assert "boom" in b_data["error"], f"B error 应含 boom，实际 {b_data}"
    finally:
        _cleanup_tools(tool_names)


async def test_safe_and_unsafe_groups_executed_in_order():
    """场景 5：safe 组先并发跑完，再串行跑 unsafe 组。

    构造 [safe, unsafe, safe, unsafe] 顺序的 tool_calls。
    验证：
    - safe 组（call_1, call_3）并发先跑完
    - unsafe 组（call_2, call_4）串行后跑
    - 最终 tool_result 按原 tool_call 顺序 [1,2,3,4] 配对
    """
    tool_names = ["f2_mix_s1", "f2_mix_u1", "f2_mix_s2", "f2_mix_u2"]
    _register_fake_tool("f2_mix_s1", is_safe=True)
    _register_fake_tool("f2_mix_u1", is_safe=False)
    _register_fake_tool("f2_mix_s2", is_safe=True)
    _register_fake_tool("f2_mix_u2", is_safe=False)
    try:
        agent = _make_minimal_agent()
        assistant_msg = _make_assistant_msg([
            {"id": "tc_1", "name": "f2_mix_s1", "arguments": '{}'},
            {"id": "tc_2", "name": "f2_mix_u1", "arguments": '{}'},
            {"id": "tc_3", "name": "f2_mix_s2", "arguments": '{}'},
            {"id": "tc_4", "name": "f2_mix_u2", "arguments": '{}'},
        ])

        # 记录每个工具的开始/结束时间戳
        events = []

        async def fake_handle(tool_name, args, **kwargs):
            t0 = time.monotonic()
            events.append(("start", tool_name, t0))
            await asyncio.sleep(0.1)
            t1 = time.monotonic()
            events.append(("end", tool_name, t1))
            return json.dumps({"who": tool_name}, ensure_ascii=False)

        await agent._dispatch_tool_calls(assistant_msg, fake_handle)

        tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
        assert len(tool_msgs) == 4, f"应有 4 个 tool 消息，实际 {len(tool_msgs)}"

        # 顺序配对（核心：tool_call.id 严格对齐）
        ids = [m["tool_call_id"] for m in tool_msgs]
        assert ids == ["tc_1", "tc_2", "tc_3", "tc_4"], (
            f"tool_result 必须按原 tool_call 顺序配对，实际 {ids}"
        )

        # 验证 safe 组（s1, s2）的开始时间接近（并发），
        # unsafe 组（u1, u2）的开始时间错开（串行）
        starts = {name: t for ev, name, t in events if ev == "start"}
        # safe 组两个应几乎同时开始（差距 < 0.05s）
        s_gap = abs(starts["f2_mix_s1"] - starts["f2_mix_s2"])
        assert s_gap < 0.05, (
            f"safe 组应并发同时开始，实际 s1/s2 开始时间差 {s_gap:.3f}s"
        )
        # unsafe 组两个应串行，u2 必须在 u1 结束后才开始（差距 > 0.05s）
        ends = {name: t for ev, name, t in events if ev == "end"}
        u_gap = starts["f2_mix_u2"] - ends["f2_mix_u1"]
        assert u_gap >= -0.01, (
            f"unsafe 组应串行，u2 不应在 u1 结束前开始，"
            f"u2_start - u1_end = {u_gap:.3f}s"
        )

        # unsafe 必须在 safe 全部结束后才开始（safe 组先跑完）
        safe_end_max = max(ends["f2_mix_s1"], ends["f2_mix_s2"])
        unsafe_start_min = min(starts["f2_mix_u1"], starts["f2_mix_u2"])
        assert unsafe_start_min >= safe_end_max - 0.05, (
            f"unsafe 组应在 safe 组全部完成后才开始，"
            f"safe_end_max={safe_end_max:.3f} unsafe_start_min={unsafe_start_min:.3f}"
        )
    finally:
        _cleanup_tools(tool_names)
