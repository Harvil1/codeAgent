"""R23 流式并发执行测试（#7）。"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from model_tools import ensure_tools_discovered
from tools.registry import registry

ensure_tools_discovered()

from agent.streaming_executor import (
    StreamingToolExecutor,
    _is_preset_safe,
    _tc_from_buf,
)


def _dtc(idx, id_, name, args_delta="", args_full=None):
    """构造 OpenAI 风格的 tool_call delta。"""
    return SimpleNamespace(
        index=idx,
        id=id_,
        function=SimpleNamespace(name=name, arguments=args_delta),
    )


# ---------------------------------------------------------------------------
# 单元：StreamingToolExecutor
# ---------------------------------------------------------------------------

class _FakeStreamClient:
    """按 delta 序列产出 chunk 的假流式 client。"""

    def __init__(self, deltas):
        self._deltas = deltas

    async def chat_completions_stream(self, messages, tools=None, **kw):
        for d in self._deltas:
            yield d


def _mk_agent(config=None, handled=None):
    from agent import AIAgent
    a = AIAgent.__new__(AIAgent)
    a.config = config or {}
    a.session_id = "t"
    a.memory_store = None
    a.session_store = None
    a.omnimate_home = None
    a.hooks_registry = None
    a.bg_manager = None
    a.team_bus = None
    a.team_coordinator = None
    a.team_name = None
    a.on_tool_call = None
    a._recent_read_files = []
    a._recent_skills = []
    a._tool_failure_streak = 0
    a._activated_conditional_skills = set()
    a._memory_touched_this_turn = False
    a._pending_skill_paths = []  # R26 #16：pre-callback 收集-批量执行新字段
    a._streaming_preset_results = {}
    a._stream_callback = None
    a.llm_client = None
    a.fallback_llm_client = None
    a.model = "m"
    a._max_tokens_escalator = None
    a._llm_usage_stats = {
        "total_calls": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
    }
    return a


def test_tc_from_buf_and_safe():
    tc = _tc_from_buf({"id": "c1", "name": "ls", "arguments": "{}"})
    assert tc.id == "c1" and tc.function.name == "ls"
    # read_file 是 safe
    assert _is_preset_safe(_tc_from_buf({"id": "x", "name": "read_file", "arguments": "{}"}))
    # write_file 是 unsafe（不预执行）
    assert not _is_preset_safe(_tc_from_buf({"id": "x", "name": "write_file", "arguments": "{}"}))
    # terminal 只读命令动态放宽
    ro = json.dumps({"command": "git status"})
    assert _is_preset_safe(_tc_from_buf({"id": "x", "name": "terminal", "arguments": ro}))
    wo = json.dumps({"command": "rm -rf build"})
    assert not _is_preset_safe(_tc_from_buf({"id": "x", "name": "terminal", "arguments": wo}))


@pytest.mark.asyncio
async def test_executor_preset_safe_skips_unsafe(monkeypatch):
    """safe call 预执行；unsafe call 不预执行；JSON 坏不预执行。"""
    a = _mk_agent()
    ex = StreamingToolExecutor(a)

    executed = []

    async def fake_handle(name, args, **kw):
        executed.append(name)
        return json.dumps({"ok": name})

    import agent.streaming_executor as se
    monkeypatch.setattr("model_tools.handle_function_call", fake_handle)

    # safe：read_file
    ex.complete(0, {"id": "c1", "name": "read_file", "arguments": '{"path": "a.py"}'})
    # unsafe：write_file（不预执行）
    ex.complete(1, {"id": "c2", "name": "write_file", "arguments": '{"path": "b.py"}'})
    # JSON 坏（截断的 arguments）
    ex.complete(2, {"id": "c3", "name": "read_file", "arguments": '{"path": "trunc'})
    await asyncio.sleep(0.05)  # 让预执行 task 跑
    results = await ex.collect()
    assert "c1" in results and "read_file" in results["c1"]
    assert "c2" not in results  # unsafe 未预执行
    assert "c3" not in results  # 坏 JSON 未预执行


@pytest.mark.asyncio
async def test_executor_drain_discards(monkeypatch):
    """drain：等 task 完成但丢弃结果（流异常路径防僵尸）。"""
    a = _mk_agent()
    ex = StreamingToolExecutor(a)

    async def fake_handle(name, args, **kw):
        await asyncio.sleep(0.02)
        return json.dumps({"ok": True})

    import agent.streaming_executor as se
    monkeypatch.setattr("model_tools.handle_function_call", fake_handle)
    ex.complete(0, {"id": "c1", "name": "read_file", "arguments": "{}"})
    await ex.drain()
    assert ex._results == {}
    assert ex._tasks == []


# ---------------------------------------------------------------------------
# 集成：_call_llm_streaming 预执行 + _dispatch_tool_calls 跳重
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_preset_end_to_end(monkeypatch):
    """开关开：流式期间两个 safe call 预执行，dispatch 跳过重复执行。"""
    # 注册假 safe 工具（记录执行次数）
    calls = {"n": 0}

    async def _h(args, **kw):
        calls["n"] += 1
        return json.dumps({"ok": calls["n"]})

    registry.register(
        name="_t_preset_read", toolset="core",
        schema={"name": "_t_preset_read", "parameters": {"type": "object", "properties": {}}},
        handler=_h, emoji="t", isConcurrencySafe=True,
    )
    try:
        a = _mk_agent(config={"agent": {"streaming_tool_execution": True}})
        # 假流：两个 tool_call（index 0 分两个 delta；index 1 一个）
        deltas = [
            {"content": "", "tool_calls": [_dtc(0, "c1", "_t_preset_read", args_delta='{"a"')], "finish_reason": None, "usage": None},
            {"content": "", "tool_calls": [_dtc(0, None, None, args_delta=': 1}')], "finish_reason": None, "usage": None},
            # index 切换 → c1 完整（预执行点）
            {"content": "", "tool_calls": [_dtc(1, "c2", "_t_preset_read", args_delta='{"a": 2}')], "finish_reason": None, "usage": None},
            {"content": "", "tool_calls": [], "finish_reason": "tool_calls", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ]
        a.llm_client = _FakeStreamClient(deltas)

        response = await a._call_llm_streaming(messages=[], tools=None)
        # 预执行结果已暂存（c1 在 index 切换时执行；c2 在流结束时执行）
        preset = a._streaming_preset_results
        assert "c1" in preset and "c2" in preset
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 2

        # dispatch 消费：预执行命中的跳过（执行次数不再增长）
        n_after_stream = calls["n"]
        a.conversation_history = []
        a._persist_session_message = lambda *ar, **kw: None
        a._maybe_handle_plan_approval = lambda tc, c: c
        a._update_failure_streak = lambda c: None
        a._idle_requested = False
        a._input_queue = None
        a.spawn_depth = 0
        a._tool_summary_task = None
        a._pending_tool_batch_summary = None

        assistant_msg = response.choices[0].message
        cont = await a._dispatch_tool_calls(
            assistant_msg, lambda name, args, **kw: (_ for _ in ()).throw(AssertionError("不应再执行")),
        )
        assert cont is True
        assert calls["n"] == n_after_stream  # 没有重复执行
        # 结果按顺序回填 history（两条 tool 消息）
        tool_msgs = [m for m in a.conversation_history if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    finally:
        registry.unregister("_t_preset_read")


@pytest.mark.asyncio
async def test_streaming_preset_disabled_by_default():
    """开关关（默认）：无预执行（结果空，行为同旧）。"""
    a = _mk_agent(config={})  # 默认关
    a.llm_client = _FakeStreamClient([
        {"content": "hi", "tool_calls": [], "finish_reason": "stop", "usage": None},
    ])
    await a._call_llm_streaming(messages=[], tools=None)
    assert a._streaming_preset_results == {}


@pytest.mark.asyncio
async def test_dispatch_preset_empty_noop():
    """无预执行结果：dispatch 行为不变（回归）。"""
    a = _mk_agent()
    a.conversation_history = []
    a._persist_session_message = lambda *ar, **kw: None
    a._idle_requested = False
    a._input_queue = None
    a.spawn_depth = 0
    a._tool_summary_task = None
    a._pending_tool_batch_summary = None

    async def _h2(args, **kw):
        return json.dumps({"ok": 2})

    registry.register(
        name="_t_preset2", toolset="core",
        schema={"name": "_t_preset2", "parameters": {"type": "object", "properties": {}}},
        handler=_h2, emoji="t", isConcurrencySafe=True,
    )
    try:
        msg = SimpleNamespace(
            content=None,
            tool_calls=[_tc_from_buf({"id": "z1", "name": "_t_preset2", "arguments": "{}"})],
            reasoning_content=None, thinking_signature=None,
        )
        cont = await a._dispatch_tool_calls(msg, lambda n, ar, **kw: json.dumps({"ok": 1}))
        assert cont is True
        tool_msgs = [m for m in a.conversation_history if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
    finally:
        registry.unregister("_t_preset2")
