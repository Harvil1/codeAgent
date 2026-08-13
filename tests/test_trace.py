"""TraceSink 本地 trace sink 测试。

CCAR8 Task 4。验证：
- 每天 jsonl 文件
- fail-open 写盘
- query / summary 聚合
- agent_id 覆盖
"""
import json
import logging
from pathlib import Path

from agent.trace import TraceSink


def test_emit_writes_jsonl(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100, model="deepseek-chat")
    files = list((tmp_path / ".trace").glob("*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["event"] == "pre_llm_call"
    assert record["input_tokens"] == 100
    assert record["model"] == "deepseek-chat"
    assert record["agent_id"] == "main"
    assert "ts" in record


def test_emit_creates_daily_file(tmp_path: Path):
    """每天一个 jsonl 文件。"""
    import datetime
    sink = TraceSink(tmp_path)
    sink.emit("event_a")
    sink.emit("event_b")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    assert (tmp_path / ".trace" / f"{date_str}.jsonl").exists()
    lines = (tmp_path / ".trace" / f"{date_str}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_emit_with_agent_id_override(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("evt", agent_id="subagent_xyz")
    record = json.loads(
        list((tmp_path / ".trace").glob("*.jsonl"))[0].read_text(encoding="utf-8")
    )
    assert record["agent_id"] == "subagent_xyz"


def test_emit_failopen_on_disk_error(tmp_path: Path, caplog):
    """写盘失败不抛，只 log warning。"""
    sink = TraceSink(tmp_path)
    # 模拟写盘失败：把 _trace_dir 改成不可写的路径
    sink._trace_dir = "/nonexistent/path/that/does/not/exist"
    with caplog.at_level(logging.WARNING):
        sink.emit("event_x")  # 不抛
    assert any("trace emit fail-open" in r.message for r in caplog.records)


def test_query_filters(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100)
    sink.emit("post_tool_use", tool="read_file")
    sink.emit("post_tool_use", tool="write_file")
    results = sink.query(event="post_tool_use")
    assert len(results) == 2
    results = sink.query(event="pre_llm_call")
    assert len(results) == 1


def test_summary_aggregates(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100)
    sink.emit("pre_llm_call", input_tokens=200)
    sink.emit("post_llm_call", output_tokens=50)
    sink.emit("tool_failed", tool="x", error="boom")
    summary = sink.summary()
    assert summary["total_events"] == 4
    assert summary["by_event"]["pre_llm_call"] == 2
    assert summary["by_event"]["post_llm_call"] == 1
    assert summary["by_event"]["tool_failed"] == 1
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 50
    assert summary["error_count"] == 1


def test_query_limit(tmp_path: Path):
    sink = TraceSink(tmp_path)
    for i in range(10):
        sink.emit("event_x", idx=i)
    results = sink.query(limit=3)
    assert len(results) == 3


# =============================================================================
# CCAR8 Task 5: trace hook 接入测试（防 silent-dead-code）
# =============================================================================


def test_register_trace_hooks_registers_all_six(tmp_path: Path):
    """_register_trace_hooks 应该把 sink 接到 6 个 hook 点。

    直接测辅助函数（不构造 AIAgent），验证 hook 真注册。
    这是 silent-dead-code 防线——单元测试过 ≠ 生产路径生效。
    """
    from agent.hooks import HookEvent, HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    # 收集所有注册的 trace_* hook 名字
    trace_hook_names = {
        h.name
        for event_hooks in hooks._hooks.values()
        for h in event_hooks
        if h.name and h.name.startswith("trace_")
    }
    expected = {
        "trace_pre_llm_call",
        "trace_post_llm_call",
        "trace_post_tool_use",
        "trace_post_tool_use_failure",
        "trace_subagent_start",
        "trace_subagent_stop",
    }
    assert expected <= trace_hook_names, (
        f"缺失 hook: {expected - trace_hook_names}"
    )


def test_register_trace_hooks_in_correct_events(tmp_path: Path):
    """6 个 hook 注册到对应的 HookEvent，不是全堆在一个 event 上。"""
    from agent.hooks import HookEvent, HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    def _names(event: HookEvent) -> set:
        return {h.name for h in hooks._hooks[event]}

    assert "trace_pre_llm_call" in _names(HookEvent.PRE_LLM_CALL)
    assert "trace_post_llm_call" in _names(HookEvent.POST_LLM_CALL)
    assert "trace_post_tool_use" in _names(HookEvent.POST_TOOL_USE)
    assert "trace_post_tool_use_failure" in _names(HookEvent.POST_TOOL_USE_FAILURE)
    assert "trace_subagent_start" in _names(HookEvent.SUBAGENT_START)
    assert "trace_subagent_stop" in _names(HookEvent.SUBAGENT_STOP)


def test_pre_llm_call_hook_signature_messages_tools(tmp_path: Path):
    """PRE_LLM_CALL hook 签名是 (messages, tools) — 不能搞错。"""
    from agent.hooks import HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    # 触发：调 run_pre_llm_call 应该不抛（签名匹配）
    messages = [{"role": "user", "content": "hello world"}]
    tools = [{"type": "function", "name": "dummy"}]
    # run_pre_llm_call 链式返回 (messages, tools)
    out_messages, out_tools = hooks.run_pre_llm_call(messages, tools, session_id="s1")
    assert out_messages == messages
    assert out_tools == tools

    # 验证 trace 记录写入
    records = sink.query(event="pre_llm_call")
    assert len(records) == 1
    # input_tokens 粗估：11 字符 // 4 = 2
    assert records[0]["input_tokens"] == len("hello world") // 4


def test_post_llm_call_hook_extracts_tokens_and_model(tmp_path: Path):
    """POST_LLM_CALL hook 从 response 提取 output_tokens 和 model。"""
    from types import SimpleNamespace

    from agent.hooks import HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    # 模拟 OpenAI 响应对象
    fake_response = SimpleNamespace(
        model="deepseek-chat",
        usage=SimpleNamespace(completion_tokens=42),
    )
    hooks.run_post_llm_call(fake_response, session_id="s1")

    records = sink.query(event="post_llm_call")
    assert len(records) == 1
    assert records[0]["output_tokens"] == 42
    assert records[0]["model"] == "deepseek-chat"


def test_post_tool_use_hook_emits_tool_name(tmp_path: Path):
    """POST_TOOL_USE hook 接收 (tool_name, args, result)，emit tool 字段。"""
    from agent.hooks import HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    hooks.run_post_tool_use("read_file", {"path": "/tmp"}, "file content", session_id="s1")

    records = sink.query(event="post_tool_use")
    assert len(records) == 1
    assert records[0]["tool"] == "read_file"


def test_post_tool_use_failure_hook_payload_signature(tmp_path: Path):
    """POST_TOOL_USE_FAILURE hook 接收 payload dict（不是 3 参数）。

    关键防回归：brief 原文写 3 参数是错的，hook 实际签名是 fn(payload: dict)。
    """
    from agent.hooks import HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    # payload 形态（来自 run_post_tool_use_failure 调用约定）
    payload = {"tool": "terminal", "error": "boom", "error_type": "RuntimeError"}
    hooks.run_post_tool_use_failure(payload)

    records = sink.query(event="tool_failed")
    assert len(records) == 1
    assert records[0]["tool"] == "terminal"
    assert "boom" in records[0]["error"]


def test_subagent_hooks_emit_type_and_session(tmp_path: Path):
    """SUBAGENT_START/STOP hook 接收 payload dict。"""
    from agent.hooks import HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)

    payload = {"subagent_type": "explore", "session_id": "sub-123"}
    hooks.run_subagent_start(payload)
    hooks.run_subagent_stop(payload)

    starts = sink.query(event="subagent_start")
    stops = sink.query(event="subagent_stop")
    assert len(starts) == 1
    assert len(stops) == 1
    assert starts[0]["subagent_type"] == "explore"
    assert starts[0]["session_id"] == "sub-123"
    assert stops[0]["subagent_type"] == "explore"


def test_register_trace_hooks_idempotent_multiple_calls(tmp_path: Path):
    """多次调用 _register_trace_hooks 应该累加（不幂等，调用方负责只调一次）。

    AIAgent __init__ 只调一次，这里验证重复调用会叠加（fail-open 不去重）。
    """
    from agent.hooks import HookEvent, HookRegistry
    from agent.trace import _register_trace_hooks

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    _register_trace_hooks(hooks, sink)
    _register_trace_hooks(hooks, sink)

    # 重复注册后 PRE_LLM_CALL 有 2 个 trace_* hook
    pre_llm_hooks = [h for h in hooks._hooks[HookEvent.PRE_LLM_CALL] if h.name and h.name.startswith("trace_")]
    assert len(pre_llm_hooks) == 2


def test_aiagent_constructor_registers_trace_hooks(tmp_path: Path):
    """端到端：AIAgent 构造时真注册 6 个 trace hook 到 hooks_registry。

    这是 Silent-Dead-Code 防线核心测试（CLAUDE.md 教训）：
    单元测试 _register_trace_hooks 过 ≠ AIAgent __init__ 真调它。
    """
    from agent import AIAgent
    from agent.hooks import HookEvent, HookRegistry

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        hooks_registry=hooks,
        trace_sink=sink,
    )
    assert agent._trace_sink is sink

    # 验证 6 个 hook 都注册了
    expected_hooks = {
        HookEvent.PRE_LLM_CALL: "trace_pre_llm_call",
        HookEvent.POST_LLM_CALL: "trace_post_llm_call",
        HookEvent.POST_TOOL_USE: "trace_post_tool_use",
        HookEvent.POST_TOOL_USE_FAILURE: "trace_post_tool_use_failure",
        HookEvent.SUBAGENT_START: "trace_subagent_start",
        HookEvent.SUBAGENT_STOP: "trace_subagent_stop",
    }
    for event, expected_name in expected_hooks.items():
        registered_names = {h.name for h in hooks._hooks[event]}
        assert expected_name in registered_names, (
            f"{event.value} 缺 hook {expected_name}——silent-dead-code 警报"
        )


def test_aiagent_constructor_no_trace_sink_is_safe(tmp_path: Path):
    """trace_sink=None（默认）时构造不抛、不注册 trace hook。"""
    from agent import AIAgent
    from agent.hooks import HookEvent, HookRegistry

    hooks = HookRegistry()
    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        hooks_registry=hooks,
        # trace_sink 不传
    )
    assert agent._trace_sink is None
    # 没有 trace_* hook
    trace_count = sum(
        1 for event_hooks in hooks._hooks.values()
        for h in event_hooks
        if h.name and h.name.startswith("trace_")
    )
    assert trace_count == 0


def test_aiagent_trace_sink_emits_on_hook_fire(tmp_path: Path):
    """完整链路：AIAgent 构造 → 手动触发 hook → trace 真落盘。

    这是最强的"生产路径生效"验证（防 silent-dead-code）。
    """
    from agent import AIAgent
    from agent.hooks import HookRegistry

    hooks = HookRegistry()
    sink = TraceSink(tmp_path)
    AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        hooks_registry=hooks,
        trace_sink=sink,
    )

    # 触发各种 hook（模拟 agent 运行时事件）
    hooks.run_pre_llm_call([{"role": "user", "content": "hello"}], None, session_id="s")
    hooks.run_post_tool_use("read_file", {"path": "/x"}, "data", session_id="s")

    # 验证 trace 落盘
    pre = sink.query(event="pre_llm_call")
    tool = sink.query(event="post_tool_use")
    assert len(pre) == 1
    assert pre[0]["input_tokens"] == len("hello") // 4  # 粗估
    assert len(tool) == 1
    assert tool[0]["tool"] == "read_file"
