"""prompt cache 检测系统测试（改造点 ③）。

测试分三层：
1. 单元测试（cache_monitor.py 模块本身）
2. 端到端测试（通过 AIAgent 真实主循环触发 hook）
3. fail-open + compact 触发 notify_compaction 端到端
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# 单元测试
# ============================================================================

def test_record_prompt_state_basic():
    """record_prompt_state 基本能用。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    state = cache_monitor.record_prompt_state(
        system_prompt="hello",
        tools=[{"function": {"name": "foo"}}],
        model="deepseek-chat",
    )
    assert state.system_hash != 0
    assert state.tools_hash != 0
    assert state.model == "deepseek-chat"


def test_check_cache_break_first_call_returns_none():
    """首次调用（无 baseline）返回 None，设 baseline。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    state = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=state, cache_read_tokens=10000,
    )
    assert result is None


def test_check_cache_break_system_changed_detected():
    """system 变了 + cache 大跌 → 返回 'system prompt 变了'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="B", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,  # 降 9000，> 5% 且 > 2000
    )
    assert result is not None
    assert "system prompt" in result


def test_check_cache_break_small_drop_not_break():
    """cache_read 从 10000 → 9600（4% < 5% 阈值），返回 None。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=9600,  # 4% < 5%
    )
    assert result is None


def test_check_cache_break_drop_below_2000_not_break():
    """cache_read 从 10000 → 9000（10% 下降但 drop=1000 < 2000 tokens）—— 注意这条数据 10000→9000 drop=1000 不满足条件，
    改成 1000→0 让 drop 仍然 < 2000（边界条件测试）。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=1000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    # drop=500 (< 2000 tokens)——虽然 50% 下降，但 drop 量小不算 break
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=500,
    )
    assert result is None


def test_check_cache_break_small_drop_under_2000_tokens():
    """cache_read 从 10000 → 9000（10% 下降 > 5%，但 drop=1000 < 2000），返回 None。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=9000,  # drop=1000 < 2000
    )
    assert result is None


def test_notify_compaction_skips_break():
    """调 notify_compaction 后，cache 大降也不算 break。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    cache_monitor.notify_compaction()

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=100,  # 大降
    )
    assert result is None


def test_reset_cache_monitor_clears_state():
    """reset_cache_monitor 后 _last_state None、_break_history 空。"""
    from agent import cache_monitor
    # 先搞点状态
    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )
    s2 = cache_monitor.record_prompt_state(
        system_prompt="B", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=100,
    )
    assert len(cache_monitor._break_history) > 0

    cache_monitor.reset_cache_monitor()
    assert cache_monitor._last_state is None
    assert cache_monitor._last_cache_read is None
    assert cache_monitor._break_history == []
    assert cache_monitor._pending_compaction is False


def test_break_history_capped_at_100():
    """构造 > 100 次 break，验证 _break_history 不超过 100。

    每次 break 后重新建立高 baseline，让下一轮真的触发 break
    （真实场景：每次 break 后下次调用 cache 恢复到高水位，然后再次 break）。
    """
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    # 先种 baseline
    s_prev = cache_monitor.record_prompt_state(
        system_prompt="base", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s_prev, cache_read_tokens=10000,
    )

    # 制造 110 次 break：每次 system 变 → cache 从 10000 跌到 100 → break
    # 但 break 后 _last_cache_read 变成 100，所以下一轮需要先"恢复"到 10000
    # 才能再次触发 > 5% + > 2000 的下降
    for i in range(110):
        # 恢复 baseline 到高水位（同一 system 不同次 OK，不算 break 因为是涨不是跌）
        s_recover = cache_monitor.record_prompt_state(
            system_prompt=f"recover_{i}", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_recover, cache_read_tokens=10000,
        )
        # 触发 break
        s_break = cache_monitor.record_prompt_state(
            system_prompt=f"break_{i}", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_break, cache_read_tokens=100,
        )

    assert len(cache_monitor._break_history) <= cache_monitor._BREAK_HISTORY_LIMIT
    assert len(cache_monitor._break_history) == cache_monitor._BREAK_HISTORY_LIMIT


def test_fail_open_record_prompt_state():
    """record_prompt_state 异常时不抛（fail-open）。"""
    from agent import cache_monitor

    # 传一个奇怪的工具列表让 hash 失败也不抛
    class Weird:
        def __repr__(self):
            raise RuntimeError("repr bomb")

    state = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[{"function": Weird()}],
        model="m",
    )
    # 不抛即可
    assert state is not None


def test_get_stats_returns_dict():
    """get_stats 返回 dict 含 total_breaks 等字段。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    stats = cache_monitor.get_stats()
    assert "total_breaks" in stats
    assert "last_break" in stats
    assert "last_cache_read" in stats
    assert stats["total_breaks"] == 0


def test_check_cache_break_tools_changed():
    """工具 schema 变了 → root_cause 含 '工具 schema'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[{"function": {"name": "foo"}}],
        model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[{"function": {"name": "bar"}}],  # 工具变了
        model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "工具" in result or "schema" in result


def test_check_cache_break_model_changed():
    """model 变了 → root_cause 含 'model'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m1",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m2",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "model" in result


# ============================================================================
# _extract_cache_read 双路径测试
# ============================================================================

def test_extract_cache_read_from_dict_deepseek_field():
    """DeepSeek 字段名 prompt_cache_hit_tokens。"""
    from agent import AIAgent
    agent, _ = _make_minimal_agent()
    usage = {"prompt_cache_hit_tokens": 8000}
    assert agent._extract_cache_read(usage) == 8000


def test_extract_cache_read_from_dict_anthropic_field():
    """Anthropic 字段名 cache_read_input_tokens。"""
    from agent import AIAgent
    agent, _ = _make_minimal_agent()
    usage = {"cache_read_input_tokens": 7500}
    assert agent._extract_cache_read(usage) == 7500


def test_extract_cache_read_from_object_deepseek_field():
    """对象形态：prompt_cache_hit_tokens 属性。"""
    from agent import AIAgent
    agent, _ = _make_minimal_agent()
    usage = SimpleNamespace(prompt_cache_hit_tokens=9000)
    assert agent._extract_cache_read(usage) == 9000


def test_extract_cache_read_from_object_anthropic_field():
    """对象形态：cache_read_input_tokens 属性。"""
    from agent import AIAgent
    agent, _ = _make_minimal_agent()
    usage = SimpleNamespace(cache_read_input_tokens=6000)
    assert agent._extract_cache_read(usage) == 6000


def test_extract_cache_read_none():
    """usage=None 返回 0。"""
    from agent import AIAgent
    agent, _ = _make_minimal_agent()
    assert agent._extract_cache_read(None) == 0


# ============================================================================
# 端到端测试：AIAgent 主循环触发 cache_monitor hook
# ============================================================================

def _make_mock_response(response_text="你好", tool_calls=None, cache_read=5000):
    """构造 mock LLM 响应。usage 含 prompt_cache_hit_tokens。"""
    msg = SimpleNamespace(content=response_text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        prompt_cache_hit_tokens=cache_read,
        cache_read_input_tokens=cache_read,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_mock_llm_client(response_text="你好", tool_calls=None, cache_read=5000):
    """构造 async mock LLMClient。"""
    response = _make_mock_response(response_text, tool_calls, cache_read)
    client = SimpleNamespace()
    client.chat_completions = AsyncMock(return_value=response)
    client.chat_completions_stream = AsyncMock(return_value=response)
    client.model = "mock-model"
    return client


def _make_minimal_agent(tmp_path=None):
    """构造最小可用 AIAgent。"""
    from agent import AIAgent

    mock_llm = _make_mock_llm_client()
    kwargs = {
        "api_key": "fake",
        "model": "mock-model",
        "enabled_toolsets": [],
    }
    if tmp_path is not None:
        kwargs["omnimate_home"] = tmp_path

    agent = AIAgent(**kwargs)
    agent.llm_client = mock_llm
    return agent, mock_llm


async def test_e2e_run_conversation_triggers_cache_monitor(tmp_path):
    """通过真实 run_conversation 跑一轮主循环，验证 cache_monitor 的 record/check 被 call。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    agent, mock_llm = _make_minimal_agent(tmp_path)

    with patch.object(cache_monitor, "record_prompt_state",
                      wraps=cache_monitor.record_prompt_state) as mock_record, \
         patch.object(cache_monitor, "check_cache_break",
                      wraps=cache_monitor.check_cache_break) as mock_check:
        await agent.run_conversation("你好")

    # 至少调过一次 record 和 check
    assert mock_record.called, "record_prompt_state 未被触发（hook 没接入）"
    assert mock_check.called, "check_cache_break 未被触发（hook 没接入）"


async def test_e2e_call_llm_with_escalation_non_stream_triggers_hook(tmp_path):
    """非流式 _call_llm_with_escalation 路径也要 hook。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    agent, _ = _make_minimal_agent(tmp_path)

    with patch.object(cache_monitor, "record_prompt_state",
                      wraps=cache_monitor.record_prompt_state) as mock_record, \
         patch.object(cache_monitor, "check_cache_break",
                      wraps=cache_monitor.check_cache_break) as mock_check:
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="你是助手",
        )

    assert mock_record.called, "非流式路径未触发 record_prompt_state"
    assert mock_check.called, "非流式路径未触发 check_cache_break"


async def test_e2e_call_llm_with_escalation_stream_triggers_hook(tmp_path):
    """流式路径（stream_callback 非空）也要 hook。

    使用真实 async generator 让 chat_completions_stream 返回有效流，
    不依赖 fallback 路径——验证流式分支本身接入 cache_monitor。
    """
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    agent, _ = _make_minimal_agent(tmp_path)
    agent._stream_callback = lambda evt: None  # 触发流式分支

    # 构造真正的 async generator mock（单 chunk 含完整内容 + usage）
    async def _fake_stream(messages, *, tools=None, **kwargs):
        yield {
            "content": "你好",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read": 5000,
                "cache_creation": 0,
            },
        }

    agent.llm_client.chat_completions_stream = _fake_stream

    with patch.object(cache_monitor, "record_prompt_state",
                      wraps=cache_monitor.record_prompt_state) as mock_record, \
         patch.object(cache_monitor, "check_cache_break",
                      wraps=cache_monitor.check_cache_break) as mock_check:
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="你是助手",
        )

    assert mock_record.called, "流式路径未触发 record_prompt_state"
    assert mock_check.called, "流式路径未触发 check_cache_break"


# ============================================================================
# fail-open 端到端测试
# ============================================================================

async def test_e2e_fail_open_record_raises(tmp_path):
    """mock record_prompt_state 抛异常，主循环不崩。"""
    from agent import cache_monitor

    agent, _ = _make_minimal_agent(tmp_path)

    with patch.object(cache_monitor, "record_prompt_state",
                      side_effect=RuntimeError("boom")):
        # 不抛异常
        result = await agent.run_conversation("你好")
    assert result is not None


async def test_e2e_fail_open_check_raises(tmp_path):
    """mock check_cache_break 抛异常，主循环不崩。"""
    from agent import cache_monitor

    agent, _ = _make_minimal_agent(tmp_path)

    with patch.object(cache_monitor, "check_cache_break",
                      side_effect=RuntimeError("boom")):
        # 不抛异常
        result = await agent.run_conversation("你好")
    assert result is not None


async def test_e2e_fail_open_extract_cache_read_raises(tmp_path):
    """response.usage 访问异常也不崩。"""
    from agent import cache_monitor

    agent, _ = _make_minimal_agent(tmp_path)
    # 替换 mock client 的响应让 usage 抛异常
    bad_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )
    agent.llm_client.chat_completions = AsyncMock(return_value=bad_response)

    result = await agent.run_conversation("你好")
    assert result is not None


# ============================================================================
# notify_compaction 端到端：通过 compress_if_needed → llm_compact 触发
# ============================================================================

async def test_e2e_llm_compact_calls_notify_compaction(tmp_path):
    """直接调 llm_compact 触发 notify_compaction。"""
    from agent import cache_monitor
    from agent.context_pipeline import llm_compact

    cache_monitor.reset_cache_monitor()

    # 构造 > token_threshold 的消息列表
    big_content = "x" * 200000  # 远超 token_threshold
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_content},
        {"role": "assistant", "content": big_content},
        {"role": "user", "content": big_content},
        {"role": "assistant", "content": big_content},
    ] + [{"role": "user", "content": "recent"}]

    # mock LLM client 返回摘要
    mock_client = SimpleNamespace()
    mock_client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary here", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    ))

    with patch.object(cache_monitor, "notify_compaction",
                      wraps=cache_monitor.notify_compaction) as mock_notify:
        new_msgs, changed = await llm_compact(
            messages,
            llm_client=mock_client,
            model="m",
            keep_recent=2,
            token_threshold=100,  # 低阈值强制触发
        )
        assert changed is True
        assert mock_notify.called, "llm_compact 未调用 notify_compaction"


async def test_e2e_compact_then_no_break(tmp_path):
    """compact 触发 notify_compaction 后，cache 大降不算 break。"""
    from agent import cache_monitor
    from agent.context_pipeline import llm_compact

    cache_monitor.reset_cache_monitor()

    # 先种 baseline
    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    # 触发 compact（mock 版本直接调 notify_compaction）
    big_content = "x" * 200000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_content},
    ] * 5 + [{"role": "user", "content": "recent"}]

    mock_client = SimpleNamespace()
    mock_client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    ))

    await llm_compact(
        messages, llm_client=mock_client, model="m",
        keep_recent=2, token_threshold=100,
    )

    # compact 后 cache 大降，应该不算 break
    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=100,  # 大降
    )
    assert result is None


# ============================================================================
# reactive_compact 触发 notify_compaction（紧急通道：prompt_too_long 时）
# ============================================================================

def test_reactive_compact_calls_notify_compaction():
    """reactive_compact 触发后，notify_compaction 应被调用。"""
    from agent import cache_monitor
    from agent.context_pipeline import reactive_compact, CompressionSessionState

    cache_monitor.reset_cache_monitor()
    assert cache_monitor._pending_compaction is False

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "old2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "old3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "recent"},
    ]
    session_state = CompressionSessionState()
    new_msgs, changed = reactive_compact(
        messages, session_state=session_state, keep_recent=3,
    )
    assert changed is True
    assert cache_monitor._pending_compaction is True, (
        "reactive_compact 未触发 notify_compaction"
    )


def test_reactive_compact_then_no_break():
    """reactive_compact 触发 notify_compaction 后，cache 大降不算 break。"""
    from agent import cache_monitor
    from agent.context_pipeline import reactive_compact, CompressionSessionState

    cache_monitor.reset_cache_monitor()

    # 先种 baseline
    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    # 触发 reactive_compact
    messages = [
        {"role": "system", "content": "sys"},
    ] + [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    session_state = CompressionSessionState()
    reactive_compact(messages, session_state=session_state, keep_recent=3)

    # reactive 后 cache 大降，应该不算 break
    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=100,  # 大降
    )
    assert result is None


# ============================================================================
# reset_cache_monitor 在 __init__ 调用测试
# ============================================================================

def test_reset_called_in_init(tmp_path):
    """AIAgent.__init__ 会调 reset_cache_monitor。"""
    from agent import AIAgent
    from agent import cache_monitor

    # 先搞脏状态
    cache_monitor._break_history.append({"fake": True})
    cache_monitor._last_cache_read = 99999

    # 创建新 agent
    agent = AIAgent(
        api_key="fake",
        model="mock",
        enabled_toolsets=[],
        omnimate_home=tmp_path,
    )

    # 状态应被 reset
    assert cache_monitor._break_history == []
    assert cache_monitor._last_state is None
    assert cache_monitor._last_cache_read is None


# ============================================================================
# CCAR4 Task A: 12 维度扩展 + per-tool hash + diff 文件 + TTL 分析
# ============================================================================

def _make_tool(name, schema=None):
    """构造 OpenAI 格式工具 schema。"""
    return {"type": "function", "function": {
        "name": name,
        "description": f"tool {name}",
        "input_schema": schema or {"type": "object", "properties": {}},
    }}


def test_record_prompt_state_captures_12_dimensions():
    """record_prompt_state 接收 12 维参数都能捕获。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    state = cache_monitor.record_prompt_state(
        system_prompt="hello",
        tools=[_make_tool("foo")],
        model="deepseek-chat",
        max_tokens=8192,
        temperature=0.5,
        stream_mode=True,
        tool_choice="auto",
        user_content_prefix="用户首条消息",
        betas={"anthropic_beta": ["tools-2024"]},
        messages_count=10,
        cache_strategy="aggressive",
    )
    # 12 维度都应有值
    assert state.system_hash != 0
    assert state.tools_hash != 0
    assert state.model == "deepseek-chat"
    assert state.max_tokens == 8192
    assert state.temperature == 0.5
    assert state.stream_mode is True
    assert state.tool_choice == "auto"
    assert state.user_content_prefix != 0
    assert state.messages_count == 10
    assert state.cache_strategy == "aggressive"
    assert state.betas_hash != 0
    # system_prompt 是 str → single 边界
    assert state.system_boundary == "single"
    # per-tool hash 应有 1 条
    assert len(state.tool_hashes) == 1
    assert state.tool_hashes[0].name == "foo"
    assert state.tool_hashes[0].schema_hash != 0


def test_record_prompt_state_system_boundary_multi_block():
    """system_prompt 是 list 时 system_boundary == 'multi-block'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    state = cache_monitor.record_prompt_state(
        system_prompt=[{"type": "text", "text": "block1"}],
        tools=[],
        model="m",
    )
    assert state.system_boundary == "multi-block"


def test_check_cache_break_reports_all_12_dimensions():
    """两个 state 在 12 维度上都不同 → 根因报告含 12 个原因。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[_make_tool("t1")], model="m1",
        max_tokens=1000, temperature=0.1, stream_mode=False,
        tool_choice="none", user_content_prefix="u1",
        betas={"k": [1]}, messages_count=5,
        cache_strategy="aggressive",
    )
    cache_monitor.check_cache_break(
        current_state=s1, cache_read_tokens=10000,
    )

    # s2 用 list 形式 system_prompt 触发 system_boundary 变化
    s2 = cache_monitor.record_prompt_state(
        system_prompt=[{"type": "text", "text": "B"}],
        tools=[_make_tool("t2")], model="m2",
        max_tokens=2000, temperature=0.9, stream_mode=True,
        tool_choice="auto", user_content_prefix="u2",
        betas={"k": [2]}, messages_count=10,
        cache_strategy="conservative",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    # 12 个维度都有报告
    assert "system prompt" in result
    assert "工具" in result
    assert "model" in result
    assert "max_tokens" in result
    assert "temperature" in result
    assert "stream" in result
    assert "tool_choice" in result
    assert "user content" in result
    assert "messages count" in result
    assert "system 边界" in result
    assert "betas" in result
    assert "cache_strategy" in result


# ============================================================================
# per-tool hash 增删改测试
# ============================================================================

def test_per_tool_hash_added():
    """tools 从 [a, b, c] → [a, b, c, d]：根因含 '+1 (d)'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("a"), _make_tool("b"), _make_tool("c")],
        model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("a"), _make_tool("b"), _make_tool("c"), _make_tool("d")],
        model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "+1" in result
    assert "d" in result


def test_per_tool_hash_removed():
    """tools 从 [a, b, c] → [a, c]：根因含 '-1 (b)'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("a"), _make_tool("b"), _make_tool("c")],
        model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("a"), _make_tool("c")],
        model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "-1" in result
    assert "b" in result


def test_per_tool_hash_changed():
    """tools 从 [a, b, c] → [a, b'(改 schema), c]：根因含 '~1 (b)'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("a"), _make_tool("b"), _make_tool("c")],
        model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    # b 改 schema
    s2 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[
            _make_tool("a"),
            _make_tool("b", schema={"type": "object", "properties": {"new_field": {"type": "string"}}}),
            _make_tool("c"),
        ],
        model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "~1" in result
    assert "b" in result


# ============================================================================
# diff 文件落盘测试
# ============================================================================

def test_diff_file_written_on_system_change(tmp_path, monkeypatch):
    """system prompt 变化触发 break 时，diff 文件写到 .cache-breaks/。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    # mock omnimate home 到 tmp_path
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    s1 = cache_monitor.record_prompt_state(
        system_prompt="original system prompt",
        tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="changed system prompt",
        tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None

    # diff 文件应存在
    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    assert len(diff_files) == 1, f"应有 1 个 diff 文件，实际 {len(diff_files)}"
    content = diff_files[0].read_text(encoding="utf-8")
    assert "## system prompt" in content
    # PromptState 只存 hash 不存原文，验证 hash 对比信息
    assert "OLD hash:" in content
    assert "NEW hash:" in content


def test_diff_file_written_on_tools_change(tmp_path, monkeypatch):
    """tools 变化触发 break 时，diff 文件含 tools schema 段。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("foo")], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A",
        tools=[_make_tool("bar")], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None

    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    assert len(diff_files) >= 1
    content = diff_files[0].read_text(encoding="utf-8")
    assert "## tools schema" in content


def test_diff_file_not_written_on_non_schema_break(tmp_path, monkeypatch):
    """非 system/tools 变化（如 model 变）不写 diff 文件。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m1",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m2",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    # 不应写 diff 文件（model 变不写 diff）
    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    assert len(diff_files) == 0


def test_diff_file_lru_cap(tmp_path, monkeypatch):
    """diff 文件超过 max_cache_break_diff_files (100) 时删旧。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    # 写 105 个 diff 文件
    for i in range(105):
        s_prev = cache_monitor.record_prompt_state(
            system_prompt=f"sys_{i}", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_prev, cache_read_tokens=10000,
        )
        s_break = cache_monitor.record_prompt_state(
            system_prompt=f"sys_{i}_changed", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_break, cache_read_tokens=100,
        )

    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    # 精确等于 100（105 次写入后 LRU 应到精确上限）
    assert len(diff_files) == 100, f"diff 文件应 == 100，实际 {len(diff_files)}"


# ============================================================================
# TTL 时长分析测试
# ============================================================================

def test_ttl_analysis_no_field_change_short_elapsed():
    """无字段变化 + elapsed < 5min → 'server-side 或未知'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "server-side" in result or "未知" in result


def test_ttl_analysis_no_field_change_5min():
    """无字段变化 + 5min < elapsed < 1h → '>5min TTL 过期'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    # mock baseline 时间为 10 分钟前
    import time as _time
    cache_monitor._last_baseline_at = _time.time() - 600  # 10 分钟前

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "5min" in result or "TTL" in result


def test_ttl_analysis_no_field_change_1h():
    """无字段变化 + elapsed > 1h → '>1h TTL 过期'。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    # mock baseline 时间为 2 小时前
    import time as _time
    cache_monitor._last_baseline_at = _time.time() - 7200

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "1h" in result or "TTL" in result


# ============================================================================
# 端到端测试：_call_llm_with_escalation 传 12 维参数
# ============================================================================

async def test_e2e_12_dimensions_captured_in_record(tmp_path):
    """通过 _call_llm_with_escalation 触发 record_prompt_state，验证新参数被捕获。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    agent, _ = _make_minimal_agent(tmp_path)
    # 设置 config 让 max_tokens 可提取
    agent.config = {"model": {"max_tokens": 4096, "temperature": 0.3}}

    captured_kwargs = {}
    original_record = cache_monitor.record_prompt_state

    def capture_record(**kwargs):
        captured_kwargs.update(kwargs)
        return original_record(**kwargs)

    with patch.object(cache_monitor, "record_prompt_state", side_effect=capture_record):
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[{"type": "function", "function": {
                "name": "test_tool", "description": "d",
                "input_schema": {"type": "object"},
            }}],
            system_prompt="你是助手",
        )

    # 新参数应被传入
    assert "max_tokens" in captured_kwargs, "max_tokens 未传入 record_prompt_state"
    assert captured_kwargs["max_tokens"] == 4096
    assert "stream_mode" in captured_kwargs, "stream_mode 未传入"
    assert "messages_count" in captured_kwargs, "messages_count 未传入"
    assert captured_kwargs["messages_count"] == 1
    assert "user_content_prefix" in captured_kwargs, "user_content_prefix 未传入"


async def test_e2e_diff_file_written_on_system_change(tmp_path):
    """端到端：两次调用 system 变化，cache 大降 → diff 文件真落盘。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    # mock omnimate home
    import constants
    original_get_home = constants.get_omnimate_home
    constants.get_omnimate_home = lambda: tmp_path
    try:
        agent, _ = _make_minimal_agent(tmp_path)

        # 第一次调用建立 baseline（cache_read=10000）
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_make_mock_response(cache_read=10000)
        )
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="system A",
        )

        # 第二次调用 system 变 + cache 大降
        agent.llm_client.chat_completions = AsyncMock(
            return_value=_make_mock_response(cache_read=1000)
        )
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="system B totally different",
        )
    finally:
        constants.get_omnimate_home = original_get_home

    # diff 文件应存在
    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    assert len(diff_files) >= 1, "端到端：diff 文件未落盘"
    content = diff_files[0].read_text(encoding="utf-8")
    assert "## system prompt" in content


async def test_e2e_break_history_includes_diff_path(tmp_path):
    """端到端：break 后 _break_history 条目含 diff_path。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    import constants
    original_get_home = constants.get_omnimate_home
    constants.get_omnimate_home = lambda: tmp_path
    try:
        agent, _ = _make_minimal_agent(tmp_path)

        agent.llm_client.chat_completions = AsyncMock(
            return_value=_make_mock_response(cache_read=10000)
        )
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="system A",
        )

        agent.llm_client.chat_completions = AsyncMock(
            return_value=_make_mock_response(cache_read=1000)
        )
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="system B different",
        )
    finally:
        constants.get_omnimate_home = original_get_home

    stats = cache_monitor.get_stats()
    assert stats["last_break"] is not None
    assert "diff_path" in stats["last_break"]
    assert stats["last_break"]["diff_path"] is not None


# ============================================================================
# fail-open 测试（新维度）
# ============================================================================

def test_fail_open_diagnose_break_exception():
    """_diagnose_break 抛异常时 check_cache_break 不崩。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="B", tools=[], model="m",
    )
    # mock _diagnose_break 抛异常
    with patch.object(cache_monitor, "_diagnose_break",
                      side_effect=RuntimeError("diagnose bomb")):
        result = cache_monitor.check_cache_break(
            current_state=s2, cache_read_tokens=1000,
        )
    # fail-open：不抛，返回 None 或字符串
    assert result is None or isinstance(result, str)


def test_fail_open_write_break_diff_exception(tmp_path, monkeypatch):
    """_write_break_diff 抛异常时不影响 check_cache_break。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="B", tools=[], model="m",
    )
    # mock _write_break_diff 抛异常
    original_write = cache_monitor._write_break_diff
    def boom(*a, **kw):
        raise RuntimeError("write bomb")
    with patch.object(cache_monitor, "_write_break_diff", side_effect=boom):
        result = cache_monitor.check_cache_break(
            current_state=s2, cache_read_tokens=1000,
        )
    # 不崩，仍然报根因
    assert result is not None


# ============================================================================
# 现有 5 维度回归（新维度加进来后老测试仍过）
# ============================================================================

def test_backward_compat_record_without_new_params():
    """不传新参数调用 record_prompt_state 仍正常（向后兼容）。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    state = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    # 老字段都有
    assert state.system_hash != 0
    assert state.model == "m"
    # 新字段有默认值
    assert state.max_tokens == 0
    assert state.temperature is None
    assert state.stream_mode is False
    assert state.tool_hashes == []


def test_backward_compat_check_cache_break_system_change():
    """老模式（只传 system/tools/model）break 检测仍工作。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="B", tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "system prompt" in result


# ============================================================================
# reset_cache_monitor 清 _last_baseline_at
# ============================================================================

def test_reset_clears_last_baseline_at():
    """reset_cache_monitor 清 _last_baseline_at。"""
    from agent import cache_monitor
    import time as _time
    cache_monitor.reset_cache_monitor()

    # 搞点状态
    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)
    assert cache_monitor._last_baseline_at is not None

    cache_monitor.reset_cache_monitor()
    assert cache_monitor._last_baseline_at is None


# ============================================================================
# Review Fix Round 1 测试
# ============================================================================

def test_system_delta_shows_real_char_count():
    """Important 1: system delta 应显示真实字符差（不是 +0）。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A" * 100, tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A" * 150, tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    # delta 应是 +50（不是 +0）
    assert "+50 chars" in result, f"期望 '+50 chars'，实际：{result}"


def test_system_delta_shows_negative_char_count():
    """Important 1: system 缩短时 delta 显示负数。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt="A" * 200, tools=[], model="m",
    )
    cache_monitor.check_cache_break(current_state=s1, cache_read_tokens=10000)

    s2 = cache_monitor.record_prompt_state(
        system_prompt="A" * 80, tools=[], model="m",
    )
    result = cache_monitor.check_cache_break(
        current_state=s2, cache_read_tokens=1000,
    )
    assert result is not None
    assert "-120 chars" in result, f"期望 '-120 chars'，实际：{result}"


def test_system_len_multi_block():
    """Important 1: system_prompt 为 list（multi-block）时 system_len 正确计算。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    s1 = cache_monitor.record_prompt_state(
        system_prompt=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        tools=[], model="m",
    )
    assert s1.system_len == 10  # "hello"(5) + "world"(5)
    assert s1.system_boundary == "multi-block"


async def test_e2e_tool_choice_betas_passed_from_config(tmp_path):
    """Important 2: _call_llm_with_escalation 应从 config 读 tool_choice/betas 传入 record_prompt_state。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    agent, _ = _make_minimal_agent(tmp_path)
    # config 里放 tool_choice 和 betas
    agent.config = {
        "model": {
            "max_tokens": 4096,
            "temperature": 0.5,
            "tool_choice": "auto",
            "betas": {"anthropic_beta": "prompt-caching-2024-07-31"},
        }
    }

    captured_kwargs = {}
    original_record = cache_monitor.record_prompt_state

    def capture_record(**kwargs):
        captured_kwargs.update(kwargs)
        return original_record(**kwargs)

    with patch.object(cache_monitor, "record_prompt_state", side_effect=capture_record):
        await agent._call_llm_with_escalation(
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            system_prompt="你是助手",
        )

    assert "tool_choice" in captured_kwargs, "tool_choice 未传入 record_prompt_state"
    assert captured_kwargs["tool_choice"] == "auto"
    assert "betas" in captured_kwargs, "betas 未传入 record_prompt_state"
    assert captured_kwargs["betas"] == {"anthropic_beta": "prompt-caching-2024-07-31"}


def test_tool_choice_none_does_not_trigger_break():
    """Important 2: tool_choice=None（常见情况）不应导致维度差异——恒定 baseline。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()

    # 两次都不传 tool_choice（默认 None）
    s1 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    s2 = cache_monitor.record_prompt_state(
        system_prompt="A", tools=[], model="m",
    )
    assert s1.tool_choice is None
    assert s2.tool_choice is None
    assert s1.tool_choice == s2.tool_choice  # 恒定


def test_diff_filename_unique_same_second(tmp_path, monkeypatch):
    """Important 3: 同秒内写多个 diff 文件，文件名不覆盖。"""
    from agent import cache_monitor
    cache_monitor.reset_cache_monitor()
    import constants
    monkeypatch.setattr(constants, "get_omnimate_home", lambda: tmp_path)

    # 快速连续触发 5 次 break（都在同一秒）
    for i in range(5):
        s_prev = cache_monitor.record_prompt_state(
            system_prompt=f"sys_{i}", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_prev, cache_read_tokens=10000,
        )
        s_break = cache_monitor.record_prompt_state(
            system_prompt=f"sys_{i}_changed", tools=[], model="m",
        )
        cache_monitor.check_cache_break(
            current_state=s_break, cache_read_tokens=100,
        )

    diff_dir = tmp_path / ".cache-breaks"
    diff_files = list(diff_dir.glob("cache-break-*.diff"))
    # 应有 5 个（不覆盖）
    assert len(diff_files) == 5, f"同秒 5 次 break 应有 5 个 diff，实际 {len(diff_files)}"
    # 文件名唯一（无重复）
    names = [f.name for f in diff_files]
    assert len(set(names)) == 5, f"文件名应唯一，实际：{names}"
