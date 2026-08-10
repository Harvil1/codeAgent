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
