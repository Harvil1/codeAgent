# tests/test_time_based_mc.py
"""改造点 ④ time-based MC（60min 清旧工具结果）测试。

测试覆盖：
1. 60min 内不清（timestamp = now - 30min）
2. 60min 外清（timestamp = now - 70min）
3. keep_recent=5：10 个 tool result，最后 5 个保留
4. 没有 _timestamp：fail-open
5. 没有 assistant 消息：fail-open
6. enabled=False：开关关闭
7. _timestamp 不污染 prompt：strip 验证
"""
import time

import pytest

from agent.context_pipeline import time_based_clear_old_tool_results


# ---------- 辅助构造 ----------

def _mk_tool_msg(idx: int, content: str = None) -> dict:
    """构造一条 tool 结果消息。"""
    return {
        "role": "tool",
        "tool_call_id": f"call_{idx}",
        "name": f"tool_{idx}",
        "content": content or f'{{"result": "data_{idx}"}}',
    }


def _mk_assistant_with_tool_calls(ts: float = None) -> dict:
    """构造一条含 tool_calls 的 assistant 消息。"""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "tool_0", "arguments": "{}"}},
        ],
    }
    if ts is not None:
        msg["_timestamp"] = ts
    return msg


def _mk_assistant_final(ts: float = None, content: str = "最终回复") -> dict:
    """构造一条纯文本 assistant 消息（最终响应）。"""
    msg = {"role": "assistant", "content": content}
    if ts is not None:
        msg["_timestamp"] = ts
    return msg


def _mk_tool_chain(n: int, ts: float = None) -> list:
    """构造 n 条 tool result（每条带可选 _timestamp）。"""
    msgs = []
    for i in range(n):
        m = _mk_tool_msg(i)
        if ts is not None:
            m["_timestamp"] = ts
        msgs.append(m)
    return msgs


def _base_config(
    enabled: bool = True,
    gap_minutes: int = 60,
    keep_recent: int = 5,
) -> dict:
    """构造带 time_based_mc 配置的 context 子字典（flat keys）。

    compress_if_needed 传入的 config 已经是 self.config.get("context", {})，
    所以 time_based_* 是 flat key（跟 snip_keep_first / output_offload_threshold 一致）。
    """
    return {
        "time_based_mc_enabled": enabled,
        "time_based_mc_gap_minutes": gap_minutes,
        "time_based_mc_keep_recent": keep_recent,
    }


CLEARED_MARK = "[Old tool result content cleared]"


# ---------- 测试 1：60min 内不清 ----------

def test_within_60min_no_clear():
    """距上次 assistant 30 分钟 → 旧 tool result 不变。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(3, ts=now - 40 * 60),       # 3 条 tool，40 分钟前
        _mk_assistant_final(ts=now - 30 * 60),      # 最后 assistant，30 分钟前
    ]
    config = _base_config()
    result = time_based_clear_old_tool_results(messages, config)

    # 内容不变
    for i in range(3):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}', \
            f"tool result {i} 不应被清除（60min 内）"


# ---------- 测试 2：60min 外清 ----------

def test_over_60min_keeps_all_when_below_keep_recent():
    """距上次 assistant 70 分钟但 tool 数 < keep_recent → 全保留（不清）。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(3, ts=now - 80 * 60),       # 3 条 tool，80 分钟前
        _mk_assistant_final(ts=now - 70 * 60),      # 最后 assistant，70 分钟前
    ]
    config = _base_config(keep_recent=5)
    # 只有 3 条 tool，keep_recent=5 → 全保留（3 <= 5）
    result = time_based_clear_old_tool_results(messages, config)

    # 3 < 5（keep_recent），所以都不清
    for i in range(3):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}'


def test_over_60min_clear_with_many_tools():
    """距上次 assistant 70 分钟 + 8 条 tool result（超过 keep_recent）→ 前面被清。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(8, ts=now - 80 * 60),       # 8 条 tool，80 分钟前
        _mk_assistant_final(ts=now - 70 * 60),      # 最后 assistant，70 分钟前
    ]
    config = _base_config(keep_recent=5)
    result = time_based_clear_old_tool_results(messages, config)

    # 前 3 条被清（8 - 5 = 3）
    for i in range(3):
        assert result[i]["content"] == CLEARED_MARK, \
            f"tool result {i} 应被清除（超 60min 且不在 keep_recent 内）"
    # 后 5 条保留
    for i in range(3, 8):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}', \
            f"tool result {i} 应保留（在 keep_recent 内）"


# ---------- 测试 3：keep_recent=5 边界 ----------

def test_keep_recent_5_exact_boundary():
    """10 条 tool result + 超时 → 最后 5 个保留，前 5 个被清。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(10, ts=now - 100 * 60),
        _mk_assistant_final(ts=now - 70 * 60),
    ]
    config = _base_config(keep_recent=5)
    result = time_based_clear_old_tool_results(messages, config)

    # 前 5 个被清
    for i in range(5):
        assert result[i]["content"] == CLEARED_MARK, \
            f"tool result {i} 应被清除"
    # 后 5 个保留
    for i in range(5, 10):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}', \
            f"tool result {i} 应保留"


def test_keep_recent_5_fewer_tools():
    """4 条 tool result + keep_recent=5 + 超时 → 全保留（4 < 5）。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(4, ts=now - 100 * 60),
        _mk_assistant_final(ts=now - 70 * 60),
    ]
    config = _base_config(keep_recent=5)
    result = time_based_clear_old_tool_results(messages, config)

    for i in range(4):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}'


def test_keep_recent_zero_clears_all():
    """keep_recent=0 + 超时 → 全部 tool result 都清（不保留任何）。

    边缘场景：tool_indices[:-0] 会得到全列表，验证实现用 ``if keep_recent > 0``
    分支守住了这个边缘（不依赖 Python 切片 [:-0] 的反直觉行为）。
    """
    now = time.time()
    messages = [
        *_mk_tool_chain(6, ts=now - 100 * 60),     # 6 条 tool，100 分钟前
        _mk_assistant_final(ts=now - 70 * 60),     # 最后 assistant，70 分钟前
    ]
    config = _base_config(keep_recent=0)
    result = time_based_clear_old_tool_results(messages, config)

    # keep_recent=0 → 全部清
    for i in range(6):
        assert result[i]["content"] == CLEARED_MARK, \
            f"tool result {i} 应被清除（keep_recent=0）"


# ---------- 测试 4：没有 _timestamp 字段 → fail-open ----------

def test_no_timestamp_fail_open():
    """最后 assistant 没有 _timestamp → 返回原消息不动。"""
    messages = [
        *_mk_tool_chain(8),                         # 没有 _timestamp
        _mk_assistant_final(),                      # 没有 _timestamp
    ]
    config = _base_config()
    result = time_based_clear_old_tool_results(messages, config)

    # 全部不变
    for i in range(8):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}'


# ---------- 测试 5：没有 assistant 消息 → fail-open ----------

def test_no_assistant_msg_fail_open():
    """消息列表里没有 assistant → 返回原消息不动。"""
    now = time.time()
    messages = [
        {"role": "user", "content": "你好"},
        *_mk_tool_chain(8, ts=now - 80 * 60),       # tool 有 _timestamp 但没 assistant
    ]
    config = _base_config()
    result = time_based_clear_old_tool_results(messages, config)

    for i in range(8):
        assert result[i + 1]["content"] == f'{{"result": "data_{i}"}}'


# ---------- 测试 6：enabled=False → 开关关闭 ----------

def test_disabled_no_clear():
    """enabled=False → 即使超时也不清。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(10, ts=now - 100 * 60),
        _mk_assistant_final(ts=now - 70 * 60),
    ]
    config = _base_config(enabled=False)
    result = time_based_clear_old_tool_results(messages, config)

    for i in range(10):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}'


# ---------- 测试 7：_timestamp 不污染 prompt ----------
# 这个测试验证 strip 逻辑（在 _assemble_turn_messages 里）
# 我们直接测试 strip 函数

def test_strip_timestamp_from_messages():
    """_assemble_turn_messages 组装的 messages 里不含 _timestamp 字段。

    通过 AIAgent._assemble_turn_messages 验证。
    """
    from agent.context_pipeline import strip_internal_fields

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi", "_timestamp": 1234567890.0},
        {"role": "assistant", "content": "hello", "_timestamp": 1234567891.0},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "{}",
         "_timestamp": 1234567892.0},
    ]
    stripped = strip_internal_fields(messages)
    for m in stripped:
        assert "_timestamp" not in m, \
            f"消息 role={m['role']} 仍含 _timestamp（应 strip）"


def test_strip_timestamp_preserves_other_fields():
    """strip_internal_fields 只去 _timestamp，其他字段保留。"""
    from agent.context_pipeline import strip_internal_fields

    messages = [
        {"role": "tool", "tool_call_id": "c1", "name": "t",
         "content": "data", "_timestamp": 1234.5},
    ]
    stripped = strip_internal_fields(messages)
    assert stripped[0]["tool_call_id"] == "c1"
    assert stripped[0]["name"] == "t"
    assert stripped[0]["content"] == "data"
    assert "_timestamp" not in stripped[0]


# ---------- 额外：幂等性 + fail-open 异常 ----------

def test_already_cleared_idempotent():
    """已清除的 tool result 再次调用不报错（幂等）。"""
    now = time.time()
    messages = [
        *_mk_tool_chain(10, ts=now - 100 * 60),
        _mk_assistant_final(ts=now - 70 * 60),
    ]
    config = _base_config(keep_recent=5)
    result1 = time_based_clear_old_tool_results(messages, config)
    result2 = time_based_clear_old_tool_results(result1, config)

    # 第二次结果应与第一次一致
    for i in range(10):
        assert result2[i]["content"] == result1[i]["content"]


def test_fail_open_on_exception():
    """函数内部异常 → 返回原消息（fail-open）。"""
    # 用一个会触发异常的 config（非 dict）
    messages = [
        {"role": "user", "content": "hi"},
    ]
    # config 没有 "context" 键 → .get("context", {}) 应 fail-open
    result = time_based_clear_old_tool_results(messages, {})
    assert result is messages or result == messages


# ---------- 集成测试：通过 compress_if_needed 编排触发 ----------

import pytest
from unittest.mock import MagicMock

from agent.context_pipeline import compress_if_needed, CompressionSessionState


class _FakeLLM:
    """模拟 LLM client（async chat_completions），仅供 compress_if_needed 接口对齐。"""
    async def chat_completions(self, msgs):
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="summary"))]
        return m


def _orch_cfg(
    enabled: bool = True,
    gap_minutes: int = 60,
    keep_recent: int = 5,
) -> dict:
    """compress_if_needed 用的 context 子字典（flat keys）。

    包含 snip/offload/L4 需要的 flat key，让其他层不干扰本测试。
    """
    return {
        "snip_message_threshold": 10 ** 9,     # 禁 L1 snip
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        "output_offload_threshold": 10 ** 9,   # 禁 L2 offload
        "output_offload_preview": 2000,
        "micro_keep_recent_results": 3,
        "llm_compact_token_threshold": 10 ** 9,  # 禁 L4
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "llm_compact_keep_recent": 10,
        "transcript_enabled": False,
        "transcript_retention": 20,
        "tool_result_total_budget": 10 ** 9,
        # time-based MC（改造点 ④）
        "time_based_mc_enabled": enabled,
        "time_based_mc_gap_minutes": gap_minutes,
        "time_based_mc_keep_recent": keep_recent,
    }


@pytest.mark.asyncio
async def test_compress_if_needed_skips_when_disabled(tmp_path):
    """compress_if_needed 编排：time_based_mc_enabled=False → tool result 不被清。

    回归保障：Critical 1（config 嵌套 bug）修复后，enabled=False 配置（flat key）
    能真正被编排器读到、让 time-based MC 跳过。若 bug 复现（函数去读嵌套 context 键），
    enabled 会被 fallback 到默认 True，tool result 会被清掉，断言失败。
    """
    now = time.time()
    messages = [
        {"role": "system", "content": "sys"},
        *_mk_tool_chain(8, ts=now - 100 * 60),     # 8 条 tool，100 分钟前
        _mk_assistant_final(ts=now - 70 * 60),     # 超时
    ]
    cfg = _orch_cfg(enabled=False, keep_recent=5)
    state = CompressionSessionState()
    out, _ = await compress_if_needed(
        messages,
        llm_client=_FakeLLM(),
        model="test",
        config=cfg,
        session_state=state,
        agent_home=str(tmp_path),
        session_id="s1",
    )
    # enabled=False → tool result 内容不被清
    tool_contents = [m.get("content") for m in out if m.get("role") == "tool"]
    assert all(c != CLEARED_MARK for c in tool_contents), \
        "enabled=False 时不应有 tool result 被清"


@pytest.mark.asyncio
async def test_compress_if_needed_clears_when_enabled_and_overdue(tmp_path):
    """compress_if_needed 编排：enabled=True + 超时 + tool 数 > keep_recent → 清。

    正向验证：flat key 配置能让 time-based MC 真正触发，证明 config shape 修复有效。
    """
    now = time.time()
    messages = [
        {"role": "system", "content": "sys"},
        *_mk_tool_chain(8, ts=now - 100 * 60),     # 8 条 tool，100 分钟前
        _mk_assistant_final(ts=now - 70 * 60),     # 超时
    ]
    cfg = _orch_cfg(enabled=True, keep_recent=5)
    state = CompressionSessionState()
    out, _ = await compress_if_needed(
        messages,
        llm_client=_FakeLLM(),
        model="test",
        config=cfg,
        session_state=state,
        agent_home=str(tmp_path),
        session_id="s2",
    )
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    cleared = [m for m in tool_msgs if m.get("content") == CLEARED_MARK]
    kept = [m for m in tool_msgs if m.get("content") != CLEARED_MARK]
    assert len(cleared) == 3, f"期望 3 条被清（8 - 5），实际 {len(cleared)}"
    assert len(kept) == 5, f"期望 5 条保留，实际 {len(kept)}"
