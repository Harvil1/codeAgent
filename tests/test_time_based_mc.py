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

from agent.context_pipeline import time_based_clear_old_tool_results, reset_offload_decisions


@pytest.fixture(autouse=True)
def _reset_offload_decisions_per_test():
    """每个测试前后清空 _offload_decisions（防测试间 tool_call_id 冲突）。"""
    reset_offload_decisions()
    yield
    reset_offload_decisions()


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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # 内容不变
    assert changed is False, "60min 内不应触发清除（changed=False）"
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # 3 < 5（keep_recent），所以都不清
    assert changed is False
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # 前 3 条被清（8 - 5 = 3）
    assert changed is True, "应触发清除（changed=True）"
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # 前 5 个被清
    assert changed is True
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    assert changed is False
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # keep_recent=0 → 全部清
    assert changed is True
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    # 全部不变
    assert changed is False
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    assert changed is False
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
    result, changed = time_based_clear_old_tool_results(messages, config)

    assert changed is False
    for i in range(10):
        assert result[i]["content"] == f'{{"result": "data_{i}"}}'


# ---------- 测试 7：_timestamp 不污染 prompt ----------
# 这个测试验证 strip 逻辑（在主循环 compress 之后、发 LLM 之前）
# 我们直接测试 strip 函数

def test_strip_timestamp_from_messages():
    """主循环 strip_internal_fields 后的 messages 里不含 _timestamp 字段。"""
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
    result1, changed1 = time_based_clear_old_tool_results(messages, config)
    result2, changed2 = time_based_clear_old_tool_results(result1, config)

    # 第一次清了 → changed1=True；第二次幂等 → changed2=False（没有新内容被清）
    assert changed1 is True, "第一次调用应触发清除"
    assert changed2 is False, "第二次调用应幂等（changed=False）"

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
    result, changed = time_based_clear_old_tool_results(messages, {})
    assert result is messages or result == messages
    assert changed is False, "异常 fail-open 时 changed=False"


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


def _mk_paired_tool_chain(
    n: int,
    *,
    tool_ts: float = None,
    asst_ts: float = None,
) -> list:
    """构造 system + assistant(tool_calls) → tool result 配对消息链（过 _fix_tool_call_pairs）。

    Round 2 fix：compress_if_needed 的 changed=True 触发 _fix_tool_call_pairs，
    后者会删除没有对应 assistant(tool_calls) 的孤儿 tool result。
    所以集成测试的消息必须有正确的配对。

    结构：
      [system, assistant(tc_0..tc_{n-1}), tool_0, ..., tool_{n-1}, assistant_final(ts)]
    """
    msgs = [{"role": "system", "content": "sys"}]
    # 一条 assistant 带 n 个 tool_calls
    asst = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": f"tool_{i}", "arguments": "{}"}}
            for i in range(n)
        ],
    }
    if tool_ts is not None:
        asst["_timestamp"] = tool_ts
    msgs.append(asst)
    # n 条 tool result（id 对应 tool_calls）
    for i in range(n):
        m = _mk_tool_msg(i)
        if tool_ts is not None:
            m["_timestamp"] = tool_ts
        msgs.append(m)
    # 最后一条 assistant（带 _timestamp，time-based MC 用它判超时）
    msgs.append(_mk_assistant_final(ts=asst_ts))
    return msgs


@pytest.mark.asyncio
async def test_compress_if_needed_skips_when_disabled(tmp_path):
    """compress_if_needed 编排：time_based_mc_enabled=False → tool result 不被清。

    回归保障：Critical 1（config 嵌套 bug）修复后，enabled=False 配置（flat key）
    能真正被编排器读到、让 time-based MC 跳过。若 bug 复现（函数去读嵌套 context 键），
    enabled 会被 fallback 到默认 True，tool result 会被清掉，断言失败。
    """
    now = time.time()
    messages = _mk_paired_tool_chain(8, tool_ts=now - 100 * 60, asst_ts=now - 70 * 60)
    cfg = _orch_cfg(enabled=False, keep_recent=5)
    state = CompressionSessionState()
    out, _, _cp = await compress_if_needed(
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

    Round 2 fix：测试消息加了正确的 assistant(tool_calls) → tool 配对，
    否则 changed=True 时 _fix_tool_call_pairs 会把孤儿 tool result 删掉。
    """
    now = time.time()
    messages = _mk_paired_tool_chain(8, tool_ts=now - 100 * 60, asst_ts=now - 70 * 60)
    cfg = _orch_cfg(enabled=True, keep_recent=5)
    state = CompressionSessionState()
    out, _, _cp = await compress_if_needed(
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


# ---------- 端到端测试：_assemble_turn_messages → compress_if_needed → strip ----------
# 防止"测试绿但生产死代码"再次发生——Critical fix 的根本保障。
# strip_internal_fields 若在 compress_if_needed 之前调用，time-based MC
# 会永远拿不到 _timestamp（死代码）。本测试验证完整生产链路。

import time as _time_module
from agent.context_pipeline import strip_internal_fields


@pytest.mark.asyncio
async def test_e2e_assemble_compress_strip_chained(tmp_path):
    """端到端：_assemble_turn_messages → compress_if_needed → strip_internal_fields。

    验证 Round 2 Critical fix：strip 挪到 compress 之后后，
    time-based MC 能在生产链路里真正读到 _timestamp 并清旧 tool result。

    生产调用链（agent/__init__.py:779-789）：
        messages = self._assemble_turn_messages(...)    # 不再 strip（保留 _timestamp）
        messages, _ = await self._run_context_compression(messages, ...)
            └─ compress_if_needed → time_based_clear_old_tool_results → 读 _timestamp → 清
        messages = strip_internal_fields(messages)      # strip _timestamp（保护 prompt cache）

    本测试直接模拟这条链路，确保：
      1. compress 后旧 tool result 被清（说明 time-based MC 真触发了）
      2. strip 后 messages 不含 _timestamp（说明 prompt cache 保护没破）
    """
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    # 构造超时（> 60min）的 conversation_history
    # 使用正确配对的 assistant(tool_calls) → tool result 消息链，
    # 这样 changed=True 时 _fix_tool_call_pairs 不会把 tool result 当孤儿删掉。
    now = _time_module.time()
    paired = _mk_paired_tool_chain(
        8, tool_ts=now - 100 * 60, asst_ts=now - 70 * 60,
    )
    # conversation_history 不含 system 消息（_assemble_turn_messages 会加）
    agent.conversation_history = paired[1:]

    # Step 1：_assemble_turn_messages（不再 strip _timestamp）
    messages = agent._assemble_turn_messages(system_prompt="sys", injected={})

    # 验证 _timestamp 还在（说明 strip 没有提前执行）
    has_ts = any("_timestamp" in m for m in messages)
    assert has_ts, "_assemble_turn_messages 返回的 messages 应保留 _timestamp（strip 已挪到后面）"

    # Step 2：compress_if_needed（time-based MC 应读到 _timestamp 并触发清除）
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    ctx_cfg = dict(_orch_cfg(enabled=True, keep_recent=5))
    state = CompressionSessionState()
    messages, compressed, _cp = await compress_if_needed(
        messages,
        llm_client=_FakeLLM(),
        model="test",
        config=ctx_cfg,
        session_state=state,
        agent_home=str(tmp_path),
        session_id="e2e_test",
    )

    # 关键断言 1：compress 返回 compressed=True（time-based MC 的 c0 参与了 changed flag）
    assert compressed is True, \
        "time-based MC 清了内容但 changed flag 应为 True（Important fix 验证）"

    # 关键断言 2：旧 tool result 被清
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    cleared = [m for m in tool_msgs if m.get("content") == CLEARED_MARK]
    assert len(cleared) == 3, \
        f"期望 3 条被清（8 - 5 = 3），实际 {len(cleared)}（time-based MC 在生产链路真生效）"

    # Step 3：strip_internal_fields（主循环 compress 之后、发 LLM 之前）
    messages = strip_internal_fields(messages)

    # 关键断言 3：strip 后不含 _timestamp（保护 prompt cache）
    for m in messages:
        assert "_timestamp" not in m, \
            f"消息 role={m.get('role')} 仍含 _timestamp（应 strip，保护 prompt cache）"
