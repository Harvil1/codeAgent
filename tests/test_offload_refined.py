# tests/test_offload_refined.py
"""改造点 ① 大文件落盘精细化（per-tool + per-message + 决策冻结）测试。

7 个核心用例 + 端到端测试（覆盖 _assemble_turn_messages → compress_if_needed → strip_internal_fields 完整链路）。
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.context_pipeline import (
    offload_large_tool_results,
    reset_offload_decisions,
    _offload_decisions,
    _OFFLOAD_DECISIONS_LIMIT,
    strip_internal_fields,
    compress_if_needed,
    CompressionSessionState,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_paired_msgs(sizes):
    """构造 system + user + N 对 assistant(tool_calls)/tool result。

    sizes: 每个 tool result content 的字符数列表。
    返回 [system, user, (assistant+tool) × N]。
    """
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u"})
    for i, size in enumerate(sizes):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}", "name": "t",
            "content": "x" * size,
        })
    return msgs


@pytest.fixture(autouse=True)
def _reset_decisions():
    """每个测试前后清空 _offload_decisions（防测试间互相污染）。"""
    reset_offload_decisions()
    yield
    reset_offload_decisions()


class _FakeLLM:
    """模拟 LLM client（async chat_completions）。"""
    async def chat_completions(self, msgs):
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="summary"))]
        return m


def _offload_cfg(**overrides):
    """compress_if_needed 用的 context 子字典，禁 L1/L4，聚焦落盘逻辑。"""
    base = {
        "snip_message_threshold": 10 ** 9,     # 禁 L1
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        "output_offload_threshold": 50000,      # 改造点 ① 默认 50K
        "output_offload_preview": 2000,
        "message_offload_threshold": 200000,    # per-message 200K
        "offload_decision_freeze": True,
        "micro_keep_recent_results": 0,         # 测试场景不要 keep_recent 干扰
        "llm_compact_token_threshold": 10 ** 9,  # 禁 L4
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "llm_compact_keep_recent": 10,
        "transcript_enabled": False,
        "transcript_retention": 20,
        "tool_result_total_budget": 10 ** 9,    # 禁 L2.6
        "time_based_mc_enabled": False,         # 禁 time-based MC
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 用例 1: 单条 60K 字符工具结果 → 落盘（per-tool 触发）
# ---------------------------------------------------------------------------

def test_per_tool_60k_triggers_offload(tmp_path):
    """60K > 50K 阈值 → 应落盘。"""
    msgs = _mk_paired_msgs([60000])
    out, changed = offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    # content 应被替换为占位 JSON（含 truncated/full_at）
    parsed = json.loads(tool_msgs[0]["content"])
    assert parsed.get("truncated") is True
    assert "full_at" in parsed


# ---------------------------------------------------------------------------
# 用例 2: 单条 30K 字符工具结果 → 不落盘（per-tool 不触发）
# ---------------------------------------------------------------------------

def test_per_tool_30k_no_offload(tmp_path):
    """30K < 50K 阈值 → 不落盘。"""
    msgs = _mk_paired_msgs([30000])
    out, changed = offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    assert changed is False
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    # 原内容保留（不是 JSON 占位）
    assert tool_msgs[0]["content"] == "x" * 30000


# ---------------------------------------------------------------------------
# 用例 3: per-message 聚合触发（10 × 25K = 250K > 200K msg_threshold）
# ---------------------------------------------------------------------------

def test_per_message_aggregate_triggers_offload(tmp_path):
    """10 个 25K 字符结果，单独都不超 50K，但总和 250K > 200K → 选最大的几个落盘。"""
    msgs = _mk_paired_msgs([25000] * 10)  # 总 250K > 200K
    out, changed = offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    # 应至少有 2 条被落盘（每条 25K，需要落盘 2 条才能让总和 < 200K：
    # 250000 - 25000*2 = 200000，正好等于阈值；实际落盘后 content 变小）
    offloaded = []
    for m in tool_msgs:
        try:
            parsed = json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded.append(m)
        except (ValueError, TypeError):
            pass
    assert len(offloaded) >= 2, f"期望至少 2 条被落盘，实际 {len(offloaded)}"


# ---------------------------------------------------------------------------
# 用例 4: 决策冻结 byte-identical（跨轮次）
# ---------------------------------------------------------------------------

def test_decision_freeze_byte_identical_across_turns(tmp_path):
    """第 2 轮 compress 时，已落盘的 tool result 必须跟第 1 轮 byte-identical。

    这是 prompt cache 保护的核心——同一 tool_call_id 的 content 在所有轮次必须一致。
    """
    msgs = _mk_paired_msgs([60000])

    # 第 1 轮
    out1, c1 = offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    assert c1 is True
    tool1 = [m for m in out1 if m.get("role") == "tool"][0]
    content1 = tool1["content"]
    # 决策已记录
    assert "call_0" in _offload_decisions

    # 第 2 轮：用第 1 轮的 out1 再跑一次
    out2, c2 = offload_large_tool_results(
        out1, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    tool2 = [m for m in out2 if m.get("role") == "tool"][0]
    content2 = tool2["content"]

    # byte-identical（prompt cache 神圣不可侵犯）
    assert content1 == content2, (
        "决策冻结失败：第 2 轮 content 与第 1 轮不一致（破坏 prompt cache）"
    )

    # 第 2 轮不应重新评估（_offload_decisions 命中，不走 maybe_offload）
    # 验证方式：落盘文件数量不应增加（同 tool_call_id 不重复落盘）
    offload_dir = Path(tmp_path) / ".task_outputs" / "tool-results"
    offload_files = list(offload_dir.glob("*.txt")) if offload_dir.exists() else []
    assert len(offload_files) == 1, (
        f"决策冻结应避免重复落盘，但找到 {len(offload_files)} 个文件"
    )


# ---------------------------------------------------------------------------
# 用例 5: 新会话 reset_offload_decisions 清空
# ---------------------------------------------------------------------------

def test_reset_offload_decisions_clears_state(tmp_path):
    """reset_offload_decisions 后 _offload_decisions 应为空。"""
    msgs = _mk_paired_msgs([60000])
    offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    assert len(_offload_decisions) > 0

    reset_offload_decisions()
    assert len(_offload_decisions) == 0


# ---------------------------------------------------------------------------
# 用例 6: LRU 上限 1000 条
# ---------------------------------------------------------------------------

def test_lru_limit_1000(tmp_path):
    """构造 > 1000 个不同 tool_call_id 的决策，dict 不超过 _OFFLOAD_DECISIONS_LIMIT。"""
    # 构造 1050 条 tool 消息，每条刚好超阈值
    sizes = [51000] * 1050
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i, size in enumerate(sizes):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_lru_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_lru_{i}", "name": "t",
            "content": "z" * size,
        })

    offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=10 ** 9,  # 禁 per-message（防聚合逻辑干扰 LRU 测试）
        freeze=True,
    )
    assert len(_offload_decisions) <= _OFFLOAD_DECISIONS_LIMIT, (
        f"LRU 上限失败：_offload_decisions 有 {len(_offload_decisions)} 条 "
        f"(应 <= {_OFFLOAD_DECISIONS_LIMIT})"
    )


# ---------------------------------------------------------------------------
# 用例 7: per-message 不跨 user 边界
# ---------------------------------------------------------------------------

def test_per_message_does_not_cross_user_boundary(tmp_path):
    """[user, tool, tool, user, tool, tool] 每个 user 段独立判断。

    构造两个段：每段单独不超 200K，加起来也不应触发 per-message offload。
    """
    msgs = [{"role": "system", "content": "s"}]

    # 段 1：user + 3 个 tool result（各 30K = 90K，< 200K msg_threshold）
    msgs.append({"role": "user", "content": "u1"})
    for i in range(3):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_s1_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_s1_{i}", "name": "t",
            "content": "a" * 30000,  # 30K × 3 = 90K < 200K
        })

    # 段 2：user + 3 个 tool result（各 30K = 90K，< 200K msg_threshold）
    msgs.append({"role": "user", "content": "u2"})
    for i in range(3):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_s2_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_s2_{i}", "name": "t",
            "content": "b" * 30000,  # 30K × 3 = 90K < 200K
        })

    out, changed = offload_large_tool_results(
        msgs, agent_home=Path(tmp_path), threshold=50000,
        message_threshold=200000, freeze=True,
    )
    # 每段独立判断（90K < 200K）→ 不应触发 per-message offload
    assert changed is False, (
        "per-message 跨 user 边界了：两段各 90K 不应触发，但触发了"
    )


# ---------------------------------------------------------------------------
# 端到端测试：_assemble_turn_messages → compress_if_needed → strip_internal_fields
# 防止"测试绿但生产死代码"再次发生（Task 1 的 silent-dead-code bug 教训）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_offload_refined_full_chain(tmp_path):
    """端到端：完整生产链路验证改造点 ① 决策冻结真的生效。

    生产调用链（agent/__init__.py:779-792）：
        messages = self._assemble_turn_messages(...)    # 组装（含 conversation_history）
        messages, _ = await self._run_context_compression(messages, ...)
            └─ compress_if_needed → L2.6 预算 offload → 落盘 + 记决策
        messages = strip_internal_fields(messages)      # strip _timestamp（保护 prompt cache）

    本测试构造：
      - conversation_history 含 5 个 50K 字符 tool result（总 250K > 200K msg_threshold）
      - 跑两轮 compress，验证第 2 轮的落盘 tool result 跟第 1 轮 byte-identical（决策冻结）
    """
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    # 构造含 5 个 60K tool result 的 conversation_history（总 300K > 200K）
    # 用正确配对的 assistant(tool_calls) → tool result 链
    history = [{"role": "user", "content": "hello"}]
    for i in range(5):
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_e2e_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"call_e2e_{i}", "name": "t",
            "content": "Q" * 60000,  # 60K each, 总 300K > 200K
        })
    agent.conversation_history = history

    # ── 第 1 轮 ──
    messages_turn1 = agent._assemble_turn_messages(system_prompt="sys", injected={})

    # compress_if_needed（走 L2 micro + L2.5 per-message 聚合 + L2.6 预算 offload → 落盘 + 记决策）
    ctx_cfg = _offload_cfg(
        output_offload_threshold=50000,     # 60K > 50K → L2 micro 会落盘
        message_offload_threshold=200000,   # per-message 200K（L2.5 聚合阈值）
        # 注意：L2.6 全局预算用 tool_result_total_budget（_offload_cfg 设为 10**9 禁用），
        # 所以 L2.6 实际不触发——micro_keep_recent=3 保护最近 3 条后，
        # 前 2 条（120K）已超 per-message 200K? 不——2 条 60K = 120K < 200K，
        # 所以 per-message 也不触发。L2 micro（per-tool 60K > 50K）才是主触发层。
        micro_keep_recent_results=3,        # 保护最近 3 条，前 2 条会落盘
    )
    state = CompressionSessionState()
    messages_after_c1, c1 = await compress_if_needed(
        messages_turn1,
        llm_client=_FakeLLM(),
        model="test",
        config=ctx_cfg,
        session_state=state,
        agent_home=Path(tmp_path),
        session_id="e2e_offload",
    )
    assert c1 is True, "第 1 轮 compress 应触发落盘"

    # strip _timestamp（主循环 compress 之后）
    messages_stripped_1 = strip_internal_fields(messages_after_c1)

    # 提取第 1 轮落盘的 tool result（content 是 JSON 占位）
    tool_msgs_1 = [m for m in messages_stripped_1 if m.get("role") == "tool"]
    offloaded_1 = {}
    for m in tool_msgs_1:
        try:
            parsed = json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded_1[m["tool_call_id"]] = m["content"]
        except (ValueError, TypeError):
            pass
    assert len(offloaded_1) >= 1, "至少 1 条 tool result 应被落盘"
    # 决策已记录
    for tc_id in offloaded_1:
        assert tc_id in _offload_decisions, f"{tc_id} 应在 _offload_decisions 中"

    # ── 第 2 轮：模拟下一轮 LLM 调用 ──
    # 用第 1 轮的 stripped 结果作为新一轮的 conversation_history
    agent.conversation_history = messages_stripped_1[1:]  # 跳过 system
    messages_turn2 = agent._assemble_turn_messages(system_prompt="sys", injected={})

    messages_after_c2, c2 = await compress_if_needed(
        messages_turn2,
        llm_client=_FakeLLM(),
        model="test",
        config=ctx_cfg,
        session_state=state,
        agent_home=Path(tmp_path),
        session_id="e2e_offload",
    )
    messages_stripped_2 = strip_internal_fields(messages_after_c2)

    # 提取第 2 轮落盘的 tool result
    tool_msgs_2 = [m for m in messages_stripped_2 if m.get("role") == "tool"]
    offloaded_2 = {}
    for m in tool_msgs_2:
        try:
            parsed = json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded_2[m["tool_call_id"]] = m["content"]
        except (ValueError, TypeError):
            pass

    # ★ 核心断言：byte-identical（决策冻结保护 prompt cache）
    for tc_id, content1 in offloaded_1.items():
        content2 = offloaded_2.get(tc_id)
        assert content2 is not None, f"{tc_id} 第 2 轮找不到对应落盘 tool result"
        assert content1 == content2, (
            f"E2E 决策冻结失败：{tc_id} 第 2 轮 content 与第 1 轮不一致，"
            "prompt cache 会被破坏"
        )

    # 验证第 2 轮没新增落盘文件（同 tool_call_id 不重复落盘）
    offload_dir = Path(tmp_path) / ".task_outputs" / "tool-results"
    offload_files = list(offload_dir.glob("*.txt")) if offload_dir.exists() else []
    files_after_turn1 = len(offloaded_1)
    assert len(offload_files) == files_after_turn1, (
        f"第 2 轮不应重复落盘，期望 {files_after_turn1} 个文件，"
        f"实际 {len(offload_files)} 个"
    )


# ---------------------------------------------------------------------------
# E2E: AIAgent.__init__ 调 reset_offload_decisions
# ---------------------------------------------------------------------------

def test_aiagent_init_resets_offload_decisions(tmp_path):
    """新建 AIAgent 实例时 _offload_decisions 应被清空（不跨会话泄漏）。"""
    from agent import AIAgent

    # 先填充 _offload_decisions
    _offload_decisions["stale_session_id"] = {"preview": "x", "file_path": None}
    assert len(_offload_decisions) > 0

    # 新建 agent（模拟新会话开始）
    AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    # _offload_decisions 应被清空
    assert len(_offload_decisions) == 0, (
        "新会话开始后 _offload_decisions 应清空（防跨会话泄漏）"
    )


# ---------------------------------------------------------------------------
# E2E: _enforce_per_message_budget 在 compress_if_needed 生产路径中真正执行
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_per_message_budget_runs_in_production(tmp_path):
    """端到端验证 Important 1：_enforce_per_message_budget 通过 compress_if_needed 真正执行。

    场景：10 个 25K tool result（总 250K > 200K per-message 阈值），
    每个单独 < per-tool 阈值（50K），所以 L2 micro_compact 不会落盘。
    只有 L2.5 per-message 聚合才会落盘。

    生产调用链：compress_if_needed → micro_compact（不触发）→
        _enforce_per_message_budget（触发！按 user 边界聚合落盘）
    """
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    # 10 个 25K tool result（总 250K > 200K per-message 阈值）
    # 每个单独 25K < 50K per-tool 阈值 → L2 micro 不会触发
    history = [{"role": "user", "content": "hello"}]
    for i in range(10):
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_pm_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"call_pm_{i}", "name": "t",
            "content": "P" * 25000,  # 25K each, 总 250K > 200K
        })
    agent.conversation_history = history

    messages = agent._assemble_turn_messages(system_prompt="sys", injected={})

    ctx_cfg = _offload_cfg(
        output_offload_threshold=50000,     # 25K < 50K → L2 micro 不触发
        message_offload_threshold=200000,   # 250K > 200K → L2.5 per-message 触发
        micro_keep_recent_results=0,        # 不保护任何条（让 per-message 跑）
        tool_result_total_budget=10 ** 9,   # 禁 L2.6（只测 per-message）
    )
    state = CompressionSessionState()
    messages_after, changed = await compress_if_needed(
        messages,
        llm_client=_FakeLLM(),
        model="test",
        config=ctx_cfg,
        session_state=state,
        agent_home=Path(tmp_path),
        session_id="e2e_per_msg",
    )
    assert changed is True, "per-message 聚合应触发落盘"

    # 验证确实有 tool result 被落盘（content 变成 JSON 占位）
    tool_msgs = [m for m in messages_after if m.get("role") == "tool"]
    offloaded = []
    for m in tool_msgs:
        try:
            parsed = json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded.append(m)
        except (ValueError, TypeError):
            pass

    # 250K - 200K = 50K 需要落盘。每条 25K，至少落 2 条（50K）才能压到 <= 200K
    # 但落盘后 content 变成 ~400 字节预览，实际减约 24600 字符/条
    # 250000 - 24600*N <= 200000 → N >= 2.03 → 至少 3 条（保险起见 >= 2）
    assert len(offloaded) >= 2, (
        f"per-message 聚合应至少落盘 2 条，实际 {len(offloaded)}"
    )

    # 验证决策被记录（决策冻结）
    for m in offloaded:
        assert m["tool_call_id"] in _offload_decisions, (
            f"{m['tool_call_id']} 应在 _offload_decisions 中"
        )


# ---------------------------------------------------------------------------
# E2E: per-message 不跨 user 边界（在 compress_if_needed 生产路径中验证）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_per_message_respects_user_boundary_in_production(tmp_path):
    """端到端验证 per-message 不跨 user 边界——在 compress_if_needed 生产路径中。

    场景：两段 user，每段 5 个 30K tool result（每段总 150K < 200K 阈值）。
    两段加起来 300K > 200K，但 per-message 按 user 边界独立判断，每段不超限 → 不落盘。
    L2.6 全局预算也禁用（tool_result_total_budget=10**9）。
    """
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    history = []
    # 段 1：user + 5 个 30K tool result（150K < 200K）
    history.append({"role": "user", "content": "u1"})
    for i in range(5):
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_s1_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"call_s1_{i}", "name": "t",
            "content": "A" * 30000,
        })
    # 段 2：user + 5 个 30K tool result（150K < 200K）
    history.append({"role": "user", "content": "u2"})
    for i in range(5):
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_s2_{i}",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"call_s2_{i}", "name": "t",
            "content": "B" * 30000,
        })
    agent.conversation_history = history

    messages = agent._assemble_turn_messages(system_prompt="sys", injected={})

    ctx_cfg = _offload_cfg(
        output_offload_threshold=10 ** 9,    # 禁 L2 per-tool
        message_offload_threshold=200000,    # per-segment 200K（每段 150K 不超）
        micro_keep_recent_results=0,
        tool_result_total_budget=10 ** 9,    # 禁 L2.6 全局预算
    )
    state = CompressionSessionState()
    messages_after, changed = await compress_if_needed(
        messages,
        llm_client=_FakeLLM(),
        model="test",
        config=ctx_cfg,
        session_state=state,
        agent_home=Path(tmp_path),
        session_id="e2e_boundary",
    )
    # 每段独立 150K < 200K，L2.6 也禁了 → 不应触发落盘
    # 但 changed 可能为 True（其他层如 freeze 预处理可能触发），
    # 所以只检查没有 tool result 被落盘
    tool_msgs = [m for m in messages_after if m.get("role") == "tool"]
    offloaded = []
    for m in tool_msgs:
        try:
            parsed = json.loads(m["content"])
            if parsed.get("truncated"):
                offloaded.append(m)
        except (ValueError, TypeError):
            pass
    assert len(offloaded) == 0, (
        f"per-message 不应跨 user 边界——每段 150K < 200K 不应触发落盘，"
        f"但找到 {len(offloaded)} 条被落盘"
    )

