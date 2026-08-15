"""T1（核心机制对齐第 1 项）：压缩阈值加"单轮增长预估"（防压缩震荡）。

- estimate_turn_growth：最近 window 轮（user 边界分组）单轮 token 最大值，
  历史不足 window 轮返回保守默认（8000）
- compress_if_needed 的 L4 判定改为 est_tokens + growth >= threshold（提前触发）
- config：context.llm_compact_growth_window（默认 3）/ llm_compact_growth_default（默认 8000）
"""
import pytest

from agent.context_pipeline import (
    CompressionSessionState, compress_if_needed, estimate_turn_growth,
)


def _turn(user: str, assistant: str = "ok") -> list:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------------------
# estimate_turn_growth 纯函数
# ---------------------------------------------------------------------------

def test_growth_insufficient_history_returns_default():
    """历史不足 window 轮 → 保守默认。"""
    msgs = [{"role": "system", "content": "s"}] + _turn("hi") + _turn("there")
    assert estimate_turn_growth(msgs, window=3, default=8000) == 8000


def test_growth_large_recent_turn():
    """最近一轮很大 → growth 反映最大单轮 token（chars/4 估算）。"""
    big = "x" * 4000  # ~1000 tokens
    msgs = (
        [{"role": "system", "content": "s"}]
        + _turn("a") + _turn("b") + _turn("c")
        + [{"role": "user", "content": big}, {"role": "assistant", "content": "ok"}]
    )
    growth = estimate_turn_growth(msgs, window=3, default=100)
    assert growth >= 900  # big turn ~1000 tokens 占主导


def test_growth_small_turns():
    """3+ 轮都很小 → growth = 最近 window 轮的最大单轮（不放大到 default）。"""
    msgs = (
        [{"role": "system", "content": "s"}]
        + _turn("aaa") + _turn("bbb") + _turn("ccc")
    )
    growth = estimate_turn_growth(msgs, window=3, default=8000)
    # 每轮 ~7 chars ≈ 2 tokens，远小于 default
    assert growth < 100


def test_growth_empty_messages():
    """空/极简消息不炸。"""
    assert estimate_turn_growth([], window=3, default=8000) == 8000
    assert estimate_turn_growth([{"role": "system", "content": "s"}], window=3, default=8000) == 8000


# ---------------------------------------------------------------------------
# compress_if_needed L4 提前触发
# ---------------------------------------------------------------------------

class _FakeLLM:
    async def chat_completions(self, messages, model=None, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary", tool_calls=None),
            finish_reason="stop",
        )], usage=None)


_CFG = {
    "snip_message_threshold": 10**9,   # 禁 L1
    "micro_keep_recent_results": 3,
    "output_offload_threshold": 10**9,  # 禁 offload（避免内容被替换影响 token 估算）
    "message_offload_threshold": 0,     # 禁 L2.5
    "tool_result_total_budget": 10**9,  # 禁 L2.6
    "llm_compact_token_threshold": 100,
    "llm_compact_keep_recent": 2,
    "llm_compact_cooldown_turns": 5,
    "max_compress_attempts": 3,
    "transcript_enabled": False,
    # T1 新键：窗口 3 轮，默认增量 0（隔离测试——只看真实历史增速）
    "llm_compact_growth_window": 3,
    "llm_compact_growth_default": 0,
}


async def test_l4_triggers_early_with_growth(tmp_path):
    """est 未到阈值但 est+growth 超线 → 提前触发 L4（防震荡）。"""
    # 构造：3 轮历史，单轮 ~160 chars ≈ 40 tokens；总量 ~120+ tokens 已超阈值不行——
    # 要 est < threshold 且 est+growth >= threshold：
    # 阈值 100，前 2 轮极小（est ~10），第 3 轮（最近窗口内）大（~90 tokens）
    big = "y" * 360   # ~90 tokens
    small = "z" * 4   # ~1 token
    msgs = (
        [{"role": "system", "content": "s"}]
        + _turn(small) + _turn(small)
        + [{"role": "user", "content": big}, {"role": "assistant", "content": "r"}]
    )
    # est ≈ 90+ 一点 < 100；growth ≈ 90 → 90+90 >= 100 → 提前触发
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert state.llm_compact_count == 1, "增速大时应提前触发 L4"


async def test_l4_not_triggered_when_low_growth(tmp_path):
    """增速小（est+growth 仍在线下）→ 不提前触发。"""
    small = "z" * 8  # ~2 tokens/轮
    msgs = (
        [{"role": "system", "content": "s"}]
        + _turn(small) + _turn(small) + _turn(small)
    )
    # est ~10 tokens，growth ~2 → 12 < 100 不触发
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs, llm_client=_FakeLLM(), model="x",
        config=_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert state.llm_compact_count == 0, "低增速且未到阈值不应触发 L4"


def test_config_defaults_present():
    """DEFAULT_CONFIG 的 context 段含 T1 新键（有真实读取点）。"""
    from config import DEFAULT_CONFIG
    ctx = DEFAULT_CONFIG["context"]
    assert ctx["llm_compact_growth_window"] == 3
    assert ctx["llm_compact_growth_default"] == 8000
