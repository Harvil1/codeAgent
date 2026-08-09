"""第 3 轮 bug 修复测试：功能失效 + LLM 韧性。"""
import inspect
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# S6: memory_store._now_iso 必须返回 UTC（naive local 会让 curator age 偏差）
# ---------------------------------------------------------------------------

def test_memory_store_now_iso_is_utc():
    """S6 fix: _now_iso() 返回的字符串解析后应含 timezone info（UTC）。"""
    from agent.memory_store import _now_iso
    iso = _now_iso()
    # 解析
    parsed = datetime.fromisoformat(iso)
    assert parsed.tzinfo is not None, (
        f"_now_iso 应返回 timezone-aware（UTC），实际: {iso}（naive）"
    )
    # 应是 UTC（offset 0）
    if hasattr(parsed, "utcoffset"):
        assert parsed.utcoffset().total_seconds() == 0, (
            f"_now_iso 应是 UTC，实际 offset: {parsed.utcoffset()}"
        )


# ---------------------------------------------------------------------------
# S5: memory_curator.apply_automatic_transitions 应走 store.list_all()
# ---------------------------------------------------------------------------

def test_memory_curator_uses_standard_interface_not_glob_md():
    """S5 fix: curator 不应 glob('*.md')，应用 store 标准接口（list_all/snapshot）。

    bug：之前扫 *.md 但 MemoryStore 写 .jsonl，扫描永远空。
    静态验证：源码不含 memory_dir.glob("*.md")。
    """
    from agent import memory_curator
    src = inspect.getsource(memory_curator.apply_automatic_transitions)
    # 不应再 glob *.md
    assert 'glob("*.md")' not in src and "glob('*.md')" not in src, (
        f"curator 不应 glob('*.md')（dead code，实际格式是 .jsonl），实际:\n{src}"
    )
    # 应该走标准接口（list_all 或 snapshot_for_prompt 或 store.delete）
    assert "list_all" in src or "snapshot" in src or "store." in src, (
        f"curator 应调 store 标准接口，实际:\n{src}"
    )


def test_memory_curator_handles_empty_dir(tmp_path):
    """回归：空 memory_dir 不崩，返回全 0 counts。"""
    from agent.memory_curator import apply_automatic_transitions
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    counts = apply_automatic_transitions(memory_dir)
    assert counts["checked"] == 0
    assert counts["archived"] == 0


# ---------------------------------------------------------------------------
# S9: 主循环每轮重新 drain（不只循环前 drain 一次）
# ---------------------------------------------------------------------------

def test_run_conversation_redrains_each_iteration():
    """S9 fix: run_conversation 在 _assemble_turn_messages 前重新 drain。

    bug：之前只在循环前 drain 一次，多轮 tool_calls 中途新到的 bg/cron/team
    消息进不去 LLM 上下文。

    静态验证：_drain_injected_messages 紧贴 _assemble_turn_messages 之前（每轮重 drain）。
    """
    from agent import AIAgent
    src = inspect.getsource(AIAgent.run_conversation)
    # 找 _drain_injected_messages 和 _assemble_turn_messages 的位置
    drain_pos = src.find("self._drain_injected_messages()")
    assemble_pos = src.find("self._assemble_turn_messages(")
    assert drain_pos != -1 and assemble_pos != -1, (
        f"run_conversation 应含 drain 和 assemble 调用"
    )
    # drain 必须在 assemble 之前（同一轮内先 drain 再 assemble）
    assert drain_pos < assemble_pos, (
        f"drain 应在 assemble 之前（每轮重新 drain），位置 drain={drain_pos} assemble={assemble_pos}"
    )
    # 关键：drain 不能只在循环外（line 688 旧逻辑）——应在 while 循环体内
    # 检查 drain 出现在 while 关键字之后
    while_pos = src.find("while ")
    assert while_pos != -1, "应含 while 循环"
    assert drain_pos > while_pos, (
        f"drain 应在 while 循环内（每轮调用），drain={drain_pos} while={while_pos}"
    )


# ---------------------------------------------------------------------------
# X6: call_with_retry max_retries=0 不能 raise None
# ---------------------------------------------------------------------------

async def test_call_with_retry_rejects_zero_retries():
    """X6 fix: max_retries=0 应抛 ValueError 或 RuntimeError，不能 raise None。"""
    from agent.llm_retry import call_with_retry

    fake_client = MagicMock()

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        await call_with_retry(fake_client, messages=[], max_retries=0)
    # 不能是 TypeError（raise None 导致的）
    assert not isinstance(exc_info.value, TypeError), (
        f"max_retries=0 不应导致 TypeError（raise None），实际: {exc_info.value}"
    )


async def test_call_with_retry_normal_path_still_works():
    """回归：max_retries>0 时正常路径仍工作。"""
    from agent.llm_retry import call_with_retry
    from unittest.mock import AsyncMock

    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_client.chat_completions = AsyncMock(return_value=fake_response)

    result = await call_with_retry(
        fake_client,
        messages=[{"role": "user", "content": "hi"}],
        max_retries=3,
    )
    assert result is fake_response


# ---------------------------------------------------------------------------
# X7: _compute_backoff 必须有上限（防止数小时 sleep）
# ---------------------------------------------------------------------------

def test_compute_backoff_capped_at_maximum():
    """X7 fix: _compute_backoff 应有 MAX_BACKOFF 上限（默认 60s 或类似）。"""
    from agent.llm_retry import _compute_backoff

    # 大 retry_after（1 小时）应被封顶
    backoff = _compute_backoff(attempt=0, initial_backoff=1.0, retry_after=3600, jitter_ratio=0)
    assert backoff <= 120, (
        f"backoff 应被 MAX_BACKOFF 封顶（≤120s），实际: {backoff}s（用户会以为 agent 挂了）"
    )

    # 大 attempt（2^20 = 12 天）也应封顶
    backoff2 = _compute_backoff(attempt=20, initial_backoff=1.0, retry_after=None, jitter_ratio=0)
    assert backoff2 <= 120, (
        f"大 attempt 的 backoff 应封顶，实际: {backoff2}s"
    )


def test_compute_backoff_normal_case_unchanged():
    """回归：正常 attempt + 小 retry_after 计算正确。"""
    from agent.llm_retry import _compute_backoff

    # attempt=2, initial=1, 无 retry_after, 无 jitter → 2^2 = 4
    backoff = _compute_backoff(attempt=2, initial_backoff=1.0, retry_after=None, jitter_ratio=0)
    assert backoff == 4.0

    # retry_after 优先于 attempt 计算（小值不封顶）
    backoff2 = _compute_backoff(attempt=5, initial_backoff=1.0, retry_after=2.0, jitter_ratio=0)
    assert backoff2 == 2.0
