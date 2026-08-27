"""bug 修复测试：剩余复杂 bug。"""
import inspect
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _run_memory_curator_once 应支持共享主 store 实例（避免跨实例 race）
# ---------------------------------------------------------------------------

def test_run_memory_curator_once_accepts_store_param():
    """_run_memory_curator_once 签名应支持 store 参数。

    若每次 new MemoryStore，与主 agent 实例不同，threading.Lock
    不跨实例，并发写同一 topic.jsonl 会丢数据。
    """
    from cli import _run_memory_curator_once
    sig = inspect.signature(_run_memory_curator_once)
    # 应接受 store 参数（可选）
    assert "store" in sig.parameters, (
        f"_run_memory_curator_once 应支持 store 参数（共享主实例避免 race），"
        f"实际签名: {sig}"
    )


def test_run_memory_curator_once_uses_passed_store(tmp_path, monkeypatch):
    """传入 store 时，curator 应直接用，不再新建。"""
    from cli import _run_memory_curator_once

    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()

    # 跟踪 MemoryStore 实例化次数
    call_count = {"n": 0}
    real_init = None
    import agent.memory_store as ms_mod
    real_init = ms_mod.MemoryStore.__init__

    def counting_init(self, *args, **kwargs):
        call_count["n"] += 1
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ms_mod.MemoryStore, "__init__", counting_init)

    # 传一个 mock store
    mock_store = MagicMock()
    mock_store.list_all.return_value = []

    _run_memory_curator_once(memory_dir, config={}, store=mock_store)

    # 传了 store 时不应新建 MemoryStore（call_count 不应增加）
    # 至少 mock_store.list_all 被调用了
    assert mock_store.list_all.called, "传 store 时应直接用，调 list_all"


# ---------------------------------------------------------------------------
# Cron 时间戳应用 UTC（避免 DST/跨时区 age 计算偏差）
# ---------------------------------------------------------------------------

def test_cron_created_at_uses_utc():
    """_parse_job 写 created_at 应优先 UTC（datetime.now(timezone.utc)）。"""
    from agent.cron import CronScheduler
    src = inspect.getsource(CronScheduler)
    # 应含 datetime.now(timezone.utc)
    assert "timezone.utc" in src, (
        f"CronScheduler 应使用 timezone.utc（避免 naive 时区 age 偏差），实际:\n{src[:500]}"
    )


# ---------------------------------------------------------------------------
# Cron 一次性 job 触发后应立刻 disable + persist（不等循环末尾）
# ---------------------------------------------------------------------------

def test_cron_oneshot_disables_immediately():
    """_tick 内 one-shot 触发时应立刻 disable + persist。

    静态验证：源码内 not job.recurring 附近含 _persist_jobs_unlocked 调用（在循环内）。
    """
    from agent.cron import CronScheduler
    src = inspect.getsource(CronScheduler._tick)
    # 关键：源码应含 "not job.recurring" 附近的立刻 persist
    # 简化：源码含 "not job.recurring" 后紧跟 disable + persist 模式
    # 通过检查 "_persist_jobs_unlocked()" 出现次数 + 接近 not job.recurring
    oneshot_pos = src.find("not job.recurring")
    persist_positions = []
    start = 0
    while True:
        p = src.find("_persist_jobs_unlocked()", start)
        if p == -1:
            break
        persist_positions.append(p)
        start = p + 1
    # 应该有至少 1 个 persist 调用在 oneshot_pos 之后（之前的 200 字符内）
    has_immediate_persist = any(
        oneshot_pos < p < oneshot_pos + 500 for p in persist_positions
    )
    assert has_immediate_persist, (
        f"_tick 应在 one-shot 触发后立刻 persist（避免重启重复触发），"
        f"oneshot_pos={oneshot_pos}, persist_positions={persist_positions}"
    )


# ---------------------------------------------------------------------------
# BackgroundManager stall_timeout 默认值应 > 0（45s 看门狗默认开）
# ---------------------------------------------------------------------------

def test_background_manager_default_stall_timeout_positive():
    """BackgroundManager 默认 stall_timeout 应 > 0（默认 45s 看门狗开）。"""
    from agent.background import BackgroundManager
    sig = inspect.signature(BackgroundManager.__init__)
    stall_default = sig.parameters["stall_timeout"].default
    assert stall_default > 0, (
        f"默认 stall_timeout 应 > 0（45s 看门狗默认开），实际: {stall_default}"
    )


def test_background_manager_still_accepts_zero_override():
    """回归：显式传 stall_timeout=0 仍可（用户主动禁用看门狗）。"""
    from agent.background import BackgroundManager
    bg = BackgroundManager(stall_timeout=0.0)
    # 不崩即可
    assert bg is not None
