"""agent/pricing.py 测试。"""

import pytest

from agent.pricing import get_pricing, estimate_cost_usd, PRICING


# ---------------------------------------------------------------------------
# get_pricing
# ---------------------------------------------------------------------------

def test_get_pricing_exact_match():
    """精确匹配 provider + model。"""
    p = get_pricing("deepseek", "deepseek-chat")
    assert p is not None
    assert p[0] == 0.27  # in_miss
    assert p[3] == 1.10  # out


def test_get_pricing_case_insensitive():
    """provider 和 model 大小写不敏感。"""
    p = get_pricing("DeepSeek", "DEEPSEEK-CHAT")
    assert p is not None


def test_get_pricing_prefix_match_for_versioned_model():
    """带版本后缀的模型名走前缀匹配。"""
    p = get_pricing("deepseek", "deepseek-chat-0324")
    assert p is not None
    assert p == PRICING["deepseek"]["deepseek-chat"]


def test_get_pricing_openrouter_prefix():
    """openrouter/<provider>/<model> 形式拆出真实 provider。"""
    p = get_pricing("openrouter", "openrouter/deepseek/deepseek-chat")
    assert p is not None
    assert p[0] == 0.27


def test_get_pricing_unknown_returns_none():
    """未收录的模型返回 None。"""
    assert get_pricing("made-up", "xxx") is None
    assert get_pricing("deepseek", "not-a-real-model") is None
    assert get_pricing("", "") is None


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------

def test_estimate_cost_simple_no_cache():
    """无 cache 命中时全部按 input 价算。"""
    # DeepSeek chat: $0.27/M input, $1.10/M output
    est = estimate_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=1_000_000,  # 1M
        completion_tokens=500_000,  # 0.5M
    )
    assert est is not None
    # 1M * 0.27 + 0.5M * 1.10 = 0.27 + 0.55 = 0.82
    assert abs(est["cost_usd"] - 0.82) < 0.0001
    assert abs(est["breakdown"]["input"] - 0.27) < 0.0001
    assert abs(est["breakdown"]["output"] - 0.55) < 0.0001
    assert est["breakdown"]["cache_hit"] == 0.0
    assert est["breakdown"]["cache_write"] == 0.0


def test_estimate_cost_with_cache_hit():
    """cache 命中按便宜价算。"""
    # DeepSeek chat: in_miss=0.27, in_hit=0.07
    est = estimate_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_tokens=800_000,  # 80% 命中
    )
    assert est is not None
    # cache miss 部分：200K * 0.27 / 1M = 0.054
    # cache hit 部分：800K * 0.07 / 1M = 0.056
    # 合计 input: 0.11
    assert abs(est["breakdown"]["input"] - 0.054) < 0.0001
    assert abs(est["breakdown"]["cache_hit"] - 0.056) < 0.0001
    assert est["cost_usd"] < 0.27  # 比 0% 命中便宜


def test_estimate_cost_with_cache_write():
    """cache 写入按 1.25x 贵价算。"""
    est = estimate_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_creation_tokens=500_000,
    )
    # cache miss 部分：500K * 0.27 / 1M = 0.135
    # cache write 部分：500K * (0.27*1.1) / 1M = 0.1485
    assert est is not None
    assert abs(est["breakdown"]["input"] - 0.135) < 0.0001
    assert abs(est["breakdown"]["cache_write"] - 0.1485) < 0.0001


def test_estimate_cost_unknown_model_returns_none():
    est = estimate_cost_usd(
        provider="xxx",
        model="yyy",
        prompt_tokens=100,
        completion_tokens=100,
    )
    assert est is None


def test_estimate_cost_zero_tokens():
    """0 tokens 应该返回 $0。"""
    est = estimate_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=0,
        completion_tokens=0,
    )
    assert est is not None
    assert est["cost_usd"] == 0.0


def test_estimate_cost_cache_more_than_prompt_clamped():
    """cache tokens 超过 prompt_tokens 时不应出负数。"""
    est = estimate_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=100,
        completion_tokens=0,
        cache_read_tokens=200,  # 异常：比 prompt 还多
        cache_creation_tokens=100,
    )
    assert est is not None
    assert est["breakdown"]["input"] >= 0  # 不应出负数
