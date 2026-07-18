"""LLM 价格表 + 成本估算。

价格单位：美元 / 百万 tokens（$/M tokens）。
数据来源：各家官方价格页（2026-07 行情）。

每条记录：
    (input_cache_miss, input_cache_hit, input_cache_write, output)

如果某模型的某项价格未知，用 0.0 占位（估算会偏低）。
"""

from typing import Dict, Tuple, Optional


# provider → model → (in_miss, in_hit, in_write, out)  全部 $/M tokens
PRICING: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "deepseek": {
        # https://api-docs.deepseek.com/quick_start/pricing
        "deepseek-chat":      (0.27, 0.07, 0.27 * 1.1, 1.10),
        "deepseek-reasoner":  (0.55, 0.14, 0.55 * 1.1, 2.19),
    },
    "openai": {
        # https://openai.com/api/pricing/
        "gpt-4o":             (2.50, 1.25, 2.50 * 1.25, 10.00),
        "gpt-4o-mini":        (0.15, 0.075, 0.15 * 1.25, 0.60),
        "gpt-4.1":            (2.00, 0.50, 2.00 * 1.25, 8.00),
        "gpt-4.1-mini":       (0.40, 0.10, 0.40 * 1.25, 1.60),
        "o1":                 (15.00, 7.50, 15.00 * 1.25, 60.00),
        "o1-mini":            (1.10, 0.55, 1.10 * 1.25, 4.40),
        "o3-mini":            (1.10, 0.55, 1.10 * 1.25, 4.40),
    },
    "anthropic": {
        # https://www.anthropic.com/api
        "claude-opus-4":      (15.00, 1.50, 18.75, 75.00),
        "claude-sonnet-4":    (3.00, 0.30, 3.75, 15.00),
        "claude-haiku-4":     (1.00, 0.10, 1.25, 5.00),
        "claude-3-5-sonnet":  (3.00, 0.30, 3.75, 15.00),
        "claude-3-5-haiku":   (0.80, 0.08, 1.00, 4.00),
    },
    "openrouter": {
        # OpenRouter 上各模型价格随源 provider；用 null 表示"按模型名前缀匹配后回退"
        # 这里不放具体条目，让调用方在匹配 openrouter/前缀时拆出真实 provider
    },
}


def get_pricing(provider: str, model: str) -> Optional[Tuple[float, float, float, float]]:
    """查询 (provider, model) 的价格。

    匹配策略：
    1. 精确匹配 provider → model
    2. 模型名前缀匹配（如 "deepseek-chat-0324" → "deepseek-chat"）
    3. OpenRouter 前缀（"openrouter/deepseek/..."）拆出真实 provider 再匹配
    4. 找不到返回 None
    """
    p = (provider or "").lower()
    m = (model or "").lower()

    # 处理 openrouter/<provider>/<model> 形式
    if p == "openrouter" and "/" in m:
        parts = m.split("/", 2)
        if len(parts) >= 3:
            # ["openrouter", "deepseek", "deepseek-chat"]
            p = parts[1]
            m = parts[2]
        elif len(parts) == 2:
            # ["openrouter", "gpt-4o"]（无中间 provider，用第一个当 model 名）
            m = parts[1]

    provider_table = PRICING.get(p)
    if not provider_table:
        return None

    # 精确
    if m in provider_table:
        return provider_table[m]

    # 前缀匹配（模型名可能带版本后缀，如 deepseek-chat-0324）
    for prefix, price in provider_table.items():
        if m.startswith(prefix):
            return price

    return None


def estimate_cost_usd(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> Optional[dict]:
    """估算美元成本。

    返回 dict：
        {
            "cost_usd": float,
            "breakdown": {
                "input": float,        # cache miss 部分
                "cache_hit": float,    # cache 命中（便宜）
                "cache_write": float,  # cache 写入（贵）
                "output": float,
            },
            "pricing_source": str,     # "exact" / "prefix" / "unknown"
        }

    找不到价格返回 None。
    """
    pricing = get_pricing(provider, model)
    if pricing is None:
        return None

    in_miss_price, in_hit_price, in_write_price, out_price = pricing

    # cache hit / write 部分从 prompt_tokens 里分出去，剩余算 cache miss
    cache_miss_tokens = max(0, prompt_tokens - cache_read_tokens - cache_creation_tokens)

    cost_input = cache_miss_tokens * in_miss_price / 1_000_000
    cost_cache_hit = cache_read_tokens * in_hit_price / 1_000_000
    cost_cache_write = cache_creation_tokens * in_write_price / 1_000_000
    cost_output = completion_tokens * out_price / 1_000_000

    return {
        "cost_usd": cost_input + cost_cache_hit + cost_cache_write + cost_output,
        "breakdown": {
            "input": cost_input,
            "cache_hit": cost_cache_hit,
            "cache_write": cost_cache_write,
            "output": cost_output,
        },
    }
