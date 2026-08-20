"""各 LLM 模型的价格表 + 花费估算。

在项目里的位置：一张静态参考表，被 usage_tracker（用量统计）调用，
把 token 数换算成美元花费。

价格单位：美元 / 百万 tokens（$/M tokens）。
数据来源：各家官方价格页（2026 年 7 月行情）。

每条价格记录是一个四元组，按顺序是：
    (input_cache_miss, input_cache_hit, input_cache_write, output)
    （输入·缓存未命中，输入·缓存命中，输入·缓存写入，输出）

某项价格查不到时用 0.0 占位（这样估算结果会偏低，宁可少报不瞎报）。
"""

from typing import Dict, Tuple, Optional


# 结构：provider → model → (in_miss, in_hit, in_write, out)，单位都是 $/M tokens
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
        # OpenRouter 只是中转，各模型价格跟原 provider 走；这里故意不放
        # 条目——匹配到 openrouter/ 前缀时由 _normalize_openrouter 拆出
        # 真实 provider 再查价
    },
}


def _normalize_openrouter(provider: str, model: str) -> Tuple[str, str]:
    """把 openrouter/<provider>/<model> 形式的名字拆开，返回 (真实provider, 真实model)。

    背景：经 OpenRouter 中转调用时，模型名里带着完整路由路径，直接查
    价格表查不到，得先拆出真正的厂家和型号。

    参数：
        provider —— 厂家名（如 "openrouter"）
        model    —— 模型名（可能是 "openrouter/deepseek/deepseek-chat" 这种带路径的）

    返回：拆好后的 (provider, model)。不是 openrouter、或路径段不够拆时，
    原样返回不改动。
    """
    if provider != "openrouter" or "/" not in model:
        return provider, model
    parts = model.split("/", 2)
    if len(parts) >= 3:
        # ["openrouter", "deepseek", "deepseek-chat"] → 中间段是真实厂家
        return parts[1], parts[2]
    if len(parts) == 2:
        # ["openrouter", "gpt-4o"]（没写中间厂家，只能拿后半当模型名）
        return provider, parts[1]
    return provider, model


def get_pricing(provider: str, model: str) -> Optional[Tuple[float, float, float, float]]:
    """查某个模型的价格四元组。

    参数：
        provider —— 厂家名（如 "deepseek"，大小写不敏感）
        model    —— 模型名（如 "deepseek-chat"）

    返回：(输入未命中, 输入命中, 缓存写入, 输出) 四个单价；查不到返回 None。

    匹配策略（按顺序试）：
    1. 精确匹配 provider → model
    2. 模型名前缀匹配（如 "deepseek-chat-0324" 这种带版本后缀的 → 按
       "deepseek-chat" 查到）
    3. OpenRouter 路径（"openrouter/deepseek/..."）拆出真实 provider 再查
    4. 都不行返回 None
    """
    p = (provider or "").lower()
    m = (model or "").lower()

    p, m = _normalize_openrouter(p, m)

    provider_table = PRICING.get(p)
    if not provider_table:
        return None

    # 先试精确名
    if m in provider_table:
        return provider_table[m]

    # 再试前缀（模型名常带版本后缀，如 deepseek-chat-0324）
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
    """把一次 LLM 调用的 token 数换算成美元花费。

    参数：
        provider              —— 厂家名（用于查价表）
        model                 —— 模型名
        prompt_tokens         —— 本次输入总 token 数（含缓存命中部分）
        completion_tokens     —— 本次输出 token 数
        cache_read_tokens     —— 其中命中缓存的部分（便宜）
        cache_creation_tokens —— 其中写缓存的部分（贵）

    返回：字典，形如：
        {
            "cost_usd": 总花费,
            "breakdown": {
                "input": float,        # 缓存未命中部分的输入花费
                "cache_hit": float,    # 命中缓存（便宜）
                "cache_write": float,  # 写入缓存（贵）
                "output": float,
            },
        }
    价格表里查不到该模型时返回 None。
    """
    pricing = get_pricing(provider, model)
    if pricing is None:
        return None

    in_miss_price, in_hit_price, in_write_price, out_price = pricing

    # prompt_tokens 是总数：把命中/写入缓存的部分扣掉，剩下的才算全价输入
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
