"""LLM 上下文自查工具（借鉴 Claude Code CtxInspectTool）。

让 LLM 决策是否 snip/compact 时，先查当前 token 用量 + cache 状态 +
messages count，比被动等自动压缩更智能。

适合：长任务中段、感觉 context 变大、调 snip/compact 前先查。
"""
import json
import logging

from agent.cache_monitor import get_stats
from agent.context_compressor import estimate_message_tokens
from tools.registry import registry

logger = logging.getLogger(__name__)


CTX_INSPECT_SCHEMA = {
    "name": "ctx_inspect",
    "description": (
        "查询当前上下文状态：token 估算、消息数、prompt cache 命中、break 历史。"
        "决策是否调 snip/compact 前先查，避免被动等自动压缩。"
        "\n\n返回字段：\n"
        "- messages_count：当前对话消息数\n"
        "- estimated_tokens：粗略 token 估算（字符数/3）\n"
        "- usage_percent：相对 llm_compact 阈值的百分比\n"
        "- recommendation：ok / snip_consider / compact_now\n"
        "- cache_last_read：最近一次 prompt cache read tokens\n"
        "- cache_total_breaks：本会话累计 cache break 次数"
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _handle_ctx_inspect(args: dict, **kwargs) -> str:
    """读 agent + cache_monitor 状态，fail-open。"""
    try:
        agent = kwargs.get("agent_ref")
        if agent is None:
            return json.dumps(
                {"error": "ctx_inspect 工具需要 agent_ref（运行时注入）",
                 "error_type": "internal_error"},
                ensure_ascii=False,
            )

        # 1. 消息数
        history = agent.conversation_history
        msg_count = len(history)

        # 2. token 估算（fail-open：异常走 fallback len//4）
        try:
            est_tokens = estimate_message_tokens(history)
        except Exception:
            est_tokens = sum(
                len(str(m.get("content", "") or "")) for m in history
            ) // 4

        # 3. cache_monitor 统计（fail-open：异常返 None 字段）
        try:
            cache_stats = get_stats()
        except Exception:
            cache_stats = {}

        # 4. llm_compact 阈值 + usage 百分比
        config = getattr(agent, "config", {}) or {}
        ctx_cfg = config.get("context", {}) if isinstance(config, dict) else {}
        llm_threshold = ctx_cfg.get("llm_compact_token_threshold", 100000)
        pct = int(est_tokens / llm_threshold * 100) if llm_threshold else 0

        # 5. recommendation（阈值：>= 70% compact_now，>= 50% snip_consider，其他 ok）
        if pct >= 70:
            recommendation = "compact_now"
        elif pct >= 50:
            recommendation = "snip_consider"
        else:
            recommendation = "ok"

        return json.dumps({
            "messages_count": msg_count,
            "estimated_tokens": est_tokens,
            "llm_compact_threshold": llm_threshold,
            "usage_percent": pct,
            "cache_last_read": cache_stats.get("last_cache_read"),
            "cache_total_breaks": cache_stats.get("total_breaks"),
            "cache_last_break": cache_stats.get("last_break"),
            "recommendation": recommendation,
            "message": (
                f"当前 {msg_count} 条消息，估算 {est_tokens} tokens"
                f"（{pct}% of compact 阈值 {llm_threshold}）。"
                f"建议：{recommendation}。"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("ctx_inspect 工具异常（fail-open）")
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动）
registry.register(
    name="ctx_inspect",
    toolset="core",
    schema=CTX_INSPECT_SCHEMA,
    handler=_handle_ctx_inspect,
    emoji="🔍",
    isConcurrencySafe=True,  # 只读，并发安全
)
