"""「上下文体检」工具：让 AI 自己查看当前对话有多重。

背景：模型在决定"要不要剪（snip）/要不要压（compact）"之前，最好先
看一眼账本——现在攒了多少条消息、估算多少 token、离自动压缩的水位线
还有多远、缓存（cache）最近命中如何。有数据再决策，比干等系统自动
触发聪明。适合长任务中段、感觉对话变长了、动手清理之前先查一下。

本文件属于工具层（tools/），被 tools/registry.py 自动发现注册；
token 估算复用 agent/context_compressor.py，缓存统计复用
agent/cache_monitor.py。
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
    """汇总当前对话的"体重报告"（消息数/token 估算/缓存状况/建议），返回 JSON。

    背景：只读查询，给模型做清理决策用。任何一步统计挂了都降级给
    粗略数字（fail-open），绝不把主对话搞崩。

    参数：
    - args：工具参数字典（本工具不收任何参数，占位满足统一签名）。
    - kwargs：运行时注入的命名上下文，本函数只用到 agent_ref
      （AIAgent 主实例，借它拿对话历史和压缩阈值配置）。

    返回：JSON 字符串，含 estimated_tokens（估算 token 数）、
    usage_percent（相对压缩水位线的百分比）、recommendation
    （ok / snip_consider / compact_now 三档建议）等字段；出错返 error。
    """
    try:
        agent = kwargs.get("agent_ref")
        if agent is None:
            return json.dumps(
                {"error": "ctx_inspect 工具需要 agent_ref（运行时注入）",
                 "error_type": "internal_error"},
                ensure_ascii=False,
            )

        # 1. 数消息条数
        history = agent.conversation_history
        msg_count = len(history)

        # 2. 估 token 数；估算器抛异常就退回"字符数除以 4"的土办法
        try:
            est_tokens = estimate_message_tokens(history)
        except Exception:
            est_tokens = sum(
                len(str(m.get("content", "") or "")) for m in history
            ) // 4

        # 3. 缓存统计；拿不到就给空表（对应字段显示 null）
        try:
            cache_stats = get_stats()
        except Exception:
            cache_stats = {}

        # 4. 从配置读压缩水位线，算出现在的用量百分比
        config = getattr(agent, "config", {}) or {}
        ctx_cfg = config.get("context", {}) if isinstance(config, dict) else {}
        llm_threshold = ctx_cfg.get("llm_compact_token_threshold", 100000)
        pct = int(est_tokens / llm_threshold * 100) if llm_threshold else 0

        # 5. 给建议：用量到 70% 就该压了，到 50% 可以考虑剪，否则不用动
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


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="ctx_inspect",
    toolset="core",
    schema=CTX_INSPECT_SCHEMA,
    handler=_handle_ctx_inspect,
    emoji="🔍",
    isConcurrencySafe=True,  # 只查不改任何东西，和其他工具同时跑也安全
)
