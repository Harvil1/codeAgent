"""LLM 主动 snip 工具（借鉴 Claude Code SnipTool）。

让 LLM 在任务边界（如"探索阶段结束"）主动调工具剪除早期历史，
比纯自动阈值更智能。

与 compact 工具的区别：
- compact = L4 LLM 摘要（有损，调 LLM 调用，慢）
- snip = L1 占位裁剪（无损，原文落 transcript，占位提示，快）

适合：探索阶段结束、长任务中段清理、对话变长想精简（不想等自动阈值）。
"""
import json
import logging
from pathlib import Path

from agent.context_pipeline import snip_compact
from agent.transcript import snapshot_if_needed
from tools.registry import registry

logger = logging.getLogger(__name__)


SNIP_SCHEMA = {
    "name": "snip",
    "description": (
        "主动剪除早期对话历史（替换为占位摘要，原文落 transcript 可找回）。"
        "适合：探索阶段结束、长任务中段清理、对话变长想精简。"
        "\n\n注意：\n"
        "- 剪除段原文可在 .transcripts/latest.jsonl 读回（无损）\n"
        "- 最近 N 条（keep_recent）不会被剪\n"
        "- 剪完会触发 prompt cache break（前缀变了）"
        "\n\n与 compact 区别：snip 快（只占位不调 LLM），compact 慢但更智能（LLM 摘要）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "为什么剪（debug/审计用，简短一句话）。"
                    "如 '代码结构探索完毕，进入实现阶段'"
                ),
            },
            "keep_recent": {
                "type": "integer",
                "description": "保留最近 N 条不剪（默认 10）",
                "default": 10,
            },
        },
        "required": [],
    },
}


def _handle_snip(args: dict, **kwargs) -> str:
    """handler：调 snip_compact 主动剪。

    通过 kwargs 拿到 agent_ref（AIAgent 实例），直接调它的 conversation_history。

    fail-open：任何异常返 error JSON，不崩主流程。
    """
    try:
        agent = kwargs.get("agent_ref")
        if agent is None:
            return json.dumps(
                {"error": "snip 工具需要 agent_ref（运行时注入）",
                 "error_type": "internal_error"},
                ensure_ascii=False,
            )

        reason = args.get("reason") or "(LLM 未给原因)"
        keep_recent = args.get("keep_recent", 10)

        history = agent.conversation_history
        before_count = len(history)

        # 短历史不值得剪（< keep_recent * 2）
        if before_count < keep_recent * 2:
            return json.dumps({
                "snipped": False,
                "reason": f"历史太短（{before_count} 条 < keep_recent×2={keep_recent*2}），无需剪",
                "messages_count": before_count,
            }, ensure_ascii=False)

        # 读 snip 配置（snip_keep_first：保留前 N 条）
        config = getattr(agent, "config", {}) or {}
        ctx_cfg = config.get("context", {}) if isinstance(config, dict) else {}
        keep_first = ctx_cfg.get("snip_keep_first", 3)

        # snip 前主动落盘 transcript（force=True），让"无损"描述变真。
        # compress_if_needed 只在 L4 llm_compact 前调 snapshot_if_needed；
        # snip_tool 绕过 compress_if_needed 直接调 snip_compact，必须自己补这一步，
        # 否则被裁掉的中间消息原文就真丢了。
        agent_home_raw = getattr(agent, "omnimate_home", None)
        session_id = getattr(agent, "session_id", None) or ""
        transcript_enabled = ctx_cfg.get("transcript_enabled", True)
        transcript_retention = ctx_cfg.get("transcript_retention", 20)
        if transcript_enabled and agent_home_raw is not None:
            try:
                agent_home = Path(agent_home_raw)
                snapshot_if_needed(
                    history,
                    agent_home=agent_home,
                    session_id=session_id,
                    force=True,
                    enabled=True,
                    retention=transcript_retention,
                )
            except Exception as e:
                # fail-open：snapshot 失败不阻塞 snip（但消息不可找回，已在 log 警告）
                logger.warning("snip 前 transcript snapshot 失败（不阻塞 snip）: %s", e)

        # snip_compact 签名：(messages, *, keep_first, keep_last, threshold=50)
        # 我们主动调用：threshold 传 keep_first+keep_recent+2 让它一定过阈值检查
        # （否则短对话会被 threshold=50 卡住）
        # 返回 (new_messages, changed: bool)
        new_history, changed = snip_compact(
            history,
            keep_first=keep_first,
            keep_last=keep_recent,
            threshold=max(1, keep_first + keep_recent + 2),  # 让它一定过阈值
        )

        if not changed:
            return json.dumps({
                "snipped": False,
                "reason": f"snip_compact 判定无需剪（消息数 {before_count}）",
                "messages_count": before_count,
            }, ensure_ascii=False)

        # 真剪了：计算剪了多少条
        after_count = len(new_history)
        snipped_count = before_count - after_count

        # 写回 agent（主循环下一轮用新 history）
        agent.conversation_history = new_history

        # 抑制下次 cache break 误报（snip 改 messages，cache 必然 break）
        try:
            from agent.cache_monitor import notify_compaction
            notify_compaction()
        except Exception:
            pass  # fail-open，cache_monitor 不可用不崩

        logger.info(
            "LLM 主动 snip: %s（剪了 %d 条，%d → %d）",
            reason, snipped_count, before_count, after_count,
        )

        return json.dumps({
            "snipped": True,
            "reason": reason,
            "snipped_count": snipped_count,
            "remaining_count": after_count,
            "message": (
                f"已剪除 {snipped_count} 条早期消息（{before_count} → {after_count}）。"
                f"原文在 .transcripts/latest.jsonl 可找回。"
                f"最近 {keep_recent} 条已保留。"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("snip 工具异常（fail-open）")
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动）
registry.register(
    name="snip",
    toolset="core",
    schema=SNIP_SCHEMA,
    handler=_handle_snip,
    emoji="✂️",
    isConcurrencySafe=False,  # 改 conversation_history，必须串行
)
