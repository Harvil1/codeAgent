"""「剪掉旧对话」工具：让 AI 自己决定什么时候裁剪历史。

背景：对话越长，每次调模型的 token 账单越贵。项目里已有一套"超过阈值就
自动压缩"的机制，但那是死板的水位线；这个工具让模型在自然的任务分界点
（比如"代码结构摸清楚了，准备动手写"）自己动手剪掉前面对话的细节。

和 compact 工具的分工（一个快一个聪明）：
- compact = 让模型把旧对话读一遍总结成摘要（有损、要再调一次模型、慢）
- snip = 直接把旧消息换成一句占位提示（原文完整存进 transcript 备查，
  算"无损"；不调模型、快）

本文件属于工具层（tools/），被 tools/registry.py 自动发现注册，
底层裁剪逻辑复用 agent/context_pipeline.py 的 snip_compact。
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
    """把早期对话消息剪掉、换成占位提示，返回执行结果 JSON。

    背景：模型主动调这个工具来瘦身对话历史。任何一步出问题都不能
    把主对话搞崩（fail-open：出错就返回带 error 的 JSON，不抛异常）。

    参数：
    - args：工具参数字典。reason 是模型给的理由（只用于日志和返回信息）；
      keep_recent 是"最近 N 条不许剪"的保护条数。
    - kwargs：运行时注入的命名上下文，本函数只用到 agent_ref
      （AIAgent 主实例，借它拿到对话历史、配置和主目录）。

    返回：JSON 字符串。成功时 snipped=True 并带剪了多少条；
    不用剪/出错时 snipped=False 或 error 字段。
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

        # 剪完至少还得剩下"保护条数"这么多消息，太短的对话剪了没意义还白折腾
        if before_count < keep_recent * 2:
            return json.dumps({
                "snipped": False,
                "reason": f"历史太短（{before_count} 条 < keep_recent×2={keep_recent*2}），无需剪",
                "messages_count": before_count,
            }, ensure_ascii=False)

        # 从配置读"开头几条也不许剪"（snip_keep_first，默认 3）
        config = getattr(agent, "config", {}) or {}
        ctx_cfg = config.get("context", {}) if isinstance(config, dict) else {}
        keep_first = ctx_cfg.get("snip_keep_first", 3)

        # 历史踩坑（无损承诺兑现）：剪之前必须先把对话原文存档到 transcript
        # （force=True 强制存）。项目里的自动压缩管线（compress_if_needed）
        # 只在自己调 LLM 摘要前会存档；本工具走的是捷径、绕过了它，
        # 所以必须自己补这一步——否则被剪掉的消息原文就真找不回来了。
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
                # 存档失败也不拦着剪（fail-open），代价是这批消息找不回来——记条警告日志
                logger.warning("snip 前 transcript snapshot 失败（不阻塞 snip）: %s", e)

        # 底层函数 snip_compact 有个"消息太少就不剪"的自检门槛（threshold）。
        # 这里是模型主动要剪，不该被默认门槛 50 拦住，所以把门槛算成
        # "开头保护 + 结尾保护 + 2"，保证只要走到这就一定剪得动。
        # 它返回 (新消息列表, 是否真的剪了)。
        new_history, changed = snip_compact(
            history,
            keep_first=keep_first,
            keep_last=keep_recent,
            threshold=max(1, keep_first + keep_recent + 2),  # 见上：保证一定过门槛
        )

        if not changed:
            return json.dumps({
                "snipped": False,
                "reason": f"snip_compact 判定无需剪（消息数 {before_count}）",
                "messages_count": before_count,
            }, ensure_ascii=False)

        # 走到这说明真剪了：算一下剪掉多少条，写回给主循环下一轮用
        after_count = len(new_history)
        snipped_count = before_count - after_count

        # 写回 agent（主循环下一轮用新 history）
        agent.conversation_history = new_history

        # 剪了消息等于把对话开头换了，模型侧的前缀缓存必然整体失效
        # （cache break）。提前打声招呼，免得缓存监控下次误报"莫名失效"。
        try:
            from agent.cache_monitor import notify_compaction
            notify_compaction()
        except Exception:
            pass  # 打招呼失败就算了，不能为一个统计模块把主流程搞崩

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


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="snip",
    toolset="core",
    schema=SNIP_SCHEMA,
    handler=_handle_snip,
    emoji="✂️",
    isConcurrencySafe=False,  # 会直接改对话历史这种共享状态，和其他工具并发跑会互相踩，只能排队执行
)
