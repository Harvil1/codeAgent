# agent/context_pipeline.py
"""分层上下文压缩管线（对话太长时逐层瘦身，裁掉或浓缩旧内容，腾出空间）。

本文件是压缩的总调度台，按「便宜到贵」的顺序套多层手段：
L1 snip（消息条数太多时裁掉中间一段，留头留尾）→ L2 micro（单条工具结果太大时
把原文存到磁盘、原地留个缩短版+文件指针）→ L2.5 按段聚合落盘 → L3.5
contextCollapse（按 token 占用比例把早期对话整段折叠成占位提示）→ L4 llm
（前面都不够时才花钱调 LLM 把旧对话写成有损摘要），外加 reactive 紧急通道
（API 报「对话超长」时立刻保命截断）。

项目里的位置：被 agent 主循环（AIAgent）每轮调用；
干活的零件来自 agent/context_compressor.py，落盘能力来自 agent/output_offload.py。
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from agent.context_compressor import (
    _summarize_conversation, _fix_tool_call_pairs, estimate_message_tokens,
    reset_compact_circuit_breaker, extract_summary_anchor, _get_model_max_tokens,
)
from agent.transcript import snapshot_if_needed

logger = logging.getLogger(__name__)


def _extract_anchor_from_slice(messages_slice: list) -> tuple:
    """从将被摘要的消息段里找上一次压缩的摘要，提取锚定段。

    多次压缩时旧摘要会被送进 LLM 重摘要，"逐字保留"的文件路径/错误
    消息/用户原话在第二代摘要里照样被改写丢失（代际损耗）。这里把
    旧摘要的三段关键内容原文提出来，由调用方原样拼进新摘要头部。

    返回：(锚定文本, 给摘要 LLM 的提示语)；没有旧摘要返回 ("", "")。
    """
    for m in messages_slice:
        c = m.get("content")
        if not isinstance(c, str) or not c or len(c) > 60000:
            continue
        # 旧摘要 placeholder 的特征标记（出现在消息开头附近）
        if "[之前的对话已自动总结]" not in c and "[对话摘要" not in c[:300]:
            continue
        anchor = extract_summary_anchor(c)
        if anchor:
            note = (
                "对话材料里含上一次压缩的摘要：其中 Files and Code Sections / "
                "Errors and fixes / All user messages 三段已原样拼在最终摘要头部，"
                "你本次输出的对应三段不要重复罗列这些旧内容（可写「见前次锚定段」），"
                "专注新增内容。"
            )
            return anchor, note
    return "", ""


def _prepend_anchor(summary: str, anchor: str) -> str:
    """把锚定段拼在摘要头部（带醒目标题，下次压缩还能识别提取）。"""
    if not anchor:
        return summary
    return (
        "### 前次压缩锚定保留（逐字未变，勿改写）\n\n"
        f"{anchor}\n\n"
        f"{summary}"
    )


def _split_system(messages: list) -> Tuple[Optional[dict], list]:
    """把开头那条 system 消息（发给 LLM 的角色设定，整场对话不变）单独摘出来。

    为什么必须摘出来：压缩层都不能动 system——动了会打穿 prompt cache
    （服务商按消息前缀复用的计费缓存，前缀一变就得全量重算，成本翻倍）。

    参数：
        messages：完整消息列表
    返回：(system 消息或 None, 其余消息组成的列表)。
    """
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages


def _reassemble(system: Optional[dict], conv: list) -> list:
    """把 system 消息和其余对话拼回一个完整列表（_split_system 的逆操作）。

    参数：
        system：system 消息（可能是 None，None 就不拼）
        conv：其余对话消息列表
    返回：拼好的完整消息列表。
    """
    return [system, *conv] if system else conv


def _has_tool_calls(msg: dict) -> bool:
    """判断某条 assistant 消息里有没有工具调用请求（tool_calls 字段非空）。"""
    tcs = msg.get("tool_calls")
    return bool(tcs)


def _is_tool_result(msg: dict) -> bool:
    """判断某条消息是不是工具执行结果的回传（role 等于 'tool'）。"""
    return msg.get("role") == "tool"


def time_based_clear_old_tool_results(
    messages: list, config: dict, agent_home=None,
) -> Tuple[list, bool]:
    """按时间清旧工具结果：距最后一次助手回复超过 N 分钟没动静，就把更早的工具结果内容清空——用户放着一两个小时没说话时，中间的工具输出基本不会再被用到。

    清空不是半丢失：agent_home 在场时，清空前先 maybe_offload 把原文
    落盘到 .task_outputs/，占位从一句纯文本升级成 offload JSON
    （开头预览 + full_at 文件指针）——模型想看全文可以自己读回。
    agent_home=None（或落盘失败）退回纯文本占位「[Old tool result
    content cleared]」（老行为，测试兼容路径）。

    在 compress_if_needed 的流水线里最先跑（不看 token 超没超），排在 L1 之前。

    参数：
        messages：完整消息列表
        config：context 配置子字典（由 compress_if_needed 从 self.config 的 "context" 段传入）
        agent_home：CodeAgent 数据目录（~/.codeAgent）；None = 不落盘、退纯文本占位
    返回：(消息列表, 本次是否真的清了内容)。第二个值让上层正确统计「这次有没有改动」。

    行为（大白话版）：
      1. 开关关着（enabled=False）→ 原样返回
      2. 找最后一条 assistant 消息的位置
      3. 读它的时间戳（_timestamp）；没有就放弃
      4. 距现在不足 gap_minutes 分钟 → 不动
      5. 超时：把这条 assistant 之前的所有工具结果清空内容，
         但最近 keep_recent 条保留不清

    fail-open：出异常不影响主流程，原样返回。
    幂等：已经清过的（内容是纯文本占位、或已带 full_at 指针的
    offload 占位）再跑一遍也不重复计「有变化」。
    """
    CLEARED_MARK = "[Old tool result content cleared]"
    try:
        # config 是 context 配置子字典（由 compress_if_needed 从 self.config.get("context", {}) 传入），
        # 按扁平 key 读，跟 snip_compact / offload_large_tool_results 的读法保持一致
        ctx_cfg = config if isinstance(config, dict) else {}
        enabled = ctx_cfg.get("time_based_mc_enabled", True)
        if not enabled:
            return messages, False

        gap_minutes = ctx_cfg.get("time_based_mc_gap_minutes", 60)
        keep_recent = ctx_cfg.get("time_based_mc_keep_recent", 5)

        # 找最后一条 assistant 消息
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx < 0:
            return messages, False

        last_ts = messages[last_assistant_idx].get("_timestamp")
        if not last_ts:
            return messages, False

        elapsed_min = (time.time() - last_ts) / 60
        if elapsed_min < gap_minutes:
            return messages, False

        # 超时：找 last_assistant_idx 之前的所有 tool result
        tool_indices = [
            i for i in range(last_assistant_idx)
            if messages[i].get("role") == "tool"
        ]
        if len(tool_indices) <= keep_recent:
            return messages, False

        # 保留最后 keep_recent 个，前面的清内容
        to_clear = tool_indices[:-keep_recent] if keep_recent > 0 else tool_indices
        cleared_count = 0
        for i in to_clear:
            content = messages[i].get("content")
            # 幂等：清过的不再计 cleared_count——纯文本占位（CLEARED_MARK）
            # 和已带 full_at 找回指针的 offload 占位（'"full_at"' 子串是它的
            # 指纹）都算「已清过」，再跑一遍结果一个字节都不变
            if not content or content == CLEARED_MARK or '"full_at"' in content:
                continue
            # agent_home 在场：清空前先落盘，占位升级成「预览 + full_at 指针」
            # ——旧内容从「半丢失」变「可找回」。threshold=0 = 无视单条
            # 大小强制落盘（时间清理看的是「多久没动」，不看内容长短）
            if agent_home is not None:
                try:
                    from agent.output_offload import maybe_offload
                    tcid = str(messages[i].get("tool_call_id") or f"timeclear_{i}")
                    new_content = maybe_offload(
                        content, tool_call_id=tcid, agent_home=agent_home,
                        threshold=0,
                    )
                    messages[i]["content"] = new_content
                    cleared_count += 1
                    continue
                except Exception as off_e:
                    logger.warning("时间清理落盘失败（退纯占位）: %s", off_e)
            # 兜底：没传 agent_home 或落盘失败 → 纯文本占位（老行为）
            messages[i]["content"] = CLEARED_MARK
            cleared_count += 1

        if cleared_count:
            logger.info(
                "time-based MC：清除了 %d 条旧工具结果（距上次 assistant %d 分钟）",
                cleared_count, int(elapsed_min),
            )
            return messages, True
        return messages, False
    except Exception as e:
        logger.warning("time-based MC 异常（fail-open）: %s", e)
        return messages, False


def strip_internal_fields(messages: list) -> list:
    """把消息里的内部字段（如 _timestamp 时间戳、_ephemeral 临时标记）洗掉，再发给 LLM。

    为什么：这些字段是程序自己记账用的，混进发给 LLM 的内容里不但没用，
    还会改变文本、打穿 prompt cache（服务商按前缀复用的缓存）。在组装
    发给 LLM 的 messages 时调用。

    参数：
        messages：消息列表
    返回：洗净内部字段后的消息列表（没脏消息就原样返回）。

    补充：`_ephemeral` 标记「这条消息只在本轮临时用」；
    洗掉标记本身，但消息的 content/role 照常保留——LLM 需要内容，
    不需要我们的记账标记。
    """
    INTERNAL_KEYS = ("_timestamp", "_ephemeral")
    out = []
    for m in messages:
        if any(k in m for k in INTERNAL_KEYS):
            new_m = {k: v for k, v in m.items() if k not in INTERNAL_KEYS}
            out.append(new_m)
        else:
            out.append(m)
    return out


def snip_compact(
    messages: list,
    *,
    keep_first: int = 3,
    keep_last: int = 47,
    threshold: int = 50,
) -> Tuple[list, bool]:
    """L1 第 1 层：消息条数超过 threshold 时，把中间一段裁掉，留头 N 条 + 尾 M 条 + 一条占位说明。

    无损：被裁内容的完整原文在 .transcripts/latest.jsonl 里，占位消息会告诉
    LLM 想看细节就去读那个文件。
    成对保护：assistant 发出的工具调用（tool_calls）和对应的工具结果（tool 消息）
    必须成对出现，拆散了 OpenAI 协议直接报错。所以裁剪边界若正好卡在
    assistant(tool_calls) 上，就往后扩几条把它的工具结果带上，不拆散。

    参数：
        messages：完整消息列表
        keep_first：头部保留条数
        keep_last：尾部保留条数
        threshold：消息总数超过这个数才裁
    返回：(新消息列表, 是否真的裁了)。
    """
    system, conv = _split_system(messages)
    # 已有占位 → 不二次裁（幂等：重复跑结果一样）
    # 判定「已有占位」要三重限定：前缀 "[snip_compact:" + role 是 user + 内容以它开头
    # 不能只搜裸字符串 "snip_compact"——用户消息里提到这词会误判；
    # 也不能只搜 "[snip_compact:"——工具读到含这子串的源码/输出同样误判。
    # 真实占位长什么样见下方构造处：role 是 user，content 以 "[snip_compact:" 开头
    placeholders = [
        m for m in conv
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("[snip_compact:")
    ]
    if placeholders:
        return messages, False
    if len(conv) <= threshold:
        return messages, False
    if len(conv) <= keep_first + keep_last:
        return messages, False

    # 头部边界成对保护：头部最后一条若是 assistant(tool_calls) 或工具结果，
    # 就把后面连续的工具结果都带上（同一次工具调用可能有多条结果，不能拆一半）。
    # - 最后是 assistant(tc)：把它所有结果都纳入头部
    # - 最后是工具结果：说明头部已装下某次工具调用的部分结果，把剩余的连续结果补齐
    head_end = keep_first
    needs_extend = (
        head_end > 0
        and head_end < len(conv)
        and (_has_tool_calls(conv[head_end - 1]) or _is_tool_result(conv[head_end - 1]))
    )
    if needs_extend:
        while head_end < len(conv) and _is_tool_result(conv[head_end]):
            head_end += 1

    # 尾部边界成对保护：尾部开头若是工具结果，它对应的 assistant(tool_calls) 一定在
    # 尾部之外（第一条就是结果，说明发起调用的消息在更前面）——留着就是孤儿，API 会拒。
    # 无条件跳过这些孤儿结果（L1 无损，原文可从 transcript 找回）。
    # 注意：就算尾部开头的前一条正好是 assistant(tc) 也不算配对完整——那条在尾部之外，
    # 尾部内的结果依然是孤儿，照样报 400。
    tail_start = len(conv) - keep_last
    while tail_start < len(conv) and _is_tool_result(conv[tail_start]) and tail_start > head_end:
        tail_start += 1

    head = conv[:head_end]
    tail = conv[tail_start:]
    omitted = tail_start - head_end
    placeholder = {
        "role": "user",
        "content": (
            f"[snip_compact: 中间 {omitted} 条已省略，"
            f"完整记录见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = head + [placeholder] + tail
    new_messages = _reassemble(system, new_conv)
    logger.info("L1 snip_compact: conv %d → %d (omitted %d, head_end=%d, tail_start=%d)",
                len(conv), len(new_conv), omitted, head_end, tail_start)
    return new_messages, True


def micro_compact(
    messages: list,
    *,
    threshold: int = 10000,
    preview_chars: int = 200,
    keep_recent: int = 3,
    agent_home=None,
) -> Tuple[list, bool]:
    """L2 第 2 层：折叠「单条特别大」的工具结果。

    触发：某条工具结果的 content 超过 threshold 字符才折叠（看大小，不看条数）。
    折叠：原文写进 .task_outputs/ 磁盘文件，原地换成带 full_at 文件指针的
    缩短版——模型想看全文可以自己调 read_file 读回来。
    保护：最近 keep_recent（默认 3）条工具结果永远不折叠（大概率马上还要用）。
    安全：只替换 content，role/tool_call_id/name 原样保留（不破坏工具调用配对）。
    幂等：已经折叠过/已经落过盘的不再动。

    参数：
        messages：完整消息列表
        threshold：单条工具结果超过多少字符才折叠
        preview_chars：折叠后保留的预览长度
        keep_recent：最近几条工具结果受保护
        agent_home：CodeAgent 数据目录（~/.codeAgent），落盘文件的存放根目录
    返回：(新消息列表, 是否折叠过至少一条)。
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return messages, False
    protected = set(tool_indices[-keep_recent:])  # 最近 keep_recent 条工具结果受保护

    folded = 0
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool" or i in protected:
            out.append(m)
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= threshold:
            out.append(m)
            continue
        if _already_micro_placeheld(m) or _already_offloaded(m):
            out.append(m)
            continue

        # 把原文写进磁盘文件，原地留带 full_at 指针的预览（模型可自己读回全文）
        if agent_home:
            try:
                from agent.output_offload import maybe_offload
                new_content = maybe_offload(
                    content,
                    tool_call_id=m.get("tool_call_id") or f"micro_{i}",
                    agent_home=agent_home,
                    threshold=0,  # 传 0 表示无视单条阈值、强制落盘
                    preview_chars=preview_chars,
                )
                if new_content != content:
                    new_m = dict(m)
                    new_m["content"] = new_content
                    out.append(new_m)
                    folded += 1
                    continue
            except Exception:
                pass

        # 兜底（没传 agent_home 或落盘失败）：换成提示性占位
        new_m = dict(m)
        new_m["content"] = json.dumps({
            "micro_compacted": True,
            "orig_chars": len(content),
            "hint": (
                f"Tool {m.get('name', '?')} 结果已折叠，"
                f"完整内容见 .transcripts/latest.jsonl 或重跑工具"
            ),
        }, ensure_ascii=False)
        out.append(new_m)
        folded += 1

    if folded == 0:
        return messages, False
    logger.info("L2 micro_compact: folded %d oversized tool results", folded)
    return out, True


def _already_micro_placeheld(msg: dict) -> bool:
    """判断某条工具消息的内容是不是已被 L2 折叠成占位（避免重复折叠）。"""
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
        return bool(parsed.get("micro_compacted"))
    except (json.JSONDecodeError, TypeError):
        return False


def _already_offloaded(msg: dict) -> bool:
    """判断某条工具消息是不是已经落过盘（output_offload 占位形态）。

    落盘占位的 JSON 里有 "truncated": true 和 "full_at" 两个字段；
    已是这种形态就不要再落一次盘。
    """
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
        return bool(parsed.get("truncated")) and "full_at" in parsed
    except (json.JSONDecodeError, TypeError):
        return False


def offload_large_tool_results(
    messages: list,
    *,
    agent_home,
    threshold: int = 50000,
    preview_chars: int = 2000,
    message_threshold: int = 200000,
    freeze: bool = True,
) -> Tuple[list, bool]:
    """L2.5：主动扫描所有工具结果消息，超阈值就落盘。

    三层触发逻辑（大白话）：
      1. 单条阈值（threshold，默认 5 万字符）：某一条工具结果太大 → 落盘
      2. 分段聚合阈值（message_threshold，默认 20 万字符）：一段连续的工具结果
         （不跨 user/assistant 边界）加起来太大 → 从最大的开始逐个落盘，
         直到总和降到阈值以下
      3. 决策冻结（freeze=True）：已经落过盘的工具调用，下次直接照抄上次生成的
         预览内容，不再重新评估——保证内容一个字节都不变，保护 prompt cache

    参数：
        messages：完整消息列表
        agent_home：CodeAgent 数据目录，落盘位置
        threshold：单条工具结果的落盘阈值（字符）
        preview_chars：落盘后保留的预览长度
        message_threshold：一段连续工具结果的聚合阈值；0 表示关闭聚合检查
        freeze：是否启用决策冻结
    返回：(新消息列表, 是否有变化)。除 content 外消息结构不变（保住工具调用配对）。
    """
    from agent.output_offload import maybe_offload

    changed = False
    out = []
    for m in messages:
        if m.get("role") != "tool":
            out.append(m)
            continue

        tc_id = m.get("tool_call_id") or ""

        # ── 决策冻结：落过盘的直接照抄上次的预览内容（一字不差，保 prompt cache）──
        if freeze and tc_id and tc_id in _offload_decisions:
            decision = _offload_decisions[tc_id]
            # 直接用记录的预览内容替换（不重新评估）
            if m.get("content") != decision["preview"]:
                new_m = dict(m)
                new_m["content"] = decision["preview"]
                out.append(new_m)
                # 冻结重放不算 changed（没新落盘，只是保持一致）
            else:
                out.append(m)
            continue

        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= threshold:
            out.append(m)
            continue
        if _already_offloaded(m):
            out.append(m)  # 已是占位（来自其他路径），不二次落盘
            continue

        # 单条阈值触发落盘
        effective_tc_id = tc_id or f"orphan_{id(m)}"
        new_content = maybe_offload(
            content,
            tool_call_id=effective_tc_id,
            agent_home=agent_home,
            threshold=threshold,
            preview_chars=preview_chars,
        )
        if new_content != content:
            new_m = dict(m)
            new_m["content"] = new_content
            out.append(new_m)
            changed = True
            if freeze and tc_id:
                _record_decision(tc_id, new_content)
        else:
            out.append(m)

    # ── per-message 聚合检查 ──
    if message_threshold > 0:
        agg_changed = _enforce_per_message_budget(
            out, message_threshold, agent_home, preview_chars, freeze,
        )
        if agg_changed:
            changed = True

    if changed:
        logger.info("L2.5 offload_large_tool_results: 至少 1 条 tool 消息已落盘（精细化）")
    return out, changed


# ---------------------------------------------------------------------------
# 决策冻结 + 按段聚合落盘
# ---------------------------------------------------------------------------

_offload_decisions: dict = {}  # 工具调用 id -> {"preview": 预览内容, "file_path": 文件路径}
_OFFLOAD_DECISIONS_LIMIT = 1000  # 表大小上限（简易 LRU 淘汰），防超长会话把内存撑爆


def _record_decision(tc_id: str, preview: str, file_path: str = None) -> None:
    """把一条落盘决策记进全局表，表满了就淘汰最老的（简易 LRU——dict 从 Python 3.7 起按插入顺序排列，删第一个元素就是删最早插入的）。

    参数：
        tc_id：工具调用的唯一 id
        preview：落盘后替换上去的预览内容
        file_path：原文落到了哪个文件
    """
    if len(_offload_decisions) >= _OFFLOAD_DECISIONS_LIMIT:
        # 淘汰最早插入的一个（dict 在 Py3.7+ 保序）
        oldest = next(iter(_offload_decisions))
        del _offload_decisions[oldest]
    _offload_decisions[tc_id] = {"preview": preview, "file_path": file_path}


def reset_offload_decisions() -> None:
    """清空全局落盘决策表（测试之间隔离用）。

    不要在新建 agent 时调这个——这张表是同进程内
    所有 agent（主代理 + 并发子代理）共享的，新建 agent 就清空，
    会把别的正在跑的 agent 的冻结决策一起抹掉，重放内容对不上、打穿 prompt cache。
    生产路径靠 1000 条上限的淘汰机制控内存；跨会话也不怕泄漏
    （工具调用 id 是服务商随机生成的，call_xxx 不会撞车）。
    """
    _offload_decisions.clear()


def _enforce_per_message_budget(
    messages: list,
    limit: int,
    agent_home,
    preview_chars: int,
    freeze: bool,
) -> bool:
    """按段聚合检查：一段连续工具结果的总和超标时，挑最大的几条落盘。

    分组规则：以 user 消息为界——一段连续的工具结果（中间可以夹着
    assistant(tool_calls)，但不能跨过 user 消息）算一组。道理：一次用户输入
    触发的一串工具调用是一个逻辑整体。

    超标的组：按大小从大到小排，逐个落盘直到总和回到限内。
    已在决策冻结表里命中的、或已经是占位的，不重复处理。

    参数：
        messages：消息列表（content 会被原地修改）
        limit：一段的总字符上限
        agent_home：落盘目录根
        preview_chars：落盘后保留的预览长度
        freeze：是否记录决策（供下次冻结照抄）
    返回：是否有消息被落盘。
    """
    from agent.output_offload import maybe_offload

    changed = False

    # 1. 按 user 消息边界分组：收集所有工具结果的索引，
    #    遇到新 user 消息就开新一段
    segments = []  # list of list of indices
    current_seg = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            current_seg.append(i)
        elif m.get("role") == "user":
            # user 消息是分组边界——user 之后的工具结果属于新一段
            if current_seg:
                segments.append(current_seg)
                current_seg = []
        # assistant / system 消息不打断分段（工具结果中间可以夹 assistant(tool_calls)）
    if current_seg:
        segments.append(current_seg)

    # 2. 对每段算总和，超 limit 的按大小从大到小逐个落盘
    for seg in segments:
        # 过滤掉已是占位或决策冻结命中的（它们的 content 已经很小）
        candidates = []
        seg_total = 0
        for idx in seg:
            m = messages[idx]
            content = m.get("content", "")
            seg_total += len(content) if isinstance(content, str) else 0
            tc_id = m.get("tool_call_id") or ""
            # 决策冻结命中的或已是占位的不进候选
            if freeze and tc_id and tc_id in _offload_decisions:
                continue
            if _already_offloaded(m):
                continue
            if not isinstance(content, str):
                continue
            candidates.append((idx, len(content)))

        if seg_total <= limit:
            continue
        if not candidates:
            continue

        # 按大小降序，逐个落盘直到总和 < limit
        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, size in candidates:
            if seg_total <= limit:
                break
            m = messages[idx]
            content = m.get("content", "")
            tc_id = m.get("tool_call_id") or f"agg_{idx}"
            new_content = maybe_offload(
                content,
                tool_call_id=tc_id,
                agent_home=agent_home,
                threshold=0,  # 传 0 表示无视单条阈值、强制落盘（聚合触发）
                preview_chars=preview_chars,
            )
            if new_content != content:
                new_m = dict(m)
                new_m["content"] = new_content
                messages[idx] = new_m
                seg_total -= size - len(new_content)
                changed = True
                if freeze and tc_id:
                    _record_decision(tc_id, new_content)
                logger.info(
                    "per-message 聚合 offload: tool 消息 idx=%d %d→%d",
                    idx, size, len(new_content),
                )

    return changed


def _split_pinned(messages: list) -> tuple:
    """把「纯文本 pinned 消息」和其余消息分开（各自保持原相对顺序）。

    只认无 tool_calls 的：带 tool_calls 的 assistant 其结果必然已被
    摘要，原样保留会造成孤儿调用（配对语义断裂）。

    参数：messages：消息列表
    返回：(pinned 列表, 其余列表)
    """
    pinned, rest = [], []
    for m in messages:
        content = m.get("content", "")
        is_pinned = (
            (isinstance(content, str) and content.startswith("[pinned]"))
            or bool(m.get("pinned"))
        )
        if is_pinned and not m.get("tool_calls"):
            pinned.append(m)
        else:
            rest.append(m)
    return pinned, rest


def apply_context_collapse(
    messages: list,
    *,
    threshold_ratio: float = 0.8,
    context_window: int = 128_000,
    keep_recent_turns: int = 3,
) -> Tuple[list, bool]:
    """L3.5 contextCollapse：按 token 占用比例，把早期对话整段折叠成一条占位说明。这一层**轻量、无损、可逆**——比 L4 花 LLM
    读一遍写摘要的有损压缩便宜得多，所以排在 L1/L2 之后、L4 之前当缓冲垫。

    触发：估算 token 数 ÷ 模型上下文窗口大小 > threshold_ratio。
    动作（大白话版）：
        1. 把 system 消息摘出来不动（保 prompt cache）
        2. 找出所有钉住的消息（pinned：内容以 [pinned] 开头，或标记 pinned=True）→ 保护
        3. 保留最近 keep_recent_turns 轮（1 轮 = 一条 user + 一条 assistant = 2 条）
        4. 剩下的中间段折叠成一条 "[context_collapse: ...]" 占位 user 消息
        5. 最终顺序：system + 钉住的 + 占位 + 最近 N 轮
        6. 最后过一遍 _fix_tool_call_pairs 兜底（防切出孤儿工具结果）

    可逆：原文由 snapshot_if_needed（编排器在 L4 前调用）或
    .transcripts/latest.jsonl 保留，占位本身会提示去哪里找。
    幂等：列表里已有 [context_collapse: 占位 → 直接返回不重复折叠。

    参数：
        messages：完整消息列表（含开头的 system）
        threshold_ratio：0-1 的占用比例，超过才触发
        context_window：模型上下文窗口大小（token 数）；默认 128K（DeepSeek/OpenAI 常见值）
        keep_recent_turns：保护最近几轮对话（1 轮 = 2 条消息）
    返回：(新消息列表, 是否发生了折叠)。新列表是浅拷贝，原列表不改。
    """
    # 幂等：已是折叠状态 → 不二次折叠
    if any(
        "[context_collapse:" in str(m.get("content", ""))
        for m in messages
    ):
        return messages, False

    # 估算 token；没超占用比例直接返回（noop = 什么都不做）
    est_tokens = estimate_message_tokens(messages)
    if est_tokens / max(context_window, 1) <= threshold_ratio:
        return messages, False

    system, conv = _split_system(messages)
    if len(conv) < (keep_recent_turns * 2 + 2):
        # 对话太少，没什么可折叠的
        return messages, False

    # 拆出钉住的 + 中间段 + 最近 N 轮
    # 最近 N 轮 = conv 末尾 keep_recent_turns*2 条（边界允许放宽以保工具调用成对：
    # 尾段开头若是工具结果，往前扩到发起它的 assistant(tool_calls)）
    tail_len = keep_recent_turns * 2
    tail_start = len(conv) - tail_len
    while tail_start > 0 and _is_tool_result(conv[tail_start]) and tail_start > 1:
        tail_start -= 1  # 往前找发起这次调用的 assistant(tool_calls)

    head_region = conv[:tail_start]  # 可折叠区域
    tail_region = conv[tail_start:]  # 最近 N 轮（保护不动）

    # 从可折叠区域里挑出钉住的消息（保留原有顺序）
    pinned = []
    collapsible_indices = []
    for idx, m in enumerate(head_region):
        content = m.get("content", "")
        is_pinned = (
            (isinstance(content, str) and content.startswith("[pinned]"))
            or bool(m.get("pinned"))
        )
        if is_pinned:
            pinned.append((idx, m))
        else:
            collapsible_indices.append(idx)

    if not collapsible_indices:
        # 全是钉住的消息，没什么可折叠的
        return messages, False

    folded_turns = len(collapsible_indices) // 2  # 粗略换算：2 条消息算 1 轮
    placeholder = {
        "role": "user",
        "content": (
            f"[context_collapse: 已折叠 {folded_turns} 轮早期对话"
            f"（{len(collapsible_indices)} 条消息），"
            "完整记录快照在 .transcripts/ 目录（latest.txt 指向最新一份）]"
        ),
    }

    # 重组：system + 钉住段（按原序）+ 占位 + 最近 N 轮
    new_conv = [m for _, m in pinned] + [placeholder] + tail_region
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    saved_tokens = est_tokens - estimate_message_tokens(new_messages)
    logger.info(
        "L3.5 context_collapse: conv %d → %d (folded %d msgs, saved ~%d tokens)",
        len(conv), len(new_conv), len(collapsible_indices), saved_tokens,
    )
    return new_messages, True


def _build_compact_boundary(
    coverage: str, preserved: str, transcript_path: Optional[str] = None,
) -> str:
    """构造压缩边界标注。

    写清楚三件事：什么时候压的 / 摘要覆盖了哪些内容 / 哪些段是原文保留的，
    再加一句提醒「保留段是原文、摘要只是转述」——帮模型分清哪些内容可信，
    引用具体数据/路径/命令输出时以保留段为准。

    参数：
        coverage：摘要覆盖的范围描述
        preserved：原样保留的段的范围描述
        transcript_path：压缩前原文快照的落盘路径（None = 没快照，不加该行）
    返回：拼好的标注文本。
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    snapshot_line = ""
    if transcript_path:
        # 找回通道：被摘要段的原文在磁盘上，模型需要细节时可分段读回
        snapshot_line = (
            f"- 原文完整快照：{transcript_path}"
            "（需要被摘要段的细节时用 read_file 读回，"
            "文件可能很大，用 offset/limit 分段）\n"
        )
    return (
        "[COMPACT_BOUNDARY]\n"
        f"- 压缩时间：{ts}\n"
        f"- 摘要覆盖范围：{coverage}（由 LLM 转述，细节可能有省略）\n"
        f"- 保留段范围：{preserved}（原样保留，含工具结果原文）\n"
        f"{snapshot_line}"
        "- 注意：保留段内容是精确的，摘要内容是转述；"
        "引用具体数据/路径/命令输出以保留段为准。\n"
    )


def _summary_shrinks(placeholder_text: str, replaced_segment: list) -> bool:
    """收敛检查：摘要占位（含框文本）的 token 是否小于被替换段。

    借鉴 dsh：压完比原来还长属于白花钱，必须算失败而不是硬塞进历史。
    """
    return (
        estimate_message_tokens([{"role": "user", "content": placeholder_text}])
        < estimate_message_tokens(replaced_segment)
    )


async def llm_compact(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    keep_recent: int = 10,
    token_threshold: int = 100000,
    msg_threshold: int = 100,
    precomputed_tokens: Optional[int] = None,
    session_memory: Optional[str] = None,
    from_idx: int = 0,
    up_to_idx: int = -1,
    tools: Optional[list] = None,
    summary_scale_thresholds=None,
    summary_files_errors_limits=None,
    transcript_path: Optional[str] = None,
    session_state: Optional["CompressionSessionState"] = None,
) -> Tuple[list, bool]:
    """L4 第 4 层：前面几层压不下去、仍超 token 阈值时，调 LLM 把旧对话写成摘要。

    有损：旧对话被摘要文本替换，细节可能丢（所以编排器在调用前会把原文
    快照到 transcript）。函数是 async 的（内部调的 _summarize_conversation
    是异步的，这里必须 await，漏了会拿到协程对象而不是结果）。

    参数：
        messages：完整消息列表
        llm_client：LLM 客户端
        model：模型名
        keep_recent：全量模式下末尾保护条数（最近这么多条不进摘要）
        token_threshold：token 阈值，超过才压
        msg_threshold：当前逻辑不用（压缩只看 token，不看消息条数）
        precomputed_tokens：调用方已算好的 token 数（省得重复遍历）；None 时内部自己算
        session_memory：预提取的会话记忆；有值就直接用它当摘要，
                        不调 LLM。获取接口尚未实现，目前永远传 None
        from_idx：局部压缩起始条数——只压这一段，
                  段外原文保留；默认 0
        up_to_idx：局部压缩的结束条数；默认 -1 = 压到末尾。两者都取默认值时
                   走全量模式（keep_recent 逻辑）；局部模式下 keep_recent 被忽略。
                   注意：局部（partial）模式成功后不写边界占位暂存——恢复裁剪
                   按「边界前全裁」语义会误裁 head 保留段，宁可不落库
        tools：当前工具 schema 列表（fork 前缀复用要用，见 _summarize_conversation）
        session_state：会话压缩记账簿；收敛检查不过时用它记一次失败账
                       （失败计数 +1 + 冷却），防下一轮立刻重试白烧。可不传
    返回：(新消息列表, 是否真的压缩了)。
    """
    system, conv = _split_system(messages)
    if precomputed_tokens is not None:
        over_token = precomputed_tokens > token_threshold
    else:
        over_token = estimate_message_tokens(messages) > token_threshold
    if not over_token:  # 压不压只看 token，不看消息条数
        return messages, False

    # 局部模式 vs 全量模式
    is_partial = from_idx != 0 or up_to_idx != -1

    if is_partial:
        # 局部模式：头段原文 + 摘要 + 尾段原文 拼装
        effective_up_to = len(conv) if up_to_idx < 0 else up_to_idx
        head = conv[:from_idx]
        tail = conv[effective_up_to:] if effective_up_to < len(conv) else []

        # 被摘要段里若有上一次压缩的摘要 → 提锚定段防代际损耗
        anchor, anchor_note = _extract_anchor_from_slice(
            conv[from_idx:effective_up_to],
        )
        # 纯文本 pinned 消息原样保留（不进重组段被摘要掉）；
        # non_pinned = 真正会被摘要替换掉的那部分（收敛检查要用）
        pinned_msgs, non_pinned = _split_pinned(conv[from_idx:effective_up_to])

        summary = await _summarize_conversation(
            conv,  # 传完整 conv，由 _summarize_conversation 内部按 from/up_to 切片
            llm_client, model=model,
            session_memory=session_memory,
            from_idx=from_idx,
            up_to_idx=effective_up_to,
            # fork 前缀 = 完整 messages（含 system），tools 与主调用一致
            fork_prefix_messages=messages,
            tools=tools,
            anchor_note=anchor_note,
            scale_thresholds=summary_scale_thresholds,
            files_limits=summary_files_errors_limits,
        )
        if not summary:
            return messages, False
        summary = _prepend_anchor(summary, anchor)

        # 收敛检查（局部模式）：被替换的是段内非 pinned 部分
        _placeholder_probe = (
            f"[COMPACT_BOUNDARY]\n[对话摘要（{from_idx}-{effective_up_to}）]\n\n"
            f"{summary}\n\n[以下是压缩段之后的对话，请继续]"
        )
        if not _summary_shrinks(_placeholder_probe, non_pinned):
            logger.warning(
                "L4 收敛检查未通过（partial %d-%d）：摘要不小于被替换段，"
                "放弃本次替换", from_idx, effective_up_to,
            )
            if session_state is not None:
                session_state.record_llm_compact_failure()
            return messages, False

        placeholder = {
            "role": "user",
            "content": (
                _build_compact_boundary(
                    f"消息 {from_idx}-{effective_up_to}",
                    f"head（消息 0~{from_idx}，{len(head)} 条原文）"
                    f"+ tail（消息 {effective_up_to}~，{len(tail)} 条原文）",
                    transcript_path=transcript_path,
                )
                + f"\n[对话摘要（{from_idx}-{effective_up_to}）]\n\n"
                f"{summary}\n\n"
                "[以下是压缩段之后的对话，请继续]"
            ),
        }
        new_conv = head + pinned_msgs + [placeholder] + tail
        new_conv = _fix_tool_call_pairs(new_conv)
        new_messages = _reassemble(system, new_conv)

        summarized_count = effective_up_to - from_idx
        logger.info(
            "L4 llm_compact (partial %d-%d): %d msgs summarized, head=%d tail=%d",
            from_idx, effective_up_to, summarized_count,
            len(head), len(tail),
        )
        try:
            from agent.cache_monitor import notify_compaction
            notify_compaction()
        except Exception as e:
            logger.debug("notify_compaction fail-open: %s", e)
        # partial 故意不写 _last_compact_placeholder（不落库边界占位）：恢复端
        # _truncate_at_last_compact_boundary 按「最后边界之前全裁」工作，落了
        # 会把 partial 明确保留的 head 段裁掉——与边界文本「保留段范围：
        # head+tail」自相矛盾。宁可不落：重启全量重载旧历史，保留段一条不丢。
        return new_messages, True

    # 全量模式（原逻辑，保持向后兼容）
    if len(conv) <= keep_recent:
        return messages, False

    to_summarize = conv[:-keep_recent]
    keep = conv[-keep_recent:]

    # 纯文本 pinned 消息免摘要：从摘要输入里挑出来原样保留
    # （原文在场，不必再进摘要；带 tool_calls 的不保——结果已被摘要，
    # 原样保留会造成孤儿调用）
    pinned_msgs, to_summarize = _split_pinned(to_summarize)

    # 被摘要段里若有上一次压缩的摘要 → 提锚定段防代际损耗
    anchor, anchor_note = _extract_anchor_from_slice(to_summarize)

    summary = await _summarize_conversation(
        to_summarize, llm_client, model=model,
        session_memory=session_memory,
        # fork 前缀 = 完整 messages（含 system），tools 与主调用一致
        fork_prefix_messages=messages,
        tools=tools,
        anchor_note=anchor_note,
        scale_thresholds=summary_scale_thresholds,
        files_limits=summary_files_errors_limits,
    )
    if not summary:
        return messages, False
    summary = _prepend_anchor(summary, anchor)

    # 收敛检查（全量模式）：被替换的是非 pinned 的 to_summarize
    _placeholder_probe = (
        f"[COMPACT_BOUNDARY]\n[之前的对话已自动总结]\n\n"
        f"{summary}\n\n[以下是最近的对话，请继续]"
    )
    if not _summary_shrinks(_placeholder_probe, to_summarize):
        logger.warning(
            "L4 收敛检查未通过：摘要不小于被替换段（%d 条），放弃本次替换",
            len(to_summarize),
        )
        if session_state is not None:
            session_state.record_llm_compact_failure()
        return messages, False

    placeholder = {
        "role": "user",
        "content": (
            _build_compact_boundary(
                f"conv 第 1~{len(to_summarize)} 条消息（共 {len(to_summarize)} 条）",
                f"最近 {len(keep)} 条消息",
                transcript_path=transcript_path,
            )
            + "\n[之前的对话已自动总结]\n\n"
            f"{summary}\n\n"
            "[以下是最近的对话，请继续]"
        ),
    }
    new_conv = pinned_msgs + [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    logger.info(
        "L4 llm_compact: %d msgs summarized, %d chars → %d chars summary",
        len(to_summarize),
        sum(len(str(m.get("content", ""))) for m in to_summarize),
        len(summary),
    )
    # 通知 cache_monitor「下次缓存命中率下降是压缩造成的，属预期」
    # 放在 return 前，确保只在真发生压缩时才通知
    try:
        from agent.cache_monitor import notify_compaction
        notify_compaction()
    except Exception as e:
        logger.debug("notify_compaction fail-open: %s", e)
    # 全量模式才写边界占位暂存（partial 分支在上面故意跳过——恢复裁剪
    # 会误裁 head 保留段）。global 声明只在全量分支用，partial 分支已不
    # 触碰该名字，放这里不会 SyntaxError。
    global _last_compact_placeholder
    _last_compact_placeholder = placeholder["content"]
    return new_messages, True


@dataclass
class CompressionSessionState:
    """一场会话内压缩相关的记账簿。

    字段（大白话）：
    - reactive_last_at：上次紧急压缩的时间戳（0 = 从没触发过）
    - reactive_count：本场紧急压缩已触发几次
    - llm_compact_count：L4 触发过几次
    - last_llm_compact_turn：上次 L4 触发时的轮次号（算冷却期用）
    - current_turn：当前 LLM 轮次号（由 agent 主循环累加）
    - llm_compact_failures：L4 连续失败计数（触发熔断用）。
      注意熔断有两层：这里管的是「要不要触发 L4」这层——连续失败就别再
      白花钱调摘要了；摘要生成那层的熔断在 context_compressor 的模块级状态里，
      两层互补。

    向后兼容：旧的 ``reacted`` 属性保留为只读别名（= reactive_count > 0），
    老代码读它不会坏。
    """
    reactive_last_at: float = 0.0
    reactive_count: int = 0
    llm_compact_count: int = 0
    last_llm_compact_turn: int = -10**6
    current_turn: int = 0
    llm_compact_failures: int = 0

    @property
    def reacted(self) -> bool:
        """向后兼容：reacted 等价于「本场触发过至少一次紧急压缩」。"""
        return self.reactive_count > 0

    def record_llm_compact(self) -> None:
        self.llm_compact_count += 1
        self.last_llm_compact_turn = self.current_turn

    def record_llm_compact_failure(self) -> None:
        """记一次 L4 软失败（如收敛检查不过）：失败计数 +1 并记冷却，
        防止下一轮立刻重试白烧一次摘要调用。"""
        self.llm_compact_failures += 1
        # 冷却记账与 record_llm_compact 同款（同取 current_turn），保持
        # 成功/失败两口径一致——失败同样进冷却期，不给「每轮重试」留缝
        self.last_llm_compact_turn = self.current_turn

    def cooldown_ok(self, cooldown_turns: int) -> bool:
        return self.current_turn - self.last_llm_compact_turn >= cooldown_turns

    def increment_turn(self) -> None:
        self.current_turn += 1


# L4 连续失败多少次就熔断（本会话不再触发 L4）
MAX_CONSECUTIVE_L4_FAILURES = 3


def _effective_llm_compact_threshold(config: dict, model: Optional[str]) -> int:
    """L4 阈值生效值：min(配置值（1M 模型抬到 70 万）, 推断窗口 × 0.9)。

    旧默认 10 万对 64k 窗口的 DeepSeek 形同虚设——优雅压缩永远晚于真实
    上限，长会话只能靠 PTL 报错后的紧急截断（只留 5 条）兜底。配置里
    显式写的值继续尊重，但封顶不超过窗口 90%（给输出和增长预估留量）。

    参数：
        config：context 配置字典
        model：模型名
    返回：生效的 token 阈值。
    """
    from agent.context_compressor import _get_model_max_tokens
    window = _get_model_max_tokens(model)
    raw = config.get("llm_compact_token_threshold", 100000)
    if model and "[1m]" in str(model):
        raw = max(raw, 700000)
    return min(raw, int(window * 0.9))


def needs_pre_send_compaction(
    messages: list,
    anchor: Optional[tuple],
    model: Optional[str],
    *,
    ratio: float = 0.9,
) -> bool:
    """发送前预检：混合估算超过推断窗口的 ratio 比例，就该再压一次。

    循环顶部的压缩之后、真正发送之前，还会追加记忆注入等增量；恢复后
    第一轮没有真实锚点时估算偏差也最大。这道预检是「PTL 报错后 reactive
    只留 5 条」灾难通道之前的最后防线——宁可多跑一次优雅压缩。
    任何异常都返回 False（fail-open：预检自己绝不能挡住发送）。

    参数：
        messages：即将发送的完整消息列表
        anchor：最近一次 usage 锚点（条数, token 数）
        model：模型名（推断窗口用）
        ratio：触发比例（默认窗口的 90%）
    返回：True = 发送前应再跑一次 force 压缩。
    """
    try:
        from agent.context_compressor import _get_model_max_tokens
        window = _get_model_max_tokens(model)
        est = estimate_tokens_hybrid(messages, anchor)
        return est > window * ratio
    except Exception:
        return False


def estimate_tokens_hybrid(
    messages: list,
    anchor: Optional[tuple] = None,
) -> int:
    """混合 token 计数——真实值打底，新增部分粗估。

    全靠估算会越估越偏，真实值又只有调完 API 才拿得到（账单里的数）。
    折中：拿最近一次主调用返回的真实输入 token 数（usage 里的 prompt +
    cache_read + cache_creation）当「锚点」，锚点之后新加的消息才用粗估——
    阈值判定的误差从「全程都在估」缩小到「只估增量」。

    参数：
        messages：完整消息列表
        anchor：锚点元组 (当时的消息条数, 当时的真实输入 token 数)
    返回：估算的 token 总数。
    消息被压缩/回退过（条数比锚点还少）或没传锚点 → 全量粗估（保守回退）。
    fail-open：锚点格式不对就当没传处理。
    """
    try:
        if anchor and len(anchor) == 2:
            anchor_count, anchor_tokens = int(anchor[0]), int(anchor[1])
            if 0 < anchor_count < len(messages):
                return anchor_tokens + estimate_message_tokens(
                    messages[anchor_count:]
                )
    except (TypeError, ValueError):
        pass
    return estimate_message_tokens(messages)


# 紧急压缩的两道保险默认值（config 可覆盖）：冷却窗口秒数 / 单会话次数上限
REACTIVE_COOLDOWN_SECONDS = 60
REACTIVE_MAX_PER_SESSION = 5


def reactive_compact(
    messages: list,
    *,
    session_state: CompressionSessionState,
    keep_recent: int = 5,
    cooldown_seconds: float = REACTIVE_COOLDOWN_SECONDS,
    max_per_session: int = REACTIVE_MAX_PER_SESSION,
    now_fn=time.time,
) -> Tuple[list, bool]:
    """紧急通道：API 报「对话超长」（prompt_too_long）时立刻调用。

    做法简单粗暴：只留 system + 一条说明占位 + 最后 keep_recent 条消息，
    其余全靠 transcript 文件找回。
    **可多次触发**：每次报错都可以再压，但有两道保险：
      1. 冷却窗口：距上次触发不足 cooldown_seconds 秒（默认 60s）→ 跳过
      2. 单会话上限：本场已触发 max_per_session 次（默认 5）→ 跳过

    参数：
        messages：完整消息列表
        session_state：会话记账簿（记触发次数和时间）
        keep_recent：末尾保留条数
        cooldown_seconds：冷却窗口秒数
        max_per_session：单会话触发上限
        now_fn：取当前时间的函数，测试注入假时钟用（生产代码不传）
    返回：(新消息列表, 是否真的压了)。
    """
    # 保护 1：冷却窗口
    now = now_fn()
    elapsed = now - session_state.reactive_last_at
    if session_state.reactive_count > 0 and elapsed < cooldown_seconds:
        logger.info(
            "reactive_compact 冷却中（距上次 %ds < %ds），跳过",
            int(elapsed), int(cooldown_seconds),
        )
        return messages, False

    # 保护 2：单会话上限
    if session_state.reactive_count >= max_per_session:
        logger.warning(
            "reactive_compact 达到单会话上限 %d 次，跳过",
            session_state.reactive_count,
        )
        return messages, False

    system, conv = _split_system(messages)
    keep = conv[-keep_recent:] if len(conv) > keep_recent else conv[:]
    placeholder = {
        "role": "user",
        "content": (
            # 边界前缀和大写统一：恢复侧 _truncate_at_last_compact_boundary
            # 只认 [COMPACT_BOUNDARY] 开头，reactive 的占位也得能被裁剪定位
            "[COMPACT_BOUNDARY]\n"
            "[紧急上下文压缩：API 返回 prompt_too_long，"
            f"已只保留最近 {len(keep)} 条消息。"
            "完整历史见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    session_state.reactive_last_at = now
    session_state.reactive_count += 1
    logger.warning(
        "reactive_compact triggered (#%d): kept last %d",
        session_state.reactive_count, len(keep),
    )
    # 通知 cache_monitor：下次缓存下降是压缩造成的，属预期（与 llm_compact 一致）
    # fail-open：通知失败只记 debug 日志，不影响压缩结果
    try:
        from agent.cache_monitor import notify_compaction
        notify_compaction()
    except Exception as e:
        logger.debug("notify_compaction fail-open: %s", e)
    global _last_compact_placeholder
    _last_compact_placeholder = placeholder["content"]
    return new_messages, True


def estimate_turn_growth(messages: list, *, window: int = 3, default: int = 8000) -> int:
    """预估「下一轮大概还要多烧多少 token」（防压缩震荡）。

    等对话真顶到线才压会出现「压完 → 下一轮一波大工具结果又顶线 → 再压」
    的来回震荡，所以提前量 = 最近 window 轮里**最大单轮 token 量**——取最大值是因为
    一轮大工具结果就能直接把下一轮顶过线。

    一轮的定义：一条 user 消息 + 它后面的 assistant/tool 消息，直到下一条 user。

    参数：
        messages：完整消息列表（开头的 system 会被跳过）
        window：观察最近几轮（config 的 llm_compact_growth_window）
        default：历史不足 window 轮时的保守默认值（config 的 llm_compact_growth_default）
    返回：预估增量（token 数）；消息为空或轮数不够 → default。
    """
    try:
        _, conv = _split_system(messages)
        # 按 user 消息边界分轮：一条 user + 后续 assistant/tool 直到下一条 user
        turns: list = []
        current: list = []
        for m in conv:
            if m.get("role") == "user":
                if current:
                    turns.append(current)
                current = [m]
            elif current:
                current.append(m)
        if current:
            turns.append(current)

        if len(turns) < max(1, window):
            return default
        recent = turns[-window:]
        return max(estimate_message_tokens(t) for t in recent)
    except Exception:
        return default


# 最近一次压缩产出的边界占位内容（llm_compact 全量分支 / reactive_compact
# 成功时写入，调用方用 take_last_compact_placeholder() 取去持久化）。
# llm_compact 的 partial 分支故意不写：恢复裁剪按「边界前全裁」语义会
# 误裁 head 保留段，宁可不落、重启全量重载也不丢保留段。
# 压缩在主循环里串行执行、无并发竞争——与 context_compressor 的
# _last_summary_degraded 同款模块级约定。
_last_compact_placeholder: Optional[str] = None


def take_last_compact_placeholder() -> Optional[str]:
    """取走并清空最近一次压缩的边界占位（一次性消费，防旧值误用）。"""
    global _last_compact_placeholder
    val = _last_compact_placeholder
    _last_compact_placeholder = None
    return val


def _persist_compact_marker(store, session_id: str, text: str) -> None:
    """往会话库落一条压缩事务标记（fail-open：落盘失败不阻塞压缩）。

    start 标记在 L4 开跑前写、成功后的边界标记（[COMPACT_BOUNDARY]，
    由 compress_if_needed 自己写；紧急压缩/手动压缩路径也各自调本函数
    补写）当 end 用——恢复时"有 start 无更晚 boundary"即压缩被中断的
    证据（可检测，无需修复：无 boundary 时保守全量载入本就是正确行为）。
    落库失败要大声（WARNING）：静默吞掉的话重启后全量载入、压缩白压，
    排查时连条日志线索都没有。
    """
    if store is None or not session_id:
        return
    try:
        store.append_message(session_id, "user", text)
    except Exception as e:
        logger.warning("压缩事务标记落盘失败（不阻塞压缩）: %s", e)


def _profile_messages(messages: list) -> dict:
    """单遍扫描消息列表，一次算出压缩各层判定要用的统计量。

    旧版时间清理/L1/L2/L2.5/L2.6 各自全量扫一遍（每轮 4-5 遍
    O(全部历史)），长会话纯 CPU 叠加可观——这里一遍算全，各层判定
    只读统计量不再重扫。

    参数：
        messages：完整消息列表（含可能的 system 头）
    返回：{"msg_count": 不含 system 的条数, "total_chars": conv 消息
          content 总字符, "max_tool_chars": 单条 tool 消息最大字符,
          "last_assistant_ts": 最后一条 assistant 的 _timestamp（无则 None）}

    口径注意（last_assistant_ts）：取的是最后一条「带」_timestamp 的
    assistant——字面上的最后一条 assistant 可能没有 _timestamp。因此
    时间清理的门可能比原版「多放行」一次调用，但函数内部仍按
    not last_ts 早退（什么都清不了），方向安全——顶多少省一次扫描，
    绝不会多清内容。
    """
    msg_count = 0
    total_chars = 0
    max_tool_chars = 0
    last_assistant_ts = None
    system, conv = _split_system(messages)
    for m in conv:
        msg_count += 1
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        total_chars += len(content)
        if m.get("role") == "tool" and len(content) > max_tool_chars:
            max_tool_chars = len(content)
        if m.get("role") == "assistant" and m.get("_timestamp") is not None:
            last_assistant_ts = m.get("_timestamp")
    return {
        "msg_count": msg_count,
        "total_chars": total_chars,
        "max_tool_chars": max_tool_chars,
        "last_assistant_ts": last_assistant_ts,
    }


async def compress_if_needed(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    config: dict,
    session_state: CompressionSessionState,
    agent_home,
    session_id: str,
    hooks_registry=None,
    tools: Optional[list] = None,
    authoritative_tokens: Optional[tuple] = None,
    session_store=None,
    force: bool = False,
) -> Tuple[list, bool, bool]:
    """分层压缩总调度（编排器）。返回 (新消息, 是否有改动, 是否发生了 LLM 摘要级压缩)。

    流水线顺序（从便宜到贵）：时间清理 → L1 裁中间 → L2 单条折叠 →
    L2.5 按段聚合 → L2.6 总量预算 → L3.5 整段折叠 → L4 LLM 摘要。
    每层各自判断要不要出手，最后统一过一遍工具调用配对修复。

    「有改动」vs「LLM 级压缩」的区分：changed 表示任何
    一层动过消息（包括无损层）——调用方只需把消息同步回对话历史；compacted
    特指 L4 的有损摘要——要做重建 prompt / [COMPACT_BOUNDARY] /
    <post_compress_brief> 那一整套后续动作。

    参数：
        messages：完整消息列表
        llm_client：LLM 客户端（L4 用）
        model：模型名
        config：context 配置子字典
        session_state：会话压缩记账簿
        agent_home：CodeAgent 数据目录
        session_id：会话 id（日志/快照用）
        hooks_registry：hook 注册表，可选；非 None 时压缩前后触发
          PRE_COMPACT/POST_COMPACT 事件，PRE_COMPACT 有 hook 要求中止就跳过本次压缩
        tools：当前工具 schema 列表（fork 前缀复用，传给 L4）
        authoritative_tokens：真实 token 锚点 (消息条数, 真实输入 token 数)，
          供混合计数用；None 走全量粗估
        session_store：会话库（可选）；L4 开跑前往库里落 [COMPACT_START]
          事务标记，成功后的 [COMPACT_BOUNDARY]（agent 主类写）当 end 用
        force：True = 发送前预检的强制压缩——L4 触发绕过冷却期
              （上下文已经顶到窗口了，等冷却就是等 PTL），但熔断照守
    返回：(新消息列表, changed, compacted)。

    其他要点：
    - L4 的判定用 session_state 里的记账，避免 L1+L2 循环白白耗掉 L4 的名额
    - L3.5：feature flag context_collapse 开启时按
      占用比例触发，无损可逆、不碰 system 和钉住消息；排在 L4 前当缓冲，
      能省下 LLM 摘要调用
    - 本函数是 async，内部必须 await llm_compact。
    """
    # PRE_COMPACT hook（可 abort）
    if hooks_registry is not None:
        try:
            # hook 链丢到线程池跑（慢 hook 别把流式输出卡住）
            abort = await asyncio.to_thread(
                hooks_registry.run_pre_compact,
                {
                    "session_id": session_id,
                    "layer": "orchestrator",
                },
            )
            if abort.get("abort"):
                logger.info("PRE_COMPACT hook 请求 abort，跳过压缩")
                return messages, False, False
        except Exception as e:
            logger.warning("PRE_COMPACT hook 触发异常（视为允许）: %s", e)

    # ── 单遍 profile：一次扫描算出下面各层触发判定要用的统计量 ──
    # 旧版时间清理/L1/L2/L2.5/L2.6 每层各自全量扫一遍（每轮 4-5 遍
    # O(全部历史)），长会话纯 CPU 叠加可观。这里开头一遍算全，各层触发
    # 判定只读统计量：统计量低于该层阈值 ⟹ 该层原本就是 no-op，跳过其
    # 扫描不改变任何行为；真要出手的层，层内精确定位扫描原样保留
    # （profile 只替触发判断，不替定位）。
    prof = _profile_messages(messages)

    # ── 时间清理（最先跑，不看 token 超没超）──
    # 距最后一条 assistant 超 gap_minutes（默认 60）分钟时，把旧工具结果
    # 内容换成占位（agent_home 在场时先落盘，占位带 full_at 找回指针）。
    # 返回的 c0=True 表示确实清了东西——必须算进
    # 最终 changed，否则 changed=False 会导致对话历史不同步，下一轮又把
    # 原始内容塞回去（等于白清，效果只活一轮）。
    # 触发判定改吃 profile 的 last_assistant_ts：开关关着/没有可用
    # 时间戳/距今不足 gap 分钟时，函数内部原本就在这几步原样返回（清都
    # 不清）——直接不调，省掉它找 assistant 的倒序扫描。真超时就照旧调，
    # 函数内部的 keep_recent/幂等检查原样保留（它只可能清得更少，不会多清）。
    c0 = False
    _tb_last_ts = prof["last_assistant_ts"]
    if (
        config.get("time_based_mc_enabled", True)
        # 与原函数同款真值判定：None/0/"" 都视为「没有可用时间戳」
        and _tb_last_ts
        and (time.time() - _tb_last_ts) / 60
        >= config.get("time_based_mc_gap_minutes", 60)
    ):
        messages, c0 = time_based_clear_old_tool_results(
            messages, config, agent_home=agent_home,
        )

    # L1 裁中间（少频繁裁中间，压缩主要靠 L4 的 token 判定）
    # 触发判定改吃 profile 的 msg_count：时间清理只换内容不动条数，
    # 统计值就是当前 conv 条数——不超过阈值时 snip_compact 内部
    # 「len(conv) <= threshold」检查原样返回，必然 no-op，不调省掉
    # 它的占位扫描；条数过线照旧调（占位幂等/头尾保留检查原样）。
    c1 = False
    _snip_threshold = config.get("snip_message_threshold", 200)
    if prof["msg_count"] > _snip_threshold:
        messages, c1 = snip_compact(
            messages,
            keep_first=config.get("snip_keep_first", 3),
            keep_last=config.get("snip_keep_last", 47),
            threshold=_snip_threshold,
        )

    # L2 单条折叠（按单条大小折叠 + 落盘留指针 + 保最近 3 条）
    # micro_compact 内部自己按大小触发 + 落盘 + 可读回
    # threshold 默认 5 万字符（精细化，小结果不落盘）
    offload_threshold = config.get("output_offload_threshold", 50000)
    offload_preview = config.get("output_offload_preview", 2000)
    offload_freeze = config.get("offload_decision_freeze", True)
    from agent.output_offload import maybe_offload

    # ── 决策冻结预处理 ──
    # 落过盘的工具调用直接照抄上次的预览内容（一字不差，保 prompt cache）
    # 放在 L2 之前——先照抄，L2 看到的就已是预览，不会重复落盘
    c_freeze = False
    if offload_freeze and _offload_decisions:
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            tc_id = m.get("tool_call_id") or ""
            if not tc_id or tc_id not in _offload_decisions:
                continue
            decision = _offload_decisions[tc_id]
            if m.get("content") != decision["preview"]:
                messages[i] = dict(m)
                messages[i]["content"] = decision["preview"]
                c_freeze = True

    # 触发判定改吃 profile 的 max_tool_chars：全场最大单条工具结果都
    # 没超过阈值时，micro_compact 内部逐条「len(content) <= threshold
    # → 跳过」检查全是跳过——必然 no-op，不调省掉全量扫描；任何一条
    # 可能过线照旧调（最近 3 条保护/幂等检查原样保留）。
    # 例外：c0/c_freeze 刚把短内容换成定长占位（清除标记/落盘预览，
    # 净变长），profile 统计的是换之前的长度——这两种情况不跳，保守
    # 走原扫描（出现频率低：一个要求闲置超 1 小时，一个只在重放有差异）。
    c2 = False
    if prof["max_tool_chars"] > offload_threshold or c0 or c_freeze:
        messages, c2 = micro_compact(
            messages,
            threshold=offload_threshold,
            preview_chars=offload_preview,
            keep_recent=config.get("micro_keep_recent_results", 3),
            agent_home=agent_home,
        )
    # 把 micro_compact 新产生的落盘决策也记进表（下次照抄）
    if offload_freeze and c2:
        for m in messages:
            if m.get("role") != "tool":
                continue
            tc_id = m.get("tool_call_id") or ""
            if not tc_id or tc_id in _offload_decisions:
                continue
            content = m.get("content", "")
            if isinstance(content, str) and _already_offloaded(m):
                _record_decision(tc_id, content)

    # ── L2.5 按段聚合预算 ──
    # 按 user 消息边界分组，一段连续工具结果总和 > message_offload_threshold 时，
    # 从最大的开始逐个落盘。比 L2.6 全局预算更精细——L2.6 只看全局总和，
    # 不区分是哪个 user 轮次的工具结果。按段处理在前，L2.6 做最后兜底。
    # 顺序：L2 单条 → L2.5 按段聚合 → L2.6 全局预算
    c_per_msg = False
    msg_threshold = config.get("message_offload_threshold", 200_000)
    # 触发判定改吃 profile 的 total_chars：一段工具结果的总和 ≤ 全部
    # 对话内容总和（段和是全量和的子集，且内容对超预览大消息只减不
    # 增）——全量和都不超「一段」阈值时任何一段必然不超，
    # _enforce_per_message_budget 逐段 continue 必然 no-op，不调省掉
    # 分组扫描；可能过线照旧调，段内挑最大落盘的精确定位原样保留。
    # 守卫必须含 c2：c2 也可能净变长——强制落盘对短消息（内容 ≤
    # preview_chars）是「全文 + JSON 包装」替换（净变长 ~200 字符/条），
    # 低 offload 阈值配置下 prof 统计会低估现场总量（原版现场重算会
    # 超阈值触发压缩），必须放行现场重算。c0/c_freeze 定长占位净变长
    # 的例外见 L2 注释，同理。
    if msg_threshold > 0 and (
        prof["total_chars"] > msg_threshold or c0 or c_freeze or c2
    ):
        c_per_msg = _enforce_per_message_budget(
            messages,
            limit=msg_threshold,
            agent_home=agent_home,
            preview_chars=offload_preview,
            freeze=offload_freeze,
        )
        if c_per_msg:
            logger.info("L2.5 per-message 聚合 offload: 按 user 边界分组落盘")

    # L2.6 总量预算：全部工具结果加起来仍超预算 → 挑最大的再落盘（全局兜底）
    # L2.6 的预算与 message_offload_threshold 语义不同——后者是「一段」的
    #   阈值，前者是全局总量上限；L2.6 读 tool_result_total_budget（默认 20 万字符）
    c26 = False
    TOTAL_TOOL_BUDGET = config.get("tool_result_total_budget", 200_000)
    # 触发判定改吃 profile 的 total_chars：工具结果全局总和 ≤ 全部对话
    # 内容总和（子集 + 内容对超预览大消息只减不增）——全和不超预算时
    # 「总和超预算→挑最大落盘」必然不触发，不扫省掉工具消息全量求和。
    # 三个保守修正：
    #   1. c0/c_freeze 定长占位净变长的例外见 L2 注释，出现就不跳；
    #   2. c2/c_per_msg 也可能净变长——强制落盘对短消息（内容 ≤
    #      preview_chars）是「全文 + JSON 包装」替换（净变长 ~200 字符/条），
    #      低 offload 阈值配置下 prof 统计会低估现场总量（原版现场重算
    #      会超预算触发落盘），必须放行现场重算；
    #   3. 下方原求和对 falsy content（如 None）取 len(str(...)) 会算出
    #      几个字符，profile 记 0——留 16×条数的富余把这个理论差盖死，
    #      等价不靠「实际不会这么配」的运气。
    _c26_slack = 16 * max(1, len(messages))
    tool_indices = (
        [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if c0
        or c_freeze
        or c2
        or c_per_msg
        or prof["total_chars"] + _c26_slack > TOTAL_TOOL_BUDGET
        else []
    )
    if tool_indices:
        tool_total = sum(
            len(str(messages[i].get("content", ""))) for i in tool_indices
        )
        if tool_total > TOTAL_TOOL_BUDGET:
            # 按当前长度排序，最大的先落盘
            sorted_indices = sorted(
                tool_indices,
                key=lambda i: len(str(messages[i].get("content", ""))),
                reverse=True,
            )
            for i in sorted_indices:
                if tool_total <= TOTAL_TOOL_BUDGET:
                    break
                content = messages[i].get("content", "")
                if not isinstance(content, str) or len(content) <= offload_threshold:
                    continue
                if _already_offloaded(messages[i]):
                    continue
                new_content = maybe_offload(
                    content,
                    tool_call_id=messages[i].get("tool_call_id") or f"budget_{i}",
                    agent_home=agent_home,
                    threshold=offload_threshold,
                    preview_chars=offload_preview,
                )
                if new_content != content:
                    messages[i] = dict(messages[i])
                    messages[i]["content"] = new_content
                    tool_total -= len(content) - len(new_content)
                    c26 = True
                    # 记录决策（下轮照抄这次的内容，一字不差）
                    if offload_freeze:
                        tc_id = messages[i].get("tool_call_id") or f"budget_{i}"
                        _record_decision(tc_id, new_content)
                    logger.info(
                        "L2.6 总量预算 offload: tool 消息 %d %d→%d",
                        i, len(content), len(new_content),
                    )

    # L3.5 整段折叠
    # 触发条件：feature flag 开 + 估算 token ÷ 窗口大小 > 占用比（默认 0.8）
    # 无损、可逆（折叠段原文在 .transcripts/latest.jsonl），不碰 system 和钉住消息
    # 排在 L4 前——便宜得多，能挡掉很多 L4 的 LLM 调用
    c35 = False
    from agent.feature_flags import is_feature_enabled, get_feature_config
    if is_feature_enabled(config, "context_collapse"):
        cc_cfg = get_feature_config(config, "context_collapse")
        cc_threshold = cc_cfg.get("threshold_ratio", 0.8)
        # context_window：优先用 config 里显式写的值；没写就按模型名推断
        # （deepseek → 64k、1m → 1M——旧版一律 128k，对 64k 模型同样虚高）
        context_window = config.get("context_collapse_context_window")
        if not context_window:
            from agent.context_compressor import _get_model_max_tokens
            context_window = _get_model_max_tokens(model)
        # 折叠前先把 transcript 快照一份（L3.5 虽是无损的，但保持原文可读回是好习惯）
        # 注意：L3.5 不像 L4 那样有损，这里不强制 force
        messages, c35 = apply_context_collapse(
            messages,
            threshold_ratio=cc_threshold,
            context_window=context_window,
            keep_recent_turns=config.get("context_collapse_keep_recent_turns", 3),
        )
        if c35:
            logger.info(
                "L3.5 contextCollapse 触发（ratio=%.2f, window=%d）",
                cc_threshold, context_window,
            )

    # L4 llm（出手条件：冷却期已过 + 超阈值 + 没熔断）
    # 不设「每会话最多压 N 次」的总量帽——设了的话长会话压满次数后
    # 永久失去 L4，退化成频繁紧急截断；只用冷却期 + 连续失败熔断控制。
    # 成功压缩受冷却期和 token 阈值双重门控，
    # 不会失控烧钱。llm_compact_count 只用于展示统计。
    c4 = False
    cooldown = config.get("llm_compact_cooldown_turns", 5)
    llm_compact_count = session_state.llm_compact_count
    conv_len = len(_split_system(messages)[1])
    # 混合计数（有锚点用 真实值+增量粗估，否则全量粗估）
    est_tokens = estimate_tokens_hybrid(messages, authoritative_tokens)

    # 方向 1：自适应压缩阈值（1M 大窗口模型放宽到 70 万，同时封顶不超过
    # 推断窗口的 90%——细则见 _effective_llm_compact_threshold）
    # 1M 窗口留 30% 给输出（30 万）、70% 给输入（70 万）
    # 压不压完全由 token 决定（接近窗口才压），不按消息条数——
    # 按条数压（如「消息数 > 100 就压」）会让长会话被反复压缩、agent 反复失忆。
    token_threshold = _effective_llm_compact_threshold(config, model)

    # 单轮增长预估：预估 token + 增量 >= 阈值就提前触发。
    # 一轮大工具结果进来会直接把下一轮顶过线，等真到线再压就会
    # 「压完 → 下一轮又到线 → 再压」来回震荡；提前量 = 最近几轮的最大单轮体量。
    growth = estimate_turn_growth(
        messages,
        window=config.get("llm_compact_growth_window", 3),
        default=config.get("llm_compact_growth_default", 8000),
    )
    over_threshold = est_tokens + growth >= token_threshold
    cooldown_ok = session_state.cooldown_ok(cooldown)
    # L4 触发熔断——连续失败达阈值后本会话不再触发（这是「触发层」熔断，
    # 与摘要生成层的熔断互补：前者省掉无效调用，后者降级用规则总结凑合）
    tripped = session_state.llm_compact_failures >= MAX_CONSECUTIVE_L4_FAILURES
    logger.info(
        "L4 trigger check: over_threshold=%s, est_tokens=%d, growth=%d, conv_msgs=%d, "
        "llm_compact_count=%d, cooldown_ok=%s, failures=%d%s",
        over_threshold, est_tokens, growth, conv_len,
        llm_compact_count, cooldown_ok,
        session_state.llm_compact_failures,
        " (TRIPPED)" if tripped else "",
    )
    if over_threshold and (force or cooldown_ok) and not tripped:
        logger.info("L4 triggered")
        # 压缩事务 start 标记：L4 真正开跑前先落盘——进程若在压缩中途
        # 崩溃，会话库里留下"有 start 无更晚 boundary"的悬挂证据
        # （恢复时由 cli 的裁剪逻辑检测并告警，无需修复）
        _persist_compact_marker(session_store, session_id, "[COMPACT_START]")
        # L4 调用前先把 transcript 落盘（force=True 强制快照，因为 L4 是有损的）
        # 快照路径传给 boundary——模型失忆后知道去哪找回被摘要段的原文
        transcript_snapshot_path: Optional[str] = None
        if config.get("transcript_enabled", True):
            try:
                _snap = snapshot_if_needed(
                    messages,
                    agent_home=agent_home,
                    session_id=session_id,
                    force=True,
                    enabled=True,
                    retention=config.get("transcript_retention", 20),
                )
                if _snap is not None:
                    transcript_snapshot_path = str(_snap)
            except Exception as e:
                logger.warning("transcript snapshot 失败（不阻塞 L4）: %s", e)

        messages, c4 = await llm_compact(
            messages,
            llm_client=llm_client,
            model=model,
            keep_recent=config.get("llm_compact_keep_recent", 30),
            token_threshold=token_threshold,  # 自适应阈值
            # 把 est+growth 传进去（提前触发时 est 本身可能还没到阈值，
            # llm_compact 内部门槛用同一个「下一轮预期水位」判定，避免二次拦截）
            precomputed_tokens=est_tokens + growth,
            tools=tools,  # fork 前缀复用要用
            summary_scale_thresholds=config.get("summary_scale_thresholds"),
            summary_files_errors_limits=config.get("summary_files_errors_limits"),
            transcript_path=transcript_snapshot_path,
            session_state=session_state,  # 收敛检查不过时记失败账（防每轮重试白烧）
        )
        if c4:
            session_state.record_llm_compact()
            # 压缩成功：把摘要占位（含 [COMPACT_BOUNDARY] 标记）落进会话库——
            # 恢复时按最后的边界裁掉压缩前旧历史，否则重启全量载入会再次
            # 撑爆上下文。（旧实现靠 agent 主类事后扫历史首条持久化，
            # 前缀对不上从未生效，现在直接在成功现场落库）
            _ph = take_last_compact_placeholder()
            if _ph:
                _persist_compact_marker(session_store, session_id, _ph)
            # 降级产出（LLM 摘要失败 → 用规则总结凑合）算一次触发质量失败——
            # 降级压缩能用但有损，连续降级就该停止触发。
            # _summarize_conversation 永不抛异常（内部自己兜底），失败信号走
            # 模块级的 _last_summary_degraded 标记（压缩在主循环里串行，无并发竞争）。
            from agent.context_compressor import _last_summary_degraded
            if _last_summary_degraded:
                session_state.llm_compact_failures += 1
                logger.warning(
                    "L4 触发但摘要降级（连续失败 %d/%d）",
                    session_state.llm_compact_failures, MAX_CONSECUTIVE_L4_FAILURES,
                )
            else:
                session_state.llm_compact_failures = 0  # 真 LLM 摘要成功则清零
        else:
            # llm_compact 自己拒绝压缩（没过内部门槛，或收敛检查不过——
            # 后者的失败账已在 llm_compact 内部记过，这里不重复计）
            pass
    elif over_threshold:
        if tripped:
            logger.info(
                "L4 skipped: 触发熔断（连续失败 %d 次）",
                session_state.llm_compact_failures,
            )
        else:
            # 总量上限分支已移除——剩下唯一的拦截原因是冷却期未过
            logger.info("L4 skipped: cooldown active (last=%d, current=%d, need=%d)",
                        session_state.last_llm_compact_turn, session_state.current_turn, cooldown)
    else:
        logger.info("L4 skipped: below threshold (est_tokens=%d, conv_msgs=%d)", est_tokens, conv_len)

    changed = c0 or c1 or c_freeze or c2 or c_per_msg or c26 or c35 or c4
    if changed:
        # 终极保险：最后再过一遍工具调用配对修复
        system, conv = _split_system(messages)
        messages = _reassemble(system, _fix_tool_call_pairs(conv))
        # 任一层真改动了消息 → 通知 cache_monitor「下次缓存下降是预期的」
        # （L4/紧急压缩内部已各自通知过；这里补齐时间清理/L1/L2*/L3.5 路径。
        # 幂等标记，重复调用无害）
        try:
            from agent.cache_monitor import notify_compaction
            notify_compaction()
        except Exception as e:
            logger.debug("notify_compaction fail-open: %s", e)

    # POST_COMPACT hook（广播压缩已完成）
    if hooks_registry is not None:
        try:
            # 丢到线程池跑（慢 hook 别卡住事件循环）
            await asyncio.to_thread(
                hooks_registry.run_post_compact,
                {
                    "session_id": session_id,
                    "layer": "orchestrator",
                },
            )
        except Exception as e:
            logger.warning("POST_COMPACT hook 触发异常（忽略）: %s", e)

    return messages, changed, c4
