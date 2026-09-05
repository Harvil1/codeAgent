"""上下文压缩的工具函数集（被 context_pipeline 压缩管线调用的零件库）。

压缩的编排调度在 agent/context_pipeline.py 的多层管线里，
本文件是管线要复用的零件：
    - ``_summarize_conversation``：调 LLM 做摘要（9 段式格式 + 超长重试 + 熔断器）
    - ``_rule_based_summary``：LLM 不可用时的降级方案（机械抽取，不花钱）
    - ``_fix_tool_call_pairs``：修复压缩切坏了的工具调用配对
    - ``estimate_message_tokens``：按字符数粗估 token
    - ``_strip_analysis_draft``：剥掉 LLM 回复里的 <analysis> 草稿区
    - ``reset_compact_circuit_breaker``：新会话开始时重置熔断器
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 9 段式结构化摘要 prompt 模板（下面这个字符串是代码，别改内容）
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT_9SECTION = """请把以下对话总结成 9 段结构化摘要。

**必须按以下 9 段输出，每段不能省略**：

1. **Primary Request and Intent**：用户的核心请求和意图
2. **Key Technical Concepts**：涉及的关键技术概念、库名、API
3. **Files and Code Sections**：涉及的文件路径（**逐字保留**）+ 关键代码段
4. **Errors and fixes**：遇到的错误（**逐字保留错误消息**）+ 修复方法
5. **Problem Solving**：问题解决过程、调试思路
6. **All user messages**：所有用户消息原文（**逐字保留**，不能改写；
   纯短确认类消息（≤8 字且无信息量，如「继续」「好」「ok」）可合并
   折叠成一行「…[已省略 N 条短确认]」省空间——长任务几百条确认
   全量罗列会把摘要撑爆且无信息量）
7. **Pending Tasks**：待办任务、未完成的工作
8. **Current Work**：当前正在做什么
9. **Optional Next Step**：可选的下一步

**铁律**：
- 文件路径、命令、错误消息、用户原话必须**逐字保留**（不能省略/改写）
- 第 6 段（All user messages）只能收录 user 角色消息的原文；user 未明说、
  由 assistant 推测或推断出的内容**不得**写成用户说过的话——如需保留必须
  明确标注"（assistant 推断）"
- 用 markdown 格式
- 第 3/4 段（Files and Code Sections / Errors and fixes）各不超过 {files_errors_limit} 字；其余每段不超过 200 字（user messages 段保留原文）

对话内容：
{dialog}
"""


# ---------------------------------------------------------------------------
# 熔断器状态（模块级全局；熔断=连续失败太多次就暂停调 LLM）
# ---------------------------------------------------------------------------

_consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 3
_compact_circuit_open = False

# 最近一次摘要是否「降级产出」（LLM 失败 → 用规则总结凑合）。
# 调度层（compress_if_needed）读它判定 L4 触发质量失败——降级摘要虽然能用
# 但有损，连续降级就该停触发 L4。
# 放模块级全局：压缩在主循环里串行执行，不存在并发竞争；子代理各有独立模块态可接受。
_last_summary_degraded = False

# PTL（prompt_too_long）重试上限
MAX_PTL_RETRIES = 3

# Files/Errors 段字数分档：被摘要消息数过阈值就放宽——长任务一屏
# 路径+报错 200 字装不下，砍了下次压缩就找不回来（锚定段只保跨代不丢）
_DEFAULT_SCALE_THRESHOLDS = (60, 150)
_DEFAULT_FILES_ERRORS_LIMITS = (200, 400, 600)


def _files_errors_limit(msg_count: int, thresholds=None, limits=None) -> int:
    """按被摘要消息数算 Files/Errors 段的字数上限（纯函数）。

    参数：
        msg_count：本次被摘要的消息条数
        thresholds：分档阈值（默认 (60, 150)）
        limits：各档上限（默认 (200, 400, 600)）
    返回：字数上限。
    """
    th = tuple(thresholds) if thresholds else _DEFAULT_SCALE_THRESHOLDS
    lm = tuple(limits) if limits else _DEFAULT_FILES_ERRORS_LIMITS
    if msg_count <= th[0]:
        return lm[0]
    if msg_count <= th[1]:
        return lm[1]
    return lm[2]


# ---------------------------------------------------------------------------
# 媒体块剥离（防压缩请求自己被图片撑爆报超长）
# ---------------------------------------------------------------------------

# 图片块类型（OpenAI 的 image_url / Anthropic 的 image / 新式 input_image）
_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image", "input_image"})
# 文档/文件块类型
_DOC_BLOCK_TYPES = frozenset({"document", "file", "input_file"})


def strip_media_blocks(messages: list) -> list:
    """把消息里的图片/文档块换成文本标记。

    为什么：压缩调用本身也受上下文长度限制，带图片的多模态消息很容易把
    「用来压缩的请求」自己撑爆（报 prompt_too_long）。

    做法：content 是块列表（多模态消息）时，图片块 → "[image]"、文档/文件块 →
    "[document]"，文字块保留；全部变文字后合并成纯字符串 content（下游的
    对话排版函数只认字符串）。content 本来就是字符串的消息原样不动。
    不改入参（有改动的消息复制新 dict）。

    与前缀复用的配合：没有多模态消息时本函数等于什么都没做（前缀
    逐字节一致，缓存照常命中）；有图片时剥掉（缓存 miss 可接受——总比压缩
    调用自己爆掉强）。

    参数：
        messages：消息列表
    返回：替换后的消息列表。
    """
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_blocks = []
        for b in content:
            if isinstance(b, dict):
                btype = b.get("type", "")
                if btype in _IMAGE_BLOCK_TYPES:
                    new_blocks.append({"type": "text", "text": "[image]"})
                    continue
                if btype in _DOC_BLOCK_TYPES:
                    new_blocks.append({"type": "text", "text": "[document]"})
                    continue
            new_blocks.append(b)
        # 全是文字块时合并成纯字符串（消除块列表形态）
        if all(
            isinstance(b, dict) and b.get("type") == "text"
            for b in new_blocks
        ):
            new_content = "\n".join(
                str(b.get("text", "")) for b in new_blocks
            )
        else:
            new_content = new_blocks
        if new_content != content:
            msg = {**msg, "content": new_content}
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# 摘要锚定：多次压缩时防"摘要被再摘要"的代际损耗
# ---------------------------------------------------------------------------

# 三段"逐字保留"内容是长任务的生命线（文件路径/错误消息/用户原话），
# 第二次压缩起它们由锚定机制原样传递，不再经过 LLM 重写
_ANCHOR_SECTION_TITLES = (
    "Files and Code Sections",
    "Errors and fixes",
    "All user messages",
)

_ANCHOR_SECTION_RE = re.compile(r"(?:^|\n)\s*(?:\d+\.\s*)?\*\*([^*\n]+)\*\*")


def extract_summary_anchor(old_summary_text: str) -> str:
    """从上一次压缩的摘要里提取要逐字保留的三段（文件/错误/用户原话）。

    9 段摘要每段标题形如 "3. **Files and Code Sections**：..."——
    按加粗标题切块，挑出三段关键内容原文。锚定头（"**Files and Code
    Sections**：..."）也是同样格式，二次锚定时能被再次提取（累积有界，
    不翻倍）。

    参数：
        old_summary_text: 旧摘要全文（placeholder 消息的 content）

    返回：三段原文拼接（markdown）；一段都提不出来返回空串。
    """
    if not old_summary_text:
        return ""
    matches = list(_ANCHOR_SECTION_RE.finditer(old_summary_text))
    if not matches:
        return ""
    sections = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(old_summary_text)
        sections.setdefault(title, old_summary_text[start:end].strip())
    parts = []
    for t in _ANCHOR_SECTION_TITLES:
        body = sections.get(t, "")
        if body:
            parts.append(f"**{t}**：{body}")
    return "\n\n".join(parts)


async def _summarize_conversation(
    messages: list,
    llm_client=None,
    *,
    model: str = None,
    summary_model: str = None,
    session_memory: str = None,
    from_idx: int = 0,
    up_to_idx: int = -1,
    fork_prefix_messages: list = None,
    tools: list = None,
    anchor_note: str = "",
    scale_thresholds=None,
    files_limits=None,
) -> str:
    """调 LLM 把一段对话历史总结成摘要文本（async；9 段式固定格式）。

    L4 压缩的核心动作，带四套保命机制：
    - **9 段式结构化 prompt**：强制逐字保留文件路径/错误消息/用户原话
    - **PTL 重试**：摘要请求自己报 prompt_too_long 时，丢掉一部分旧消息再试
      （最多 MAX_PTL_RETRIES 次）
    - **熔断器**：连续 MAX_CONSECUTIVE_FAILURES 次失败后不再调 LLM，直接走规则总结
    - **session_memory 替代**：有预提取的会话记忆就直接用，一次 LLM 都不调

    参数：
        messages：要摘要的对话段
        llm_client：LLM 客户端；None 时走规则总结（不花钱的降级）
        model：主模型名
        summary_model：摘要专用模型（作为 model 的兜底——只传它时才生效；
            两者都传时 model 优先。此优先级是锁定 spec，见
            tests/test_summarize_9section.py「应优先 model」用例）
        session_memory：预提取的会话记忆；非空就直接返回它当摘要
        from_idx：从第几条开始摘要（默认 0 = 从头）
        up_to_idx：摘要到第几条为止（默认 -1 = 到末尾）——两者配合实现局部压缩
          （只摘要中间一段）；局部段不足 2 条返回空串不压
        fork_prefix_messages：完整对话（含 system）。有值且没配 summary_model 时，
          摘要请求 = 完整对话前缀 + 追加一句摘要指令（tools 与主调用相同）——
          请求开头和主对话一模一样，能命中服务商的前缀缓存，省一次全量缓存写入；
          不设 max_tokens（设了参数不一致会破缓存）。fork 失败/空回复
          就降级走独立调用（前缀已失效可接受）；配了 summary_model 时不用 fork
          （专用小模型和主对话的缓存空间不同，用户显式配置优先）
        tools：当前工具 schema 列表，fork 请求带上（保持和主调用一致）
    返回：摘要文本；拿不到 LLM 摘要时返回规则总结的降级版本。
    """
    global _consecutive_failures, _compact_circuit_open, _last_summary_degraded
    _last_summary_degraded = False  # 每次调用先重置

    # 局部提取（from_idx/up_to_idx）
    is_partial = from_idx != 0 or up_to_idx != -1
    effective_up_to = len(messages) if up_to_idx < 0 else up_to_idx
    to_summarize = messages[from_idx:effective_up_to]
    if is_partial and len(to_summarize) < 2:
        logger.info(
            "partial compact: 段消息数 %d < 2，跳过压缩",
            len(to_summarize),
        )
        return ""

    # 1. 熔断器检查（连续失败达上限 → 直接走规则总结，不再花钱调 LLM）
    if _compact_circuit_open:
        logger.warning(
            "摘要熔断器开启（连续 %d 次失败），跳过 LLM 摘要",
            _consecutive_failures,
        )
        _last_summary_degraded = True  # 标记这次是降级产出
        return _rule_based_summary(to_summarize)

    # 2. 预提取的会话记忆优先（有就直接用，省一次 LLM 调用）
    if session_memory and session_memory.strip():
        logger.info("用 session memory 替代 LLM 摘要")
        return session_memory

    # 3. 无客户端时走规则总结
    if llm_client is None:
        return _rule_based_summary(to_summarize)

    # 4. 排版对话 + 拼 9 段式 prompt（用 to_summarize 段，不是全量 messages）
    # 先剥媒体块（image/document → [image]/[document] 文本标记），
    # 防多模态消息把压缩调用自己撑爆；纯文本消息不受影响。
    working_messages = strip_media_blocks(to_summarize)  # 不改入参（PTL 重试还要再切片）
    fe_th = tuple(scale_thresholds) if scale_thresholds else _DEFAULT_SCALE_THRESHOLDS
    fe_lm = tuple(files_limits) if files_limits else _DEFAULT_FILES_ERRORS_LIMITS
    dialog = _format_dialog_for_summary(working_messages)
    prompt = SUMMARIZE_PROMPT_9SECTION.format(
        dialog=dialog,
        files_errors_limit=_files_errors_limit(len(working_messages), fe_th, fe_lm),
    )
    # 锚定提示：三段关键内容由调用方原样拼接，LLM 不必重复罗列（省输出 + 防重写矛盾）
    if anchor_note:
        prompt += "\n\n" + anchor_note

    # === fork 前缀复用（先试它，失败降级独立调用路径）===
    if fork_prefix_messages and not summary_model:
        fork_messages = strip_media_blocks(fork_prefix_messages) + [
            {"role": "user", "content": prompt},
        ]
        try:
            response = await llm_client.chat_completions(
                fork_messages, tools=tools,
            )
            summary = response.choices[0].message.content or ""
            if summary.strip():
                # 成功：重置熔断器（与独立调用路径同一套语义）
                _consecutive_failures = 0
                _compact_circuit_open = False
                logger.info(
                    "fork 摘要成功（前缀 %d 条消息复用主对话缓存）",
                    len(fork_prefix_messages),
                )
                return _strip_analysis_draft(summary)
            logger.warning("fork 摘要响应为空（降级独立调用）")
        except Exception as e:
            # fork 失败不计入熔断——后面的独立路径带完整的 PTL 重试与熔断逻辑
            logger.warning("fork 摘要失败（降级独立调用）: %s", e)

    # 5. PTL 重试（最多 MAX_PTL_RETRIES 次）
    # 独立调用路径：传 system message（你是技术对话摘要助手）+ model 参数
    # （model 优先于 summary_model，由 tests/test_summarize_9section.py 锁定）。
    # OpenAICompatClient 会把 model kwarg 弹掉、用自己构造时绑定的模型，
    # 但 aux_llm_router 或未来的其他 client 实现可能用外部传入的 model——保持一致。
    summary_system_prompt = "你是技术对话摘要助手。"
    effective_model = model or summary_model
    for retry in range(MAX_PTL_RETRIES + 1):
        try:
            response = await llm_client.chat_completions(
                [
                    {"role": "system", "content": summary_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=effective_model,
            )
            summary = response.choices[0].message.content or ""
            # 成功：重置熔断器
            _consecutive_failures = 0
            _compact_circuit_open = False
            # 剥掉 <analysis> 草稿（模型内部推敲，不属于最终摘要）
            return _strip_analysis_draft(summary)
        except Exception as e:
            err_str = str(e).lower()
            is_ptl = "prompt_too_long" in err_str or "context_length" in err_str
            if is_ptl and retry < MAX_PTL_RETRIES:
                # 用 tokenGap 精确算法计算该丢多少条
                # _compute_ptl_drop_count 内部在错误消息解析不出数字时也会退回 20%
                drop_count = _compute_ptl_drop_count(
                    working_messages, str(e),
                    model_max_tokens=_get_model_max_tokens(effective_model or ""),
                )
                old_drop = max(1, len(working_messages) // 5)
                working_messages = working_messages[drop_count:]
                dialog = _format_dialog_for_summary(working_messages)
                # 丢消息后按当时的实际条数重算分档
                prompt = SUMMARIZE_PROMPT_9SECTION.format(
                    dialog=dialog,
                    files_errors_limit=_files_errors_limit(
                        len(working_messages), fe_th, fe_lm,
                    ),
                )
                logger.warning(
                    "PTL 重试 %d/%d：tokenGap 精确算法丢 %d 条（旧 20%% 会丢 %d 条）",
                    retry + 1, MAX_PTL_RETRIES, drop_count, old_drop,
                )
                continue
            # 其他错误、或 PTL 重试耗尽 → 走规则总结 + 熔断器计数加一
            _consecutive_failures += 1
            if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _compact_circuit_open = True
                logger.error(
                    "摘要熔断器开启（连续 %d 次失败）",
                    _consecutive_failures,
                )
            logger.warning("LLM 摘要失败（降级规则总结）: %s", e)
            _last_summary_degraded = True  # 标记这次是降级产出
            return _rule_based_summary(working_messages)

    return _rule_based_summary(working_messages)


def _head_tail(text: str, n: int = 300) -> str:
    """取文本头尾各 n 字符，中间用省略标注代替（错误堆栈常在尾部）。

    参数：
        text：原文
        n：头/尾各取的字符数
    返回：头尾拼接文本（不够长就原样返回）。
    """
    if not isinstance(text, str) or len(text) <= 2 * n:
        return text
    return f"{text[:n]} …[中间 {len(text) - 2 * n} 字符省略]… {text[-n:]}"


def _compact_tool_result(content) -> str:
    """工具结果进摘要输入前的压形：占位 JSON 提关键字段，普通文本头尾截取。

    大白话：已经落盘的大结果是一张「提货单」（JSON，带 full_at 指针），
    摘要器最需要的就是这张单子上的指针——直接头尾截会把指针截丢，
    所以占位单独把 error/full_at/hint 等关键字段挑出来。

    参数：
        content：工具结果的原始 content
    返回：压形后的文本。
    """
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and ("full_at" in parsed or "error" in parsed):
                keys = ("error", "full_at", "truncated", "orig_chars", "hint")
                picked = {k: parsed[k] for k in keys if k in parsed}
                if "preview" in parsed:
                    picked["preview"] = _head_tail(str(parsed["preview"]), 150)
                return json.dumps(picked, ensure_ascii=False)
        except ValueError:  # JSONDecodeError 本就是 ValueError 子类
            pass
    return _head_tail(text, 300)


def _format_dialog_for_summary(messages: list) -> str:
    """把消息列表排版成给摘要 LLM 看的纯文本对话（原料保真版）。

    摘要 prompt 要求「文件路径/错误消息逐字保留」，原料里就得真有这些：
    - 工具调用带参数（read_file(path=…) 紧凑形式，截 300 字符）——
      文件路径、命令、搜索词都在参数里
    - 工具结果取头 300 + 尾 300 字符（旧版只留头 200，报错堆栈在尾部全丢）
    - 已落盘的大结果（offload 占位 JSON）提取 full_at/error 等关键字段——
      「去哪找回原文」的指针不能被截断丢掉
    - _ephemeral 瞬时消息（后台任务提醒等）不进摘要，省摘要预算

    参数：
        messages：消息列表
    返回：排版好的对话文本。
    """
    formatted = []
    for msg in messages:
        if msg.get("_ephemeral"):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "tool":
            formatted.append(f"[工具结果] {_compact_tool_result(content)}")
        elif role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                formatted.append(
                    f"[助手调用了工具: {fn.get('name', '?')}({_head_tail(args, 300)})]"
                )
            if content:
                formatted.append(f"[助手补充] {_head_tail(content, 200)}")
        else:
            formatted.append(f"[{role}] {content}")
    return "\n\n".join(formatted)


def _strip_analysis_draft(summary: str) -> str:
    """剥掉回复里的 <analysis>…</analysis> 草稿区（模型推敲用的，不是正文）。

    某些模型（如 Claude）习惯先写一段 <analysis>思考过程</analysis>
    再给答案，这段内部推理不属于摘要内容，剥掉。

    参数：
        summary：LLM 原始回复文本
    返回：剥掉草稿区后的文本。
    """
    return re.sub(r'<analysis>.*?</analysis>\s*', '', summary, flags=re.DOTALL)


def reset_compact_circuit_breaker() -> None:
    """新会话开始时重置熔断器（别让上一场的失败计数殃及这一场）。

    在 AIAgent.__init__ 调用。
    """
    global _consecutive_failures, _compact_circuit_open
    _consecutive_failures = 0
    _compact_circuit_open = False


def _rule_based_summary(messages: list) -> str:
    """降级方案：不调 LLM，机械地把对话要点抽出来拼成文本。

    保留所有 user 消息 + 有文字内容的 assistant 消息（纯工具调用消息跳过），
    各截 150 字，整体限 1500 字。质量差但零成本、永远不会失败。

    参数：
        messages：消息列表
    返回：拼好的降级摘要文本。
    """
    lines = ["[规则提取（LLM 不可用）]"]
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:150]
        if role == "user" and content:
            lines.append(f"- 用户: {content}")
        elif role == "assistant" and content and not msg.get("tool_calls"):
            lines.append(f"- 助手: {content}")
    return "\n".join(lines)[:1500]


def _fix_tool_call_pairs(messages: list) -> list:
    """修复压缩切坏了的「工具调用 ↔ 工具结果」配对。

    协议要求：assistant 发出的每个工具调用（tool_calls 里的 id），后面必须
    跟着对应 id 的工具结果（tool 消息）；反过来，工具结果前面必须有发起它的
    工具调用。压缩裁切很容易把一对拆散，拆散了 API 直接报 400。

    处理两类问题：
    - 正向孤儿（调用了但没结果）：在 assistant(tool_calls) 后面的结果序列末尾
      补一条假结果（内容注明「因压缩丢失」）。⚠️ 位置必须紧贴结果序列末尾，
      **不能**补到整个消息列表最后——中间隔了别的消息照样违反「紧跟其后」
      （Anthropic 查得最严），还是 400
    - 反向孤儿（有结果但前面没发起它的调用）：直接删掉
      （压缩边界或上游 bug 都可能造出这种孤儿）

    参数：
        messages：消息列表
    返回：修复配对后的消息列表。
    """
    # 第一遍：收集全部出现过的工具调用 id（当「全场已知 id 集合」用；
    # 反向孤儿判定实际用下面的 seen_so_far）
    seen_tool_call_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                seen_tool_call_ids.add(tc.get("id"))

    # 反向孤儿要按「截至当前位置见过哪些 id」判断（逐步累加）；
    # 用全量集合会漏判错序——结果 B 出现在发起 B 的调用之前也会放行
    # （B 已在全量集合里），导致 API 400。
    seen_so_far = set()

    # 第二遍：单遍扫描重建列表，pending 记录当前 assistant(tc) 还缺哪些结果 id。
    # 一遇到非工具消息（user/system/新 assistant）就立刻把缺的补上，
    # 保证假结果紧跟在前一个结果序列末尾（也就是发起调用的 assistant 后面）。
    fixed = []
    pending_tool_calls = {}  # {id: name}：当前 assistant(tc) 还缺结果的调用 id
    dropped_orphans = 0
    MAX_PENDING_REPAIR = 100

    def _flush_pending():
        nonlocal pending_tool_calls
        if not pending_tool_calls:
            return
        if len(pending_tool_calls) > MAX_PENDING_REPAIR:
            logger.warning(
                "_fix_tool_call_pairs: 未配对 tool_call 数 %d 超过上限 %d，仅补全前 %d 条",
                len(pending_tool_calls), MAX_PENDING_REPAIR, MAX_PENDING_REPAIR,
            )
            pending_tool_calls = dict(list(pending_tool_calls.items())[:MAX_PENDING_REPAIR])
        for call_id, name in pending_tool_calls.items():
            fixed.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps({
                    "note": "此工具调用的结果因上下文压缩而丢失",
                }, ensure_ascii=False),
            })
        pending_tool_calls = {}

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # 新的 assistant(tc) → 前一个结果序列结束，先把旧 pending 补完
            _flush_pending()
            for tc in msg["tool_calls"]:
                tid = tc.get("id")
                if tid:
                    pending_tool_calls[tid] = tc["function"]["name"]
                    # 累加到 seen_so_far（之后的位置才能引用这个 id）
                    seen_so_far.add(tid)
            fixed.append(msg)
        elif msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            # 反向孤儿按「截至当前位置」判断（seen_so_far），不用全局集合
            if call_id not in seen_so_far:
                # 反向孤儿：有结果但前面没发起它的调用 → 删掉
                dropped_orphans += 1
                continue
            pending_tool_calls.pop(call_id, None)
            fixed.append(msg)
        else:
            # 非工具消息：结果序列结束 → 立刻把 pending 补完
            _flush_pending()
            fixed.append(msg)

    # 末尾可能还剩 pending（最后一条正好是 assistant(tc)，后面没消息了）
    _flush_pending()

    if dropped_orphans:
        logger.warning(
            "_fix_tool_call_pairs: 删除 %d 条孤儿 tool result（无对应 tool_calls）",
            dropped_orphans,
        )

    return fixed


def _estimate_text_tokens(text: str) -> int:
    """单段文本的 CJK 感知 token 估算（len 和 encode 都是 C 级速度）。

    大白话：utf-8 里一个汉字占 3 个字节、一个 ASCII 字符占 1 个字节，
    「字节数 − 字符数」再除以 2 就是汉字个数（每个汉字多占 2 字节）。
    汉字按保守上界 1 token/字计（旧公式统一 ÷3 对纯中文低估约一半，
    低估 → 压得太晚 → 撞上限紧急截断丢信息），ASCII 维持 4 字符 1 token。

    参数：
        text：待估文本
    返回：估算 token 数。
    """
    if not text:
        return 0
    chars = len(text)
    extra_bytes = len(text.encode("utf-8", errors="ignore")) - chars
    cjk = max(0, extra_bytes // 2)
    return cjk + (chars - cjk) // 4


def estimate_message_tokens(messages: list) -> int:
    """按字符构成粗估 token（CJK 感知版，见 _estimate_text_tokens）。

    不追求精确，只求方向正确——压缩阈值的判定用它。

    参数：
        messages：消息列表
    返回：估算的 token 总数。
    """
    total_tokens = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        total_tokens += _estimate_text_tokens(content)
        for tc in msg.get("tool_calls", []) or []:
            args = tc.get("function", {}).get("arguments", "")
            if not isinstance(args, str):
                args = str(args)
            total_tokens += _estimate_text_tokens(args)
    return total_tokens


# ---------------------------------------------------------------------------
# PTL（对话超长报错）的 tokenGap 精确丢条数算法
# ---------------------------------------------------------------------------


def _get_model_max_tokens(model_name: str) -> int:
    """按模型名猜它的上下文窗口大小（简化查表，覆盖常见服务商）。

    参数：
        model_name：模型名字符串
    返回：窗口 token 数。规则：``[1m]``/``1m`` 后缀 → 1M（Claude 扩展上下文）；
    v4/deepseek → 65536；claude-3-5/sonnet/haiku → 200000；其他默认 64000。
    """
    if not model_name:
        return 64000
    name = model_name.lower()
    if "[1m]" in name or "1m" in name:
        return 1_000_000
    if "v4" in name or "deepseek" in name:
        return 65536
    if "claude-3-5" in name or "sonnet" in name or "haiku" in name:
        return 200000
    return 64000


def _compute_ptl_drop_count(
    messages: list,
    error_msg: str,
    model_max_tokens: int = 64000,
    safety_margin: float = 0.85,
) -> int:
    """报 prompt_too_long 之后，精确算出该丢几条消息再重试。

    tokenGap 算法：
      1. 从报错文本里提取 token 上限和实际 token 数（能解析出来的话）
      2. 算预算（上限 × safety_margin，留 15% 给输出等开销）
      3. 算超了多少
      4. 按平均每条大小换算成要丢的条数（多丢 10% 保险，免得丢完还超）

    兜底：报错文本解析不出数字时，退回「丢 20%」算法（max(1, len // 5)）。
    保护：无论算出丢多少，至少留 2 条（不能丢光）。

    参数：
        messages：当前消息列表
        error_msg：API 报的原始错误文本
        model_max_tokens：模型窗口上限（错误里解析不出来时用）
        safety_margin：安全系数（默认 0.85）
    返回：该丢弃的消息条数。
    """
    # 兜底：错误消息解析不了（空 / 不含数字）退回「丢 20%」
    if not error_msg:
        return max(1, len(messages) // 5)

    err_lower = error_msg.lower()

    # 检测是不是「对话超长」类错误（不是的话走兜底）
    is_ptl = (
        "prompt_too_long" in err_lower
        or "context_length" in err_lower
        or "too long" in err_lower
        or "maximum context" in err_lower
    )
    if not is_ptl:
        return max(1, len(messages) // 5)

    # 尝试提取 token 数——三种服务商的报错格式：
    # DeepSeek: "prompt_too_long: input 75000 > 65536"
    # Anthropic: "prompt is too long: 75000 > 64000"
    # OpenAI: "maximum context length is 65536 tokens, however you requested 75000"
    #
    # 策略：
    #   actual = "(\d+) >"  或  "requested (\d+)"
    #   limit  = "> (\d+)"  或  "max... is (\d+)" 或 "max...[:=]\s*(\d+)"
    actual_tokens = None
    limit_tokens = None

    # actual: "75000 >" 格式（DeepSeek / Anthropic）
    m = re.search(r"(\d{3,7})\s*>", error_msg)
    if m:
        actual_tokens = int(m.group(1))

    # actual: "requested 75000" 格式（OpenAI）
    if actual_tokens is None:
        m = re.search(r"requested\s+(\d{3,7})", err_lower)
        if m:
            actual_tokens = int(m.group(1))

    # limit: "> 65536" 格式（DeepSeek / Anthropic）
    m = re.search(r">\s*(\d{3,7})", error_msg)
    if m:
        limit_tokens = int(m.group(1))

    # limit: "is 65536 tokens" 或 "max... 65536" 格式（OpenAI）
    if limit_tokens is None:
        m = re.search(r"(?:is|max[a-z_]*)\s*[:=]?\s*(\d{3,7})", err_lower)
        if m:
            limit_tokens = int(m.group(1))

    # 提取不到 → 退回「丢 20%」
    if actual_tokens is None and limit_tokens is None:
        return max(1, len(messages) // 5)

    # 能提取到就用提取值覆盖默认
    if limit_tokens:
        model_max_tokens = limit_tokens
    if actual_tokens is None:
        actual_tokens = estimate_message_tokens(messages)

    # 算预算（留 15% 给输出和 prompt 本身的开销）
    budget = int(model_max_tokens * safety_margin)
    overflow = max(0, actual_tokens - budget)

    if overflow == 0:
        return 1  # 兜底丢 1 条（报了超长但算出来没超——可能是安全系数太严）

    # 算平均每条消息多大
    msg_count = max(1, len(messages))
    avg_tokens_per_msg = actual_tokens / msg_count

    # 算要丢多少条（多丢 10% 保险，避免丢完还超、连续报错）
    drop_count = int(overflow / avg_tokens_per_msg * 1.1) + 1

    # 保护：至少留 2 条
    return min(drop_count, max(1, len(messages) - 2))
