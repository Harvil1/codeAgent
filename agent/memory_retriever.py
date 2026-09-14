"""相关记忆检索器：每次调主 LLM 之前跑一次。

记忆（AI 对用户/项目沉淀下来的事实条目，跨会话保留）不能全塞给 LLM，
得先挑出和当前问题最相关的几条。本文件就是那个"挑"的环节：

输入：当前用户消息 + 记忆索引（MEMORY.md 目录）
输出：最相关的 N 个记忆 ID 列表

失败策略是 fail-open（出了问题就当没有）：LLM 超时、返回坏 JSON、
空响应等任何异常都返回空列表，绝不影响主对话。

两个反噪音设计：
- active_tools（正在使用的工具）：不召回这些工具的"用法文档"类记忆
  （用户正在用，不需要教程）；"坑/警告"类仍然召回
- exclude_ids（跨轮去重）：之前轮次已注入过的记忆不再占名额
"""
import json
import logging
import re
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


RETRIEVAL_PROMPT_TEMPLATE = """你是记忆检索助手。当前用户消息：

<query>
{query}
</query>

可用记忆索引（每行一条）：

<index>
{index_text}
</index>
{active_tools_rule}{exclude_rule}
返回最多 {max_results} 条与当前 query 最相关的记忆 ID（从索引行的 `.memory/{{topic}}.jsonl#{{uid}}` 路径中提取 `{{topic}}#{{uid}}` 部分）。
格式：JSON 数组，元素是 ID 字符串。例如：["general#1720870000000a1b2c3", "debugging#1720870000000d4e5f6"]
只返回 JSON 数组，不要其他文本。若无相关的，返回 []。

年龄标注说明：每条索引行末尾的 [age: Nd] 表示该记忆最后更新距今天数（[age: unknown] 表示未知）。
排序规则：相关性同等的两条记忆，优先返回更新的那条（N 更小）；新旧记忆内容冲突时，
新记忆优先，旧记忆只作历史背景（例如"用户在用 React 16"是旧信息，"已升到 React 19"是新信息，应返回后者）。
"""

# 设计取舍：用户正在用的工具，不需要再教用法；而且查询里带工具名 +
# 记忆描述里也带工具名，纯关键词匹配会假阳性（看着相关其实没用）。
# 但"这工具有坑"的记忆此时反而最有价值，所以要保留。"坑/警告"类放行。
_ACTIVE_TOOLS_RULE = """

当前对话正在使用这些工具：{tools}
选择规则：**不要**选择这些工具的「用法/操作指南/如何使用」类记忆；
这些工具的「坑/警告/注意事项/已知问题」类记忆**仍然可选**。
"""

# 设计取舍：同一批记忆反复注入只是浪费上下文名额
_EXCLUDE_RULE = """

以下记忆 ID 已在之前轮次注入过对话，不要重复选择（除非 query 与之强相关
且此前注入的信息明显不完整）：
{excluded}
"""


def balanced_truncate_index(index_text: str, limit: int = 25000) -> str:
    """索引超长时按主题分节均衡截取，而不是从头硬砍。

    硬砍的问题：排在索引后面的主题（新主题/项目区主题）对检索彻底隐身，
    而且模型不知道"还有更多没看到"。均衡截取让每个主题都留代表条目，
    并在尾部标注可用 memory_recall 深查。

    参数：
    - index_text：完整索引文本（MEMORY.md 结构，每主题一节 "## xxx"）
    - limit：总字符预算

    返回：截取后的文本（末尾带截断标注）；不超限时原样返回。
    """
    if len(index_text) <= limit:
        return index_text
    chunks = re.split(r"(?m)^(?=## )", index_text)
    head = ""
    sections = []
    for c in chunks:
        if c.lstrip().startswith("## "):
            sections.append(c)
        elif c.strip():
            head = c
    if not sections:
        return index_text[:limit] + "\n…（索引超长已截断，可用 memory_recall 深查完整记忆库）"
    budget = max(1000, limit - len(head) - 200)
    per = max(500, budget // len(sections))
    out = [head] if head else []
    used = 0
    for s in sections:
        take = s if len(s) <= per else s[:per] + "\n…（本主题截断）\n"
        out.append(take)
        used += len(take)
        if used >= budget:
            break
    out.append(
        "\n（索引超长已按主题均衡截取；记忆库可能还有未列出的条目，"
        "可用 memory_recall 工具深查）"
    )
    return "".join(out)


def annotate_index_with_age(index_text: str, link_age_days: dict) -> str:
    """给索引的每一行末尾加上年龄标注 `[age: Nd]`（防召回过期记忆）。

    LLM 分不清哪条记忆是三年前的哪条是今天的，把"这记忆几天没更新了"
    写在行尾让它自己判断新旧。

    参数：
    - index_text：完整的索引文本（和 MEMORY.md 正文同构，每行一条记忆）
    - link_age_days：markdown 链接路径 → 年龄天数的映射（查不到或值为 None
      就标 unknown）

    返回：加了年龄标注的索引文本。纯 prompt 层改造——不改存储格式；
    没有链接的行（如标题）原样保留。任何解析问题只是不标注（fail-open）。
    """
    out_lines = []
    for line in index_text.splitlines():
        match = re.search(r"\(([^()]+\.jsonl#[^()]+)\)", line)
        if not match:
            out_lines.append(line)
            continue
        days = link_age_days.get(match.group(1))
        age = f"{days}d" if isinstance(days, int) and days >= 0 else "unknown"
        out_lines.append(f"{line} [age: {age}]")
    return "\n".join(out_lines)


async def retrieve_relevant(
    *,
    query: str,
    index_text: str,
    llm_client,
    model: str,
    max_results: int = 5,
    active_tools: Optional[List[str]] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> List[str]:
    """让 LLM 从索引里挑出最相关的 N 个记忆 ID。失败返回空列表。

    本函数必须是 async 并 await 底层调用：用同步方式调 async 方法
    拿到的是 coroutine 对象，会被 except 捕获后静默返回空列表——
    记忆检索看起来正常其实一直没工作。

    参数：
    - query：当前用户消息（检索的依据）
    - index_text：记忆索引全文
    - llm_client：LLM 客户端（chat_completions 已是 async）
    - model：检索用的模型名
    - max_results：最多返回几个 ID（默认 5）
    - active_tools：当前对话正在使用的工具名列表（反噪音：
      非空时 prompt 注入"用法文档类不选"规则）
    - exclude_ids：已注入过的记忆 ID 集合（prompt 提示之外
      还做**确定性后过滤**——LLM 不听话也能滤掉）

    返回：记忆 ID 字符串列表；失败或无相关返回 []。
    """
    if not query.strip() or not index_text.strip():
        return []

    tools_rule = ""
    if active_tools:
        tools_rule = _ACTIVE_TOOLS_RULE.format(
            tools=", ".join(active_tools[:10]),
        )
    exclude_rule = ""
    if exclude_ids:
        exclude_rule = _EXCLUDE_RULE.format(
            excluded="\n".join(f"- {x}" for x in sorted(exclude_ids)[:30]),
        )

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        query=query[:1000],  # 防查询过长撑爆 prompt
        index_text=balanced_truncate_index(index_text, 25000),  # 超长按主题均衡截取
        max_results=max_results,
        active_tools_rule=tools_rule,
        exclude_rule=exclude_rule,
    )

    try:
        response = await llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("memory retrieval LLM 调用失败（fail-open）: %s", e)
        return []

    # 解析 JSON（模型常在 JSON 外多说话，要容错）
    try:
        # 先试整段直接解析
        result = json.loads(content)
    except json.JSONDecodeError:
        # 退而求其次：正则抠出 [ ... ] 再解析
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not match:
            logger.warning("memory retrieval 输出非合法 JSON: %s", content[:200])
            return []
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(result, list):
        return []
    # 只留字符串元素；确定性排除已注入过的（不信任 LLM 会听话）；最后掐到上限
    picked = [str(x) for x in result if isinstance(x, str)]
    if exclude_ids:
        picked = [x for x in picked if x not in exclude_ids]
    return picked[:max_results]


# 索引行里记忆 ID 的形状：真实索引（memory_store._entry_link 生成）括号里
# 是 markdown 链接路径（.memory/{topic}.jsonl#{短id} 或
# .memory/projects/{键}/{topic}.jsonl#{短id}），所以正则锚定 ".jsonl#"——
# 只认这种链接形态，索引行正文里随手写的 (v2#dev) 之类的括号对不会被
# 误当成记忆 ID（误配了兜底会选中它，get() 落空、该轮 0 注入）。
# 抠出来后还要过一道 _normalize_index_id 裁成裸 ID。
_ID_IN_LINE = re.compile(r"\(([^()\s]*\.jsonl#[^()\s]+)\)")


def _normalize_index_id(inner: str) -> str:
    """把括号里抠出来的东西裁成裸记忆 ID（{topic}#{短id}）。

    大白话：索引行括号里可能装两种货——裸 ID（proj#abc123）或带路径的
    markdown 链接（.memory/projects/键/project.jsonl#abc123）。下游
    （retrieve_relevant 的返回值、MemoryStore.get、exclude_ids 去重）
    认的全是裸 ID，所以这里统一裁剪：

    - 先砍掉路径前缀：取最后一个 / 之后的部分
    - 再把 .jsonl# 换成 #：project.jsonl#abc123 → project#abc123

    本来就是裸 ID 的原样返回（两步都不命中）。
    """
    tail = inner.rpartition("/")[-1]
    return tail.replace(".jsonl#", "#", 1)


def _iter_index_ids(index_text: str) -> List[str]:
    """从索引文本里抠出全部记忆 ID（保持出现顺序，已裁成裸 ID）。"""
    return [
        _normalize_index_id(m.group(1))
        for m in _ID_IN_LINE.finditer(index_text or "")
    ]


def _query_tokens(query: str) -> List[str]:
    """把检索 query 拆成小写词（按空白/常见标点切，丢单字噪声）。

    中文 2-gram 兜底：中文没空格，整句「怎么配置缓存」切成 1 个长
    token 后做子串匹配基本必 miss——中文用户在 aux LLM 挂掉时
    兜底检索等于瘫痪。所以对含 CJK 字符的 token 追加两两组合的
    2-gram（原 token 保留）：「怎么配置缓存」→ 原 token +
    ["怎么","么配","配置","置缓","缓存"]，短粒度的子词就能命中
    索引行了。打分用的，重复无所谓、不严格去重。
    """
    tokens = []
    for t in re.split(
            r"[\s,.;:!?，。；：！？/\\|()（）\[\]【】]+", query or ""):
        if len(t) < 2:
            continue
        t = t.lower()
        tokens.append(t)
        # 含 CJK 的 token 追加 2-gram 切分（纯英文/数字 token 不动）
        cjk = re.findall(r"[\u4e00-\u9fff]", t)
        if len(cjk) >= 2:
            tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def keyword_fallback_ids(
    query: str,
    index_text: str,
    *,
    max_results: int = 5,
    exclude_ids=None,
) -> List[str]:
    """确定性兜底检索：aux LLM 挑选失败/为空时，用关键词包含匹配顶上。

    大白话：LLM 检索是「让秘书翻目录挑条目」，秘书请假（调用失败）或
    空手而归时，这里用最笨但永远可用的办法——把 query 拆词，看哪些
    索引行包含这些词，按命中词数排序取前 N。零依赖、零 LLM。

    参数：
    - query：当前检索依据
    - index_text：记忆索引全文（调用方手里就有）
    - max_results：最多返回几条
    - exclude_ids：已注入过的记忆 ID（跳过，跨轮去重）

    返回：记忆 ID 列表（可能为空）。
    """
    tokens = _query_tokens(query)
    if not tokens or not index_text:
        return []
    exclude = set(exclude_ids or [])
    scored = []
    for line in index_text.splitlines():
        m = _ID_IN_LINE.search(line)
        if not m:
            continue
        mid = _normalize_index_id(m.group(1))
        if mid in exclude:
            continue
        low = line.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits > 0:
            scored.append((hits, mid))
    scored.sort(key=lambda pair: -pair[0])
    return [mid for _hits, mid in scored[:max_results]]


def correct_memory_id(bad_id: str, index_text: str) -> Optional[str]:
    """ID 纠错：LLM 抄错 ID（大小写/截尾）时按归一化和前缀匹配救回。

    记忆 ID 长得像 proj#abc123，LLM 抄错一个字符 get() 就落空——
    旧版直接静默丢条。这里从索引里把真 ID 找回来：
    1. 大小写归一后完全相等 → 直接换回真 ID
    2. 归一后一方是另一方前缀（且被截方 >= 6 字符）且候选唯一 → 换回

    参数：
    - bad_id：LLM 抄出来的（可能残缺的）ID
    - index_text：记忆索引全文

    返回：真实 ID；救不回返回 None。
    """
    if not bad_id:
        return None
    real_ids = _iter_index_ids(index_text)
    norm = lambda s: s.strip().lower()
    for rid in real_ids:
        if norm(rid) == norm(bad_id):
            return rid
    nb = norm(bad_id)
    if len(nb) >= 6:
        cands = [
            rid for rid in real_ids
            if norm(rid).startswith(nb) or nb.startswith(norm(rid))
        ]
        if len(cands) == 1:
            return cands[0]
    return None
