"""相关记忆检索器：每轮主 LLM 调用前调一次。

输入：当前 user message + memory 索引
输出：top-N 最相关的 memory_id 列表

失败 fail-open：任何异常（LLM 超时/坏 JSON/空响应）返回空 list。
"""
import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


RETRIEVAL_PROMPT_TEMPLATE = """你是记忆检索助手。当前用户消息：

<query>
{query}
</query>

可用记忆索引（每行一条）：

<index>
{index_text}
</index>

返回最多 {max_results} 条与当前 query 最相关的记忆 ID（从索引行的 `.memory/{{topic}}.jsonl#{{uid}}` 路径中提取 `{{topic}}#{{uid}}` 部分）。
格式：JSON 数组，元素是 ID 字符串。例如：["general#1720870000000a1b2c3", "debugging#1720870000000d4e5f6"]
只返回 JSON 数组，不要其他文本。若无相关的，返回 []。

年龄标注说明：每条索引行末尾的 [age: Nd] 表示该记忆最后更新距今天数（[age: unknown] 表示未知）。
排序规则：相关性同等的两条记忆，优先返回更新的那条（N 更小）；新旧记忆内容冲突时，
新记忆优先，旧记忆只作历史背景（例如"用户在用 React 16"是旧信息，"已升到 React 19"是新信息，应返回后者）。
"""


def annotate_index_with_age(index_text: str, link_age_days: dict) -> str:
    """给索引行末尾附年龄标注 `[age: Nd]`（T4，防召回过期记忆）。

    Args:
        index_text: 完整索引文本（MEMORY.md 正文同构）
        link_age_days: markdown 链接路径 → 年龄天数（None/查不到 = unknown）

    纯 prompt 层改造：不改存储格式、不改 Top5 语义。
    无链接的行（标题等）原样返回；fail-open——任何解析问题只是不标注。
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
) -> List[str]:
    """调 LLM 选 top-N 相关 memory_id。失败返回 []（async：LLMClient.chat_completions 已改 async）。

    Task D4 fix: 改 async + await chat_completions。之前 sync 调 async 方法
    返回 coroutine，被 except 捕获 TypeError 后返回空 list（记忆检索静默失效）。
    """
    if not query.strip() or not index_text.strip():
        return []

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        query=query[:1000],  # 防止 query 太长
        index_text=index_text[:25000],  # 检索 index 上限对齐 25KB（记忆多时检索更完整）
        max_results=max_results,
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

    # 提取 JSON 数组（容忍模型输出多余文本）
    try:
        # 尝试直接 parse
        result = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 [ ... ] 子串
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
    # 只保留字符串元素 + 截断到 max_results
    return [str(x) for x in result if isinstance(x, str)][:max_results]
