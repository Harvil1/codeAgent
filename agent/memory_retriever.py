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

返回最多 {max_results} 条与当前 query 最相关的记忆 ID（从索引的 .memory/{{id}}.md 路径中提取 {{id}} 部分）。
格式：JSON 数组，元素是 ID 字符串。例如：["1720870000000a1b2c3", "1720870000000d4e5f6"]
只返回 JSON 数组，不要其他文本。若无相关的，返回 []。
"""


def retrieve_relevant(
    *,
    query: str,
    index_text: str,
    llm_client,
    model: str,
    max_results: int = 5,
) -> List[str]:
    """调 LLM 选 top-N 相关 memory_id。失败返回 []。"""
    if not query.strip() or not index_text.strip():
        return []

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        query=query[:1000],  # 防止 query 太长
        index_text=index_text[:5000],  # 防止 index 太长
        max_results=max_results,
    )

    try:
        response = llm_client.chat_completions(
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
