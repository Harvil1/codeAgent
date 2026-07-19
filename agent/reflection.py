"""任务级反思引擎（CCALS-P0-2）。

每个 run_conversation 结束后调一次，用 aux_llm（便宜模型）从对话轨迹中提炼
3 类经验，自动 memory_save 写入对应分类：
  - user 维度：新偏好、习惯、输出要求
  - feedback 维度：有效策略、需避开的坑、工具组合技巧
  - project 维度：项目规则、技术栈决策、业务逻辑

设计：
- aux_llm 不可用 → no-op（fail-open，不影响主流程）
- LLM 输出 JSON 数组，每项 {type, name, description, summary, body}
- 已有同类记忆时跳过（按 name + type 去重）
- 异步触发（不阻塞用户拿到响应）—— 调用方决定同步还是异步
"""
import json
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


REFLECTION_PROMPT_TEMPLATE = """你是经验提炼助手。从以下对话轨迹中提炼可复用的长期记忆。

<trajectory>
{trajectory}
</trajectory>

请提炼 0-5 条**值得长期记住**的经验，分三类：
- user：用户偏好、习惯、输出要求（例："用户喜欢简洁回复，不超过 3 句话"）
- feedback：有效策略、需避开的坑、工具组合技巧（例："用 delegate_task 并行处理多个文件比串行快 3 倍"）
- project：项目规则、技术栈决策、业务逻辑（例："本项目用 uv 不用 pip 管理依赖"）

要求：
1. 只输出真正非平凡的、跨会话有用的经验（不要记琐碎细节）
2. 没有值得记的就返回 []
3. 每条字段：
   - type: user / feedback / project 之一
   - name: ≤20 字标题（L0）
   - description: ≤40 字索引钩子（L0.5）
   - summary: 80-100 字摘要层（L1）
   - body: 完整说明（L2，可选）

输出格式：JSON 数组，不要其他文本。示例：
[
  {{
    "type": "user",
    "name": "偏好简洁回复",
    "description": "用户喜欢≤3 句话回复",
    "summary": "用户多次要求简短直接回复，不喜欢长篇解释。回答时优先给结论，展开用列表。",
    "body": ""
  }}
]
"""


def extract_trajectory(messages: List[dict], max_chars: int = 4000) -> str:
    """把 messages 列表压缩成供反思用的轨迹文本。

    策略：
    - 取最近 N 条（不超过 max_chars 字符）
    - 跳过 system / tool 长结果（只留 role+短 content）
    - 工具调用记一行"调用 X 工具"
    """
    if not messages:
        return ""
    # 倒序取，超 max_chars 截断
    lines = []
    total = 0
    for m in reversed(messages):
        role = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)[:200]
        else:
            content = str(content)[:300]
        # 工具调用单独标记
        tool_calls = m.get("tool_calls")
        if tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            content = f"[调用工具: {', '.join(names)}] " + content
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.insert(0, line)
        total += len(line)
    return "\n".join(lines)


def run_reflection(
    *,
    messages: List[dict],
    llm_client,
    model: Optional[str] = None,
) -> List[dict]:
    """跑一次反思，返回提炼出的经验列表。

    每项格式：{type, name, description, summary, body}
    失败（LLM 异常/坏 JSON/空响应）返回空 list（fail-open）。
    """
    trajectory = extract_trajectory(messages)
    if not trajectory.strip():
        return []

    prompt = REFLECTION_PROMPT_TEMPLATE.format(trajectory=trajectory)
    try:
        kwargs = {}
        if model:
            kwargs["model"] = model
        response = llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("反思 LLM 调用失败（fail-open）: %s", e)
        return []

    # 解析 JSON 数组（容忍模型输出多余文本）
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            logger.debug("反思输出非合法 JSON: %s", content[:200])
            return []
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(result, list):
        return []

    # 过滤+规范化每条
    valid = []
    valid_types = {"user", "feedback", "project"}
    for item in result:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        if t not in valid_types or not name or not desc:
            continue  # 必需字段缺失跳过
        valid.append({
            "type": t,
            "name": name[:60],  # 防止 LLM 给太长
            "description": desc[:80],
            "summary": (item.get("summary") or "").strip()[:200],
            "body": (item.get("body") or "").strip(),
        })
    return valid


def apply_reflection(
    *,
    messages: List[dict],
    memory_store,
    llm_client,
    model: Optional[str] = None,
) -> int:
    """端到端：跑反思 + 写入 memory_store。返回写入条数。

    去重：同 type 下 name 已存在则跳过（避免重复堆积）。
    """
    if memory_store is None or llm_client is None:
        return 0

    insights = run_reflection(
        messages=messages, llm_client=llm_client, model=model,
    )
    if not insights:
        return 0

    # 已有记忆 name 集合（按 type 分组）
    existing = memory_store.list_all()
    existing_keys = {(e.type, e.name) for e in existing}

    written = 0
    for ins in insights:
        key = (ins["type"], ins["name"])
        if key in existing_keys:
            continue  # 同类同名跳过
        try:
            memory_store.save(
                name=ins["name"],
                description=ins["description"],
                type=ins["type"],
                body=ins["body"],
                summary=ins["summary"],
            )
            existing_keys.add(key)  # 防同批多次重复
            written += 1
        except Exception as e:
            logger.warning("反思写入 memory 失败（跳过）: %s", e)

    if written:
        logger.info("反思写入 %d 条新经验", written)
    return written
