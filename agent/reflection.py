"""任务级反思引擎（CCALS-P0-2）。

每个 run_conversation 结束后调一次，用 aux_llm（便宜模型）从对话轨迹中提炼
4 类经验，自动 memory_save 写入对应分类：
  - user 维度：新偏好、习惯、输出要求
  - feedback 维度：有效策略、需避开的坑、工具组合技巧
  - project 维度：项目规则、技术栈决策、业务逻辑
  - reference 维度：外部系统指针（Linear/Slack/GitHub/URL/共享路径）

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

<existing_memories>
{existing_memories}
</existing_memories>

请提炼 0-5 条**值得长期记住**的经验,分四类:
- user：用户偏好、习惯、输出要求
- feedback：有效策略、需避开的坑、工具组合技巧
- project：项目规则、技术栈决策、业务逻辑
- **reference**: 外部系统指针——对话中提到的、值得跨会话记住的外部资源位置。
  例：Linear 工单 ID（PROJ-123）、Slack 频道（#incident-xxx）、GitHub repo
  （owner/repo）、文档 URL（https://...）、共享路径（/share/docs/xxx）。
  提炼时机：用户明确指向某外部资源作为后续工作上下文。
  不要把一次性查阅的 URL 算 reference（那不入记忆）。

要求:
1. 只输出真正非平凡的、跨会话有用的经验
2. **跟 existing_memories 语义重复的不要输出**(避免碎片化)
3. 没有值得记的就返回 []
4. 如果新经验**推翻/修正**已有记忆(如"改用 pip"推翻"用 uv"),加字段:
   - supersedes: 被推翻的旧记忆 name
   - confidence: 1.0(最新观察覆盖旧观察)
5. 每条字段:
   - type: user / feedback / project / reference 之一
   - name: ≤20 字标题(L0)
   - description: ≤40 字索引钩子(L0.5)
   - summary: 80-100 字摘要层(L1)
   - body: 完整说明(L2,可选)
   - confidence: 0.0-1.0(信心分,默认 0.8)

输出格式: JSON 数组,不要其他文本。示例:
[
  {{
    "type": "user",
    "name": "偏好简洁回复",
    "description": "用户喜欢≤3 句话回复",
    "summary": "用户多次要求简短直接回复...",
    "body": "",
    "confidence": 0.9
  }},
  {{
    "type": "project",
    "name": "改用 pip",
    "description": "项目改用 pip 管理依赖",
    "summary": "用户说项目从 uv 迁移到 pip...",
    "body": "",
    "confidence": 1.0,
    "supersedes": "用 uv 不用 pip"
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
    memory_store=None,
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

    # 批次 B: 传入已有记忆(让 LLM 避免重复 + 检测矛盾)
    try:
        existing = memory_store.list_all()
        existing_memories = "\n".join(
            f"- [{e.type}] {e.name}: {e.description}"
            + (f" | {e.summary}" if e.summary else "")
            for e in existing[:50]  # 最多 50 条,避免 prompt 太长
        )
    except Exception:
        existing_memories = "(无法读取已有记忆)"

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        trajectory=trajectory,
        existing_memories=existing_memories or "(暂无已有记忆)",
    )
    try:
        kwargs = {}
        if model:
            kwargs["model"] = model
        # Task D4 follow-up: llm_client.chat_completions 已改 async。
        # run_reflection 经 apply_reflection 在 _bg() daemon thread 里跑（无事件循环），
        # 用 asyncio.run 驱动；与 agent/user_profile.py:build_and_save_profile 同模式。
        import asyncio
        response = asyncio.run(llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
            **kwargs,
        ))
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
    valid_types = {"user", "feedback", "project", "reference"}
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
    session_id: str = "",
) -> int:
    """端到端：跑反思 + 写入 memory_store。返回写入条数。

    去重：同 type 下 name 已存在则跳过（避免重复堆积）。
    """
    if memory_store is None or llm_client is None:
        return 0

    insights = run_reflection(
        messages=messages, memory_store=memory_store, llm_client=llm_client, model=model,
    )
    if not insights:
        return 0

    # 已有记忆 name 集合（按 type 分组）
    existing = memory_store.list_all()
    existing_keys = {(e.type, e.name) for e in existing}

    # X14 fix: 两阶段——先全部 save，再统一处理 supersedes。
    # 之前在 save 循环内 supersede，导致"批内新写入"不在 existing 快照里，
    # 同批后写入的 insight 想 supersede 同批刚写入的会失效。
    written = 0
    written_records = []  # [(ins, mem_id)] 记录写入的，用于阶段 2 supersede
    for ins in insights:
        key = (ins["type"], ins["name"])
        if key in existing_keys:
            continue
        try:
            mem_id = memory_store.save(
                name=ins["name"],
                description=ins["description"],
                type=ins["type"],
                body=ins["body"],
                summary=ins["summary"],
                confidence=ins.get("confidence", 0.8),
                source_session_id=session_id or "",
            )
            existing_keys.add(key)
            written += 1
            written_records.append((ins, mem_id))
        except Exception as e:
            logger.warning("反思写入 memory 失败（跳过）: %s", e)

    # 阶段 2：处理 supersedes（X5 fix: 加 type 一致性 + X14 fix: 含批内新写入）
    for ins, _ in written_records:
        supersedes = ins.get("supersedes")
        if not supersedes or supersedes == ins["name"]:
            continue
        target_id = None
        target_body = ""
        # 先在 existing（旧记忆）找，X5 fix: 必须 name + type 都匹配
        for old_entry in existing:
            if (old_entry.name == supersedes
                    and old_entry.type == ins["type"]):
                target_id = old_entry.id
                target_body = old_entry.body or ""
                break
        # X14 fix: 再在批内新写入里找（同批刚 save 的也能被 supersede）
        if target_id is None:
            for other_ins, other_id in written_records:
                if (other_ins["name"] == supersedes
                        and other_ins["type"] == ins["type"]
                        and other_ins["name"] != ins["name"]):
                    target_id = other_id
                    target_body = other_ins.get("body", "")
                    break
        if target_id:
            try:
                memory_store.update(
                    target_id,
                    confidence=0.1,
                    body=f"[已被 '{ins['name']}' 推翻] " + target_body,
                )
                logger.info(
                    "记忆 '%s' 被 '%s' 推翻,confidence 降到 0.1",
                    supersedes, ins["name"],
                )
            except Exception:
                pass

    if written:
        logger.info("反思写入 %d 条新经验", written)
    return written
