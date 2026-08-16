"""对话级轻量记忆提取（R19 #21）。

对齐 CC extractMemories：每轮对话结束后后台提取记忆，与任务级 reflection
互补（reflection 是触发式全轨迹复盘，本模块是每 N 回合的增量轻提取）。

与 CC 的差异（如实记录）：
- CC 用 forked agent + 工具白名单（Read/Grep/Glob + Bash 只读）可查证；
  OmniMate 用 aux_llm **单轮无工具**提取（轻量优先——每轮跑的成本敏感，
  查证类需求留给 reflection/curator）
- CC 的 maxTurns=5 防跑偏不适用（无工具循环）

互斥（对齐 CC「主 agent 写入互斥」）：
- 本轮 LLM 调过 memory save/update → 跳过本次提取并推进游标
  （主 agent 已写过，自动提取不重复抢写）

节流：每 every_n_turns 个回合跑一次（config memory.auto_extract）。
防重复：预注入已有记忆 manifest（与 reflection 共用 build_memory_manifest）。
秘密扫描：memory_store.save 内置（R19 #24），提取产物命中自动拒绝。
整链 fail-open：任何异常不影响主对话。
"""
import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# 提取产物上限（对齐 reflection 的防堆积语义）
MAX_EXTRACT_ITEMS = 3


async def run_auto_extract(agent, start_idx: int) -> int:
    """后台任务：从 start_idx 到当前 history 末尾的增量轨迹提取记忆。

    返回保存条数。fail-open 全吞。
    """
    try:
        messages = list(agent.conversation_history[start_idx:])
        if len(messages) < 2:
            return 0
        aux = getattr(agent, "aux_llm_router", None)
        memory_store = getattr(agent, "memory_store", None)
        if aux is None or memory_store is None:
            return 0

        from agent.reflection import (
            REFLECTION_PROMPT_TEMPLATE,
            build_memory_manifest,
            extract_trajectory,
        )
        trajectory = extract_trajectory(messages)
        if not trajectory.strip():
            return 0
        manifest = build_memory_manifest(memory_store)
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            trajectory=trajectory,
            existing_memories=manifest or "(暂无已有记忆)",
        )

        resp = await aux.chat_completions(
            [{"role": "user", "content": prompt}],
        )
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return 0

        items = _parse_items(content)
        valid_types = {"user", "feedback", "project", "reference"}
        saved = 0
        for item in items[:MAX_EXTRACT_ITEMS]:
            try:
                t = item.get("type", "")
                name = (item.get("name") or "").strip()
                desc = (item.get("description") or "").strip()
                if t not in valid_types or not name or not desc:
                    continue
                memory_store.save(
                    name=name[:60],
                    description=desc[:200],
                    type=t,
                    summary=(item.get("summary") or "").strip()[:200],
                    body=(item.get("body") or "").strip(),
                    source_session_id=f"auto_extract:{getattr(agent, 'session_id', '')}",
                )
                saved += 1
            except Exception as e:
                # 含 R19 #24 秘密拒绝——单条失败不影响其余
                logger.debug("auto_extract 保存单条失败: %s", e)
        if saved:
            logger.info("auto_extract 增量提取保存 %d 条记忆", saved)
        return saved
    except Exception as e:
        logger.debug("auto_extract 失败（fail-open）: %s", e)
        return 0


def _parse_items(content: str) -> List[dict]:
    """解析 LLM 输出为条目列表（容忍多余文本，与 reflection 同款策略）。"""
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return [x for x in result if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        result = json.loads(match.group(0))
        return [x for x in result if isinstance(x, dict)] if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []
