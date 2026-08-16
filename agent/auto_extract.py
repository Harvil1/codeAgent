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

机械查证（R26 #13）：
- CC 用带工具的 fork agent 查证；OmniMate 裁决为**机械验证**（提取产物中
  出现的文件路径须真实存在，否则丢弃该条）——零 LLM 成本、覆盖最主要的
  幻觉形态（编造路径入库）。完整带工具查证不搬（每轮成本不可接受）。
"""
import json
import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# 提取产物上限（对齐 reflection 的防堆积语义）
MAX_EXTRACT_ITEMS = 3

# 形如 path/to/file.py 或 file.py 的 token（含扩展名；跨平台分隔符）。
# 注：目录段可为零——简报测试锚定裸文件名（fake_path.py）也须查证，
# 故用 * 而非 +（纯单词无扩展名仍不验）。
_PATH_TOKEN_RE = re.compile(r"(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z]{1,4}")


def _filter_verified_items(items: List[dict], base_dir: str) -> List[dict]:
    """R26 #13：机械查证——条目 body/summary 里提到的文件路径必须存在。

    LLM 提取最常见的幻觉是编造文件路径；命中不存在路径的条目直接丢弃
    （宁缺毋滥）。验证只认"像路径"的 token（file.py / a/b.py 形态），
    纯单词不验。
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    kept = []
    for item in items:
        text = f"{item.get('body', '')} {item.get('summary', '')}"
        paths = _PATH_TOKEN_RE.findall(text)
        ok = True
        for p in paths:
            cand = Path(p)
            if cand.is_absolute():
                continue  # 绝对路径跨机器不稳，不验（保守放行）
            if not (cand.exists() or (base / cand).exists()):
                ok = False
                logger.info("auto_extract 丢弃幻觉路径条目（%s 不存在）: %s", p, item.get("name"))
                break
        if ok:
            kept.append(item)
    return kept


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
        # R26 #13：机械查证（幻觉路径条目丢弃）
        from agent.workspace_context import get_workspace_cwd
        try:
            base_dir = get_workspace_cwd()
        except Exception:
            base_dir = ""
        items = _filter_verified_items(items, base_dir)
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
