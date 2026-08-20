"""对话级轻量记忆提取（R19 #21 引入）。

记忆（AI 对用户/项目沉淀下来的事实条目，跨会话保留）的自动积累有两条路：
- reflection（agent/reflection.py）：任务做完后对整段轨迹做一次完整复盘，
  触发式、重量级
- 本模块：每隔几轮对话就悄悄做一次增量小提取，每轮成本极低

两者互补，像"每次做完项目写总结报告"和"随手记便签"的区别。

与 CC（被对标的实现）的差异（如实记录）：
- CC 用 fork 出的子代理 + 工具白名单（Read/Grep/Glob + Bash 只读）能实地
  查证；OmniMate 用辅助模型（aux_llm，便宜模型）单轮无工具提取——
  因为每轮都跑，成本敏感；需要查证的重活留给 reflection/curator
- CC 的 maxTurns=5（防子代理跑偏）在这里不适用（我们没有工具循环）

互斥规则（对齐 CC"主 agent 写入互斥"）：本轮 LLM 自己调过 memory
save/update 的话，自动提取就让位——主 agent 已经写过，别重复抢写；
游标照样推进（跳过的不回头补看）。

节流：每 every_n_turns 轮才跑一次（config memory.auto_extract）。
防重复：把已有记忆清单（与 reflection 共用 build_memory_manifest）预先
给 LLM 看，让它别写重复的。
秘密扫描：memory_store.save 内置（R19 #24），提取产物里混进密钥会被自动拒绝。
整条链 fail-open：任何异常都不影响主对话。

机械查证（R26 #13 的裁决）：CC 用带工具的 fork agent 查证；OmniMate 裁决
为**机械验证**——提取产物里出现的文件路径必须真实存在，不存在就丢弃该条。
零 LLM 成本，且正好挡住最主要的幻觉形态（编造路径入库）。
完整的带工具查证不搬（每轮都跑，成本不可接受）。
"""
import json
import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# 单次提取的条数上限（对齐 reflection 的"防堆积"语义——宁少勿滥）
MAX_EXTRACT_ITEMS = 3

# 匹配"像文件路径"的 token：形如 path/to/file.py 或裸的 file.py
# （必须带扩展名；分隔符兼容跨平台）。
# 注意：目录段允许为零——测试里裸文件名（fake_path.py）也必须查证，
# 所以用 * 不用 +（纯单词没扩展名的仍然不验，避免误伤普通词）。
_PATH_TOKEN_RE = re.compile(r"(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z]{1,4}")


def _filter_verified_items(items: List[dict], base_dir: str) -> List[dict]:
    """机械查证（R26 #13）：条目里提到的文件路径必须真实存在。

    背景：LLM 提取最常见的幻觉是编造文件路径——编出来的路径一旦入库，
    之后的会话会把它当真。所以宁缺毋滥：提到不存在路径的条目直接丢弃。
    只验"长得像路径"的 token（file.py / a/b.py 形态），纯单词不验。

    参数：
    - items：LLM 提取出的条目列表
    - base_dir：相对路径的基准目录（路径会同时按原样和拼上 base 试）

    返回：通过查证的条目子列表。
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
                continue  # 绝对路径跨机器必然不稳，不验（保守放行）
            if not (cand.exists() or (base / cand).exists()):
                ok = False
                logger.info("auto_extract 丢弃幻觉路径条目（%s 不存在）: %s", p, item.get("name"))
                break
        if ok:
            kept.append(item)
    return kept


async def run_auto_extract(agent, start_idx: int) -> int:
    """后台任务：从上次游标处到当前对话末尾，增量提取记忆。

    背景：这是节流器到点后真正跑的函数，主循环把它丢到后台执行。

    参数：
    - agent：主 agent 实例（提供对话历史、aux 路由、记忆库、会话 ID）
    - start_idx：上次处理到的位置（游标），只看它之后的新消息

    返回：本次实际保存的条数（int）。任何失败都吞掉返回 0（fail-open）。
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
        # R26 #13：机械查证（提到不存在路径的条目丢弃）
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
                # 包括 R19 #24 的秘密拒绝——单条失败不连累其余条目
                logger.debug("auto_extract 保存单条失败: %s", e)
        if saved:
            logger.info("auto_extract 增量提取保存 %d 条记忆", saved)
        return saved
    except Exception as e:
        logger.debug("auto_extract 失败（fail-open）: %s", e)
        return 0


def _parse_items(content: str) -> List[dict]:
    """把 LLM 回复解析成条目列表，容忍 JSON 外的多余文字。

    背景：模型经常在 JSON 前后加解释性文字。先试整段解析，不行再用
    正则从文本里抠出 [...] 片段解析（与 reflection 同款策略）。

    参数：
    - content：LLM 的原始回复文本

    返回：合法的条目 dict 列表；解析不出返回空列表。
    """
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
