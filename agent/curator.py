"""Curator：技能（沉淀成 Markdown 的"怎么做"经验）的后台维护工。

触发方式：不是定时器，而是 agent 启动时问一句 should_run_now()——
距上次运行超过 interval_hours（默认 7 天）才触发。

第一次运行不马上干活，只记一个起始时间（种子化 last_run_at），
等满一个完整周期再跑（防止刚装好就大改技能库）。

干活分三步：
  第 1 步：确定性状态转换（纯时间规则，不调 LLM）
  第 2 步：LLM 合并审查（可选，默认关闭）
  第 3 步（可选）：跨会话记录整理 consolidate_transcripts——
      从最近 5 个会话的原始轨迹里提炼跨会话的共性经验。
      和前两步互补：前两步整理"已有"的技能/记忆，这一步从原始会话里挖"新"知识。

手动触发：codeagent curator run [--dry-run]
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from tools import skill_usage as _u

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置（实际值从 config.yaml 读，这里只是兜底默认值）
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_HOURS = 24 * 7        # 每 7 天跑一次
DEFAULT_STALE_AFTER_DAYS = 30          # 30 天没动静 → 标 stale（疑似过时）
DEFAULT_ARCHIVE_AFTER_DAYS = 90        # 90 天没动静 → 归档
DEFAULT_MIN_IDLE_HOURS = 1             # agent 至少空闲 1 小时才跑（防打扰）


def is_enabled() -> bool:
    """curator 的开关是否打开（从配置读；读不到默认开）。"""
    try:
        from config import load_config
        return bool(load_config().get("curator", {}).get("enabled", True))
    except Exception:
        return True


def is_paused(skills_dir: Path = None) -> bool:
    """判断是否被用户手动暂停。

    参数：
    - skills_dir：技能目录（状态文件在它的上级目录）；不传直接返回"未暂停"
    """
    if skills_dir is None:
        return False
    state = load_state(skills_dir)
    return bool(state.get("paused", False))


def _curator_cfg(key: str, default: int) -> int:
    """读 curator 配置项；配置缺失/类型不对/读配置失败都回退模块常量。"""
    try:
        from config import load_config
        value = load_config().get("curator", {}).get(key, default)
        return int(value)
    except Exception:
        return default


def get_interval_hours() -> int:
    """返回运行间隔（小时）。"""
    return _curator_cfg("interval_hours", DEFAULT_INTERVAL_HOURS)


def get_stale_after_days() -> int:
    """返回"多少天没动静标 stale"的天数。"""
    return _curator_cfg("stale_after_days", DEFAULT_STALE_AFTER_DAYS)


def get_archive_after_days() -> int:
    """返回"多少天没动静归档"的天数。"""
    return _curator_cfg("archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS)


# ---------------------------------------------------------------------------
# 状态文件：记上次运行时间，判断周期到没到
# ---------------------------------------------------------------------------

def _state_file(skills_dir: Path) -> Path:
    # skills_dir 的上一级就是 agent_home（~/.codeAgent），状态文件放那里
    return Path(skills_dir).parent / ".curator_state.json"


def load_state(skills_dir: Path) -> dict:
    """读 curator 状态（上次运行时间等）。文件不存在或读坏返回空 dict。

    参数：
    - skills_dir：技能目录
    """
    path = _state_file(skills_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(skills_dir: Path, state: dict) -> None:
    """保存 curator 状态（原子写，写一半断电也不会写坏文件）。"""
    from agent.atomic_io import atomic_write_text
    path = _state_file(skills_dir)
    atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))


def _parse_iso(value) -> Optional[datetime]:
    """解析 ISO 时间戳（转手调 agent.utils.parse_iso，失败返回 None）。"""
    from agent.utils import parse_iso
    return parse_iso(value, on_failure=None)


# ---------------------------------------------------------------------------
# 触发判断：现在到底该不该跑
# ---------------------------------------------------------------------------

def should_run_now(
    skills_dir: Path,
    now: Optional[datetime] = None,
) -> bool:
    """判断 curator 现在是否应该立即运行。

    三道门，全过才跑：
      - 配置里开了
      - 没被暂停
      - 距上次运行超过 interval_hours

    从没跑过的话：只记下当前时间当起点（种子化 last_run_at），等满一个周期。

    参数：
    - skills_dir：技能目录
    - now：当前时间（不传用系统时间）
    """
    if not is_enabled():
        return False
    if is_paused(skills_dir):
        return False

    if now is None:
        now = datetime.now(timezone.utc)

    state = load_state(skills_dir)
    last = _parse_iso(state.get("last_run_at"))

    if last is None:
        # 从没跑过：播下种子（记当前时间），等满一个周期
        state["last_run_at"] = now.isoformat()
        state["last_run_summary"] = (
            "首次运行已推迟——curator 已种子化，"
            "将在一个周期后运行。用 `curator run --dry-run` 立即预览。"
        )
        save_state(skills_dir, state)
        return False

    interval = timedelta(hours=get_interval_hours())
    return (now - last) >= interval


# ---------------------------------------------------------------------------
# 第 1 步：确定性状态转换（纯规则，不调 LLM）
# ---------------------------------------------------------------------------

def apply_automatic_transitions(
    skills_dir: Path,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """遍历所有 curator 管的技能，按最后活动时间转换状态。

    规则（活动时间 = 使用/查看/修补里最新的一次；从没活动过用创建时间）：
      90 天没动静 → 归档（挪 .archive/）
      30 天没动静 → 标 stale
      又被用了   → 恢复 active
    用户钉住（pinned）的技能永远不碰；只管 agent 自己创建的技能。

    参数：
    - skills_dir：技能目录
    - now：当前时间（不传用系统时间）
    返回：计数 dict。
    """
    if now is None:
        now = datetime.now(timezone.utc)

    stale_cutoff = now - timedelta(days=get_stale_after_days())
    archive_cutoff = now - timedelta(days=get_archive_after_days())

    counts = {
        "marked_stale": 0,
        "archived": 0,
        "reactivated": 0,
        "checked": 0,
    }

    usage = _u.load_usage(skills_dir)

    for name, rec in usage.items():
        # 只管 agent 自己创建的技能（用户手建的不能动）
        if rec.get("created_by") != "agent":
            continue

        # 被钉住（pinned）的技能免疫一切操作
        if rec.get("pinned"):
            continue

        counts["checked"] += 1

        # 算最后活动时间（使用/查看/修补里最新的）
        last_activity = _latest_activity(rec)
        # 从没活动过，就拿创建时间当锚点
        anchor = last_activity or _parse_iso(rec.get("created_at")) or now

        current = rec.get("state", _u.STATE_ACTIVE)

        # 从没用过的技能给宽限期
        never_used = int(rec.get("use_count", 0) or 0) == 0
        if never_used and anchor > stale_cutoff:
            # 还年轻——可能只是触发场景还没出现
            if current == _u.STATE_STALE:
                _u.set_state(skills_dir, name, _u.STATE_ACTIVE)
                counts["reactivated"] += 1
            continue

        # 状态转换（30/90 天两道线）
        if anchor <= archive_cutoff and current != _u.STATE_ARCHIVED:
            ok, _msg = _u.archive_skill(skills_dir, name)
            if ok:
                counts["archived"] += 1
        elif anchor <= stale_cutoff and current == _u.STATE_ACTIVE:
            _u.set_state(skills_dir, name, _u.STATE_STALE)
            counts["marked_stale"] += 1
        elif anchor > stale_cutoff and current == _u.STATE_STALE:
            # 又被使用了 → 重新激活
            _u.set_state(skills_dir, name, _u.STATE_ACTIVE)
            counts["reactivated"] += 1

    return counts


def _latest_activity(rec: dict) -> Optional[datetime]:
    """取这条技能记录的最新活动时间（使用/查看/修补三个时间里最晚的）。

    参数：
    - rec：单条技能的使用记录 dict
    返回：最新活动时间；三个都没有返回 None。
    """
    candidates = [
        _parse_iso(rec.get("last_used_at")),
        _parse_iso(rec.get("last_viewed_at")),
        _parse_iso(rec.get("last_patched_at")),
    ]
    valid = [c for c in candidates if c is not None]
    return max(valid) if valid else None


# ---------------------------------------------------------------------------
# 第 2 步：LLM 合并审查（下面的 CURATOR_REVIEW_PROMPT 是发给 LLM 的提示词）
# ---------------------------------------------------------------------------

CURATOR_REVIEW_PROMPT = (
    "你是后台技能库管理员。这是一次**伞形构建的合并任务**，"
    "不是被动审计，也不是简单的去重。\n\n"

    "目标：把技能库整理成**类级别的指令和经验知识库**。"
    "如果技能库里有几百个各自只记录一次会话的窄技能，"
    "那是失败——不是成功。agent 搜索技能是按描述匹配，"
    "不是按精确名字；一个宽泛的伞形技能（带子章节）"
    "比五个窄技能更容易被发现。\n\n"

    "硬规则——不可违反：\n"
    "1. 不要动用户手动创建的技能（created_by='user'）\n"
    "2. 不要删除任何技能。归档（移到 .archive/）是最大破坏性操作。"
    "归档可恢复，删除不可恢复。\n"
    "3. 不要碰 pinned 的技能。完全跳过。\n"
    "4. 不要把 use_count=0 当作跳过合并的理由。计数是新的，大多是零。"
    "按内容判断重叠，不是按 use_count。\n"
    "5. 不要以'每个技能有不同触发场景'为由拒绝合并。"
    "正确的判断是：'人类维护者会把这个写成 N 个独立技能，"
    "还是一个带 N 个子章节的技能？'如果答案是后者，合并。\n\n"

    "工作方法——必须执行：\n"
    "1. 扫描全部候选技能列表。识别**前缀簇**（共享首词或领域关键词的技能）。\n"
    "2. 对每个 2+ 成员的簇，问'这些技能服务的**伞形类**是什么？维护者会命名这个类"
    "   并写一个技能吗？'如果会，选择（或创建）伞形，吸收同级技能。\n"
    "3. 三种合并方式：\n"
    "   a. 合并到已有伞形——簇中一个技能已经足够宽泛。用 patch 加章节，然后归档同级。\n"
    "   b. 创建新伞形——没有现有成员足够宽泛。用 skill_manage(create) 写新的类级别技能，归档窄的同级。\n"
    "   c. 降级为 references/templates/scripts——同级有窄但有价值的会话特定内容。"
    "   移到伞形的适当子目录，然后归档旧的。\n"
    "4. 迭代。一轮合并后，扫描剩余集合，找下一个伞形机会。\n\n"

    "工具集：skills_list, skill_view, skill_manage(patch/create/write_file/delete)\n"
    "合并时 skill_manage(action='delete') 必须传 absorbed_into=<伞形名>。\n\n"

    "完成后写人类可读总结和结构化 YAML 块：\n\n"
    "## 结构化总结（必需）\n"
    "```yaml\n"
    "consolidations:\n"
    "  - from: <旧技能名>\n"
    "    into: <伞形技能名>\n"
    "    reason: <一句话>\n"
    "prunings:\n"
    "  - name: <技能名>\n"
    "    reason: <一句话>\n"
    "```\n"
)


def run_curator_review(
    skills_dir: Path,
    *,
    agent_factory=None,
    dry_run: bool = False,
    session_store=None,
    memory_store=None,
    llm=None,
) -> Dict:
    """跑一次完整的 curator 审查（三步走）。

    第 1 步：确定性状态转换（总是跑）
    第 2 步：LLM 合并审查（可选，agent_factory 传入且非 dry_run 才跑）
    第 3 步：跨会话记录整理（可选）——session_store/memory_store/llm
        都传入才启用；门控是"距上次 ≥24 小时 + 新增 ≥5 个会话"（should_consolidate），
        门控状态（last_consolidate_at / sessions_seen）随本次运行一起落盘

    参数：
    - skills_dir：技能目录
    - agent_factory：造后台 agent 的工厂，签名
        agent_factory(enabled_toolsets=[...], is_background_review=True) -> AIAgent
    - dry_run：True 只统计不动手
    - session_store / memory_store / llm：第 3 步需要的组件（不传就跳过第 3 步）
    返回：运行报告 dict。
    """
    now = datetime.now(timezone.utc)
    start = now

    # 第 1 步：确定性转换
    if not dry_run:
        counts = apply_automatic_transitions(skills_dir, now=now)
    else:
        counts = {"marked_stale": 0, "archived": 0, "reactivated": 0, "checked": 0}

    # 第 2 步：LLM 合并审查（有工厂且非 dry_run 才跑）
    consolidation_result = {"consolidations": [], "prunings": []}
    if agent_factory and not dry_run:
        # 看看有没有 agent 自己创建的技能
        usage = _u.load_usage(skills_dir)
        agent_skills = [
            n for n, r in usage.items()
            if r.get("created_by") == "agent" and r.get("state") != "archived"
        ]

        if agent_skills:
            # 拼候选清单
            candidate_list = _render_candidate_list(skills_dir, agent_skills)

            # 拼完整 prompt
            prompt = f"{CURATOR_REVIEW_PROMPT}\n\n{candidate_list}"

            # 起一个后台 agent 来干活
            review_agent = None
            try:
                review_agent = agent_factory(
                    enabled_toolsets=["core"],  # core 工具集里有技能管理工具
                    is_background_review=True,
                )
                # 本函数是 sync 的（在 curator 后台线程里跑，不在宿主循环
                # 线程），AIAgent.chat 是 async 的 → 交给进程级常驻循环宿主
                # 同步等结果（等价旧的 asyncio.run，client 绑定常驻循环不漂移）
                from agent.loop_host import loop_host
                raw_output = loop_host.run_async(review_agent.chat(prompt), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）

                # 解析 LLM 输出里的结构化块
                consolidation_result = _parse_consolidation_output(raw_output)
            except Exception as e:
                logger.warning("curator LLM 审查失败: %s", e)
                consolidation_result = {
                    "consolidations": [],
                    "prunings": [],
                    "error": str(e),
                }
            finally:
                # 一次性 review agent 的 client 用完就关——chat 走常驻循环真分配了
                # 连接池，不关就滞留到进程退出（fail-open）。agent_factory
                # 本身抛异常时 review_agent 还是 None，跳过
                if review_agent is not None:
                    try:
                        from agent.llm_client import aclose_llm_client
                        from agent.loop_host import loop_host
                        loop_host.run_async(aclose_llm_client(review_agent.llm_client))
                    except (asyncio.CancelledError, concurrent.futures.CancelledError):
                        pass  # 取消不打穿关闭（concurrent 版是 fut.result() 搬运后的实际类型）
                    except Exception as e:
                        logger.warning("review agent client 关闭失败（fail-open）: %s", e)

    # 写报告 + 更新状态
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    state = load_state(skills_dir)
    state["last_run_at"] = now.isoformat()

    # 跨会话记录整理（可选第 3 步）。门控状态直接写进 state，
    # 和 last_run_at 一起随下面的 save_state 落盘
    consolidated = _maybe_consolidate_transcripts(
        state,
        session_store=session_store, memory_store=memory_store, llm=llm,
        dry_run=dry_run,
    )

    state["last_run_summary"] = (
        f"转换: {counts}; "
        f"合并: {len(consolidation_result.get('consolidations', []))}; "
        f"沉淀: {consolidated}; "
        f"耗时: {duration:.1f}s"
    )
    save_state(skills_dir, state)

    return {
        "transitions": counts,
        "consolidations": consolidation_result,
        "consolidated_memories": consolidated,
        "duration_seconds": duration,
        "dry_run": dry_run,
    }


def _render_candidate_list(skills_dir: Path, skill_names: list) -> str:
    """把候选技能清单拼成文本给 LLM 看。

    参数：
    - skills_dir：技能目录
    - skill_names：候选技能名列表
    """
    usage = _u.load_usage(skills_dir)
    lines = ["# 候选技能列表（curator 管理范围）\n"]

    for name in skill_names:
        rec = usage.get(name, {})
        lines.append(
            f"- {name}: use={rec.get('use_count', 0)}, "
            f"patch={rec.get('patch_count', 0)}, "
            f"state={rec.get('state', 'active')}, "
            f"pinned={'yes' if rec.get('pinned') else 'no'}"
        )

    return "\n".join(lines)


def _parse_consolidation_output(output: str) -> dict:
    """从 LLM 输出里抠出 ```yaml ... ``` 结构化块并解析。

    参数：
    - output：LLM 的原始输出
    返回：{consolidations, prunings}；没找到块或解析失败就给空列表。
    """
    # 找 ```yaml ... ``` 代码块
    match = re.search(r"```yaml\n(.*?)```", output, re.DOTALL)
    if not match:
        return {"consolidations": [], "prunings": []}

    yaml_block = match.group(1)
    try:
        import yaml
        parsed = yaml.safe_load(yaml_block) or {}
        return {
            "consolidations": parsed.get("consolidations", []),
            "prunings": parsed.get("prunings", []),
            "raw_yaml": yaml_block,
        }
    except Exception:
        return {"consolidations": [], "prunings": [], "raw_yaml": yaml_block}


# ---------------------------------------------------------------------------
# 跨会话记录整理（第 3 步，可选）
# ---------------------------------------------------------------------------

CONSOLIDATE_MIN_HOURS = 24
CONSOLIDATE_MIN_SESSIONS = 5

CONSOLIDATE_PROMPT = """以下是最近 {n} 个会话的轨迹摘要。请提炼出**跨会话反复出现、
值得长期记住**的项目经验（现有记忆里没有的）。

要求：
- 只提炼跨会话共性（单个会话的临时细节不要）
- type 只能是 project 或 reference
- 输出 JSON 数组，每条 {{"type": "project", "name": "...", "description": "...", "summary": "...", "body": "..."}}
- 最多 {max_items} 条；没有值得提炼的输出 []

轨迹：
{trajectories}
"""


def should_consolidate(state: dict, *, now: float, new_sessions_since: int) -> bool:
    """第 3 步的门控：距上次 ≥24 小时 且 新增 ≥5 个会话才跑。

    （curator 是单进程触发的，不需要跨进程锁。）

    参数：
    - state：curator 状态 dict（读 last_consolidate_at）
    - now：当前 Unix 时间戳（秒）
    - new_sessions_since：距上次整理新增的会话数
    """
    last = float(state.get("last_consolidate_at") or 0)
    if (now - last) < CONSOLIDATE_MIN_HOURS * 3600:
        return False
    return new_sessions_since >= CONSOLIDATE_MIN_SESSIONS


def consolidate_transcripts(session_store, memory_store, *, llm) -> int:
    """跨会话记录整理：从最近几个会话的轨迹里提炼值得长期记的经验。

    把散落在多个会话里的碎片经验沉淀成完整记忆条目。
    和 curator 其他动作的区别：那些整理"已有记忆"，这里从"原始会话"挖新知识。
    整体 fail-open（出任何错吞掉返回 0）；save 自带秘密扫描（命中拒收单条）。

    参数：
    - session_store：会话库（取最近会话和消息）
    - memory_store：记忆库（结果存这里）
    - llm：LLM 句柄，要有 chat_completions（LLMClient / AuxLLMRouter 都行；
        现场都是 async 接口，这里交给常驻循环宿主驱动——和 reflection 同模式）
    返回：成功沉淀的条数。
    """
    try:
        from agent.reflection import extract_trajectory
        sessions = session_store.list_sessions(limit=CONSOLIDATE_MIN_SESSIONS)
        parts = []
        for s in sessions:
            msgs = session_store.get_messages(s["id"], limit=200)
            traj = extract_trajectory(msgs)
            if traj.strip():
                parts.append(f"## 会话：{s.get('title') or s['id'][:8]}\n{traj}")
        if len(parts) < CONSOLIDATE_MIN_SESSIONS:
            return 0
        prompt = CONSOLIDATE_PROMPT.format(
            n=len(parts), max_items=5,
            trajectories="\n\n".join(parts)[:60000],
        )
        # chat_completions 是 async 的，而 curator 在后台线程 / CLI 的同步
        # 上下文里跑（不在宿主循环线程）→ 交给进程级常驻循环宿主同步等结果
        # （等价旧的 asyncio.run，client 绑定常驻循环不漂移）
        from agent.loop_host import loop_host
        resp = loop_host.run_async(llm.chat_completions([{"role": "user", "content": prompt}]), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
        content = resp.choices[0].message.content or ""
        import json as _json
        import re as _re
        try:
            items = _json.loads(content)
        except _json.JSONDecodeError:
            m = _re.search(r"\[.*\]", content, _re.DOTALL)
            items = _json.loads(m.group(0)) if m else []
        saved = 0
        for item in (items or [])[:5]:
            if not isinstance(item, dict) or item.get("type") not in ("project", "reference"):
                continue
            try:
                memory_store.save(
                    name=str(item.get("name", ""))[:60],
                    description=str(item.get("description", ""))[:200],
                    type=item["type"],
                    summary=str(item.get("summary", ""))[:200],
                    body=str(item.get("body", "")),
                    source_session_id="curator:consolidate",
                )
                saved += 1
            except Exception as e:
                logger.warning("consolidate 单条保存失败（含秘密拒绝）: %s", e)
        if saved:
            logger.info("curator consolidate_transcripts 沉淀 %d 条跨会话记忆", saved)
        return saved
    except Exception as e:
        logger.warning("consolidate_transcripts fail-open: %s", e)
        return 0


def _maybe_consolidate_transcripts(
    state: dict,
    *,
    session_store,
    memory_store,
    llm,
    dry_run: bool,
) -> int:
    """第 3 步的接线：过了门控（≥24h + ≥5 新会话）才真正跑。

    门控状态（last_consolidate_at / sessions_seen）直接写进 state dict，
    由调用方（run_curator_review）统一落盘。组件缺失 / 门控没过 / dry_run
    都返回 0 且不动门控状态（缺 llm 时留到下次补跑）。

    参数：
    - state：curator 状态 dict（门控的读和写都在它身上）
    - session_store / memory_store / llm：第 3 步需要的组件
    - dry_run：True 直接跳过
    返回：沉淀条数。
    """
    if dry_run:
        return 0
    if session_store is None or memory_store is None:
        logger.debug("consolidate 跳过：session_store/memory_store 未提供")
        return 0
    try:
        total_sessions = len(session_store.list_sessions(limit=1000))
    except Exception as e:
        logger.warning("consolidate 会话计数失败（本轮跳过）: %s", e)
        return 0
    new_sessions = max(0, total_sessions - int(state.get("sessions_seen") or 0))
    if not should_consolidate(state, now=time.time(), new_sessions_since=new_sessions):
        return 0
    if llm is None:
        logger.info("curator consolidate 跳过：未提供 llm 句柄")
        return 0
    saved = consolidate_transcripts(session_store, memory_store, llm=llm)
    state["last_consolidate_at"] = time.time()
    state["sessions_seen"] = total_sessions
    return saved
