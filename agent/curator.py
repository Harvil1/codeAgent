"""Curator：后台技能维护系统。

不是 cron 定时，而是在 agent 启动时检查 should_run_now()。
如果距离上次运行超过 interval_hours（默认 7 天），则触发。

第一运行不立即跑，而是种子 last_run_at，等一个完整周期。
（避免刚安装就大改技能库）

两阶段：
  第 1 阶段：确定性状态转换（纯时间规则，无 LLM）
  第 2 阶段：LLM 合并审查（可选，默认关闭）

手动触发：omnimate curator run [--dry-run]
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from tools import skill_usage as _u

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置（从 config.yaml 读取，这里提供默认值）
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_HOURS = 24 * 7        # 7 天跑一次
DEFAULT_STALE_AFTER_DAYS = 30          # 30 天无活动 → stale
DEFAULT_ARCHIVE_AFTER_DAYS = 90        # 90 天无活动 → archived
DEFAULT_MIN_IDLE_HOURS = 1             # agent 至少空闲 1 小时才跑


def is_enabled() -> bool:
    """curator 是否启用（从 config 读）。"""
    try:
        from config import load_config
        return bool(load_config().get("curator", {}).get("enabled", True))
    except Exception:
        return True


def is_paused(skills_dir: Path = None) -> bool:
    """是否被手动暂停。"""
    if skills_dir is None:
        return False
    state = load_state(skills_dir)
    return bool(state.get("paused", False))


def get_interval_hours() -> int:
    return DEFAULT_INTERVAL_HOURS


def get_stale_after_days() -> int:
    return DEFAULT_STALE_AFTER_DAYS


def get_archive_after_days() -> int:
    return DEFAULT_ARCHIVE_AFTER_DAYS


# ---------------------------------------------------------------------------
# 状态文件：记录上次运行时间
# ---------------------------------------------------------------------------

def _state_file(skills_dir: Path) -> Path:
    # skills_dir 的 parent 是 agent_home（~/.OmniMate）
    return Path(skills_dir).parent / ".curator_state.json"


def load_state(skills_dir: Path) -> dict:
    """加载 curator 状态（上次运行时间等）。"""
    path = _state_file(skills_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(skills_dir: Path, state: dict) -> None:
    """保存 curator 状态（原子写）。"""
    from agent.atomic_io import atomic_write_text
    path = _state_file(skills_dir)
    atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))


def _parse_iso(value) -> Optional[datetime]:
    """解析 ISO 时间戳(委托给 agent.utils.parse_iso,保留 None 兜底)。"""
    from agent.utils import parse_iso
    return parse_iso(value, on_failure=None)


# ---------------------------------------------------------------------------
# 触发判断
# ---------------------------------------------------------------------------

def should_run_now(
    skills_dir: Path,
    now: Optional[datetime] = None,
) -> bool:
    """判断 curator 是否应该立即运行。

    门控：
      - 必须启用
      - 未暂停
      - 距离上次运行超过 interval_hours

    首次运行：不立即跑，种子 last_run_at，延后一个周期。
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
        # 从未运行过：种子 last_run_at，等一个周期
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
# 第 1 阶段：确定性状态转换（纯函数，无 LLM）
# ---------------------------------------------------------------------------

def apply_automatic_transitions(
    skills_dir: Path,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    """遍历所有 curator 管理的技能，根据活动时间戳转换状态。

    Pinned 技能永不被碰。
    返回计数 dict。
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
        # 只管 agent 创建的技能
        if rec.get("created_by") != "agent":
            continue

        # Pinned 免疫一切
        if rec.get("pinned"):
            continue

        counts["checked"] += 1

        # 计算最后活动时间（use/view/patch 中最新的）
        last_activity = _latest_activity(rec)
        # 如果从未活动，用创建时间作为锚点
        anchor = last_activity or _parse_iso(rec.get("created_at")) or now

        current = rec.get("state", _u.STATE_ACTIVE)

        # 从未使用的技能给予宽限期
        never_used = int(rec.get("use_count", 0) or 0) == 0
        if never_used and anchor > stale_cutoff:
            # 还年轻，可能只是触发场景还没出现
            if current == _u.STATE_STALE:
                _u.set_state(skills_dir, name, _u.STATE_ACTIVE)
                counts["reactivated"] += 1
            continue

        # 状态转换
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
    """获取记录的最新活动时间（use/view/patch 中最新的）。"""
    candidates = [
        _parse_iso(rec.get("last_used_at")),
        _parse_iso(rec.get("last_viewed_at")),
        _parse_iso(rec.get("last_patched_at")),
    ]
    valid = [c for c in candidates if c is not None]
    return max(valid) if valid else None


# ---------------------------------------------------------------------------
# 第 2 阶段：LLM 合并审查
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
) -> Dict:
    """执行一次 curator 审查。

    第 1 步：确定性状态转换（总是跑）
    第 2 步：LLM 合并审查（可选）

    agent_factory：创建后台 agent 的工厂函数，签名：
        agent_factory(enabled_toolsets=[...], is_background_review=True) -> AIAgent

    返回运行报告。
    """
    now = datetime.now(timezone.utc)
    start = now

    # 第 1 步：确定性转换
    if not dry_run:
        counts = apply_automatic_transitions(skills_dir, now=now)
    else:
        counts = {"marked_stale": 0, "archived": 0, "reactivated": 0, "checked": 0}

    # 第 2 步：LLM 合并审查
    consolidation_result = {"consolidations": [], "prunings": []}
    if agent_factory and not dry_run:
        # 检查是否有 agent 创建的技能
        usage = _u.load_usage(skills_dir)
        agent_skills = [
            n for n, r in usage.items()
            if r.get("created_by") == "agent" and r.get("state") != "archived"
        ]

        if agent_skills:
            # 构造候选列表
            candidate_list = _render_candidate_list(skills_dir, agent_skills)

            # 构造完整 prompt
            prompt = f"{CURATOR_REVIEW_PROMPT}\n\n{candidate_list}"

            # 启动后台 agent 执行
            try:
                review_agent = agent_factory(
                    enabled_toolsets=["core"],  # 给技能管理工具
                    is_background_review=True,
                )
                # Task D4 fix: AIAgent.chat 已改 async。run_curator_review 是 sync 函数，
                # 可能由 threading 后台或 CLI sync 调用 → asyncio.run 驱动。
                import asyncio
                raw_output = asyncio.run(review_agent.chat(prompt))

                # 解析结构化输出
                consolidation_result = _parse_consolidation_output(raw_output)
            except Exception as e:
                logger.warning("curator LLM 审查失败: %s", e)
                consolidation_result = {
                    "consolidations": [],
                    "prunings": [],
                    "error": str(e),
                }

    # 写报告 + 更新状态
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    state = load_state(skills_dir)
    state["last_run_at"] = now.isoformat()
    state["last_run_summary"] = (
        f"转换: {counts}; "
        f"合并: {len(consolidation_result.get('consolidations', []))}; "
        f"耗时: {duration:.1f}s"
    )
    save_state(skills_dir, state)

    return {
        "transitions": counts,
        "consolidations": consolidation_result,
        "duration_seconds": duration,
        "dry_run": dry_run,
    }


def _render_candidate_list(skills_dir: Path, skill_names: list) -> str:
    """渲染候选技能列表给 LLM 看。"""
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
    """从 LLM 输出中解析结构化 YAML 块。"""
    # 找 ```yaml ... ``` 块
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
