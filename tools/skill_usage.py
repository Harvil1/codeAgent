"""技能使用统计和来龙去脉记录。

记录每个技能「多久没用、被看过/用过几次、是谁创建的」，供决定哪些技能
该归档、哪些值得推荐。这些数字单独存在 ~/.codeAgent/skills/.usage.json 里
（键是技能名），不写进技能文件本身。

谁来用：计数由 skill_view / skill_manage 等工具在干活时顺手触发；
后台维护工人（curator，定期整理技能/记忆的程序）读活动时间戳来决定技能的生命周期转换。

设计原则（为什么做成这样）：
1. 统计单独放一个「伴生文件」（sidecar），不写进 SKILL.md——技能正文是给用户/LLM 看的，
   塞进计数会污染内容
2. 写盘用原子写（先写临时文件再一步替换），断电/中断也不会留下半个坏文件
3. 所有计数都是「尽力而为」：失败了只记日志，绝不影响工具本身的调用
4. 只有 created_by="agent"（后台 curator 创建）的技能才受 curator 自动管理，
   用户建的技能 curator 不动

生命周期状态（像食品保质期一样随闲置时间流转）：
    active   - 默认的健康状态
    stale    - 超过 30 天（stale_after_days）没活动，标记为「陈旧」
    archived - 超过 90 天（archive_after_days）没活动；目录挪到 .archive/ 下
    pinned   - 用户手动钉住，免疫一切自动转换（跟 state 平行的一个独立开关）
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_file(skills_dir: Path) -> Path:
    return Path(skills_dir) / ".usage.json"


# 内存缓存 + 懒写盘：记数先攒在内存里，等收尾时一次性落盘。
# 为什么：查看/使用很频繁，每次都全量读写 .usage.json 太浪费。
_usage_cache: Dict[str, dict] = {}


def load_usage(skills_dir: Path) -> Dict[str, Dict[str, Any]]:
    """读出某目录的全部使用统计。带内存缓存：第一次读盘之后就一直用内存里的副本。

    参数：
    - skills_dir：技能目录路径

    返回：字典，键是技能名，值是该技能的统计记录；文件不存在或损坏时返回空字典。
    """
    key = str(skills_dir)
    if key in _usage_cache:
        return _usage_cache[key]["data"]
    path = _usage_file(Path(skills_dir))
    if not path.exists():
        data = {}
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    _usage_cache[key] = {"data": data, "dirty": False}
    return data


def save_usage(skills_dir: Path, data: Dict[str, Dict[str, Any]]) -> None:
    """把改过的统计存回缓存并打上「待写盘」标记（dirty），但不立刻写文件。

    参数：
    - skills_dir：技能目录路径
    - data：完整的统计数据字典
    """
    _usage_cache[str(skills_dir)] = {"data": data, "dirty": True}


def flush_usage(skills_dir: Path = None) -> None:
    """把攒在内存里改过的统计真正写进磁盘文件。

    平时只动内存缓存，由本函数在收尾时机（agent 退出、每轮对话结束）统一落盘。

    参数：
    - skills_dir：只写这个目录的缓存；传 None 表示把所有目录的都写一遍

    返回：无。写失败的只记 debug 日志，不抛错（统计是小事，不值得打断主流程）。
    """
    from agent.atomic_io import atomic_write_text
    for key, entry in _usage_cache.items():
        if not entry["dirty"]:
            continue
        if skills_dir and key != str(skills_dir):
            continue
        try:
            path = _usage_file(Path(key))
            atomic_write_text(path, json.dumps(entry["data"], ensure_ascii=False, indent=2))
            entry["dirty"] = False
        except Exception as e:
            logger.debug("flush_usage 失败 %s: %s", key, e)


def _ensure_record(data: Dict, skill_name: str) -> Dict:
    """确保统计字典里有这个技能的记录（没有就补一条空白模板），并返回这条记录。

    参数：
    - data：全部统计数据的字典（会原地修改）
    - skill_name：技能名

    返回：该技能的统计记录字典。
    """
    if skill_name not in data:
        data[skill_name] = {
            "created_by": "user",
            "use_count": 0,
            "view_count": 0,
            "patch_count": 0,
            "last_used_at": None,
            "last_viewed_at": None,
            "last_patched_at": None,
            "created_at": _now_iso(),
            "state": STATE_ACTIVE,
            "pinned": False,
            "archived_at": None,
        }
    return data[skill_name]


def bump_view(skills_dir: Path, skill_name: str) -> None:
    """把某技能的「被查看次数」加一，并记下这次查看的时间。

    skill_view 工具看技能时调用，给推荐排序和闲置判定提供数据。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：无。失败只记 debug 日志（尽力而为）。
    """
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["view_count"] = int(rec.get("view_count", 0)) + 1
        rec["last_viewed_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_view 失败: %s", e)


def bump_use(skills_dir: Path, skill_name: str) -> None:
    """把某技能的「实际使用次数」加一，并记下这次使用的时间。

    技能被当成斜杠命令（在输入框里打 /技能名 触发）真正执行时调用。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：无。失败只记 debug 日志。
    """
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["use_count"] = int(rec.get("use_count", 0)) + 1
        rec["last_used_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_use 失败: %s", e)


def bump_patch(skills_dir: Path, skill_name: str) -> None:
    """把某技能的「被修改次数」加一，并记下这次修改的时间。

    skill_manage 工具做 patch（小修）或 edit（重写）后调用。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：无。失败只记 debug 日志。
    """
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["patch_count"] = int(rec.get("patch_count", 0)) + 1
        rec["last_patched_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_patch 失败: %s", e)


def mark_agent_created(skills_dir: Path, skill_name: str) -> None:
    """把技能标记为「agent 创建」，从此受后台维护工人（curator）自动管理。

    关键区分：只有 curator 在后台自主审查时创建的技能才标记；用户当面让
    agent 建的不标记——用户亲手要的东西，后台不该自作主张去归档或改它。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：无。失败只记 debug 日志。
    """
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["created_by"] = "agent"
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("mark_agent_created 失败: %s", e)


def set_state(skills_dir: Path, skill_name: str, state: str) -> None:
    """设置技能的生命周期状态（active/stale/archived）。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名
    - state：目标状态；不在三个合法值里就直接忽略

    返回：无。只管 agent 创建的技能；技能没有记录或出错时静默跳过。
    """
    if state not in _VALID_STATES:
        return
    try:
        data = load_usage(skills_dir)
        if skill_name not in data:
            return
        rec = data[skill_name]
        # 只自动管理 agent 创建的技能；用户建的动状态属于越权
        if rec.get("created_by") != "agent":
            return
        rec["state"] = state
        if state == STATE_ARCHIVED:
            rec["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            rec["archived_at"] = None
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("set_state 失败: %s", e)


def set_pinned(skills_dir: Path, skill_name: str, pinned: bool) -> None:
    """给技能钉上/取消「钉住」（pinned）标记。被钉住的技能免疫所有自动转换。

    这是用户表达「这个技能我要留着，别动」的开关——用户的明确意图优先于算法。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名
    - pinned：True 钉住 / False 取消

    返回：无。同样只对 agent 创建的技能生效；出错静默跳过。
    """
    try:
        data = load_usage(skills_dir)
        if skill_name not in data:
            return
        rec = data[skill_name]
        if rec.get("created_by") != "agent":
            return
        rec["pinned"] = bool(pinned)
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("set_pinned 失败: %s", e)


def archive_skill(skills_dir: Path, skill_name: str) -> tuple:
    """把技能整个目录挪到 .archive/ 目录下（回收站式的「软删除」）。

    永不真删除，归档的东西随时能捞回来（项目铁律「完全可逆」）。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：(是否成功, 给人看的消息) 二元组。
    技能不存在、归档位置已被占用（同名冲突）或移动失败都会返回失败。
    """
    skill_dir = Path(skills_dir) / skill_name
    if not skill_dir.exists():
        return False, f"技能不存在: {skill_name}"

    archive_dir = Path(skills_dir) / ".archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    target = archive_dir / skill_name
    if target.exists():
        return False, f"归档已存在: {skill_name}"

    try:
        skill_dir.rename(target)
        set_state(skills_dir, skill_name, STATE_ARCHIVED)
        return True, f"已归档到 {target}"
    except Exception as e:
        return False, f"归档失败: {e}"


def restore_skill(skills_dir: Path, skill_name: str) -> tuple:
    """把技能从 .archive/ 挪回原位（撤销归档）。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名

    返回：(是否成功, 消息) 二元组。归档里没有它、原位置已被占用或移动失败都返回失败。
    """
    archive_dir = Path(skills_dir) / ".archive"
    src = archive_dir / skill_name
    if not src.exists():
        return False, f"归档中不存在: {skill_name}"

    target = Path(skills_dir) / skill_name
    if target.exists():
        return False, f"技能已存在（与归档冲突）: {skill_name}"

    try:
        src.rename(target)
        set_state(skills_dir, skill_name, STATE_ACTIVE)
        return True, f"已恢复: {skill_name}"
    except Exception as e:
        return False, f"恢复失败: {e}"


# ---------------------------------------------------------------------------
# 技能评分 + 推荐功能
# ---------------------------------------------------------------------------

def set_rating(skills_dir: Path, skill_name: str, rating: int) -> tuple:
    """给技能打 1-5 星的评分（用户的喜好信号，喂给推荐排序用）。

    参数：
    - skills_dir：技能目录路径
    - skill_name：技能名
    - rating：星数，必须是 1 到 5 的整数

    返回：(是否成功, 消息) 二元组。评分不在 1-5 或技能文件不存在时拒绝。
    """
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return False, f"评分越界：{rating}（应为 1-5 整数）"

    # 只要求技能文件真实存在，不要求它已经有统计记录
    skill_path = Path(skills_dir) / skill_name / "SKILL.md"
    if not skill_path.exists():
        return False, f"技能不存在: {skill_name}"

    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["rating"] = rating
        rec["rated_at"] = _now_iso()
        save_usage(skills_dir, data)
        return True, f"已给 /{skill_name} 打 {rating} 星"
    except Exception as e:
        return False, f"评分失败: {e}"


def get_recommendations(skills_dir: Path, limit: int = 5) -> list:
    """算出最值得推荐的技能，按综合分从高到低返回前几个。

    给「你现在可能用得上哪些技能」提供排序依据。
    综合分公式 = 使用次数 × 1.0 + 评分 × 2.0 + 查看次数 × 0.1
    （评分权重最大，因为那是用户的直接喜好；查看只值 0.1，看过不等于有用。）

    参数：
    - skills_dir：技能目录路径
    - limit：最多返回几个，默认 5

    返回：列表，每项是 {"name", "score", "use_count", "rating", "view_count",
    "pinned", "description"}。被钉住（pinned）的技能加 10 分强力置顶；
    已归档的不参与。
    """
    data = load_usage(skills_dir)
    sd = Path(skills_dir)

    candidates = []
    for skill_md in sorted(sd.glob("*/SKILL.md")):
        name = skill_md.parent.name
        rec = data.get(name, {})

        # 归档的不推荐
        if rec.get("state") == STATE_ARCHIVED:
            continue

        use_count = int(rec.get("use_count", 0))
        rating = int(rec.get("rating", 0))
        view_count = int(rec.get("view_count", 0))
        pinned = bool(rec.get("pinned", False))

        # 综合分；钉住的额外加 10 分，保证排在最前
        score = use_count * 1.0 + rating * 2.0 + view_count * 0.1
        if pinned:
            score += 10.0

        # 顺手取一份技能描述，推荐列表里展示用
        description = ""
        try:
            content = skill_md.read_text(encoding="utf-8")
            # 用笨办法从文件头元信息区里抠出 description 那一行（这里不值得上完整解析器）
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    for line in parts[1].splitlines():
                        if line.strip().startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip('"').strip("'")
                            break
        except Exception:
            pass

        candidates.append({
            "name": name,
            "score": round(score, 2),
            "use_count": use_count,
            "rating": rating,
            "view_count": view_count,
            "pinned": pinned,
            "description": description,
        })

    # 分高的排前面，只留前 limit 个
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]
