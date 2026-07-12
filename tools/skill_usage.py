"""技能使用统计 + provenance 追踪。

存在 ~/.agent/skills/.usage.json，键是技能名。
计数器由 skill_view / skill_manage 工具触发。
curator 读取活动时间戳决定生命周期转换。

设计原则：
1. Sidecar 文件，不写进 SKILL.md（避免污染用户内容）
2. 原子写入（tempfile + os.replace）
3. 所有计数 best-effort，失败不影响工具调用
4. 只有 created_by="agent" 的技能受 curator 管理

生命周期状态：
    active   - 默认
    stale    - 超过 stale_after_days（30天）无活动
    archived - 超过 archive_after_days（90天）；移动到 .archive/
    pinned   - 免疫自动转换（正交于 state 的布尔标志）
"""

import json
import logging
import os
import tempfile
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


def load_usage(skills_dir: Path) -> Dict[str, Dict[str, Any]]:
    """加载使用统计。"""
    path = _usage_file(skills_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_usage(skills_dir: Path, data: Dict[str, Dict[str, Any]]) -> None:
    """原子写入使用统计。"""
    path = _usage_file(skills_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入：先写临时文件，再 rename
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    except Exception:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _ensure_record(data: Dict, skill_name: str) -> Dict:
    """确保技能有使用记录，返回该记录。"""
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
    """查看次数 +1。skill_view() 调用。"""
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["view_count"] = int(rec.get("view_count", 0)) + 1
        rec["last_viewed_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_view 失败: %s", e)


def bump_use(skills_dir: Path, skill_name: str) -> None:
    """使用次数 +1。技能作为 slash 命令被调用时。"""
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["use_count"] = int(rec.get("use_count", 0)) + 1
        rec["last_used_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_use 失败: %s", e)


def bump_patch(skills_dir: Path, skill_name: str) -> None:
    """修改次数 +1。skill_manage(patch/edit) 调用。"""
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["patch_count"] = int(rec.get("patch_count", 0)) + 1
        rec["last_patched_at"] = _now_iso()
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("bump_patch 失败: %s", e)


def mark_agent_created(skills_dir: Path, skill_name: str) -> None:
    """标记技能为 agent 创建（使其受 curator 管理）。

    关键：只有后台 curator 审查时创建的技能才标记。
    用户手动让 agent 创建的不标记。
    """
    try:
        data = load_usage(skills_dir)
        rec = _ensure_record(data, skill_name)
        rec["created_by"] = "agent"
        save_usage(skills_dir, data)
    except Exception as e:
        logger.debug("mark_agent_created 失败: %s", e)


def set_state(skills_dir: Path, skill_name: str, state: str) -> None:
    """设置生命周期状态。"""
    if state not in _VALID_STATES:
        return
    try:
        data = load_usage(skills_dir)
        if skill_name not in data:
            return
        rec = data[skill_name]
        # 只管理 agent 创建的技能
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
    """设置/取消 pin。pinned 技能免疫所有自动转换。"""
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
    """把技能目录移动到 .archive/。

    返回 (是否成功, 消息)。
    永不删除！归档是可恢复的。
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
    """从 .archive/ 恢复技能。"""
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
