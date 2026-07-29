"""MemoryCurator:后台记忆维护系统。

两阶段:
  第 1 阶段(本模块 apply_automatic_transitions):
    纯函数,按 expected_valid_days 判断过期,转换 state。
  第 2 阶段(run_memory_review,后续 task 实现):
    LLM 在 type 桶内找重复/矛盾,改写 body + 归档。

设计原则(沿用 CLAUDE.md "完全可逆"):
  - 永不物理删除
  - archived 是终态,移到 .archive/
  - 所有改动可回滚(.archive/ 完整保留)
"""

import datetime
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _parse_iso(value) -> Optional[datetime.datetime]:
    """解析 ISO 时间戳。失败返回 None。"""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def apply_automatic_transitions(
    memory_dir: Path,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, int]:
    """第 1 阶段:确定性状态转换。纯函数,无 LLM。

    规则:
      age > 2 * valid_days  → archived(移 .archive/)
      age > valid_days      → stale
      age ≤ valid_days + state==stale → active(reactivated)
      archived 终态,不动

    memory_dir: ~/.agent/.memory/
    返回计数 dict。
    """
    # 延迟导入避免循环依赖
    from agent.memory_store import MemoryStore, _format_frontmatter

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}

    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        return counts

    # 用 memory_dir 的 parent 当 harvil_home
    harvil_home = memory_dir.parent
    store = MemoryStore(harvil_home=harvil_home)

    for path in sorted(memory_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            from agent.memory_store import _parse_frontmatter
            meta, body = _parse_frontmatter(text)
            if meta is None:
                continue
        except Exception as e:
            logger.warning("读取记忆文件失败 %s: %s", path, e)
            continue

        state = meta.get("state", "active") or "active"
        if state == "archived":
            continue  # 终态,不动

        counts["checked"] += 1

        updated_at = _parse_iso(meta.get("updated_at"))
        if updated_at is None:
            continue  # 时间戳损坏,跳过(保守)
        # 确保 timezone-aware
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)

        age_days = (now - updated_at).total_seconds() / 86400
        valid_days = int(meta.get("expected_valid_days", 365) or 365)
        threshold_stale = valid_days
        threshold_archive = valid_days * 2

        if age_days > threshold_archive:
            # → archived(用 store.delete 走标准归档流程)
            mem_id = path.stem
            try:
                store.delete(mem_id)
                # delete 不写 state 字段(文件已移走),需要额外标记
                # 已移到 .archive/memory-{ts}/,state 字段保留在文件里
                counts["archived"] += 1
                logger.info("记忆 %s archived(age=%.0f days > 2×%d)", mem_id, age_days, valid_days)
            except Exception as e:
                logger.warning("归档记忆 %s 失败: %s", mem_id, e)
        elif age_days > threshold_stale:
            if state != "stale":
                _set_state_in_file(path, "stale", meta, body, now)
                counts["marked_stale"] += 1
                logger.info("记忆 %s stale(age=%.0f days > %d)", path.stem, age_days, valid_days)
        else:
            # age ≤ valid_days
            if state == "stale":
                _set_state_in_file(path, "active", meta, body, now)
                counts["reactivated"] += 1
                logger.info("记忆 %s reactivated(age=%.0f days ≤ %d)", path.stem, age_days, valid_days)

    return counts


def _set_state_in_file(path, new_state, meta, body, now):
    """原地改 frontmatter 的 state 字段 + last_reviewed_at。原子写。"""
    from agent.memory_store import _format_frontmatter
    from agent.atomic_io import atomic_write_text

    meta["state"] = new_state
    meta["last_reviewed_at"] = now.isoformat(timespec="seconds")
    # state=active 时不写(保持老文件干净)
    if new_state == "active":
        meta.pop("state", None)
    atomic_write_text(path, _format_frontmatter(meta) + body)


# ---------------------------------------------------------------------------
# 状态文件 + 门控(照搬 skill Curator 模式)
# ---------------------------------------------------------------------------

def _state_file_path(memory_dir: Path) -> Path:
    """状态文件路径:~/.agent/.memory/.curator_state.json"""
    return Path(memory_dir) / ".curator_state.json"


def load_memory_curator_state(memory_dir: Path) -> Dict:
    """加载状态文件。不存在返回空 dict。"""
    path = _state_file_path(memory_dir)
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("读取 curator 状态失败 %s: %s", path, e)
        return {}


def save_memory_curator_state(memory_dir: Path, state: Dict) -> None:
    """原子写状态文件。"""
    import json
    from agent.atomic_io import atomic_write_text
    path = _state_file_path(memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))


def should_run_now_memory(
    memory_dir: Path,
    now: Optional[datetime.datetime] = None,
    interval_hours: int = 168,
) -> bool:
    """门控:enabled + not paused + 距上次 ≥ interval_hours + 首次种子化。"""
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    state = load_memory_curator_state(memory_dir)
    if state.get("paused"):
        return False

    last_str = state.get("last_run_at")
    if not last_str:
        # 首次运行:种子化,等一个周期
        state["last_run_at"] = now.isoformat()
        state["last_run_summary"] = "首次运行已推迟——curator 已种子化,等一个周期"
        state["paused"] = False
        save_memory_curator_state(memory_dir, state)
        return False

    last = _parse_iso(last_str)
    if last is None:
        # 时间戳损坏,重置
        state["last_run_at"] = now.isoformat()
        save_memory_curator_state(memory_dir, state)
        return False

    if last.tzinfo is None:
        last = last.replace(tzinfo=datetime.timezone.utc)

    elapsed = (now - last).total_seconds() / 3600
    return elapsed >= interval_hours
