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
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional

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


# ---------------------------------------------------------------------------
# 第 2 阶段:候选收集 + 分桶 + 分批
# ---------------------------------------------------------------------------

MEMORY_REVIEW_PROMPT_TEMPLATE = """你是后台记忆库管理员。下面是同一个分类(type={type_name})下的 {n} 条记忆。
请逐对检查,识别以下三种关系之一:

1. **重复**: 多条记忆描述实质相同的事实
   操作: 选一条最完整/最新的作为主条目,其余归档
   YAML 输出:
     - action: merge_duplicate
       keep: <主条目 id>
       archive: [<弃用 id>, ...]
       reason: <一句话>

2. **矛盾**: 用户偏好/习惯发生变化,新旧冲突
   操作: 把新信息整合进旧条目的 body,然后归档新条目
   YAML 输出:
     - action: resolve_contradiction
       update_id: <旧条目 id>
       new_body: |
         <整合后的完整 body,含"原 X,YYYY-MM 改为 Y"说明>
       archive: <新条目 id>
       reason: <一句话>

3. **无关**: 只是名字或主题相近,内容不重叠
   不输出任何东西

完整记忆列表(JSON):
{entries_json}

只输出 ```yaml ... ``` 代码块,不要其他文字。
"""


def collect_review_candidates(memory_dir: Path) -> Dict[str, List]:
    """收集 state=active 的记忆,按 type 分桶。

    返回 dict:{type_name: [MemoryEntry, ...]}
    只保留 2+ 条的桶(单条不可能重复/矛盾)。
    """
    from agent.memory_store import MemoryStore
    store = MemoryStore(harvil_home=Path(memory_dir).parent)
    all_entries = store.list_all()
    buckets: Dict[str, List] = {}
    for entry in all_entries:
        if entry.state != "active":
            continue
        buckets.setdefault(entry.type, []).append(entry)
    # 只保留 2+ 条
    return {k: v for k, v in buckets.items() if len(v) >= 2}


def chunk_batch(entries: List, size: int = 30) -> Iterator[List]:
    """把列表切成 size 大小的批。"""
    for i in range(0, len(entries), size):
        yield entries[i:i + size]


# ---------------------------------------------------------------------------
# Task 7: YAML 解析 + action 执行 + 改写备份
# ---------------------------------------------------------------------------


def parse_yaml_actions(raw: str) -> List[Dict]:
    """从 LLM 输出解析 YAML action 列表。

    支持格式:包含 ```yaml ... ``` 代码块。
    损坏/无块返回空列表。
    """
    match = re.search(r"```yaml\n(.*?)```", raw, re.DOTALL)
    if not match:
        return []
    try:
        import yaml
        parsed = yaml.safe_load(match.group(1))
        if not isinstance(parsed, list):
            return []
        return [a for a in parsed if isinstance(a, dict) and "action" in a]
    except Exception as e:
        logger.warning("YAML 解析失败: %s", e)
        return []


def safe_rewrite_body(store, entry_id: str, new_body: str, archive_root: Path) -> Path:
    """改写 body 前备份原文到 .archive/memory-rewrites-{ts}/。

    返回备份文件路径。
    """
    entry = store.get(entry_id)
    if entry is None:
        raise KeyError(f"记忆不存在: {entry_id}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(archive_root) / f"memory-rewrites-{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{entry_id}.md"
    backup_path.write_text(
        f"---\nname: {entry.name}\ndescription: {entry.description}\n"
        f"original_updated_at: {entry.updated_at.isoformat(timespec='seconds')}\n---\n"
        f"{entry.body}",
        encoding="utf-8",
    )
    store.update(entry_id, body=new_body)
    logger.info("记忆 %s body 改写,原文备份: %s", entry_id, backup_path)
    return backup_path


def execute_action(action: Dict, store, archive_root: Path) -> str:
    """执行单个 curator action。返回结果描述(用于日志/报告)。

    支持:
      merge_duplicate {keep, archive: [ids]}
      resolve_contradiction {update_id, new_body, archive}
    未知 action 跳过。
    """
    act_type = action.get("action")

    if act_type == "merge_duplicate":
        keep_id = action.get("keep")
        archive_ids = action.get("archive", [])
        if isinstance(archive_ids, str):
            archive_ids = [archive_ids]
        for aid in archive_ids:
            if aid and aid != keep_id:
                try:
                    store.delete(aid)
                except Exception as e:
                    logger.warning("merge_duplicate 归档 %s 失败: %s", aid, e)
        return f"merge_duplicate: keep={keep_id}, archived={archive_ids}"

    if act_type == "resolve_contradiction":
        update_id = action.get("update_id")
        new_body = action.get("new_body", "")
        archive_id = action.get("archive")
        if update_id and new_body:
            safe_rewrite_body(store, update_id, new_body, archive_root)
        if archive_id and archive_id != update_id:
            try:
                store.delete(archive_id)
            except Exception as e:
                logger.warning("resolve_contradiction 归档 %s 失败: %s", archive_id, e)
        return f"resolve_contradiction: updated={update_id}, archived={archive_id}"

    logger.warning("未知 curator action: %s", act_type)
    return f"skip: 未知 action {act_type}"
