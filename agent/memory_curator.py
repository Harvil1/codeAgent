"""MemoryCurator:后台记忆维护系统。

两阶段:
  第 1 阶段(本模块 apply_automatic_transitions):
    纯函数,按 expected_valid_days 判断过期,转换 state。
  第 2 阶段(run_memory_review,LLM 合并 + 矛盾检测):
    LLM 在 type 桶内找重复/矛盾,改写 body + 归档。

设计原则(沿用 OMNIMATE.md "完全可逆"):
  - 永不物理删除
  - archived 是终态,移到 .archive/
  - 所有改动可回滚(.archive/ 完整保留)
"""

import datetime
import logging
import re
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
    *,
    store=None,
) -> Dict[str, int]:
    """第 1 阶段:确定性状态转换。纯函数,无 LLM。

    规则:
      age > 2 * valid_days  → archived(移 .archive/)
      age > valid_days      → stale
      age ≤ valid_days + state==stale → active(reactivated)
      archived 终态,不动

    memory_dir: ~/.OmniMate/.memory/
    返回计数 dict。
    """
    # 延迟导入避免循环依赖
    from agent.memory_store import MemoryStore

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}

    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        return counts

    # 用 memory_dir 的 parent 当 omnimate_home
    omnimate_home = memory_dir.parent
    # X1 fix: 优先用传入的 store（共享主 agent 实例，threading.Lock 跨线程互斥）
    # 之前每次新建 MemoryStore，与主 agent 实例不同，并发写同一 topic.jsonl 会丢数据
    if store is None:
        store = MemoryStore(omnimate_home=omnimate_home)

    # S5 fix: 用标准接口 list_all() 拿所有条目，不再扫老 .md 文件（dead code）
    # 之前直接读 markdown 文件但 MemoryStore 写 .jsonl + 索引 MEMORY.md，扫不到任何条目
    try:
        all_entries = store.list_all()
    except Exception as e:
        logger.warning("curator list_all 失败: %s", e)
        return counts

    for entry in all_entries:
        try:
            # 跳过已 archived
            state = getattr(entry, "state", "active") or "active"
            if state == "archived":
                continue  # 终态,不动

            counts["checked"] += 1

            # 时间戳：MemoryEntry.updated_at 是 datetime 对象（或字符串，兼容）
            updated_at_raw = getattr(entry, "updated_at", None)
            if isinstance(updated_at_raw, str):
                updated_at = _parse_iso(updated_at_raw)
            else:
                updated_at = updated_at_raw  # datetime 对象
            if updated_at is None:
                continue  # 时间戳损坏,跳过(保守)
            # 确保 timezone-aware
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=datetime.timezone.utc)

            age_days = (now - updated_at).total_seconds() / 86400
            valid_days = int(getattr(entry, "expected_valid_days", 365) or 365)
            threshold_stale = valid_days
            threshold_archive = valid_days * 2

            mem_id = entry.id
            if age_days > threshold_archive:
                try:
                    store.delete(mem_id)
                    counts["archived"] += 1
                    logger.info("记忆 %s archived(age=%.0f days > 2×%d)", mem_id, age_days, valid_days)
                except Exception as e:
                    logger.warning("归档记忆 %s 失败: %s", mem_id, e)
            elif age_days > threshold_stale:
                # R30b-A4：update() 已支持 state 字段，stale 标记真正落盘
                # （此前是 no-op 只打日志——两阶段维护第 1 阶段形同虚设）。
                # state-only 更新不刷新 updated_at（memory_store.update 语义），
                # 年龄按内容年龄算，不会翻转。
                try:
                    store.update(mem_id, state="stale")
                    counts["marked_stale"] += 1
                    logger.info(
                        "记忆 %s marked stale(age=%.0f days > %d)",
                        mem_id, age_days, valid_days,
                    )
                except Exception as e:
                    logger.warning("标记 stale 失败 %s: %s", mem_id, e)
            else:
                if state == "stale":
                    # 年龄回落（内容被重新验证过）→ 恢复 active
                    try:
                        store.update(mem_id, state="active")
                        counts["reactivated"] += 1
                        logger.info("记忆 %s reactivated(age=%.0f days ≤ %d)", mem_id, age_days, valid_days)
                    except Exception as e:
                        logger.warning("恢复 active 失败 %s: %s", mem_id, e)
        except Exception as e:
            logger.warning("处理记忆条目 %s 失败: %s", getattr(entry, "id", "?"), e)

    return counts


# ---------------------------------------------------------------------------
# 状态文件 + 门控(照搬 skill Curator 模式)
# ---------------------------------------------------------------------------

def _state_file_path(memory_dir: Path) -> Path:
    """状态文件路径:~/.OmniMate/.memory/.curator_state.json"""
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
    config: Optional[Dict] = None,
) -> bool:
    """门控:enabled + not paused + 距上次 ≥ interval_hours + 首次种子化。

    config 参数(Optional):
      config["memory"]["curator"]["enabled"] = False  → 整个 memory curator 关闭
      config["memory"]["curator"]["interval_hours"]   → 覆盖默认 168(7 天)
    缺省/老 config 无此段时按默认值跑(向后兼容)。
    """
    # config 门控:enabled=False 直接拒绝
    if config is not None:
        cur_cfg = config.get("memory", {}).get("curator", {})
        if not cur_cfg.get("enabled", True):
            return False
        # config 覆盖 interval_hours
        interval_hours = cur_cfg.get("interval_hours", interval_hours)

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
请逐条/逐对检查,识别以下五种情况之一(R19 #22 对齐 CC autoDream 三动作,矛盾解决已有):

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

3. **被证伪**: 记忆陈述的事实已被后续信息明确推翻(不是偏好变化,是事实错误——
   如环境已迁移/结论已过时/诊断被更正),保留会误导未来决策
   操作: 归档该条目
   YAML 输出:
     - action: delete_falsified
       archive: <条目 id>
       evidence: <被哪条记忆/什么信息证伪,一句话>

4. **相对日期**: body 里有"昨天/上周/最近/目前"等相对时间表述,随时间推移会失真
   操作: 改写 body,把相对日期替换为绝对日期(按 updated_at 推算,如"昨天"→"YYYY-MM-DD")
   YAML 输出:
     - action: normalize_dates
       update_id: <条目 id>
       new_body: |
         <改写后的完整 body(只替换相对日期,其余逐字保留)>

5. **无关**: 只是名字或主题相近,内容不重叠
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
    store = MemoryStore(omnimate_home=Path(memory_dir).parent)
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
    R19 #24：new_body 是 LLM 重写产物——秘密扫描命中则拒绝改写保留原文
    （返回 None 表示拒绝，调用方无需感知差异）。
    """
    from agent.secret_scanner import find_secrets_in
    if find_secrets_in(new_body):
        logger.warning(
            "curator 改写产物疑似含密钥（entry %s），拒绝改写保留原文",
            entry_id,
        )
        return None
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
      delete_falsified {archive, evidence}          R19 #22：删除被证伪事实
      normalize_dates {update_id, new_body}         R19 #22：相对日期转绝对日期
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

    if act_type == "delete_falsified":
        # R19 #22：被证伪事实——软删除（archive，完全可逆，对齐设计原则 3）
        archive_id = action.get("archive")
        evidence = action.get("evidence", "")
        if archive_id:
            try:
                store.delete(archive_id)
            except Exception as e:
                logger.warning("delete_falsified 归档 %s 失败: %s", archive_id, e)
        return f"delete_falsified: archived={archive_id} ({evidence[:60]})"

    if act_type == "normalize_dates":
        # R19 #22：相对日期转绝对日期（只改 body，原文经 safe_rewrite_body 备份）
        update_id = action.get("update_id")
        new_body = action.get("new_body", "")
        if update_id and new_body:
            safe_rewrite_body(store, update_id, new_body, archive_root)
        return f"normalize_dates: updated={update_id}"

    logger.warning("未知 curator action: %s", act_type)
    return f"skip: 未知 action {act_type}"


# ---------------------------------------------------------------------------
# Task 8: run_memory_review 主入口(第 2 阶段 LLM 合并 + 矛盾检测)
# ---------------------------------------------------------------------------


def run_memory_review(
    memory_dir: Path,
    *,
    agent_factory,
    dry_run: bool = False,
    max_batch_size: int = 30,
    config: Optional[Dict] = None,
) -> Dict:
    """第 2 阶段:LLM 合并 + 矛盾检测。

    agent_factory: () -> AIAgent(主模型后台 agent),每次调用只构造一次
    dry_run: True 时只统计候选,不调 LLM(避免成本)
    max_batch_size: 每批发给 LLM 的最大记忆条数
    config: Optional 配置字典。读取 config["memory"]["curator"]:
      - llm_review_enabled = False → 直接跳过第 2 阶段,不构造 agent
      - max_batch_size            → 覆盖默认 30
    缺省/老 config 无此段时按默认值跑(向后兼容)。

    流程:
      1. collect_review_candidates 按 type 分桶
      2. dry_run → 直接返回候选统计,不构造 agent
      3. 否则构造一次 agent,遍历每桶每批
      4. 每批构造 prompt → agent.chat() → parse_yaml_actions → execute_action
      5. LLM 失败该批跳过(errors+1),action 执行失败也 errors+1

    返回报告 dict:
      {
        "dry_run": bool,
        "buckets_reviewed": int,   # 实际跑过 LLM 的批数
        "candidates_found": int,   # 候选总数(2+ 条的桶)
        "executed_actions": int,
        "errors": int,
      }
    """
    memory_dir = Path(memory_dir)
    archive_root = memory_dir.parent / ".archive"

    # config 门控:llm_review_enabled=False 直接跳过(避免 LLM 成本)
    if config is not None:
        cur_cfg = config.get("memory", {}).get("curator", {})
        if not cur_cfg.get("llm_review_enabled", True):
            logger.info("run_memory_review 跳过:llm_review_enabled=False")
            return {
                "dry_run": dry_run,
                "buckets_reviewed": 0,
                "executed_actions": 0,
                "errors": 0,
                "candidates_found": 0,
                "skipped": "llm_review_enabled=False",
            }
        max_batch_size = cur_cfg.get("max_batch_size", max_batch_size)

    # dry_run 短路:不构造 agent,避免 LLM 成本
    if dry_run:
        buckets = collect_review_candidates(memory_dir)
        return {
            "dry_run": True,
            "buckets_reviewed": 0,
            "candidates_found": sum(len(v) for v in buckets.values()),
            "executed_actions": 0,
            "errors": 0,
        }

    buckets = collect_review_candidates(memory_dir)
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=memory_dir.parent)

    total_actions = 0
    errors = 0
    buckets_reviewed = 0

    # 复用一个 agent 实例(避免每个批都重新构造)
    try:
        review_agent = agent_factory()
    except Exception as e:
        logger.warning("创建 review agent 失败: %s", e)
        return {
            "dry_run": False, "buckets_reviewed": 0,
            "candidates_found": sum(len(v) for v in buckets.values()),
            "executed_actions": 0, "errors": 1,
        }

    import json
    for type_name, entries in buckets.items():
        for batch in chunk_batch(entries, size=max_batch_size):
            buckets_reviewed += 1
            # 构造 prompt:把 batch 序列化成 JSON
            entries_json = json.dumps([
                {
                    "id": e.id, "name": e.name, "description": e.description,
                    "body": e.body,
                    "updated_at": e.updated_at.isoformat(timespec="seconds"),
                }
                for e in batch
            ], ensure_ascii=False, indent=2)

            prompt = MEMORY_REVIEW_PROMPT_TEMPLATE.format(
                type_name=type_name, n=len(batch), entries_json=entries_json,
            )

            # LLM 调用单批 try/except —— 一批失败不污染其他批
            try:
                # Task D4 fix: AIAgent.chat 已改 async。run_memory_review 是 sync 函数。
                import asyncio
                raw_output = asyncio.run(review_agent.chat(prompt))
            except Exception as e:
                logger.warning("LLM 调用失败(type=%s): %s", type_name, e)
                errors += 1
                continue

            actions = parse_yaml_actions(raw_output)
            for action in actions:
                # execute_action 内部已对 store.delete/rewrite 做了 try/except
                # 这里再兜一层,确保任何异常都不中断主循环
                try:
                    result = execute_action(action, store, archive_root)
                    total_actions += 1
                    logger.info("执行: %s", result)
                except Exception as e:
                    logger.warning("action 执行失败: %s | action=%s", e, action)
                    errors += 1

    return {
        "dry_run": False,
        "buckets_reviewed": buckets_reviewed,
        "candidates_found": sum(len(v) for v in buckets.values()),
        "executed_actions": total_actions,
        "errors": errors,
    }
