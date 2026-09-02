"""记忆管理员（MemoryCurator）：后台自动维护记忆库的系统。

记忆会越攒越多、越放越旧，需要有个"图书管理员"定期整理，分两个阶段：

第 1 阶段（本模块的 apply_automatic_transitions）：
  纯时间规则，不调 LLM——按 expected_valid_days（预期有效天数）
  判断过期，转换状态（标旧/归档）。
第 2 阶段（run_memory_review）：
  调 LLM 在同类型记忆里找重复/矛盾，改写正文 + 归档。

设计原则（沿用 CODEAGENT.md 的"完全可逆"）：
  - 永不物理删除文件
  - archived（归档）是终点状态，条目挪到 .archive/
  - 所有改动都可回滚（.archive/ 里完整保留原文）
"""

import datetime
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def _parse_iso(value) -> Optional[datetime.datetime]:
    """解析 ISO 格式的时间戳字符串。空值或解析失败返回 None。"""
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
    """第 1 阶段：按纯时间规则转换记忆状态，不调 LLM。

    判断标准（年龄 = 现在减最后更新时间）：
      年龄 > 2 × 有效天数 → 归档（挪 .archive/）
      年龄 > 有效天数     → 标 stale（疑似过时）
      年龄 ≤ 有效天数 且当前是 stale → 恢复 active（内容被重新验证过了）
      已归档的是终点状态，不再动

    参数：
    - memory_dir：记忆目录（~/.codeAgent/.memory/）
    - now：当前时间（不传用系统时间，测试可注入）
    - store：MemoryStore 实例（不传就现建一个）
    返回：各动作的计数 dict。
    """
    # 函数内 import，防止模块加载时互相依赖成环
    from agent.memory_store import MemoryStore

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}

    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        return counts

    # 拿 memory_dir 的上一级当 codeagent_home
    codeagent_home = memory_dir.parent
    # 优先用传进来的 store（和主 agent 共用一个实例，锁才能跨线程互斥）；
    # 现建 MemoryStore 的话和主实例不是同一把锁，
    # 并发写同一个 topic.jsonl 会丢数据
    if store is None:
        store = MemoryStore(codeagent_home=codeagent_home)

    # 必须用标准接口 list_all() 拿条目，不要直接扫文件——
    # MemoryStore 实际写的是 .jsonl，直接读 markdown 根本扫不到东西（死代码）
    try:
        all_entries = store.list_all()
    except Exception as e:
        logger.warning("curator list_all 失败: %s", e)
        return counts

    for entry in all_entries:
        try:
            # 已归档的直接跳过
            state = getattr(entry, "state", "active") or "active"
            if state == "archived":
                continue  # 终点状态，不再动

            counts["checked"] += 1

            # 时间戳：updated_at 可能是 datetime 对象也可能是字符串（两种都兼容）
            updated_at_raw = getattr(entry, "updated_at", None)
            if isinstance(updated_at_raw, str):
                updated_at = _parse_iso(updated_at_raw)
            else:
                updated_at = updated_at_raw  # datetime 对象
            if updated_at is None:
                continue  # 时间戳坏了就跳过这条（保守处理）
            # 统一成带时区的时间
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
                # 标 stale 走 update() 的 state 字段，真正落盘。
                # 只改 state 不刷新 updated_at（memory_store.update 的语义），
                # 年龄按内容年龄算，不会 stale↔active 反复翻转。
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
# 状态文件 + 运行门控（照搬技能 Curator 的模式）
# ---------------------------------------------------------------------------

def _state_file_path(memory_dir: Path) -> Path:
    """状态文件路径：~/.codeAgent/.memory/.curator_state.json（记上次运行时间等）。"""
    return Path(memory_dir) / ".curator_state.json"


def load_memory_curator_state(memory_dir: Path) -> Dict:
    """读状态文件。文件不存在或读失败返回空 dict。"""
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
    """原子写状态文件（先写临时文件再改名，中途断电不会写坏）。"""
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
    """判断现在该不该跑 memory curator：开了 + 没暂停 + 距上次 ≥ 间隔 + 首次只播种。

    参数：
    - memory_dir：记忆目录
    - now：当前时间（不传用系统时间）
    - interval_hours：间隔小时数，默认 168（7 天）
    - config：配置字典（可选）。读 config["memory"]["curator"]：
      - enabled = False → 整个 memory curator 关掉
      - interval_hours → 覆盖默认 168
      配置缺这一节时按默认跑。
    返回：该跑 True / 不该跑 False。
    """
    # 配置门控：enabled=False 直接不跑
    if config is not None:
        cur_cfg = config.get("memory", {}).get("curator", {})
        if not cur_cfg.get("enabled", True):
            return False
        # 配置覆盖间隔时长
        interval_hours = cur_cfg.get("interval_hours", interval_hours)

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    state = load_memory_curator_state(memory_dir)
    if state.get("paused"):
        return False

    last_str = state.get("last_run_at")
    if not last_str:
        # 首次运行：只播下种子（记下当前时间），等满一个周期再跑
        state["last_run_at"] = now.isoformat()
        state["last_run_summary"] = "首次运行已推迟——curator 已种子化,等一个周期"
        state["paused"] = False
        save_memory_curator_state(memory_dir, state)
        return False

    last = _parse_iso(last_str)
    if last is None:
        # 时间戳坏了，重置重新计时
        state["last_run_at"] = now.isoformat()
        save_memory_curator_state(memory_dir, state)
        return False

    if last.tzinfo is None:
        last = last.replace(tzinfo=datetime.timezone.utc)

    elapsed = (now - last).total_seconds() / 3600
    return elapsed >= interval_hours


# ---------------------------------------------------------------------------
# 第 2 阶段：收集候选 + 按类型分桶 + 切批
# ---------------------------------------------------------------------------

MEMORY_REVIEW_PROMPT_TEMPLATE = """你是后台记忆库管理员。下面是同一个分类(type={type_name})下的 {n} 条记忆。
请逐条/逐对检查,识别以下五种情况之一:

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
    """收集所有 state=active 的记忆，按 type 分桶。

    只保留有 2 条以上的桶——单独一条不可能和自己重复/矛盾。

    参数：
    - memory_dir：记忆目录
    返回：{类型名: [记忆条目, ...]}
    """
    from agent.memory_store import MemoryStore
    store = MemoryStore(codeagent_home=Path(memory_dir).parent)
    all_entries = store.list_all()
    buckets: Dict[str, List] = {}
    for entry in all_entries:
        if entry.state != "active":
            continue
        buckets.setdefault(entry.type, []).append(entry)
    # 只留 2 条以上的桶
    return {k: v for k, v in buckets.items() if len(v) >= 2}


def chunk_batch(entries: List, size: int = 30) -> Iterator[List]:
    """把列表切成每批最多 size 条（防止一次发给 LLM 的内容太多）。

    参数：
    - entries：条目列表
    - size：每批大小，默认 30
    """
    for i in range(0, len(entries), size):
        yield entries[i:i + size]


# ---------------------------------------------------------------------------
# YAML 解析 + action 执行 + 改写前备份
# ---------------------------------------------------------------------------


def parse_yaml_actions(raw: str) -> List[Dict]:
    """从 LLM 输出里解析出 YAML 格式的 action 列表。

    认的格式：```yaml ... ``` 代码块。没有代码块或解析失败返回空列表
    （LLM 输出不可信，坏了就当没有这批）。

    参数：
    - raw：LLM 的原始输出文本
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
    """改写记忆正文前，先把原文备份到 .archive/memory-rewrites-{时间戳}/。

    返回备份文件路径；条目不存在抛 KeyError。
    安全项：新正文是 LLM 重写的产物——疑似含密钥就拒绝改写、
    保留原文（返回 None 表示拒绝，调用方不用区分这种情况）。

    参数：
    - store：MemoryStore 实例
    - entry_id：记忆完整 id
    - new_body：要写入的新正文
    - archive_root：归档根目录（.archive/）
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
    """执行一条 curator 动作。返回结果描述（写日志/报告用）。

    支持：
      merge_duplicate {keep, archive: [ids]}        合并重复：留一条，归档其余
      resolve_contradiction {update_id, new_body, archive}  矛盾解决：改旧条目并归档新条目
      delete_falsified {archive, evidence}          删除被证伪事实
      normalize_dates {update_id, new_body}         相对日期转绝对日期
    不认识的 action 跳过。

    参数：
    - action：LLM 输出的一条动作（dict）
    - store：MemoryStore 实例
    - archive_root：归档根目录
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
        # 被证伪的事实——软删除到归档，完全可逆
        archive_id = action.get("archive")
        evidence = action.get("evidence", "")
        if archive_id:
            try:
                store.delete(archive_id)
            except Exception as e:
                logger.warning("delete_falsified 归档 %s 失败: %s", archive_id, e)
        return f"delete_falsified: archived={archive_id} ({evidence[:60]})"

    if act_type == "normalize_dates":
        # 相对日期转绝对日期——只改正文，原文经 safe_rewrite_body 备份过
        update_id = action.get("update_id")
        new_body = action.get("new_body", "")
        if update_id and new_body:
            safe_rewrite_body(store, update_id, new_body, archive_root)
        return f"normalize_dates: updated={update_id}"

    logger.warning("未知 curator action: %s", act_type)
    return f"skip: 未知 action {act_type}"


# ---------------------------------------------------------------------------
# 第 2 阶段主入口：LLM 合并 + 矛盾检测
# ---------------------------------------------------------------------------


def run_memory_review(
    memory_dir: Path,
    *,
    agent_factory,
    dry_run: bool = False,
    max_batch_size: int = 30,
    config: Optional[Dict] = None,
) -> Dict:
    """第 2 阶段主入口：调 LLM 做记忆合并 + 矛盾检测。

    流程：
      1. collect_review_candidates 按类型分桶
      2. dry_run=True → 只统计候选数，不调 LLM（省成本）
      3. 否则构造一次后台 agent，遍历每桶每批
      4. 每批拼 prompt → agent.chat() → parse_yaml_actions → execute_action
      5. LLM 失败该批跳过（errors+1），单个 action 失败也 errors+1

    参数：
    - memory_dir：记忆目录
    - agent_factory：造后台 agent 的工厂，() -> AIAgent（整个流程只造一次）
    - dry_run：True 只统计不调 LLM
    - max_batch_size：每批发给 LLM 的最大条数
    - config：配置字典（可选），读 config["memory"]["curator"]：
      - llm_review_enabled = False → 跳过整个第 2 阶段（不造 agent，省成本）
      - max_batch_size → 覆盖默认 30
      配置缺这一节时按默认跑。
    返回：报告 dict（dry_run / buckets_reviewed（实际跑过 LLM 的批数）/
      candidates_found（候选总数）/ executed_actions / errors）。
    """
    memory_dir = Path(memory_dir)
    archive_root = memory_dir.parent / ".archive"

    # 配置门控：llm_review_enabled=False 直接跳过（省 LLM 成本）
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

    # dry_run 短路：不造 agent，不花 LLM 的钱
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
    store = MemoryStore(codeagent_home=memory_dir.parent)

    total_actions = 0
    errors = 0
    buckets_reviewed = 0

    # 全程复用一个 agent（省得每批都重新构造一遍）
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
            # 拼 prompt：把这一批条目序列化成 JSON
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

            # 每批单独 try/except——一批失败不连累其他批
            try:
                # 本函数是 sync 的，AIAgent.chat 是 async 的，用 asyncio.run 驱动
                import asyncio
                raw_output = asyncio.run(review_agent.chat(prompt))
            except Exception as e:
                logger.warning("LLM 调用失败(type=%s): %s", type_name, e)
                errors += 1
                continue

            actions = parse_yaml_actions(raw_output)
            for action in actions:
                # execute_action 内部已对删除/改写做了异常兜底；
                # 这里再包一层，保证任何异常都打不断主循环
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
