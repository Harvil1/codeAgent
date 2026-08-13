"""多文件记忆存储：对齐 Claude Code 主题组织。

存储结构（对齐 Claude Code topic 文件）：
- ~/.OmniMate/.memory/{topic}.jsonl：一个主题一个文件，每行一条记忆（JSON）
- ~/.OmniMate/MEMORY.md：索引（自动生成，按主题分组，注入 system prompt 截断 200行/25KB）

原则：
- 写入即维护：同主题同 name 的记忆自动更新（不无限堆积）
- snapshot_for_prompt() 返回截断索引（省 token + 保 prompt cache），retriever 用 full_index_text()
- 旧格式（每记忆一个 .md 文件）启动时迁移到 topic jsonl
- 删除软删除到 .archive/memory-{ts}/
"""
import json
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from agent.atomic_io import atomic_write_text

# 记忆索引注入 system prompt 的上限（对齐 Claude Code：200 行 / 25KB，先到者）
_INDEX_MAX_LINES = 200
_INDEX_MAX_BYTES = 25000

logger = logging.getLogger(__name__)

VALID_TYPES = {"user", "feedback", "project", "reference", "other"}


@dataclass
class MemoryEntry:
    """单条记忆。"""
    id: str  # 对外格式 {topic}#{uid}
    name: str
    description: str
    type: str
    body: str
    created_at: datetime
    updated_at: datetime
    # 主题（对齐 Claude Code topic 文件）：记忆按主题组织到 .memory/{topic}.jsonl
    topic: str = "general"
    # CCALS-P0-1: L1 摘要层
    summary: str = ""
    confidence: float = 1.0
    expected_valid_days: int = 365
    source_session_id: str = ""
    state: str = "active"
    last_reviewed_at: str = ""


def _generate_id() -> str:
    """生成时间排序的唯一 ID（topic 文件内的块 uid）。"""
    ts = int(datetime.now().timestamp() * 1000)
    short_uuid = uuid.uuid4().hex[:6]
    return f"{ts}{short_uuid}"


def _now_iso() -> str:
    # S6 fix: 用 UTC，避免与 curator 的 UTC 比较时偏差（local vs UTC 错 8 小时）
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_entry_id(entry_id: str) -> Tuple[str, str]:
    """把对外 id（{topic}#{uid}）拆成 (topic, uid)。无 # 时兜底 general。"""
    if "#" in entry_id:
        topic, _, uid = entry_id.rpartition("#")
        return topic or "general", uid
    return "general", entry_id


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """解析 `---\\n...yaml...\\n---\\nbody` 格式（迁移用）。"""
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        meta = yaml.safe_load(parts[1])
        if not isinstance(meta, dict):
            return None, text
        return meta, parts[2].lstrip("\n")
    except yaml.YAMLError as e:
        logger.warning("frontmatter 解析失败: %s", e)
        return None, text


def _format_frontmatter(meta: dict) -> str:
    """把 dict 序列化成 frontmatter 文本（`---\\n...yaml...\\n---\\n`）。

    与 `_parse_frontmatter` 对称。主要用于测试 fixture 构造老格式 .md 文件
    （测 legacy 迁移到 jsonl 的逻辑），以及任何需要写 frontmatter 的工具脚本。
    meta 为空 dict 时返回空串（不带 frontmatter）。
    """
    if not meta:
        return ""
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n"


class MemoryStore:
    """主题组织的多文件记忆存储。"""

    def __init__(self, *, omnimate_home: Path):
        self._home = Path(omnimate_home)
        self._memory_dir = self._home / ".memory"
        self._index_path = self._home / "MEMORY.md"
        self._lock = threading.Lock()
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._cached_snapshot: str = ""
        # topic 行内存缓存（锁内访问）：读路径不再每次读盘，写路径全量重写
        # 仅发生在更新/删除（低频）；新建走文件 append（O(1)）。
        # mtime 失效：跨 MemoryStore 实例写文件（如 curator 自建实例）自动重读。
        self._rows_cache: dict = {}  # topic -> (mtime_at_load, rows)
        # 压力测试优化（Round 3）：写路径只标 dirty，读 snapshot/index 时才
        # 惰性 rebuild。原实现每次 save 全量重扫所有 topic + 重写 MEMORY.md，
        # n 条记忆批量写入是 O(n²)（500 条 2.45s，万条估算 100s+）。
        # 语义安全：记忆写入后本会话不注入（prompt cache 保护设计），
        # 下次会话构造时 build_index_text 会 ensure fresh。
        self._index_dirty = False
        # 启动时迁移老格式（每记忆一 .md → topic jsonl）
        self._migrate_legacy_if_any()
        self._rebuild_index()

    # ------------------------------------------------------------------
    # topic 文件读写
    # ------------------------------------------------------------------

    def _topic_path(self, topic: str) -> Path:
        """topic 文件路径（安全文件名）。"""
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", (topic or "general"))
        return self._memory_dir / f"{safe}.jsonl"

    def _topic_mtime(self, topic: str) -> tuple:
        """topic 文件的 (mtime, size) 双因子缓存键（不存在返回 (-1.0, -1)）。

        双因子防 mtime 精度窗口（Windows ~15ms）内写入误判缓存有效。
        """
        try:
            st = self._topic_path(topic).stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return (-1.0, -1)

    def _read_topic_rows(self, topic: str) -> List[dict]:
        """读 topic 全部行（内存缓存 + mtime 失效，miss 时读盘一次）。

        返回的是缓存 list 本身——调用方（锁内）对其的 mutation
        会同步到缓存，这是有意设计（save/update 就地改 rows 后写回）。
        外部实例改了文件（mtime 变）自动重读。
        """
        mtime = self._topic_mtime(topic)
        cached = self._rows_cache.get(topic)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        path = self._topic_path(topic)
        rows = []
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("topic 文件 %s 有损坏行，跳过", path)
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            except Exception as e:
                logger.warning("读 topic 文件失败 %s: %s", path, e)
        self._rows_cache[topic] = (mtime, rows)
        return rows

    def _write_topic_rows(self, topic: str, rows: List[dict]) -> None:
        """原子写 topic 文件（JSONL）+ 同步缓存（更新/删除路径用）。"""
        path = self._topic_path(topic)
        text = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in rows
        ) + ("\n" if rows else "")
        atomic_write_text(path, text)
        self._rows_cache[topic] = (self._topic_mtime(topic), rows)

    def _append_topic_row(self, topic: str, row: dict) -> None:
        """新建路径：缓存 append + 文件 append（O(1)，免全量重写）。"""
        rows = self._read_topic_rows(topic)  # 确保缓存已加载
        rows.append(row)
        path = self._topic_path(topic)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("append topic 行失败 %s: %s", path, e)
        self._rows_cache[topic] = (self._topic_mtime(topic), rows)

    def _find_row(self, topic: str, uid: str) -> Optional[dict]:
        for r in self._read_topic_rows(topic):
            if r.get("id") == uid:
                return r
        return None

    def _row_to_entry(self, topic: str, row: dict) -> MemoryEntry:
        return MemoryEntry(
            id=f"{topic}#{row.get('id', '')}",
            name=row.get("name", ""),
            description=row.get("description", ""),
            type=row.get("type", "other"),
            body=row.get("body", ""),
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
            topic=topic,
            summary=row.get("summary", "") or "",
            confidence=float(row.get("confidence", 1.0) or 1.0),
            expected_valid_days=int(row.get("expected_valid_days", 365) or 365),
            source_session_id=row.get("source_session_id", "") or "",
            state=row.get("state", "active") or "active",
            last_reviewed_at=row.get("last_reviewed_at", "") or "",
        )

    def _scan_all_entries(self) -> List[MemoryEntry]:
        """扫描所有 topic 文件，解析全部记忆。"""
        entries = []
        for path in sorted(self._memory_dir.glob("*.jsonl")):
            topic = path.stem
            for row in self._read_topic_rows(topic):
                try:
                    entries.append(self._row_to_entry(topic, row))
                except (ValueError, TypeError) as e:
                    logger.warning("memory 行解析失败 %s: %s", path, e)
        return entries

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """重建 MEMORY.md：按主题分组。"""
        type_priority = {"feedback": 0, "user": 1, "project": 2, "reference": 3, "other": 4}
        entries = self._scan_all_entries()
        entries = [e for e in entries if e.state != "archived"]
        entries.sort(key=lambda e: str(e.updated_at), reverse=True)
        entries.sort(key=lambda e: e.confidence, reverse=True)
        entries.sort(key=lambda e: type_priority.get(e.type, 99))

        # 按 topic 分组（保持 topic 内排序）
        by_topic: Dict[str, list] = {}
        for e in entries:
            by_topic.setdefault(e.topic, []).append(e)

        lines = [
            "# Memory Index",
            "",
            "自动生成，请勿手动编辑。⭐ 表示 feedback 类(用户纠正过的),永远优先显示。",
            "",
        ]
        for topic in sorted(by_topic.keys()):
            lines.append(f"## 主题：{topic}")
            for e in by_topic[topic]:
                marker = "⭐ " if e.type == "feedback" else ""
                uid = e.id.split("#")[-1]
                if e.summary:
                    lines.append(
                        f"- {marker}[{e.name}](.memory/{topic}.jsonl#{uid}) — "
                        f"{e.description} | 摘要：{e.summary}"
                    )
                else:
                    lines.append(
                        f"- {marker}[{e.name}](.memory/{topic}.jsonl#{uid}) — {e.description}"
                    )
            lines.append("")

        atomic_write_text(self._index_path, "\n".join(lines) + "\n")
        # 缓存 snapshot(跳过前 4 行头)
        self._cached_snapshot = "\n".join(lines[4:]) if len(lines) > 4 else ""
        self._index_dirty = False

    def snapshot_for_prompt(self) -> str:
        """索引注入 system prompt（截断：200 行 / 25KB，先到者，对齐 Claude Code）。"""
        self._ensure_index_fresh()
        snap = self._cached_snapshot
        lines = snap.splitlines()
        if len(lines) > _INDEX_MAX_LINES:
            snap = "\n".join(lines[:_INDEX_MAX_LINES]) + (
                "\n... [记忆索引超出行数上限，其余按需检索]"
            )
        if len(snap) > _INDEX_MAX_BYTES:
            snap = snap[:_INDEX_MAX_BYTES] + "\n... [记忆索引超出字节上限，其余按需检索]"
        return snap

    def full_index_text(self) -> str:
        """完整记忆索引（供 memory_retriever 按需检索，不被注入截断影响）。"""
        self._ensure_index_fresh()
        return self._cached_snapshot

    def _mark_index_dirty(self) -> None:
        """写路径调用：标记索引待重建（不立即 rebuild，防批量写 O(n²)）。"""
        self._index_dirty = True

    def _ensure_index_fresh(self) -> None:
        """读路径调用：dirty 时才 rebuild（惰性）。线程安全（拿锁）。"""
        if self._index_dirty:
            with self._lock:
                if self._index_dirty:
                    self._rebuild_index()

    def build_index_text(self) -> str:
        """重建并返回索引（启动时用）。"""
        with self._lock:
            self._rebuild_index()
        return self.snapshot_for_prompt()

    # ------------------------------------------------------------------
    # 公开：读
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            row = self._find_row(topic, uid)
            return self._row_to_entry(topic, row) if row else None

    def load_body(self, memory_id: str) -> Optional[str]:
        entry = self.get(memory_id)
        return entry.body if entry else None

    def list_all(self) -> List[MemoryEntry]:
        with self._lock:
            return self._scan_all_entries()

    def find_by_topic_name(self, topic: str, name: str) -> Optional[MemoryEntry]:
        """同主题同 name 查重（写入即维护）。"""
        if not name:
            return None
        with self._lock:
            for row in self._read_topic_rows(topic or "general"):
                if row.get("name") == name and row.get("state", "active") != "archived":
                    return self._row_to_entry(topic or "general", row)
            return None

    # ------------------------------------------------------------------
    # 公开：写
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        name: str,
        description: str,
        type: str,
        body: str = "",
        summary: str = "",
        confidence: float = 1.0,
        expected_valid_days: int = 365,
        source_session_id: str = "",
        topic: str = "general",
    ) -> str:
        """创建或更新记忆（写入即维护：同 topic 同 name → 更新）。返回 entry id。"""
        if not name or not description:
            raise ValueError("name 和 description 必需")
        if type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一，实际: {type}")
        # 注意：save 内联查重/更新，避免嵌套持锁（threading.Lock 不可重入）
        with self._lock:
            rows = self._read_topic_rows(topic)
            existing = next(
                (r for r in rows
                 if r.get("name") == name and r.get("state", "active") != "archived"),
                None,
            )
            if existing is not None:
                # 写入即维护：同 topic 同 name → 更新而非新建
                existing.update({
                    "description": description, "type": type, "body": body,
                    "summary": summary, "confidence": confidence,
                    "expected_valid_days": expected_valid_days,
                    "source_session_id": source_session_id,
                    "updated_at": _now_iso(),
                })
                self._write_topic_rows(topic, rows)
                self._mark_index_dirty()
                return f"{topic}#{existing['id']}"

            now = datetime.now()
            uid = _generate_id()
            row = {
                "id": uid, "name": name, "description": description,
                "type": type, "body": body, "summary": summary,
                "created_at": now.isoformat(timespec="seconds"),
                "updated_at": now.isoformat(timespec="seconds"),
                "confidence": confidence,
                "expected_valid_days": expected_valid_days,
                "source_session_id": source_session_id,
                "state": "active",
            }
            self._append_topic_row(topic, row)
            self._mark_index_dirty()
            return f"{topic}#{uid}"

    def update(
        self,
        memory_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
        summary: Optional[str] = None,
        confidence: Optional[float] = None,
        expected_valid_days: Optional[int] = None,
        source_session_id: Optional[str] = None,
    ) -> MemoryEntry:
        """更新字段。不存在的 id 抛 KeyError。"""
        if type is not None and type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一")
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            rows = self._read_topic_rows(topic)
            target = None
            for r in rows:
                if r.get("id") == uid:
                    target = r
                    break
            if target is None:
                raise KeyError(f"memory not found: {memory_id}")
            if name is not None:
                target["name"] = name
            if description is not None:
                target["description"] = description
            if type is not None:
                target["type"] = type
            if body is not None:
                target["body"] = body
            if summary is not None:
                target["summary"] = summary
            if confidence is not None:
                target["confidence"] = confidence
            if expected_valid_days is not None:
                target["expected_valid_days"] = expected_valid_days
            if source_session_id is not None:
                target["source_session_id"] = source_session_id
            target["updated_at"] = _now_iso()
            self._write_topic_rows(topic, rows)
            self._mark_index_dirty()
            return self._row_to_entry(topic, target)

    def delete(self, memory_id: str) -> bool:
        """软删除：把条目移到 .archive/，并从 topic 文件移除。"""
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            rows = self._read_topic_rows(topic)
            target = None
            for r in rows:
                if r.get("id") == uid:
                    target = r
                    break
            if target is None:
                return False
            # 软删除：条目副本存档
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_dir = self._home / ".archive" / f"memory-{ts}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                (archive_dir / f"{topic}-{uid}.json").write_text(
                    json.dumps(target, ensure_ascii=False), encoding="utf-8",
                )
            except Exception as e:
                logger.warning("记忆软删除存档失败: %s", e)
            # 从 topic 文件移除
            rows = [r for r in rows if r.get("id") != uid]
            self._write_topic_rows(topic, rows)
            self._mark_index_dirty()
            return True

    def clear_all(self) -> int:
        """软删除全部记忆（topic 文件整体移到 .archive，可恢复）。返回删除数。"""
        with self._lock:
            topics = sorted(p.stem for p in self._memory_dir.glob("*.jsonl"))
            total = 0
            for topic in topics:
                rows = self._read_topic_rows(topic)
                total += len(rows)
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_dir = self._home / ".archive" / f"memory-{ts}"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    src = self._topic_path(topic)
                    dst = archive_dir / f"{topic}.jsonl"
                    # 避免覆盖已存在的归档
                    n = 1
                    while dst.exists():
                        dst = archive_dir / f"{topic}-{n}.jsonl"
                        n += 1
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    logger.warning("清空 topic %s 失败: %s", topic, e)
                    continue
            self._mark_index_dirty()
            return total

    # ------------------------------------------------------------------
    # 旧格式迁移（每记忆一 .md → topic jsonl）
    # ------------------------------------------------------------------

    def _migrate_legacy_if_any(self) -> None:
        """把旧 `.memory/{id}.md`（frontmatter 单块）迁移到 topic jsonl。"""
        legacy_mds = [p for p in self._memory_dir.glob("*.md") if p.name != "latest.md"]
        if not legacy_mds:
            return
        logger.info("发现 %d 个旧格式记忆文件，迁移到主题组织", len(legacy_mds))
        for path in legacy_mds:
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text)
                if meta is None:
                    continue
                topic = meta.get("topic") or _infer_topic(meta.get("name", ""), meta.get("type", "other"))
                row = {
                    "id": path.stem, "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "type": meta.get("type", "other"),
                    "body": body,
                    "created_at": meta.get("created_at", _now_iso()),
                    "updated_at": meta.get("updated_at", _now_iso()),
                    "summary": meta.get("summary", "") or "",
                    "confidence": float(meta.get("confidence", 1.0) or 1.0),
                    "expected_valid_days": int(meta.get("expected_valid_days", 365) or 365),
                    "source_session_id": meta.get("source_session_id", "") or "",
                    "state": meta.get("state", "active") or "active",
                }
                rows = self._read_topic_rows(topic)
                rows.append(row)
                self._write_topic_rows(topic, rows)
                # 归档旧文件
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_dir = self._home / ".archive" / f"legacy-memory-{ts}"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), archive_dir / path.name)
                except Exception as e:
                    logger.warning("归档旧记忆 %s 失败: %s", path, e)
            except Exception as e:
                logger.warning("迁移旧记忆 %s 失败: %s", path, e)

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------

    def format_for_system_prompt(self, target: str) -> str:
        """旧接口兼容：返回索引（忽略 target 参数）。"""
        return self.snapshot_for_prompt()

    def add(self, target: str, content: str) -> bool:
        """旧接口兼容：等价于 save（target 当 type 用）。"""
        try:
            t = target if target in VALID_TYPES else "other"
            self.save(name=content[:30], description=content, type=t, body=content)
            return True
        except Exception as e:
            logger.warning("旧 add() 兼容失败: %s", e)
            return False

    def modify(self, action: str, target: str, content: str, old_content: str = "") -> bool:
        """旧接口兼容：粗略映射到 save。"""
        if action == "add":
            return self.add(target, content)
        logger.warning("旧 modify(action=%s) 不再支持，请用 memory 工具新 action", action)
        return False


def _parse_dt(value) -> datetime:
    """解析 ISO 时间，失败返回 now。"""
    try:
        return datetime.fromisoformat(str(value or _now_iso()))
    except (ValueError, TypeError):
        return datetime.now()


def _infer_topic(name: str, type: str) -> str:
    """从 name/type 推断主题（旧格式迁移用，兜底 general）。"""
    if type == "feedback":
        return "feedback"
    if type == "project":
        return "project"
    first = (name or "").strip().split()[0] if (name or "").strip() else ""
    return first[:12] if first else "general"
