"""多文件记忆存储：Claude Code 风格。

存储结构：
- ~/.agent/.memory/{ulid}.md：单条记忆，YAML frontmatter + body
- ~/.agent/MEMORY.md：索引（自动生成，每次写后重建）

原则：
- 写入立即落盘 + 重建索引
- snapshot_for_prompt() 返回索引文本，会话内 frozen
- 老格式（无 frontmatter）启动时备份到 .archive/legacy-memory-{ts}/
- 删除软删除到 .archive/memory-{ts}/{id}.md
"""
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from agent.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)

VALID_TYPES = {"user", "feedback", "project", "reference", "other"}


@dataclass
class MemoryEntry:
    """单条记忆。"""
    id: str
    name: str
    description: str
    type: str
    body: str
    created_at: datetime
    updated_at: datetime
    # CCALS-P0-1: L1 摘要层（80-100 字符，比 description 更详细）
    # 老文件无此字段时兼容读为空串
    summary: str = ""
    # confidence + expected_valid_days(借鉴 DeerFlow DeerMem):
    # confidence: LLM 给的信心分 0.0-1.0,低于阈值的候选不写入
    # expected_valid_days: 预期有效期(天),到期后进 staleness 评审
    # 老文件无这俩字段时兼容读为默认值(1.0 / 365 天)
    confidence: float = 1.0
    expected_valid_days: int = 365
    # 来源追溯:这条记忆来自哪次对话(便于回溯原始上下文)
    source_session_id: str = ""
    # curator 用:active/stale/archived 状态 + 上次评估时间
    state: str = "active"
    last_reviewed_at: str = ""


def _generate_id() -> str:
    """生成时间排序的唯一 ID（不用 ulid 库，简化为 uuid 拼时间戳）。"""
    ts = int(datetime.now().timestamp() * 1000)
    short_uuid = uuid.uuid4().hex[:6]
    return f"{ts}{short_uuid}"  # 如 "1720870000000a1b2c3"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """解析 `---\\n...yaml...\\n---\\nbody` 格式。返回 (meta, body) 或 (None, text)。"""
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
    """格式化 dict 为 frontmatter 字符串。"""
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n"


class MemoryStore:
    """多文件记忆存储。"""

    def __init__(self, *, harvil_home: Path):
        self._home = Path(harvil_home)
        self._memory_dir = self._home / ".memory"
        self._index_path = self._home / "MEMORY.md"
        self._lock = threading.Lock()
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._cached_snapshot: str = ""  # P1-5: snapshot_for_prompt 缓存
        # 启动时迁移老格式（若存在）
        self._migrate_legacy_if_any()
        # 重建索引（保证一致）
        self._rebuild_index()

    # ---- 内部：写 ----
    def _write_entry_file(self, entry: MemoryEntry) -> None:
        """写单条 .md 文件。"""
        meta = {
            "name": entry.name,
            "description": entry.description,
            "type": entry.type,
            "created_at": entry.created_at.isoformat(timespec="seconds"),
            "updated_at": entry.updated_at.isoformat(timespec="seconds"),
        }
        # CCALS-P0-1: summary 非空时才写入 frontmatter（避免老格式文件多出空字段）
        if entry.summary:
            meta["summary"] = entry.summary
        # confidence / expected_valid_days:非默认值才写(老文件兼容)
        if entry.confidence != 1.0:
            meta["confidence"] = entry.confidence
        if entry.expected_valid_days != 365:
            meta["expected_valid_days"] = entry.expected_valid_days
        if entry.source_session_id:
            meta["source_session_id"] = entry.source_session_id
        # curator 状态字段(state=active 不写,保持老文件干净)
        if entry.state != "active":
            meta["state"] = entry.state
        if entry.last_reviewed_at:
            meta["last_reviewed_at"] = entry.last_reviewed_at
        content = _format_frontmatter(meta) + entry.body
        path = self._memory_dir / f"{entry.id}.md"
        atomic_write_text(path, content)

    def _rebuild_index(self) -> None:
        """扫描 .memory/ 重建 MEMORY.md。

        CCALS-P0-1: 索引行包含 summary（若有），让 retriever 拿到的
        index 自动含 L1 摘要层，无需读全文就能判断更细的相关性。

        correction 优先注入(借鉴 DeerFlow guaranteed_categories):
        - feedback 类型(用户纠正过的)排最前面 + 加 ⭐ 标记
        - 让 LLM 看到 system prompt 时优先注意到纠正类记忆
        - 防止"用户说不要用 pip,但 agent 又用 pip"这种重复纠正
        """
        # type 优先级:feedback(纠正) 最优先,然后 user/project,最后 other
        type_priority = {"feedback": 0, "user": 1, "project": 2, "reference": 3, "other": 4}
        # 三步稳定排序(从最细粒度到最粗粒度,利用 Python sorted 稳定性):
        # 1. updated_at 倒序(新的在前)
        # 2. confidence 倒序(高在前)
        # 3. type_priority 升序(feedback=0 最前)
        # 最终顺序:feedback 优先 → 同 type 内 confidence 高的 → 同 confidence 内最新的
        entries = self._scan_all_entries()
        # curator:archived 不进索引(不出现在 system prompt)
        entries = [e for e in entries if e.state != "archived"]
        entries.sort(key=lambda e: str(e.updated_at), reverse=True)
        entries.sort(key=lambda e: e.confidence, reverse=True)
        entries.sort(key=lambda e: type_priority.get(e.type, 99))

        lines = [
            "# Memory Index",
            "",
            "自动生成，请勿手动编辑。⭐ 表示 feedback 类(用户纠正过的),永远优先显示。",
            "",
        ]
        for entry in entries:
            marker = "⭐ " if entry.type == "feedback" else ""
            if entry.summary:
                lines.append(
                    f"- {marker}[{entry.name}](.memory/{entry.id}.md) — {entry.description}"
                    f" | 摘要：{entry.summary}"
                )
            else:
                lines.append(
                    f"- {marker}[{entry.name}](.memory/{entry.id}.md) — {entry.description}"
                )
        atomic_write_text(self._index_path, "\n".join(lines) + "\n")
        # P1-5: 缓存 snapshot(跳过前 4 行头)
        self._cached_snapshot = "\n".join(lines[4:]) if len(lines) > 4 else ""

    def _scan_all_entries(self) -> List[MemoryEntry]:
        """扫描 .memory/ 下所有 .md，解析为 MemoryEntry。失败的跳过。"""
        entries = []
        for path in sorted(self._memory_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            if meta is None:
                logger.warning("跳过无 frontmatter 的 memory 文件: %s", path)
                continue
            try:
                entry = MemoryEntry(
                    id=path.stem,
                    name=meta.get("name", ""),
                    description=meta.get("description", ""),
                    type=meta.get("type", "other"),
                    body=body,
                    created_at=datetime.fromisoformat(str(meta.get("created_at", _now_iso()))),
                    updated_at=datetime.fromisoformat(str(meta.get("updated_at", _now_iso()))),
                    summary=meta.get("summary", "") or "",  # CCALS-P0-1: 兼容老文件
                    confidence=float(meta.get("confidence", 1.0) or 1.0),
                    expected_valid_days=int(meta.get("expected_valid_days", 365) or 365),
                    source_session_id=meta.get("source_session_id", "") or "",
                    state=meta.get("state", "active") or "active",
                    last_reviewed_at=meta.get("last_reviewed_at", "") or "",
                )
                entries.append(entry)
            except (ValueError, TypeError) as e:
                logger.warning("memory 文件字段解析失败 %s: %s", path, e)
        return entries

    def _migrate_legacy_if_any(self) -> None:
        """检测旧格式 MEMORY.md / USER.md，备份到 .archive/legacy-memory-{ts}/。

        判断标准：文件存在 + 内容不以 `---` 开头（无 frontmatter）。
        新建的多文件 MEMORY.md 索引以 `# Memory Index` 开头，不会被误判。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_any = False
        archive_dir = self._home / ".archive" / f"legacy-memory-{ts}"
        for filename in ("MEMORY.md", "USER.md"):
            path = self._home / filename
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            if content.startswith("---") or content.startswith("# Memory Index"):
                continue  # 新格式，不动
            # 老格式 → 备份
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive_dir / filename)
            logger.info("旧 %s 已备份到 %s/", filename, archive_dir)
            path.unlink()
            archived_any = True
        if archived_any:
            logger.info("老格式记忆文件已迁移。新格式 MEMORY.md 索引将在下一步重建。")

    # ---- 公开：读 ----
    def _get_unlocked(self, memory_id: str) -> Optional[MemoryEntry]:
        """读取单条记忆（调用方已持锁或无需锁）。"""
        path = self._memory_dir / f"{memory_id}.md"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        if meta is None:
            return None
        return MemoryEntry(
            id=memory_id,
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            type=meta.get("type", "other"),
            body=body,
            created_at=datetime.fromisoformat(str(meta.get("created_at", _now_iso()))),
            updated_at=datetime.fromisoformat(str(meta.get("updated_at", _now_iso()))),
            summary=meta.get("summary", "") or "",  # CCALS-P0-1
            state=meta.get("state", "active") or "active",
            last_reviewed_at=meta.get("last_reviewed_at", "") or "",
        )

    def list_all(self) -> List[MemoryEntry]:
        with self._lock:
            return self._scan_all_entries()

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            return self._get_unlocked(memory_id)

    def load_body(self, memory_id: str) -> Optional[str]:
        entry = self.get(memory_id)
        return entry.body if entry else None

    def snapshot_for_prompt(self) -> str:
        """返回索引文本(frozen,会话内不变)。

        P1-5: 从内存缓存返回(_rebuild_index 时更新),避免每次 read_text + splitlines。
        """
        return self._cached_snapshot

    def build_index_text(self) -> str:
        """重建并返回索引（启动时用）。"""
        with self._lock:
            self._rebuild_index()
        return self.snapshot_for_prompt()

    # ---- 公开：写 ----
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
    ) -> str:
        """创建新记忆。返回 memory_id。"""
        if not name or not description:
            raise ValueError("name 和 description 必需")
        if type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一，实际: {type}")
        with self._lock:
            now = datetime.now()
            mid = _generate_id()
            entry = MemoryEntry(
                id=mid, name=name, description=description,
                type=type, body=body, summary=summary,
                created_at=now, updated_at=now,
                confidence=float(confidence),
                expected_valid_days=int(expected_valid_days),
                source_session_id=source_session_id,
            )
            self._write_entry_file(entry)
            self._rebuild_index()
        return mid

    def update(
        self,
        memory_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
        summary: Optional[str] = None,  # CCALS-P0-1
    ) -> MemoryEntry:
        """更新字段。不存在的 id 抛 KeyError。"""
        if type is not None and type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一")
        with self._lock:
            entry = self._get_unlocked(memory_id)
            if entry is None:
                raise KeyError(f"memory not found: {memory_id}")
            if name is not None:
                entry.name = name
            if description is not None:
                entry.description = description
            if type is not None:
                entry.type = type
            if body is not None:
                entry.body = body
            if summary is not None:
                entry.summary = summary
            entry.updated_at = datetime.now()
            self._write_entry_file(entry)
            self._rebuild_index()
        return entry

    def delete(self, memory_id: str) -> bool:
        """软删除：移到 .archive/memory-{ts}/{id}.md。"""
        with self._lock:
            path = self._memory_dir / f"{memory_id}.md"
            if not path.exists():
                return False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self._home / ".archive" / f"memory-{ts}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive_dir / path.name))
            self._rebuild_index()
        return True

    # ---- 兼容旧接口（被 prompt_builder 等调用，留 stub 避免破坏） ----
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
        """旧接口兼容：粗略映射到 save（行为不完全等价）。"""
        if action == "add":
            return self.add(target, content)
        # replace / remove 在旧接口下行为模糊，旧数据已弃用，直接返 False 提示用户用新工具
        logger.warning("旧 modify(action=%s) 不再支持，请用 memory 工具新 action", action)
        return False
