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
        content = _format_frontmatter(meta) + entry.body
        path = self._memory_dir / f"{entry.id}.md"
        path.write_text(content, encoding="utf-8")

    def _rebuild_index(self) -> None:
        """扫描 .memory/ 重建 MEMORY.md。"""
        lines = ["# Memory Index", ""]
        lines.append("自动生成，请勿手动编辑。每行：`- [name](.memory/{id}.md) — description`")
        lines.append("")
        for entry in self._scan_all_entries():
            lines.append(
                f"- [{entry.name}](.memory/{entry.id}.md) — {entry.description}"
            )
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
                    created_at=datetime.fromisoformat(meta.get("created_at", _now_iso())),
                    updated_at=datetime.fromisoformat(meta.get("updated_at", _now_iso())),
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
            created_at=datetime.fromisoformat(meta.get("created_at", _now_iso())),
            updated_at=datetime.fromisoformat(meta.get("updated_at", _now_iso())),
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
        """返回索引文本（frozen，会话内不变）。"""
        # 读当前 MEMORY.md 内容（仅索引行，跳过头部 3 行）
        if not self._index_path.exists():
            return ""
        text = self._index_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # 跳过前 3 行（标题 + 空 + 说明）和第 4 行空行
        return "\n".join(lines[4:]) if len(lines) > 4 else ""

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
                type=type, body=body,
                created_at=now, updated_at=now,
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
