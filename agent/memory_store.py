"""内置文件记忆系统。

设计原则：
1. 声明式事实，不是指令
   ✅ "用户偏好简洁回复"
   ❌ "必须简洁回复"
2. 字数限制（防止膨胀）
   - MEMORY.md: ~2200 字符（环境事实）
   - USER.md: ~1375 字符（用户画像）
3. Frozen snapshot
   - 会话开始时加载一次
   - 写入立即落盘，但下次会话才注入
   - 保护 prompt cache
"""

import logging
import threading
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375


class MemoryStore:
    """文件记忆存储。每个 agent 实例一份。"""

    def __init__(
        self,
        harvil_home: Path,
        memory_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
        user_char_limit: int = DEFAULT_USER_CHAR_LIMIT,
    ):
        self._home = Path(harvil_home)
        self._memory_path = self._home / "MEMORY.md"
        self._user_path = self._home / "USER.md"

        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit

        # 内存中的条目列表（frozen snapshot 来自启动时加载）
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []

        self._lock = threading.Lock()

        # 启动时从磁盘加载（之后会话内不再 reload，保护 prompt cache）
        self.load_from_disk()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def load_from_disk(self) -> None:
        """从磁盘文件加载记忆。"""
        with self._lock:
            self.memory_entries = self._read_file(self._memory_path)
            self.user_entries = self._read_file(self._user_path)

    def _read_file(self, path: Path) -> List[str]:
        """读取记忆文件，返回条目列表。"""
        if not path.exists():
            return []
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("读取记忆文件失败 %s: %s", path, e)
            return []

        # 每行一条记忆（以 "- " 开头的 bullet）
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- "):
                entries.append(line[2:])
            elif line and not line.startswith("#"):
                # 容忍没有 bullet 的行
                entries.append(line)
        return entries

    def _write_file(self, path: Path, entries: List[str], header: str) -> None:
        """写入记忆文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {header}", ""]
        for entry in entries:
            lines.append(f"- {entry}")
        # 关键：必须指定 encoding（Windows 默认 cp1252 会乱码）
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # CRUD（由 memory 工具调用）
    # ------------------------------------------------------------------

    def modify(
        self,
        action: str,
        target: str,
        content: str,
        old_content: str = "",
    ) -> bool:
        """统一的修改入口。

        action: add / replace / remove
        target: memory / user
        """
        if action == "add":
            return self.add(target, content)
        elif action == "replace":
            return self.replace(target, old_content or content, content)
        elif action == "remove":
            return self.remove(target, content)
        logger.warning("未知记忆操作: %s", action)
        return False

    def add(self, target: str, content: str) -> bool:
        """添加一条记忆。返回是否成功。"""
        if not content or not content.strip():
            return False

        with self._lock:
            if target == "memory":
                if self._total_chars("memory") + len(content) > self.memory_char_limit:
                    logger.warning("MEMORY.md 超出字数限制，拒绝写入")
                    return False
                if content in self.memory_entries:
                    return True  # 幂等
                self.memory_entries.append(content)
                self._write_file(self._memory_path, self.memory_entries, "Agent Memory")
            elif target == "user":
                if self._total_chars("user") + len(content) > self.user_char_limit:
                    logger.warning("USER.md 超出字数限制，拒绝写入")
                    return False
                if content in self.user_entries:
                    return True
                self.user_entries.append(content)
                self._write_file(self._user_path, self.user_entries, "User Profile")
            else:
                return False
        return True

    def replace(self, target: str, old_content: str, new_content: str) -> bool:
        """替换一条记忆（按子串匹配）。"""
        if not new_content.strip():
            return False
        with self._lock:
            entries = self.memory_entries if target == "memory" else self.user_entries
            for i, e in enumerate(entries):
                if old_content and old_content in e:
                    entries[i] = new_content
                    self._persist(target, entries)
                    return True
        return False

    def remove(self, target: str, content: str) -> bool:
        """删除一条记忆（按子串匹配）。"""
        with self._lock:
            entries = self.memory_entries if target == "memory" else self.user_entries
            for i, e in enumerate(entries):
                if content and content in e:
                    entries.pop(i)
                    self._persist(target, entries)
                    return True
        return False

    def _persist(self, target: str, entries: List[str]) -> None:
        """落盘（调用方已持锁）。"""
        if target == "memory":
            self._write_file(self._memory_path, entries, "Agent Memory")
        else:
            self._write_file(self._user_path, entries, "User Profile")

    def _total_chars(self, target: str) -> int:
        entries = self.memory_entries if target == "memory" else self.user_entries
        return sum(len(e) for e in entries)

    # ------------------------------------------------------------------
    # Prompt 注入
    # ------------------------------------------------------------------

    def format_for_system_prompt(self, target: str) -> str:
        """格式化为 system prompt 的一段。"""
        entries = self.memory_entries if target == "memory" else self.user_entries
        if not entries:
            return ""

        header = "# 记忆（agent 笔记）" if target == "memory" else "# 用户画像"
        lines = [header]
        for entry in entries:
            lines.append(f"- {entry}")
        return "\n".join(lines)

    def snapshot_for_prompt(self) -> str:
        """返回合并的 memory + user 快照，供 prompt_builder 注入。

        这是 frozen snapshot：会话开始时调用一次，之后会话内不再调用。
        """
        parts = []
        mem = self.format_for_system_prompt("memory")
        usr = self.format_for_system_prompt("user")
        if mem:
            parts.append(mem)
        if usr:
            parts.append(usr)
        return "\n\n".join(parts)
