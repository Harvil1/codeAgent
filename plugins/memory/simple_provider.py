"""最简单的记忆 provider：把对话存到 JSON 文件，支持关键词搜索。

作为 MemoryProvider ABC 的参考实现。
生产场景可换成 Honcho/Mem0/Supermemory 等。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agent.memory_provider import MemoryProvider


class SimpleProvider(MemoryProvider):
    """简单的文件记忆 provider。"""

    def __init__(self, harvil_home: Path):
        self._home = Path(harvil_home)
        self._store_path = self._home / "turns.json"
        self._turns: List[Dict] = []
        self._session_id = ""

    @property
    def name(self) -> str:
        return "simple"

    def is_available(self) -> bool:
        return True  # 总是可用

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        if self._store_path.exists():
            try:
                self._turns = json.loads(self._store_path.read_text(encoding="utf-8"))
            except Exception:
                self._turns = []

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:
        """异步写入一轮对话。"""
        self._turns.append({
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "user": user_content,
            "assistant": assistant_content,
        })
        # 只保留最近 1000 轮
        self._turns = self._turns[-1000:]
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps(self._turns, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def prefetch(self, query: str, **kwargs) -> str:
        """简单的关键词搜索：返回包含查询词的历史轮次。"""
        if not query:
            return ""
        keywords = query.lower().split()
        matches = []
        for turn in self._turns[-50:]:  # 只搜最近 50 轮
            text = (turn.get("user", "") + " " + turn.get("assistant", "")).lower()
            if any(kw in text for kw in keywords):
                matches.append(turn)
        if not matches:
            return ""
        # 格式化为上下文文本
        lines = ["# 相关历史记忆（来自 SimpleProvider）"]
        for m in matches[-3:]:  # 最多 3 条
            lines.append(f"- {m.get('user', '')[:100]}")
        return "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict]:
        return []  # 不暴露额外工具

    def shutdown(self) -> None:
        pass  # 每轮都落盘了，无需额外清理
