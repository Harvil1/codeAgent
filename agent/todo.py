"""TodoManager：内存中的任务清单状态机。

解决的问题：多步任务中 LLM 会"漂移"——重复步骤、跳过步骤、忘记目标。
对话越长越严重。TodoWrite 让 LLM 显式追踪进度。

约束：
  - 同时只能 1 个 in_progress（强制顺序聚焦）
  - 3 轮没更新时触发 reminder（系统注入提醒）
  - 仅存内存，单会话内有效（不持久化，跨会话用 Task System）

LLM 主动调 todo_write 工具更新清单。
系统（AIAgent 主循环）调用 increment_round + should_remind 决定是否提醒。
"""

import threading
from dataclasses import dataclass
from typing import List, Optional


VALID_STATUSES = {"pending", "in_progress", "completed"}


@dataclass
class TodoItem:
    id: str
    text: str
    status: str  # pending / in_progress / completed


class TodoManager:
    """任务清单状态机（内存，单会话）。"""

    def __init__(self):
        self._todos: List[TodoItem] = []
        self._rounds_since_update: int = 0
        self._lock = threading.Lock()

    def write(self, items: List[dict]) -> dict:
        """替换整个 todo 列表。

        校验：同时只能 1 个 in_progress。
        成功返回 {success, count}，失败返回 {success: False, error}。
        """
        with self._lock:
            new_todos: List[TodoItem] = []
            in_progress_count = 0
            invalid_items = []

            for i, item in enumerate(items):
                status = item.get("status", "pending")
                if status not in VALID_STATUSES:
                    invalid_items.append(f"#{i}: 非法 status '{status}'")
                    continue
                if status == "in_progress":
                    in_progress_count += 1
                new_todos.append(TodoItem(
                    id=str(item.get("id") or i),
                    text=item.get("text", ""),
                    status=status,
                ))

            if invalid_items:
                return {"success": False, "error": "; ".join(invalid_items)}

            if in_progress_count > 1:
                return {
                    "success": False,
                    "error": f"同时只能 1 个 in_progress（当前 {in_progress_count} 个）",
                }

            self._todos = new_todos
            self._rounds_since_update = 0  # 写入即重置计数
            return {
                "success": True,
                "count": len(self._todos),
                "completed": sum(1 for t in self._todos if t.status == "completed"),
                "remaining": sum(1 for t in self._todos if t.status != "completed"),
            }

    def increment_round(self) -> None:
        """每轮 LLM 调用后由主循环调用。"""
        with self._lock:
            self._rounds_since_update += 1

    def should_remind(self) -> bool:
        """是否该注入 reminder。"""
        with self._lock:
            return (
                self._rounds_since_update >= 3
                and any(t.status != "completed" for t in self._todos)
            )

    def format_for_reminder(self) -> str:
        """格式化为 reminder 文本（注入到 messages 末尾）。"""
        with self._lock:
            if not self._todos:
                return ""
            lines = [
                f"<todo_reminder>{self._rounds_since_update} 轮未更新任务清单。当前进度：",
            ]
            for t in self._todos:
                mark = {
                    "pending": "[ ]",
                    "in_progress": "[~]",
                    "completed": "[x]",
                }.get(t.status, "?")
                lines.append(f"  {mark} {t.text}")
            lines.append("如有进展，调用 todo_write 更新清单。</todo_reminder>")
            return "\n".join(lines)

    def format_for_display(self) -> str:
        """格式化为用户可读文本（/todo 命令用）。"""
        with self._lock:
            if not self._todos:
                return "(清单为空)"
            lines = []
            for t in self._todos:
                mark = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "✓",
                }.get(t.status, "?")
                lines.append(f"  {mark} {t.text}")
            return "\n".join(lines)

    def reset(self) -> None:
        """清空清单（新会话时）。"""
        with self._lock:
            self._todos = []
            self._rounds_since_update = 0

    @property
    def todos(self) -> List[TodoItem]:
        with self._lock:
            return list(self._todos)

    @property
    def rounds_since_update(self) -> int:
        with self._lock:
            return self._rounds_since_update


# 全局单例（每个 agent 实例应有自己的，这里简化为全局）
_todo_manager = TodoManager()


def get_todo_manager() -> TodoManager:
    return _todo_manager
