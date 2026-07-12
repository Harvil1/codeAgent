"""持久化任务图（Task System）。

和 TodoWrite 的区别：
  - TodoWrite：内存清单，单会话，扁平，LLM 维护
  - Task System：持久化 .tasks/{id}.json，跨会话，DAG 依赖

支持：
  - 依赖追踪（blocked_by）
  - 状态机（pending → in_progress → completed）
  - 所有权认领（owner）
  - 自动解锁检查（can_start）
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


VALID_STATUSES = {"pending", "in_progress", "completed", "deleted"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tasks_dir(harvil_home=None) -> Path:
    if harvil_home:
        home = Path(harvil_home)
    else:
        try:
            from constants import get_agent_home
            home = get_agent_home()
        except Exception:
            home = Path.home() / ".agent"
    d = home / ".tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TaskStore:
    """持久化任务图。每个任务一个 JSON 文件。"""

    def __init__(self, harvil_home=None):
        self._dir = _tasks_dir(harvil_home)

    def _task_file(self, task_id: str) -> Path:
        return self._dir / f"{task_id}.json"

    def _write(self, task_id: str, task: dict) -> None:
        f = self._task_file(task_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps(task, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> dict:
        """创建任务。"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": owner,
            "blocked_by": list(blocked_by or []),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._write(task_id, task)
        logger.info("创建任务 %s: %s", task_id, subject)
        return task

    def get(self, task_id: str) -> Optional[dict]:
        """获取单个任务。"""
        f = self._task_file(task_id)
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    def update(self, task_id: str, **fields) -> Optional[dict]:
        """更新任务字段（除 id 外）。"""
        task = self.get(task_id)
        if task is None:
            return None
        for k, v in fields.items():
            if k != "id":
                task[k] = v
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def set_status(self, task_id: str, status: str) -> Optional[dict]:
        if status not in VALID_STATUSES:
            return None
        return self.update(task_id, status=status)

    def claim(self, task_id: str, owner: str) -> Optional[dict]:
        """认领任务（设置 owner 并置为 in_progress）。"""
        return self.update(task_id, owner=owner, status="in_progress")

    def complete(self, task_id: str) -> Optional[dict]:
        """完成任务。"""
        return self.set_status(task_id, "completed")

    def delete(self, task_id: str) -> Optional[dict]:
        """软删除任务（标 deleted，不真删文件）。"""
        return self.set_status(task_id, "deleted")

    def list_all(self, status: Optional[str] = None) -> List[dict]:
        """列出所有任务（可选按状态过滤）。"""
        tasks = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
                if status is None or t.get("status") == status:
                    tasks.append(t)
            except Exception:
                continue
        tasks.sort(key=lambda t: t.get("created_at", ""))
        return tasks

    # ------------------------------------------------------------------
    # 依赖
    # ------------------------------------------------------------------

    def can_start(self, task_id: str) -> bool:
        """检查任务的所有依赖是否已完成。"""
        task = self.get(task_id)
        if task is None:
            return False
        for dep_id in task.get("blocked_by", []):
            dep = self.get(dep_id)
            if dep is None or dep.get("status") != "completed":
                return False
        return True

    def find_ready(self) -> List[dict]:
        """找出所有 pending 且依赖已满足的任务（可认领）。"""
        ready = []
        for t in self.list_all(status="pending"):
            if self.can_start(t["id"]):
                ready.append(t)
        return ready

    def find_blocked(self) -> List[dict]:
        """找出依赖未满足的 pending 任务。"""
        blocked = []
        for t in self.list_all(status="pending"):
            if not self.can_start(t["id"]):
                blocked.append(t)
        return blocked


# 全局单例
_task_store: Optional[TaskStore] = None


def get_task_store(harvil_home=None) -> TaskStore:
    global _task_store
    if _task_store is None or harvil_home is not None:
        _task_store = TaskStore(harvil_home)
    return _task_store
