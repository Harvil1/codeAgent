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


VALID_STATUSES = {"pending", "in_progress", "completed", "deleted", "blocked", "triage"}

# 06 NEW: 同 kind 阻塞达到阈值时升级到 triage（避免死循环重试）
TRIAGE_THRESHOLD = 3
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}


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

    def __init__(self, harvil_home=None, *, sqlite_store=None):
        self._dir = _tasks_dir(harvil_home)
        # 可选 SQLite 双写
        self._sqlite = sqlite_store

    def attach_sqlite(self, sqlite_store) -> None:
        """运行时注入 SQLite store（用于 cli.py 启动顺序解耦）。"""
        self._sqlite = sqlite_store

    def _task_file(self, task_id: str) -> Path:
        return self._dir / f"{task_id}.json"

    def _write(self, task_id: str, task: dict) -> None:
        from agent.atomic_io import atomic_write_text
        f = self._task_file(task_id)
        atomic_write_text(f, json.dumps(task, ensure_ascii=False, indent=2))
        # SQLite 双写（失败不阻塞主流程）
        if self._sqlite is not None:
            try:
                self._sqlite.save_task(
                    id=task_id,
                    subject=task.get("subject", ""),
                    description=task.get("description", "") or "",
                    status=task.get("status", "pending"),
                    owner=task.get("owner"),
                    created_at=task.get("created_at", _now_iso()),
                    updated_at=task.get("updated_at", _now_iso()),
                    blocked_by=task.get("blocked_by", []),
                    body_json=json.dumps(task, ensure_ascii=False),
                )
            except Exception as e:
                logger.warning("task SQLite 双写失败 %s: %s", task_id, e)

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
            "last_heartbeat_at": None,
            "comments": [],
            "artifacts": [],
            "block_reason": None,
            "block_kind": None,
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

    # ------------------------------------------------------------------
    # 06 NEW: 阻塞升级（mark_blocked + block_history）
    # ------------------------------------------------------------------

    def mark_blocked(
        self,
        task_id: str,
        *,
        kind: str = "transient",
        reason: str = "",
    ) -> dict:
        """记录阻塞，自动升级重复阻塞到 triage。

        同 kind 阻塞 >= TRIAGE_THRESHOLD 次时状态升级到 triage，
        让 find_ready 不再选它，等人工或上级介入。

        返回 {"status": "blocked"|"triage", "block_count": int, "kind": str}
        """
        if kind not in VALID_BLOCK_KINDS:
            kind = "transient"

        task = self.get(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")

        history = task.setdefault("block_history", [])
        counts = task.setdefault("block_count_by_kind", {})
        counts[kind] = counts.get(kind, 0) + 1
        history.append({
            "kind": kind,
            "reason": reason,
            "at": _now_iso(),
        })
        # 限制历史长度（防无限增长，滚动覆盖）
        if len(history) > 20:
            del history[: len(history) - 20]

        # 升级判定
        new_status = "blocked"
        triage_reason: Optional[str] = None
        if counts[kind] >= TRIAGE_THRESHOLD:
            new_status = "triage"
            triage_reason = (
                f"task 阻塞超过 {TRIAGE_THRESHOLD} 次（kind={kind}），"
                f"已升级到 triage，等待人工或上级介入"
            )

        task["status"] = new_status
        task["block_kind"] = kind
        task["block_reason"] = triage_reason or reason
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        logger.info(
            "task %s 标记阻塞 kind=%s count=%d status=%s",
            task_id, kind, counts[kind], new_status,
        )
        return {
            "status": new_status,
            "block_count": counts[kind],
            "kind": kind,
        }

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
        """找出所有 pending 且依赖已满足的任务（可认领）。

        排除 triage 状态（已升级等待人工介入）。
        """
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

    # ------------------------------------------------------------------
    # Kanban 增强（heartbeat / comments / artifacts）
    # ------------------------------------------------------------------

    def heartbeat(self, task_id: str) -> Optional[dict]:
        """更新 last_heartbeat_at 为当前时间。"""
        return self.update(task_id, last_heartbeat_at=_now_iso())

    def add_comment(
        self, task_id: str, *, author: str, content: str,
    ) -> Optional[dict]:
        """追加一条 comment。comments 只增不删。"""
        task = self.get(task_id)
        if task is None:
            return None
        task.setdefault("comments", []).append({
            "author": author,
            "content": content,
            "created_at": _now_iso(),
        })
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def add_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """把 paths 去重追加到 artifacts。"""
        task = self.get(task_id)
        if task is None:
            return None
        existing = task.setdefault("artifacts", [])
        for p in paths:
            if p not in existing:
                existing.append(p)
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def remove_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """从 artifacts 移除 paths。"""
        task = self.get(task_id)
        if task is None:
            return None
        task["artifacts"] = [
            p for p in task.get("artifacts", []) if p not in paths
        ]
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    # ------------------------------------------------------------------
    # DAG 增强（cycle-safe dependency management）
    # ------------------------------------------------------------------

    def has_path(self, start_id: str, target_id: str) -> bool:
        """DFS：从 start_id 沿 blocked_by 边走，能否到达 target_id？

        blocked_by 语义：A.blocked_by=[B] 表示 A 依赖 B。
        所以"沿 blocked_by 走"= "查 start 依赖谁、间接依赖谁"。
        """
        visited = set()
        stack = [start_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_task = self.get(cur)
            if cur_task is None:
                continue
            for dep in cur_task.get("blocked_by", []):
                if dep == target_id:
                    return True
                stack.append(dep)
        return False

    def add_dependency(
        self, child_id: str, parent_id: str,
        *, validate: bool = True,
    ) -> Optional[dict]:
        """加 child 依赖 parent 的边（child.blocked_by += [parent]）。

        validate=True 时做 cycle 检测：若 parent 已经（直接或间接）依赖 child，拒绝。
        self-link 永远拒绝（即使 validate=False）。
        """
        if parent_id == child_id:
            raise ValueError("self-link forbidden")
        child = self.get(child_id)
        if child is None:
            return None
        if validate:
            if self.has_path(parent_id, child_id):
                raise ValueError(
                    f"cycle detected: {parent_id} 已经依赖 {child_id}，"
                    f"再加 {child_id} → {parent_id} 边会成环"
                )
        blocked_by = child.setdefault("blocked_by", [])
        if parent_id not in blocked_by:
            blocked_by.append(parent_id)
        child["updated_at"] = _now_iso()
        self._write(child_id, child)
        return child


# 全局单例
_task_store: Optional[TaskStore] = None


def get_task_store(harvil_home=None) -> TaskStore:
    global _task_store
    if _task_store is None or harvil_home is not None:
        _task_store = TaskStore(harvil_home)
    return _task_store
