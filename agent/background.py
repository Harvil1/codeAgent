"""后台任务管理器：长命令异步执行 + 完成时通知。

设计目标：
- subprocess.Popen 跑命令（不共享 GIL）
- 每任务一个 daemon thread 读 stdout 并 wait
- 完成时 push 到 _notifications deque（lock 保护）
- 主循环每轮 drain_notifications 清空
- 默认非 detach（agent 退出清理）；detach=True 用 start_new_session/CREATE_NEW_PROCESS_GROUP
- 并发上限默认 5

跨平台：subprocess 必须 text=True, encoding="utf-8"（CLAUDE.md 强制）。
"""
import copy
import logging
import secrets
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    """后台任务状态。daemon thread 写，工具 handler 读，都用 manager.lock。"""
    task_id: str
    command: list
    cwd: Optional[Path]
    status: str  # "running" | "completed" | "failed" | "stopped"
    pid: Optional[int]
    started_at: datetime
    detach: bool = False
    ended_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)


class BackgroundManager:
    """单进程内的后台任务管理器。实例由 RuntimeContext 持有。"""

    def __init__(
        self,
        *,
        max_concurrent: int = 5,
        notification_stdout_cap: int = 500,
        result_stdout_cap: int = 5000,
        default_timeout: float = 600.0,
    ):
        self._tasks: dict = {}
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._max_concurrent = max_concurrent
        self._notification_stdout_cap = notification_stdout_cap
        self._result_stdout_cap = result_stdout_cap
        self._default_timeout = default_timeout

    # ---- 启动 ----
    def start(
        self,
        command: list,
        *,
        cwd: Optional[Path] = None,
        detach: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """启动后台任务。返回 task_id。

        Raises RuntimeError if at max_concurrent.
        """
        with self._lock:
            running_count = sum(
                1 for t in self._tasks.values() if t.status == "running"
            )
            if running_count >= self._max_concurrent:
                raise RuntimeError(
                    f"max concurrent tasks reached ({self._max_concurrent})"
                )

            task_id = "bg_" + secrets.token_hex(4)
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "cwd": str(cwd) if cwd else None,
            }
            if detach:
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    popen_kwargs["start_new_session"] = True

            try:
                proc = subprocess.Popen(command, **popen_kwargs)
            except (FileNotFoundError, OSError) as e:
                # 命令不存在等：直接创建 failed 任务
                task = BackgroundTask(
                    task_id=task_id,
                    command=command,
                    cwd=cwd,
                    status="failed",
                    pid=None,
                    started_at=datetime.now(),
                    detach=detach,
                    ended_at=datetime.now(),
                    exit_code=-1,
                    stderr=str(e),
                )
                self._tasks[task_id] = task
                self._push_notification_locked(task)
                return task_id

            task = BackgroundTask(
                task_id=task_id,
                command=command,
                cwd=cwd,
                status="running",
                pid=proc.pid,
                started_at=datetime.now(),
                detach=detach,
                _proc=proc,
            )
            self._tasks[task_id] = task

        # 起 daemon thread 监控
        effective_timeout = timeout if timeout is not None else self._default_timeout
        t = threading.Thread(
            target=self._watch,
            args=(task_id, proc, effective_timeout),
            daemon=True,
        )
        t.start()
        return task_id

    def _watch(self, task_id: str, proc: subprocess.Popen, timeout: float):
        """daemon thread：读 stdout/stderr、wait、push 通知。"""
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
            with self._lock:
                task = self._tasks.get(task_id)
                # I-3 修复：如果 stop() 已改状态（非 running），不覆盖，不重复 push 通知
                if task is None or task.status != "running":
                    return
                task.stdout = (stdout or "")[: self._result_stdout_cap]
                task.stderr = (stderr or "")[: self._result_stdout_cap]
                task.exit_code = exit_code
                task.ended_at = datetime.now()
                task.status = "completed" if exit_code == 0 else "failed"
                self._push_notification_locked(task)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.status != "running":
                    return
                task.stdout = (stdout or "")[: self._result_stdout_cap]
                task.stderr = (stderr or "")[: self._result_stdout_cap]
                task.exit_code = -1
                task.ended_at = datetime.now()
                task.status = "failed"
                self._push_notification_locked(task)
        except Exception as e:
            logger.warning("bg task %s watcher 异常: %s", task_id, e)
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.status != "running":
                    return
                task.status = "failed"
                task.ended_at = datetime.now()
                task.exit_code = -1
                task.stderr = str(e)[: self._result_stdout_cap]
                self._push_notification_locked(task)

    def _push_notification_locked(self, task: BackgroundTask):
        """必须在持有 self._lock 时调。push 一个通知到 deque。"""
        self._notifications.append({
            "task_id": task.task_id,
            "status": task.status,
            "exit_code": task.exit_code,
            "stdout": (task.stdout or "")[: self._notification_stdout_cap],
            "stderr": (task.stderr or "")[: self._notification_stdout_cap],
            "command": task.command,
            "ended_at": task.ended_at.isoformat() if task.ended_at else None,
        })

    # ---- 查询 ----
    def status(self, task_id: str) -> Optional[BackgroundTask]:
        """返回任务的浅拷贝（I-4 修复：防止调用方读到部分更新）。"""
        with self._lock:
            task = self._tasks.get(task_id)
            return copy.copy(task) if task else None

    def result(self, task_id: str) -> Optional[BackgroundTask]:
        # 同 status，工具层会用 task.stdout（cap result_stdout_cap）
        return self.status(task_id)

    def list_tasks(self) -> list:
        """返回所有任务的浅拷贝列表（I-4 修复）。"""
        with self._lock:
            return [copy.copy(t) for t in self._tasks.values()]

    # ---- 控制 ----
    def stop(self, task_id: str) -> bool:
        """终止任务。已完成的任务返回 True（幂等）。未知任务返回 False。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in ("completed", "failed", "stopped"):
                return True
            # I-3 修复：先标记 "stopping"，让 _watch() 的 communicate() 返回后
            # 检测到 status != "running" 而跳过覆盖
            task.status = "stopping"
            proc = task._proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("stop task %s 失败: %s", task_id, e)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return True
            # 无论 _watch 是否介入，stop 都设最终状态 "stopped"
            task.status = "stopped"
            task.ended_at = datetime.now()
            task.exit_code = task.exit_code if task.exit_code is not None else -1
            self._push_notification_locked(task)
        return True

    # ---- 通知 ----
    def drain_notifications(self) -> list:
        """主循环每轮调。返回并清空通知队列。"""
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
        return notifications

    # ---- 生命周期 ----
    def shutdown(self) -> None:
        """agent 退出时调；终止所有非 detach 的 running 任务。"""
        with self._lock:
            to_stop = [
                (tid, t) for tid, t in self._tasks.items()
                if t.status == "running" and not t.detach
            ]
        for tid, _ in to_stop:
            self.stop(tid)
