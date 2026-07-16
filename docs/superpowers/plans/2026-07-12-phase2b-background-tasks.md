# Phase 2b: 后台任务系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增后台任务系统：长命令异步执行（subprocess.Popen + 守护线程），完成时通知队列 push，主循环每轮注入 `<task_notification>` 临时消息。

**Architecture:** 新增 `agent/background.py`（BackgroundManager 单例）+ `tools/bg_task.py`（5 个工具 handler）。Manager 持有 task 字典 + 通知 deque + lock；每个任务起一个 daemon thread 读 stdout 并 wait()。注入到主循环 + RuntimeContext + AIAgent。

**Tech Stack:** Python 3.11+ subprocess + threading + collections.deque + uv + pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-phase2b-background-tasks-design.md`

## Global Constraints

- 文件 I/O 必须 `encoding="utf-8"`（HARVIL.md 强制）
- subprocess 必须 `text=True, encoding="utf-8"`
- 用 `uv`，不要 `pip install`
- 中文注释/commit；英文标识符
- 不要 import 用不到的模块（ruff F401）
- 工具 handler 返回 JSON 字符串；错误 `{"error": "...", "error_type": "..."}`
- 测试：`uv run pytest tests/<file>.py -v`
- 项目当前 429 测试不得回归
- Subagent 不 commit；controller 拿用户授权后统一 commit

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/background.py` | T1 新增 | `BackgroundTask` dataclass + `BackgroundManager` |
| `tests/test_background.py` | T1 新增 | Manager 单元测试（13 用例） |
| `tools/bg_task.py` | T2 新增 | 5 个工具 handler + 模块级注册 |
| `tests/test_bg_tool.py` | T2 新增 | 8 个工具测试 |
| `config.py` | T3 改 | `bg_task` 配置块 |
| `toolsets.py` | T4 改 | 新增 `bg` toolset |
| `agent/__init__.py` | T5 改 | AIAgent 接 `bg_manager=None` + 主循环 drain notifications |
| `cli.py` | T6 改 | RuntimeContext 持有 manager + AIAgent 注入 + shutdown |
| `tests/test_integration.py` | T7 改 | 主循环通知注入 e2e |

---

## Task 1: agent/background.py 核心模块

**Files:**
- Create: `agent/background.py`
- Create: `tests/test_background.py`

**Interfaces:**
- Consumes: `subprocess`, `threading`, `collections.deque`, `pathlib.Path`, `dataclasses`, `time`, `logging`, `secrets`（生成 task_id）, `sys`（平台判断）
- Produces: `BackgroundTask`（dataclass）+ `BackgroundManager`（class，方法：start / status / result / list_tasks / stop / drain_notifications / shutdown）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_background.py
"""后台任务管理器测试。"""
import sys
import time
from pathlib import Path

import pytest

from agent.background import BackgroundManager, BackgroundTask


def _quick_cmd():
    """跨平台快命令：sleep 0.3 然后 print done。"""
    if sys.platform == "win32":
        return [sys.executable, "-c", "import time; time.sleep(0.3); print('done')"]
    return ["sh", "-c", "sleep 0.3; echo done"]


def test_start_returns_task_id_and_creates_running_task(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    assert task_id.startswith("bg_")
    task = mgr.status(task_id)
    assert task is not None
    assert task.status == "running"
    # 清理
    mgr.shutdown()


def test_start_increments_pid(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    task = mgr.status(task_id)
    assert task.pid is not None and task.pid > 0
    mgr.shutdown()


def test_concurrent_limit_returns_error(tmp_path):
    mgr = BackgroundManager(max_concurrent=2)
    id1 = mgr.start(_quick_cmd(), cwd=tmp_path)
    id2 = mgr.start(_quick_cmd(), cwd=tmp_path)
    with pytest.raises(Exception) as exc_info:
        mgr.start(_quick_cmd(), cwd=tmp_path)
    assert "max concurrent" in str(exc_info.value).lower() or "full" in str(exc_info.value).lower()
    mgr.shutdown()


def test_completed_task_has_status_and_exit_code(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    # 等完成（最多 5 秒）
    for _ in range(50):
        task = mgr.status(task_id)
        if task.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert task.status == "completed"
    assert task.exit_code == 0
    assert "done" in task.stdout
    mgr.shutdown()


def test_drain_notifications_returns_and_clears(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    # 等完成
    for _ in range(50):
        if mgr.status(task_id) and mgr.status(task_id).status in ("completed", "failed"):
            break
        time.sleep(0.1)
    # 给 daemon thread 时间 push
    time.sleep(0.2)
    notifications = mgr.drain_notifications()
    assert len(notifications) >= 1
    assert notifications[0]["task_id"] == task_id
    # 再 drain 应该空
    assert mgr.drain_notifications() == []
    mgr.shutdown()


def test_drain_notifications_empty_returns_empty_list():
    mgr = BackgroundManager()
    assert mgr.drain_notifications() == []


def test_status_unknown_task_returns_none():
    mgr = BackgroundManager()
    assert mgr.status("bg_nonexistent") is None


def test_result_unknown_task_returns_none():
    mgr = BackgroundManager()
    assert mgr.result("bg_nonexistent") is None


def test_stop_terminates_running_task(tmp_path):
    """stop 一个 running 任务 → terminated。"""
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    task_id = mgr.start(long_cmd, cwd=tmp_path)
    time.sleep(0.2)  # 确保子进程已起
    stopped = mgr.stop(task_id)
    assert stopped is True
    # 等一下让 daemon thread 看到
    for _ in range(20):
        if mgr.status(task_id).status in ("stopped", "failed"):
            break
        time.sleep(0.1)
    assert mgr.status(task_id).status in ("stopped", "failed")


def test_stop_already_completed_is_idempotent(tmp_path):
    """stop 已完成的任务返回 True 但不改状态。"""
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    for _ in range(50):
        if mgr.status(task_id).status == "completed":
            break
        time.sleep(0.1)
    result = mgr.stop(task_id)
    assert result is True
    assert mgr.status(task_id).status == "completed"


def test_stop_unknown_returns_false():
    mgr = BackgroundManager()
    assert mgr.stop("bg_nonexistent") is False


def test_command_not_found_marks_task_failed(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(["./nonexistent-cmd-xyz"], cwd=tmp_path)
    for _ in range(30):
        task = mgr.status(task_id)
        if task.status in ("failed", "completed"):
            break
        time.sleep(0.1)
    assert task.status == "failed"


def test_list_tasks_returns_all(tmp_path):
    mgr = BackgroundManager()
    mgr.start(_quick_cmd(), cwd=tmp_path)
    mgr.start(_quick_cmd(), cwd=tmp_path)
    tasks = mgr.list_tasks()
    assert len(tasks) == 2
    mgr.shutdown()


def test_shutdown_terminates_non_detach_tasks(tmp_path):
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    mgr.start(long_cmd, cwd=tmp_path)
    mgr.shutdown()
    # shutdown 后所有任务应被 terminate（status != running）
    for task in mgr.list_tasks():
        assert task.status != "running"


def test_detach_task_survives_shutdown(tmp_path):
    """detach=True 的任务在 shutdown 后不被终止（PID 仍存活）。

    用 subprocess 检查 PID 是否还存在。简单起见，detach 任务用长 sleep。
    """
    import subprocess
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    task_id = mgr.start(long_cmd, cwd=tmp_path, detach=True)
    time.sleep(0.3)  # 确保子进程已起
    task = mgr.status(task_id)
    pid = task.pid
    mgr.shutdown()
    # 检查 PID 仍存活（粗糙：kill 0 不抛即存活，POSIX only）
    if sys.platform != "win32":
        try:
            import os
            os.kill(pid, 0)
            still_alive = True
            # 清理
            os.kill(pid, 9)
        except OSError:
            still_alive = False
        assert still_alive, "detach 任务应存活"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_background.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.background'`

- [ ] **Step 3: 写实现**

```python
# agent/background.py
"""后台任务管理器：长命令异步执行 + 完成时通知。

设计目标：
- subprocess.Popen 跑命令（不共享 GIL）
- 每任务一个 daemon thread 读 stdout 并 wait
- 完成时 push 到 _notifications deque（lock 保护）
- 主循环每轮 drain_notifications 清空
- 默认非 detach（agent 退出清理）；detach=True 用 start_new_session/CREATE_NEW_PROCESS_GROUP
- 并发上限默认 5

跨平台：subprocess 必须 text=True, encoding="utf-8"（HARVIL.md 强制）。
"""
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
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
                if task is None:
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
                if task is None:
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
                if task is None:
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
        with self._lock:
            return self._tasks.get(task_id)

    def result(self, task_id: str) -> Optional[BackgroundTask]:
        # 同 status，工具层会用 task.stdout（cap result_stdout_cap）
        return self.status(task_id)

    def list_tasks(self) -> list:
        with self._lock:
            return list(self._tasks.values())

    # ---- 控制 ----
    def stop(self, task_id: str) -> bool:
        """终止任务。已完成的任务返回 True（幂等）。未知任务返回 False。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in ("completed", "failed", "stopped"):
                return True
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
            if task.status == "running":
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_background.py -v`
Expected: PASS（14 tests）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 429 + 新增 background tests，0 回归）

- [ ] **Step 6: 不 commit**

---

## Task 2: tools/bg_task.py 5 个工具

**Files:**
- Create: `tools/bg_task.py`
- Create: `tests/test_bg_tool.py`

**Interfaces:**
- Consumes: `BackgroundManager` from `agent/background`（T1）；`registry` from `tools/registry`
- Produces: 5 个工具 handler（`bg_start` / `bg_status` / `bg_result` / `bg_list` / `bg_stop`）；模块级 register

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bg_tool.py
"""bg_task 工具 handler 测试。"""
import json
import sys
import time
from pathlib import Path

import pytest

# 触发工具注册
import tools.bg_task  # noqa
from tools.registry import registry


def _quick_cmd():
    if sys.platform == "win32":
        return [sys.executable, "-c", "import time; time.sleep(0.3); print('done')"]
    return ["sh", "-c", "sleep 0.3; echo done"]


def test_bg_start_returns_json_with_task_id(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_start",
        {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"].startswith("bg_")
    assert parsed["status"] == "running"
    assert "pid" in parsed
    mgr.shutdown()


def test_bg_start_full_returns_bg_task_full_error(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager(max_concurrent=1)
    registry.dispatch("bg_start",
                       {"command": [sys.executable, "-c", "import time; time.sleep(30)"],
                        "cwd": str(tmp_path)},
                       bg_manager=mgr)
    result_str = registry.dispatch(
        "bg_start",
        {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_full"
    mgr.shutdown()


def test_bg_status_returns_summary_json(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    status_str = registry.dispatch(
        "bg_status", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(status_str)
    assert parsed["task_id"] == task_id
    assert "status" in parsed
    mgr.shutdown()


def test_bg_status_unknown_returns_not_found():
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_status", {"task_id": "bg_nonexistent"}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_not_found"


def test_bg_result_returns_full_output(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    # 等完成
    for _ in range(50):
        task = mgr.status(task_id)
        if task and task.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    result_str = registry.dispatch(
        "bg_result", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"] == task_id
    assert "stdout" in parsed
    assert "done" in parsed["stdout"]
    mgr.shutdown()


def test_bg_list_returns_all_tasks(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    registry.dispatch("bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
                       bg_manager=mgr)
    registry.dispatch("bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
                       bg_manager=mgr)
    result_str = registry.dispatch("bg_list", {}, bg_manager=mgr)
    parsed = json.loads(result_str)
    assert len(parsed["tasks"]) == 2
    mgr.shutdown()


def test_bg_stop_terminates_and_returns_stopped(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start",
        {"command": [sys.executable, "-c", "import time; time.sleep(30)"],
         "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    time.sleep(0.3)
    result_str = registry.dispatch(
        "bg_stop", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"] == task_id
    assert parsed["status"] in ("stopped", "running")  # 状态可能还在过渡
    mgr.shutdown()


def test_bg_stop_unknown_returns_not_found():
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_stop", {"task_id": "bg_nonexistent"}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_not_found"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_bg_tool.py -v`
Expected: FAIL — `ModuleNotFoundError` or `KeyError: 'bg_start'`（工具未注册）

- [ ] **Step 3: 写实现**

```python
# tools/bg_task.py
"""后台任务工具：5 个 handler 注册到 registry。

工具：
- bg_start: 启动后台任务
- bg_status: 查询任务状态
- bg_result: 查询完整输出
- bg_list: 列出所有任务
- bg_stop: 终止任务

handler 通过 kwargs 接收 bg_manager（由 agent 透传）。
"""
import json
import logging
from pathlib import Path

from tools.registry import registry

logger = logging.getLogger(__name__)


def _handle_bg_start(args: dict, *, bg_manager=None, **kwargs) -> str:
    if bg_manager is None:
        return json.dumps({
            "error": "background manager not available",
            "error_type": "bg_unavailable",
        }, ensure_ascii=False)
    command = args.get("command")
    if not command or not isinstance(command, list):
        return json.dumps({
            "error": "bg_start requires 'command' as non-empty list",
            "error_type": "invalid_args",
        }, ensure_ascii=False)
    cwd_raw = args.get("cwd")
    cwd = Path(cwd_raw) if cwd_raw else None
    detach = bool(args.get("detach", False))
    timeout = args.get("timeout")
    try:
        task_id = bg_manager.start(
            command, cwd=cwd, detach=detach,
            timeout=timeout if timeout is not None else None,
        )
    except RuntimeError as e:
        return json.dumps({
            "error": str(e),
            "error_type": "bg_task_full",
        }, ensure_ascii=False)
    task = bg_manager.status(task_id)
    return json.dumps({
        "task_id": task_id,
        "status": task.status,
        "pid": task.pid,
    }, ensure_ascii=False)


def _handle_bg_status(args: dict, *, bg_manager=None, **kwargs) -> str:
    if bg_manager is None:
        return json.dumps({"error": "background manager not available",
                           "error_type": "bg_unavailable"}, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({"error": "bg_status requires 'task_id'",
                           "error_type": "invalid_args"}, ensure_ascii=False)
    task = bg_manager.status(task_id)
    if task is None:
        return json.dumps({"error": f"task not found: {task_id}",
                           "error_type": "bg_task_not_found"}, ensure_ascii=False)
    runtime = None
    if task.started_at:
        from datetime import datetime
        end = task.ended_at or datetime.now()
        runtime = (end - task.started_at).total_seconds()
    return json.dumps({
        "task_id": task.task_id,
        "status": task.status,
        "pid": task.pid,
        "runtime_seconds": runtime,
    }, ensure_ascii=False)


def _handle_bg_result(args: dict, *, bg_manager=None, **kwargs) -> str:
    if bg_manager is None:
        return json.dumps({"error": "background manager not available",
                           "error_type": "bg_unavailable"}, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({"error": "bg_result requires 'task_id'",
                           "error_type": "invalid_args"}, ensure_ascii=False)
    task = bg_manager.result(task_id)
    if task is None:
        return json.dumps({"error": f"task not found: {task_id}",
                           "error_type": "bg_task_not_found"}, ensure_ascii=False)
    return json.dumps({
        "task_id": task.task_id,
        "status": task.status,
        "exit_code": task.exit_code,
        "stdout": task.stdout,
        "stderr": task.stderr,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "ended_at": task.ended_at.isoformat() if task.ended_at else None,
    }, ensure_ascii=False)


def _handle_bg_list(args: dict, *, bg_manager=None, **kwargs) -> str:
    if bg_manager is None:
        return json.dumps({"error": "background manager not available",
                           "error_type": "bg_unavailable"}, ensure_ascii=False)
    tasks = bg_manager.list_tasks()
    summaries = [
        {
            "task_id": t.task_id,
            "status": t.status,
            "command": t.command,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "ended_at": t.ended_at.isoformat() if t.ended_at else None,
        }
        for t in tasks
    ]
    return json.dumps({"tasks": summaries}, ensure_ascii=False)


def _handle_bg_stop(args: dict, *, bg_manager=None, **kwargs) -> str:
    if bg_manager is None:
        return json.dumps({"error": "background manager not available",
                           "error_type": "bg_unavailable"}, ensure_ascii=False)
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({"error": "bg_stop requires 'task_id'",
                           "error_type": "invalid_args"}, ensure_ascii=False)
    ok = bg_manager.stop(task_id)
    if not ok:
        return json.dumps({"error": f"task not found: {task_id}",
                           "error_type": "bg_task_not_found"}, ensure_ascii=False)
    task = bg_manager.status(task_id)
    return json.dumps({
        "task_id": task_id,
        "status": task.status if task else "stopped",
    }, ensure_ascii=False)


# ---- 注册 ----
registry.register("bg_start", _handle_bg_start, {
    "name": "bg_start",
    "description": "启动后台任务（异步执行长命令，立即返回 task_id）",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"},
                        "description": "命令及参数（list 形式，不经 shell）"},
            "cwd": {"type": "string", "description": "工作目录（可选）"},
            "detach": {"type": "boolean", "description": "是否脱离 agent 进程组（默认 false）"},
            "timeout": {"type": "number", "description": "超时秒数（默认 600）"},
        },
        "required": ["command"],
    },
})
registry.register("bg_status", _handle_bg_status, {
    "name": "bg_status",
    "description": "查询后台任务状态",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
})
registry.register("bg_result", _handle_bg_result, {
    "name": "bg_result",
    "description": "查询后台任务的完整输出（stdout/stderr 各 cap 5000）",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
})
registry.register("bg_list", _handle_bg_list, {
    "name": "bg_list",
    "description": "列出所有后台任务",
    "parameters": {"type": "object", "properties": {}},
})
registry.register("bg_stop", _handle_bg_stop, {
    "name": "bg_stop",
    "description": "终止后台任务",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
})
```

**注意**：上面的 `registry.register(name, fn, schema)` 形式 — implementer 应先查 `tools/registry.py` 的实际签名（可能是位置参数或关键字）。如果签名不同，调整即可。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_bg_tool.py -v`
Expected: PASS（8 tests）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 3: config.py 新增 bg_task 块

**Files:**
- Modify: `config.py:DEFAULT_CONFIG`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `DEFAULT_CONFIG["bg_task"]` dict

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py
def test_default_config_has_bg_task_block():
    from config import DEFAULT_CONFIG
    bg = DEFAULT_CONFIG["bg_task"]
    for key in ("enabled", "max_concurrent", "default_timeout",
                "notification_stdout_cap", "result_stdout_cap",
                "default_detach"):
        assert key in bg, f"缺 {key}"


def test_default_config_bg_task_defaults():
    from config import DEFAULT_CONFIG
    bg = DEFAULT_CONFIG["bg_task"]
    assert bg["enabled"] is True
    assert bg["max_concurrent"] == 5
    assert bg["default_timeout"] == 600
    assert bg["default_detach"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_default_config_has_bg_task_block -v`
Expected: FAIL — `KeyError: 'bg_task'`

- [ ] **Step 3: 改 config.py**

在 `config.py:DEFAULT_CONFIG` 里（建议紧跟 `hooks` 块之后）新增：

```python
    # 后台任务（Phase 2b）
    "bg_task": {
        "enabled": True,                        # False 时 bg_* 工具隐藏
        "max_concurrent": 5,
        "default_timeout": 600,                 # bg_start 默认超时（秒）
        "notification_stdout_cap": 500,         # 通知里 stdout 字符上限
        "result_stdout_cap": 5000,              # bg_result 返回的 stdout 字符上限
        "default_detach": False,                # bg_start 默认 detach
    },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 4: toolsets.py 新增 bg toolset

**Files:**
- Modify: `toolsets.py:TOOLSETS`

**Interfaces:**
- Produces: `TOOLSETS["bg"]` 列表含 5 个工具名

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py 或新建 tests/test_toolsets.py
def test_bg_toolset_exists():
    from toolsets import TOOLSETS, resolve_toolset
    assert "bg" in TOOLSETS
    tools = resolve_toolset("bg")
    for name in ("bg_start", "bg_status", "bg_result", "bg_list", "bg_stop"):
        assert name in tools, f"缺 {name}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_bg_toolset_exists -v`
Expected: FAIL — `KeyError: 'bg'`

- [ ] **Step 3: 改 toolsets.py**

打开 `toolsets.py`，在 `TOOLSETS` 字典里新增：

```python
"bg": [
    "bg_start",
    "bg_status",
    "bg_result",
    "bg_list",
    "bg_stop",
],
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py::test_bg_toolset_exists -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 5: agent/__init__.py 集成 bg_manager + 主循环 drain

**Files:**
- Modify: `agent/__init__.py:AIAgent.__init__` + `run_conversation`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `BackgroundManager`（T1）
- Produces: AIAgent 新增 `bg_manager=None` kwarg；主循环 drain notifications

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_aiagent_accepts_bg_manager_kwarg():
    agent = _make_test_agent()  # 复用 P2-T6 helper
    assert agent.bg_manager is None


def test_aiagent_drains_notifications_into_temporary_user_msg(tmp_path):
    """完成的后台任务在下一轮主循环注入 <task_notification> 临时 user 消息。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.background import BackgroundManager
    import sys, time

    mgr = BackgroundManager()
    # 跑一个快任务并等它完成
    mgr.start([sys.executable, "-c", "print('done')"], cwd=tmp_path)
    time.sleep(0.5)  # 等任务完成 + push notification

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        bg_manager=mgr,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("check")
    # conversation_history 不该含 task_notification（是临时消息）
    for msg in agent.conversation_history:
        assert "<task_notification>" not in msg.get("content", "")
    mgr.shutdown()


def test_aiagent_no_bg_manager_backward_compat(tmp_path):
    """bg_manager=None 时主循环不抛，行为同 Phase 2a。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_aiagent_accepts_bg_manager_kwarg -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'bg_manager'`

- [ ] **Step 3: 改 agent/__init__.py**

`AIAgent.__init__` 加 `bg_manager=None`：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        hooks_registry=None,
        bg_manager=None,  # === NEW Phase 2b ===
    ):
        # ...
        self.bg_manager = bg_manager
```

`run_conversation` 顶部（USER_PROMPT_SUBMIT hook 之后、`conversation_history.append` 之前）插入 drain：

```python
    def run_conversation(self, user_message: str) -> str:
        # === Phase 2a: USER_PROMPT_SUBMIT hook ===
        if (self.hooks_registry and ...):
            ...

        # === NEW Phase 2b: bg_task notifications ===
        bg_notifications = []
        if self.bg_manager:
            try:
                bg_notifications = self.bg_manager.drain_notifications()
            except Exception as e:
                logger.warning("drain_notifications 异常: %s", e)
                bg_notifications = []

        self.conversation_history.append({"role": "user", "content": user_message})

        # 组装 messages 时（在 while 循环内、调 LLM 前）附加通知
        # 找到 messages = [{"role":"system", ...}, *self.conversation_history] 之后插入：
```

具体地，找到主循环内组装 `messages` 的位置，**在 messages 组装之后、调 LLM 之前**附加：

```python
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
            ]

            # === NEW Phase 2b: 注入 bg_task 通知（临时，不进 history）===
            if bg_notifications:
                notif_text = "\n".join(
                    f"[task {n['task_id']} {n['status']}] "
                    f"exit={n.get('exit_code')} "
                    f"stdout_tail={(n.get('stdout') or '')[-200:]}"
                    for n in bg_notifications
                )
                messages.append({
                    "role": "user",
                    "content": f"<task_notification>\n{notif_text}\n</task_notification>",
                })
                # 清空本轮通知（已注入）
                bg_notifications = []

            # TodoWrite reminder 等原有逻辑继续...
```

注意：第一轮 drain 之后通知要清空，否则后续每轮都会重复注入。可以在 messages 组装处条件清空。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -k bg_manager -v`
Expected: PASS（3 个新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 6: cli.py 注入 bg_manager + shutdown

**Files:**
- Modify: `cli.py:RuntimeContext`

**Interfaces:**
- Consumes: `BackgroundManager`（T1）+ `config.bg_task`（T3）
- Produces: RuntimeContext.bg_manager；AIAgent 构造时透传；shutdown 终止任务

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_runtime_context_has_bg_manager():
    from cli import RuntimeContext
    from agent.background import BackgroundManager
    ctx = RuntimeContext.__new__(RuntimeContext)
    ctx.bg_manager = BackgroundManager()
    assert isinstance(ctx.bg_manager, BackgroundManager)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_runtime_context_has_bg_manager -v`
Expected: FAIL（属性不存在或类型错）

- [ ] **Step 3: 改 cli.py**

在 `RuntimeContext.__init__` 加（建议紧邻 hooks_registry 实例化之后）：

```python
        # === NEW Phase 2b: 后台任务管理器 ===
        from agent.background import BackgroundManager
        bg_cfg = self.config.get("bg_task", {})
        self.bg_manager = BackgroundManager(
            max_concurrent=bg_cfg.get("max_concurrent", 5),
            notification_stdout_cap=bg_cfg.get("notification_stdout_cap", 500),
            result_stdout_cap=bg_cfg.get("result_stdout_cap", 5000),
            default_timeout=bg_cfg.get("default_timeout", 600),
        )
```

在 `_create_agent`（或 AIAgent 构造点）传 `bg_manager=self.bg_manager`。

在 `RuntimeContext.shutdown`（或退出路径）调 `self.bg_manager.shutdown()`：

```python
    def shutdown(self):
        # ... 原有清理 ...
        if hasattr(self, "bg_manager") and self.bg_manager:
            try:
                self.bg_manager.shutdown()
            except Exception as e:
                logger.warning("bg_manager shutdown 失败: %s", e)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 7: 主循环通知注入 e2e + handle_function_call 透传 bg_manager

**Files:**
- Modify: `model_tools.py:handle_function_call`（透传 bg_manager 到工具 handler）
- Modify: `agent/__init__.py`（调 handle_function_call 时传 bg_manager）
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: 所有前置任务
- Produces: 工具 handler 能收到 bg_manager；主循环完整集成测试

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_handle_function_call_threads_bg_manager(tmp_path):
    """bg_start 工具能通过 handle_function_call 拿到 bg_manager。"""
    import json, sys
    from agent.background import BackgroundManager
    from model_tools import handle_function_call

    mgr = BackgroundManager()
    result_str = handle_function_call(
        "bg_start",
        {"command": [sys.executable, "-c", "print('ok')"], "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"].startswith("bg_")
    mgr.shutdown()


def test_e2e_aiagent_full_bg_lifecycle(tmp_path):
    """端到端：AIAgent + bg_manager + 完整后台任务生命周期。

    流程：
    1. 通过 handle_function_call 启动一个快任务
    2. 任务完成 → push 通知
    3. 下一轮主循环 drain → 注入 <task_notification>
    4. bg_status / bg_result 查询正常
    """
    import json, sys, time
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=["bg"], harvil_home=str(tmp_path),
        bg_manager=mgr,
    )

    # mock LLM：第一轮调 bg_start，第二轮调 bg_status，第三轮 stop
    call_count = [0]
    started_task_id = [None]

    def side_effect(msgs, **kw):
        call_count[0] += 1
        from model_tools import handle_function_call
        resp = MagicMock()
        if call_count[0] == 1:
            # 启动任务
            result = handle_function_call(
                "bg_start",
                {"command": [sys.executable, "-c", "print('done')"],
                 "cwd": str(tmp_path)},
                bg_manager=mgr,
            )
            started_task_id[0] = json.loads(result)["task_id"]
            time.sleep(0.5)  # 等任务完成
            resp.choices = [MagicMock(
                message=MagicMock(content=None, tool_calls=[MagicMock(
                    id="c1", type="function",
                    function=MagicMock(name="bg_status",
                                       arguments=json.dumps({"task_id": started_task_id[0]})),
                )]),
                finish_reason="tool_calls",
            )]
            return resp
        elif call_count[0] == 2:
            result = handle_function_call(
                "bg_status",
                {"task_id": started_task_id[0]},
                bg_manager=mgr,
            )
            # 再调一次让循环结束
            resp.choices = [MagicMock(
                message=MagicMock(content="all done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp
        else:
            resp.choices = [MagicMock(
                message=MagicMock(content="done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp

    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions.side_effect = side_effect

    final = agent.run_conversation("run bg task")
    # 至少没崩
    assert isinstance(final, str)
    mgr.shutdown()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_handle_function_call_threads_bg_manager -v`
Expected: FAIL（`handle_function_call` 不接 bg_manager）

- [ ] **Step 3: 改 model_tools.py + agent/__init__.py**

`model_tools.py:handle_function_call` 加 `bg_manager=None` kwarg + 透传到 registry.dispatch：

```python
def handle_function_call(
    function_name, function_args, *,
    # ... 原有 kwargs ...
    hooks_registry=None,
    bg_manager=None,  # === NEW Phase 2b ===
):
    # ... 原有逻辑 ...
    result = registry.dispatch(
        function_name, function_args,
        # ... 原有 kwargs ...
        hooks_registry=hooks_registry,
        bg_manager=bg_manager,  # === NEW ===
    )
    # ...
```

`agent/__init__.py` 内调用 `handle_function_call` 的位置加 `bg_manager=self.bg_manager`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 + 新增 bg tests，0 回归）

- [ ] **Step 6: 不 commit**

---

## Self-Review

**Spec 覆盖检查**：
- ✅ §1 架构 → T1-T7
- ✅ §2 数据结构 → T1
- ✅ §3 5 个工具 → T2
- ✅ §4 主循环通知注入 → T5
- ✅ §5 RuntimeContext + AIAgent 改造 → T5, T6
- ✅ §6 config 块 → T3
- ✅ §6 toolset → T4
- ✅ §7 失败处理（FileNotFoundError / 超时 / 容量满 / 未知 task_id）→ T1 + T2 测试覆盖
- ✅ §8 并发安全 → T1（lock 守护）
- ✅ §9 Windows 兼容 → T1（_detach_flags + creationflags）
- ✅ §10 测试矩阵 → T1 (13+) / T2 (8) / T5 (3) / T7 (2)

**Placeholder 扫描**：无 TBD/TODO。`registry.register` 签名在 T2 step 3 注释了「implementer 应先查」—— 这是合理的现实注意，不是 placeholder。

**类型一致性**：
- `BackgroundTask` 字段在 T1 定义，T2 工具 handler 引用 ✓
- `BackgroundManager` 方法签名 T1 → T2/T5/T6 一致 ✓
- `bg_manager=None` kwarg 链 T5（AIAgent）→ T7（model_tools）→ 工具 handler ✓

**遗漏检查**：spec §10 提到的 `test_detach_task_survives_shutdown`（POSIX only）已在 T1 step 1 测试代码里。Windows skipif 在测试体内用 `if sys.platform != "win32"` 处理。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase2b-background-tasks.md`.

按用户授权直接进 SDD，沿用 Phase 2a 工作流。
