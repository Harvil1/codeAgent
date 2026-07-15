# Phase 2b: 后台任务系统设计

- **日期**：2026-07-12
- **状态**：用户授权直接推进
- **范围**：仅 Phase 2b（后台任务）。2c Cron 留给 Phase 2.3
- **对应 Spec**：`docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 2b 展开
- **依赖**：Phase 2a Hooks 已就绪（可选用 PostToolUse hook 通知，但非必须）

---

## 摘要

新增后台任务系统：长命令异步执行（`subprocess.Popen` + 守护线程），完成时 push 到线程安全队列，主循环每轮 LLM 调用前 drain 并作为 `<task_notification>` 临时 user 消息注入（不进 conversation_history）。

5 个工具：`bg_start` / `bg_status` / `bg_result` / `bg_list` / `bg_stop`。默认同 process group（agent 退出时子进程被带杀），`detach=True` 让子进程独立存活。并发上限默认 5。

---

## §1 架构

```
┌─────────────────────────────────────────────────────────────┐
│ LLM 调 bg_start("python long_script.py")                    │
│     ↓                                                       │
│ BackgroundManager.start()                                   │
│     ├─ subprocess.Popen(command, cwd, encoding="utf-8")     │
│     ├─ daemon thread: read stdout/stderr, wait()            │
│     └─ return task_id immediately                           │
│                                                             │
│ 主循环每轮 LLM 前：                                          │
│   notifications = bg_manager.drain_notifications()           │
│   if notifications:                                          │
│       messages.append({"role":"user",                        │
│                         "content":"<task_notification>..."}) │
│                                                             │
│ 完成时（daemon thread 内）：                                  │
│   with lock: notifications_queue.append({task_id, result})  │
└─────────────────────────────────────────────────────────────┘
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/background.py` | 🆕 新增 | `BackgroundManager` + `BackgroundTask` dataclass |
| `tools/bg_task.py` | 🆕 新增 | 5 个工具 handler（注册到 registry） |
| `agent/__init__.py` | ♻️ 改 | 主循环每轮 drain notifications |
| `cli.py` | ♻️ 改 | RuntimeContext 持有 `bg_manager`；注入 AIAgent |
| `config.py` | ♻️ 改 | `bg_task` 配置块 |
| `tests/test_background.py` | 🆕 新增 | BackgroundManager 单元测试 |
| `tests/test_bg_tool.py` | 🆕 新增 | 5 个工具 handler 测试 |
| `tests/test_integration.py` | ♻️ 改 | 主循环通知注入 e2e |

### 关键决策

1. **执行模型**：`subprocess.Popen` + 守护线程读输出 + `wait()`。每个任务一个线程。
2. **默认非 detach**：同 process group。`detach=True` 时用 `start_new_session=True`（POSIX）/ `creationflags=CREATE_NEW_PROCESS_GROUP`（Windows）。
3. **队列**：内存 `collections.deque` + `threading.Lock`。不持久化。
4. **通知注入**：每轮 LLM 前 drain，作为临时 user 消息（不进 conversation_history，类似 TodoWrite reminder）。
5. **并发上限**：默认 5；超过返回 `error_type=bg_task_full`。
6. **输出 cap**：每通知 stdout/stderr 各 cap 500 字符；完整输出 `bg_result` 可查（cap 5000）。

---

## §2 数据结构

```python
@dataclass
class BackgroundTask:
    task_id: str               # ulid-like，如 "bg_abc123"
    command: list              # 已解析的命令 list（不经 shell）
    cwd: Optional[Path]
    status: str                # "running" | "completed" | "failed" | "stopped"
    pid: Optional[int]
    started_at: datetime
    ended_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""           # 累积，cap 5000 字符
    stderr: str = ""           # 累积，cap 5000 字符
    detach: bool = False
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
```

```python
class BackgroundManager:
    def __init__(self, *, max_concurrent: int = 5,
                 notification_stdout_cap: int = 500,
                 result_stdout_cap: int = 5000):
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._max_concurrent = max_concurrent
        ...

    def start(self, command, *, cwd=None, detach=False, timeout=600) -> str:
        """返回 task_id。超容量抛/返回 error。"""

    def status(self, task_id: str) -> Optional[BackgroundTask]: ...
    def result(self, task_id: str) -> Optional[BackgroundTask]: ...
    def list_tasks(self) -> list[BackgroundTask]: ...
    def stop(self, task_id: str) -> bool: ...

    def drain_notifications(self) -> list[dict]:
        """主循环每轮调用。返回并清空累积通知。"""

    def shutdown(self) -> None:
        """agent 退出时调用；终止所有非 detach 任务。"""
```

---

## §3 工具表面（5 个工具）

所有工具 handler 在 `tools/bg_task.py`，模块 import 时注册到 `registry`。返回 JSON 字符串。

### `bg_start`

```python
def bg_start(command: list[str], *, cwd: str = None,
              detach: bool = False, timeout: int = 600) -> str:
    """启动后台任务。立即返回 task_id。"""
```

返回：
```json
{"task_id": "bg_abc123", "status": "running", "pid": 12345}
```

错误：
```json
{"error": "max concurrent tasks reached (5)", "error_type": "bg_task_full"}
```

### `bg_status`

```json
{"task_id": "bg_abc123", "status": "running", "pid": 12345,
 "runtime_seconds": 12.3}
```

### `bg_result`

返回完整 task 对象（stdout/stderr 各 cap 5000）：
```json
{"task_id": "...", "status": "completed", "exit_code": 0,
 "stdout": "...", "stderr": "...",
 "started_at": "...", "ended_at": "..."}
```

### `bg_list`

```json
{"tasks": [{...task summary...}, ...]}
```

### `bg_stop`

```json
{"task_id": "...", "status": "stopped"}
```

非 detach 任务用 `proc.terminate()` 然后 `proc.wait(timeout=5)`；超时 `proc.kill()`。

---

## §4 主循环通知注入

**位置**：`agent/__init__.py:run_conversation` 顶部（在 USER_PROMPT_SUBMIT hook 之后、组装 messages 之前）。

**实现**：
```python
# === NEW: bg_task notifications ===
if self.bg_manager:
    try:
        notifications = self.bg_manager.drain_notifications()
    except Exception as e:
        logger.warning("drain_notifications 异常: %s", e)
        notifications = []
    if notifications:
        notif_text = "\n".join(
            f"[task {n['task_id']} {n['status']}] "
            f"exit={n.get('exit_code')} "
            f"stdout_tail={n.get('stdout', '')[-200:]}"
            for n in notifications
        )
        # 临时 user 消息，不进 conversation_history
        messages.append({
            "role": "user",
            "content": f"<task_notification>\n{notif_text}\n</task_notification>",
        })
```

**关键**：这条消息**不进** `self.conversation_history`，每轮重新构造。和 TodoWrite reminder 同模式。

---

## §5 改造 RuntimeContext + AIAgent

### RuntimeContext

```python
# cli.py:RuntimeContext.__init__
from agent.background import BackgroundManager

self.bg_manager = BackgroundManager(
    max_concurrent=self.config.get("bg_task", {}).get("max_concurrent", 5),
    notification_stdout_cap=self.config.get("bg_task", {}).get(
        "notification_stdout_cap", 500),
    result_stdout_cap=self.config.get("bg_task", {}).get(
        "result_stdout_cap", 5000),
)
```

### AIAgent

```python
# agent/__init__.py:AIAgent.__init__
def __init__(self, ..., bg_manager=None, ...):
    self.bg_manager = bg_manager
```

### AIAgent 退出

在 cli.py 的退出路径调 `bg_manager.shutdown()`（终止所有非 detach 任务）。具体位置：`RuntimeContext.shutdown` 或 `__exit__`。

---

## §6 配置

```python
# config.py:DEFAULT_CONFIG
"bg_task": {
    "enabled": True,                       # False 时 bg_* 工具隐藏
    "max_concurrent": 5,
    "default_timeout": 600,                # bg_start 默认超时
    "notification_stdout_cap": 500,        # 通知里 stdout 字符上限
    "result_stdout_cap": 5000,             # bg_result 返回的 stdout 字符上限
    "default_detach": False,               # bg_start 默认 detach
},
```

### toolsets.py

把 `bg_start` / `bg_status` / `bg_result` / `bg_list` / `bg_stop` 加入 `TOOLSETS["core"]`（或独立 `TOOLSETS["bg"]`）。建议独立 `bg` toolset，用户可单独启用。

---

## §7 失败处理

| 故障 | 行为 |
|---|---|
| 达到 max_concurrent | `bg_start` 返回 `bg_task_full` error |
| 命令不存在 | Popen 抛 FileNotFoundError → task 立即 status=failed, exit_code=-1, stderr=str(e) |
| 超时 | daemon thread 在 timeout 后调 proc.kill()，task status=failed |
| daemon thread 异常 | log warning + task status=failed |
| 主循环 drain 异常 | log warning + 当轮跳过通知 |
| bg_status/result 查不到 task_id | 返回 `{"error": "task not found", "error_type": "bg_task_not_found"}` |
| bg_stop 任务已结束 | 返回当前 status（幂等，不报错） |

---

## §8 并发安全

- `self._tasks` 字典 + `self._notifications` deque 都用同一个 `self._lock` 守护
- daemon thread 写 task 状态前先 `with self._lock:`
- 主循环 drain 前 `with self._lock: notifications = list(deque); deque.clear()`
- 工具 handler 读 task 前 `with self._lock:` 浅拷贝返回

---

## §9 Windows 兼容

- `subprocess.Popen` 用 `text=True, encoding="utf-8"`（CLAUDE.md 强制）
- `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` 用于 detach（Windows）
- `start_new_session=True` 用于 detach（POSIX）
- 平台分支封装在 `_detach_flags(detach: bool) -> dict` helper

---

## §10 测试矩阵

### `tests/test_background.py`

- `test_start_returns_task_id_and_status_running`
- `test_concurrent_limit_returns_error`
- `test_daemon_thread_pushes_notification_on_complete`
- `test_drain_notifications_clears_queue`
- `test_drain_notifications_empty_returns_empty_list`
- `test_status_unknown_task_returns_none`
- `test_stop_terminates_running_task`
- `test_stop_already_completed_is_idempotent`
- `test_detach_does_not_die_with_manager`（POSIX skip on Windows）
- `test_command_not_found_marks_task_failed`
- `test_timeout_kills_task`
- `test_shutdown_terminates_non_detach_tasks`
- `test_shutdown_preserves_detach_tasks`

### `tests/test_bg_tool.py`

- `test_bg_start_returns_json_with_task_id`
- `test_bg_start_full_returns_bg_task_full_error`
- `test_bg_status_returns_summary_json`
- `test_bg_status_unknown_returns_not_found`
- `test_bg_result_returns_full_output_capped`
- `test_bg_list_returns_all_tasks`
- `test_bg_stop_terminates_and_returns_stopped`
- `test_bg_stop_unknown_returns_not_found`

### `tests/test_integration.py`

- `test_aiagent_drains_notifications_into_temporary_user_msg`：mock 一个任务完成 → 主循环看到 `<task_notification>` 消息
- `test_aiagent_no_bg_manager_backward_compat`：`bg_manager=None` 时主循环不抛

---

## §11 已知限制 / 非目标

1. **不持久化任务状态**：agent 重启后所有 task 信息丢失。生产用需要查询 OS 进程表辅助。
2. **不实现任务依赖**：`bg_start` 不能等另一个 `bg_task` 完成。
3. **不实现 streaming stdout**：daemon thread 累积到 cap；用户用 `bg_result` 查快照。
4. **detach 任务不可回收**：detach=True 后，agent 无法 stop 它（pid 已独立）。`bg_stop` 返回 not_found。
5. **不做跨进程锁**：单进程 agent 内的 threading.Lock 足够。多 agent 实例共享任务不在范围。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代 | 理由 |
|---|---|---|---|
| 执行模型 | subprocess.Popen + 守护线程 | threading/os.system / multiprocessing | 跨平台、独立进程、不共享 GIL |
| 默认 detach | False | True | 默认安全（agent 退出清理）；显式 opt-in 才独立 |
| 队列存储 | 内存 deque | 文件持久化 | 通知用完即弃，跨会话价值低 |
| 通知注入 | 临时 user 消息（不进 history） | 持久 user 消息 | 不占 context；类似 TodoWrite reminder 模式 |
| 并发上限 | 默认 5 | 无限 / 配置化 | 防失控；可配置 |
| toolset 归属 | 独立 `bg` toolset | 加入 core | 用户可单独禁用 |

## 附录 B: 与现有原则对齐

| 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰 | 后台任务是工具（数据驱动），不是核心循环代码 |
| Prompt Caching 神圣 | 通知是 user 消息，不改 system prompt |
| 完全可逆 | 任务可 bg_stop；通知队列 drain 即清空 |
| 用户意图优先 | detach 是用户显式选择 |
| 安全默认 | 默认非 detach（agent 退出清理）；命令走 PermissionChecker |
| 发现 ≠ 可见 | bg_* 工具注册到独立 toolset；config.bg_task.enabled 控制可见 |
