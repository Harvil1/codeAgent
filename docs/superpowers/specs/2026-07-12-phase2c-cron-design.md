# Phase 2c: Cron 调度系统设计

- **日期**：2026-07-12
- **状态**：用户授权直接推进
- **范围**：仅 Phase 2c（Cron）。完成 Phase 2 全部
- **对应 Spec**：`docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 2c 展开
- **依赖**：Phase 1（主循环）+ Phase 2a Hooks（非必须但可搭配）

---

## 摘要

新增 Cron 调度系统：后台 daemon 线程每 30s 扫描 `~/.agent/.cron/jobs.json`，匹配到期的 job push 到线程安全队列，主循环每轮 LLM 前 drain 并作为 `<scheduled_message>` 临时 user 消息注入。手写 5-field cron 表达式 parser（无依赖）。错过默认不补救（`catch_up: false`）。`config.cron.enabled=False` 完全关闭。

---

## §1 架构

```
┌────────────────────────────────────────────────────────┐
│ 启动时（RuntimeContext.__init__）                       │
│   CronScheduler()                                      │
│     ├─ 加载 ~/.agent/.cron/jobs.json                   │
│     ├─ 起后台 daemon thread（30s tick）                │
│     └─ agent_lock = threading.Lock()                   │
│                                                        │
│ 后台 daemon thread (每 30s)：                           │
│   for job in enabled_jobs:                             │
│     if cron_match(job.cron, now) and                   │
│        last_fired[job.id] != minute_marker(now):       │
│       notifications.append({job_id, message})          │
│       last_fired[job.id] = minute_marker(now)          │
│                                                        │
│ 主循环每轮 LLM 前：                                     │
│   sched_msgs = scheduler.drain_due()                   │
│   if sched_msgs:                                       │
│     messages.append({"role":"user",                    │
│                      "content":"<scheduled_message>"}) │
└────────────────────────────────────────────────────────┘
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/cron_parser.py` | 🆕 新增 | 5-field cron 表达式解析 + `cron_match(expr, now) -> bool` |
| `agent/cron.py` | 🆕 新增 | `CronScheduler`（加载 jobs、daemon thread、drain_due、shutdown） |
| `agent/__init__.py` | ♻️ 改 | 主循环每轮 LLM 前 drain_due 注入 `<scheduled_message>` |
| `cli.py` | ♻️ 改 | RuntimeContext 持有 scheduler；shutdown |
| `config.py` | ♻️ 改 | `cron` 配置块 |
| `tests/test_cron_parser.py` | 🆕 新增 | parser 单元测试 |
| `tests/test_cron.py` | 🆕 新增 | CronScheduler 单元测试 |
| `tests/test_integration.py` | ♻️ 改 | 主循环注入 e2e |

---

## §2 数据结构

### jobs.json 格式

```json
{
  "jobs": [
    {
      "id": "daily_report",
      "cron": "0 9 * * *",
      "message": "请生成今日工作报告",
      "enabled": true,
      "catch_up": false
    },
    {
      "id": "weekly_backup",
      "cron": "0 2 * * 0",
      "message": "执行每周数据备份",
      "enabled": true
    }
  ]
}
```

字段：
- `id`（必需）：唯一标识，用于 last_fired 去重
- `cron`（必需）：标准 5-field 表达式
- `message`（必需）：触发时注入的 user 消息文本
- `enabled`（可选，默认 true）
- `catch_up`（可选，默认 false）：错过的运行是否补救

### CronJob dataclass

```python
@dataclass
class CronJob:
    id: str
    cron: str
    message: str
    enabled: bool = True
    catch_up: bool = False
```

---

## §3 cron 表达式解析（agent/cron_parser.py）

支持 5 个字段：`minute hour day_of_month month day_of_week`

每个字段支持的语法：
- `*`：任意值
- `N`：精确值（如 `5`）
- `*/N`：步进（如 `*/15` = 每 15 单位）
- `N-M`：范围（如 `9-17`）
- `N,M,K`：列表（如 `0,15,30,45`）
- 组合：`1-5,10-14`、`*/15,30`

### 接口

```python
def parse_field(expr: str, min_val: int, max_val: int) -> set[int]:
    """解析单字段为允许值的集合。"""

def cron_match(cron_expr: str, dt: datetime) -> bool:
    """检查 dt 是否匹配 cron 表达式（精确到分钟）。"""

# 字段范围
FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 0 = Sunday
}
```

### day_of_week vs day_of_month 的 OR 语义

标准 cron：当 `day_of_month` 和 `day_of_week` 都不是 `*` 时，匹配任一即触发（OR）。其他情况是 AND。本 parser 遵循此约定。

---

## §4 CronScheduler（agent/cron.py）

```python
class CronScheduler:
    def __init__(
        self,
        *,
        jobs_path: Path,
        poll_interval_seconds: float = 30.0,
        enabled: bool = True,
    ):
        self._jobs: list[CronJob] = []
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._last_fired: dict[str, str] = {}  # job_id -> minute_marker
        self._poll_interval = poll_interval_seconds
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load_jobs(jobs_path)

    def start(self) -> None:
        """启动后台 daemon thread。幂等。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="cron-scheduler"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止后台 thread。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def drain_due(self) -> list[dict]:
        """主循环每轮调。返回并清空到期通知。"""
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
        return notifications

    def reload(self, jobs_path: Path) -> int:
        """重新加载 jobs.json。返回加载数量。"""
        ...

    def shutdown(self) -> None:
        """同 stop。"""
        self.stop()

    # ---- 内部 ----
    def _load_jobs(self, jobs_path: Path) -> int: ...
    def _run_loop(self) -> None: ...
    def _tick(self, now: datetime) -> None: ...
    def _push_notification(self, job_id: str, message: str) -> None: ...
```

### `_run_loop` 实现

```python
def _run_loop(self):
    while not self._stop_event.is_set():
        try:
            self._tick(datetime.now())
        except Exception as e:
            logger.warning("cron tick 异常: %s", e)
        self._stop_event.wait(self._poll_interval)

def _tick(self, now: datetime):
    if not self._enabled:
        return
    minute_marker = now.strftime("%Y-%m-%d %H:%M")
    with self._lock:
        jobs_snapshot = list(self._jobs)
    for job in jobs_snapshot:
        if not job.enabled:
            continue
        try:
            if not cron_match(job.cron, now):
                continue
        except Exception as e:
            logger.warning("cron %s 表达式无效 '%s': %s", job.id, job.cron, e)
            continue
        with self._lock:
            if self._last_fired.get(job.id) == minute_marker:
                continue  # 同分钟已触发，防抖
            self._last_fired[job.id] = minute_marker
            self._notifications.append({
                "job_id": job.id,
                "message": job.message,
                "fired_at": now.isoformat(timespec="seconds"),
            })
```

---

## §5 主循环注入

**位置**：`agent/__init__.py:run_conversation` 顶部，紧邻 bg_task notifications drain。

```python
# === Phase 2c: cron scheduled messages ===
cron_messages = []
if self.cron_scheduler:
    try:
        cron_messages = self.cron_scheduler.drain_due()
    except Exception as e:
        logger.warning("cron drain_due 异常: %s", e)
        cron_messages = []
```

主循环内（同 bg_notifications 位置）注入：

```python
if cron_messages:
    sched_text = "\n".join(
        f"[Scheduled: {m['job_id']}] {m['message']}"
        for m in cron_messages
    )
    messages.append({
        "role": "user",
        "content": f"<scheduled_message>\n{sched_text}\n</scheduled_message>",
    })
    cron_messages = []  # 本轮注入后清空
```

**关键**：同 bg_notifications，**不进 conversation_history**。

---

## §6 改造 RuntimeContext + AIAgent

### RuntimeContext

```python
# cli.py:RuntimeContext.__init__
from agent.cron import CronScheduler
from pathlib import Path

cron_cfg = self.config.get("cron", {})
if cron_cfg.get("enabled", True):
    cron_path = cron_cfg.get("jobs_path") or (
        Path(self.home) / ".cron" / "jobs.json"
    )
    self.cron_scheduler = CronScheduler(
        jobs_path=cron_path,
        poll_interval_seconds=cron_cfg.get("poll_interval_seconds", 30.0),
        enabled=True,
    )
    self.cron_scheduler.start()
else:
    self.cron_scheduler = None
```

### AIAgent

```python
# agent/__init__.py:AIAgent.__init__
def __init__(self, ..., cron_scheduler=None, ...):
    self.cron_scheduler = cron_scheduler
```

### RuntimeContext.shutdown

```python
if self.cron_scheduler:
    self.cron_scheduler.shutdown()
```

---

## §7 配置

```python
# config.py:DEFAULT_CONFIG
"cron": {
    "enabled": True,                       # False 时整个 cron 关闭
    "jobs_path": None,                     # None → 默认 ~/.agent/.cron/jobs.json
    "poll_interval_seconds": 30.0,         # 后台线程 tick 间隔
},
```

---

## §8 失败处理

| 故障 | 行为 |
|---|---|
| jobs.json 不存在 | 静默启动，无 job 加载（log info） |
| jobs.json 格式错 | 启动时 fail-fast（raise ValueError） |
| 单个 job 缺 id/cron/message | 跳过 + log warning（不阻塞其他） |
| cron 表达式不合法 | 该 job 跳过（log warning）+ 不退出 |
| daemon thread tick 异常 | log + 继续下一个 tick |
| drain_due 异常 | log + 当轮跳过注入 |

---

## §9 并发安全

- `self._lock` 守护 `_jobs` / `_notifications` / `_last_fired`
- daemon thread 读 `_jobs` 前先 lock + snapshot
- 主循环 drain_due 用 lock + clear
- 没有 agent_lock（cron 只 push 队列，不直接调 LLM）

---

## §10 测试矩阵

### `tests/test_cron_parser.py`（10 个）

- `test_parse_star_returns_full_range`
- `test_parse_single_int`
- `test_parse_step`
- `test_parse_range`
- `test_parse_list`
- `test_parse_combined_range_list`
- `test_cron_match_basic`
- `test_cron_match_step`
- `test_cron_match_dom_or_dow`（验证 OR 语义）
- `test_cron_match_invalid_expr_raises`

### `tests/test_cron.py`（8 个）

- `test_load_jobs_valid`
- `test_load_jobs_missing_file_returns_empty`
- `test_load_jobs_malformed_raises`
- `test_load_jobs_skips_invalid_entries`
- `test_drain_due_returns_and_clears`
- `test_start_stop_idempotent`
- `test_tick_fires_matching_job_once_per_minute`
- `test_tick_skips_disabled_job`

### `tests/test_integration.py`（2 个）

- `test_aiagent_drains_cron_into_temporary_user_msg`
- `test_aiagent_no_cron_scheduler_backward_compat`

---

## §11 已知限制 / 非目标

1. **不支持秒级精度**：最小粒度 1 分钟（标准 cron）。
2. **不支持时区**：用本地时间。需要时区的用户在 OS 层调整。
3. **不实现 catch_up**（默认 false）。即使配置 `catch_up: true` 也忽略——本 phase 不实现补救逻辑。
4. **不支持 hot reload**：改 jobs.json 后需 `cron_reload` 工具或重启 agent。本 phase 不实现 reload 工具（留 Phase 2.x）。
5. **不持久化 last_fired**：agent 重启后所有 last_fired 清零——可能在同一分钟内重复触发（被 minute_marker 防抖兜底，最多多触发 1 次）。
6. **不实现 cron 管理 UI/工具**：用户直接编辑 jobs.json。Phase 2.x 可加 `cron_add` / `cron_remove` 工具。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代 | 理由 |
|---|---|---|---|
| 表达式解析 | 手写 100 行 | croniter / crontab | 零依赖；5-field 90% 用法足够 |
| 调度循环 | 后台 daemon thread | 主循环 poll | LLM 调用间隔不规则，daemon 保证精度 |
| 集成方式 | 主循环独立 drain | UserPromptSubmit hook | hook 只在用户消息时触发；cron 直接走主循环更通用 |
| 通知注入 | 临时 user 消息（不进 history） | 持久 user 消息 | 同 bg_task 模式，省 context |
| catch_up | 默认 false，本 phase 不实现 true | 实现完整 catch_up | YAGNI；多数用户不需要 |
| Hot reload | 不实现 | 文件 watcher / 工具 | YAGNI；用户重启或加 reload 工具是后续 |
| agent_lock | 不引入 | threading.Lock | cron 只 push，不调 LLM；主循环本身同步无 race |

## 附录 B: 与现有原则对齐

| 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰 | cron 是数据（jobs.json）+ 单 scheduler；非核心循环代码 |
| Prompt Caching 神圣 | 通知是临时 user 消息，不进 system prompt |
| 完全可逆 | 删 jobs.json 即清空；stop() 即停 |
| 用户意图优先 | enabled 开关；catch_up=false 默认 |
| 安全默认 | daemon thread 异常 fail-open；jobs.json 不存在静默 |
