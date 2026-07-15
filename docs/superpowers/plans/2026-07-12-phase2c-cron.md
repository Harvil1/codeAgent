# Phase 2c: Cron 调度系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 Cron 调度系统：后台 daemon 线程每 30s 扫 jobs.json，匹配到期的 job push 到队列，主循环每轮注入 `<scheduled_message>` 临时消息。手写 5-field cron parser（无依赖）。

**Architecture:** 新增 `agent/cron_parser.py`（5-field 解析）+ `agent/cron.py`（CronScheduler + daemon thread + drain_due）。注入主循环 + RuntimeContext + AIAgent。

**Tech Stack:** Python 3.11+ threading + datetime + deque + uv + pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-phase2c-cron-design.md`

## Global Constraints

- 文件 I/O 必须 `encoding="utf-8"`
- 用 `uv`
- 中文注释/commit；英文标识符
- 不要 import 用不到的模块
- daemon thread 必须 `daemon=True`（agent 退出自动清理）
- 测试：`uv run pytest tests/<file>.py -v`
- 项目当前 464 测试不得回归
- Subagent 不 commit；controller 统一 commit

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/cron_parser.py` | T1 新增 | `parse_field` + `cron_match` |
| `tests/test_cron_parser.py` | T1 新增 | 10 个 parser 测试 |
| `agent/cron.py` | T2 新增 | `CronJob` + `CronScheduler` |
| `tests/test_cron.py` | T2 新增 | 8 个 scheduler 测试 |
| `config.py` | T3 改 | `cron` 配置块 |
| `agent/__init__.py` | T4 改 | AIAgent 接 `cron_scheduler=None` + 主循环 drain_due |
| `cli.py` | T5 改 | RuntimeContext 实例化 + start/shutdown |
| `tests/test_integration.py` | T6 改 | 主循环注入 e2e |

---

## Task 1: agent/cron_parser.py

**Files:**
- Create: `agent/cron_parser.py`
- Create: `tests/test_cron_parser.py`

**Interfaces:**
- Consumes: 无
- Produces: `parse_field(expr, min_val, max_val) -> set[int]`、`cron_match(cron_expr, dt) -> bool`、`FIELD_RANGES`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cron_parser.py
"""cron 表达式解析器测试。"""
from datetime import datetime

import pytest

from agent.cron_parser import parse_field, cron_match


# ---------------------------------------------------------------------------
# parse_field
# ---------------------------------------------------------------------------

def test_parse_star_returns_full_range():
    assert parse_field("*", 0, 5) == {0, 1, 2, 3, 4, 5}
    assert parse_field("*", 1, 3) == {1, 2, 3}


def test_parse_single_int():
    assert parse_field("5", 0, 59) == {5}


def test_parse_step():
    assert parse_field("*/15", 0, 59) == {0, 15, 30, 45}


def test_parse_range():
    assert parse_field("9-17", 0, 23) == {9, 10, 11, 12, 13, 14, 15, 16, 17}


def test_parse_list():
    assert parse_field("0,15,30,45", 0, 59) == {0, 15, 30, 45}


def test_parse_combined_range_list():
    assert parse_field("1-3,10-12", 0, 23) == {1, 2, 3, 10, 11, 12}


def test_parse_step_with_offset():
    """'2-10/2' = 从 2 到 10 步进 2。"""
    assert parse_field("2-10/2", 0, 59) == {2, 4, 6, 8, 10}


def test_parse_invalid_int_raises():
    with pytest.raises(ValueError):
        parse_field("abc", 0, 59)


def test_parse_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_field("60", 0, 59)


# ---------------------------------------------------------------------------
# cron_match
# ---------------------------------------------------------------------------

def test_cron_match_every_minute():
    """'* * * * *' 匹配任意时间。"""
    assert cron_match("* * * * *", datetime(2026, 7, 12, 15, 30)) is True


def test_cron_match_specific_minute():
    """'30 * * * *' 只匹配 minute=30。"""
    assert cron_match("30 * * * *", datetime(2026, 7, 12, 15, 30)) is True
    assert cron_match("30 * * * *", datetime(2026, 7, 12, 15, 31)) is False


def test_cron_match_step_minute():
    """'*/15 * * * *' 匹配 0/15/30/45 分。"""
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 0)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 15)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 45)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 7)) is False


def test_cron_match_daily_at_9am():
    """'0 9 * * *' 匹配每天 9:00。"""
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 9, 0)) is True
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 10, 0)) is False
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 9, 30)) is False


def test_cron_match_dom_or_dow():
    """当 day_of_month 和 day_of_week 都不是 '*'，匹配任一即触发（OR 语义）。"""
    # 2026-07-12 是周日（weekday 6 in Python is Sunday=6 → 我们的 day_of_week 0=Sunday）
    # 在 cron_expr "0 0 1 * 0" 中：
    #   day_of_month=1 (每月 1 号)
    #   day_of_week=0 (每周日)
    # 2026-07-12 是周日 → day_of_week 匹配 → 整体匹配（即使 dom != 1）
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 12, 0, 0)) is True
    # 2026-07-01 是周三，但是 dom=1 → 匹配
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 1, 0, 0)) is True
    # 2026-07-15 是周三，dom != 1 且 dow != 0 → 不匹配
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 15, 0, 0)) is False


def test_cron_match_dom_and_dow_when_one_is_star():
    """当 day_of_month='*' 而 day_of_week='0'，AND 语义（其他字段全 AND）。"""
    # 2026-07-12 是周日 → dow=0 匹配 → 整体匹配
    assert cron_match("0 0 * * 0", datetime(2026, 7, 12, 0, 0)) is True
    # 2026-07-13 是周一 → dow != 0 → 不匹配
    assert cron_match("0 0 * * 0", datetime(2026, 7, 13, 0, 0)) is False


def test_cron_match_too_few_fields_raises():
    with pytest.raises(ValueError):
        cron_match("* * * *", datetime.now())


def test_cron_match_invalid_expr_raises():
    with pytest.raises(ValueError):
        cron_match("60 * * * *", datetime.now())  # 60 越界
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cron_parser.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# agent/cron_parser.py
"""5-field cron 表达式解析器。

字段顺序：minute hour day_of_month month day_of_week
每字段语法：* | N | */N | N-M | N,M,K | N-M/S
"""

import logging
from datetime import datetime
from typing import Set

logger = logging.getLogger(__name__)

FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 0=Sunday, 6=Saturday
}


def parse_field(expr: str, min_val: int, max_val: int) -> Set[int]:
    """解析单字段为允许值的集合。

    支持：'*'、'N'、'*/N'、'N-M'、'N,M,K'、'N-M/S'
    Raises ValueError on invalid syntax or out-of-range values.
    """
    if not expr:
        raise ValueError("empty field")

    result: Set[int] = set()
    for part in expr.split(","):
        result.update(_parse_part(part.strip(), min_val, max_val))
    return result


def _parse_part(part: str, min_val: int, max_val: int) -> Set[int]:
    """解析逗号分隔的单段。"""
    # 处理 step：N-M/S 或 */S
    step = 1
    if "/" in part:
        range_part, step_str = part.split("/", 1)
        try:
            step = int(step_str)
        except ValueError:
            raise ValueError(f"invalid step '{step_str}' in part '{part}'")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        part = range_part

    if part == "*":
        start, end = min_val, max_val
    elif "-" in part:
        bounds = part.split("-", 1)
        try:
            start = int(bounds[0])
            end = int(bounds[1])
        except ValueError:
            raise ValueError(f"invalid range '{part}'")
    else:
        try:
            v = int(part)
        except ValueError:
            raise ValueError(f"invalid value '{part}'")
        if "/" in part or step > 1:
            # '5/N' 形式：从 5 到 max_val，步进 N
            start, end = v, max_val
        else:
            _check_range(v, min_val, max_val, part)
            return {v}

    _check_range(start, min_val, max_val, part)
    _check_range(end, min_val, max_val, part)
    if start > end:
        raise ValueError(f"range start > end in '{part}'")

    return set(range(start, end + 1, step))


def _check_range(val: int, min_val: int, max_val: int, raw: str):
    if val < min_val or val > max_val:
        raise ValueError(f"value {val} out of range [{min_val}, {max_val}] in '{raw}'")


def cron_match(cron_expr: str, dt: datetime) -> bool:
    """检查 dt 是否匹配 cron 表达式。

    Raises ValueError 如果表达式格式错误。
    """
    fields = cron_expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron expression must have 5 fields, got {len(fields)}: '{cron_expr}'"
        )

    minute_set = parse_field(fields[0], *FIELD_RANGES["minute"])
    hour_set = parse_field(fields[1], *FIELD_RANGES["hour"])
    dom_set = parse_field(fields[2], *FIELD_RANGES["day_of_month"])
    month_set = parse_field(fields[3], *FIELD_RANGES["month"])
    dow_set = parse_field(fields[4], *FIELD_RANGES["day_of_week"])

    if dt.minute not in minute_set:
        return False
    if dt.hour not in hour_set:
        return False
    if dt.month not in month_set:
        return False

    # day_of_month 和 day_of_week 的特殊 OR 语义：
    # 当两者都不是 '*' 时，匹配任一即触发
    dom_is_star = fields[2] == "*"
    dow_is_star = fields[4] == "*"

    # Python weekday(): Monday=0 ... Sunday=6
    # cron day_of_week: Sunday=0 ... Saturday=6
    cron_dow = (dt.weekday() + 1) % 7

    dom_match = dt.day in dom_set
    dow_match = cron_dow in dow_set

    if not dom_is_star and not dow_is_star:
        return dom_match or dow_match
    elif not dom_is_star:
        return dom_match
    elif not dow_is_star:
        return dow_match
    else:
        return True  # 都是 '*'，必然匹配
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cron_parser.py -v`
Expected: PASS（约 17 tests，包含 parse_field 9 + cron_match 8）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 2: agent/cron.py CronScheduler

**Files:**
- Create: `agent/cron.py`
- Create: `tests/test_cron.py`

**Interfaces:**
- Consumes: `cron_match` from `agent.cron_parser`（T1）
- Produces: `CronJob` dataclass + `CronScheduler` 类（load_jobs / start / stop / drain_due / reload / shutdown / _tick）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cron.py
"""CronScheduler 测试。"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from agent.cron import CronScheduler, CronJob
from agent.cron_parser import cron_match


def _write_jobs(path: Path, jobs: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False), encoding="utf-8")


def test_load_jobs_valid(tmp_path: Path):
    p = tmp_path / "jobs.json"
    _write_jobs(p, [
        {"id": "j1", "cron": "*/5 * * * *", "message": "hello"},
        {"id": "j2", "cron": "0 9 * * *", "message": "9am", "enabled": False},
    ])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 2
    assert sched._jobs[0].id == "j1"
    assert sched._jobs[1].enabled is False


def test_load_jobs_missing_file_starts_empty(tmp_path: Path):
    """文件不存在不抛，jobs 为空。"""
    sched = CronScheduler(jobs_path=tmp_path / "nonexistent.json", enabled=False)
    assert sched._jobs == []


def test_load_jobs_malformed_raises(tmp_path: Path):
    p = tmp_path / "jobs.json"
    p.write_text("not a json {{{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        CronScheduler(jobs_path=p, enabled=False)


def test_load_jobs_skips_invalid_entries(tmp_path: Path, caplog):
    """缺关键字段的 job 跳过 + warning。"""
    p = tmp_path / "jobs.json"
    _write_jobs(p, [
        {"id": "good", "cron": "* * * * *", "message": "ok"},
        {"id": "no_cron", "message": "misses cron"},
        {"no_message": "x"},
    ])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 1
    assert sched._jobs[0].id == "good"


def test_drain_due_returns_and_clears(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._notifications.append({"job_id": "test", "message": "hi"})
    result = sched.drain_due()
    assert len(result) == 1
    assert result[0]["job_id"] == "test"
    assert sched.drain_due() == []


def test_start_stop_idempotent(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched.start()
    sched.start()  # 幂等
    assert sched._thread is not None
    sched.stop()
    sched.stop()  # 幂等


def test_tick_fires_matching_job(tmp_path: Path):
    """manual tick 一个匹配的 job，notification 被 push。"""
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    notifs = sched.drain_due()
    assert len(notifs) == 1
    assert notifs[0]["job_id"] == "j1"
    assert notifs[0]["message"] == "hi"


def test_tick_skips_disabled_job(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi", enabled=False)]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    assert sched.drain_due() == []


def test_tick_dedupes_within_same_minute(tmp_path: Path):
    """同一分钟内多次 tick 只触发一次。"""
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    sched._tick(datetime(2026, 7, 12, 15, 30))
    notifs = sched.drain_due()
    assert len(notifs) == 1


def test_tick_fires_again_in_next_minute(tmp_path: Path):
    sched = CronScheduler(jobs_path=tmp_path / "x.json", enabled=False)
    sched._jobs = [CronJob(id="j1", cron="* * * * *", message="hi")]
    sched._tick(datetime(2026, 7, 12, 15, 30))
    sched._tick(datetime(2026, 7, 12, 15, 31))
    notifs = sched.drain_due()
    assert len(notifs) == 2


def test_reload_replaces_jobs(tmp_path: Path):
    p = tmp_path / "jobs.json"
    _write_jobs(p, [{"id": "j1", "cron": "* * * * *", "message": "old"}])
    sched = CronScheduler(jobs_path=p, enabled=False)
    assert len(sched._jobs) == 1
    _write_jobs(p, [
        {"id": "j1", "cron": "* * * * *", "message": "new"},
        {"id": "j2", "cron": "0 0 * * *", "message": "daily"},
    ])
    sched.reload(p)
    assert len(sched._jobs) == 2
    assert sched._jobs[0].message == "new"


def test_daemon_thread_starts_and_stops(tmp_path: Path):
    """集成：start + sleep + stop 不抛。"""
    sched = CronScheduler(
        jobs_path=tmp_path / "x.json",
        enabled=False,
        poll_interval_seconds=0.2,
    )
    sched.start()
    time.sleep(0.5)
    sched.stop()
    # thread 应已结束
    assert not sched._thread.is_alive()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cron.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# agent/cron.py
"""CronScheduler：后台 daemon 线程扫 jobs.json + 到期 push 通知。

设计：
- 后台线程 30s tick 一次（可配置）
- 每次扫所有 enabled jobs，cron_match + minute_marker 去重
- 主循环每轮 drain_due 清空通知
- 不持久化 last_fired（重启清零）
"""
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.cron_parser import cron_match

logger = logging.getLogger(__name__)


@dataclass
class CronJob:
    """单个 cron 任务配置。"""
    id: str
    cron: str
    message: str
    enabled: bool = True
    catch_up: bool = False  # 本 phase 不实现 catch_up，仅占位


class CronScheduler:
    """Cron 调度器。实例由 RuntimeContext 持有，注入 AIAgent。"""

    def __init__(
        self,
        *,
        jobs_path: Path,
        poll_interval_seconds: float = 30.0,
        enabled: bool = True,
    ):
        self._jobs: list = []
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._last_fired: dict = {}  # job_id -> minute_marker
        self._poll_interval = poll_interval_seconds
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jobs_path = jobs_path
        if enabled:
            self._load_jobs(jobs_path)

    # ---- 启停 ----
    def start(self) -> None:
        """启动后台 thread。幂等。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="cron-scheduler"
        )
        self._thread.start()
        logger.info("CronScheduler 后台线程已启动 (poll=%ss)", self._poll_interval)

    def stop(self) -> None:
        """停止后台 thread。幂等。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def shutdown(self) -> None:
        """同 stop。"""
        self.stop()

    # ---- 通知 drain ----
    def drain_due(self) -> list:
        """主循环每轮调。返回并清空通知。"""
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
        return notifications

    # ---- reload ----
    def reload(self, jobs_path: Optional[Path] = None) -> int:
        """重新加载 jobs.json。返回加载数量。"""
        path = jobs_path or self._jobs_path
        with self._lock:
            self._jobs = []
            self._last_fired = {}
            self._notifications.clear()
        return self._load_jobs(path)

    # ---- 内部 ----
    def _load_jobs(self, jobs_path: Path) -> int:
        """从 jobs.json 加载。文件不存在 = 静默返回 0。"""
        if not jobs_path.exists():
            logger.info("cron jobs.json 不存在 (%s)，无 job 加载", jobs_path)
            return 0
        text = jobs_path.read_text(encoding="utf-8")
        config = json.loads(text)  # 解析失败抛
        if "jobs" not in config:
            raise ValueError(f"jobs.json 缺少 'jobs' 字段: {jobs_path}")
        raw_jobs = config["jobs"]
        if not isinstance(raw_jobs, list):
            raise ValueError(
                f"jobs.json 'jobs' 必须是 list，实际 {type(raw_jobs).__name__}"
            )

        jobs = []
        for h in raw_jobs:
            job = self._parse_job(h)
            if job is not None:
                jobs.append(job)
        with self._lock:
            self._jobs = jobs
        logger.info("cron 加载了 %d 个 jobs", len(jobs))
        return len(jobs)

    def _parse_job(self, h: dict) -> Optional[CronJob]:
        """解析单个 job dict。缺关键字段返回 None + log warning。"""
        job_id = h.get("id")
        cron = h.get("cron")
        message = h.get("message")
        if not job_id:
            logger.warning("cron job 缺 id 字段，跳过: %s", h)
            return None
        if not cron:
            logger.warning("cron job '%s' 缺 cron 字段，跳过", job_id)
            return None
        if not message:
            logger.warning("cron job '%s' 缺 message 字段，跳过", job_id)
            return None
        return CronJob(
            id=job_id,
            cron=cron,
            message=message,
            enabled=h.get("enabled", True),
            catch_up=h.get("catch_up", False),
        )

    def _run_loop(self):
        """后台 daemon thread：每 poll_interval 秒 tick 一次。"""
        while not self._stop_event.is_set():
            try:
                self._tick(datetime.now())
            except Exception as e:
                logger.warning("cron tick 异常: %s", e)
            self._stop_event.wait(self._poll_interval)

    def _tick(self, now: datetime):
        """核心调度逻辑。可独立测试。"""
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
            except ValueError as e:
                logger.warning("cron %s 表达式无效 '%s': %s", job.id, job.cron, e)
                continue
            with self._lock:
                if self._last_fired.get(job.id) == minute_marker:
                    continue  # 同分钟去重
                self._last_fired[job.id] = minute_marker
                self._notifications.append({
                    "job_id": job.id,
                    "message": job.message,
                    "fired_at": now.isoformat(timespec="seconds"),
                })
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cron.py -v`
Expected: PASS（约 11 tests）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 3: config.py cron 块

**Files:**
- Modify: `config.py:DEFAULT_CONFIG`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `DEFAULT_CONFIG["cron"]` dict

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py
def test_default_config_has_cron_block():
    from config import DEFAULT_CONFIG
    c = DEFAULT_CONFIG["cron"]
    for key in ("enabled", "jobs_path", "poll_interval_seconds"):
        assert key in c, f"缺 {key}"


def test_default_config_cron_defaults():
    from config import DEFAULT_CONFIG
    c = DEFAULT_CONFIG["cron"]
    assert c["enabled"] is True
    assert c["jobs_path"] is None
    assert c["poll_interval_seconds"] == 30.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_default_config_has_cron_block -v`
Expected: FAIL

- [ ] **Step 3: 改 config.py**

在 `bg_task` 块之后加：

```python
    # Cron 调度（Phase 2c）
    "cron": {
        "enabled": True,                        # False 时整个 cron 关闭
        "jobs_path": None,                      # None → 默认 ~/.agent/.cron/jobs.json
        "poll_interval_seconds": 30.0,          # 后台线程 tick 间隔
    },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 4: agent/__init__.py 主循环集成

**Files:**
- Modify: `agent/__init__.py:AIAgent.__init__` + `run_conversation`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: `CronScheduler.drain_due()`（T2）
- Produces: AIAgent 新增 `cron_scheduler=None` kwarg；主循环注入 `<scheduled_message>`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_aiagent_accepts_cron_scheduler_kwarg():
    agent = _make_test_agent()
    assert agent.cron_scheduler is None


def test_aiagent_drains_cron_into_temporary_user_msg(tmp_path):
    """cron drain_due 返回的消息作为 <scheduled_message> 临时 user 消息注入。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.cron import CronScheduler

    sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
    # 手动 push 一条通知
    sched._notifications.append({
        "job_id": "j1",
        "message": "time to check backups",
        "fired_at": "2026-07-12T15:30:00",
    })

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        cron_scheduler=sched,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hi")
    # conversation_history 不该含 scheduled_message
    for msg in agent.conversation_history:
        assert "<scheduled_message>" not in msg.get("content", "")
    sched.shutdown()


def test_aiagent_no_cron_scheduler_backward_compat(tmp_path):
    """cron_scheduler=None 时主循环不抛。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_aiagent_accepts_cron_scheduler_kwarg -v`
Expected: FAIL

- [ ] **Step 3: 改 agent/__init__.py**

`AIAgent.__init__` 加 `cron_scheduler=None`：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        hooks_registry=None,
        bg_manager=None,
        cron_scheduler=None,  # === NEW Phase 2c ===
    ):
        # ...
        self.cron_scheduler = cron_scheduler
```

`run_conversation` 顶部 drain（紧邻 bg_manager drain 之后）：

```python
        # === Phase 2b: bg_task notifications ===
        bg_notifications = []
        if self.bg_manager:
            try:
                bg_notifications = self.bg_manager.drain_notifications()
            except Exception as e:
                logger.warning("drain_notifications 异常: %s", e)
                bg_notifications = []

        # === NEW Phase 2c: cron scheduled messages ===
        cron_messages = []
        if self.cron_scheduler:
            try:
                cron_messages = self.cron_scheduler.drain_due()
            except Exception as e:
                logger.warning("cron drain_due 异常: %s", e)
                cron_messages = []
```

主循环内 messages 组装后注入（紧邻 bg_notifications 注入之后）：

```python
            # === Phase 2c: cron scheduled messages 临时注入 ===
            if cron_messages:
                sched_text = "\n".join(
                    f"[Scheduled: {m['job_id']}] {m['message']}"
                    for m in cron_messages
                )
                messages.append({
                    "role": "user",
                    "content": f"<scheduled_message>\n{sched_text}\n</scheduled_message>",
                })
                cron_messages = []
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -k cron_scheduler -v`
Expected: PASS（3 个新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 5: cli.py RuntimeContext 注入 + start/shutdown

**Files:**
- Modify: `cli.py:RuntimeContext`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: T2 + T3 + T4
- Produces: RuntimeContext.cron_scheduler；启动时 start；退出时 shutdown

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_runtime_context_has_cron_scheduler():
    from cli import RuntimeContext
    from agent.cron import CronScheduler
    ctx = RuntimeContext.__new__(RuntimeContext)
    sched = CronScheduler(jobs_path=Path("/tmp/x.json"), enabled=False)
    ctx.cron_scheduler = sched
    assert isinstance(ctx.cron_scheduler, CronScheduler)
    sched.shutdown()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_runtime_context_has_cron_scheduler -v`
Expected: FAIL

- [ ] **Step 3: 改 cli.py**

在 `RuntimeContext.__init__` 加（紧邻 bg_manager 实例化之后）：

```python
        # === NEW Phase 2c: Cron 调度器 ===
        from agent.cron import CronScheduler
        cron_cfg = self.config.get("cron", {})
        if cron_cfg.get("enabled", True):
            from pathlib import Path
            cron_path = cron_cfg.get("jobs_path")
            if cron_path is None:
                cron_path = Path(self.home) / ".cron" / "jobs.json"
            try:
                self.cron_scheduler = CronScheduler(
                    jobs_path=Path(cron_path),
                    poll_interval_seconds=cron_cfg.get("poll_interval_seconds", 30.0),
                    enabled=True,
                )
                self.cron_scheduler.start()
            except Exception as e:
                logger.error("CronScheduler 启动失败: %s", e)
                self.cron_scheduler = None
        else:
            self.cron_scheduler = None
```

在 `_create_agent`（或 AIAgent 构造点）传 `cron_scheduler=self.cron_scheduler`。

在 `RuntimeContext.shutdown` 加：

```python
        if hasattr(self, "cron_scheduler") and self.cron_scheduler:
            try:
                self.cron_scheduler.shutdown()
            except Exception as e:
                logger.warning("cron_scheduler shutdown 失败: %s", e)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 6: 端到端集成测试

**Files:**
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: 所有前置任务
- Produces: 完整 e2e 验证

- [ ] **Step 1: 写 e2e 测试**

```python
# 追加到 tests/test_integration.py
def test_e2e_cron_full_lifecycle(tmp_path):
    """端到端：jobs.json 配置 → scheduler tick → drain_due → 主循环注入 <scheduled_message>。

    流程：
    1. 写一个匹配当前时间的 jobs.json
    2. 构造 AIAgent + cron_scheduler
    3. 手动调 _tick(now) 触发
    4. agent.run_conversation() drain_due → 验证 LLM 看到 <scheduled_message>
    """
    import json
    from datetime import datetime
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.cron import CronScheduler

    jobs_path = tmp_path / ".cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(json.dumps({
        "jobs": [
            {"id": "test_job", "cron": "* * * * *", "message": "cron test message"},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    sched = CronScheduler(jobs_path=jobs_path, enabled=True, poll_interval_seconds=999)
    # 手动触发一次 tick
    sched._tick(datetime.now())

    captured_messages = []
    def capture_llm_call(msgs, **kw):
        captured_messages.append(list(msgs))  # snapshot
        resp = MagicMock()
        resp.choices = [MagicMock(
            message=MagicMock(content="ok", tool_calls=None),
            finish_reason="stop",
        )]
        return resp

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        cron_scheduler=sched,
    )
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions.side_effect = capture_llm_call

    agent.run_conversation("check")
    sched.shutdown()

    # 第 1 次 LLM 调用的 messages 应含 <scheduled_message>
    assert len(captured_messages) >= 1
    first_msgs = captured_messages[0]
    contents = [m.get("content", "") for m in first_msgs]
    assert any("<scheduled_message>" in c for c in contents), \
        f"应当含 <scheduled_message>，实际: {contents}"
    assert any("cron test message" in c for c in contents)
```

- [ ] **Step 2: 跑测试**

Run: `uv run pytest tests/test_integration.py::test_e2e_cron_full_lifecycle -v`
Expected: PASS

- [ ] **Step 3: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 + 新增 cron tests，0 回归）

- [ ] **Step 4: 不 commit**

---

## Self-Review

**Spec 覆盖检查**：
- ✅ §1 架构 → T1-T6
- ✅ §2 jobs.json + CronJob dataclass → T2
- ✅ §3 cron_parser 5-field → T1（含 step / range / list / dom-or-dow 语义）
- ✅ §4 CronScheduler → T2
- ✅ §5 主循环注入 → T4
- ✅ §6 RuntimeContext + AIAgent → T4, T5
- ✅ §7 config → T3
- ✅ §8 失败处理（文件不存在/malformed/缺字段/表达式错/tick 异常/drain 异常）→ T1/T2 测试覆盖
- ✅ §9 并发安全（lock 守护 _jobs/_notifications/_last_fired）→ T2 实现
- ✅ §10 测试矩阵 → T1 (17) + T2 (11) + T4 (3) + T6 (1)
- ✅ §11 已知限制（不实现 catch_up / hot reload / 时区 / 持久化 last_fired）→ 全遵守

**Placeholder 扫描**：无 TBD/TODO。

**类型一致性**：
- `cron_match(expr, dt) -> bool` 在 T1 定义，T2 调用 ✓
- `CronJob` 字段在 T2 定义，测试中 dict 字面量一致 ✓
- `CronScheduler.drain_due()` 在 T2 定义，T4/T6 调用 ✓
- `cron_scheduler=None` kwarg 链 T4 → T5 → AIAgent ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase2c-cron.md`.

按用户授权直接进 SDD。
