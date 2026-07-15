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
            if self._enabled:
                try:
                    self._tick(datetime.now())
                except Exception as e:
                    logger.warning("cron tick 异常: %s", e)
            self._stop_event.wait(self._poll_interval)

    def _tick(self, now: datetime):
        """核心调度逻辑。可独立测试（不检查 _enabled）。"""
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
