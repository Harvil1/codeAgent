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
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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
    catch_up: bool = False  # P1-7: True 时启动会补跑错过的一次触发
    # === CronRecurringExpiry NEW ===
    created_at: str = ""    # ISO 时间戳；老 jobs.json 缺时 _parse_job 补 now
    recurring: bool = True  # False = 一次性，触发后自动 disable（不删除）
    # === P1-7 NEW ===
    last_fired_at: str = ""  # 上次实际触发时间（ISO timespec=minutes）；空=首次
    # catch_up 扫描窗口上限（小时），避免 last_fired_at 太久远时扫太多分钟
    # 默认 24h：超过的不补跑（用户重启间隔通常 < 24h；过长间隔视为废弃 job）


# P1-7: catch_up 扫描窗口上限（小时）
DEFAULT_CATCH_UP_WINDOW_HOURS = 24


class CronScheduler:
    """Cron 调度器。实例由 RuntimeContext 持有，注入 AIAgent。"""

    def __init__(
        self,
        *,
        jobs_path: Path,
        poll_interval_seconds: float = 30.0,
        enabled: bool = True,
        max_age_days: int = 7,  # === CronRecurringExpiry NEW ===
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
        # === CronRecurringExpiry NEW: 从构造参数读 max_age_days ===
        # RuntimeContext 从 config["cron"]["max_age_days"] 传入；测试可直接构造时覆盖
        self._max_age_days = max_age_days
        self._load_jobs(jobs_path)

    # ---- 启停 ----
    def start(self) -> None:
        """启动后台 thread。幂等。"""
        if self._thread and self._thread.is_alive():
            return
        # P1-7: 启动时先补跑错过的触发（catch_up=True 的 jobs）
        try:
            self._apply_catch_up(datetime.now())
        except Exception as e:
            logger.warning("catch_up 启动补跑失败（不阻塞调度）: %s", e)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="cron-scheduler"
        )
        self._thread.start()
        logger.info("CronScheduler 后台线程已启动 (poll=%ss)", self._poll_interval)

    def _apply_catch_up(
        self, now: datetime, *, window_hours: int = DEFAULT_CATCH_UP_WINDOW_HOURS,
    ):
        """P1-7: 启动时补跑错过的一次触发。

        对每个 enabled + catch_up=True 的 job：
          - last_fired_at 为空（首次）→ 跳过
          - last_fired_at 距 now > window_hours → 跳过（避免大量扫描）
          - 扫 [last_fired_at, now] 每分钟，第一个 cron_match 就 push 补跑通知 + break

        补跑只 push 一次（不重复），通知里标 catch_up=True 区分实时触发。
        扫描结果不影响 last_fired_at（下次正常 tick 才更新）。
        """
        from datetime import timedelta
        catch_up_count = 0
        with self._lock:
            jobs_snapshot = list(self._jobs)

        for job in jobs_snapshot:
            if not job.enabled or not job.catch_up:
                continue
            if not job.last_fired_at:
                continue
            try:
                last_fired = datetime.fromisoformat(job.last_fired_at)
            except (ValueError, TypeError) as e:
                logger.debug(
                    "catch_up: job %s last_fired_at 解析失败 %s（跳过）",
                    job.id, e,
                )
                continue

            elapsed_hours = (now - last_fired).total_seconds() / 3600
            if elapsed_hours > window_hours:
                logger.info(
                    "catch_up: job %s last_fired_at 距今 %.1fh 超过 %dh 上限，跳过",
                    job.id, elapsed_hours, window_hours,
                )
                continue
            if elapsed_hours <= 0:
                continue  # last_fired 在未来（时钟漂移）→ 跳过

            # 扫每分钟（不含端点：从 last_fired+1min 到 now-1min）
            # 即不重复 last_fired 那次，也不预触发 now 这分钟（让正常 tick 处理）
            scan_start = last_fired + timedelta(minutes=1)
            scan_end = now - timedelta(minutes=1)
            cursor = scan_start
            found = False
            while cursor <= scan_end:
                try:
                    if cron_match(job.cron, cursor):
                        found = True
                        break
                except ValueError:
                    break  # 表达式无效，跳过
                cursor += timedelta(minutes=1)

            if found:
                with self._lock:
                    self._notifications.append({
                        "job_id": job.id,
                        "message": job.message,
                        "fired_at": now.isoformat(timespec="seconds"),
                        "catch_up": True,
                        "missed_between": f"{scan_start.isoformat(timespec='minutes')} ~ {scan_end.isoformat(timespec='minutes')}",
                    })
                catch_up_count += 1
                logger.info(
                    "catch_up: job %s 补跑 1 次（窗口 %s ~ %s）",
                    job.id,
                    scan_start.isoformat(timespec="minutes"),
                    scan_end.isoformat(timespec="minutes"),
                )

        if catch_up_count:
            # 持久化更新（虽然 last_fired_at 没变，但通知里需要标记，最好不持久化
            # 因为 catch_up 不算"实际触发"。让下次正常 tick 更新 last_fired_at。）
            pass

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

    # ---- CRUD（CCAR12 Task 3：tools/cron_tool.py 包装，LLM 可自主管理定时任务）----
    def add_job(
        self,
        cron: str,
        message: str,
        *,
        catch_up: bool = False,
        job_id: Optional[str] = None,
    ) -> CronJob:
        """新增 job（自动生成 id + 持久化）。

        先用 cron_match 校验表达式（非法抛 ValueError——工具层捕获转
        invalid_cron_expr 错误，不走 try/except 吞掉）。
        job_id 显式传入且已存在时抛 ValueError（防覆盖）。
        """
        # 校验表达式：cron_match 对非法 expr 抛 ValueError
        cron_match(cron, datetime.now())

        new_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        job = CronJob(
            id=new_id,
            cron=cron,
            message=message,
            catch_up=catch_up,
            created_at=created_at,
        )
        with self._lock:
            if any(j.id == new_id for j in self._jobs):
                raise ValueError(f"cron job id 已存在: {new_id}")
            self._jobs.append(job)
            self._persist_jobs_unlocked()
        logger.info("cron 新增 job %s (%s)", new_id, cron)
        return job

    def remove_job(self, job_id: str) -> bool:
        """删除 job（持久化）。不存在返回 False。"""
        with self._lock:
            before = len(self._jobs)
            self._jobs = [j for j in self._jobs if j.id != job_id]
            removed = len(self._jobs) < before
            if removed:
                self._last_fired.pop(job_id, None)
                self._persist_jobs_unlocked()
        if removed:
            logger.info("cron 删除 job %s", job_id)
        return removed

    def list_jobs(self) -> list:
        """列出所有 job（快照，dict 形式）。"""
        with self._lock:
            jobs_snapshot = list(self._jobs)
        return [
            {
                "id": j.id,
                "cron": j.cron,
                "message": j.message,
                "enabled": j.enabled,
                "catch_up": j.catch_up,
                "recurring": j.recurring,
                "created_at": j.created_at,
            }
            for j in jobs_snapshot
        ]

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
        # === CronRecurringExpiry NEW: 老格式兼容 ===
        # X10 fix: created_at 默认用 UTC（避免跨时区/DST age 跳变）
        created_at = h.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        recurring = h.get("recurring", True)
        # P1-7: last_fired_at 兼容（老 jobs.json 缺时为空，首次 _tick 后才填）
        last_fired_at = h.get("last_fired_at", "")
        return CronJob(
            id=job_id,
            cron=cron,
            message=message,
            enabled=h.get("enabled", True),
            catch_up=h.get("catch_up", False),
            created_at=created_at,
            recurring=recurring,
            last_fired_at=last_fired_at,
        )

    def _run_loop(self):
        """后台 daemon thread：每 poll_interval 秒 tick 一次。"""
        # === CronRecurringExpiry NEW: 从配置读 max_age_days ===
        max_age_days = self._max_age_days
        while not self._stop_event.is_set():
            if self._enabled:
                try:
                    self._tick(datetime.now(), max_age_days=max_age_days)
                except Exception as e:
                    logger.warning("cron tick 异常: %s", e)
            self._stop_event.wait(self._poll_interval)

    def _tick(self, now: datetime, *, max_age_days: int = 7):
        """核心调度逻辑。可独立测试（不检查 _enabled）。

        NEW:
        - 7 天过期检查（created_at 缺失/解析失败时跳过检查）
        - non-recurring 触发后自动 disable
        - 任何 enable 变化都通过 _persist_jobs_unlocked 持久化
        """
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with self._lock:
            jobs_snapshot = list(self._jobs)

        expired_ids: set = set()
        fired_oneshot_ids: set = set()
        fired_any = False  # P1-7: 任何触发都需持久化 last_fired_at

        for job in jobs_snapshot:
            if not job.enabled:
                continue

            # === CronRecurringExpiry NEW: 7 天过期检查 ===
            if job.created_at:
                try:
                    created = datetime.fromisoformat(job.created_at)
                    if (now - created).days >= max_age_days:
                        logger.info(
                            "cron job %s 已超 %d 天，自动 disable",
                            job.id, max_age_days,
                        )
                        expired_ids.add(job.id)
                        continue
                except (ValueError, TypeError) as e:
                    logger.debug(
                        "created_at 解析失败 %s: %s（跳过过期检查）",
                        job.id, e,
                    )

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
                # P1-7: 持久化 last_fired_at（让下次启动 catch_up 能用）
                job.last_fired_at = now.isoformat(timespec="minutes")
                self._notifications.append({
                    "job_id": job.id,
                    "message": job.message,
                    "fired_at": now.isoformat(timespec="seconds"),
                })
                fired_any = True

                # X11 fix: 一次性 job 触发后立刻 disable + persist（不等循环末尾）
                # 避免进程被 kill 后下次启动 catch_up 重复触发
                if not job.recurring:
                    job.enabled = False
                    self._persist_jobs_unlocked()
                    logger.info("cron 一次性 job %s 触发后立即 disable", job.id)

            # === CronRecurringExpiry NEW: 记录一次性任务（已立刻 disable，仍 add 用于末尾 log 统计） ===
            if not job.recurring:
                fired_oneshot_ids.add(job.id)

        # === CronRecurringExpiry NEW: 统一 disable + persist ===
        to_disable = expired_ids | fired_oneshot_ids
        if to_disable:
            with self._lock:
                for job in self._jobs:
                    if job.id in to_disable:
                        job.enabled = False
                self._persist_jobs_unlocked()
            logger.info(
                "cron disable 了 %d 个 job（过期 %d + 一次性 %d）",
                len(to_disable), len(expired_ids), len(fired_oneshot_ids),
            )
        elif fired_any:
            # P1-7: 没有 disable 但有触发 → 也持久化（更新 last_fired_at）
            with self._lock:
                self._persist_jobs_unlocked()

    def _persist_jobs_unlocked(self) -> None:
        """把当前 _jobs 写回 jobs.json（原子替换）。

        调用方必须已持 _lock（方法名 _unlocked 提示）。
        失败时 log warning 不抛——不能让持久化失败杀掉调度线程。
        """
        from agent.atomic_io import atomic_write_text
        try:
            data = {
                "jobs": [
                    {
                        "id": j.id,
                        "cron": j.cron,
                        "message": j.message,
                        "enabled": j.enabled,
                        "catch_up": j.catch_up,
                        "created_at": j.created_at,
                        "recurring": j.recurring,
                        "last_fired_at": j.last_fired_at,  # P1-7
                    }
                    for j in self._jobs
                ]
            }
            content = json.dumps(data, ensure_ascii=False, indent=2)
            atomic_write_text(self._jobs_path, content)
        except Exception as e:
            logger.warning("cron jobs 持久化失败（不阻塞调度）: %s", e)
