"""Cron 调度器（cron=定时任务——到点自动执行，像闹钟）：后台小线程盯着任务表，到点就发通知。

在项目里的位置：下层用 agent/cron_parser.py 判断"到点没"，
上层被主循环（agent/__init__.py）每轮取走到期通知；任务列表存在 jobs.json。

工作方式：
- 后台线程每 30 秒（可配置）醒来查一次表
- 每次把所有启用（enabled）的任务过一遍，靠"表达式匹配 + 分钟标记"防止同一分钟重复触发
- 主循环每轮调 drain_due 把攒下的通知一次性取走
- 任务的上次触发时间会存盘（供重启后补跑判断用）
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
    """一条定时任务的配置（什么时候触发、触发时说什么）。

    字段大白话：
    - id：任务唯一编号
    - cron：cron 表达式（定闹钟的时间规则）
    - message：到点要通知的内容
    - enabled：开关；False=暂停这条任务但不删除
    - catch_up：补跑开关——True 时，程序重启后会补一次停机期间错过的触发
    - created_at：创建时间（ISO 格式字符串）；缺失时读取自动补当前时间
    - recurring：True=循环任务（到点每次都响）；False=一次性任务，响一次就自动停用（不删除）
    - last_fired_at：上次真正触发的时间（精确到分钟）；空=还没触发过
    """
    id: str
    cron: str
    message: str
    enabled: bool = True
    catch_up: bool = False  # 补跑开关：True 时启动会补跑错过的一次触发
    # === 任务生命周期字段 ===
    created_at: str = ""    # ISO 时间戳；缺失时 _parse_job 自动补当前时间
    recurring: bool = True  # False = 一次性，触发后自动停用（不删除，留档可查）
    # === 补跑相关字段 ===
    last_fired_at: str = ""  # 上次实际触发时间（ISO 格式精确到分钟）；空=从没触发过
    # 补跑扫描窗口上限（小时）：上次触发太久远就不补了，免得逐分钟扫描太费劲。
    # 默认 24 小时：用户重启间隔通常不到一天；隔更久的基本算废弃任务，不补。


# 默认值：补跑扫描窗口上限（小时），超过就不补
DEFAULT_CATCH_UP_WINDOW_HOURS = 24


def _to_local_naive(dt: datetime) -> datetime:
    """把解析出来的时间统一成 naive 本地时间再拿去做差。

    存档里的时间戳口径不统一：created_at 写的是带时区的 UTC，
    last_fired_at 写的是 naive 本地。aware 和 naive 直接相减会抛
    TypeError，所以比较前都先过这一道。
    """
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


class CronScheduler:
    """定时任务调度器本体。实例由 CLI 的 RuntimeContext 统一持有，再注入给 AIAgent 用。"""

    def __init__(
        self,
        *,
        jobs_path: Path,
        poll_interval_seconds: float = 30.0,
        enabled: bool = True,
        max_age_days: int = 7,  # 任务最多活几天，超龄自动停用
    ):
        """建一个调度器。

        参数：
        - jobs_path：任务表文件 jobs.json 的路径（必填）
        - poll_interval_seconds：后台线程多久查一次表，默认 30 秒
        - enabled：总开关；False=线程空转不触发任何任务
        - max_age_days：任务创建超过多少天就自动停用（防僵尸任务），默认 7

        返回：无（构造函数）。构造时会立刻从 jobs_path 加载任务表。
        """
        self._jobs: list = []
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._last_fired: dict = {}  # 记录每个任务最近触发的"分钟标记"（job_id -> "年-月-日 时:分"），用来同分钟去重
        self._poll_interval = poll_interval_seconds
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jobs_path = jobs_path
        # 超龄天数从构造参数读——
        # 正式运行时 RuntimeContext 从 config["cron"]["max_age_days"] 传入，
        # 测试可以直接构造时覆盖，不用改全局配置
        self._max_age_days = max_age_days
        self._load_jobs(jobs_path)

    # ---- 启停 ----
    def start(self) -> None:
        """启动后台扫描线程。重复调用不会起第二个线程（幂等）。

        参数：无。返回：无。
        """
        if self._thread and self._thread.is_alive():
            return
        # 起线程前先补跑停机期间错过的触发（只针对开了 catch_up 的任务）
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
        """启动时补跑：把停机期间错过的一次触发补上通知。

        停机期间可能错过触发（闹钟定的是每小时响、电脑关了一晚）：
        逐分钟扫停机时间段，第一个该触发的点就补一条通知。

        对每条"启用 + 开了补跑"的任务：
          - 从没触发过（last_fired_at 为空）→ 不补（没基线可对）
          - 上次触发距现在超过 window_hours 小时 → 不补（太久远，逐分钟扫描太费劲）
          - 在 [上次触发, 现在] 区间逐分钟扫，第一个命中的时间点就发补跑通知，然后停

        补跑只发一条（不重复轰炸），通知里带 catch_up=True 和真实触发区分开。
        扫描本身不更新 last_fired_at（留给下次正常扫描更新）。

        明确语义：每条任务每次启动只补 1 次——停机 8 小时
        错过 8 次每小时触发也只补 1 次（补跑是"提醒你错过了"，不是
        "把历史重放一遍"，防止通知风暴；完整错过区间用 missed_between
        字段标注）。

        参数：
        - now：当前时间
        - window_hours：补跑扫描窗口上限（小时），超窗不补

        返回：无（补的通知直接进通知队列）。
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
                last_fired = _to_local_naive(datetime.fromisoformat(job.last_fired_at))
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
                continue  # 上次触发时间在未来（多半是时钟漂移）→ 没法补，跳过

            # 逐分钟扫，两头都不含：从"上次触发+1分钟"到"现在-1分钟"——
            # 既不重复上次已触发那次，也不抢当前这分钟（留给正常扫描处理）
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
                    break  # 表达式写错了，这条任务不补
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
        # 注意：补跑后不更新 last_fired_at 存档（补跑不算"实际触发"，
        # 留给下次正常扫描更新）。

    def stop(self) -> None:
        """停掉后台线程（最多等它 5 秒）。重复调用无害（幂等）。参数无，返回无。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def shutdown(self) -> None:
        """和 stop() 一回事，给生命周期管理统一叫法用的别名。"""
        self.stop()

    # ---- 通知 drain ----
    def drain_due(self) -> list:
        """主循环每轮调一次：把攒下的到期通知全部取走并清空队列。

        参数：无。
        返回：通知列表（每条含 job_id/message/fired_at 等字段）。
        """
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
        return notifications

    # ---- reload ----
    def reload(self, jobs_path: Optional[Path] = None) -> int:
        """把任务表整个重读一遍（旧任务和通知全清掉重来）。

        参数：
        - jobs_path：从哪个文件读；不传就用构造时的老路径

        返回：这次加载进来的任务条数。
        """
        # 注意：reload 会清空去重标记和通知——文件被外部改过后用它刷新
        path = jobs_path or self._jobs_path
        with self._lock:
            self._jobs = []
            self._last_fired = {}
            self._notifications.clear()
        return self._load_jobs(path)

    # ---- 增删查（tools/cron_tool.py 把这些方法包装成工具，
    # 让 LLM 能自己管理定时任务）----
    def add_job(
        self,
        cron: str,
        message: str,
        *,
        catch_up: bool = False,
        job_id: Optional[str] = None,
        recurring: bool = True,  # 模板的一次性属性要透传到这里（False=一次性）
    ) -> CronJob:
        """新增一条定时任务（自动生成编号 + 立刻存盘）。

        先拿 cron_match 试解析一次表达式做校验——写错的直接抛 ValueError
        （错误不能被 try/except 吞掉——否则工具层转不成 invalid_cron_expr
        报错，用户看不到为什么失败）。
        显式传了 job_id 且已存在时也抛 ValueError（防止悄悄覆盖旧任务）。

        参数：
        - cron：cron 表达式（时间规则）
        - message：到点要通知的内容
        - catch_up：是否开启重启补跑
        - job_id：自定义任务编号；不传自动生成
        - recurring：True=循环任务；False=一次性（响一次自动停用）

        返回：新建的 CronJob 对象。
        """
        # 校验表达式：cron_match 遇到非法写法会抛 ValueError
        cron_match(cron, datetime.now())

        new_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        job = CronJob(
            id=new_id,
            cron=cron,
            message=message,
            catch_up=catch_up,
            recurring=recurring,
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
        """删掉一条任务（立刻存盘）。

        参数：
        - job_id：要删的任务编号

        返回：True=删掉了；False=本来就没有这条任务。
        """
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
        """列出所有任务（拷贝一份的快照，dict 格式，改它不影响内部状态）。

        参数：无。返回：任务 dict 列表。
        """
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
        """从 jobs.json 读任务表。

        文件不存在不算错（第一次用还没建过），安静返回 0。
        文件存在但格式不对（缺 jobs 字段等）抛 ValueError。

        参数：
        - jobs_path：任务表文件路径

        返回：加载进来的任务条数。
        """
        if not jobs_path.exists():
            logger.info("cron jobs.json 不存在 (%s)，无 job 加载", jobs_path)
            return 0
        text = jobs_path.read_text(encoding="utf-8")
        config = json.loads(text)  # JSON 本身坏了就直接抛出去，不吞
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
        """把一个 dict 转成 CronJob 对象，缺关键字段就丢弃。

        jobs.json 是给人也是给程序改的，字段可能缺——
        id/cron/message 三样缺任何一个都记条 warning 后丢弃。

        参数：
        - h：单个任务的原始 dict

        返回：CronJob 对象；不合格返回 None。
        """
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
        # === 字段缺省兜底（jobs.json 可能缺这些字段） ===
        # created_at 默认用 UTC 时间——不然跨时区/夏令时会让"任务年龄"跳来跳去
        created_at = h.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        recurring = h.get("recurring", True)
        # last_fired_at 缺省就当空（首次触发后才写上）
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
        """后台线程的主循环：每隔 poll_interval 秒醒来查一次表。

        单次查询出错只记日志不退出——调度线程挂了所有闹钟就全哑了。
        参数无，返回无。
        """
        # 超龄天数启动时从配置读好，循环里直接用
        max_age_days = self._max_age_days
        while not self._stop_event.is_set():
            if self._enabled:
                try:
                    self._tick(datetime.now(), max_age_days=max_age_days)
                except Exception as e:
                    logger.warning("cron tick 异常: %s", e)
            self._stop_event.wait(self._poll_interval)

    def _tick(self, now: datetime, *, max_age_days: int = 7):
        """一次扫描的核心逻辑：判断哪些任务到点该触发（设计成可直接单测，不查总开关）。

        干的事：
        - 超龄检查：创建超过 max_age_days 天的任务自动停用
          （created_at 缺失或解析失败就跳过这项检查，不误杀）
        - 一次性任务（recurring=False）触发后立刻自动停用
        - 只要有触发或停用变化，都会写回 jobs.json 存档

        参数：
        - now：拿哪个时间点做判断（测试可以传假时间）
        - max_age_days：任务最多活几天

        返回：无（触发的通知直接进队列）。
        """
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with self._lock:
            jobs_snapshot = list(self._jobs)

        expired_ids: set = set()
        fired_oneshot_ids: set = set()
        fired_any = False  # 只要有触发就得存档（更新 last_fired_at 给补跑用）

        for job in jobs_snapshot:
            if not job.enabled:
                continue

            # === 超龄检查，过期的僵尸任务自动停用 ===
            if job.created_at:
                try:
                    created = _to_local_naive(datetime.fromisoformat(job.created_at))
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
                    continue  # 这分钟已经触发过了，不重复
                self._last_fired[job.id] = minute_marker
                # 触发时间写进任务对象（后面统一存档，下次启动补跑靠它）
                job.last_fired_at = now.isoformat(timespec="minutes")
                self._notifications.append({
                    "job_id": job.id,
                    "message": job.message,
                    "fired_at": now.isoformat(timespec="seconds"),
                })
                fired_any = True

                # 一次性任务触发后立刻停用并存盘，不等这轮循环扫完——
                # 不然进程在这中间被杀，下次启动补跑会把它再触发一遍
                if not job.recurring:
                    job.enabled = False
                    self._persist_jobs_unlocked()
                    logger.info("cron 一次性 job %s 触发后立即 disable", job.id)

            # === 记下一次性任务（上面已立刻停用，
            # 这里只是加进集合，凑给循环末尾的日志统计用） ===
            if not job.recurring:
                fired_oneshot_ids.add(job.id)

        # === 把过期和一次性任务统一停用 + 存盘 ===
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
            # 没有要停用的但有触发 → 也要存盘（把新的 last_fired_at 写进去）
            with self._lock:
                self._persist_jobs_unlocked()

    def _persist_jobs_unlocked(self) -> None:
        """把内存里的任务表整个写回 jobs.json（先写临时文件再原子替换，写一半断电也不会留下半个坏文件）。

        规矩：调用前必须已经拿着 self._lock（方法名里的 _unlocked 就是提醒这个，
        锁外调用会数据打架）。
        写失败只记 warning 不抛错——不能因为存档失败把调度线程搞崩。

        参数：无。返回：无。
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
                        "last_fired_at": j.last_fired_at,  # 上次触发时间（补跑要用）
                    }
                    for j in self._jobs
                ]
            }
            content = json.dumps(data, ensure_ascii=False, indent=2)
            atomic_write_text(self._jobs_path, content)
        except Exception as e:
            logger.warning("cron jobs 持久化失败（不阻塞调度）: %s", e)
