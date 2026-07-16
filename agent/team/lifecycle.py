"""AutonomousLifecycle: WORK/IDLE/SHUTDOWN 三态状态机。

WORK: 调 work_fn(task) 跑一轮
IDLE: 轮询 inbox + unclaimed tasks；拿到新工作回 WORK；超时进 SHUTDOWN
SHUTDOWN: 调 on_shutdown_fn 后退出
"""
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STATE_WORK = "work"
STATE_IDLE = "idle"
STATE_SHUTDOWN = "shutdown"


class AutonomousLifecycle:
    """三态生命周期。"""

    def __init__(
        self,
        *,
        work_fn: Callable[[str], str],
        poll_inbox_fn: Callable[[], list],
        poll_tasks_fn: Callable[[], list],
        claim_task_fn: Callable[[str], bool],
        on_shutdown_fn: Optional[Callable] = None,
        idle_timeout: float = 60.0,
        poll_interval: float = 5.0,
    ):
        self._work_fn = work_fn
        self._poll_inbox_fn = poll_inbox_fn
        self._poll_tasks_fn = poll_tasks_fn
        self._claim_task_fn = claim_task_fn
        self._on_shutdown_fn = on_shutdown_fn
        self._idle_timeout = idle_timeout
        self._poll_interval = poll_interval
        self._state = STATE_WORK
        self._idle_started_at: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def run(self, *, initial_task: str) -> None:
        """运行生命周期直到 SHUTDOWN。"""
        current_task = initial_task
        while True:
            # WORK
            self._state = STATE_WORK
            try:
                self._work_fn(current_task)
            except Exception as e:
                logger.exception("WORK 异常: %s", e)

            # IDLE
            self._state = STATE_IDLE
            self._idle_started_at = time.time()
            next_task = None
            while self._state == STATE_IDLE:
                if self._idle_timed_out():
                    logger.info("IDLE 超时（%.1fs），进入 SHUTDOWN",
                                self._idle_timeout)
                    break
                next_task = self._try_get_work()
                if next_task is not None:
                    break
                time.sleep(self._poll_interval)

            if next_task is None:
                # IDLE 超时
                break
            current_task = next_task

        # SHUTDOWN
        self._state = STATE_SHUTDOWN
        if self._on_shutdown_fn:
            try:
                self._on_shutdown_fn()
            except Exception:
                logger.exception("on_shutdown 异常")

    def _idle_timed_out(self) -> bool:
        if self._idle_started_at is None:
            return False
        return (time.time() - self._idle_started_at) >= self._idle_timeout

    def _try_get_work(self) -> Optional[str]:
        """检查 inbox + unclaimed tasks。返回拿到的 task prompt 或 None。"""
        # 1. 先看 inbox
        try:
            msgs = self._poll_inbox_fn()
            if msgs:
                return msgs[0].content
        except Exception as e:
            logger.warning("poll_inbox 异常: %s", e)

        # 2. 看 unclaimed tasks
        try:
            ready = self._poll_tasks_fn()
            for t in ready:
                if self._claim_task_fn(t["id"]):
                    return t.get("subject") or t.get("description") or t["id"]
        except Exception as e:
            logger.warning("poll_tasks 异常: %s", e)

        return None
