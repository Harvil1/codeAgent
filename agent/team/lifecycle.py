"""自主工人的三态生命周期：干活（WORK）/ 等活（IDLE）/ 收工（SHUTDOWN）。

打比方：一个自由职业者的一天——先干手里的活（WORK）；干完没新活了就
每几分钟刷一下接单平台（IDLE，轮询收件箱和没人认领的任务）；刷到新活
回去继续干；要是等太久（超时）就收工走人（SHUTDOWN）。

状态流转：
  WORK：调 work_fn(task) 跑一轮
  IDLE：反复查 inbox 和 unclaimed tasks；拿到新工作回 WORK；等超时进 SHUTDOWN
  SHUTDOWN：调 on_shutdown_fn 收尾后退出

被谁用：worker.py 的 autonomous 模式拿它驱动子 agent 进程的整个生命周期。
"""
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STATE_WORK = "work"
STATE_IDLE = "idle"
STATE_SHUTDOWN = "shutdown"


class AutonomousLifecycle:
    """三态生命周期状态机（WORK / IDLE / SHUTDOWN）。

    设计成纯状态机 + 全回调注入：本类只管「什么时候干什么」，
    「具体怎么干」全部由调用方通过构造参数塞进来，方便测试时换成假函数。
    """

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
        """组装状态机。

        参数：
            work_fn：干活的函数，传任务文本，返回结果（WORK 态用）
            poll_inbox_fn：查收件箱，返回消息列表（IDLE 态找新活用）
            poll_tasks_fn：查没人认领的任务，返回任务 dict 列表
            claim_task_fn：认领任务（传任务 ID），返回是否抢到手
            on_shutdown_fn：收工回调（可选），SHUTDOWN 时调；出错也不影响退出
            idle_timeout：等多久没新活就收工，单位秒，默认 60
            poll_interval：IDLE 态两次轮询之间的间隔秒数，默认 5
        """
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
        """开跑，一直循环到收工（SHUTDOWN）为止，阻塞直到结束。

        参数：
            initial_task：开局要干的第一份活（任务文本）

        无返回值。干活抛异常只记日志不中断——坏一份活不该让工人罢工。
        """
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
        """判断等活是否超时（还没开始等或没到时限都算没超）。无参数。"""
        if self._idle_started_at is None:
            return False
        return (time.time() - self._idle_started_at) >= self._idle_timeout

    def _try_get_work(self) -> Optional[str]:
        """找新活：先看收件箱，再看没人认领的任务。

        返回：新任务文本（inbox 消息内容，或认领成功的任务描述）；
        没活返回 None，等下一轮再查。查询出错只记 warning，不当成死活。
        """
        # 1. 收件箱优先：别人直接发来的指令比抢任务急
        try:
            msgs = self._poll_inbox_fn()
            if msgs:
                return msgs[0].content
        except Exception as e:
            logger.warning("poll_inbox 异常: %s", e)

        # 2. 收件箱空着才去任务池捞没人认领的（认领成功才算拿到）
        try:
            ready = self._poll_tasks_fn()
            for t in ready:
                if self._claim_task_fn(t["id"]):
                    return t.get("subject") or t.get("description") or t["id"]
        except Exception as e:
            logger.warning("poll_tasks 异常: %s", e)

        return None
