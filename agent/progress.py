"""工具执行进度摘要（P1-10）。

长工具调用（delegate_task / 后台任务）执行期间，周期性调用 aux_llm 生成进度摘要，
通过 stream_callback 推送给前端，让用户知道"还在做什么"。

设计：
- daemon thread 每 interval 秒触发一次
- 有 aux_llm_router：调 LLM 生成基于 goal 的简短进度提示（用便宜模型）
- 无 aux_llm_router：发心跳 "任务仍在执行..."
- interval=0 时禁用（测试用）

用法：
    with ProgressReporter(
        goal="执行测试",
        stream_callback=cb,
        aux_llm_router=router,
    ):
        run_long_task()
"""
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


DEFAULT_PROGRESS_INTERVAL = 30.0  # 秒
DEFAULT_HEARTBEAT_MESSAGE = "任务仍在执行..."


class ProgressReporter:
    """周期性发送进度通知的辅助类（P1-10）。

    用法见模块 docstring。
    """

    def __init__(
        self,
        *,
        goal: str,
        stream_callback: Optional[Callable[[dict], None]] = None,
        aux_llm_router=None,
        interval: float = DEFAULT_PROGRESS_INTERVAL,
        aux_model: Optional[str] = None,
    ):
        self.goal = goal
        self.stream_callback = stream_callback
        self.aux_llm_router = aux_llm_router
        self.interval = interval
        self.aux_model = aux_model
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self) -> None:
        """启动 daemon thread。幂等。interval<=0 或 stream_callback=None 时 no-op。"""
        if self.interval <= 0 or self.stream_callback is None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="progress-reporter",
        )
        self._thread.start()

    def stop(self) -> None:
        """停止 thread。幂等。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def tick_once(self) -> dict:
        """同步触发一次进度事件（测试用，不走 thread）。

        返回推送的事件 dict。
        """
        self._tick_count += 1
        msg = self.generate_message()
        event = {
            "type": "progress",
            "goal": self.goal,
            "tick": self._tick_count,
            "elapsed_seconds": int(self._tick_count * self.interval),
            "message": msg,
        }
        if self.stream_callback is not None:
            try:
                self.stream_callback(event)
            except Exception as e:
                logger.debug("stream_callback 异常（progress）: %s", e)
        return event

    def _run(self) -> None:
        """daemon thread 主循环：每 interval 秒 tick 一次。"""
        while not self._stop_event.wait(self.interval):
            try:
                self.tick_once()
            except Exception as e:
                logger.debug("progress reporter tick 异常: %s", e)

    def generate_message(self) -> str:
        """生成进度消息。有 aux_llm 时调 LLM，否则发心跳。

        fail-open：LLM 失败时降级为心跳，绝不抛异常。
        """
        if self.aux_llm_router is None:
            return DEFAULT_HEARTBEAT_MESSAGE
        try:
            prompt = (
                f"用户在等子任务执行结果：{self.goal}\n"
                f"已经过了约 {self._tick_count * self.interval:.0f} 秒。"
                f"用 10 个字以内简短描述一个等待中的进度提示（不要复述任务）："
            )
            resp = self.aux_llm_router.chat_completions(
                [{"role": "user", "content": prompt}],
            )
            choice = resp.choices[0]
            text = getattr(choice.message, "content", None)
            if text and text.strip():
                return text.strip()[:80]
        except Exception as e:
            logger.debug("aux_llm 进度摘要失败，发心跳: %s", e)
        return DEFAULT_HEARTBEAT_MESSAGE
