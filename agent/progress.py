"""工具执行进度摘要——长时间任务跑着的时候，定时告诉用户"我还在干活、在干什么"。

场景：subagent（子代理）或后台任务一跑就是几分钟，界面上毫无动静用户
会以为卡死了。这个模块在背后开一个小线程，每隔一阵推一条进度消息给
前端，就像外卖 App 上"骑手已取餐"那种提示。

怎么干活：
- daemon thread（守护线程——主程序退出时它跟着死，不挡路）每 interval 秒触发一次
- 配了 aux_llm_router（辅助小模型路由，专门干杂活省钱）：让便宜模型根据
  任务目标生成一句简短的进度提示
- 没配：就发固定心跳文案"任务仍在执行..."
- interval=0 时整个功能关闭（测试用）

用法（模块用法就这一种，直接套）：
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


DEFAULT_PROGRESS_INTERVAL = 30.0  # 默认多久报一次进度（秒）
DEFAULT_HEARTBEAT_MESSAGE = "任务仍在执行..."


class ProgressReporter:
    """周期性发进度通知的小助手。

    用法：像文件一样 with 打开（详见本模块开头的 docstring，那里有完整示例）：
    进 with 时自动 start 起后台线程，出 with 时自动 stop。
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
        """记下配置，先不起线程（start/with 进去才起）。

        参数：
            goal：这次在干什么（任务目标，会出现在进度消息里）。
            stream_callback：推送函数，收事件 dict；传 None 则整个功能不启动。
            aux_llm_router：辅助小模型路由，用来生成更聪明的进度文案；
                传 None 就只发固定心跳。
            interval：多少秒报一次（默认 30；<=0 关闭功能）。
            aux_model：指定辅助模型名；None 用路由默认的。
        """
        self.goal = goal
        self.stream_callback = stream_callback
        self.aux_llm_router = aux_llm_router
        self.interval = interval
        self.aux_model = aux_model
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

    def __enter__(self):
        """进 with 块：自动启动进度线程。"""
        self.start()
        return self

    def __exit__(self, *args):
        """出 with 块：自动停线程。"""
        self.stop()

    def start(self) -> None:
        """启动后台进度线程（守护线程）。调多次也无害。

        interval<=0 或没配 stream_callback 时什么都不做（等于关闭功能）。
        """
        if self.interval <= 0 or self.stream_callback is None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="progress-reporter",
        )
        self._thread.start()

    def stop(self) -> None:
        """停掉进度线程（置停止标记 + 最多等 1 秒收尾）。调多次也无害。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def tick_once(self) -> dict:
        """立刻同步触发一次进度事件——不走线程，测试专用。

        参数：无。
        返回：本次推送的事件 dict（含 type/goal/tick/耗时/消息文案）。
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
        """进度线程的主循环：每 interval 秒报一次，收到停止标记就退出。"""
        while not self._stop_event.wait(self.interval):
            try:
                self.tick_once()
            except Exception as e:
                logger.debug("progress reporter tick 异常: %s", e)

    def generate_message(self) -> str:
        """生成一条进度文案：有小模型就让它现写一句，否则发固定心跳。

        fail-open：调小模型失败就降级成心跳文案，这个函数绝不抛异常
        ——进度提示是锦上添花，不能反过来把正事搅黄。

        参数：无。
        返回：进度文案字符串（小模型写的截到 80 字符）。
        """
        if self.aux_llm_router is None:
            return DEFAULT_HEARTBEAT_MESSAGE
        try:
            prompt = (
                f"用户在等子任务执行结果：{self.goal}\n"
                f"已经过了约 {self._tick_count * self.interval:.0f} 秒。"
                f"用 10 个字以内简短描述一个等待中的进度提示（不要复述任务）："
            )
            # aux_llm_router.chat_completions 是异步函数，而本函数跑在
            # 守护线程里（不在宿主循环线程）——交给进程级常驻循环宿主
            # 同步等结果（等价旧的 asyncio.run，aux 缓存 client 不再
            # 每次绑定新循环漂移）。
            from agent.loop_host import loop_host
            resp = loop_host.run_async(self.aux_llm_router.chat_completions(
                [{"role": "user", "content": prompt}],
            ))
            choice = resp.choices[0]
            text = getattr(choice.message, "content", None)
            if text and text.strip():
                return text.strip()[:80]
        except Exception as e:
            logger.debug("aux_llm 进度摘要失败，发心跳: %s", e)
        return DEFAULT_HEARTBEAT_MESSAGE
