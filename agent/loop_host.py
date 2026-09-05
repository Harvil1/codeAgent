"""事件循环宿主——进程级常驻事件循环，取代「每回合 asyncio.run 现建现拆」。

大白话：以前 CLI 每处理一条用户消息就搭一个临时舞台（事件循环），
演完拆台。问题：HTTP 连接池第一次使用时会跟当时的舞台绑死，下一回合
换了新舞台，旧连接全断（"Event loop is closed" 的结构根源）；后台
线程也各自搭各自的舞台（一线程一循环，线程数随任务涨）。

现在：进程首次用到时搭一个永久舞台（daemon 线程 + 永不关闭的事件
循环），所有要用 async client 的地方（主回合、斜杠命令、后台任务）
统一到这个舞台上跑——连接池绑得稳，后台任务也只是舞台上的一个 task。

三种姿势：
- 外部线程同步等结果：``loop_host.run_async(coro)``
- 外部线程发出去就不管：``loop_host.submit(coro, name=...)``
- 跑一个「回合」：``loop_host.run_turn(coro)``——结束时清场（回合栅栏）

回合栅栏（为什么要有）：asyncio.run 关循环会把回合内没跑完的 task 全部
取消（免费清场）；常驻舞台不会自己清场——run_turn 在回合结束时显式取消
「本回合新冒出来且没完成、又不在后台豁免名单」的 task，保住旧语义。
submit 注册的后台任务（自动记忆提取/批间摘要这类）永不被栅栏误杀。
"""
import asyncio
import contextvars
import logging
import threading
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class AgentLoopHost:
    """进程级常驻事件循环宿主（用模块级单例 loop_host，不要自己 new）。

    一个 daemon 线程跑一个永不关闭的事件循环；外部线程通过
    run_coroutine_threadsafe 把协程交上来。惰性启动：第一次调用
    run_async/submit/run_turn 才拉线程。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        # submit 注册的后台任务集合（回合栅栏豁免名单）。只在宿主循环
        # 线程内增删（_wrapped 协程体里），无锁安全
        self._bg_tasks: set = set()
        # 当前在跑的回合 future（force_exit 主动取消用；无回合时为 None）
        self._current_turn_fut: Optional[ConcurrentFuture] = None

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        """确保宿主线程和循环就绪（惰性 + 幂等 + 线程安全）。"""
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_forever, daemon=True, name="agent-loop-host",
            )
            self._thread.start()
            return self._loop

    def _run_forever(self) -> None:
        """宿主线程主体：跑循环到 stop，退出前把遗留 task 收干净。"""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    def stop(self, timeout: float = 2.0) -> None:
        """优雅停机：停循环并收线程（幂等；异常退出路径由 os._exit 兜底）。"""
        with self._lock:
            loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # 循环已关
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def run_async(self, coro: Awaitable, *, timeout: Optional[float] = None) -> Any:
        """外部线程：把协程交给宿主循环跑，阻塞等结果，异常原样穿透。

        等价于 asyncio.run(coro)，但协程跑在常驻循环上（连接池绑定稳定）。
        ⚠️ 不得在宿主循环线程内调用（自己等自己 = 死锁）。
        """
        loop = self._ensure_started()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout=timeout)
        except KeyboardInterrupt:
            # 调用线程收到 Ctrl+C（防御路径——真实通道是 UI 键位的
            # 协作式中断）：取消协程让中断传进去，再重抛给调用方
            fut.cancel()
            raise

    def submit(self, coro: Awaitable, *, name: str = "bg") -> ConcurrentFuture:
        """外部线程：发出去就不管的后台任务（fire-and-forget）。

        任务注册进豁免名单（回合栅栏不取消）；异常打 WARNING（fail-open
        但要大声）；contextvars 从调用方复制带上（工作目录等上下文可见）。
        """
        loop = self._ensure_started()
        ctx = contextvars.copy_context()

        async def _wrapped():
            me = asyncio.current_task()
            self._bg_tasks.add(me)
            try:
                await coro
            except Exception as e:
                logger.warning("loop_host 后台任务 %s 失败（fail-open）: %s", name, e)
            finally:
                self._bg_tasks.discard(me)

        return asyncio.run_coroutine_threadsafe(ctx.run(_wrapped), loop)

    def run_turn(self, coro: Awaitable) -> Any:
        """跑一个「回合」：正常执行 + 结束时清场（回合栅栏）。

        asyncio.run 关循环会把回合内遗留 task 全部取消——常驻循环后这层
        免费清场没了，这里显式补上（见 _fenced_turn）。当前回合的 future
        记在 _current_turn_fut 上，强退路径可用 cancel_current_turn() 主动取消。
        """
        loop = self._ensure_started()
        fut = asyncio.run_coroutine_threadsafe(self._fenced_turn(coro), loop)
        self._current_turn_fut = fut
        try:
            return fut.result()
        except KeyboardInterrupt:
            fut.cancel()
            raise
        finally:
            self._current_turn_fut = None

    async def _fenced_turn(self, coro: Awaitable) -> Any:
        """回合栅栏：结束时取消本回合新生的非后台 task（等价旧 asyncio.run 清场）。"""
        before = set(asyncio.all_tasks())
        try:
            return await coro
        finally:
            me = asyncio.current_task()
            leftover = [
                t for t in asyncio.all_tasks()
                if t is not me and t not in before and t not in self._bg_tasks
            ]
            for t in leftover:
                t.cancel()
            if leftover:
                await asyncio.gather(*leftover, return_exceptions=True)

    def call_soon_threadsafe(self, fn: Callable) -> None:
        """线程安全地在宿主循环上调度一个同步函数（收尾/取消用）。"""
        loop = self._ensure_started()
        loop.call_soon_threadsafe(fn)

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """宿主循环（未启动时为 None；只读探查用，别拿去跑协程）。"""
        return self._loop


# 进程级单例——全项目统一用它
loop_host = AgentLoopHost()


def cancel_current_turn() -> None:
    """主动取消当前在跑的回合（强退路径用；没有回合时是安全空操作）。

    fut.cancel() 会让 CancelledError 传进回合协程（与旧 asyncio.run 被
    KeyboardInterrupt 打断后的清理语义对齐）。
    """
    fut = loop_host._current_turn_fut
    if fut is not None and not fut.done():
        try:
            fut.cancel()
        except Exception:
            pass  # fail-open：强退兜底还有 os._exit
