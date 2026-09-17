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

硬约束（回合并发契约）：同一时刻只允许一个回合在跑（run_turn 不得
并发——两个回合的栅栏会把对方回合的新生 task 当遗留互删）；回合进行
期间，其他线程经 run_async 提交、且活到回合结束仍未完成的协程，会被
回合栅栏当成回合遗留清理掉——跨回合的长活儿必须走 submit 或
run_async(exempt_from_fence=True)（两者都进豁免名单）。

回合栅栏（为什么要有）：asyncio.run 关循环会把回合内没跑完的 task 全部
取消（免费清场）；常驻舞台不会自己清场——run_turn 在回合结束时显式取消
「本回合新冒出来且没完成、又不在后台豁免名单」的 task，保住旧语义。
submit 注册的后台任务（自动记忆提取/批间摘要这类）永不被栅栏误杀。
"""
import asyncio
import contextvars
import logging
import threading
# _chain_future 是 asyncio.run_coroutine_threadsafe 内部同款的私有 API：
# 把 asyncio future 的结果/异常/取消搬进 concurrent Future（fut.cancel()
# 也会反向取消 task）——语义跟随标准库。标准库若哪天改签名，verify 的
# loop_host 检查会先炸给我们看。
from asyncio.futures import _chain_future
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Awaitable, Callable, Coroutine, Optional

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
        # pending 豁免集（C-1 注册时机竞态兜底）：submit 在提交前把协程
        # 对象同步登记到这里。为什么需要：_bg_tasks 要等 _wrapped 首步
        # 真正跑起来才注册，而「task 已创建、首步还没跑」的窗口里（task
        # 创建排在回合栅栏落下之前、首步排在之后），回合栅栏的
        # all_tasks() 会看到这个既不在 _bg_tasks 里的新 task，把它当
        # 回合遗留误杀——协程体一行不跑、WARNING 都不打，后台任务
        # 静默消失。栅栏按协程对象在这里豁免兜底（见 _fenced_turn）。
        # set 增删是 GIL 原子的，跨线程调用够安全。
        self._pending_bg_coros: set = set()
        # 回合是否在跑（并发契约警示用；调用方线程读写在容忍窗口内）
        self._turn_active = False
        # 停机旗：stop() 置位后永不复位——_ensure_started 见到它就拒绝
        # 再拉新循环（防 straggler 线程在进程收尾窗口复活宿主）
        self._stopped: bool = False
        # 当前在跑的回合 future（force_exit 主动取消用；无回合时为 None）
        self._current_turn_fut: Optional[ConcurrentFuture] = None

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        """确保宿主线程和循环就绪（惰性 + 幂等 + 线程安全）。"""
        with self._lock:
            if self._stopped:
                # 停机后拒绝复活：shutdown 后某个 straggler 线程再来调
                # run_async/submit，旧版会悄悄拉起一个新循环（进程将退，
                # 白造线程还可能半路死锁在垂死循环上）——fail-open 由
                # 调用方处理，这里抛 RuntimeError 拒绝（异常消息即提示）
                raise RuntimeError("loop_host 已停机，拒绝再启动")
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
            except Exception as e:
                logger.warning("loop_host 停机排空异常: %s", e)
            self._loop.close()

    def stop(self, timeout: float = 2.0) -> None:
        """优雅停机：停循环并收线程（幂等；异常退出路径由 os._exit 兜底）。

        置 _stopped 旗：之后任何线程再来 run_async/submit 都被拒绝复活
        （一次性宿主——stop 即退役，进程收尾后不该再有新活儿上舞台）。
        """
        with self._lock:
            self._stopped = True
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
    def run_async(self, coro: Coroutine, *, timeout: Optional[float] = None,
                  exempt_from_fence: bool = False) -> Any:
        """外部线程：把协程交给宿主循环跑，阻塞等结果，异常原样穿透。

        等价于 asyncio.run(coro)，但协程跑在常驻循环上（连接池绑定稳定）。
        timeout 只限制「等结果」的时长；超时会顺手取消协程（wait_for 语义），
        不然调用方都超时走了，协程还赖在常驻循环上白跑到天荒地老。
        ⚠️ 不得在宿主循环线程内调用（自己等自己 = 死锁）。
        停机后调用抛 RuntimeError（stop 之后拒绝复活，见 _ensure_started）。
        exempt_from_fence：True = 该协程注册进回合栅栏豁免名单（出生即
        登记，机制同 submit）——给「跨回合/回合期间的后台线程长活」用
        （curator 审查、进度播报这类）。回合栅栏只清回合自己的遗留，
        不该碰这些活。默认 False（回合内的短调用无需豁免）。
        """
        loop = self._ensure_started()
        if exempt_from_fence:
            # 出生即豁免（机制同 submit 的 pending 集）：提交前同步登记，
            # 栅栏按协程对象放行；finally 退场（正常/异常/取消收尾都走）
            self._pending_bg_coros.add(coro)
            try:
                return self._submit_and_wait(coro, loop, timeout)
            finally:
                self._pending_bg_coros.discard(coro)
        return self._submit_and_wait(coro, loop, timeout)

    def _submit_and_wait(self, coro: Coroutine, loop: asyncio.AbstractEventLoop,
                         timeout: Optional[float]) -> Any:
        """提交协程到宿主循环并阻塞等结果；中断/超时顺手取消协程再重抛。"""
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout=timeout)
        except KeyboardInterrupt:
            # 调用线程收到 Ctrl+C（防御路径——真实通道是 UI 键位的
            # 协作式中断）：取消协程让中断传进去，再重抛给调用方
            fut.cancel()
            raise
        except TimeoutError:
            # 等超时了=调用方不等了：取消协程别让它白跑（wait_for 语义）；
            # 若协程其实刚好已完成，cancel 是无害空操作
            fut.cancel()
            raise

    def submit(self, coro: Coroutine, *, name: str = "bg") -> ConcurrentFuture:
        """外部线程：发出去就不管的后台任务（fire-and-forget）。

        任务注册进豁免名单（回合栅栏不取消）；异常打 WARNING（fail-open
        但要大声）；Task 以调用方（submit 线程）的 contextvars 上下文创建
        （create_task(context=...)），工作目录等上下文对后台任务可见。
        停机后调用不炸调用方：返回一个已设 RuntimeError 的 future
        （fail-open 但要大声——WARNING 打进日志）。
        """
        try:
            loop = self._ensure_started()
        except RuntimeError as e:
            # straggler 线程停机后才来交活：不给拉新循环，也别把
            # RuntimeError 直接砸给「发出去就不管」的调用方——给个
            # 已设异常的 future，等 result 的人自己看到（fail-open 大声）
            logger.warning("submit 被拒（loop_host 已停机）: %s", e)
            fut: ConcurrentFuture = ConcurrentFuture()
            fut.set_exception(e)
            return fut
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
                # 首步已跑、_bg_tasks 已接管豁免职责：协程对象从 pending
                # 豁免集退场（正常/异常/取消收尾都会走到这里）
                self._pending_bg_coros.discard(asyncio.current_task().get_coro())

        wrapped = ctx.run(_wrapped)
        # 注册时机竞态（C-1）：必须在提交前同步登记 pending 豁免集——
        # 建任务的回调跑完后、task 首步运行前，回合栅栏的 all_tasks()
        # 会看到这个「已创建未启动」的 task，此时它还没把自己加进
        # _bg_tasks，会被当回合遗留误杀。栅栏按协程对象豁免兜底
        # （见 _fenced_turn），首步跑起来后再由 _bg_tasks 接管。
        self._pending_bg_coros.add(wrapped)
        fut: ConcurrentFuture = ConcurrentFuture()

        def _schedule():
            # Task 的上下文显式用调用方（submit 线程）的快照 ctx 创建
            # （3.11+ 的 create_task 支持 context 参数）——contextvars
            # 传播由这里白纸黑字保证，不依赖 call_soon_threadsafe 的
            # Handle 自带「上下文顺路拷贝」那种隐式通道（旧实现靠的
            # 就是隐式通道，行为对不对全凭实现细节，无从审计）。
            # fut.cancel() 经 _chain_future 的取消回调反向取消 task——
            # 与 run_coroutine_threadsafe 完全同款语义。
            try:
                task = loop.create_task(wrapped, context=ctx)
                _chain_future(task, fut)
            except Exception as e:
                fut.set_exception(e)

        try:
            loop.call_soon_threadsafe(_schedule)
        except RuntimeError:
            # 循环已关（进程收尾窗口）：豁免集退场，异常原样穿透
            # （与旧 run_coroutine_threadsafe 路径行为一致）
            self._pending_bg_coros.discard(wrapped)
            raise
        return fut

    def run_turn(self, coro: Coroutine) -> Any:
        """跑一个「回合」：正常执行 + 结束时清场（回合栅栏）。

        asyncio.run 关循环会把回合内遗留 task 全部取消——常驻循环后这层
        免费清场没了，这里显式补上（见 _fenced_turn）。当前回合的 future
        记在 _current_turn_fut 上，强退路径可用 cancel_current_turn() 主动取消。
        """
        loop = self._ensure_started()
        if self._turn_active:
            # 并发契约（模块 docstring 硬约束）：两个回合的栅栏会把对方
            # 回合的新生 task 当遗留互删，行为未定义——fail-open 但要
            # 大声，只警示不拦截，后到者继续执行
            logger.warning("run_turn 并发调用（契约禁止，后到者继续执行）")
        self._turn_active = True
        fut = asyncio.run_coroutine_threadsafe(self._fenced_turn(coro), loop)
        self._current_turn_fut = fut
        try:
            return fut.result()
        except KeyboardInterrupt:
            fut.cancel()
            raise
        finally:
            self._turn_active = False
            self._current_turn_fut = None

    async def _fenced_turn(self, coro: Awaitable) -> Any:
        """回合栅栏：结束时取消本回合新生的非后台 task（等价旧 asyncio.run 清场）。"""
        before = set(asyncio.all_tasks())
        try:
            return await coro
        finally:
            me = asyncio.current_task()
            # C-1 兜底：submit 的后台任务「已创建未启动」时不在 _bg_tasks
            # （注册要等首步），按 pending 豁免集里的协程对象放行
            leftover = [
                t for t in asyncio.all_tasks()
                if t is not me and t not in before and t not in self._bg_tasks
                and t.get_coro() not in self._pending_bg_coros
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
