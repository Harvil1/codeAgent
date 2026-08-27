"""后台任务管理器：让跑得久的命令在后台执行，不卡住对话，跑完再通知。

打个比方：你让师傅装软件（要 10 分钟），不用站在旁边干等——先去干别的，
装完了师傅喊你一声。这个文件就是那个"安排后台干活 + 干完喊人"的管家。

在项目里的位置：由 CLI 的 RuntimeContext 持有，工具层（后台任务工具）
调用它启动/查询/停止任务，主循环每轮来把完成通知取走。

工作方式：
- 用 subprocess.Popen 跑命令（独立进程，不受 Python 全局锁拖累）
- 每个任务配一个后台守护线程，负责收输出、等进程结束
- 任务结束时把通知塞进队列（有锁保护，防打架）
- 主循环每轮调 drain_notifications 把通知一次取走
- 默认跟着 agent 同生共死（agent 退出就清理）；detach=True 则让任务
  独立成新进程组，agent 退了它也继续跑
- 同时最多跑 5 个后台任务（防失控）
- 停滞看门狗：连续 45 秒没有任何新输出 → 发通知提醒 LLM
  "这任务可能卡住了"（比如命令在等交互确认）

跨平台注意：subprocess 必须开 text=True, encoding="utf-8"
（Windows 默认编码是 cp1252，不开会中文乱码——项目铁律）。
"""
import copy
import logging
import queue
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 默认停滞超时（秒）：连续这么久标准输出没有新增一个字节，
# 就判定为"可能卡在交互提示"。0 表示禁用这功能；
# BackgroundManager 构造时显式传 45.0 才真正启用。
DEFAULT_STALL_TIMEOUT = 45.0


@dataclass
class BackgroundTask:
    """一个后台任务的全部状态（跑什么命令、跑到哪步、输出了什么）。

    读写规矩：后台守护线程写它，工具层读它，两边都得先拿 manager 的锁。
    字段大白话：
    - task_id：任务唯一编号（形如 bg_ab12cd34）
    - command：要跑的命令（列表形式）
    - cwd：在哪个目录跑；None=用当前目录
    - status：任务状态——running(跑着)/completed(正常结束)/failed(失败)/stopped(被手动停)
    - pid：进程号
    - started_at / ended_at：开始/结束时间
    - detach：True=agent 退出后任务继续独立跑
    - exit_code：进程退出码（0=成功）
    - stdout / stderr：捕捉到的标准输出/错误输出（会截断到上限）
    - _proc：进程对象本体（repr 里藏起来，防打印一大坨）
    - monitor："监视模式"——跑 tail -f/watch 这类持续观察命令时
      开这个：不算卡住（安静是常态），输出同步落盘
    - output_file：监视模式的输出落盘文件，随时 read_file 看新增内容
    """
    task_id: str
    command: list
    cwd: Optional[Path]
    status: str  # 取值：running(跑着) / completed(正常结束) / failed(失败) / stopped(被手动停)
    pid: Optional[int]
    started_at: datetime
    detach: bool = False
    ended_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    # 监视模式（流式观察——tail -f/watch/轮询类命令）
    monitor: bool = False                  # 开了就不发"卡住"提醒（这类命令安静是常态）
    output_file: Optional[str] = None      # 输出同步落盘的文件路径（read_file 随时查新增）


class BackgroundManager:
    """后台任务管理器本体（管启动、盯梢、通知、停止）。实例由 RuntimeContext 持有。"""

    def __init__(
        self,
        *,
        max_concurrent: int = 5,
        notification_stdout_cap: int = 500,
        result_stdout_cap: int = 5000,
        default_timeout: float = 600.0,
        stall_timeout: float = 45.0,  # 默认 45 秒看门狗开启（传 0 等于关掉）
    ):
        """建管理器。

        参数：
        - max_concurrent：最多同时跑几个后台任务，默认 5
        - notification_stdout_cap：发给 LLM 的通知里输出截到多长，默认 500 字符
        - result_stdout_cap：查询任务结果时的输出上限，默认 5000 字符
        - default_timeout：任务默认最长跑多久（秒），默认 600=10 分钟
        - stall_timeout：多久没新输出算"可能卡住"（秒）；0=关闭看门狗

        返回：无（构造函数）。
        """
        self._tasks: dict = {}
        self._lock = threading.Lock()
        self._notifications: deque = deque()
        self._max_concurrent = max_concurrent
        self._notification_stdout_cap = notification_stdout_cap
        self._result_stdout_cap = result_stdout_cap
        self._default_timeout = default_timeout
        # 停滞看门狗超时。0=关闭（走单次阻塞等待的兼容路径）；
        # >0 开启：用专门的读输出线程 + 主线程定期巡查，超过这个秒数没新输出就发提醒。
        self._stall_timeout = stall_timeout
        # idle wake（后台唤醒）：任务结束通知入队后要敲一下的回调。
        # 由 CLI 注册（往输入队列塞唤醒哨兵）；None = 没人注册，行为照旧。
        self._wake_callback = None

    # ---- 启动 ----
    def start(
        self,
        command: list,
        *,
        cwd: Optional[Path] = None,
        detach: bool = False,
        timeout: Optional[float] = None,
        monitor: bool = False,
        monitor_dir=None,
    ) -> str:
        """启动一个后台任务，立刻返回任务编号（不等命令跑完）。

        monitor=True 走"监视模式"（跑 tail -f/watch 这类持续观察命令）：
        - 新输出实时抄送到 <monitor_dir>/<task_id>.log，随时 read_file 看增量
        - 不发"卡住"提醒（这类命令安静是常态）
        - 默认超时放宽到 24 小时（本来就是长跑的）
        - 进程退出时照常发通知

        参数：
        - command：要跑的命令（列表形式）
        - cwd：在哪个目录跑；None=用当前目录
        - detach：True=agent 退出后任务继续独立跑
        - timeout：最长跑多久（秒）；None=用默认值
        - monitor：是否监视模式
        - monitor_dir：监视输出文件放哪个目录；None=用默认目录

        返回：task_id（任务编号）。
        后台任务已满员时抛 RuntimeError；命令本身不存在不抛——
        会建成一个 failed 任务并通过通知报告。
        """
        with self._lock:
            running_count = sum(
                1 for t in self._tasks.values() if t.status == "running"
            )
            if running_count >= self._max_concurrent:
                raise RuntimeError(
                    f"max concurrent tasks reached ({self._max_concurrent})"
                )

            task_id = "bg_" + secrets.token_hex(4)
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "cwd": str(cwd) if cwd else None,
            }
            if detach:
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    popen_kwargs["start_new_session"] = True

            try:
                proc = subprocess.Popen(command, **popen_kwargs)
            except (FileNotFoundError, OSError) as e:
                # 命令不存在之类的启动失败：不抛异常，直接建成一个 failed 任务
                task = BackgroundTask(
                    task_id=task_id,
                    command=command,
                    cwd=cwd,
                    status="failed",
                    pid=None,
                    started_at=datetime.now(),
                    detach=detach,
                    ended_at=datetime.now(),
                    exit_code=-1,
                    stderr=str(e),
                )
                self._tasks[task_id] = task
                self._push_notification_locked(task)
                return task_id

            task = BackgroundTask(
                task_id=task_id,
                command=command,
                cwd=cwd,
                status="running",
                pid=proc.pid,
                started_at=datetime.now(),
                detach=detach,
                _proc=proc,
                monitor=monitor,
            )
            # 监视模式的输出文件（输出实时抄送到这，read_file 随时查增量）
            if monitor:
                try:
                    m_dir = Path(monitor_dir) if monitor_dir else (
                        Path.home() / ".OmniMate" / ".task_outputs" / "monitor"
                    )
                    m_dir.mkdir(parents=True, exist_ok=True)
                    task.output_file = str(m_dir / f"{task_id}.log")
                except Exception:
                    task.output_file = None  # 求稳：落盘失败也不影响任务照跑
            self._tasks[task_id] = task

        # 起一个守护线程专门盯这个任务
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if monitor:
            # 监视器默认长跑 24 小时 + 强制走轮询路径（实时抄送输出需要它）
            if timeout is None:
                effective_timeout = 86400.0
        t = threading.Thread(
            target=self._watch,
            args=(task_id, proc, effective_timeout, self._stall_timeout),
            daemon=True,
        )
        t.start()
        return task_id

    def _watch(
        self,
        task_id: str,
        proc: subprocess.Popen,
        timeout: float,
        stall_timeout: float = 0.0,
    ):
        """守护线程的活儿：收输出、等进程结束、发通知。

        看门狗开启时（stall_timeout > 0）：
          - 用两个专门的读输出子线程把 stdout/stderr 源源不断搬进队列
          - 本线程定期巡查三件事：进程结束没 / 卡住没 / 总超时到没
          - 超过 stall_timeout 秒没有新输出 → 发"可能卡住"提醒
            （只提醒不杀进程——怎么办交给 LLM 决定）
        stall_timeout == 0 时走单次阻塞等待路径（无看门狗）。
        监视模式任务（task.monitor=True）也走轮询路径（要实时抄送输出），
        但**跳过卡住提醒**（持续观察类命令安静是常态）。

        参数：
        - task_id：任务编号
        - proc：要盯的进程对象
        - timeout：总超时（秒）
        - stall_timeout：卡住判定阈值（秒）；0=不启用

        返回：无（结果通过通知队列汇报）。
        """
        task = self._tasks.get(task_id)
        is_monitor = bool(task and task.monitor)
        if stall_timeout > 0 or is_monitor:
            return self._watch_with_stall(
                task_id, proc, timeout,
                stall_timeout if stall_timeout > 0 else 30.0,
            )
        return self._watch_classic(task_id, proc, timeout)

    def _watch_classic(
        self, task_id: str, proc: subprocess.Popen, timeout: float,
    ):
        """一次性阻塞等到命令结束或超时（看门狗关闭时用）。

        参数：
        - task_id：任务编号
        - proc：要盯的进程对象
        - timeout：总超时（秒）

        返回：无（结果通过通知队列汇报）。
        """
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode
            with self._lock:
                task = self._tasks.get(task_id)
                # stop() 抢先把状态改了（不再是 running）时，
                # 这里绝不能再覆盖状态、再发一条重复通知
                if task is None or task.status != "running":
                    return
                task.stdout = (stdout or "")[: self._result_stdout_cap]
                task.stderr = (stderr or "")[: self._result_stdout_cap]
                task.exit_code = exit_code
                task.ended_at = datetime.now()
                task.status = "completed" if exit_code == 0 else "failed"
                self._push_notification_locked(task)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.status != "running":
                    return
                task.stdout = (stdout or "")[: self._result_stdout_cap]
                task.stderr = (stderr or "")[: self._result_stdout_cap]
                task.exit_code = -1
                task.ended_at = datetime.now()
                task.status = "failed"
                self._push_notification_locked(task)
        except Exception as e:
            logger.warning("bg task %s watcher 异常: %s", task_id, e)
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.status != "running":
                    return
                task.status = "failed"
                task.ended_at = datetime.now()
                task.exit_code = -1
                task.stderr = str(e)[: self._result_stdout_cap]
                self._push_notification_locked(task)

    def _watch_with_stall(
        self,
        task_id: str,
        proc: subprocess.Popen,
        timeout: float,
        stall_timeout: float,
    ):
        """带看门狗的盯梢方式：读输出线程 + 本线程定期巡查。

        流程：
          1. 起两个读输出线程，把 stdout/stderr 逐行搬进队列
          2. 本线程每 0.5 秒醒一次，依次查：
             - 把队列里攒的输出搬进缓存
             - 有新输出 → "最近输出时间"重新计时
             - 进程结束了吗（poll）→ 结束就收尾、发最终通知
             - 总超时到了吗 → 到了就杀进程、标记失败
             - 超过 stall_timeout 没新输出 → 发"可能卡住"提醒（任务保持 running）

        参数：
        - task_id：任务编号
        - proc：要盯的进程对象
        - timeout：总超时（秒）
        - stall_timeout：卡住判定阈值（秒）

        返回：无（结果通过通知队列汇报）。
        """
        stdout_q: queue.Queue = queue.Queue()
        stderr_q: queue.Queue = queue.Queue()

        def _reader(stream, q: queue.Queue):
            try:
                for line in iter(stream.readline, ""):
                    q.put(line)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(
            target=_reader, args=(proc.stdout, stdout_q), daemon=True,
        )
        t_err = threading.Thread(
            target=_reader, args=(proc.stderr, stderr_q), daemon=True,
        )
        t_out.start()
        t_err.start()

        stdout_parts: list = []
        stderr_parts: list = []
        last_output_time = time.monotonic()
        stall_notified = False
        deadline = time.monotonic() + timeout

        # 监视模式的两件套——输出实时抄送到文件 + 跳过"卡住"提醒
        with self._lock:
            _task_ref = self._tasks.get(task_id)
            is_monitor = bool(_task_ref and _task_ref.monitor)
            output_file = (_task_ref.output_file if _task_ref else None)
        tee_f = None
        if is_monitor and output_file:
            try:
                tee_f = open(output_file, "a", encoding="utf-8")
            except Exception:
                tee_f = None  # 求稳：抄送文件打不开就不抄，监视照常

        try:
            while True:
                # 发现 stop() 已抢先改状态，立刻退出别再抢
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None or task.status != "running":
                        return

                # 把队列里攒的输出搬出来（队列空就立刻走人，不傻等）
                got_new = False
                try:
                    while True:
                        line = stdout_q.get_nowait()
                        stdout_parts.append(line)
                        if tee_f is not None:
                            try:
                                tee_f.write(line)
                                tee_f.flush()
                            except Exception:
                                pass
                        got_new = True
                except queue.Empty:
                    pass
                try:
                    while True:
                        stderr_parts.append(stderr_q.get_nowait())
                        got_new = True
                except queue.Empty:
                    pass
                if got_new:
                    last_output_time = time.monotonic()
                    stall_notified = False

                # 进程结束了吗？
                rc = proc.poll()
                if rc is not None:
                    # 给读输出线程 1 秒时间把尾巴读完
                    t_out.join(timeout=1.0)
                    t_err.join(timeout=1.0)
                    try:
                        while True:
                            stdout_parts.append(stdout_q.get_nowait())
                    except queue.Empty:
                        pass
                    try:
                        while True:
                            stderr_parts.append(stderr_q.get_nowait())
                    except queue.Empty:
                        pass
                    stdout = "".join(stdout_parts)
                    stderr = "".join(stderr_parts)
                    with self._lock:
                        task = self._tasks.get(task_id)
                        if task is None or task.status != "running":
                            return
                        task.stdout = stdout[: self._result_stdout_cap]
                        task.stderr = stderr[: self._result_stdout_cap]
                        task.exit_code = rc
                        task.ended_at = datetime.now()
                        task.status = "completed" if rc == 0 else "failed"
                        self._push_notification_locked(task)
                    return

                now = time.monotonic()
                # 总超时到了吗？
                if now >= deadline:
                    proc.kill()
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        pass
                    with self._lock:
                        task = self._tasks.get(task_id)
                        if task is None or task.status != "running":
                            return
                        task.stdout = "".join(stdout_parts)[: self._result_stdout_cap]
                        task.stderr = "".join(stderr_parts)[: self._result_stdout_cap]
                        task.exit_code = -1
                        task.ended_at = datetime.now()
                        task.status = "failed"
                        self._push_notification_locked(task)
                    return

                # 卡住检测（监视模式任务跳过——持续观察安静是常态）
                if (
                    not stall_notified
                    and not is_monitor
                    and (now - last_output_time) >= stall_timeout
                ):
                    with self._lock:
                        task = self._tasks.get(task_id)
                        if task is not None and task.status == "running":
                            # 发"可能卡住"提醒（任务状态保持 running 不动）
                            self._notifications.append({
                                "task_id": task_id,
                                "status": "running",
                                "stall": True,
                                "stall_seconds": stall_timeout,
                                "stdout": "".join(stdout_parts)[
                                    : self._notification_stdout_cap
                                ],
                                "stderr": "".join(stderr_parts)[
                                    : self._notification_stdout_cap
                                ],
                                "command": task.command,
                                "hint": (
                                    f"任务已 {stall_timeout:.0f}s 无 stdout 输出，"
                                    "可能卡在交互提示或死锁。可用 bg_stop 终止。"
                                ),
                            })
                    stall_notified = True
                    logger.info(
                        "bg task %s stall detected (%.1fs 无输出)",
                        task_id, stall_timeout,
                    )

                time.sleep(0.5)  # 巡查间隔半秒：够及时，也不至于空转太勤
        except Exception as e:
            logger.warning("bg task %s stall watcher 异常: %s", task_id, e)
            try:
                proc.kill()
            except Exception:
                pass
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.status != "running":
                    return
                task.status = "failed"
                task.ended_at = datetime.now()
                task.exit_code = -1
                task.stderr = str(e)[: self._result_stdout_cap]
                self._push_notification_locked(task)
        finally:
            # 抄送文件必须在所有退出路径都关掉（漏关会占着文件句柄）
            if tee_f is not None:
                try:
                    tee_f.close()
                except Exception:
                    pass

    def _push_notification_locked(self, task: BackgroundTask):
        """把一条任务结束通知塞进队列（内部方法）。

        规矩：调用前必须已经拿着 self._lock（方法名 _locked 就是提醒这个）。
        顺带敲一下 idle wake 回调（注册了才有）：通知到了就提醒主循环
        "有后台任务收工了，空闲的话来处理"。注意"可能卡住"提醒
        （_watch_with_stall 里直接 append 的 stall 通知）不走这里——
        任务还没完，不唤醒。

        参数：
        - task：刚结束的任务对象

        返回：无。
        """
        self._notifications.append({
            "task_id": task.task_id,
            "status": task.status,
            "exit_code": task.exit_code,
            "stdout": (task.stdout or "")[: self._notification_stdout_cap],
            "stderr": (task.stderr or "")[: self._notification_stdout_cap],
            "command": task.command,
            "ended_at": task.ended_at.isoformat() if task.ended_at else None,
        })
        if self._wake_callback is not None:
            try:
                self._wake_callback()
            except Exception as e:
                logger.debug("bg wake 回调失败（fail-open）: %s", e)

    # ---- 查询 ----
    def status(self, task_id: str) -> Optional[BackgroundTask]:
        """查一个任务现在的状态。

        返回拷贝而非原件——防调用方读到守护线程改一半的自相矛盾数据。

        参数：
        - task_id：任务编号

        返回：任务状态对象的拷贝；编号不存在返回 None。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return copy.copy(task) if task else None

    def result(self, task_id: str) -> Optional[BackgroundTask]:
        """查任务结果——就是 status 换个名字（工具层习惯叫"取结果"，
        拿到后读 task.stdout，输出已按结果上限截好）。"""
        return self.status(task_id)

    def list_tasks(self) -> list:
        """列出所有任务（每个都是拷贝，理由同 status：防读到改一半的数据）。

        参数：无。返回：任务拷贝的列表。
        """
        with self._lock:
            return [copy.copy(t) for t in self._tasks.values()]

    # ---- 控制 ----
    def stop(self, task_id: str) -> bool:
        """停掉一个后台任务。

        参数：
        - task_id：任务编号

        返回：True=停了（已经结束的任务也算停成功，重复调用无害）；
        False=没有这个任务。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in ("completed", "failed", "stopped"):
                return True
            # 先把状态标成 "stopping"：盯梢线程看到非 running
            # 就知道别人抢先了，不会覆盖结果、重复发通知
            task.status = "stopping"
            proc = task._proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("stop task %s 失败: %s", task_id, e)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return True
            # 不管盯梢线程有没有抢先处理，最终状态都由 stop 定成 "stopped"
            task.status = "stopped"
            task.ended_at = datetime.now()
            task.exit_code = task.exit_code if task.exit_code is not None else -1
            self._push_notification_locked(task)
        return True

    # ---- 通知 ----
    def set_wake_callback(self, fn) -> None:
        """注册 idle wake 回调：任务结束通知入队后被敲一下（不传参）。

        CLI 用它往输入队列塞唤醒哨兵，实现"主对话空闲时后台任务
        完成 → 自动跑一轮处理结果"。回调在盯梢线程里执行，必须便宜、
        非阻塞、永不拖垮通知本身（调用处已 try/except 兜底）。

        参数：
        - fn：无参回调；传 None 等于注销。

        返回：无。
        """
        self._wake_callback = fn

    def has_notifications(self) -> bool:
        """队列里有没有没取走的通知（给"要不要发起唤醒轮"做预检用）。

        参数：无。返回：True=有通知待取。
        """
        with self._lock:
            return len(self._notifications) > 0

    def drain_notifications(self) -> list:
        """主循环每轮调一次：把攒下的通知全部取走并清空队列。

        参数：无。返回：通知列表（任务结束/卡住提醒等）。
        """
        with self._lock:
            notifications = list(self._notifications)
            self._notifications.clear()
        return notifications

    # ---- 生命周期 ----
    def shutdown(self) -> None:
        """agent 退出时调：把所有还在跑、且没开独立存活（detach）的任务停掉。

        开了 detach 的不碰——用户明确要它脱离 agent 继续跑。
        参数无，返回无。
        """
        with self._lock:
            to_stop = [
                (tid, t) for tid, t in self._tasks.items()
                if t.status == "running" and not t.detach
            ]
        for tid, _ in to_stop:
            self.stop(tid)
