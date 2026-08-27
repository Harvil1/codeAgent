"""团队协调员：管成员名单（谁在队里、什么状态），负责拉起和收拾工人进程。

打比方：工地包工头——新工人来了登记花名册（registry.json，一直记在
硬盘上，重启也不丢）；要干活了就去招工（spawn，用 subprocess.Popen 起
一个真正的操作系统子进程跑 agent/team/worker.py）；收工时负责送走
所有人（shutdown_all）。

被谁用：主 agent 和工具层通过它组队；worker.py 自己也会建一个，
用来回报状态。
"""
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agent.team.bus import MessageBus, _with_lock

logger = logging.getLogger(__name__)


@dataclass
class TeamMember:
    """花名册上的一行：一个团队成员的档案。"""
    name: str
    role: str
    pid: Optional[int]
    status: str             # 生命周期：刚登记 spawning / 干活中 running / 干完了 completed / 出事了 failed
    created_at: str
    task: Optional[str] = None


class TeamCoordinator:
    """包工头本头：登记成员、拉起进程、改状态、收尾清理。"""

    def __init__(self, *, team_dir: Path, omnimate_home: Path,
                 config: dict):
        """开工准备：建目录、定文件位置、顺手建一个消息总线。

        参数：
            team_dir：团队工作目录（花名册 registry.json 和收件箱都在里面）
            omnimate_home：agent 的家目录（~/.OmniMate），传给子进程用
            config：全局配置 dict（从中读 team.max_members 等设置）
        """
        self._team_dir = Path(team_dir)
        self._omnimate_home = Path(omnimate_home)
        self._config = config
        self._registry_path = self._team_dir / "registry.json"
        self._registry_lock = self._team_dir / "registry.lock"
        self._team_dir.mkdir(parents=True, exist_ok=True)
        self._bus = MessageBus(team_dir=self._team_dir)
        # 必须把 Popen 对象存下来，
        # shutdown_all 靠它们才能找到子进程去 terminate——只记 pid 不够
        self._processes: dict = {}

    def _load_registry(self) -> dict:
        if not self._registry_path.exists():
            return {"members": []}
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"members": []}

    def _save_registry(self, registry: dict) -> None:
        """把花名册写回硬盘（原子写：先写临时文件再替换，中途断电不丢原文件）。

        注意：这里不加锁——约定由调用者先拿锁再调（_with_lock 包装）。

        参数：registry：完整的花名册 dict。
        """
        from agent.atomic_io import atomic_write_text
        atomic_write_text(
            self._registry_path,
            json.dumps(registry, ensure_ascii=False, indent=2),
        )

    def _max_members(self) -> int:
        return self._config.get("team", {}).get("max_members", 10)

    # ---- 登记 ----
    def register(self, *, name: str, role: str,
                 status: str = "running") -> TeamMember:
        """新工人入队登记（写进花名册）。

        参数：
            name：成员名（队内唯一，重名抛 ValueError）
            role：角色（干什么的，如 coder/reviewer）
            status：初始状态，默认 "running"

        返回：新造的 TeamMember 档案。
        队伍满员（超过配置的 team.max_members，默认 10）抛 RuntimeError。
        """
        def _add():
            reg = self._load_registry()
            if any(m["name"] == name for m in reg["members"]):
                raise ValueError(f"name exists: {name}")
            if len(reg["members"]) >= self._max_members():
                raise RuntimeError(
                    f"max members reached ({self._max_members()})"
                )
            member = TeamMember(
                name=name, role=role, pid=None, status=status,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            reg["members"].append(asdict(member))
            self._save_registry(reg)
            return member
        return _with_lock(self._registry_lock, _add)

    def spawn(self, *, name: str, role: str, task: str,
              depth: int = 1,
              task_id: Optional[str] = None,
              command: Optional[list] = None) -> TeamMember:
        """招工上岗：登记 + 拉起一个真正的子进程跑子 agent。

        参数：
            name：成员名
            role：角色
            task：给它的任务文本
            depth：嵌套层级（防子进程再 spawn 子进程无限套娃），
                主 agent 是 0，默认 1（第一层子 agent）
            task_id：可选，绑定的任务 ID。给了就做两件事：
                1. 启动前先到任务库 claim（认领，owner 写成成员名，持久化）
                2. 把任务 ID 塞进子进程环境变量 OMNIMATE_KANBAN_TASK
                   （工人只准动自己名下的任务，见 task_binding.py）
            command：可选，自定义启动命令；不传就用默认的
                agent.team.worker 入口

        返回：TeamMember 档案（带上了 pid）。
        出错处理：task_id 不存在或进程起不来时，花名册标 failed 并抛异常。
        """
        # 先登记再启动：万一启动失败，花名册里也能查到这次尝试
        member = self.register(name=name, role=role, status="spawning")
        member.task = task

        cmd = command or [
            sys.executable, "-m", "agent.team.worker",
            "--name", name,
            "--task", task,
            "--team-dir", str(self._team_dir),
            "--agent-home", str(self._omnimate_home),
            "--depth", str(depth),
        ]

        # 任务绑定两步：先认领（持久化），再塞工牌进环境变量（进程级）
        env = os.environ.copy()
        if task_id is not None:
            from agent.task_store import get_task_store
            from agent.team.task_binding import ENV_VAR
            store = get_task_store(omnimate_home=str(self._omnimate_home))
            claimed = store.claim(task_id, owner=name)
            if claimed is None:
                self.update_status(name, "failed")
                raise ValueError(f"task_id {task_id} 不存在")
            env[ENV_VAR] = task_id

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            member.pid = proc.pid
            self._processes[name] = proc
            self.update_status(name, "running", pid=proc.pid)
            logger.info("spawned team member %s (pid=%d, task_id=%s)",
                        name, proc.pid, task_id or "<none>")
        except OSError as e:
            logger.error("spawn 失败 %s: %s", name, e)
            self.update_status(name, "failed")
            raise

        return member

    def update_status(self, name: str, status: str,
                       pid: Optional[int] = None) -> None:
        """改档案：更新某成员的状态。

        参数：
            name：成员名
            status：新状态（running/completed/failed 等）
            pid：可选，顺带更新进程号

        无返回值；成员不存在就静默不改（保存原样花名册）。
        """
        def _update():
            reg = self._load_registry()
            for m in reg["members"]:
                if m["name"] == name:
                    m["status"] = status
                    if pid is not None:
                        m["pid"] = pid
                    break
            self._save_registry(reg)
        _with_lock(self._registry_lock, _update)

    def list_members(self) -> List[TeamMember]:
        def _read():
            reg = self._load_registry()
            return [TeamMember(**m) for m in reg["members"]]
        return _with_lock(self._registry_lock, _read)

    def shutdown(self, name: str, timeout: float = 10.0) -> bool:
        """送走一个工人：先礼后兵——发关停消息，然后干等它退出。

        参数：
            name：成员名
            timeout：最多等几秒，默认 10

        返回：True=确认退出了（本来就不在跑也算）；False=等到超时还活着。
        """
        members = self.list_members()
        target = next((m for m in members if m.name == name), None)
        if target is None:
            return False
        if target.status != "running":
            return True  # 已退出

        # 先发「请退出」的消息（温和版，给对方体面收尾的机会）
        self._bus.send(
            from_="coordinator", to=name,
            type_="shutdown", content="please exit",
        )

        # 然后干等：每半秒看一眼进程还在不在
        if target.pid:
            try:
                # 简化实现：轮询查活，不引入 wait/句柄那套复杂机制
                import time
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if not _pid_alive(target.pid):
                        self.update_status(name, "completed")
                        return True
                    time.sleep(0.5)
            except Exception as e:
                logger.warning("shutdown wait 失败: %s", e)
        return False

    def shutdown_all(self, timeout: float = 5.0) -> None:
        """全员收工：把本协调员拉起过的所有子进程挨个送走。

        什么时候用：主 agent 退出时调，防止留下一堆孤儿进程占资源。

        手法分级：先礼貌 terminate（相当于 Ctrl+C），等 timeout 秒
        还不走就 kill（强杀），再等 2 秒。最后把花名册状态改成 completed
        （改状态失败也不拦着清理下一个）。

        参数：timeout：terminate 后的宽限秒数，默认 5。无返回值。
        """
        for name, proc in list(self._processes.items()):
            if proc.poll() is not None:
                # 自己已经退了，只需补个状态
                try:
                    self.update_status(name, "completed")
                except Exception:
                    pass
                continue
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.warning("shutdown_all: %s kill 后仍未退出", name)
                logger.info("shutdown_all: 已终止 %s", name)
            except Exception as e:
                logger.warning("shutdown_all: 终止 %s 失败: %s", name, e)
            finally:
                try:
                    self.update_status(name, "completed")
                except Exception:
                    pass


def _pid_alive(pid: int) -> bool:
    """看一眼某个进程号还活着没（跨平台的简化实现）。

    参数：pid：进程号。
    返回：True=还活着；False=已经没了。
    Windows 用系统 API OpenProcess 探，类 Unix 用 kill(pid, 0) 探
    （信号 0 只探测不真发）。
    """
    try:
        if sys.platform == "win32":
            # Windows: 用 OpenProcess 检查进程是否存活
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            import os
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False
