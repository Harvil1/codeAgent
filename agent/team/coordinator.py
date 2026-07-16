"""团队成员注册 + spawn 管理。

registry.json 持久化所有成员状态。spawn 用 subprocess.Popen 起子进程。
"""
import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agent.team.bus import MessageBus, _with_lock

logger = logging.getLogger(__name__)


@dataclass
class TeamMember:
    """单个团队成员。"""
    name: str
    role: str
    pid: Optional[int]
    status: str             # "spawning" / "running" / "completed" / "failed"
    created_at: str
    task: Optional[str] = None


class TeamCoordinator:
    """团队成员管理。"""

    def __init__(self, *, team_dir: Path, harvil_home: Path,
                 config: dict):
        self._team_dir = Path(team_dir)
        self._harvil_home = Path(harvil_home)
        self._config = config
        self._registry_path = self._team_dir / "registry.json"
        self._registry_lock = self._team_dir / "registry.lock"
        self._team_dir.mkdir(parents=True, exist_ok=True)
        self._bus = MessageBus(team_dir=self._team_dir)

    def _load_registry(self) -> dict:
        if not self._registry_path.exists():
            return {"members": []}
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"members": []}

    def _save_registry(self, registry: dict) -> None:
        """写入 registry（不加锁——调用者负责持锁）。"""
        self._registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _max_members(self) -> int:
        return self._config.get("team", {}).get("max_members", 10)

    # ---- 注册 ----
    def register(self, *, name: str, role: str,
                 status: str = "running") -> TeamMember:
        """注册成员。名字冲突抛 ValueError。超上限抛 RuntimeError。"""
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
              command: Optional[list] = None) -> TeamMember:
        """启动子 agent 进程。command 默认是 agent.team.worker 入口。"""
        # 先注册（status=spawning）
        member = self.register(name=name, role=role, status="spawning")
        member.task = task

        cmd = command or [
            sys.executable, "-m", "agent.team.worker",
            "--name", name,
            "--task", task,
            "--team-dir", str(self._team_dir),
            "--agent-home", str(self._harvil_home),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            member.pid = proc.pid
            self.update_status(name, "running")
            logger.info("spawned team member %s (pid=%d)", name, proc.pid)
        except OSError as e:
            logger.error("spawn 失败 %s: %s", name, e)
            self.update_status(name, "failed")
            raise

        return member

    def update_status(self, name: str, status: str,
                       pid: Optional[int] = None) -> None:
        """更新成员状态。"""
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
        reg = self._load_registry()
        return [TeamMember(**m) for m in reg["members"]]

    def shutdown(self, name: str, timeout: float = 10.0) -> bool:
        """发 shutdown 消息 + 等 pid 退出。"""
        members = self.list_members()
        target = next((m for m in members if m.name == name), None)
        if target is None:
            return False
        if target.status != "running":
            return True  # 已退出

        # 发 shutdown 消息
        self._bus.send(
            from_="coordinator", to=name,
            type_="shutdown", content="please exit",
        )

        # 等 pid 退出
        if target.pid:
            try:
                # 简化：直接轮询检查 pid 是否还在
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


def _pid_alive(pid: int) -> bool:
    """检查 pid 是否还存活。跨平台粗糙实现。"""
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
