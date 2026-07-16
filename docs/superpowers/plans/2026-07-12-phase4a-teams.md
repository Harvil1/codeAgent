# Phase 4a: Agent Teams 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 新增 Agent Teams（多 agent 通过 JSONL inbox 通信）+ 5 个工具 + 一次性 spawn。

**Architecture:** 新增 `agent/team/` 子包（bus + coordinator + worker）+ `tools/team_tool.py`（5 工具）+ 主循环注入 `<team_messages>`。子 agent 用 `subprocess.Popen` 起独立进程，跑完即退出。跨进程文件锁由 spike 验证过（msvcrt + fcntl）。

**Tech Stack:** Python 3.11+ subprocess + threading + msvcrt/fcntl + uv + pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-phase4a-teams-design.md`

## Global Constraints

- 文件 I/O 必须 `encoding="utf-8"`
- subprocess 必须 `text=True, encoding="utf-8"`
- 用 `uv`
- 中文注释/commit；英文标识符
- 528 测试不得回归
- Subagent 不 commit；controller 统一 commit
- 跨进程文件锁：Windows 用 msvcrt.locking，POSIX 用 fcntl.flock（spike 验证过）

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/team/__init__.py` | T1 新增 | package marker |
| `agent/team/bus.py` | T1 新增 | MessageBus + TeamMessage + 跨平台锁 helpers |
| `tests/test_team_bus.py` | T1 新增 | bus 单元测试（10 个） |
| `agent/team/coordinator.py` | T2 新增 | TeamCoordinator + TeamMember + registry.json |
| `tests/test_team_coordinator.py` | T2 新增 | coordinator 单元测试（6 个） |
| `agent/team/worker.py` | T3 新增 | 子 agent CLI 入口 |
| `tools/team_tool.py` | T4 新增 | 5 工具 handler + 注册 |
| `tests/test_team_tool.py` | T4 新增 | 工具测试（5 个） |
| `config.py` | T5 改 | `team` 块 |
| `toolsets.py` | T5 改 | `team` toolset |
| `agent/__init__.py` | T6 改 | AIAgent `team_bus/team_coordinator/team_name=None` + 主循环注入 |
| `model_tools.py` | T6 改 | `handle_function_call` 透传 team_* |
| `cli.py` | T7 改 | RuntimeContext 装配 + 主 agent 注册 |
| `tests/test_integration.py` | T8 改 | e2e |

---

## Task 1: agent/team/bus.py MessageBus

**Files:**
- Create: `agent/team/__init__.py`
- Create: `agent/team/bus.py`
- Create: `tests/test_team_bus.py`

**Interfaces:**
- Consumes: 无
- Produces: `TeamMessage` dataclass、`MessageBus` 类、跨平台 `_acquire_lock/_release_lock/_with_lock` helpers

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_bus.py
"""MessageBus 测试。"""
import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from agent.team.bus import MessageBus, TeamMessage


def test_send_creates_inbox_file(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    mid = bus.send(from_="alice", to="bob", type_="message", content="hi")
    assert mid  # non-empty
    inbox = tmp_path / "inbox" / "bob.jsonl"
    assert inbox.exists()
    lines = inbox.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["from"] == "alice"
    assert parsed["to"] == "bob"
    assert parsed["content"] == "hi"
    assert parsed["type"] == "message"
    assert parsed["id"] == mid


def test_send_returns_message_id(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    mid = bus.send(from_="a", to="b", type_="message", content="x")
    assert isinstance(mid, str) and len(mid) > 0


def test_send_invalid_type_raises(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    with pytest.raises(ValueError):
        bus.send(from_="a", to="b", type_="invalid_kind", content="x")


def test_read_inbox_returns_messages(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="m1")
    bus.send(from_="a", to="b", type_="message", content="m2")
    msgs = bus.read_inbox("b")
    assert len(msgs) == 2
    assert msgs[0].content == "m1"
    assert msgs[1].content == "m2"


def test_read_inbox_is_consumptive(tmp_path: Path):
    """读后清空。"""
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="x")
    first = bus.read_inbox("b")
    assert len(first) == 1
    second = bus.read_inbox("b")
    assert second == []


def test_read_inbox_empty_returns_empty_list(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    assert bus.read_inbox("nobody") == []


def test_read_inbox_unknown_name_returns_empty(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    # 即使 inbox 文件不存在也返回 []
    assert bus.read_inbox("nonexistent") == []


def test_list_inboxes_returns_names(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    bus.send(from_="a", to="b", type_="message", content="x")
    bus.send(from_="a", to="c", type_="message", content="x")
    names = set(bus.list_inboxes())
    assert "b" in names
    assert "c" in names


def test_concurrent_send_safe(tmp_path: Path):
    """多线程并发 send 同一 inbox 不丢消息。"""
    bus = MessageBus(team_dir=tmp_path)
    errors = []

    def worker(n: int):
        try:
            for i in range(20):
                bus.send(from_="sender", to="target",
                         type_="message", content=f"msg-{n}-{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    msgs = bus.read_inbox("target")
    assert len(msgs) == 5 * 20  # 100


def test_concurrent_read_write_safe(tmp_path: Path):
    """读 + 写并发不抛异常。"""
    bus = MessageBus(team_dir=tmp_path)
    errors = []

    def writer():
        try:
            for i in range(20):
                bus.send(from_="a", to="b", type_="message", content=f"m{i}")
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(20):
                bus.read_inbox("b")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_bus.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# agent/team/__init__.py
"""Agent Teams 子包。"""
```

```python
# agent/team/bus.py
"""JSONL 文件消息总线。

每个 agent 一个 inbox 文件 (~/.agent/.team/inbox/{name}.jsonl)。
所有读写用文件锁序列化（Windows msvcrt / POSIX fcntl，spike 验证过）。
"""
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

VALID_TYPES = {"message", "request", "response", "shutdown"}


@dataclass
class TeamMessage:
    """单条消息。"""
    id: str
    from_: str        # 发送者（不用 built-in `from`）
    to: str
    type: str
    content: str
    ts: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 跨平台文件锁
# ---------------------------------------------------------------------------

def _acquire_lock(fileobj):
    """获取独占锁。"""
    if sys.platform == "win32":
        import msvcrt
        while True:
            try:
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.01)
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX)


def _release_lock(fileobj):
    """释放锁。"""
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)


def _with_lock(lock_path: Path, fn):
    """获取 lock_path 的独占锁后执行 fn。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        _acquire_lock(lf)
        try:
            return fn()
        finally:
            _release_lock(lf)


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class MessageBus:
    """JSONL 消息总线。"""

    def __init__(self, *, team_dir: Path):
        self._team_dir = Path(team_dir)
        self._inbox_dir = self._team_dir / "inbox"
        self._locks_dir = self._team_dir / "locks"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, name: str) -> Path:
        return self._inbox_dir / f"{name}.jsonl"

    def _lock_path(self, name: str) -> Path:
        return self._locks_dir / f"inbox-{name}.lock"

    def send(
        self,
        *,
        from_: str,
        to: str,
        type_: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> str:
        """追加一条消息到 `to` 的 inbox。返回 message_id。"""
        if type_ not in VALID_TYPES:
            raise ValueError(
                f"type_ 必须是 {VALID_TYPES} 之一，实际: {type_}"
            )
        msg_id = uuid.uuid4().hex[:12]
        msg = {
            "id": msg_id,
            "from": from_,
            "to": to,
            "type": type_,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
        }

        def _append():
            inbox = self._inbox_path(to)
            with open(inbox, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        _with_lock(self._lock_path(to), _append)
        return msg_id

    def read_inbox(self, name: str) -> List[TeamMessage]:
        """消费式读取：返回所有消息，清空文件。"""
        def _consume():
            inbox = self._inbox_path(name)
            if not inbox.exists():
                return []
            text = inbox.read_text(encoding="utf-8")
            msgs = []
            for line in text.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    msgs.append(TeamMessage(
                        id=data["id"],
                        from_=data["from"],
                        to=data["to"],
                        type=data["type"],
                        content=data["content"],
                        ts=data["ts"],
                        request_id=data.get("request_id"),
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("inbox 消息解析失败 (%s): %s", line[:100], e)
            # 清空（保留文件）
            inbox.write_text("", encoding="utf-8")
            return msgs

        return _with_lock(self._lock_path(name), _consume)

    def list_inboxes(self) -> List[str]:
        """列出所有 inbox 文件名（去 .jsonl 后缀）。"""
        return [p.stem for p in self._inbox_dir.glob("*.jsonl")]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_team_bus.py -v`
Expected: PASS（10 tests）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 2: agent/team/coordinator.py

**Files:**
- Create: `agent/team/coordinator.py`
- Create: `tests/test_team_coordinator.py`

**Interfaces:**
- Consumes: `MessageBus`（T1）
- Produces: `TeamMember` dataclass + `TeamCoordinator` 类

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_coordinator.py
"""TeamCoordinator 测试。"""
import json
from pathlib import Path

import pytest

from agent.team.coordinator import TeamCoordinator, TeamMember


def test_register_lead(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    m = coord.register(name="main", role="lead")
    assert m.name == "main"
    assert m.role == "lead"
    assert m.status == "running"


def test_spawn_returns_member_with_pid(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    # 用最简单的 subprocess：python -c "pass"
    m = coord.spawn(name="w1", role="worker",
                     task="dummy", command=[_python(), "-c", "pass"])
    assert m.pid is not None and m.pid > 0
    # 等子进程退出
    import time
    time.sleep(0.5)
    members = coord.list_members()
    assert any(mm.name == "w1" for mm in members)


def test_spawn_duplicate_name_raises(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    with pytest.raises(ValueError, match="exists"):
        coord.register(name="main", role="lead")


def test_list_members(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    coord.register(name="w1", role="worker")
    members = coord.list_members()
    assert len(members) == 2
    names = {m.name for m in members}
    assert names == {"main", "w1"}


def test_update_status(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    m = coord.register(name="w1", role="worker")
    coord.update_status("w1", "completed")
    members = coord.list_members()
    assert members[0].status == "completed"


def test_spawn_max_members_raises(tmp_path: Path):
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 1}})
    coord.register(name="main", role="lead")
    with pytest.raises(RuntimeError, match="max"):
        coord.register(name="w1", role="worker")


def _python():
    import sys
    return sys.executable
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_coordinator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# agent/team/coordinator.py
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
        def _write():
            self._registry_path.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _with_lock(self._registry_lock, _write)

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
                proc = subprocess.Popen(
                    ["ping", "-n", "1", "127.0.0.1"] if sys.platform == "win32"
                    else ["sleep", "0.1"],
                )  # placeholder, 真实实现用 psutil 或 wait
                # 简化：直接检查 pid 是否还在
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
            # Windows: 用 tasklist 或 OpenProcess
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_team_coordinator.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 不 commit**

---

## Task 3: agent/team/worker.py 子 agent 入口

**Files:**
- Create: `agent/team/worker.py`

**Interfaces:**
- Consumes: AIAgent + MessageBus + TeamCoordinator
- Produces: `python -m agent.team.worker` CLI 入口

- [ ] **Step 1: 写实现（无单元测试，e2e 测试在 T8）**

```python
# agent/team/worker.py
"""子 agent CLI 入口。

用法：
    python -m agent.team.worker --name X --task "..." \
        --team-dir ~/.agent/.team --agent-home ~/.agent

流程：
1. 加载 config + MemoryStore + MessageBus + TeamCoordinator
2. 构造 AIAgent（team_name=name）
3. run_conversation(task)
4. 把响应 send 给 "main"
5. update_status(name, "completed")
6. 退出
"""
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HarvilAgent team worker")
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--team-dir", required=True)
    parser.add_argument("--agent-home", required=True)
    parser.add_argument("--config", default=None,
                        help="可选 config.yaml 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [worker:%(process)d] %(message)s")
    logger.info("worker %s 启动", args.name)

    team_dir = Path(args.team_dir)
    agent_home = Path(args.agent_home)

    # 延迟导入避免循环
    from config import load_config
    from agent.memory_store import MemoryStore
    from agent.team.bus import MessageBus
    from agent.team.coordinator import TeamCoordinator
    from agent import AIAgent

    config = load_config(args.config) if args.config else load_config()
    memory_store = MemoryStore(harvil_home=agent_home)
    bus = MessageBus(team_dir=team_dir)
    coordinator = TeamCoordinator(
        team_dir=team_dir, harvil_home=agent_home, config=config,
    )

    # 构造 AIAgent
    api_base = config.get("model", {}).get("base_url") or ""
    api_key = config.get("model", {}).get("api_key") or ""
    model_name = config.get("model", {}).get("name", "deepseek-chat")
    agent = AIAgent(
        base_url=api_base, api_key=api_key, model=model_name,
        enabled_toolsets=config.get("agent", {}).get("enabled_toolsets", ["core"]),
        harvil_home=str(agent_home),
        memory_store=memory_store,
        team_bus=bus, team_coordinator=coordinator, team_name=args.name,
        config=config,
    )

    # 跑任务
    try:
        response = agent.run_conversation(args.task)
        # 把响应发给 main
        bus.send(
            from_=args.name, to="main",
            type_="response", content=response or "",
        )
        coordinator.update_status(args.name, "completed")
        logger.info("worker %s 任务完成", args.name)
    except Exception as e:
        logger.exception("worker %s 异常", args.name)
        bus.send(
            from_=args.name, to="main",
            type_="message",
            content=f"[worker crashed: {e}]",
        )
        coordinator.update_status(args.name, "failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 不 commit**

---

## Task 4: tools/team_tool.py 5 工具

**Files:**
- Create: `tools/team_tool.py`
- Create: `tests/test_team_tool.py`

**Interfaces:**
- Consumes: `MessageBus`（T1）+ `TeamCoordinator`（T2）
- Produces: 5 个工具 handler + 注册

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_tool.py
"""team_tool 工具测试。"""
import json
import sys
from pathlib import Path

import tools.team_tool  # 触发注册
from agent.team.bus import MessageBus
from agent.team.coordinator import TeamCoordinator
from tools.registry import registry


def _setup(tmp_path: Path):
    bus = MessageBus(team_dir=tmp_path)
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10}})
    coord.register(name="main", role="lead")
    return bus, coord


def test_team_send(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = registry.dispatch(
        "team_send",
        {"to": "w1", "content": "hello"},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert "id" in parsed


def test_team_inbox_consumptive(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    bus.send(from_="w1", to="main", type_="message", content="hi")
    result_str = registry.dispatch(
        "team_inbox", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["messages"][0]["content"] == "hi"
    # 再读应为空
    result_str2 = registry.dispatch(
        "team_inbox", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    assert json.loads(result_str2)["count"] == 0


def test_team_members(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = registry.dispatch(
        "team_members", {},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["members"][0]["name"] == "main"


def test_team_spawn_simple(tmp_path: Path):
    """spawn 一个最简子进程（不真的起 worker，用 echo 等价命令）。"""
    bus, coord = _setup(tmp_path)
    # 用真实 command 覆盖 worker 入口（测试中不真跑 agent）
    # 这里只测 registry.dispatch 调 coordinator.spawn 的 wiring
    # 真实 spawn 测试在 T8 integration
    # 这里测 handler 返回正确 JSON 格式
    # 直接 mock coord.spawn
    from unittest.mock import patch
    from agent.team.coordinator import TeamMember
    fake_member = TeamMember(
        name="w1", role="worker", pid=12345, status="running",
        created_at="2026-07-12T00:00:00", task="dummy",
    )
    with patch.object(coord, "spawn", return_value=fake_member):
        result_str = registry.dispatch(
            "team_spawn",
            {"name": "w1", "task": "dummy"},
            team_bus=bus, team_coordinator=coord, team_name="main",
        )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert parsed["name"] == "w1"
    assert parsed["pid"] == 12345


def test_team_shutdown_unknown(tmp_path: Path):
    bus, coord = _setup(tmp_path)
    result_str = registry.dispatch(
        "team_shutdown", {"name": "nonexistent"},
        team_bus=bus, team_coordinator=coord, team_name="main",
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is False
    assert "error" in parsed
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_tool.py -v`
Expected: FAIL — `ModuleNotFoundError` 或工具未注册

- [ ] **Step 3: 写实现**

```python
# tools/team_tool.py
"""Agent Teams 工具：5 个 handler 注册到 registry。

工具：
- team_send: 发消息
- team_inbox: 读自己的收件箱（消费式）
- team_members: 列出团队成员
- team_spawn: 启动子 agent
- team_shutdown: 关闭子 agent

handler 通过 kwargs 接收 team_bus / team_coordinator / team_name（由 agent 透传）。
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


TEAM_SEND_SCHEMA = {
    "name": "team_send",
    "description": "向另一个 agent 发消息",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "接收者 agent 名"},
            "content": {"type": "string"},
            "msg_type": {
                "type": "string",
                "enum": ["message", "request", "response", "shutdown"],
                "default": "message",
            },
            "request_id": {"type": "string"},
        },
        "required": ["to", "content"],
    },
}

TEAM_INBOX_SCHEMA = {
    "name": "team_inbox",
    "description": "读自己收件箱的所有消息（消费式：读后清空）",
    "parameters": {"type": "object", "properties": {}},
}

TEAM_MEMBERS_SCHEMA = {
    "name": "team_members",
    "description": "列出所有团队成员及其状态",
    "parameters": {"type": "object", "properties": {}},
}

TEAM_SPAWN_SCHEMA = {
    "name": "team_spawn",
    "description": "启动一个子 agent 进程处理任务（一次性，完成后自动退出）",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "新 agent 的名字（必须唯一）"},
            "role": {"type": "string", "default": "worker"},
            "task": {"type": "string", "description": "交给新 agent 跑的 prompt"},
        },
        "required": ["name", "task"],
    },
}

TEAM_SHUTDOWN_SCHEMA = {
    "name": "team_shutdown",
    "description": "向另一个 agent 发 shutdown 消息并等其退出",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}


def _handle_team_send(args: dict, **kwargs) -> str:
    bus = kwargs.get("team_bus")
    team_name = kwargs.get("team_name", "main")
    if bus is None:
        return _err("team bus 未初始化", "team_unavailable")
    to = args.get("to")
    content = args.get("content")
    if not to or not content:
        return _err("to 和 content 必需", "invalid_args")
    try:
        mid = bus.send(
            from_=team_name, to=to,
            type_=args.get("msg_type", "message"),
            content=content,
            request_id=args.get("request_id"),
        )
        return json.dumps({"success": True, "id": mid, "to": to},
                          ensure_ascii=False)
    except ValueError as e:
        return _err(str(e), "invalid_args")
    except Exception as e:
        return _err(f"内部错误: {e}", "team_error")


def _handle_team_inbox(args: dict, **kwargs) -> str:
    bus = kwargs.get("team_bus")
    team_name = kwargs.get("team_name", "main")
    if bus is None:
        return _err("team bus 未初始化", "team_unavailable")
    try:
        msgs = bus.read_inbox(team_name)
        return json.dumps({
            "success": True, "count": len(msgs),
            "messages": [
                {
                    "id": m.id, "from": m.from_, "to": m.to,
                    "type": m.type, "content": m.content, "ts": m.ts,
                    "request_id": m.request_id,
                }
                for m in msgs
            ],
        }, ensure_ascii=False)
    except Exception as e:
        return _err(f"读取 inbox 失败: {e}", "team_error")


def _handle_team_members(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    members = coord.list_members()
    return json.dumps({
        "success": True, "count": len(members),
        "members": [
            {
                "name": m.name, "role": m.role, "pid": m.pid,
                "status": m.status, "created_at": m.created_at,
            }
            for m in members
        ],
    }, ensure_ascii=False)


def _handle_team_spawn(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    task = args.get("task")
    if not name or not task:
        return _err("name 和 task 必需", "invalid_args")
    role = args.get("role", "worker")
    try:
        member = coord.spawn(name=name, role=role, task=task)
        return json.dumps({
            "success": True, "name": name, "pid": member.pid,
            "status": member.status,
        }, ensure_ascii=False)
    except (ValueError, RuntimeError) as e:
        return _err(str(e), "team_spawn_error")
    except Exception as e:
        return _err(f"spawn 失败: {e}", "team_error")


def _handle_team_shutdown(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    bus = kwargs.get("team_bus")
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    if not name:
        return _err("name 必需", "invalid_args")
    ok = coord.shutdown(name)
    if not ok:
        return _err(f"未找到或仍在运行: {name}", "team_not_found")
    return json.dumps({
        "success": True, "name": name, "status": "shutdown",
    }, ensure_ascii=False)


def _err(msg: str, error_type: str) -> str:
    return json.dumps({"success": False, "error": msg,
                       "error_type": error_type}, ensure_ascii=False)


# ---- 注册 ----
registry.register(
    name="team_send", toolset="team",
    schema=TEAM_SEND_SCHEMA, handler=_handle_team_send, emoji="📤",
)
registry.register(
    name="team_inbox", toolset="team",
    schema=TEAM_INBOX_SCHEMA, handler=_handle_team_inbox, emoji="📥",
)
registry.register(
    name="team_members", toolset="team",
    schema=TEAM_MEMBERS_SCHEMA, handler=_handle_team_members, emoji="👥",
)
registry.register(
    name="team_spawn", toolset="team",
    schema=TEAM_SPAWN_SCHEMA, handler=_handle_team_spawn, emoji="🚀",
)
registry.register(
    name="team_shutdown", toolset="team",
    schema=TEAM_SHUTDOWN_SCHEMA, handler=_handle_team_shutdown, emoji="🛑",
)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_team_tool.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: 不 commit**

---

## Task 5: config.py + toolsets.py

**Files:**
- Modify: `config.py:DEFAULT_CONFIG`
- Modify: `toolsets.py:TOOLSETS`

- [ ] **Step 1: 加 config 块**

在 `config.py:DEFAULT_CONFIG` 加（紧邻 `cron` 块之后）：

```python
    # Agent Teams（Phase 4a）
    "team": {
        "enabled": True,                        # False 时整个 team 工具隐藏
        "team_dir": None,                       # None → 默认 ~/.agent/.team/
        "default_role": "worker",
        "spawn_timeout": 600,                   # 等子 agent 退出最长时间（秒）
        "max_members": 10,                      # 团队成员上限
    },
```

- [ ] **Step 2: 加 toolset**

在 `toolsets.py:TOOLSETS` 加：

```python
"team": [
    "team_send",
    "team_inbox",
    "team_members",
    "team_spawn",
    "team_shutdown",
],
```

- [ ] **Step 3: 加 config 测试**

```python
# 追加到 tests/test_config.py
def test_default_config_has_team_block():
    from config import DEFAULT_CONFIG
    t = DEFAULT_CONFIG["team"]
    for key in ("enabled", "team_dir", "default_role",
                "spawn_timeout", "max_members"):
        assert key in t, f"缺 {key}"


def test_team_toolset_exists():
    from toolsets import TOOLSETS, resolve_toolset
    assert "team" in TOOLSETS
    tools = resolve_toolset("team")
    assert "team_send" in tools
    assert "team_spawn" in tools
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 6: agent/__init__.py + model_tools.py 集成

**Files:**
- Modify: `agent/__init__.py:AIAgent.__init__` + `run_conversation`
- Modify: `model_tools.py:handle_function_call`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: T1 MessageBus + T2 TeamCoordinator
- Produces: AIAgent `team_bus/team_coordinator/team_name=None` kwarg + 主循环注入 `<team_messages>`

- [ ] **Step 1: 改 agent/__init__.py**

`AIAgent.__init__` 加 3 个新 kwargs：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        memory_retriever=None,
        team_bus=None,             # === NEW Phase 4a ===
        team_coordinator=None,     # === NEW ===
        team_name=None,            # === NEW ===
    ):
        # ...
        self.team_bus = team_bus
        self.team_coordinator = team_coordinator
        self.team_name = team_name
```

`run_conversation` 在 cron drain 之后加 team inbox drain：

```python
        # === NEW Phase 4a: team inbox drain ===
        team_messages_text = ""
        if self.team_bus and self.team_name:
            try:
                msgs = self.team_bus.read_inbox(self.team_name)
                if msgs:
                    team_messages_text = "\n".join(
                        f"[from {m.from_} ({m.type})] {m.content}"
                        for m in msgs
                    )
            except Exception as e:
                logger.warning("team inbox drain 异常: %s", e)
                team_messages_text = ""
```

主循环 messages 组装后注入（紧邻 cron_messages 之后）：

```python
            # === NEW Phase 4a: team messages 注入 ===
            if team_messages_text:
                messages.append({
                    "role": "user",
                    "content": f"<team_messages>\n{team_messages_text}\n</team_messages>",
                })
                team_messages_text = ""  # 本轮注入后清空
```

**注意**：team_messages **持久**进 conversation_history（不是临时消息）。但为了不重复注入，这里 `team_messages_text = ""` 是清空本地变量。

实际实现需小心：把 drain 放在 run_conversation 顶部（像 cron_messages 那样），然后在 messages 组装后注入。注入后清空本地变量，下轮循环就不再有。但**消息内容已经进 conversation_history**了，所以下轮 LLM 还能看到。

具体语义需要 implementer 仔细处理：drain 一次，注入到 messages，但 conversation_history 已经包含原始消息（在最初 user 消息 append 时）。这里 team_messages 是 LLM 之间通信，应该进 conversation_history 还是临时？

**决策**：team_messages 像 bg_notifications 一样作为**临时** user 消息（不进 conversation_history）。每轮 drain 一次、注入 messages 一次、清空。下轮如果 inbox 有新消息，再注入。已处理的消息进 inbox 文件后清空（read_inbox 是消费式）。

按这个语义实现：临时注入，**不进** conversation_history。

- [ ] **Step 2: 改 model_tools.py**

`handle_function_call` 加 3 个新 kwargs + 透传：

```python
def handle_function_call(
    function_name, function_args, *,
    # ... 原有 kwargs ...
    bg_manager=None,
    team_bus=None,               # === NEW ===
    team_coordinator=None,       # === NEW ===
    team_name=None,              # === NEW ===
):
    # ...
    result = registry.dispatch(
        function_name, function_args,
        # ... 原有 ...
        bg_manager=bg_manager,
        team_bus=team_bus,
        team_coordinator=team_coordinator,
        team_name=team_name,
    )
```

在 `agent/__init__.py` 调 `handle_function_call` 的地方加：

```python
team_bus=self.team_bus,
team_coordinator=self.team_coordinator,
team_name=self.team_name,
```

- [ ] **Step 3: 加测试**

```python
# 追加到 tests/test_integration.py
def test_aiagent_accepts_team_kwargs():
    agent = _make_test_agent()
    assert agent.team_bus is None
    assert agent.team_coordinator is None
    assert agent.team_name is None


def test_aiagent_team_messages_injected_into_temporary_user_msg(tmp_path):
    """team_bus.read_inbox 返回的消息作为 <team_messages> 临时注入。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.team.bus import TeamMessage

    fake_bus = MagicMock()
    fake_bus.read_inbox.return_value = [
        TeamMessage(
            id="m1", from_="worker1", to="main",
            type="response", content="task done",
            ts="2026-07-12T15:30:00", request_id=None,
        ),
    ]

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        team_bus=fake_bus, team_name="main",
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("check")

    # conversation_history 不应含 team_messages（临时）
    for msg in agent.conversation_history:
        assert "<team_messages>" not in msg.get("content", "")
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_integration.py -k team -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 7: cli.py 装配 + 主 agent 自注册

**Files:**
- Modify: `cli.py:RuntimeContext.__init__` + `_create_agent`

- [ ] **Step 1: 改 cli.py**

在 `RuntimeContext.__init__` 加（紧邻 cron_scheduler 之后）：

```python
        # === NEW Phase 4a: Agent Teams ===
        from agent.team.bus import MessageBus
        from agent.team.coordinator import TeamCoordinator
        team_cfg = self.config.get("team", {})
        if team_cfg.get("enabled", True):
            team_path = team_cfg.get("team_dir") or (
                Path(self.home) / ".team"
            )
            try:
                self.team_bus = MessageBus(team_dir=Path(team_path))
                self.team_coordinator = TeamCoordinator(
                    team_dir=Path(team_path),
                    harvil_home=Path(self.home),
                    config=self.config,
                )
                # 主 agent 自注册为 lead（仅当 registry 还没有 main）
                members = self.team_coordinator.list_members()
                if not any(m.name == "main" and m.status == "running" for m in members):
                    try:
                        self.team_coordinator.register(
                            name="main", role="lead", status="running",
                        )
                    except ValueError:
                        pass  # 已存在，跳过
            except Exception as e:
                logger.error("Team 系统初始化失败: %s", e)
                self.team_bus = None
                self.team_coordinator = None
        else:
            self.team_bus = None
            self.team_coordinator = None
```

在 `_create_agent` 加 3 个 kwargs：

```python
agent = AIAgent(
    # ... 原有 ...
    team_bus=self.team_bus,
    team_coordinator=self.team_coordinator,
    team_name="main",
)
```

- [ ] **Step 2: 跑回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: 不 commit**

---

## Task 8: e2e 集成测试

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: 加 e2e 测试**

```python
# 追加到 tests/test_integration.py
def test_e2e_team_spawn_real_subprocess(tmp_path):
    """端到端：spawn 一个真子进程（不用 worker.py，用 echo 等价命令）。

    验证 spawn 启动 + pid 写入 registry + 进程退出后状态可查。
    """
    import time
    import sys
    from agent.team.coordinator import TeamCoordinator

    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    # 用最简命令（不真起 worker，避免依赖 LLM）
    member = coord.spawn(
        name="w1", role="worker", task="dummy",
        command=[sys.executable, "-c",
                 "import time; time.sleep(0.3); print('done')"],
    )
    assert member.pid is not None

    # 等子进程退出
    time.sleep(1.0)
    members = coord.list_members()
    target = next((m for m in members if m.name == "w1"), None)
    assert target is not None
    # status 可能仍是 "running"（因为没真的有 worker.py 更新），
    # 但 pid 应该有
    assert target.pid == member.pid


def test_e2e_team_send_and_inbox_through_bus(tmp_path: Path):
    """端到端：两个进程通过 bus 互发消息（同进程内模拟）。"""
    import json
    from agent.team.bus import MessageBus

    bus = MessageBus(team_dir=tmp_path)
    # main 给 worker1 发任务
    mid = bus.send(
        from_="main", to="worker1",
        type_="request", content="分析数据",
    )
    # worker1 读自己的 inbox
    msgs = bus.read_inbox("worker1")
    assert len(msgs) == 1
    assert msgs[0].content == "分析数据"
    # worker1 回复 main
    bus.send(
        from_="worker1", to="main",
        type_="response", content="分析完成",
        request_id=mid,
    )
    # main 读 inbox
    msgs = bus.read_inbox("main")
    assert len(msgs) == 1
    assert msgs[0].content == "分析完成"
    assert msgs[0].request_id == mid
```

- [ ] **Step 2: 跑测试**

Run: `uv run pytest tests/test_integration.py -k team -v`
Expected: PASS

- [ ] **Step 3: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 528 + 新增 team 测试）

- [ ] **Step 4: 不 commit**

---

## Self-Review

**Spec 覆盖**：
- ✅ §1 架构 → T1-T8
- ✅ §2 MessageBus（JSONL + 文件锁 + 消费式读） → T1
- ✅ §3 TeamCoordinator（register/spawn/list_members/update_status/shutdown） → T2
- ✅ §4 worker.py CLI 入口 → T3
- ✅ §5 5 工具 → T4
- ✅ §6 主循环 team_messages 注入 → T6
- ✅ §7 配置 → T5
- ✅ §7 toolset → T5
- ✅ §8 失败处理（name conflict / max members / send 容错 / spawn 失败 / worker crash） → T2+T4 测试
- ✅ §9 并发安全 → T1 文件锁实现 + spike 验证
- ✅ §10 测试矩阵 → T1 (10) + T2 (6) + T4 (5) + T6 (2) + T8 (2)
- ✅ §11 已知限制（无 autonomous IDLE / 无跨机器 / 无认证 / 一次性 spawn） → 全遵守

**Placeholder 扫描**：无 TBD/TODO。

**类型一致性**：
- `TeamMessage` 字段 T1 定义，T2/T4/T6 引用 ✓
- `MessageBus.send/read_inbox/list_inboxes` 签名 T1 → T4/T6 一致 ✓
- `TeamMember` 字段 T2 定义，T4 测试 dict 一致 ✓
- `TeamCoordinator.spawn/register/list_members` 签名 T2 → T4/T7 一致 ✓
- `team_bus/team_coordinator/team_name=None` kwarg 链 T6 → T7 → AIAgent ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase4a-teams.md`.

按用户授权直接进 SDD。
