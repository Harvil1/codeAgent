# Phase 4b: Autonomous Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 给 worker 加 WORK/IDLE/SHUTDOWN 三态生命周期；`--autonomous` 启用；IDLE 轮询 inbox 找新工作；60s 无事则退出；`depth` 防递归 spawn。

**Architecture:** 新增 `agent/team/lifecycle.py`（AutonomousLifecycle 类）+ 改 `tools/team_tool.py`（加 `idle` 工具）+ 改 `agent/team/worker.py`（加 `--autonomous` 模式）+ 改 `agent/__init__.py`（`_idle_requested` 标志）+ depth 防 spawn 失控。

**Tech Stack:** Python 3.11+、uv、pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-phase4b-autonomous-design.md`

## Global Constraints

- 文件 I/O `encoding="utf-8"`
- 用 `uv`
- 中文注释/commit；英文标识符
- 556 测试不得回归
- Subagent 不 commit
- IDLE 轮询 sleep 用 `time.sleep(poll_interval)`，不引入 asyncio

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/team/lifecycle.py` | T1 新增 | AutonomousLifecycle 状态机 |
| `tests/test_team_lifecycle.py` | T1 新增 | 5 单元测试 |
| `tools/team_tool.py` | T2 改 | 加 `idle` 工具 + spawn depth 检查 |
| `agent/__init__.py` | T2 改 | AIAgent `_idle_requested` + `spawn_depth` |
| `agent/team/worker.py` | T3 改 | `--autonomous` + `--depth` + lifecycle 集成 |
| `config.py` | T4 改 | team.autonomous_idle_timeout / max_depth |
| `tests/test_integration.py` | T5 改 | idle 工具 + max_depth e2e |

---

## Task 1: agent/team/lifecycle.py AutonomousLifecycle

**Files:**
- Create: `agent/team/lifecycle.py`
- Create: `tests/test_team_lifecycle.py`

**Interfaces:**
- Consumes: 无（纯状态机）
- Produces: `STATE_WORK/STATE_IDLE/STATE_SHUTDOWN` 常量 + `AutonomousLifecycle` 类

- [ ] **Step 1: 写失败测试**

```python
# tests/test_team_lifecycle.py
"""AutonomousLifecycle 测试。"""
import time
from unittest.mock import MagicMock

import pytest

from agent.team.lifecycle import (
    AutonomousLifecycle, STATE_WORK, STATE_IDLE, STATE_SHUTDOWN,
)


def test_lifecycle_work_then_shutdown_no_messages():
    """无 inbox 消息 + 无任务 → IDLE 超时 → SHUTDOWN。"""
    work_fn = MagicMock(return_value="done")
    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.2,  # 短超时
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="hello")
    work_fn.assert_called_once_with("hello")
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_idle_picks_up_inbox_message():
    """IDLE 中 inbox 来消息 → 回 WORK。"""
    from agent.team.bus import TeamMessage
    calls = {"count": 0, "tasks": []}

    def work_fn(task):
        calls["tasks"].append(task)
        calls["count"] += 1
        return "ok"

    msg = TeamMessage(
        id="m1", from_="main", to="worker",
        type="request", content="next task",
        ts="2026-07-12T15:00:00", request_id=None,
    )
    inbox_states = [[], [msg], []]  # 第 1 次（初始）空，第 2 次有消息，第 3 次空
    state_idx = {"i": 0}

    def poll_inbox():
        i = state_idx["i"]
        state_idx["i"] += 1
        return inbox_states[i] if i < len(inbox_states) else []

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=poll_inbox,
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.2,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="initial")
    # 应该跑了 2 次（initial + next task）
    assert calls["count"] == 2
    assert "next task" in calls["tasks"]
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_work_exception_continues_to_idle():
    """work_fn 抛异常 → 进 IDLE 不崩。"""
    call_count = {"n": 0}

    def work_fn(task):
        call_count["n"] += 1
        raise RuntimeError("boom")

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.1,
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="x")
    assert call_count["n"] == 1  # 异常后进 IDLE，超时后 SHUTDOWN，不再调 work
    assert lifecycle.state == STATE_SHUTDOWN


def test_lifecycle_multiple_work_cycles():
    """多次 WORK→IDLE→WORK 循环（inbox 给 2 条消息）。"""
    from agent.team.bus import TeamMessage
    work_count = {"n": 0}

    def work_fn(task):
        work_count["n"] += 1
        return "ok"

    msgs = [
        TeamMessage(id="m1", from_="main", to="w",
                    type="message", content="t1", ts="x", request_id=None),
        TeamMessage(id="m2", from_="main", to="w",
                    type="message", content="t2", ts="x", request_id=None),
    ]
    # inbox 序列：空 → m1 → m2 → 空 → 空...
    seq = [[], [msgs[0]], [msgs[1]], []]
    idx = {"i": 0}

    def poll_inbox():
        i = idx["i"]
        idx["i"] += 1
        return seq[i] if i < len(seq) else []

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=poll_inbox,
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        idle_timeout=0.15,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="initial")
    # initial + t1 + t2 = 3 次工作
    assert work_count["n"] == 3


def test_lifecycle_on_shutdown_called():
    """SHUTDOWN 时调 on_shutdown_fn。"""
    called = {"x": False}
    def on_shutdown():
        called["x"] = True
    lifecycle = AutonomousLifecycle(
        work_fn=lambda t: None,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: True,
        on_shutdown_fn=on_shutdown,
        idle_timeout=0.05,
        poll_interval=0.02,
    )
    lifecycle.run(initial_task="x")
    assert called["x"] is True
    assert lifecycle.state == STATE_SHUTDOWN
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_lifecycle.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 写实现**

```python
# agent/team/lifecycle.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_team_lifecycle.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 2: idle 工具 + AIAgent `_idle_requested` + spawn depth 检查

**Files:**
- Modify: `tools/team_tool.py`（加 idle 工具 + spawn depth 检查）
- Modify: `agent/__init__.py`（`_idle_requested` 标志 + `spawn_depth` 属性 + run_conversation 检查标志）
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: T1 lifecycle（间接）
- Produces: `idle` 工具 + AIAgent._idle_requested/spawn_depth

- [ ] **Step 1: 改 agent/__init__.py**

`AIAgent.__init__` 加 2 个新字段：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        team_name=None,
        spawn_depth: int = 0,      # === NEW Phase 4b ===
    ):
        # ... 原有 ...
        self._idle_requested = False     # === NEW ===
        self.spawn_depth = spawn_depth   # === NEW ===
```

`run_conversation` 在每轮 LLM 调用后检查（找到 assistant_msg 处理之后）：

```python
            # === NEW Phase 4b: idle 标志检查 ===
            if self._idle_requested:
                logger.info("idle 已请求，退出 run_conversation")
                break
```

具体行号：在 `assistant_msg = response.choices[0].message` 处理完 tool_calls 之后、下一轮 while 检查之前加。

- [ ] **Step 2: 改 tools/team_tool.py**

加 `idle` 工具 schema + handler：

```python
IDLE_SCHEMA = {
    "name": "idle",
    "description": (
        "声明当前没有更多工作要做，进入 IDLE 状态等新任务。"
        "仅在 autonomous worker 模式下有意义；主 agent 调用是 no-op。"
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _handle_idle(args: dict, **kwargs) -> str:
    agent = kwargs.get("agent_ref")
    if agent is None:
        # 主 agent 调（无 agent_ref）：返回 ok 但实际无副作用
        return json.dumps({
            "success": True, "message": "idle requested (no agent ref)",
        }, ensure_ascii=False)
    agent._idle_requested = True
    return json.dumps({
        "success": True, "message": "idle requested",
    }, ensure_ascii=False)
```

注册：
```python
registry.register(
    name="idle", toolset="team",
    schema=IDLE_SCHEMA, handler=_handle_idle, emoji="💤",
)
```

**改 `_handle_team_spawn`** 加 depth 检查：

```python
def _handle_team_spawn(args: dict, **kwargs) -> str:
    coord = kwargs.get("team_coordinator")
    agent = kwargs.get("agent_ref")
    config = kwargs.get("config") or {}
    if coord is None:
        return _err("team coordinator 未初始化", "team_unavailable")
    name = args.get("name")
    task = args.get("task")
    if not name or not task:
        return _err("name 和 task 必需", "invalid_args")

    # === NEW Phase 4b: depth 检查 ===
    current_depth = getattr(agent, "spawn_depth", 0) if agent else 0
    max_depth = config.get("team", {}).get("max_depth", 2)
    if current_depth >= max_depth:
        return json.dumps({
            "success": False,
            "error": f"max_depth {max_depth} reached (current: {current_depth})",
            "error_type": "team_max_depth",
        }, ensure_ascii=False)

    role = args.get("role", "worker")
    try:
        # 子进程 command 加 --depth current_depth+1
        member = coord.spawn(
            name=name, role=role, task=task,
            extra_args=["--depth", str(current_depth + 1)],
            autonomous=False,  # 一次性；后续 worker 加 --autonomous
        )
        return json.dumps({
            "success": True, "name": name, "pid": member.pid,
            "status": member.status,
        }, ensure_ascii=False)
    except (ValueError, RuntimeError) as e:
        return _err(str(e), "team_spawn_error")
    except Exception as e:
        return _err(f"spawn 失败: {e}", "team_error")
```

注意：`coord.spawn` 需要支持 `extra_args` + `autonomous` 参数（T3 会改 coordinator.spawn 签名，或者 worker.py 命令行本身就接受这些参数，不需要 coord 介入）。

**简化方案**：coord.spawn 不改签名，worker.py 默认接受 `--depth` 和 `--autonomous`（默认 depth=1, autonomous=False）。team_spawn 工具只检查 depth 限制，不传 extra_args。子进程 worker.py 启动时 depth 默认 1（通过命令行 --depth），实际递归限制由 team_spawn 工具的检查决定。

更简化：worker.py 命令行加 `--depth`，team_spawn 工具根据当前 agent.spawn_depth 计算子 agent 的 depth 并写到 task 文本里？太复杂。

**最终方案**：spawn 的 command 默认带 `--depth {current+1}`。`coord.spawn` 接受 `command_override` 参数，工具层传完整 command。

具体改 `coordinator.spawn`（**T2 不动 coordinator，由 T3 在 worker.py 命令行默认接 `--depth`**）：

T2 只做两件事：
1. AIAgent 加 spawn_depth + _idle_requested
2. team_tool 加 idle 工具 + team_spawn 的 depth 检查（拿 agent.spawn_depth 比 max_depth）

子进程 spawn 的 depth 传递由 worker.py 默认参数 + coord.spawn 内部处理。**T3 改 coord.spawn** 接受 depth 参数。

- [ ] **Step 3: 改 coordinator.spawn**（移到 T2，因为 T3 只改 worker.py 入口）

`agent/team/coordinator.py:spawn` 加 `depth: int = 1` + `autonomous: bool = False` 参数：

```python
def spawn(
    self,
    *,
    name: str,
    role: str,
    task: str,
    depth: int = 1,
    autonomous: bool = False,
    command: Optional[list] = None,
) -> TeamMember:
    """启动子 agent。depth 用于递归限制，autonomous 启用 IDLE 轮询。"""
    member = self.register(name=name, role=role, status="spawning")
    member.task = task

    if command is None:
        cmd = [
            sys.executable, "-m", "agent.team.worker",
            "--name", name,
            "--task", task,
            "--team-dir", str(self._team_dir),
            "--agent-home", str(self._harvil_home),
            "--depth", str(depth),
        ]
        if autonomous:
            cmd.append("--autonomous")
    else:
        cmd = command

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        member.pid = proc.pid
        self._processes[name] = proc  # I3 已添加
        self.update_status(name, "running", pid=proc.pid)
        logger.info("spawned %s (pid=%d, depth=%d, autonomous=%s)",
                    name, proc.pid, depth, autonomous)
    except OSError as e:
        logger.error("spawn 失败 %s: %s", name, e)
        self.update_status(name, "failed")
        raise
    return member
```

- [ ] **Step 4: 改 tools/team_tool.py `_handle_team_spawn`** 传 depth 给 coord：

```python
    # 替换原 spawn 调用
    current_depth = getattr(agent, "spawn_depth", 0) if agent else 0
    member = coord.spawn(
        name=name, role=role, task=task,
        depth=current_depth + 1,
    )
```

- [ ] **Step 5: 加测试**

```python
# 追加到 tests/test_integration.py
def test_idle_tool_sets_flag():
    """调 idle 工具 → AIAgent._idle_requested = True。"""
    import tools.team_tool  # 触发注册
    from tools.registry import registry
    from agent import AIAgent

    agent = _make_test_agent()
    agent._idle_requested = False
    result_str = registry.dispatch(
        "idle", {},
        agent_ref=agent,
    )
    import json
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert agent._idle_requested is True


def test_team_spawn_max_depth_blocks(tmp_path):
    """depth >= max_depth 时 spawn 返回 team_max_depth error。"""
    import json
    import tools.team_tool
    from tools.registry import registry
    from agent import AIAgent
    from agent.team.bus import MessageBus
    from agent.team.coordinator import TeamCoordinator

    bus = MessageBus(team_dir=tmp_path)
    coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path,
                             config={"team": {"max_members": 10, "max_depth": 2}})
    coord.register(name="main", role="lead")

    # 假装主 agent 已经 depth=2
    class FakeAgent:
        spawn_depth = 2
    agent = FakeAgent()

    result_str = registry.dispatch(
        "team_spawn",
        {"name": "w1", "task": "x"},
        team_bus=bus, team_coordinator=coord, team_name="main",
        agent_ref=agent,
        config={"team": {"max_members": 10, "max_depth": 2}},
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is False
    assert parsed["error_type"] == "team_max_depth"
```

- [ ] **Step 6: handle_function_call 透传 agent_ref**

`model_tools.py:handle_function_call` 加 `agent_ref=None` kwarg + 透传 dispatch。`agent/__init__.py` 调用时传 `agent_ref=self`。

- [ ] **Step 7: 跑测试**

Run: `uv run pytest tests/test_integration.py -k "idle_tool or max_depth" -v`
Expected: PASS（2 tests）

Run: `uv run pytest tests/ -v`
Expected: PASS（无回归）

- [ ] **Step 8: 不 commit**

---

## Task 3: worker.py --autonomous + --depth

**Files:**
- Modify: `agent/team/worker.py`

- [ ] **Step 1: 改 worker.py**

```python
# agent/team/worker.py
"""子 agent CLI 入口。

用法：
    python -m agent.team.worker --name X --task "..." \
        --team-dir ~/.agent/.team --agent-home ~/.agent \
        [--autonomous] [--depth N]
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
    parser.add_argument("--config", default=None)
    parser.add_argument("--autonomous", action="store_true",
                        help="启用 IDLE 轮询模式")
    parser.add_argument("--depth", type=int, default=1,
                        help="递归 spawn 深度（主 agent=0，子 agent=1+）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [worker:%(process)d] %(message)s")
    logger.info("worker %s 启动 (depth=%d, autonomous=%s)",
                args.name, args.depth, args.autonomous)

    team_dir = Path(args.team_dir)
    agent_home = Path(args.agent_home)

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

    api_base = config.get("model", {}).get("base_url") or ""
    api_key = config.get("model", {}).get("api_key") or ""
    model_name = config.get("model", {}).get("name", "deepseek-chat")
    agent = AIAgent(
        base_url=api_base, api_key=api_key, model=model_name,
        enabled_toolsets=config.get("agent", {}).get("enabled_toolsets", ["core"]),
        harvil_home=str(agent_home),
        memory_store=memory_store,
        team_bus=bus, team_coordinator=coordinator, team_name=args.name,
        spawn_depth=args.depth,
        config=config,
    )

    if args.autonomous:
        from agent.team.lifecycle import AutonomousLifecycle
        team_cfg = config.get("team", {})
        lifecycle = AutonomousLifecycle(
            work_fn=lambda task: agent.run_conversation(task),
            poll_inbox_fn=lambda: bus.read_inbox(args.name),
            poll_tasks_fn=lambda: [],  # 暂不接入 task_store
            claim_task_fn=lambda tid: False,
            on_shutdown_fn=lambda: coordinator.update_status(args.name, "completed"),
            idle_timeout=team_cfg.get("autonomous_idle_timeout", 60.0),
            poll_interval=team_cfg.get("autonomous_poll_interval", 5.0),
        )
        lifecycle.run(initial_task=args.task)
        logger.info("worker %s autonomous 生命周期结束", args.name)
    else:
        # 一次性模式
        try:
            response = agent.run_conversation(args.task)
            bus.send(from_=args.name, to="main",
                     type_="response", content=response or "")
            coordinator.update_status(args.name, "completed")
            logger.info("worker %s 任务完成", args.name)
        except Exception as e:
            logger.exception("worker %s 异常", args.name)
            bus.send(from_=args.name, to="main",
                     type_="message",
                     content=f"[worker crashed: {e}]")
            coordinator.update_status(args.name, "failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: 不 commit**

---

## Task 4: config.py 新增字段

**Files:**
- Modify: `config.py:DEFAULT_CONFIG["team"]`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 加字段**

在 `team` 块加：

```python
        "autonomous_idle_timeout": 60.0,        # IDLE 状态超时秒数
        "autonomous_poll_interval": 5.0,        # IDLE 轮询间隔
        "max_depth": 2,                         # spawn 递归深度上限
```

- [ ] **Step 2: 加测试**

```python
# 追加到 tests/test_config.py
def test_default_config_team_autonomous_fields():
    from config import DEFAULT_CONFIG
    t = DEFAULT_CONFIG["team"]
    assert t["autonomous_idle_timeout"] == 60.0
    assert t["autonomous_poll_interval"] == 5.0
    assert t["max_depth"] == 2
```

- [ ] **Step 3: 跑测试**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 4: 不 commit**

---

## Task 5: 全量回归 + autonomous e2e（mock）

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: 加 autonomous e2e 测试（mock 短超时）**

```python
# 追加到 tests/test_integration.py
def test_e2e_autonomous_lifecycle_with_mock_agent(tmp_path):
    """端到端：用 mock agent 跑 AutonomousLifecycle，验证状态转换。

    避免 spawn 真子进程（依赖 LLM）。直接在测试进程内跑 lifecycle。
    """
    from agent.team.bus import MessageBus, TeamMessage
    from agent.team.lifecycle import (
        AutonomousLifecycle, STATE_WORK, STATE_SHUTDOWN,
    )

    bus = MessageBus(team_dir=tmp_path)
    # 给 "w1" 准备一条消息（在初始任务跑完后，IDLE 中会拿到）
    # 不预先放（测无消息超时）

    work_count = {"n": 0}
    def work_fn(task):
        work_count["n"] += 1
        return f"done-{task}"

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: bus.read_inbox("w1"),
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: False,
        on_shutdown_fn=lambda: None,
        idle_timeout=0.2,
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="initial task")
    assert work_count["n"] == 1  # 只跑了 initial，IDLE 无消息超时
    assert lifecycle.state == STATE_SHUTDOWN


def test_e2e_autonomous_lifecycle_picks_up_message_mid_idle(tmp_path):
    """IDLE 中 inbox 来消息 → 回 WORK。"""
    import threading
    import time
    from agent.team.bus import MessageBus, TeamMessage
    from agent.team.lifecycle import AutonomousLifecycle, STATE_SHUTDOWN

    bus = MessageBus(team_dir=tmp_path)

    work_log = []
    def work_fn(task):
        work_log.append(task)
        return "ok"

    # 后台线程在 100ms 后给 w1 发消息
    def delayed_msg():
        time.sleep(0.1)
        bus.send(from_="main", to="w1",
                 type_="message", content="late task")

    t = threading.Thread(target=delayed_msg)
    t.start()

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: bus.read_inbox("w1"),
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: False,
        idle_timeout=0.5,  # 长一点，等消息到
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="initial")
    t.join()

    # 应跑了 2 次（initial + late task）
    assert len(work_log) == 2
    assert work_log[0] == "initial"
    assert work_log[1] == "late task"
    assert lifecycle.state == STATE_SHUTDOWN
```

- [ ] **Step 2: 跑 e2e**

Run: `uv run pytest tests/test_integration.py -k autonomous -v`
Expected: PASS

- [ ] **Step 3: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（556 + 新增 autonomous tests）

- [ ] **Step 4: 不 commit**

---

## Self-Review

**Spec 覆盖**：
- ✅ §1 架构（WORK/IDLE/SHUTDOWN） → T1 lifecycle + T3 worker.py 集成
- ✅ §2 AutonomousLifecycle → T1
- ✅ §3 状态机循环 → T1 run 方法
- ✅ §4 idle 工具 → T2
- ✅ §5 depth 防失控 → T2（spawn_depth 检查）+ T3（worker 接 --depth）
- ✅ §6 worker.py autonomous 模式 → T3
- ✅ §7 配置 → T4
- ✅ §8 测试矩阵 → T1 (5) + T2 (2) + T5 (2)
- ✅ §9 已知限制（不接 task_store / 不持久化 / 不传 history / max_depth 软限制） → 全遵守

**Placeholder 扫描**：无 TBD/TODO。

**类型一致性**：
- `AutonomousLifecycle.run(initial_task)` 签名 T1 定义，T3 worker.py 调用 ✓
- `STATE_WORK/STATE_IDLE/STATE_SHUTDOWN` 常量 T1 定义，T1/T5 引用 ✓
- `AIAgent._idle_requested` / `spawn_depth` T2 定义，T2 测试 + T3 worker.py 引用 ✓
- `coord.spawn(depth=N, autonomous=False)` T2 改造签名，T2 测试通过 + T3 worker.py 不需传（用 default） ✓

---

## Execution Handoff

按用户授权直接进 SDD。
