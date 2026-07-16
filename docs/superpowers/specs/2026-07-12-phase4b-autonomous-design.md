# Phase 4b: Autonomous Agent 设计

- **日期**：2026-07-12
- **状态**：用户授权直接推进
- **范围**：Phase 4b（Autonomous 生命周期）。完成 Phase 4 全部
- **依赖**：Phase 4a Agent Teams（已就绪）
- **对应 Spec**：`docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 4b 展开

---

## 摘要

给 worker.py 加 WORK/IDLE/SHUTDOWN 三态生命周期。`--autonomous` flag 启用。worker 完成首个任务后不退出，进入 IDLE 轮询 inbox + unclaimed tasks；收到新工作回到 WORK；60s 无事可做则 SHUTDOWN。`depth` 参数防递归 spawn 失控（默认 max_depth=2）。

---

## §1 架构

```
worker.py 启动 (--autonomous --depth N)
   ↓
WORK 状态：run_conversation(task)
   ↓ 完成或调 idle 工具
IDLE 状态：
   while not timeout:
       sleep(POLL_INTERVAL)
       if inbox 有消息 → WORK（user_msg=消息内容）
       if task_store 有 unclaimed task → WORK（认领 + user_msg=任务）
   ↓ IDLE_TIMEOUT（60s）超时
SHUTDOWN：update_status("completed") + 退出
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/team/lifecycle.py` | 🆕 新增 | AutonomousLifecycle 类（WORK/IDLE/SHUTDOWN 状态机） |
| `agent/team/worker.py` | ♻️ 改 | 加 `--autonomous` + `--depth` 参数；autonomous 模式调 lifecycle |
| `tools/team_tool.py` | ♻️ 改 | 加 `idle` 工具（仅 worker 用，触发 IDLE） |
| `agent/__init__.py` | ♻️ 改 | AIAgent 加 `_idle_requested` 标志 + `idle` 工具能设置它 |
| `config.py` | ♻️ 改 | team.autonomous_idle_timeout / max_depth |
| `tests/test_team_lifecycle.py` | 🆕 新增 | lifecycle 单元测试 |
| `tests/test_integration.py` | ♻️ 改 | autonomous e2e（mock LLM 短轮询） |

---

## §2 AutonomousLifecycle

```python
# agent/team/lifecycle.py
"""WORK/IDLE/SHUTDOWN 状态机。"""
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STATE_WORK = "work"
STATE_IDLE = "idle"
STATE_SHUTDOWN = "shutdown"


class AutonomousLifecycle:
    """三态生命周期管理。

    使用方式：
        lifecycle = AutonomousLifecycle(
            work_fn=lambda task: agent.run_conversation(task),
            poll_inbox_fn=lambda: bus.read_inbox(name),
            poll_tasks_fn=lambda: task_store.find_ready(),
            claim_task_fn=lambda tid: task_store.claim(tid, name),
            idle_timeout=60.0,
            poll_interval=5.0,
        )
        lifecycle.run(initial_task="...")
    """

    def __init__(
        self,
        *,
        work_fn: Callable[[str], str],            # task → response
        poll_inbox_fn: Callable[[], list],        # → list[TeamMessage]（消费式）
        poll_tasks_fn: Callable[[], list],        # → list[task dict]
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
        # WORK: 跑初始任务
        self._state = STATE_WORK
        try:
            self._work_fn(initial_task)
        except Exception as e:
            logger.exception("WORK 异常: %s", e)

        # IDLE 循环
        self._state = STATE_IDLE
        self._idle_started_at = time.time()
        while self._state == STATE_IDLE:
            if self._idle_timed_out():
                logger.info("IDLE 超时，进入 SHUTDOWN")
                break
            if self._try_get_work():
                # 拿到新工作，回 WORK
                self._state = STATE_WORK
                self._idle_started_at = None
                # 工作循环留在外层
            else:
                time.sleep(self._poll_interval)

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
        """检查 inbox + unclaimed tasks。返回拿到的 task prompt。"""
        # 1. 先看 inbox
        try:
            msgs = self._poll_inbox_fn()
            if msgs:
                # 取第一条消息作为新 task
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

⚠️ **修正**：上述伪代码的 `_try_get_work` 拿到工作后需要回到 WORK 跑 work_fn，不是简单 return。修正在 spec §3 给出最终版本。

---

## §3 最终状态机

```python
def run(self, *, initial_task: str) -> None:
    """完整 WORK→IDLE→WORK→...→SHUTDOWN 循环。"""
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
                break
            next_task = self._try_get_work()
            if next_task is not None:
                break
            time.sleep(self._poll_interval)

        if next_task is None:
            # IDLE 超时，无新工作
            break

        current_task = next_task
        # 回到 while 顶部 → WORK

    # SHUTDOWN
    self._state = STATE_SHUTDOWN
    if self._on_shutdown_fn:
        try: self._on_shutdown_fn()
        except Exception: logger.exception("on_shutdown 异常")
```

---

## §4 idle 工具

`tools/team_tool.py` 加第 6 个工具：

```python
IDLE_SCHEMA = {
    "name": "idle",
    "description": "声明当前没有更多工作，进入 IDLE 状态等新任务（仅 autonomous 模式可用）",
    "parameters": {"type": "object", "properties": {}},
}

def _handle_idle(args: dict, **kwargs) -> str:
    agent = kwargs.get("agent_ref")  # 通过 kwargs 传 AIAgent 引用
    if agent is None:
        return json.dumps({"success": False, "error": "no agent ref"}, ensure_ascii=False)
    agent._idle_requested = True
    return json.dumps({"success": True, "message": "idle requested"}, ensure_ascii=False)
```

**机制**：worker 调 `idle` 工具后，`AIAgent._idle_requested = True`。`run_conversation` 检测到此标志后退出循环（类似 interrupt）。

`AIAgent.__init__` 加 `self._idle_requested = False`。

`run_conversation` 在每轮 LLM 调用后检查：
```python
if self._idle_requested:
    logger.info("idle requested, exiting run_conversation")
    break
```

**注意**：`idle` 工具不在 `team` toolset 里（避免主 agent 误调），独立放在 `core` toolset 但 check_fn 控制只有 `team_name != "main"` 时可见。或更简单：放进 `team` toolset，主 agent 调了也无害（只是设置标志，主 agent 不读这个标志）。

决策：**放进 team toolset**（简单），无 check_fn。主 agent 调 idle 是 no-op（它没读这个标志）。

---

## §5 depth 防失控

```python
# worker.py 启动时
parser.add_argument("--depth", type=int, default=1)

# 传给 AIAgent
agent = AIAgent(..., spawn_depth=args.depth)

# AIAgent.__init__
self.spawn_depth = spawn_depth or 1
```

`team_spawn` 工具 handler 检查 depth：
```python
def _handle_team_spawn(args, **kwargs):
    agent = kwargs.get("agent_ref")
    current_depth = getattr(agent, "spawn_depth", 1) if agent else 1
    max_depth = kwargs.get("config", {}).get("team", {}).get("max_depth", 2)
    if current_depth >= max_depth:
        return json.dumps({
            "success": False,
            "error": f"max_depth {max_depth} reached (current: {current_depth})",
            "error_type": "team_max_depth",
        }, ensure_ascii=False)
    # ... 正常 spawn，子进程 command 加 --depth current_depth+1
```

主 agent depth=0；它 spawn 的 worker depth=1；worker spawn 的子 worker depth=2。max_depth=2 时 worker 不能再 spawn。

---

## §6 worker.py autonomous 模式

```python
# agent/team/worker.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--team-dir", required=True)
    parser.add_argument("--agent-home", required=True)
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    # ... 加载 config + components ...

    agent = AIAgent(
        ...,
        team_bus=bus, team_coordinator=coord, team_name=args.name,
        spawn_depth=args.depth,
    )

    if args.autonomous:
        from agent.team.lifecycle import AutonomousLifecycle
        lifecycle = AutonomousLifecycle(
            work_fn=lambda task: agent.run_conversation(task),
            poll_inbox_fn=lambda: bus.read_inbox(args.name),
            poll_tasks_fn=lambda: [],  # Phase 4b 暂不接入 task_store
            claim_task_fn=lambda tid: False,
            on_shutdown_fn=lambda: coord.update_status(args.name, "completed"),
            idle_timeout=config.get("team", {}).get("autonomous_idle_timeout", 60.0),
            poll_interval=5.0,
        )
        lifecycle.run(initial_task=args.task)
    else:
        # 一次性模式（原 Phase 4a 行为）
        response = agent.run_conversation(args.task)
        bus.send(from_=args.name, to="main",
                 type_="response", content=response or "")
        coord.update_status(args.name, "completed")
```

---

## §7 配置

```python
# config.py:DEFAULT_CONFIG["team"] 新增
"autonomous_idle_timeout": 60.0,        # IDLE 状态超时秒数
"autonomous_poll_interval": 5.0,        # IDLE 轮询间隔
"max_depth": 2,                         # spawn 递归深度上限
```

---

## §8 测试矩阵

### `tests/test_team_lifecycle.py`（5 个）

- `test_lifecycle_work_then_shutdown_no_messages`：无 inbox 消息 + 无任务 → IDLE 超时 → SHUTDOWN
- `test_lifecycle_idle_picks_up_inbox_message`：IDLE 中 inbox 来消息 → 回 WORK
- `test_lifecycle_idle_picks_up_unclaimed_task`：IDLE 中 task_store 有 ready → 回 WORK
- `test_lifecycle_multiple_work_cycles`：多次 WORK→IDLE→WORK 循环
- `test_lifecycle_work_exception_continues_to_idle`：work_fn 抛异常 → 进 IDLE 不崩

### `tests/test_integration.py`（1 个）

- `test_idle_tool_sets_flag`：调 idle 工具 → AIAgent._idle_requested = True
- `test_team_spawn_max_depth_blocks`：depth=max_depth 时 spawn 返回 team_max_depth error

---

## §9 已知限制

1. **不接入 task_store**：Phase 4b 的 poll_tasks_fn 是 stub（返 []）。真实 autonomous 找 unclaimed task 留给后续。inbox 消息驱动已经足够 demo。
2. **不持久化 IDLE 状态**：worker 重启后从 WORK 开始（不是从 IDLE 恢复）。
3. **不实现 work 之间的状态传递**：每次 WORK 是独立 run_conversation（无对话历史延续）。下个 WORK 是全新会话。
4. **不实现 cooperative cancellation**：IDLE 中收到 shutdown 消息不会立即退出（要等当前 sleep + 检查超时）。
5. **max_depth 是软限制**：worker 可以手动写 worker.py 启动命令绕过。生产场景假设互信。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代 | 理由 |
|---|---|---|---|
| 状态机位置 | 独立 lifecycle.py 类 | 内嵌 worker.py | 可单元测试；复用可能 |
| IDLE 检测 | 轮询 inbox + tasks | 事件驱动（watcher） | 轮询简单；事件需要文件 watcher |
| idle 工具 | 设置标志位 | run_conversation 直接退出 | 协作式；类比 interrupt |
| depth 防失控 | CLI 参数 + handler 检查 | 全局计数器 | 简单；进程隔离 |
| 工作循环独立 | 每次新 run_conversation | 延续 history | 简化；避免 context 爆炸 |
