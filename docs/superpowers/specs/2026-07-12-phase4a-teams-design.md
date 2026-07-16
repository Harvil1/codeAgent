# Phase 4a: Agent Teams 设计

- **日期**：2026-07-12
- **状态**：用户授权直接推进
- **范围**：仅 Phase 4a（Agent Teams 消息总线 + 5 工具 + 一次性 spawn）。Phase 4b Autonomous 留 Phase 4.2
- **Spike 已验证**：Windows 上跨进程文件锁 + JSONL append 可行（msvcrt + open("a") 都通过）
- **对应 Spec**：`docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 4a 展开

---

## 摘要

新增 Agent Teams：多个 agent 通过 JSONL 收件箱互相通信。主 agent 通过 `team_spawn` 工具 `subprocess.Popen` 起独立 agent 进程，通过 `team_send` 发消息，`team_inbox` 读自己的收件箱。一次性 spawn（子 agent 跑完一个 prompt 即退出），无 IDLE 轮询（Phase 4b 做）。

---

## §1 架构

```
主 agent (Process A)
  │
  ├─ LLM 调 team_spawn(name="worker1", role="...", task="...")
  │     ↓
  │  subprocess.Popen([sys.executable, "-m", "agent.team.worker",
  │                    "--name", "worker1", "--task", "..."])
  │     ↓
  │  Process B 起来：构造 AIAgent + run_conversation(task)
  │     ↓
  │  完成后 team_send(to="main", type="response", content=result)
  │     ↓
  │  Process B 退出
  │
  └─ LLM 后续轮 team_inbox() → 读到 worker1 的 response
```

### 文件布局

```
~/.agent/.team/
├── inbox/
│   ├── main.jsonl          # 主 agent 的收件箱
│   ├── worker1.jsonl       # 子 agent 1 的收件箱
│   └── ...
├── registry.json           # 团队成员名册（name/role/pid/status）
└── locks/
    └── inbox-{name}.lock   # 每个收件箱的 lock 文件
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/team/__init__.py` | 🆕 新增 | package marker |
| `agent/team/bus.py` | 🆕 新增 | MessageBus（send/read_inbox/list_inboxes） |
| `agent/team/coordinator.py` | 🆕 新增 | TeamCoordinator（注册/查询成员） |
| `agent/team/worker.py` | 🆕 新增 | 子 agent 入口（`python -m agent.team.worker`） |
| `tools/team_tool.py` | 🆕 新增 | 5 个工具 handler + 注册 |
| `agent/__init__.py` | ♻️ 改 | AIAgent 加 `team_name=None` kwarg + 主循环每轮 drain inbox |
| `cli.py` | ♻️ 改 | RuntimeContext 装配 team_coordinator；注入 AIAgent |
| `config.py` | ♻️ 改 | `team` 配置块 |
| `tests/test_team_bus.py` | 🆕 新增 | MessageBus 单元测试 |
| `tests/test_team_tool.py` | 🆕 新增 | 工具 handler 测试 |
| `tests/test_integration.py` | ♻️ 改 | 主循环 + spawn e2e |

---

## §2 消息总线（`agent/team/bus.py`）

### 数据结构

```python
@dataclass
class TeamMessage:
    id: str               # ulid-like
    from_: str            # 发送者 agent 名（避免 built-in `from`）
    to: str               # 接收者 agent 名
    type: str             # "message" | "request" | "response" | "shutdown"
    content: str
    ts: str               # ISO timestamp
    request_id: Optional[str] = None  # request/response 配对
```

### MessageBus 接口

```python
class MessageBus:
    """JSONL 文件消息总线。所有读写用文件锁序列化。"""

    def __init__(self, *, team_dir: Path):
        self._team_dir = team_dir
        self._inbox_dir = team_dir / "inbox"
        self._locks_dir = team_dir / "locks"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def send(self, *, from_: str, to: str, type_: str, content: str,
             request_id: Optional[str] = None) -> str:
        """追加一条消息到 to 的收件箱。返回 message_id。

        用文件锁 + atomic append（spike 验证过）。
        type_ 必须是 'message' / 'request' / 'response' / 'shutdown'。
        """

    def read_inbox(self, name: str) -> List[TeamMessage]:
        """读取 name 的所有消息，**消费式**（读后清空）。

        用文件锁保证原子性：lock → read all → truncate → unlock。
        """

    def list_inboxes(self) -> List[str]:
        """列出所有有消息或曾经有消息的 agent 名。"""
```

### 文件锁实现

跨平台 helper（spike 验证过）：

```python
def _with_lock(lock_path: Path, fn):
    """获取独占锁后执行 fn，确保释放。"""
    with open(lock_path, "w", encoding="utf-8") as lf:
        _acquire_lock(lf)
        try:
            return fn()
        finally:
            _release_lock(lf)


def _acquire_lock(fileobj):
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
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)
```

---

## §3 TeamCoordinator（`agent/team/coordinator.py`）

```python
@dataclass
class TeamMember:
    name: str
    role: str               # "lead" / "worker" / 自定义
    pid: Optional[int]      # subprocess.Popen.pid（spawn 后填）
    status: str             # "spawning" / "running" / "completed" / "failed"
    created_at: str
    task: Optional[str]     # 一次性 spawn 的任务 prompt

class TeamCoordinator:
    """团队成员注册 + spawn 管理。"""

    def __init__(self, *, team_dir: Path, team_name: str,
                 config: dict, harvil_home: Path):
        ...

    def register(self, name: str, role: str) -> TeamMember:
        """主 agent 启动时自注册（name=team_name，role='lead'）。"""

    def spawn(self, *, name: str, role: str, task: str) -> TeamMember:
        """启动子 agent 进程。返回 TeamMember。

        流程：
        1. 注册到 registry.json
        2. subprocess.Popen([sys.executable, "-m", "agent.team.worker",
                              "--name", name, "--task", task])
        3. 记录 pid
        """

    def list_members(self) -> List[TeamMember]:
        """列出所有成员。"""

    def update_status(self, name: str, status: str):
        """更新成员状态（子 agent 退出时由 worker.py 调）。"""

    def shutdown(self, name: str) -> bool:
        """发送 shutdown 消息 + 等 pid 退出（最多 10s）。"""
```

### registry.json 格式

```json
{
  "members": [
    {
      "name": "main",
      "role": "lead",
      "pid": 12345,
      "status": "running",
      "created_at": "2026-07-12T15:30:00",
      "task": null
    },
    {
      "name": "worker1",
      "role": "worker",
      "pid": 12346,
      "status": "running",
      "created_at": "2026-07-12T15:31:00",
      "task": "处理数据分析"
    }
  ]
}
```

---

## §4 worker.py 子 agent 入口

```python
# agent/team/worker.py
"""子 agent 入口。CLI: python -m agent.team.worker --name X --task "..."

流程：
1. 解析 --name / --task
2. 加载 config + MemoryStore + LLM client（独立实例）
3. 创建 AIAgent，team_name=name
4. run_conversation(task)
5. 把最终响应 send 给 "main"
6. 注册自己 status=completed
7. 退出
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--team-dir", required=True)
    parser.add_argument("--agent-home", required=True)
    args = parser.parse_args()

    # 加载配置 + 组件
    config = _load_config(args.agent_home)
    memory_store = MemoryStore(harvil_home=args.agent_home)
    bus = MessageBus(team_dir=Path(args.team_dir))
    coordinator = TeamCoordinator(team_dir=Path(args.team_dir), ...)

    # 构造 AIAgent
    agent = AIAgent(
        ...,
        team_name=args.name,
        team_bus=bus,
        memory_store=memory_store,
    )

    # 跑一次
    try:
        response = agent.run_conversation(args.task)
        bus.send(from_=args.name, to="main", type_="response",
                 content=response)
        coordinator.update_status(args.name, "completed")
    except Exception as e:
        bus.send(from_=args.name, to="main", type_="message",
                 content=f"[worker crashed: {e}]")
        coordinator.update_status(args.name, "failed")

if __name__ == "__main__":
    main()
```

---

## §5 工具表面（`tools/team_tool.py`）

5 个工具：

```python
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
    "description": "列出所有团队成员",
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
```

### handler 实现

通过 kwargs 接收 `team_coordinator` + `team_bus` + `team_name`（由 agent 透传）。

- `_handle_team_send`：调 `bus.send(from_=team_name, ...)`
- `_handle_team_inbox`：调 `bus.read_inbox(team_name)`
- `_handle_team_members`：调 `coordinator.list_members()`
- `_handle_team_spawn`：调 `coordinator.spawn(name=..., role=..., task=...)`
- `_handle_team_shutdown`：调 `coordinator.shutdown(name=...)`

每个返回 JSON 字符串。

---

## §6 主循环集成（`agent/__init__.py`）

`AIAgent.__init__` 加 `team_bus=None`, `team_coordinator=None`, `team_name=None` kwargs。

主循环每轮 LLM 前（紧邻 cron drain 之后），如果有新消息，**作为持久 user 消息**注入（不是临时，因为消息是上下文）：

```python
# === NEW Phase 4a: team inbox 注入 ===
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
if team_messages_text:
    messages.append({
        "role": "user",
        "content": f"<team_messages>\n{team_messages_text}\n</team_messages>",
    })
```

**关键**：team_messages **持久**进 conversation_history（区别于 bg/cron 的临时消息）。原因：消息是 agent 的实质上下文。

---

## §7 配置

```python
# config.py:DEFAULT_CONFIG
"team": {
    "enabled": True,                       # False 时整个 team 工具隐藏
    "team_dir": None,                      # None → 默认 ~/.agent/.team/
    "default_role": "worker",
    "spawn_timeout": 600,                  # 等子 agent 退出最长时间（秒）
    "max_members": 10,                     # 团队成员上限
},
```

---

## §8 失败处理

| 故障 | 行为 |
|---|---|
| spawn 时名字已存在 | 返回 `{"error": "name exists", "error_type": "team_name_conflict"}` |
| 团队成员达 max_members | 返回 `{"error": "max members reached", "error_type": "team_full"}` |
| send 时接收者 inbox 不存在 | 自动创建 inbox 文件（不报错） |
| send 时文件锁获取失败 | log warning + 返回 error（不阻塞主循环） |
| read_inbox 时锁失败 | log warning + 返回空 list |
| spawn subprocess 失败 | log error + 注册成员 status=failed + 返回 error |
| 子 agent crash | 子 agent 自己 try/except + status=failed + 发 message |
| worker.py 入口参数错 | 进程立即退出 + 主 agent 不感知（用 team_members 查询时看到 failed） |

---

## §9 并发安全

- 每个收件箱用独立 lock 文件（`locks/inbox-{name}.lock`），互不影响
- `send` → lock → append → unlock
- `read_inbox` → lock → read all → truncate → unlock
- registry.json 用单独的 `registry.lock`
- spike 验证：Windows msvcrt + POSIX fcntl 都工作

---

## §10 测试矩阵

### `tests/test_team_bus.py`（10 个）

- `test_send_creates_inbox_file`
- `test_send_returns_message_id`
- `test_send_invalid_type_raises`
- `test_read_inbox_returns_messages`
- `test_read_inbox_is_consumptive`（读后清空）
- `test_read_inbox_empty_returns_empty_list`
- `test_read_inbox_unknown_name_returns_empty`
- `test_list_inboxes_returns_names`
- `test_concurrent_send_safe`（多线程并发 send 同一 inbox 不丢）
- `test_concurrent_read_write_safe`（一个线程 read 一个线程 send 不出错）

### `tests/test_team_coordinator.py`（6 个）

- `test_register_lead`
- `test_spawn_returns_member_with_pid`
- `test_spawn_duplicate_name_raises`
- `test_list_members`
- `test_update_status`
- `test_shutdown`

### `tests/test_team_tool.py`（5 个）

- 每个 handler 至少一个 happy path 测试

### `tests/test_integration.py`（2 个）

- `test_team_messages_injected_into_main_loop`：mock bus.read_inbox 返回消息 → 主循环注入 `<team_messages>`
- `test_team_spawn_e2e`：spawn 一个真实子进程跑简单 task → 验证 message 回流（用 quick prompt）

---

## §11 已知限制 / 非目标

1. **不实现 Autonomous IDLE 轮询**（Phase 4b）。子 agent 跑完 task 即退出，不会主动找新工作。
2. **不实现跨机器**：所有 agent 进程在同一机器上，共享 `~/.agent/`。
3. **不实现消息持久化超 7 天**：inbox 消息消费后立即清空。需要历史回查用 session_search。
4. **不实现消息加密/认证**：所有 agent 共享同一文件系统，假设互信。
5. **不实现任务分配算法**：主 agent 显式 `team_send` 派任务；没有「任务市场」自动认领。
6. **子 agent 进程资源独占**：每个子进程独立 MemoryStore、独立 LLM client、独立 system prompt cache。5 个子 agent = 5x 内存。
7. **spawn 异步**：team_spawn 立即返回，不等子 agent 完成。主 agent 用 `team_inbox` 轮询响应。
8. **死循环防护**：子 agent 调 team_spawn 自己 spawn 子 agent → 允许但限制 `max_members` 总数。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代 | 理由 |
|---|---|---|---|
| 并发模型 | subprocess.Popen | threading / asyncio | crash 隔离；spike 验证可行；spec §6 设计 |
| 范围 | 仅 4a | 4a+4b 一起 | 验证文件总线架构；4b 依赖 4a |
| 启动方式 | 主 agent team_spawn 工具 spawn | CLI 子命令 | agent 自管理 |
| 消息总线 | JSONL 文件 + 文件锁 | SQLite / Redis / message queue | 零依赖；跨进程可见；spike 验证 |
| 通知注入 | 持久进 history（区别于 bg/cron 临时） | 临时 | 消息是上下文，不是状态 |
| 子 agent 生命周期 | 一次性（run_conversation 一次后退出） | 长驻 IDLE 轮询 | 4a 简化；4b 实现 IDLE |

## 附录 B: 与现有原则对齐

| 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰 | team 工具是数据；MessageBus 独立模块 |
| Prompt Caching 神圣 | team 消息进 user turn，不改 system prompt |
| 完全可逆 | inbox 文件消费后 truncate（不是 unlink）；shutdown 软停止 |
| 用户意图优先 | team.enabled 开关；max_members 上限 |
| 安全默认 | subprocess.Popen 不经 shell；文件锁原子；spawn 失败 error JSON |
| 发现 ≠ 可见 | team toolset 独立；config.team.enabled 控制 |
