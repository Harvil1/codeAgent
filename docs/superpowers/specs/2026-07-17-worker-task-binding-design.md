# Worker 任务归属强制（Task Binding）设计

- **日期**：2026-07-17
- **状态**：用户授权直接推进
- **范围**：第 4 批改进之 ⑫（快速项）。给 spawned worker 加 task_id 绑定，防止 prompt 注入跨任务操作
- **依赖**：Phase 4a Agent Teams（已就绪）、TaskStore（已就绪）
- **参考**：`D:\project\hermes-agent-main\项目文档\04-多Agent-协调.md` §5.4

---

## 摘要

team_spawn 加 `task_id` 参数。Coordinator 在 spawn 前自动 claim，并把 `HARVIL_KANBAN_TASK=<id>` 注入子进程 env。Worker 内的 `task_update` / `task_complete` 工具读 env，发现调用参数里的 task_id 与 env 不一致时返回 `permission_denied`。CLI 主 agent 和不传 task_id 的老调用不受影响（向后兼容）。

---

## §1 问题与目标

### 问题

当前 `team_spawn` 把任意 prompt 交给 spawned worker，worker 进程内可调用 `task_update` / `task_complete` 操作**任何** task_id。一旦 worker 的 prompt 被注入（比如读邮件时被邮件内容里的指令污染），就可能改动兄弟 worker 的任务、跨租户任务的状态。

### 目标

1. 每个 spawned worker 在进程级被绑定到**至多一个** task_id
2. Worker 内的写工具（`task_update` / `task_complete`）只能动它绑定的那个 task
3. 主 agent 和 legacy 调用零行为变化（向后兼容）
4. 集中 env 读取逻辑到独立模块，便于下一批 Kanban heartbeat 复用

### 非目标

- 不引入 kanban DB（用现有 flat TaskStore）
- 不实现 heartbeat / run_id / claim_lock 等 dispatcher 复杂语义（下一批做）
- 不门控 `task_create` / `task_list`（创建新任务和读取不算跨任务操纵）
- 不约束 `delegate_task`（同步子进程，无 LLM 自主性风险）

---

## §2 架构

```
父 agent (main)                            子进程 (worker)
─────────────                              ─────────────
1. task_create(subject="解析数据")
   → task_id = "task_abc123"

2. team_spawn(name="w1", task="...",
              task_id="task_abc123")
       │
       ▼
   Coordinator.spawn(task_id=...)
       ├─ TaskStore.claim("task_abc123", owner="w1")
       │   [持久化绑定：owner=w1, status=in_progress]
       ├─ env["HARVIL_KANBAN_TASK"] = "task_abc123"
       │   [进程绑定：env var]
       └─ subprocess.Popen(cmd, env=env)  ──►  worker.py main()
                                                env var 自动可见

                                            3. LLM 调 task_complete(id="task_abc123")
                                               ├─ task_binding.assert_owned("task_abc123")
                                               │   bound="task_abc123" ✓ pass
                                               └─ TaskStore.complete(...)

                                            4. (注入场景) LLM 调 task_complete(id="task_xyz")
                                               ├─ task_binding.assert_owned("task_xyz")
                                               │   bound="task_abc123" ≠ "task_xyz"
                                               │   → raise PermissionError
                                               └─ handler 捕获 → 返回
                                                   {"error": "...", "error_type": "permission_denied"}
```

### 关键不变量

1. **Env var 是唯一身份来源** —— worker 不读 CLI 参数，env 注入后子进程自带身份（与 04 号文档一致）
2. **绑定不可改** —— 子进程无法 unset env 来绕过（攻击模型不考虑 OS 级 env 篡改）
3. **task_id 可选** —— `team_spawn` 不传时 env 不设，`get_bound_task_id()` 返 None，`assert_owned` 直接 pass
4. **CLI 主 agent 不受限** —— `python main.py` 不设 env，工具行为不变
5. **多 worker 隔离** —— 每个 spawned worker 有独立 env，互不影响

---

## §3 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/team/task_binding.py` | 🆕 新增 | `get_bound_task_id()` 读 env + `assert_owned(id)` 校验 |
| `agent/team/coordinator.py` | ♻️ 改 | `spawn()` 加 `task_id` 可选参数：claim + 注入 env |
| `tools/team_tool.py` | ♻️ 改 | `TEAM_SPAWN_SCHEMA` 加 `task_id` 字段；handler 转发 |
| `tools/task_tools.py` | ♻️ 改 | `_handle_task_update` / `_handle_task_complete` 头部加 `assert_owned(id)`，捕获 `PermissionError` 返回 `permission_denied` |
| `tests/test_team_task_binding.py` | 🆕 新增 | 模块单元测试 + 集成场景 |

---

## §4 组件设计

### §4.1 `agent/team/task_binding.py`（新增）

```python
"""Worker 任务归属强制（Task Binding）。

防止 spawned worker 被 prompt 注入后跨任务操作。
Coordinator.spawn() 时把 HARVIL_KANBAN_TASK=<id> 注入子进程 env，
task_tools 的写工具调用前用 assert_owned() 校验。
"""
import os
from typing import Optional


ENV_VAR = "HARVIL_KANBAN_TASK"


class TaskOwnershipError(PermissionError):
    """Worker 试图操作未绑定的 task。"""


def get_bound_task_id() -> Optional[str]:
    """返回当前进程绑定的 task_id；未绑定返回 None。"""
    return os.environ.get(ENV_VAR) or None


def assert_owned(task_id: str) -> None:
    """断言当前进程有权操作 task_id。

    - 未绑定（主 agent / legacy 调用）：直接 pass
    - 绑定且匹配：pass
    - 绑定但不匹配：raise TaskOwnershipError
    """
    bound = get_bound_task_id()
    if bound is not None and bound != task_id:
        raise TaskOwnershipError(
            f"worker bound to task {bound!r}, cannot operate on {task_id!r}"
        )
```

**设计权衡**：
- 抛 `PermissionError` 子类（`TaskOwnershipError`），handler 可精准捕获，不会被宽泛的 `except Exception` 吞掉
- `get_bound_task_id` 单独导出，下一批 heartbeat 桥接复用

### §4.2 `agent/team/coordinator.py`（改 spawn）

```python
# 现有签名
def spawn(self, *, name: str, role: str, task: str,
          depth: int = 1,
          command: Optional[list] = None) -> TeamMember:

# 改为
def spawn(self, *, name: str, role: str, task: str,
          depth: int = 1,
          task_id: Optional[str] = None,
          command: Optional[list] = None) -> TeamMember:
    """...

    task_id: 可选。若提供：
      1. spawn 前 TaskStore.claim(task_id, owner=name)
      2. 注入 HARVIL_KANBAN_TASK=task_id 到子进程 env
    """
    member = self.register(name=name, role=role, status="spawning")
    member.task = task

    cmd = command or [
        sys.executable, "-m", "agent.team.worker",
        "--name", name,
        "--task", task,
        "--team-dir", str(self._team_dir),
        "--agent-home", str(self._harvil_home),
        "--depth", str(depth),
    ]

    env = os.environ.copy()
    if task_id is not None:
        # 1. claim 到 worker name（持久化绑定）
        from agent.task_store import get_task_store
        from agent.team.task_binding import ENV_VAR
        store = get_task_store(harvil_home=str(self._harvil_home))
        claimed = store.claim(task_id, owner=name)
        if claimed is None:
            self.update_status(name, "failed")
            raise ValueError(f"task_id {task_id} 不存在")
        # 2. 注入 env（进程绑定）
        env[ENV_VAR] = task_id

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        # ... 原有 Popen 后逻辑（pid/members/processes/update_status）
```

**注意**：
- `import os` 已经在文件顶部
- claim 在 register 之后、Popen 之前——若 claim 失败（task_id 不存在）抛 `ValueError`，update_status("failed") 后让 caller 处理
- env 是 `os.environ.copy()` + 修改，不污染父进程

### §4.3 `tools/team_tool.py`（schema + handler）

```python
TEAM_SPAWN_SCHEMA = {
    "name": "team_spawn",
    "description": (
        "启动一个子 agent 进程处理任务（一次性，完成后自动退出）。\n"
        "可选 task_id：若提供，worker 进程被绑定到该 task，"
        "只能通过 task_update/task_complete 操作该任务。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "新 agent 的名字（必须唯一）"},
            "role": {"type": "string", "default": "worker"},
            "task": {"type": "string", "description": "交给新 agent 跑的 prompt"},
            "task_id": {
                "type": "string",
                "description": "可选。绑定的 TaskStore 任务 ID",
            },
        },
        "required": ["name", "task"],
    },
}

def _handle_team_spawn(args: dict, **kwargs) -> str:
    name = args.get("name", "").strip()
    task = args.get("task", "").strip()
    task_id = args.get("task_id") or None  # 空字符串归一化为 None
    ...
    try:
        member = coordinator.spawn(
            name=name, role=role, task=task,
            depth=depth, task_id=task_id,
        )
    except ValueError as e:
        return _err(str(e), "invalid_task_id")
    ...
```

### §4.4 `tools/task_tools.py`（两个 handler 加校验）

```python
from agent.team.task_binding import assert_owned, TaskOwnershipError

def _handle_task_update(args: dict, **kwargs) -> str:
    task_id = args.get("id", "").strip()
    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return json.dumps(
            {"error": str(e), "error_type": "permission_denied"},
            ensure_ascii=False,
        )
    # ... 原有逻辑

def _handle_task_complete(args: dict, **kwargs) -> str:
    task_id = args.get("id", "").strip()
    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return json.dumps(
            {"error": str(e), "error_type": "permission_denied"},
            ensure_ascii=False,
        )
    # ... 原有逻辑
```

---

## §5 错误处理

| 场景 | 行为 | 返回 |
|---|---|---|
| 主 agent / legacy 调用（无 env） | `assert_owned` 直接 pass | 原工具行为不变 |
| Worker 调用自己的 task | 匹配 pass | 正常执行 |
| Worker 调用别的 task | `TaskOwnershipError` | `{"error_type": "permission_denied"}` |
| team_spawn 的 task_id 不存在 | `coordinator.spawn` 抛 `ValueError` | `{"error_type": "invalid_task_id"}` |
| team_spawn 的 task 已被 claim | 当前 TaskStore.claim 是覆盖式（直接重写 owner），先不做并发检测；下一批 Kanban heartbeat 引入 dispatcher 后处理 |

---

## §6 测试设计

### §6.1 `tests/test_team_task_binding.py`（新增）

```python
class TestTaskBinding:
    """task_binding 模块单元测试。"""

    def test_get_bound_task_id_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
        assert get_bound_task_id() is None

    def test_get_bound_task_id_returns_env(self, monkeypatch):
        monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_abc")
        assert get_bound_task_id() == "task_abc"

    def test_assert_owned_passes_when_unset(self, monkeypatch):
        """主 agent / legacy 调用不受限。"""
        monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
        assert_owned("any_task")  # 不抛

    def test_assert_owned_passes_when_match(self, monkeypatch):
        monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_abc")
        assert_owned("task_abc")  # 不抛

    def test_assert_owned_raises_on_mismatch(self, monkeypatch):
        monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_abc")
        with pytest.raises(TaskOwnershipError):
            assert_owned("task_xyz")


class TestCoordinatorSpawnBinding:
    """Coordinator.spawn(task_id=...) 集成测试。"""

    def test_spawn_without_task_id_no_env(self, tmp_path, monkeypatch):
        """不传 task_id → 子进程 env 无 HARVIL_KANBAN_TASK。"""
        monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
        coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path, config={})
        # 用 mock command 抓 env
        captured_env = {}
        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            class P: pid = 12345; poll = lambda self: 0
            return P()
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        coord.spawn(name="w1", role="worker", task="...", depth=1)
        assert "HARVIL_KANBAN_TASK" not in captured_env

    def test_spawn_with_task_id_claims_and_sets_env(
        self, tmp_path, monkeypatch, fake_task_store,
    ):
        """传 task_id → claim + env var。"""
        # 1. 建任务
        store = get_task_store(harvil_home=str(tmp_path))
        task = store.create(subject="测试")
        # 2. spawn
        coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path, config={})
        captured_env = {}
        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            class P: pid = 12345; poll = lambda self: 0
            return P()
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        coord.spawn(name="w1", role="worker", task="...", depth=1, task_id=task["id"])
        # 3. 校验
        assert captured_env["HARVIL_KANBAN_TASK"] == task["id"]
        refreshed = store.get(task["id"])
        assert refreshed["owner"] == "w1"
        assert refreshed["status"] == "in_progress"

    def test_spawn_with_invalid_task_id_raises(self, tmp_path, monkeypatch):
        coord = TeamCoordinator(team_dir=tmp_path, harvil_home=tmp_path, config={})
        with pytest.raises(ValueError):
            coord.spawn(name="w1", role="worker", task="...", task_id="task_nonexistent")


class TestTaskToolsBinding:
    """task_update/task_complete handler 在绑定下的行为。"""

    def test_task_complete_blocks_foreign_id(self, monkeypatch):
        """env 绑定 task_A → task_complete(task_B) 返回 permission_denied。"""
        monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
        result = _handle_task_complete({"id": "task_B"})
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_task_update_blocks_foreign_id(self, monkeypatch):
        monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
        result = _handle_task_update({"id": "task_B", "status": "completed"})
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_task_complete_allows_matching_id(self, monkeypatch, tmp_path):
        """env 绑定 → task_complete(同 id) 正常执行。"""
        store = get_task_store(harvil_home=str(tmp_path))
        task = store.create(subject="X")
        monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])
        result = _handle_task_complete({"id": task["id"]})
        data = json.loads(result)
        assert data.get("status") == "completed"
        refreshed = store.get(task["id"])
        assert refreshed["status"] == "completed"

    def test_task_complete_allows_when_unset(self, monkeypatch, tmp_path):
        """主 agent / legacy 调用（无 env）正常执行。"""
        monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
        store = get_task_store(harvil_home=str(tmp_path))
        task = store.create(subject="X")
        result = _handle_task_complete({"id": task["id"]})
        data = json.loads(result)
        assert data.get("status") == "completed"
```

### §6.2 现有测试无回归

跑 `uv run pytest tests/test_team_coordinator.py tests/test_team_tool.py tests/test_task_system.py tests/test_integration.py`，确认现有 spawn / task 调用零行为变化。

---

## §7 实现顺序

1. 新建 `agent/team/task_binding.py` + 单元测试
2. 改 `coordinator.spawn()` + 测试
3. 改 `team_tool.py` schema + handler + 测试
4. 改 `task_tools.py` 两个 handler + 测试
5. 跑全测：`uv run pytest tests/`，确认 668 → 668+N（新增测试数）
6. `scripts/verify.py` 回归
7. commit + push origin/master

---

## §8 未来扩展（本批不做）

- **§5.5 自动心跳桥**：worker runtime 活动 → `heartbeat_current_worker_from_env()` 把 env var 解出后桥接到 TaskStore（每 60s 更新 `last_heartbeat_at`）。复用 `task_binding.get_bound_task_id()`
- **claim 并发检测**：当前 claim 是覆盖式，多 worker 同时抢同一 task_id 时后到者覆盖。引入 dispatcher 后再加 mutex
- **goal-mode 评审**：完成前辅助 LLM 评审（04 号文档 §7.4）
