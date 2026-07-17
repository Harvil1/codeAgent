# Worker 任务归属强制（Task Binding）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 spawned worker 加 task_id 绑定（HARVIL_KANBAN_TASK env var），防止 prompt 注入跨任务操作。

**Architecture:** team_spawn 加 task_id 参数 → Coordinator.spawn 在 Popen 前 claim + 注入 env → task_binding.assert_owned 在 task_update/task_complete 调用前校验 → 不匹配抛 TaskOwnershipError → handler 返回 permission_denied。

**Tech Stack:** Python 3.11+、pytest、uv、subprocess.Popen、os.environ

**Spec:** `docs/superpowers/specs/2026-07-17-worker-task-binding-design.md`

## Global Constraints

- **语言约定**：所有注释、commit message、文档用中文；代码标识符用英文（CLAUDE.md 约定）
- **文件 I/O**：必须 `encoding="utf-8"`（CLAUDE.md 强制，Windows 默认 cp1252 会乱码）
- **依赖管理**：用 `uv`，不要用 `pip install`
- **工具结果契约**：所有 handler 返回 JSON 字符串；错误用 `{"error": "...", "error_type": "..."}`（CLAUDE.md 铁律）
- **测试基线**：当前 668 测试通过；本计划完成后应为 668 + 新增测试数（无回归）
- **env var 名**：`HARVIL_KANBAN_TASK`（常量 `agent/team/task_binding.py:ENV_VAR`）

---

## 文件结构

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/team/task_binding.py` | 🆕 新增 | env 读取 + ownership 校验（唯一身份来源） |
| `agent/team/coordinator.py` | ♻️ 改 | `spawn()` 加 `task_id` 参数，claim + 注入 env |
| `tools/team_tool.py` | ♻️ 改 | `TEAM_SPAWN_SCHEMA` 加 `task_id` 字段；handler 转发 |
| `tools/task_tools.py` | ♻️ 改 | 两个写 handler 头部加 `assert_owned` |
| `tests/test_team_task_binding.py` | 🆕 新增 | 模块单元测试 + Coordinator 集成 + handler 行为 |

---

## Task 1: 新建 task_binding 模块 + 单元测试

**Files:**
- Create: `agent/team/task_binding.py`
- Test: `tests/test_team_task_binding.py`

**Interfaces:**
- Produces: `ENV_VAR: str`（常量 "HARVIL_KANBAN_TASK"）、`TaskOwnershipError(PermissionError)`、`get_bound_task_id() -> Optional[str]`、`assert_owned(task_id: str) -> None`

- [ ] **Step 1.1: 写失败测试（模块未实现）**

Create `tests/test_team_task_binding.py`:

```python
"""task_binding 模块单元测试。"""
import pytest

from agent.team.task_binding import (
    ENV_VAR,
    TaskOwnershipError,
    get_bound_task_id,
    assert_owned,
)


# ---------------------------------------------------------------------------
# get_bound_task_id
# ---------------------------------------------------------------------------

def test_env_var_constant():
    assert ENV_VAR == "HARVIL_KANBAN_TASK"


def test_get_bound_task_id_none_when_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert get_bound_task_id() is None


def test_get_bound_task_id_returns_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    assert get_bound_task_id() == "task_abc"


def test_get_bound_task_id_returns_none_for_empty(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "")
    assert get_bound_task_id() is None


# ---------------------------------------------------------------------------
# assert_owned
# ---------------------------------------------------------------------------

def test_assert_owned_passes_when_unset(monkeypatch):
    """主 agent / legacy 调用（env 未设）不受限。"""
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert_owned("any_task_id")  # 不抛


def test_assert_owned_passes_when_match(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    assert_owned("task_abc")  # 不抛


def test_assert_owned_raises_on_mismatch(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "task_abc")
    with pytest.raises(TaskOwnershipError) as exc_info:
        assert_owned("task_xyz")
    msg = str(exc_info.value)
    assert "task_abc" in msg
    assert "task_xyz" in msg


def test_task_ownership_error_is_permission_error():
    """TaskOwnershipError 是 PermissionError 子类（handler 用 except PermissionError 精准捕获）。"""
    assert issubclass(TaskOwnershipError, PermissionError)
```

- [ ] **Step 1.2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_task_binding.py -v`

Expected: `ModuleNotFoundError: No module named 'agent.team.task_binding'`

- [ ] **Step 1.3: 创建模块实现**

Create `agent/team/task_binding.py`:

```python
"""Worker 任务归属强制（Task Binding）。

防止 spawned worker 被 prompt 注入后跨任务操作。

工作方式：
  Coordinator.spawn(task_id=X) 时把 HARVIL_KANBAN_TASK=X 注入子进程 env。
  task_tools 的写工具（task_update / task_complete）调用前用
  assert_owned(task_id) 校验：
    - env 未设（主 agent / legacy 调用）：直接 pass
    - env 与传入 task_id 匹配：pass
    - 不匹配：抛 TaskOwnershipError

注意：本模块是「身份来源」的唯一处。下一批 Kanban heartbeat 桥接
会复用 get_bound_task_id()。
"""
import os
from typing import Optional


ENV_VAR = "HARVIL_KANBAN_TASK"


class TaskOwnershipError(PermissionError):
    """Worker 试图操作未绑定的 task。"""


def get_bound_task_id() -> Optional[str]:
    """返回当前进程绑定的 task_id；未绑定（env 未设或空）返回 None。"""
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

- [ ] **Step 1.4: 跑测试确认通过**

Run: `uv run pytest tests/test_team_task_binding.py -v`

Expected: 8 passed

- [ ] **Step 1.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `668 + 8 = 676 passed`

- [ ] **Step 1.6: Commit**

```bash
git add agent/team/task_binding.py tests/test_team_task_binding.py
git commit -m "feat(team): task_binding 模块——env 读取 + ownership 校验

防止 spawned worker 被 prompt 注入后跨任务操作。
为下一批 team_spawn task_id 参数和 task_tools 门控做准备。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Coordinator.spawn 接受 task_id

**Files:**
- Modify: `agent/team/coordinator.py:97-132`（spawn 方法）
- Test: `tests/test_team_coordinator.py`（追加 3 个测试）

**Interfaces:**
- Consumes: `agent.team.task_binding.ENV_VAR`、`agent.task_store.get_task_store`、`TaskStore.claim(task_id, owner)`
- Produces: `TeamCoordinator.spawn(*, name, role, task, depth=1, task_id=None, command=None) -> TeamMember`

- [ ] **Step 2.1: 写失败测试（追加到 test_team_coordinator.py）**

Append to `tests/test_team_coordinator.py`:

```python
def _python():
    import sys
    return sys.executable


class _FakePopen:
    """假 Popen，捕获 env 用于断言。"""
    def __init__(self, cmd, **kwargs):
        self.captured_env = dict(kwargs.get("env") or {})
        self.pid = 12345
        self._poll = 0

    def poll(self):
        return self._poll


def test_spawn_without_task_id_no_env(tmp_path, monkeypatch):
    """不传 task_id → 子进程 env 不含 HARVIL_KANBAN_TASK。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    coord.spawn(name="w1", role="worker", task="...", depth=1)
    assert "HARVIL_KANBAN_TASK" not in captured["env"]


def test_spawn_with_task_id_claims_and_sets_env(tmp_path, monkeypatch):
    """传 task_id → TaskStore.claim + env 注入。"""
    from agent.task_store import get_task_store
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="测试任务")

    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    captured = {}
    def fake_popen(cmd, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    coord.spawn(
        name="w1", role="worker", task="...",
        depth=1, task_id=task["id"],
    )

    # env 注入
    assert captured["env"]["HARVIL_KANBAN_TASK"] == task["id"]
    # claim 持久化
    refreshed = store.get(task["id"])
    assert refreshed["owner"] == "w1"
    assert refreshed["status"] == "in_progress"


def test_spawn_with_invalid_task_id_raises(tmp_path, monkeypatch):
    """task_id 不存在 → ValueError，且不调 Popen。"""
    coord = TeamCoordinator(
        team_dir=tmp_path, harvil_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    popen_called = []
    def fake_popen(cmd, **kwargs):
        popen_called.append(cmd)
        return _FakePopen(cmd, **kwargs)
    monkeypatch.setattr("agent.team.coordinator.subprocess.Popen", fake_popen)

    import pytest
    with pytest.raises(ValueError, match="不存在"):
        coord.spawn(
            name="w1", role="worker", task="...",
            task_id="task_nonexistent",
        )
    assert popen_called == []  # 没启动子进程
    # registry 里状态应为 failed
    members = coord.list_members()
    assert any(m.name == "w1" and m.status == "failed" for m in members)
```

- [ ] **Step 2.2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_coordinator.py::test_spawn_with_task_id_claims_and_sets_env -v`

Expected: FAIL with `TypeError: spawn() got an unexpected keyword argument 'task_id'`

- [ ] **Step 2.3: 改 Coordinator.spawn**

Modify `agent/team/coordinator.py:97-132`. First add `import os` at the top (after `import json`):

```python
import json
import logging
import os
import subprocess
import sys
```

Then replace the spawn method (lines 97-132) with:

```python
    def spawn(self, *, name: str, role: str, task: str,
              depth: int = 1,
              task_id: Optional[str] = None,
              command: Optional[list] = None) -> TeamMember:
        """启动子 agent 进程。command 默认是 agent.team.worker 入口。

        depth 用于递归限制（Phase 4b），默认 1（第一层子 agent）。

        task_id: 可选。若提供：
          1. spawn 前 TaskStore.claim(task_id, owner=name)（持久化绑定）
          2. 注入 HARVIL_KANBAN_TASK=task_id 到子进程 env（进程绑定）
          task_id 不存在时抛 ValueError，registry 标 failed，不启动子进程。
        """
        # 先注册（status=spawning）
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

        # task_id 绑定：claim + env 注入
        env = os.environ.copy()
        if task_id is not None:
            from agent.task_store import get_task_store
            from agent.team.task_binding import ENV_VAR
            store = get_task_store(harvil_home=str(self._harvil_home))
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
```

- [ ] **Step 2.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_team_coordinator.py -v`

Expected: 所有原有 + 3 个新测试 passed

- [ ] **Step 2.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `676 + 3 = 679 passed`（Task 1 的 676 + 新增 3）

- [ ] **Step 2.6: Commit**

```bash
git add agent/team/coordinator.py tests/test_team_coordinator.py
git commit -m "feat(team): Coordinator.spawn 支持 task_id 绑定

spawn(task_id=X) 时先 claim 到 worker name，再注入 HARVIL_KANBAN_TASK
到子进程 env。task_id 不存在抛 ValueError 且不启动子进程。
不传 task_id 时行为完全不变（向后兼容）。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: team_spawn 工具 schema 加 task_id

**Files:**
- Modify: `tools/team_tool.py:51-63`（TEAM_SPAWN_SCHEMA）和 `tools/team_tool.py:149-183`（_handle_team_spawn）
- Test: `tests/test_team_tool.py`

**Interfaces:**
- Consumes: `TeamCoordinator.spawn(task_id=...)`
- Produces: `team_spawn` 工具 schema 加可选 `task_id` 字段；handler 转发给 coordinator；task_id 不存在时返回 `{"error_type": "invalid_task_id"}`

- [ ] **Step 3.1: 写失败测试（追加到 test_team_tool.py 末尾）**

先读 test_team_tool.py 现有结构，找到夹具/导入区，然后追加：

```python
def test_team_spawn_with_task_id_forwards(monkeypatch, tmp_path):
    """team_spawn(args={..., 'task_id': X}) → coord.spawn(task_id=X)。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    result = _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": "task_abc"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    import json
    data = json.loads(result)
    assert data["success"] is True
    assert captured["task_id"] == "task_abc"
    assert captured["name"] == "w1"


def test_team_spawn_without_task_id_passes_none(monkeypatch):
    """不传 task_id → coord.spawn 不收 task_id 参数（None）。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    _handle_team_spawn(
        args={"name": "w1", "task": "do X"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    assert captured.get("task_id") is None


def test_team_spawn_empty_task_id_treated_as_absent(monkeypatch):
    """空字符串 task_id → 归一化为 None。"""
    from tools.team_tool import _handle_team_spawn

    captured = {}
    class _FakeMember:
        pid = 12345
        status = "running"
    class _FakeCoord:
        def spawn(self, **kwargs):
            captured.update(kwargs)
            return _FakeMember()

    _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": ""},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    assert captured.get("task_id") is None


def test_team_spawn_invalid_task_id_returns_error(monkeypatch):
    """coord.spawn 抛 ValueError → 返回 invalid_task_id。"""
    from tools.team_tool import _handle_team_spawn

    class _FakeCoord:
        def spawn(self, **kwargs):
            raise ValueError("task_id task_xxx 不存在")

    result = _handle_team_spawn(
        args={"name": "w1", "task": "do X", "task_id": "task_xxx"},
        team_coordinator=_FakeCoord(),
        agent_ref=None,
        config={"team": {"max_depth": 2}},
    )
    import json
    data = json.loads(result)
    assert data["success"] is False
    assert data["error_type"] == "invalid_task_id"
```

- [ ] **Step 3.2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_tool.py -v -k task_id`

Expected: FAIL with `AssertionError: 'task_id' not in captured` 或类似（因为现有 handler 不传 task_id）

- [ ] **Step 3.3: 改 TEAM_SPAWN_SCHEMA**

Modify `tools/team_tool.py:51-63`. Replace the schema:

```python
TEAM_SPAWN_SCHEMA = {
    "name": "team_spawn",
    "description": (
        "启动一个子 agent 进程处理任务（一次性，完成后自动退出）。\n"
        "可选 task_id：若提供，worker 进程被绑定到该 task，"
        "其内部的 task_update / task_complete 只能操作该任务（防止 prompt 注入跨任务操作）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "新 agent 的名字（必须唯一）"},
            "role": {"type": "string", "default": "worker"},
            "task": {"type": "string", "description": "交给新 agent 跑的 prompt"},
            "task_id": {
                "type": "string",
                "description": (
                    "可选。绑定的 TaskStore 任务 ID。"
                    "若提供，Coordinator 会先 claim 该任务（owner=name, status=in_progress），"
                    "并把 HARVIL_KANBAN_TASK 注入子进程 env。"
                ),
            },
        },
        "required": ["name", "task"],
    },
}
```

- [ ] **Step 3.4: 改 _handle_team_spawn**

Modify `tools/team_tool.py:149-183`. Replace the handler body:

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

    # === P4b-T2 NEW: depth 检查 ===
    current_depth = getattr(agent, "spawn_depth", 0) if agent else 0
    max_depth = config.get("team", {}).get("max_depth", 2)
    if current_depth >= max_depth:
        return json.dumps({
            "success": False,
            "error": f"max_depth {max_depth} reached (current: {current_depth})",
            "error_type": "team_max_depth",
        }, ensure_ascii=False)

    role = args.get("role", "worker")
    # task_id 归一化：空字符串/None 都视为「不绑定」
    task_id = args.get("task_id") or None

    try:
        member = coord.spawn(
            name=name, role=role, task=task,
            depth=current_depth + 1,
            task_id=task_id,
        )
        return json.dumps({
            "success": True, "name": name, "pid": member.pid,
            "status": member.status,
            "task_id": task_id,
        }, ensure_ascii=False)
    except ValueError as e:
        # task_id 不存在等
        msg = str(e)
        error_type = "invalid_task_id" if "不存在" in msg else "team_spawn_error"
        return _err(msg, error_type)
    except RuntimeError as e:
        return _err(str(e), "team_spawn_error")
    except Exception as e:
        return _err(f"spawn 失败: {e}", "team_error")
```

- [ ] **Step 3.5: 跑新测试确认通过**

Run: `uv run pytest tests/test_team_tool.py -v -k task_id`

Expected: 4 passed

- [ ] **Step 3.6: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `679 + 4 = 683 passed`

- [ ] **Step 3.7: Commit**

```bash
git add tools/team_tool.py tests/test_team_tool.py
git commit -m "feat(team): team_spawn 工具加 task_id 参数

LLM 调 team_spawn(name, task, task_id=X) 时，handler 把 task_id
转发给 Coordinator.spawn。空字符串归一化为 None（向后兼容）。
task_id 不存在时返回 invalid_task_id 错误。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: task_update / task_complete 加 ownership 校验

**Files:**
- Modify: `tools/task_tools.py:118-136`（_handle_task_update）和 `tools/task_tools.py:139-158`（_handle_task_complete）
- Test: `tests/test_team_task_binding.py`（追加 handler 行为测试）

**Interfaces:**
- Consumes: `agent.team.task_binding.assert_owned`、`TaskOwnershipError`
- Produces: `_handle_task_update` / `_handle_task_complete` 在绑定不匹配时返回 `{"error_type": "permission_denied"}`

- [ ] **Step 4.1: 写失败测试（追加到 test_team_task_binding.py）**

Append to `tests/test_team_task_binding.py`:

```python
# ---------------------------------------------------------------------------
# task_tools handler 集成
# ---------------------------------------------------------------------------

import json

from agent.task_store import get_task_store
from tools.task_tools import _handle_task_update, _handle_task_complete


def test_task_complete_blocks_foreign_id(monkeypatch, tmp_path):
    """env 绑定 task_A → task_complete(task_B) 返回 permission_denied。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    result = _handle_task_complete(
        {"id": "task_B"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("error_type") == "permission_denied"
    assert "task_A" in data.get("error", "")
    assert "task_B" in data.get("error", "")


def test_task_update_blocks_foreign_id(monkeypatch, tmp_path):
    """env 绑定 task_A → task_update(task_B) 返回 permission_denied。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    result = _handle_task_update(
        {"id": "task_B", "status": "completed"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("error_type") == "permission_denied"


def test_task_complete_allows_matching_id(monkeypatch, tmp_path):
    """env 绑定 → task_complete(同 id) 正常执行。"""
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])

    result = _handle_task_complete(
        {"id": task["id"]},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True
    refreshed = store.get(task["id"])
    assert refreshed["status"] == "completed"


def test_task_update_allows_matching_id(monkeypatch, tmp_path):
    """env 绑定 → task_update(同 id) 正常执行。"""
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])

    result = _handle_task_update(
        {"id": task["id"], "status": "in_progress"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_complete_allows_when_unset(monkeypatch, tmp_path):
    """主 agent / legacy 调用（env 未设）正常执行。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    store = get_task_store(harvil_home=str(tmp_path))
    task = store.create(subject="X")

    result = _handle_task_complete(
        {"id": task["id"]},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_create_not_gated(monkeypatch, tmp_path):
    """task_create 不受门控——worker 可创建新任务（不算跨任务操纵）。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    from tools.task_tools import _handle_task_create
    result = _handle_task_create(
        {"subject": "新任务"},
        harvil_home=str(tmp_path),
    )
    data = json.loads(result)
    assert data.get("success") is True


def test_task_list_not_gated(monkeypatch, tmp_path):
    """task_list 不受门控——读取不敏感。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_A")
    from tools.task_tools import _handle_task_list
    result = _handle_task_list({}, harvil_home=str(tmp_path))
    data = json.loads(result)
    # 不报 permission_denied 即可
    assert data.get("error_type") != "permission_denied"
```

- [ ] **Step 4.2: 跑测试确认失败**

Run: `uv run pytest tests/test_team_task_binding.py -v -k blocks`

Expected: 2 FAIL — `assert data.get("error_type") == "permission_denied"` 失败（因为 handler 还没加校验，会去尝试 store.complete 失败，但不是 permission_denied）

- [ ] **Step 4.3: 改 _handle_task_update 和 _handle_task_complete**

Modify `tools/task_tools.py`. Add import at top (after existing imports):

```python
import json
from typing import Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry


def _ownership_denied(msg: str) -> str:
    """把 TaskOwnershipError 消息包成 permission_denied JSON 错误。

    msg 来自 assert_owned 抛出的异常，已包含 bound task_id 和 attempted
    task_id（如 "worker bound to task 'task_A', cannot operate on 'task_B'"）。
    """
    return json.dumps({
        "error": msg,
        "error_type": "permission_denied",
    }, ensure_ascii=False)
```

Then modify `_handle_task_update` (was line 118). Replace the first few lines:

```python
def _handle_task_update(args: dict, **kwargs) -> str:
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return json.dumps({"error": "id 不能为空"}, ensure_ascii=False)

    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))

    store = _get_store(kwargs)
    fields = {}
    for key in ("status", "owner", "description", "subject"):
        if key in args and args[key] is not None:
            if key == "status" and args[key] not in VALID_STATUSES:
                return json.dumps(
                    {"error": f"非法 status: {args[key]}"}, ensure_ascii=False,
                )
            fields[key] = args[key]

    task = store.update(task_id, **fields)
    if task is None:
        return json.dumps({"error": f"任务不存在: {task_id}"}, ensure_ascii=False)
    return json.dumps({"success": True, "task": task}, ensure_ascii=False)
```

And `_handle_task_complete` (was line 139). Replace:

```python
def _handle_task_complete(args: dict, **kwargs) -> str:
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return json.dumps({"error": "id 不能为空"}, ensure_ascii=False)

    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))

    store = _get_store(kwargs)
    task = store.complete(task_id)
    if task is None:
        return json.dumps({"error": f"任务不存在: {task_id}"}, ensure_ascii=False)

    # 检查解锁了哪些任务（含 id/subject/status，方便 LLM 判断下一步）
    ready = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in store.find_ready()
    ]
    return json.dumps({
        "success": True,
        "task": task,
        "unblocked": ready,
    }, ensure_ascii=False)
```

- [ ] **Step 4.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_team_task_binding.py -v`

Expected: 8（Task 1） + 7（Task 4）= 15 passed

- [ ] **Step 4.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `683 + 7 = 690 passed`

- [ ] **Step 4.6: Commit**

```bash
git add tools/task_tools.py tests/test_team_task_binding.py
git commit -m "feat(task): task_update/task_complete 加 ownership 校验

spawned worker（HARVIL_KANBAN_TASK 已设）调用 task_update/task_complete
时，先 assert_owned(id) 校验。绑定不匹配返回 permission_denied，
让 LLM 自行修正。task_create/task_list 不受门控。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 全测回归 + verify.py + push

**Files:**
- No code changes — verification only

- [ ] **Step 5.1: 跑完整测试套件**

Run: `uv run pytest tests/ -v --tb=short`

Expected: `690 passed`（668 起点 + 22 新增）

- [ ] **Step 5.2: 跑复刻检查清单**

Run: `uv run python scripts/verify.py`

Expected: 22/22 PASS（与起点一致）

- [ ] **Step 5.3: 确认 git 状态干净**

Run: `git status --short`

Expected: 只剩 `?? .codegraph/`（索引目录，不提交）

- [ ] **Step 5.4: 跑测试覆盖场景手动验证（可选但推荐）**

模拟一次 e2e：

```python
# scripts/manual_verify_task_binding.py（临时脚本，跑完删）
import os
import tempfile
from pathlib import Path

# 1. 模拟主 agent 建 task
from agent.task_store import TaskStore
tmp = Path(tempfile.mkdtemp())
store = TaskStore(harvil_home=tmp)
task = store.create(subject="测试")
print(f"1. 创建 task: {task['id']}")

# 2. 模拟 worker 进程（env 已注入）
os.environ["HARVIL_KANBAN_TASK"] = task["id"]
from agent.team.task_binding import assert_owned, get_bound_task_id
print(f"2. worker 绑定: {get_bound_task_id()}")
assert_owned(task["id"])
print("   ✓ 自家 task 校验通过")

# 3. 注入场景
try:
    assert_owned("task_other")
    print("   ✗ 注入场景应失败")
except PermissionError as e:
    print(f"   ✓ 注入场景被拒: {e}")

# 4. 模拟主 agent（无 env）
del os.environ["HARVIL_KANBAN_TASK"]
assert_owned("any_task")
print("3. 主 agent 无 env：✓ 不受限")

# 清理
import shutil
shutil.rmtree(tmp)
```

Run: `uv run python scripts/manual_verify_task_binding.py`

Expected:
```
1. 创建 task: task_xxx
2. worker 绑定: task_xxx
   ✓ 自家 task 校验通过
   ✓ 注入场景被拒: worker bound to task 'task_xxx', cannot operate on 'task_other'
3. 主 agent 无 env：✓ 不受限
```

- [ ] **Step 5.5: Push 到 origin**

```bash
git push origin master
```

Expected: 4 个新 commit（Task 1/2/3/4）+ 之前的 rename commit + spec commit 都推送成功

- [ ] **Step 5.6: 更新交接文档**

修改 `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`，在「第 3 批改进」后加：

```markdown
### 第 4 批改进（690 → 第 4 批完成）

| # | 能力 | 说明 |
|---|---|---|
| ⑫ | Worker 任务归属强制 | team_spawn 加 task_id；env 注入 HARVIL_KANBAN_TASK；task_update/complete 校验 ownership |
```

并把「当前状态」从 668 改为 690，把「⑫ Worker 任务归属强制」从「剩余待做」移到「已完成」。

---

## 完成标准

全部满足才算完成：

- [ ] 4 个新 commit 已 push 到 origin/master
- [ ] `uv run pytest tests/ -q` 显示 690 passed
- [ ] `uv run python scripts/verify.py` 22/22 PASS
- [ ] 手动验证脚本输出正确（自家 pass / 注入拒绝 / 主 agent 不受限）
- [ ] 交接文档已更新
