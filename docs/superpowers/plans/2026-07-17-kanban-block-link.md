# Kanban Block/Unblock/Link + 自动心跳桥 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 TaskStore 加 blocked status + cycle-safe task_link + 自动心跳桥（POST_TOOL_USE hook）。

**Architecture:** TaskStore 加 `blocked` status + 2 字段 + `has_path`/`add_dependency` 方法（cycle 检测）；3 个新 handler（task_block/task_unblock/task_link）；新模块 `agent/team/auto_heartbeat.py` 注册到 `AIAgent.__init__` 的 POST_TOOL_USE hook。

**Tech Stack:** Python 3.11+、pytest、uv

**Spec:** `docs/superpowers/specs/2026-07-17-kanban-block-link-auto-heartbeat-design.md`

## Global Constraints

- **语言约定**：注释/文档/commit 用中文；代码标识符用英文（CLAUDE.md）
- **文件 I/O**：必须 `encoding="utf-8"`
- **工具结果契约**：handler 返回 JSON 字符串；错误用 `{"success": False, "error": "...", "error_type": "..."}`
- **依赖管理**：用 `uv`，不要 `pip install`
- **测试基线**：当前 712 测试通过；本计划完成后预期 712 + 23 = 735
- **Ownership 门控**：所有新 handler 复用 `_get_owned_task`（⑪a 已建立）；task_link 双重门控（parent + child 都要过）
- **Hook 签名**：`POST_TOOL_USE` callback 是 `fn(tool_name, args, result) -> Optional[str]`（链式 transform）
- **HookRegistry API**：`hooks_registry.register_post_tool_use(fn, name=...)`

---

## 文件结构

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/task_store.py` | ♻️ 改 | `VALID_STATUSES` 加 `blocked`；`create()` 加 2 字段；加 `has_path` + `add_dependency` 方法 |
| `tools/task_tools.py` | ♻️ 改 | 加 3 schema + 3 handler；`TASK_UPDATE_SCHEMA` status enum 加 `blocked` |
| `agent/team/auto_heartbeat.py` | 🆕 新增 | `maybe_heartbeat()` + `_post_tool_use_hook` + `register(hooks_registry)` |
| `agent/__init__.py` | ♻️ 改 | `AIAgent.__init__` 末尾注册 auto_heartbeat hook |
| `tests/test_kanban_block_link.py` | 🆕 新增 | TaskStore DAG 方法 + 3 handler 测试（~18 个） |
| `tests/test_auto_heartbeat.py` | 🆕 新增 | auto_heartbeat 模块单元测试（~5 个） |

---

## Task 1: TaskStore 加 blocked status + has_path + add_dependency

**Files:**
- Modify: `agent/task_store.py`（VALID_STATUSES、create、加 2 方法）
- Test: `tests/test_kanban_block_link.py`（新建）

**Interfaces:**
- Consumes: 无（foundation）
- Produces:
  - `VALID_STATUSES` 含 `"blocked"`
  - `TaskStore.create()` 返回 dict 含 `block_reason: None` + `block_kind: None`
  - `TaskStore.has_path(start_id, target_id) -> bool` — DFS 沿 blocked_by 边
  - `TaskStore.add_dependency(child_id, parent_id, *, validate=True) -> Optional[dict]` — 加边，含 cycle/self-link 检测（抛 ValueError）

- [ ] **Step 1.1: 写失败测试**

Create `tests/test_kanban_block_link.py`:

```python
"""Kanban block/unblock/link + DAG 测试。"""
import json

import pytest

from agent.task_store import TaskStore, VALID_STATUSES


@pytest.fixture
def store(tmp_path):
    return TaskStore(harvil_home=tmp_path)


# ---------------------------------------------------------------------------
# Task 1: TaskStore has_path + add_dependency
# ---------------------------------------------------------------------------

def test_valid_statuses_includes_blocked():
    assert "blocked" in VALID_STATUSES


def test_create_includes_block_fields(store):
    task = store.create(subject="X")
    assert task["block_reason"] is None
    assert task["block_kind"] is None


def test_has_path_direct_dependency(store):
    """A.blocked_by=[B] → has_path(A, B) == True。"""
    b = store.create(subject="B")
    a = store.create(subject="A", blocked_by=[b["id"]])
    assert store.has_path(a["id"], b["id"]) is True


def test_has_path_indirect(store):
    """A→B→C（A 依赖 B，B 依赖 C）→ has_path(A, C) == True。"""
    c = store.create(subject="C")
    b = store.create(subject="B", blocked_by=[c["id"]])
    a = store.create(subject="A", blocked_by=[b["id"]])
    assert store.has_path(a["id"], c["id"]) is True


def test_has_path_no_path(store):
    """无关任务 → False。"""
    a = store.create(subject="A")
    z = store.create(subject="Z")
    assert store.has_path(a["id"], z["id"]) is False


def test_add_dependency_basic(store):
    """add_dependency(child, parent) → child.blocked_by 含 parent。"""
    parent = store.create(subject="P")
    child = store.create(subject="C")
    updated = store.add_dependency(child["id"], parent["id"])
    assert updated is not None
    assert parent["id"] in updated["blocked_by"]


def test_add_dependency_self_link_raises(store):
    """add_dependency(X, X) → ValueError。"""
    x = store.create(subject="X")
    with pytest.raises(ValueError, match="self-link"):
        store.add_dependency(x["id"], x["id"])


def test_add_dependency_cycle_rejected(store):
    """已有 A→B，再加 B→A → ValueError。"""
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # 现在 B 依赖 A。如果加 A 依赖 B（add_dependency(A, B)），成环
    with pytest.raises(ValueError, match="cycle"):
        store.add_dependency(a["id"], b["id"])


def test_add_dependency_idempotent(store):
    """加同样的边两次 → blocked_by 只一条。"""
    parent = store.create(subject="P")
    child = store.create(subject="C")
    store.add_dependency(child["id"], parent["id"])
    updated = store.add_dependency(child["id"], parent["id"])
    assert updated["blocked_by"].count(parent["id"]) == 1


def test_add_dependency_no_validate_bypasses_cycle(store):
    """validate=False 跳过 cycle 检测（紧急逃生口）。"""
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # 加 A→B 不验证（明知会成环）
    updated = store.add_dependency(a["id"], b["id"], validate=False)
    assert updated is not None
    assert b["id"] in updated["blocked_by"]
```

- [ ] **Step 1.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_block_link.py -v`

Expected: 多数 FAIL — `"blocked" not in VALID_STATUSES`、`KeyError: 'block_reason'`、`AttributeError: 'TaskStore' object has no attribute 'has_path'`

- [ ] **Step 1.3: 改 VALID_STATUSES 加 blocked**

In `agent/task_store.py` line 24, change:

```python
VALID_STATUSES = {"pending", "in_progress", "completed", "deleted"}
```

To:

```python
VALID_STATUSES = {"pending", "in_progress", "completed", "deleted", "blocked"}
```

- [ ] **Step 1.4: 改 create 加 2 字段**

In `agent/task_store.py`, the `create` method's task dict (around line 75-90). Add `block_reason` and `block_kind` after `artifacts`:

```python
task = {
    "id": task_id,
    "subject": subject,
    "description": description,
    "status": "pending",
    "owner": owner,
    "blocked_by": list(blocked_by or []),
    "created_at": _now_iso(),
    "updated_at": _now_iso(),
    "last_heartbeat_at": None,
    "comments": [],
    "artifacts": [],
    "block_reason": None,
    "block_kind": None,
}
```

- [ ] **Step 1.5: 加 has_path + add_dependency 方法**

In `agent/task_store.py`, find the `remove_artifacts` method (added in ⑪a). Add these 2 methods after `remove_artifacts` and BEFORE the `# 全局单例` divider:

```python
    def remove_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """从 artifacts 移除 paths。"""
        task = self.get(task_id)
        if task is None:
            return None
        task["artifacts"] = [
            p for p in task.get("artifacts", []) if p not in paths
        ]
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    # ------------------------------------------------------------------
    # DAG 增强（cycle-safe dependency management）
    # ------------------------------------------------------------------

    def has_path(self, start_id: str, target_id: str) -> bool:
        """DFS：从 start_id 沿 blocked_by 边走，能否到达 target_id？

        blocked_by 语义：A.blocked_by=[B] 表示 A 依赖 B。
        所以"沿 blocked_by 走"= "查 start 依赖谁、间接依赖谁"。
        """
        visited = set()
        stack = [start_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_task = self.get(cur)
            if cur_task is None:
                continue
            for dep in cur_task.get("blocked_by", []):
                if dep == target_id:
                    return True
                stack.append(dep)
        return False

    def add_dependency(
        self, child_id: str, parent_id: str,
        *, validate: bool = True,
    ) -> Optional[dict]:
        """加 child 依赖 parent 的边（child.blocked_by += [parent]）。

        validate=True 时做 cycle 检测：若 parent 已经（直接或间接）依赖 child，拒绝。
        self-link 永远拒绝（即使 validate=False）。
        """
        if parent_id == child_id:
            raise ValueError("self-link forbidden")
        child = self.get(child_id)
        if child is None:
            return None
        if validate:
            if self.has_path(parent_id, child_id):
                raise ValueError(
                    f"cycle detected: {parent_id} 已经依赖 {child_id}，"
                    f"再加 {child_id} → {parent_id} 边会成环"
                )
        blocked_by = child.setdefault("blocked_by", [])
        if parent_id not in blocked_by:
            blocked_by.append(parent_id)
        child["updated_at"] = _now_iso()
        self._write(child_id, child)
        return child
```

- [ ] **Step 1.6: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_block_link.py -v`

Expected: 10 passed

- [ ] **Step 1.7: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `712 + 10 = 722 passed`

- [ ] **Step 1.8: Commit**

```bash
git add agent/task_store.py tests/test_kanban_block_link.py
git commit -m "feat(task): TaskStore 加 blocked status + has_path + add_dependency

VALID_STATUSES 加 'blocked'；create 加 block_reason/block_kind 字段。
新方法 has_path（DFS 沿 blocked_by）和 add_dependency（含 cycle+self-link 检测）。
validate=False 紧急逃生口跳过 cycle 检测。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: task_block + task_unblock handler

**Files:**
- Modify: `tools/task_tools.py`（加 2 schema + 2 handler + 注册；TASK_UPDATE_SCHEMA status enum 加 blocked）
- Test: `tests/test_kanban_block_link.py`（追加 7 个测试）

**Interfaces:**
- Consumes: Task 1 的 `TaskStore.update`（写 status + block_reason + block_kind）；⑪a 的 `_get_owned_task` helper
- Produces: `task_block` / `task_unblock` 工具注册到 `core` toolset

- [ ] **Step 2.1: 写失败测试（追加到 test_kanban_block_link.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 2: task_block + task_unblock handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_block, _handle_task_unblock


def test_block_sets_status_and_reason(store, monkeypatch):
    """task_block → status=blocked, block_reason=reason, block_kind=kind。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "waiting for API", "kind": "needs_input"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "blocked"
    assert data["task"]["block_reason"] == "waiting for API"
    assert data["task"]["block_kind"] == "needs_input"


def test_block_rejects_empty_reason(store, monkeypatch):
    """reason="" → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "   "},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "reason" in data["error"]


def test_block_rejects_invalid_kind(store, monkeypatch):
    """kind="random" → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_block(
        {"id": task["id"], "reason": "X", "kind": "random"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "kind" in data["error"]


def test_block_blocks_foreign_id(store, monkeypatch):
    """跨任务 block 被 permission_denied。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_block(
        {"id": task_b["id"], "reason": "X"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_unblock_resets_to_pending(store, monkeypatch):
    """task_unblock 默认回 pending，清空 block_reason/kind。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    store.update(task["id"], status="blocked",
                 block_reason="X", block_kind="needs_input")
    result = _handle_task_unblock(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "pending"
    assert data["task"]["block_reason"] is None
    assert data["task"]["block_kind"] is None


def test_unblock_custom_status(store, monkeypatch):
    """new_status='in_progress' → status=in_progress。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    store.update(task["id"], status="blocked", block_reason="X")
    result = _handle_task_unblock(
        {"id": task["id"], "new_status": "in_progress"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "in_progress"


def test_unblock_non_blocked_rejected(store, monkeypatch):
    """对 pending 任务调 unblock → error。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_unblock(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "blocked" in data["error"]
```

- [ ] **Step 2.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_block_link.py -v -k "block"`

Expected: ImportError: `cannot import name '_handle_task_block'`

- [ ] **Step 2.3: 加 schema 到 task_tools.py**

After `TASK_ARTIFACTS_SCHEMA` (added in ⑪a Task 4), add:

```python
TASK_BLOCK_SCHEMA = {
    "name": "task_block",
    "description": (
        "把任务标记为 blocked（卡住）。必须填 reason 解释为什么卡住。"
        "可选 kind：'dependency'（等其他任务）/ 'needs_input'（等人决策）/ "
        "'capability'（缺权限/凭证）/ 'transient'（偶发失败可能恢复）。"
        "kind 仅作人类可读标签，不影响自动化行为。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "reason": {"type": "string", "description": "为什么卡住（必填）"},
            "kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": "可选，block 类型标签",
            },
        },
        "required": ["id", "reason"],
    },
}

TASK_UNBLOCK_SCHEMA = {
    "name": "task_unblock",
    "description": (
        "解除 task 的 blocked 状态。默认回到 pending；可选 new_status='in_progress'。"
        "清空 block_reason / block_kind。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "new_status": {
                "type": "string",
                "enum": ["pending", "in_progress"],
                "default": "pending",
            },
        },
        "required": ["id"],
    },
}
```

Also update `TASK_UPDATE_SCHEMA`'s status enum to include `"blocked"`. Find the existing:

```python
"status": {
    "type": "string",
    "enum": ["pending", "in_progress", "completed"],
    "description": "新状态",
},
```

Change to:

```python
"status": {
    "type": "string",
    "enum": ["pending", "in_progress", "completed", "blocked"],
    "description": "新状态",
},
```

- [ ] **Step 2.4: 加 handler 到 task_tools.py**

After `_handle_task_artifacts` (added in ⑪a Task 4), add:

```python
def _handle_task_block(args: dict, **kwargs) -> str:
    """task_block: status=blocked + block_reason + block_kind。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    reason = (args.get("reason") or "").strip()
    if not reason:
        return json.dumps(
            {"error": "reason 不能为空"}, ensure_ascii=False,
        )
    kind = args.get("kind")
    if kind is not None and kind not in (
        "dependency", "needs_input", "capability", "transient",
    ):
        return json.dumps(
            {"error": f"非法 kind: {kind}"}, ensure_ascii=False,
        )
    store = _get_store(kwargs)
    updated = store.update(
        task["id"],
        status="blocked",
        block_reason=reason,
        block_kind=kind,
    )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)


def _handle_task_unblock(args: dict, **kwargs) -> str:
    """task_unblock: 解除 blocked，清空 block_reason/block_kind。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    new_status = args.get("new_status") or "pending"
    if new_status not in ("pending", "in_progress"):
        return json.dumps(
            {"error": f"非法 new_status: {new_status}（只允许 pending 或 in_progress）"},
            ensure_ascii=False,
        )
    if task.get("status") != "blocked":
        return json.dumps(
            {"error": f"任务不是 blocked 状态（当前: {task.get('status')}）"},
            ensure_ascii=False,
        )
    store = _get_store(kwargs)
    updated = store.update(
        task["id"],
        status=new_status,
        block_reason=None,
        block_kind=None,
    )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

Finally, after the existing `registry.register(name="task_artifacts", ...)` line, add:

```python
registry.register(
    name="task_block", toolset="core",
    schema=TASK_BLOCK_SCHEMA, handler=_handle_task_block, emoji="⏸",
)
registry.register(
    name="task_unblock", toolset="core",
    schema=TASK_UNBLOCK_SCHEMA, handler=_handle_task_unblock, emoji="▶",
)
```

- [ ] **Step 2.5: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_block_link.py -v -k "block"`

Expected: 7 passed

- [ ] **Step 2.6: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `722 + 7 = 729 passed`

- [ ] **Step 2.7: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_block_link.py
git commit -m "feat(task): task_block + task_unblock 工具

task_block(id, reason, kind?): status='blocked' + 存 reason/kind。
task_unblock(id, new_status='pending'): 解除 blocked，清空 reason/kind。
TASK_UPDATE_SCHEMA status enum 加 'blocked'。
都受 ownership 门控。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: task_link handler（双重门控 + cycle 检测）

**Files:**
- Modify: `tools/task_tools.py`（加 schema + handler + 注册）
- Test: `tests/test_kanban_block_link.py`（追加 4 个测试）

**Interfaces:**
- Consumes: Task 1 的 `TaskStore.add_dependency(child, parent, validate=True)`；⑫ 的 `assert_owned`
- Produces: `task_link` 工具注册到 `core` toolset

- [ ] **Step 3.1: 写失败测试（追加到 test_kanban_block_link.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 3: task_link handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_link


def test_link_adds_edge(store, monkeypatch):
    """task_link(parent, child) → child.blocked_by 含 parent。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    parent = store.create(subject="P")
    child = store.create(subject="C")
    result = _handle_task_link(
        {"parent_id": parent["id"], "child_id": child["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert parent["id"] in data["task"]["blocked_by"]


def test_link_self_rejected(store, monkeypatch):
    """parent_id == child_id → cycle_detected。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    x = store.create(subject="X")
    result = _handle_task_link(
        {"parent_id": x["id"], "child_id": x["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "cycle_detected"


def test_link_cycle_rejected(store, monkeypatch):
    """已有 A→B，再 link B→A → cycle_detected。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    a = store.create(subject="A")
    b = store.create(subject="B", blocked_by=[a["id"]])
    # 现在 B 依赖 A；尝试让 A 依赖 B（加 A→B 边，即 link parent=B, child=A）
    result = _handle_task_link(
        {"parent_id": b["id"], "child_id": a["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "cycle_detected"


def test_link_respects_ownership(store, monkeypatch):
    """parent 或 child 跨任务 → permission_denied。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    # worker A 想给 B 加依赖（parent=A, child=B），但 B 不是 A 的
    result = _handle_task_link(
        {"parent_id": task_a["id"], "child_id": task_b["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"
```

- [ ] **Step 3.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_block_link.py -v -k link`

Expected: ImportError: `cannot import name '_handle_task_link'`

- [ ] **Step 3.3: 加 schema + handler + 注册**

After `TASK_UNBLOCK_SCHEMA` (added in Task 2), add schema:

```python
TASK_LINK_SCHEMA = {
    "name": "task_link",
    "description": (
        "post-creation 加依赖边：让 child 依赖 parent（parent 完成前 child 不能开始）。"
        "含 cycle 检测和 self-link 拒绝。"
        "parent_id 和 child_id 都过 ownership 门控。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "string", "description": "被依赖的任务 ID"},
            "child_id": {"type": "string", "description": "加依赖的任务 ID"},
        },
        "required": ["parent_id", "child_id"],
    },
}
```

After `_handle_task_unblock` (added in Task 2), add handler:

```python
def _handle_task_link(args: dict, **kwargs) -> str:
    """task_link: 加 child 依赖 parent 的边（含 cycle+self-link 检测，双重 ownership 门控）。"""
    parent_id = (args.get("parent_id") or "").strip()
    child_id = (args.get("child_id") or "").strip()
    if not parent_id or not child_id:
        return json.dumps(
            {"error": "parent_id 和 child_id 必需"}, ensure_ascii=False,
        )
    # 双重 ownership 门控：parent 和 child 都要过
    try:
        assert_owned(parent_id)
        assert_owned(child_id)
    except TaskOwnershipError as e:
        return _ownership_denied(str(e))
    store = _get_store(kwargs)
    try:
        updated = store.add_dependency(child_id, parent_id, validate=True)
    except ValueError as e:
        msg = str(e)
        error_type = (
            "cycle_detected" if "cycle" in msg or "self-link" in msg
            else "invalid_args"
        )
        return json.dumps(
            {"error": msg, "error_type": error_type},
            ensure_ascii=False,
        )
    if updated is None:
        return json.dumps(
            {"error": f"任务不存在: {child_id}"}, ensure_ascii=False,
        )
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

Finally, after `registry.register(name="task_unblock", ...)`, add:

```python
registry.register(
    name="task_link", toolset="core",
    schema=TASK_LINK_SCHEMA, handler=_handle_task_link, emoji="🔗",
)
```

- [ ] **Step 3.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_block_link.py -v -k link`

Expected: 4 passed

- [ ] **Step 3.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `729 + 4 = 733 passed`

- [ ] **Step 3.6: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_block_link.py
git commit -m "feat(task): task_link 工具——post-creation DAG edge 含 cycle 检测

task_link(parent_id, child_id): child.blocked_by += [parent]。
self-link 直接拒绝；加边前 DFS 检测环；双重 ownership 门控（parent 和
child 都要过 assert_owned）。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: auto_heartbeat 模块 + AIAgent hook 注册 + 全测 push

**Files:**
- Create: `agent/team/auto_heartbeat.py`
- Modify: `agent/__init__.py`（AIAgent.__init__ 末尾加 3 行注册）
- Test: `tests/test_auto_heartbeat.py`（新建）

**Interfaces:**
- Consumes: ⑫ 的 `task_binding.get_bound_task_id()`；⑪a 的 `TaskStore.heartbeat(id)`；现有 `HookRegistry.register_post_tool_use`
- Produces: `auto_heartbeat.maybe_heartbeat()` / `auto_heartbeat.register(hooks_registry)` / `auto_heartbeat._post_tool_use_hook(tool_name, args, result)`

- [ ] **Step 4.1: 写失败测试**

Create `tests/test_auto_heartbeat.py`:

```python
"""auto_heartbeat 模块单元测试。"""
import pytest

from agent.team.auto_heartbeat import (
    maybe_heartbeat,
    reset_for_test,
    register,
    _post_tool_use_hook,
)
from agent.task_store import TaskStore


@pytest.fixture(autouse=True)
def _reset_state():
    """每个测试前重置 rate-limit 计时。"""
    reset_for_test()
    yield
    reset_for_test()


def test_maybe_heartbeat_noop_without_env(monkeypatch, tmp_path):
    """env 未设 → return False，不调 store。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    # env 没设，task 存在也不应触发
    result = maybe_heartbeat()
    assert result is False
    refreshed = store.get(task["id"])
    assert refreshed["last_heartbeat_at"] is None


def test_maybe_heartbeat_writes_with_env(monkeypatch, tmp_path):
    """env 设了 + task 存在 → 写入 last_heartbeat_at。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])
    # 注意：maybe_heartbeat 用全局 task_store 单例，需要把 task 加到单例能找到的地方
    # 测试里直接用 monkeypatch 替换 get_task_store
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", store)
    result = maybe_heartbeat()
    assert result is True
    refreshed = store.get(task["id"])
    assert refreshed["last_heartbeat_at"] is not None


def test_maybe_heartbeat_rate_limit(monkeypatch, tmp_path):
    """连续调两次 < 60s → 第二次 no-op。"""
    store = TaskStore(harvil_home=tmp_path)
    task = store.create(subject="X")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task["id"])
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", store)

    first = maybe_heartbeat()
    assert first is True
    first_ts = store.get(task["id"])["last_heartbeat_at"]
    # 立即再调
    second = maybe_heartbeat()
    assert second is False
    second_ts = store.get(task["id"])["last_heartbeat_at"]
    assert first_ts == second_ts  # 没更新


def test_maybe_heartbeat_silent_failure(monkeypatch, tmp_path):
    """store.heartbeat 抛异常 → maybe_heartbeat 返 False 不上抛。"""
    monkeypatch.setenv("HARVIL_KANBAN_TASK", "task_nonexistent")
    # 全局单例为 None → get_task_store() 走默认路径，找不到 task 返 None
    # maybe_heartbeat 应静默
    import agent.task_store as ts_module
    monkeypatch.setattr(ts_module, "_task_store", None)
    result = maybe_heartbeat()
    assert result is False  # 不抛


def test_post_tool_use_hook_returns_none(monkeypatch, tmp_path):
    """_post_tool_use_hook 永远返回 None（不替换 result）。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    ret = _post_tool_use_hook("some_tool", {"x": 1}, "result string")
    assert ret is None


def test_register_none_noop():
    """register(None) 不抛。"""
    register(None)  # 不应抛


def test_register_real_registry():
    """register 真实 HookRegistry → 内部 register_post_tool_use 被调用。"""
    from agent.hooks import HookRegistry
    reg = HookRegistry()
    register(reg)
    # 验证：注册后 POST_TOOL_USE hook 列表非空
    from agent.hooks import HookEvent
    assert len(reg._hooks[HookEvent.POST_TOOL_USE]) >= 1
```

- [ ] **Step 4.2: 跑测试确认失败**

Run: `uv run pytest tests/test_auto_heartbeat.py -v`

Expected: `ModuleNotFoundError: No module named 'agent.team.auto_heartbeat'`

- [ ] **Step 4.3: 创建 auto_heartbeat.py 模块**

Create `agent/team/auto_heartbeat.py`:

```python
"""自动心跳桥：runtime 活动每 60s 自动 bump task.last_heartbeat_at。

防 spawned worker 跑长任务时 dispatcher watchdog 误回收（HarvilAgent 暂时
没有 watchdog，但字段值得维护，未来 dispatcher 接入即可用）。

逻辑：
  POST_TOOL_USE hook 触发（每次工具调用结束）→
    读 HARVIL_KANBAN_TASK env →
      未设（主 agent / legacy）→ no-op
      已设 → rate-limit（60s/进程）→ TaskStore.heartbeat(tid)

所有失败静默（log debug），不能影响 agent 主循环。

Hook 签名约定：fn(tool_name, args, result) -> Optional[str]
本 hook 是 side-effect only，永远返回 None（不替换 result）。
"""
import logging
import time

logger = logging.getLogger(__name__)


_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_last_attempt: float = 0.0


def maybe_heartbeat() -> bool:
    """Best-effort 自动心跳。返回 True 表示真的写了 heartbeat，False 表示跳过/失败。

    不会抛异常——调用方（hook）不用 try/except。
    使用 TaskStore 全局单例（spawned worker 的 agent_home 与主 agent 一致）。
    """
    global _last_attempt
    try:
        from agent.team.task_binding import get_bound_task_id
        tid = get_bound_task_id()
        if not tid:
            return False
        now = time.monotonic()
        if (now - _last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return False
        _last_attempt = now
        from agent.task_store import get_task_store
        store = get_task_store()
        result = store.heartbeat(tid)
        return result is not None
    except Exception:
        logger.debug("auto-heartbeat failed", exc_info=True)
        return False


def reset_for_test() -> None:
    """测试用：重置 rate-limit 计时。"""
    global _last_attempt
    _last_attempt = 0.0


def _post_tool_use_hook(tool_name: str, args: dict, result: str):
    """POST_TOOL_USE hook 签名：fn(tool_name, args, result) -> Optional[str]。

    本 hook 是 side-effect only，永远返回 None（不替换 result）。
    异常被 HookRegistry 吞掉（视为 None），maybe_heartbeat 内部已 try/except 是双保险。
    """
    maybe_heartbeat()
    return None


def register(hooks_registry) -> None:
    """注册到 HookRegistry。

    hooks_registry: agent/hooks.py 的 HookRegistry 实例。
    传入 None 时 no-op（hooks 系统未启用）。
    """
    if hooks_registry is None:
        return
    hooks_registry.register_post_tool_use(
        _post_tool_use_hook, name="auto_heartbeat",
    )
```

- [ ] **Step 4.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_auto_heartbeat.py -v`

Expected: 7 passed

- [ ] **Step 4.5: 注册 hook 到 AIAgent.__init__**

In `agent/__init__.py`, find `self.hooks_registry = hooks_registry` (around line 149). After the existing `__init__` body but before the end of the method, add:

```python
        # === ⑪b NEW: 自动心跳桥 ===
        # spawned worker 每次工具调用后自动 bump task.last_heartbeat_at
        # 主 agent 无 HARVIL_KANBAN_TASK env，no-op
        try:
            from agent.team.auto_heartbeat import register as _register_auto_heartbeat
            _register_auto_heartbeat(self.hooks_registry)
        except Exception as e:
            logger.warning("注册 auto_heartbeat hook 失败（不影响主流程）: %s", e)
```

**注意**：必须放在 `self.hooks_registry = hooks_registry` 之后。包裹 try/except 防止注册失败影响 agent 启动。

- [ ] **Step 4.6: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `733 + 7 = 740 passed`

- [ ] **Step 4.7: 手动验证 hook 集成（可选）**

```python
# scripts/manual_verify_auto_heartbeat.py（临时脚本）
import os
import tempfile
from pathlib import Path

from agent.task_store import TaskStore
from agent.team.auto_heartbeat import maybe_heartbeat, reset_for_test

# 模拟 spawned worker
tmp = Path(tempfile.mkdtemp())
store = TaskStore(harvil_home=tmp)
task = store.create(subject="测试")

# 注入全局单例（agent 启动时 cli.py 会做这事）
import agent.task_store as ts
ts._task_store = store

os.environ["HARVIL_KANBAN_TASK"] = task["id"]
reset_for_test()

print(f"1. 创建 task: {task['id']}")
print(f"2. heartbeat 前: last_heartbeat_at={task['last_heartbeat_at']}")
result = maybe_heartbeat()
print(f"3. heartbeat 结果: {result}")
refreshed = store.get(task["id"])
print(f"4. heartbeat 后: last_heartbeat_at={refreshed['last_heartbeat_at']}")

# 主 agent 场景
del os.environ["HARVIL_KANBAN_TASK"]
reset_for_test()
result2 = maybe_heartbeat()
print(f"5. 主 agent 无 env: maybe_heartbeat={result2}（应为 False）")

import shutil
shutil.rmtree(tmp)
print("\n全部通过。")
```

Run: `uv run python scripts/manual_verify_auto_heartbeat.py`

Expected:
```
1. 创建 task: task_xxx
2. heartbeat 前: last_heartbeat_at=None
3. heartbeat 结果: True
4. heartbeat 后: last_heartbeat_at=2026-...
5. 主 agent 无 env: maybe_heartbeat=False（应为 False）

全部通过。
```

- [ ] **Step 4.8: 删临时脚本 + Commit**

```bash
rm scripts/manual_verify_auto_heartbeat.py
git add agent/team/auto_heartbeat.py agent/__init__.py tests/test_auto_heartbeat.py
git commit -m "feat(team): auto_heartbeat 自动心跳桥 + AIAgent hook 注册

新模块 agent/team/auto_heartbeat.py：maybe_heartbeat() + 60s rate-limit
+ POST_TOOL_USE hook 签名 + register(hooks_registry)。spawned worker
每次工具调用后自动 bump task.last_heartbeat_at；主 agent no-op；
失败静默 log debug。AIAgent.__init__ 自动注册。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 全测回归 + push + 更新交接文档

**Files:**
- No code changes — verification only

- [ ] **Step 5.1: 跑完整测试套件**

Run: `uv run pytest tests/ -v --tb=short | tail -20`

Expected: `740 passed`

- [ ] **Step 5.2: 跑复刻检查清单**

Run: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python scripts/verify.py 2>&1 | tail -3`

Expected: `总计 22：19 通过，3 失败`（3 个 pre-existing MemoryStore 签名问题，与本批无关）

- [ ] **Step 5.3: 确认 git 状态干净**

Run: `git status --short`

Expected: 只剩 `?? .codegraph/`

- [ ] **Step 5.4: Push 到 origin**

```bash
git push origin master
```

Expected: 5 个新 commit（Task 1-4 + 本验证 task）+ spec + plan commit 推送成功

- [ ] **Step 5.5: 更新交接文档**

修改 `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`：
- 「当前状态」从 712 改为 740
- 在「第 5 批改进（690 → 712）」后加：

```markdown
### 第 6 批改进（712 → 740）

| # | 能力 | 说明 |
|---|---|---|
| ⑪b | Kanban block/unblock/link + 自动心跳桥 | 新 status `blocked` + block_reason/block_kind；3 新工具 task_block/task_unblock/task_link（含 cycle 检测）；auto_heartbeat 模块挂 POST_TOOL_USE hook，每 60s 自动续心跳 |
```

- 从「剩余待做」删除 ⑪b 行（已完成）

---

## 完成标准

- [ ] 4 个新 commit 已 push 到 origin/master
- [ ] `uv run pytest tests/ -q` 显示 740 passed
- [ ] verify.py 与起点一致（19 通过 + 3 pre-existing 失败）
- [ ] 交接文档已更新（712 → 740，⑪b 从剩余移入完成）
