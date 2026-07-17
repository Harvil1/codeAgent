# Kanban Heartbeat / Comment / Artifacts 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 TaskStore 加 heartbeat / comment / artifacts 三类增强，加 3 个新工具 + 扩展 task_complete。

**Architecture:** TaskStore JSON 加 3 字段 + 4 方法；task_tools.py 加 3 个新 handler 复用 helper `_get_owned_task`；artifacts 走严格验证（存在+可读+≤100MB）；所有新工具受 ⑫ ownership 门控。

**Tech Stack:** Python 3.11+、pytest、uv

**Spec:** `docs/superpowers/specs/2026-07-17-kanban-enhancements-design.md`

## Global Constraints

- **语言约定**：注释/文档/commit 用中文；代码标识符用英文（CLAUDE.md 约定）
- **文件 I/O**：必须 `encoding="utf-8"`（CLAUDE.md 强制）
- **工具结果契约**：所有 handler 返回 JSON 字符串；错误用 `{"success": False, "error": "...", "error_type": "..."}`（CLAUDE.md 铁律）
- **依赖管理**：用 `uv`，不要用 `pip install`
- **测试基线**：当前 690 测试通过；本计划完成后预期 690 + 22 = 712（无回归）
- **Ownership 门控常量**：`agent.team.task_binding.ENV_VAR = "HARVIL_KANBAN_TASK"`（⑫ 已建立）
- **Artifacts 上限**：单文件 `MAX_ARTIFACT_SIZE = 100 * 1024 * 1024`（100MB）

---

## 文件结构

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/task_store.py` | ♻️ 改 | TaskStore 加 3 字段（create 初始化）+ 4 方法 |
| `tools/task_tools.py` | ♻️ 改 | 加 3 schema + 3 handler；扩展 task_complete；加 helper `_get_owned_task` / `_infer_author` / `_validate_artifact_path` / `_artifact_error_type`；现有 update/complete 重构走 helper |
| `tests/test_kanban_enhancements.py` | 🆕 新增 | ~22 个测试覆盖所有新行为 |

---

## Task 1: TaskStore 加字段 + 4 方法

**Files:**
- Modify: `agent/task_store.py`（`create` 方法 + 类尾部加 4 个新方法）
- Test: `tests/test_kanban_enhancements.py`（新建）

**Interfaces:**
- Consumes: 无（foundation task）
- Produces: `TaskStore.create()` 现在返回的 dict 含 `last_heartbeat_at` / `comments` / `artifacts` 字段；新方法 `heartbeat(task_id)` / `add_comment(task_id, *, author, content)` / `add_artifacts(task_id, paths)` / `remove_artifacts(task_id, paths)`

- [ ] **Step 1.1: 写失败测试**

Create `tests/test_kanban_enhancements.py`:

```python
"""Kanban heartbeat / comment / artifacts 测试。"""
import json
from pathlib import Path

import pytest

from agent.task_store import TaskStore


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return TaskStore(harvil_home=tmp_path)


# ---------------------------------------------------------------------------
# Task 1: TaskStore 新字段 + 4 方法
# ---------------------------------------------------------------------------

def test_create_includes_new_fields(store):
    """create 出来的 task 含 last_heartbeat_at / comments / artifacts。"""
    task = store.create(subject="X")
    assert task["last_heartbeat_at"] is None
    assert task["comments"] == []
    assert task["artifacts"] == []


def test_heartbeat_sets_timestamp(store):
    import time
    task = store.create(subject="X")
    before = task["updated_at"]
    time.sleep(0.01)  # 确保 timestamp 不同
    updated = store.heartbeat(task["id"])
    assert updated is not None
    assert updated["last_heartbeat_at"] is not None
    assert updated["last_heartbeat_at"] > before


def test_add_comment_appends(store):
    task = store.create(subject="X")
    updated = store.add_comment(task["id"], author="w1", content="hi")
    assert len(updated["comments"]) == 1
    c = updated["comments"][0]
    assert c["author"] == "w1"
    assert c["content"] == "hi"
    assert "created_at" in c
    # 再加一条
    updated2 = store.add_comment(task["id"], author="w2", content="yo")
    assert len(updated2["comments"]) == 2


def test_add_artifacts_dedups(store):
    task = store.create(subject="X")
    updated = store.add_artifacts(task["id"], ["/a/b.txt", "/c/d.txt"])
    assert updated["artifacts"] == ["/a/b.txt", "/c/d.txt"]
    # 重复加同样的 → 去重
    updated2 = store.add_artifacts(task["id"], ["/a/b.txt", "/e/f.txt"])
    assert updated2["artifacts"] == ["/a/b.txt", "/c/d.txt", "/e/f.txt"]


def test_remove_artifacts(store):
    task = store.create(subject="X")
    store.add_artifacts(task["id"], ["/a", "/b", "/c"])
    updated = store.remove_artifacts(task["id"], ["/b"])
    assert updated["artifacts"] == ["/a", "/c"]


def test_legacy_task_compat(store, tmp_path):
    """旧 task JSON 没新字段时，setdefault 兜底。"""
    # 手动写一个不带新字段的 task JSON
    legacy = {
        "id": "task_legacy",
        "subject": "old",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    f = store._task_file("task_legacy")
    f.write_text(json.dumps(legacy), encoding="utf-8")
    # 用新方法不应崩
    t = store.add_comment("task_legacy", author="x", content="y")
    assert t is not None
    assert len(t["comments"]) == 1
    t2 = store.add_artifacts("task_legacy", ["/p"])
    assert t2["artifacts"] == ["/p"]
```

- [ ] **Step 1.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_enhancements.py -v`

Expected: 6 FAIL — `KeyError: 'last_heartbeat_at'` for the first test, `AttributeError: 'TaskStore' object has no attribute 'heartbeat'` for the rest.

- [ ] **Step 1.3: 改 TaskStore.create 加 3 字段**

In `agent/task_store.py`, find the `create` method. Replace the `task = {...}` block (around line 75-84):

```python
    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> dict:
        """创建任务。"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
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
        }
        self._write(task_id, task)
        logger.info("创建任务 %s: %s", task_id, subject)
        return task
```

- [ ] **Step 1.4: 加 4 个新方法到 TaskStore 类**

Append 4 methods to `TaskStore` class, right BEFORE the `# 依赖` section divider (around line 140). Find the existing `find_blocked` method and add AFTER it:

```python
    def find_blocked(self) -> List[dict]:
        """找出依赖未满足的 pending 任务。"""
        blocked = []
        for t in self.list_all(status="pending"):
            if not self.can_start(t["id"]):
                blocked.append(t)
        return blocked

    # ------------------------------------------------------------------
    # Kanban 增强（heartbeat / comments / artifacts）
    # ------------------------------------------------------------------

    def heartbeat(self, task_id: str) -> Optional[dict]:
        """更新 last_heartbeat_at 为当前时间。"""
        return self.update(task_id, last_heartbeat_at=_now_iso())

    def add_comment(
        self, task_id: str, *, author: str, content: str,
    ) -> Optional[dict]:
        """追加一条 comment。comments 只增不删。"""
        task = self.get(task_id)
        if task is None:
            return None
        task.setdefault("comments", []).append({
            "author": author,
            "content": content,
            "created_at": _now_iso(),
        })
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def add_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """把 paths 去重追加到 artifacts。"""
        task = self.get(task_id)
        if task is None:
            return None
        existing = task.setdefault("artifacts", [])
        for p in paths:
            if p not in existing:
                existing.append(p)
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

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
```

- [ ] **Step 1.5: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_enhancements.py -v`

Expected: 6 passed

- [ ] **Step 1.6: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `690 + 6 = 696 passed`

- [ ] **Step 1.7: Commit**

```bash
git add agent/task_store.py tests/test_kanban_enhancements.py
git commit -m "feat(task): TaskStore 加 heartbeat/comments/artifacts 字段+方法

create 时初始化 3 个新字段。新增 4 方法：
heartbeat / add_comment / add_artifacts / remove_artifacts。
setdefault 兜底兼容旧 task JSON。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: task_heartbeat 工具 + 共用 helper

**Files:**
- Modify: `tools/task_tools.py`（加 helper `_get_owned_task` / `_infer_author`，加 schema + handler，注册）
- Test: `tests/test_kanban_enhancements.py`（追加 4 个测试）

**Interfaces:**
- Consumes: Task 1 的 `TaskStore.heartbeat(id)` / `TaskStore.add_comment(id, *, author, content)`；⑫ 的 `assert_owned` / `TaskOwnershipError`
- Produces:
  - `_get_owned_task(args: dict, kwargs: dict) -> Tuple[Optional[dict], Optional[str]]` — 返回 `(task, None)` 或 `(None, error_json)`
  - `_infer_author(kwargs: dict) -> str` — `"main"` 或 team_name
  - `task_heartbeat` 工具注册到 `core` toolset

- [ ] **Step 2.1: 写失败测试（追加到 test_kanban_enhancements.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 2: task_heartbeat handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_heartbeat, _infer_author


def test_heartbeat_handler_updates_task(store, monkeypatch):
    """handler 调用后 task.last_heartbeat_at 非 None。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_heartbeat(
        {"id": task["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["last_heartbeat_at"] is not None


def test_heartbeat_handler_with_note_adds_comment(store, monkeypatch):
    """带 note 的 heartbeat 同时加一条 comment，author=main（无 team_name）。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_heartbeat(
        {"id": task["id"], "note": "epoch 50/100"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    refreshed = store.get(task["id"])
    assert len(refreshed["comments"]) == 1
    assert refreshed["comments"][0]["content"] == "epoch 50/100"
    assert refreshed["comments"][0]["author"] == "main"


def test_heartbeat_handler_blocks_foreign_id(store, monkeypatch):
    """env 绑 task_A 时 heartbeat(task_B) → permission_denied。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_heartbeat(
        {"id": task_b["id"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_infer_author():
    """_infer_author: 有 team_name → team_name；无 → 'main'。"""
    assert _infer_author({"team_name": "w1"}) == "w1"
    assert _infer_author({}) == "main"
    assert _infer_author({"team_name": None}) == "main"
```

- [ ] **Step 2.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k heartbeat`

Expected: ImportError: `cannot import name '_handle_task_heartbeat' from 'tools.task_tools'`

- [ ] **Step 2.3: 加 helper + schema + handler 到 task_tools.py**

In `tools/task_tools.py`, the file currently has these imports at top:

```python
import json
from typing import Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry
```

And the existing `_ownership_denied` helper from ⑫. Below `_ownership_denied`, add 2 new helpers:

```python
def _ownership_denied(msg: str) -> str:
    """把 TaskOwnershipError 消息包成 permission_denied JSON 错误。"""
    return json.dumps({
        "error": msg,
        "error_type": "permission_denied",
    }, ensure_ascii=False)


def _get_owned_task(args: dict, kwargs: dict):
    """过 ownership + 取任务。失败返回 (None, error_json)；成功返回 (task, None)。

    所有 task_* 写工具共用此 helper，避免 empty-id + ownership 检查重复。
    """
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return None, json.dumps(
            {"error": "id 不能为空"}, ensure_ascii=False,
        )
    try:
        assert_owned(task_id)
    except TaskOwnershipError as e:
        return None, _ownership_denied(str(e))
    store = _get_store(kwargs)
    task = store.get(task_id)
    if task is None:
        return None, json.dumps(
            {"error": f"任务不存在: {task_id}"}, ensure_ascii=False,
        )
    return task, None


def _infer_author(kwargs: dict) -> str:
    """从上下文推断 comment author。"""
    team_name = kwargs.get("team_name")
    if team_name:
        return team_name
    return "main"
```

Then, after the existing `TASK_LIST_SCHEMA` (around line 91), add the new schema:

```python
TASK_HEARTBEAT_SCHEMA = {
    "name": "task_heartbeat",
    "description": (
        "报告当前任务仍在进行（更新 last_heartbeat_at）。"
        "长任务（训练/编码/爬虫）每几分钟调一次。"
        "可选 note 会作为 comment 追加。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "note": {"type": "string", "description": "可选，附加为 comment"},
        },
        "required": ["id"],
    },
}
```

Then, after `_handle_task_list` (around line 169), add the new handler:

```python
def _handle_task_heartbeat(args: dict, **kwargs) -> str:
    """更新 last_heartbeat_at；note 非空时附加为 comment。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    note = args.get("note")
    store = _get_store(kwargs)
    if note:
        author = _infer_author(kwargs)
        store.add_comment(task["id"], author=author, content=note)
    updated = store.heartbeat(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

Finally, register the new tool. Find the existing `registry.register(name="task_list", ...)` line and add AFTER it:

```python
registry.register(
    name="task_heartbeat", toolset="core",
    schema=TASK_HEARTBEAT_SCHEMA, handler=_handle_task_heartbeat, emoji="💓",
)
```

- [ ] **Step 2.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k "heartbeat or infer_author"`

Expected: 4 passed

- [ ] **Step 2.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `696 + 4 = 700 passed`

- [ ] **Step 2.6: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_enhancements.py
git commit -m "feat(task): task_heartbeat 工具 + 共用 helper

新 helper：_get_owned_task（empty-id + ownership + 取任务三合一）、
_infer_author（从 team_name 推断作者）。新工具 task_heartbeat：
更新 last_heartbeat_at，note 非空时附加为 comment。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: task_comment 工具

**Files:**
- Modify: `tools/task_tools.py`（加 schema + handler + 注册）
- Test: `tests/test_kanban_enhancements.py`（追加 3 个测试）

**Interfaces:**
- Consumes: Task 1 的 `TaskStore.add_comment(id, *, author, content)`；Task 2 的 `_get_owned_task` / `_infer_author`
- Produces: `task_comment` 工具注册到 `core` toolset

- [ ] **Step 3.1: 写失败测试（追加到 test_kanban_enhancements.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 3: task_comment handler
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_comment


def test_comment_handler_appends(store, monkeypatch):
    """handler 后 comments 多一条。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_comment(
        {"id": task["id"], "content": "first comment"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    refreshed = store.get(task["id"])
    assert len(refreshed["comments"]) == 1
    assert refreshed["comments"][0]["content"] == "first comment"


def test_comment_handler_rejects_empty_content(store, monkeypatch):
    """content 为空 → 错误。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_comment(
        {"id": task["id"], "content": "   "},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert "error" in data
    assert "content" in data["error"]


def test_comment_handler_blocks_foreign_id(store, monkeypatch):
    """跨任务 comment 被拒。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    result = _handle_task_comment(
        {"id": task_b["id"], "content": "hi"},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"
```

- [ ] **Step 3.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k comment_handler`

Expected: ImportError: `cannot import name '_handle_task_comment'`

- [ ] **Step 3.3: 加 schema + handler 到 task_tools.py**

After the `TASK_HEARTBEAT_SCHEMA` you added in Task 2, add:

```python
TASK_COMMENT_SCHEMA = {
    "name": "task_comment",
    "description": (
        "给任务追加一条持久化留言（写进任务本，跨会话保留）。"
        "用于：给下一个 worker 留问题、记录部分发现、记设计决策。"
        "临时推理不要写这里，放普通回复里。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "content": {"type": "string", "description": "留言内容"},
        },
        "required": ["id", "content"],
    },
}
```

After `_handle_task_heartbeat` (added in Task 2), add:

```python
def _handle_task_comment(args: dict, **kwargs) -> str:
    """追加 comment 到 task.comments。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    content = (args.get("content") or "").strip()
    if not content:
        return json.dumps(
            {"error": "content 不能为空"}, ensure_ascii=False,
        )
    author = _infer_author(kwargs)
    store = _get_store(kwargs)
    updated = store.add_comment(task["id"], author=author, content=content)
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

Finally, after the `registry.register(name="task_heartbeat", ...)` line, add:

```python
registry.register(
    name="task_comment", toolset="core",
    schema=TASK_COMMENT_SCHEMA, handler=_handle_task_comment, emoji="💬",
)
```

- [ ] **Step 3.4: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k comment_handler`

Expected: 3 passed

- [ ] **Step 3.5: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `700 + 3 = 703 passed`

- [ ] **Step 3.6: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_enhancements.py
git commit -m "feat(task): task_comment 工具——持久化留言

LLM 可给任务追加 {author, content, created_at} 结构的 comment。
跨会话保留。受 ownership 门控。空 content 拒绝。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: task_artifacts 工具 + 验证

**Files:**
- Modify: `tools/task_tools.py`（加 schema + handler + `_validate_artifact_path` / `_artifact_error_type` helper + `MAX_ARTIFACT_SIZE` 常量 + 顶部 import；注册）
- Test: `tests/test_kanban_enhancements.py`（追加 7 个测试）

**Interfaces:**
- Consumes: Task 1 的 `TaskStore.add_artifacts(id, paths)` / `TaskStore.remove_artifacts(id, paths)`；Task 2 的 `_get_owned_task`
- Produces:
  - `MAX_ARTIFACT_SIZE = 100 * 1024 * 1024`（模块级常量）
  - `_validate_artifact_path(path: str) -> Optional[str]` — 返回 None=OK 或错误描述
  - `_artifact_error_type(msg: str) -> str` — `"artifact_too_large"` 或 `"invalid_artifact_path"`
  - `task_artifacts` 工具注册到 `core` toolset

- [ ] **Step 4.1: 写失败测试（追加到 test_kanban_enhancements.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 4: task_artifacts handler + validation
# ---------------------------------------------------------------------------

from tools.task_tools import (
    _handle_task_artifacts,
    _validate_artifact_path,
    _artifact_error_type,
    MAX_ARTIFACT_SIZE,
)


def test_artifacts_handler_add_valid_path(store, monkeypatch, tmp_path):
    """加真实文件 → 成功。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    f = tmp_path / "out.txt"
    f.write_text("hello", encoding="utf-8")
    result = _handle_task_artifacts(
        {"id": task["id"], "add": [str(f)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert str(f) in data["task"]["artifacts"]


def test_artifacts_handler_add_nonexistent(store, monkeypatch):
    """加不存在的路径 → invalid_artifact_path，列表不变。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_artifacts(
        {"id": task["id"], "add": ["/nonexistent/file.txt"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "invalid_artifact_path"
    refreshed = store.get(task["id"])
    assert refreshed["artifacts"] == []


def test_artifacts_handler_add_directory(store, monkeypatch, tmp_path):
    """加目录（不是文件）→ invalid_artifact_path。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    result = _handle_task_artifacts(
        {"id": task["id"], "add": [str(tmp_path)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "invalid_artifact_path"
    assert "不是文件" in data["error"]


def test_artifacts_handler_atomic_failure(store, monkeypatch, tmp_path):
    """add [valid, invalid, valid] → 整批失败，列表不变。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    valid = tmp_path / "ok.txt"
    valid.write_text("ok", encoding="utf-8")
    result = _handle_task_artifacts(
        {"id": task["id"], "add": [str(valid), "/nonexistent", str(valid)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "invalid_artifact_path"
    refreshed = store.get(task["id"])
    assert refreshed["artifacts"] == []  # 整批失败


def test_artifacts_handler_remove(store, monkeypatch, tmp_path):
    """add 后 remove → 列表清空。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    f1 = tmp_path / "a.txt"
    f1.write_text("a", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("b", encoding="utf-8")
    # add
    _handle_task_artifacts(
        {"id": task["id"], "add": [str(f1), str(f2)]},
        harvil_home=str(store._dir.parent),
    )
    # remove f1
    result = _handle_task_artifacts(
        {"id": task["id"], "remove": [str(f1)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert str(f1) not in data["task"]["artifacts"]
    assert str(f2) in data["task"]["artifacts"]


def test_artifacts_handler_blocks_foreign_id(store, monkeypatch, tmp_path):
    """跨任务加附件被拒。"""
    task_a = store.create(subject="A")
    task_b = store.create(subject="B")
    monkeypatch.setenv("HARVIL_KANBAN_TASK", task_a["id"])
    f = tmp_path / "x.txt"
    f.write_text("x", encoding="utf-8")
    result = _handle_task_artifacts(
        {"id": task_b["id"], "add": [str(f)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_validate_artifact_path_oversized(monkeypatch, tmp_path):
    """超过 100MB → artifact_too_large。"""
    f = tmp_path / "big.txt"
    f.write_text("small", encoding="utf-8")  # 真实小文件
    # mock stat 返回超大 size
    class _FakeStat:
        st_size = MAX_ARTIFACT_SIZE + 1
    monkeypatch.setattr(Path, "stat", lambda self: _FakeStat())
    err = _validate_artifact_path(str(f))
    assert err is not None
    assert "过大" in err
    assert _artifact_error_type(err) == "artifact_too_large"
```

- [ ] **Step 4.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k "artifacts_handler or validate_artifact"`

Expected: ImportError: `cannot import name '_handle_task_artifacts'`

- [ ] **Step 4.3: 加顶部 import**

In `tools/task_tools.py`, change the import block at the top from:

```python
import json
from typing import Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry
```

To:

```python
import json
import os
from pathlib import Path
from typing import List, Optional

from agent.task_store import get_task_store, VALID_STATUSES
from agent.team.task_binding import assert_owned, TaskOwnershipError
from tools.registry import registry


MAX_ARTIFACT_SIZE = 100 * 1024 * 1024  # 100MB
```

- [ ] **Step 4.4: 加 validation helpers**

After the `_infer_author` helper (added in Task 2), add:

```python
def _validate_artifact_path(path: str) -> Optional[str]:
    """验证单个附件路径。返回 None=OK，否则返回错误描述。"""
    p = Path(path)
    if not p.exists():
        return f"路径不存在: {path}"
    if not p.is_file():
        return f"不是文件（拒绝目录）: {path}"
    if not os.access(path, os.R_OK):
        return f"不可读: {path}"
    size = p.stat().st_size
    if size > MAX_ARTIFACT_SIZE:
        return f"文件过大: {size} bytes（上限 {MAX_ARTIFACT_SIZE}）"
    return None


def _artifact_error_type(msg: str) -> str:
    """根据验证错误消息选 error_type。"""
    if "过大" in msg:
        return "artifact_too_large"
    return "invalid_artifact_path"
```

- [ ] **Step 4.5: 加 schema + handler**

After the `TASK_COMMENT_SCHEMA` you added in Task 3, add:

```python
TASK_ARTIFACTS_SCHEMA = {
    "name": "task_artifacts",
    "description": (
        "管理任务的交付物文件路径列表（add / remove）。"
        "路径必须存在、可读、单个文件 ≤100MB。"
        "用于让下游（gateway notifier 等）知道交付物在哪。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要加入的文件绝对路径列表",
            },
            "remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要移除的文件路径列表",
            },
        },
        "required": ["id"],
    },
}
```

After `_handle_task_comment` (added in Task 3), add:

```python
def _handle_task_artifacts(args: dict, **kwargs) -> str:
    """管理 task.artifacts 列表（add / remove）。原子性：批量 add 任一失败整批拒绝。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    add_paths: List[str] = args.get("add") or []
    remove_paths: List[str] = args.get("remove") or []
    store = _get_store(kwargs)

    # 原子性：先全部验证 add，任一失败 → 整批拒绝
    for p in add_paths:
        verr = _validate_artifact_path(p)
        if verr:
            return json.dumps(
                {"error": verr, "error_type": _artifact_error_type(verr)},
                ensure_ascii=False,
            )

    if add_paths:
        store.add_artifacts(task["id"], add_paths)
    if remove_paths:
        store.remove_artifacts(task["id"], remove_paths)
    updated = store.get(task["id"])
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

Finally, after the `registry.register(name="task_comment", ...)` line, add:

```python
registry.register(
    name="task_artifacts", toolset="core",
    schema=TASK_ARTIFACTS_SCHEMA, handler=_handle_task_artifacts, emoji="📎",
)
```

- [ ] **Step 4.6: 跑新测试确认通过**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k "artifacts_handler or validate_artifact"`

Expected: 7 passed

- [ ] **Step 4.7: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `703 + 7 = 710 passed`

- [ ] **Step 4.8: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_enhancements.py
git commit -m "feat(task): task_artifacts 工具 + 严格路径验证

新 helper：_validate_artifact_path（存在/可读/≤100MB）、
_artifact_error_type（区分 invalid_artifact_path / artifact_too_large）。
新工具 task_artifacts：add/remove，原子性批量验证。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: task_complete 扩展 + 现有 handler 重构 + 全测回归

**Files:**
- Modify: `tools/task_tools.py`（`TASK_COMPLETE_SCHEMA` 加 artifacts 参数；`_handle_task_complete` 改造；`_handle_task_update` 重构走 `_get_owned_task`）
- Test: `tests/test_kanban_enhancements.py`（追加 2 个测试）

**Interfaces:**
- Consumes: Task 2 的 `_get_owned_task`；Task 4 的 `_validate_artifact_path` / `_artifact_error_type`；Task 1 的 `TaskStore.add_artifacts`
- Produces: `task_complete` 工具现接受可选 `artifacts: List[str]`；`_handle_task_update` / `_handle_task_complete` 都走统一 helper

- [ ] **Step 5.1: 写失败测试（追加到 test_kanban_enhancements.py 末尾）**

Append:

```python
# ---------------------------------------------------------------------------
# Task 5: task_complete artifacts 扩展
# ---------------------------------------------------------------------------

from tools.task_tools import _handle_task_complete


def test_complete_with_artifacts(store, monkeypatch, tmp_path):
    """complete 时带 artifacts → 完成后 artifacts 已填。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    f = tmp_path / "final.txt"
    f.write_text("done", encoding="utf-8")
    result = _handle_task_complete(
        {"id": task["id"], "artifacts": [str(f)]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["task"]["status"] == "completed"
    assert str(f) in data["task"]["artifacts"]


def test_complete_with_invalid_artifacts(store, monkeypatch):
    """complete 时带无效路径 → 不完成（status 不变），artifacts 不动。"""
    monkeypatch.delenv("HARVIL_KANBAN_TASK", raising=False)
    task = store.create(subject="X")
    store.claim(task["id"], owner="test")  # status → in_progress
    result = _handle_task_complete(
        {"id": task["id"], "artifacts": ["/nonexistent/bad.txt"]},
        harvil_home=str(store._dir.parent),
    )
    data = json.loads(result)
    assert data["error_type"] == "invalid_artifact_path"
    refreshed = store.get(task["id"])
    assert refreshed["status"] == "in_progress"  # 不变
    assert refreshed["artifacts"] == []  # 不动
```

- [ ] **Step 5.2: 跑测试确认失败**

Run: `uv run pytest tests/test_kanban_enhancements.py -v -k "complete_with"`

Expected: 2 FAIL —
- `test_complete_with_artifacts`：现有 handler 忽略 artifacts 参数，task.artifacts 仍为 []，断言 `str(f) in data["task"]["artifacts"]` 失败
- `test_complete_with_invalid_artifacts`：现有 handler 不验证路径，直接 complete 返回 success，断言 `error_type == "invalid_artifact_path"` 失败

- [ ] **Step 5.3: 改 TASK_COMPLETE_SCHEMA 加 artifacts 参数**

Find the existing `TASK_COMPLETE_SCHEMA` (around line 66). Replace it:

```python
TASK_COMPLETE_SCHEMA = {
    "name": "task_complete",
    "description": (
        "标记任务完成。会自动解锁依赖本任务的其他任务。"
        "可选 artifacts：完成时一并加入交付物路径（与 task_artifacts 同款验证）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "任务 ID"},
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选。完成时一并加入的交付物路径",
            },
        },
        "required": ["id"],
    },
}
```

- [ ] **Step 5.4: 改 _handle_task_complete 走 helper + 支持 artifacts**

Find the existing `_handle_task_complete` (around line 162). Replace it with:

```python
def _handle_task_complete(args: dict, **kwargs) -> str:
    """标记完成。可选 artifacts 一次性提交（先验证，原子性）。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    # 新增：先验证 artifacts，再 complete（任一失败 → 整个调用不变）
    artifacts: List[str] = args.get("artifacts") or []
    store = _get_store(kwargs)
    if artifacts:
        for p in artifacts:
            verr = _validate_artifact_path(p)
            if verr:
                return json.dumps(
                    {"error": verr, "error_type": _artifact_error_type(verr)},
                    ensure_ascii=False,
                )
        store.add_artifacts(task["id"], artifacts)

    completed = store.complete(task["id"])
    # 检查解锁了哪些任务（含 id/subject/status，方便 LLM 判断下一步）
    ready = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in store.find_ready()
    ]
    return json.dumps({
        "success": True,
        "task": completed,
        "unblocked": ready,
    }, ensure_ascii=False)
```

- [ ] **Step 5.5: 重构 _handle_task_update 走 helper**

Find the existing `_handle_task_update` (around line 140). Replace it with:

```python
def _handle_task_update(args: dict, **kwargs) -> str:
    """更新 task 字段（status / owner / description / subject）。"""
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    store = _get_store(kwargs)
    fields = {}
    for key in ("status", "owner", "description", "subject"):
        if key in args and args[key] is not None:
            if key == "status" and args[key] not in VALID_STATUSES:
                return json.dumps(
                    {"error": f"非法 status: {args[key]}"}, ensure_ascii=False,
                )
            fields[key] = args[key]

    updated = store.update(task["id"], **fields)
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

- [ ] **Step 5.6: 跑新测试 + 受影响测试确认通过**

Run: `uv run pytest tests/test_kanban_enhancements.py tests/test_team_task_binding.py tests/test_task_system.py -v`

Expected: all passed (含 Task 5 的 2 个新测试 + 现有 task_update/task_complete 测试无回归)

- [ ] **Step 5.7: 跑全测确认无回归**

Run: `uv run pytest tests/ -q`

Expected: `710 + 2 = 712 passed`

- [ ] **Step 5.8: Commit**

```bash
git add tools/task_tools.py tests/test_kanban_enhancements.py
git commit -m "feat(task): task_complete 加 artifacts 参数 + 重构 helper

task_complete 现接受可选 artifacts（与 task_artifacts 同款验证）。
_handle_task_update / _handle_task_complete 重构走 _get_owned_task，
消除 empty-id + ownership 检查重复。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: 全测回归 + push + 更新交接文档

**Files:**
- No code changes — verification only

- [ ] **Step 6.1: 跑完整测试套件**

Run: `uv run pytest tests/ -v --tb=short | tail -20`

Expected: `712 passed`

- [ ] **Step 6.2: 跑复刻检查清单**

Run: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run python scripts/verify.py 2>&1 | tail -5`

Expected: `总计 22：19 通过，3 失败`（3 个失败是 pre-existing MemoryStore 签名问题，与本批无关）

- [ ] **Step 6.3: 确认 git 状态干净**

Run: `git status --short`

Expected: 只剩 `?? .codegraph/`（索引目录，不提交）

- [ ] **Step 6.4: 手动 e2e 验证（可选）**

写一个临时脚本验证完整流程：建任务 → claim → 模拟 worker（env 注入） → heartbeat + comment + artifacts + complete → 校验 task JSON 状态。脚本跑完即删。

Run: `uv run python scripts/manual_verify_kanban.py`

Expected:
```
1. 创建 task: task_xxx
2. worker 绑定: task_xxx
3. heartbeat + note ✓ (last_heartbeat_at set, comment added)
4. comment 独立调用 ✓
5. artifacts 加文件 ✓
6. complete + artifacts ✓
7. 主 agent 无 env: ✓ 不受限
```

- [ ] **Step 6.5: Push 到 origin**

```bash
git push origin master
```

Expected: 5 个新 commit（Task 1-5）+ spec commit + plan commit 都推送成功

- [ ] **Step 6.6: 更新交接文档**

修改 `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`：
- 「当前状态」从 690 改为 712
- 在「第 4 批改进」后加：

```markdown
### 第 5 批改进（690 → 712）

| # | 能力 | 说明 |
|---|---|---|
| ⑪a | Kanban heartbeat/comment/artifacts | TaskStore 加 3 字段；3 新工具 task_heartbeat / task_comment / task_artifacts；task_complete 加 artifacts 参数；artifacts 严格验证 |
```

- 在「剩余待做」的 ⑪ 描述改为：

```markdown
| ⑪b | Kanban block/unblock/link + 自动心跳桥 | 新 status 'blocked' + block_kind + task_block/unblock 工具；task_link post-creation DAG edge（含 cycle 检测）；agent 循环 hook 自动 heartbeat |
```

---

## 完成标准

全部满足才算完成：

- [ ] 5 个新 commit 已 push 到 origin/master
- [ ] `uv run pytest tests/ -q` 显示 712 passed
- [ ] `verify.py` 与起点相同（19 通过 + 3 pre-existing 失败）
- [ ] 手动验证脚本输出正确
- [ ] 交接文档已更新（690 → 712，⑪ 拆为 ⑪a 已完成 + ⑪b 待做）
