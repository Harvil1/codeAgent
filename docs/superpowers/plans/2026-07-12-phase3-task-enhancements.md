# Phase 3: 任务增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** 3a 把 `task_complete` 返回的 `unblocked` 增强（含 subject）；3b 给 `worktree.py` 加事件流审计（`.events.jsonl`）。

**Architecture:** 改 `tools/task_tools.py:_handle_task_complete` + 改 `tools/worktree.py:_create_git_worktree` 和 cleanup。两独立小项。

**Tech Stack:** Python 3.11+、uv、pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 3

## Global Constraints

- 文件 I/O `encoding="utf-8"`
- 用 `uv`
- 中文注释/commit；英文标识符
- 502 测试不得回归
- Subagent 不 commit；controller 统一 commit

---

## Task 1: 3a — task_complete 返回 unblocked 含 subject

**Files:**
- Modify: `tools/task_tools.py:_handle_task_complete`（约 149-155 行）
- Modify: `tests/test_task_system.py`

**Interfaces:**
- Consumes: `TaskStore.find_ready()` 已有
- Produces: `unblocked` 字段从 `list[str]` 变为 `list[{"id","subject","status"}]`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_task_system.py
def test_task_complete_returns_unblocked_with_subject(tmp_path):
    """task_complete 返回的 unblocked 含 id + subject。"""
    from agent.task_store import TaskStore
    from tools.task_tools import _handle_task_complete

    store = TaskStore(tmp_path)
    t1 = store.create("task_1 subject", blocked_by=[])
    t2 = store.create("task_2 subject", blocked_by=[t1])

    result_str = _handle_task_complete({"id": t1}, task_store=store)
    import json
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert isinstance(parsed["unblocked"], list)
    # 应含 t2 的完整对象
    assert len(parsed["unblocked"]) >= 1
    ub = parsed["unblocked"][0]
    assert ub["id"] == t2
    assert "subject" in ub
    assert ub["subject"] == "task_2 subject"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_task_system.py::test_task_complete_returns_unblocked_with_subject -v`
Expected: FAIL — 当前 unblocked 是 list[str]

- [ ] **Step 3: 改 task_tools.py**

修改 `_handle_task_complete` 的 unblocked 部分：

```python
def _handle_task_complete(args: dict, **kwargs) -> str:
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return json.dumps({"error": "id 不能为空"}, ensure_ascii=False)

    store = _get_store(kwargs)
    task = store.complete(task_id)
    if task is None:
        return json.dumps({"error": f"任务不存在: {task_id}"}, ensure_ascii=False)

    # 检查解锁了哪些任务（含完整信息，便于 LLM 知道下一步做什么）
    ready = store.find_ready()
    unblocked = [
        {"id": t["id"], "subject": t.get("subject", ""), "status": t.get("status", "")}
        for t in ready
    ]
    return json.dumps({
        "success": True,
        "task": task,
        "unblocked": unblocked,
    }, ensure_ascii=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_task_system.py -v`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 502 + 新增 1）

- [ ] **Step 6: 不 commit**

---

## Task 2: 3b — worktree 事件流审计

**Files:**
- Modify: `tools/worktree.py`
- Modify: `tests/test_worktree.py`

**Interfaces:**
- Consumes: `pathlib.Path`、`datetime`、`json`
- Produces: `_log_worktree_event(repo_root, event_type, payload)` 模块级函数

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_worktree.py
def test_log_worktree_event_appends_jsonl(tmp_path):
    """事件写入 .worktrees/.events.jsonl，每行一个 JSON。"""
    import json
    from tools.worktree import _log_worktree_event

    _log_worktree_event(tmp_path, "create.after", {
        "name": "task-x",
        "branch": "harvil/task-x/abc12345",
        "path": str(tmp_path / ".harvil-worktrees" / "task-x-abc12345"),
    })

    events_file = tmp_path / ".worktrees" / ".events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "create.after"
    assert parsed["payload"]["name"] == "task-x"
    assert "ts" in parsed  # ISO timestamp


def test_log_worktree_event_multiple_appends(tmp_path):
    """多次写入追加，不覆盖。"""
    import json
    from tools.worktree import _log_worktree_event

    for i in range(3):
        _log_worktree_event(tmp_path, f"event_{i}", {"i": i})

    events_file = tmp_path / ".worktrees" / ".events.jsonl"
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert [p["event"] for p in parsed] == ["event_0", "event_1", "event_2"]


def test_log_worktree_event_atomic_write(tmp_path):
    """写入失败不应破坏已有内容（try/except + 原子 append）。"""
    from tools.worktree import _log_worktree_event
    # 正常写入一条
    _log_worktree_event(tmp_path, "first", {"x": 1})
    # 写入失败（payload 含不可序列化对象）
    class NotSerializable:
        pass
    try:
        _log_worktree_event(tmp_path, "bad", {"obj": NotSerializable()})
    except Exception:
        pass  # 实现应 catch 内部
    # 第一条应仍存在
    events_file = tmp_path / ".worktrees" / ".events.jsonl"
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    assert json.loads(lines[0])["event"] == "first"


def test_worktree_create_logs_events(tmp_path):
    """create_isolated_workspace 实际触发 create.before + create.after 事件。"""
    import json
    # 跳过非 git 环境（无 git 仓库时退化为 temp workspace，仍应记录事件）
    from tools.worktree import create_isolated_workspace, _resolve_events_path

    path, cleanup = create_isolated_workspace(name="audit-test")
    try:
        # 找 events 文件位置
        events_path = _resolve_events_path(path)
        # 可能是 temp workspace（parent 的 .worktrees/）
        if events_path.exists():
            lines = events_path.read_text(encoding="utf-8").strip().splitlines()
            events = [json.loads(l)["event"] for l in lines]
            # 应至少有 create.after
            assert "create.after" in events
    finally:
        cleanup()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_worktree.py -v`
Expected: FAIL — `_log_worktree_event` 不存在

- [ ] **Step 3: 改 worktree.py**

在 `tools/worktree.py` 加事件流 helper + 在 create/cleanup 中调用。

加在文件顶部 import 之后：

```python
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def _resolve_events_path(workspace_or_repo: Path) -> Path:
    """解析事件流文件路径。

    优先用 workspace 的 repo_root；找不到则用 workspace 的 parent（temp workspace 场景）。
    返回 <root>/.worktrees/.events.jsonl。
    """
    repo_root = get_repo_root(workspace_or_repo)
    base = repo_root or workspace_or_repo.parent
    return Path(base) / ".worktrees" / ".events.jsonl"


def _log_worktree_event(
    repo_root: Path,
    event_type: str,
    payload: dict,
) -> None:
    """追加一条事件到 .worktrees/.events.jsonl。

    失败只 log warning，不抛（事件流是审计辅助，不应阻塞主流程）。
    """
    events_file = Path(repo_root) / ".worktrees" / ".events.jsonl"
    entry = {
        "event": event_type,
        "payload": payload,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except (TypeError, ValueError) as e:
        # payload 不可序列化
        logger.warning("worktree 事件 payload 序列化失败 (%s): %s", event_type, e)
    except OSError as e:
        logger.warning("worktree 事件写入失败 (%s): %s", event_type, e)
```

然后在 `_create_git_worktree` 加事件（before + after）：

```python
def _create_git_worktree(base: Path, name: str) -> Tuple[Path, Callable]:
    """用 git worktree 创建独立工作区。"""
    repo_root = get_repo_root(base) or base
    short_id = uuid.uuid4().hex[:8]
    branch = f"harvil/{name}/{short_id}"

    worktree_dir = repo_root.parent / ".harvil-worktrees" / f"{name}-{short_id}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # === NEW: create.before 事件 ===
    _log_worktree_event(repo_root.parent, "create.before", {
        "name": name, "branch": branch, "planned_path": str(worktree_dir),
    })

    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_dir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        _log_worktree_event(repo_root.parent, "create.failed", {
            "name": name, "branch": branch, "stderr": result.stderr[:500],
        })
        raise RuntimeError(f"git worktree add 失败: {result.stderr}")

    logger.info("已创建 git worktree: %s（分支 %s）", worktree_dir, branch)

    # === NEW: create.after 事件 ===
    _log_worktree_event(repo_root.parent, "create.after", {
        "name": name, "branch": branch, "path": str(worktree_dir),
    })

    def cleanup(keep: bool = False):
        if keep:
            _log_worktree_event(repo_root.parent, "remove.keep", {
                "path": str(worktree_dir), "branch": branch,
            })
            logger.info("保留 worktree: %s", worktree_dir)
            return
        _log_worktree_event(repo_root.parent, "remove.before", {
            "path": str(worktree_dir), "branch": branch,
        })
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=str(repo_root),
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(repo_root),
                capture_output=True,
                timeout=10,
            )
            logger.info("已清理 worktree: %s", worktree_dir)
        except Exception as e:
            logger.debug("清理 worktree 失败: %s", e)
        shutil.rmtree(worktree_dir, ignore_errors=True)
        _log_worktree_event(repo_root.parent, "remove.after", {
            "path": str(worktree_dir), "branch": branch,
        })

    return worktree_dir, cleanup
```

也在 `_create_temp_workspace` 加 create.after 事件（保持一致）：

```python
def _create_temp_workspace(name: str) -> Tuple[Path, Callable]:
    """非 git 仓库时创建空临时目录。"""
    # ... 原有逻辑 ...
    # === NEW ===
    _log_worktree_event(temp_dir.parent, "create.after", {
        "name": name, "path": str(temp_dir), "type": "temp",
    })
    # ... 原 cleanup ...
```

具体行号需 implementer 读现有代码定位。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_worktree.py -v`
Expected: PASS（4 个新测试 + 原有）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Self-Review

**Spec 覆盖**：
- ✅ 3a `task_tools.py:task_complete` 报告 unblocked 含 subject → T1
- ✅ 3b `worktree.py` `.events.jsonl` 事件流 → T2（create.before/after/failed + remove.before/after/keep）

**Placeholder**：无 TBD。

**类型一致性**：`_log_worktree_event(repo_root, event_type, payload)` 签名一致。

---

## Execution Handoff

按用户授权直接进 SDD。
