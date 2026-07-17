# Kanban Heartbeat / Comment / Artifacts 设计

- **日期**：2026-07-17
- **状态**：用户授权直接推进
- **范围**：第 5 批改进之 ⑪ 第一部分。给 TaskStore 加 heartbeat / comment / artifacts 三类增强
- **依赖**：⑫ Worker 任务归属强制（已完成）；TaskStore / task_tools.py（已就绪）
- **参考**：`D:\project\hermes-agent-main\项目文档\04-多Agent-协调.md` §5.2 / §5.5

---

## 摘要

TaskStore 任务 JSON 加 3 个字段（`last_heartbeat_at` / `comments` / `artifacts`），task_tools.py 加 3 个新工具（`task_heartbeat` / `task_comment` / `task_artifacts`），并扩展现有 `task_complete` 支持 `artifacts` 参数。所有新工具复用 ⑫ 的 ownership 门控（`assert_owned`），artifacts 走严格验证（存在 + 可读 + ≤100MB）。

**不在范围**（留给第 6 批）：block/unblock（新 status 语义）、link（cycle 检测）、自动心跳桥（agent 循环 hook）。

---

## §1 问题与目标

### 问题

当前 task_* 工具集只支持「创建/更新/完成/列出」。Worker 跑长任务（比如 2 小时训练）时无法报告进度，无法留笔记，无法记录交付物文件路径。这导致：

1. **未来 dispatcher watchdog 无法判断 worker 是否还活着**（缺 `last_heartbeat_at`）
2. **Worker 重启时丢失上下文**（无持久化 comment）
3. **任务完成时无法记录"产出了哪些文件"**（gateway notifier 拿不到 artifact 清单）

### 目标

1. Worker 能定期 ping 心跳，可选附 note
2. Worker / 主 agent 能给任务追加持久化留言（跨会话保留）
3. Worker 能维护任务的交付物文件路径列表（add / remove，严格验证）
4. task_complete 一次性提交 artifacts（常见场景：完成时刚好产出文件）
5. 所有新工具受 ⑫ ownership 门控保护

### 非目标

- 不引入新 task status（status 仍为 pending/in_progress/completed/deleted）
- 不做 cycle 检测的 link 工具
- 不做 block/unblock（依赖现有 `blocked_by` + `can_start`）
- 不挂 agent 循环做自动心跳桥（留给下一批）
- 不验证 artifacts 文件内容（仅元数据：存在/可读/大小）

---

## §2 架构

```
LLM 调用工具
    │
    ▼
tools/task_tools.py handler
    │
    ├─ _get_owned_task(args, kwargs)   ← 过 2 道关：工牌 + 取任务
    │   ├─ assert_owned(id)             （⑫ 已有）
    │   └─ store.get(id)
    │
    ▼
agent/task_store.py TaskStore
    │
    ├─ heartbeat(id)                    ← 更新 last_heartbeat_at
    ├─ add_comment(id, author, content) ← 追加到 comments
    ├─ add_artifacts(id, paths)         ← 去重追加到 artifacts
    └─ remove_artifacts(id, paths)      ← 从 artifacts 移除
    │
    ▼
.tasks/{id}.json  ← 加 3 个字段
    ├─ last_heartbeat_at: Optional[ISO string]
    ├─ comments: [{author, content, created_at}, ...]
    └─ artifacts: ["/abs/path/to/file", ...]
```

### 关键不变量

1. **门控一致**：所有新工具都走 `assert_owned`（与 ⑫ 一致）
2. **Artifacts 严格验证**：每个路径必须存在 + 可读 + ≤100MB，否则 `invalid_artifact_path` / `artifact_too_large`
3. **Artifacts 原子性**：批量 add 时任一路径无效 → 整批失败，列表不变
4. **Comment 只追加**：无 delete 操作；想删只能 task_update 改 comments
5. **向后兼容**：旧任务 JSON 没新字段时读出来用默认值（None / []），首次 update 时回填

### Author 推断

```python
def _infer_author(kwargs: dict) -> str:
    """从上下文推断 comment author。"""
    team_name = kwargs.get("team_name")
    if team_name:
        return team_name
    return "main"
```

主 agent 或 CLI 调用（无 team_name 透传）→ `"main"`；spawned worker →  worker name。

---

## §3 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/task_store.py` | ♻️ 改 | TaskStore 加 3 字段（create 初始化）+ 4 方法（heartbeat / add_comment / add_artifacts / remove_artifacts） |
| `tools/task_tools.py` | ♻️ 改 | 加 3 个 schema + 3 个 handler；扩展 TASK_COMPLETE_SCHEMA 加 artifacts 参数；加 helper `_get_owned_task` / `_infer_author` / `_validate_artifact_path` |
| `tests/test_kanban_enhancements.py` | 🆕 新增 | 20 个测试覆盖新行为 |

---

## §4 组件设计

### §4.1 `agent/task_store.py` 扩展

#### 加字段（create 时初始化）

```python
def create(self, subject, description="", blocked_by=None, owner=None) -> dict:
    ...
    task = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "owner": owner,
        "blocked_by": list(blocked_by or []),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        # ↓ 新字段 ↓
        "last_heartbeat_at": None,
        "comments": [],
        "artifacts": [],
    }
    ...
```

#### 加方法

```python
def heartbeat(self, task_id: str) -> Optional[dict]:
    """更新 last_heartbeat_at 为当前时间。"""
    return self.update(task_id, last_heartbeat_at=_now_iso())

def add_comment(self, task_id: str, *, author: str, content: str) -> Optional[dict]:
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
    task["artifacts"] = [p for p in task.get("artifacts", []) if p not in paths]
    task["updated_at"] = _now_iso()
    self._write(task_id, task)
    return task
```

### §4.2 `tools/task_tools.py` 扩展

#### Helper：过两道关（工牌 + 取任务）

```python
def _get_owned_task(args: dict, kwargs: dict):
    """过 ownership + 取任务。失败返回 (None, error_json)；成功返回 (task, None)。"""
    task_id = (args.get("id") or "").strip()
    if not task_id:
        return None, json.dumps({"error": "id 不能为空"}, ensure_ascii=False)
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

#### Helper：附件路径验证

```python
MAX_ARTIFACT_SIZE = 100 * 1024 * 1024  # 100MB

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
```

#### 新工具 schema

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

#### 扩展 TASK_COMPLETE_SCHEMA

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

#### Handler：task_heartbeat

```python
def _handle_task_heartbeat(args: dict, **kwargs) -> str:
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

#### Handler：task_comment

```python
def _handle_task_comment(args: dict, **kwargs) -> str:
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    content = (args.get("content") or "").strip()
    if not content:
        return json.dumps({"error": "content 不能为空"}, ensure_ascii=False)
    author = _infer_author(kwargs)
    store = _get_store(kwargs)
    updated = store.add_comment(task["id"], author=author, content=content)
    return json.dumps({"success": True, "task": updated}, ensure_ascii=False)
```

#### Handler：task_artifacts（含原子性）

```python
def _handle_task_artifacts(args: dict, **kwargs) -> str:
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    add_paths = args.get("add") or []
    remove_paths = args.get("remove") or []
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


def _artifact_error_type(msg: str) -> str:
    """根据验证错误消息选 error_type。"""
    if "过大" in msg:
        return "artifact_too_large"
    return "invalid_artifact_path"
```

#### 扩展 _handle_task_complete

```python
def _handle_task_complete(args: dict, **kwargs) -> str:
    task, err = _get_owned_task(args, kwargs)
    if err:
        return err
    # 新增：先验证 artifacts，再 complete
    artifacts = args.get("artifacts") or []
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
    # 原有 complete 逻辑
    completed = store.complete(task["id"])
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

**重构**：现有 `_handle_task_update` / `_handle_task_complete` 也改成走 `_get_owned_task` helper（去掉重复的 empty-id + ownership 检查）。

#### 注册新工具

```python
registry.register(
    name="task_heartbeat", toolset="core",
    schema=TASK_HEARTBEAT_SCHEMA, handler=_handle_task_heartbeat, emoji="💓",
)
registry.register(
    name="task_comment", toolset="core",
    schema=TASK_COMMENT_SCHEMA, handler=_handle_task_comment, emoji="💬",
)
registry.register(
    name="task_artifacts", toolset="core",
    schema=TASK_ARTIFACTS_SCHEMA, handler=_handle_task_artifacts, emoji="📎",
)
```

---

## §5 错误处理

| 场景 | error_type | 说明 |
|---|---|---|
| 工人跨任务操作 | `permission_denied` | "worker bound to task 'X', cannot operate on 'Y'" |
| 任务不存在 | （无 error_type） | "任务不存在: task_xxx" |
| id 参数为空 | （无 error_type） | "id 不能为空" |
| 附件路径不存在 | `invalid_artifact_path` | "路径不存在: /xxx" |
| 附件路径是目录 | `invalid_artifact_path` | "不是文件（拒绝目录）: /xxx" |
| 附件不可读 | `invalid_artifact_path` | "不可读: /xxx" |
| 附件文件过大 | `artifact_too_large` | "文件过大: N bytes（上限 104857600）" |
| content 参数为空 | （无 error_type） | "content 不能为空" |

所有错误返回 JSON 字符串：`{"success": False, "error": "...", "error_type": "..."}`（项目铁律）。

---

## §6 测试设计

新建 `tests/test_kanban_enhancements.py`，约 20 个测试：

### §6.1 TaskStore 单元测试（直接测方法，不经 handler）

- `test_task_store_heartbeat_sets_timestamp` — heartbeat() 后 last_heartbeat_at 非 None
- `test_task_store_add_comment_appends` — add_comment() 后 comments 长度 +1，结构齐全
- `test_task_store_add_artifacts_dedups` — 同一路径 add 两次，列表只 1 条
- `test_task_store_remove_artifacts` — add 后 remove，列表清空
- `test_task_store_legacy_task_compat` — 手动写一个无新字段的 task JSON，读取/更新不崩

### §6.2 task_heartbeat handler

- `test_heartbeat_updates_task` — 调 handler 后 task.last_heartbeat_at 非 None
- `test_heartbeat_with_note_adds_comment` — 带 note 同时加 comment，author 正确
- `test_heartbeat_blocks_foreign_id` — env 绑 task_A 时 heartbeat(task_B) → permission_denied
- `test_heartbeat_main_agent_unrestricted` — 无 env 时可在任意任务 heartbeat

### §6.3 task_comment handler

- `test_comment_appends` — handler 后 comments 多一条
- `test_comment_empty_content_rejected` — content="" → 错误
- `test_comment_blocks_foreign_id` — 跨任务被拒
- `test_comment_infer_author_worker` — 有 team_name → author=team_name
- `test_comment_infer_author_main` — 无 team_name → author="main"

### §6.4 task_artifacts handler

- `test_artifacts_add_valid_path` — 加 tmp 文件 → 成功，列表含该路径
- `test_artifacts_add_nonexistent` — 加不存在的路径 → invalid_artifact_path，列表不变
- `test_artifacts_add_directory_rejected` — 加目录 → invalid_artifact_path
- `test_artifacts_add_oversized_file` — 加 >100MB（用 monkeypatch 改 MAX 或 mock stat）→ artifact_too_large
- `test_artifacts_atomic_failure` — add [valid, invalid, valid] → 整批失败，列表不变
- `test_artifacts_remove` — add 后 remove，列表清空
- `test_artifacts_blocks_foreign_id` — 跨任务加附件被拒

### §6.5 task_complete 扩展

- `test_complete_with_artifacts` — complete(artifacts=[valid]) → 完成 + artifacts 已填
- `test_complete_with_invalid_artifacts` — complete(artifacts=[invalid]) → 不完成（status 不变），artifacts 不动

### §6.6 现有测试无回归

跑 `uv run pytest tests/test_task_system.py tests/test_team_task_binding.py tests/test_integration.py`，确认现有 690 测试零行为变化。

---

## §7 实现顺序（5 个 task）

| Task | 做什么 | 新增测试数 | 累计测试 |
|---|---|---|---|
| 1 | TaskStore 加 3 字段 + 4 方法 + 直接测试 | ~5 | 695 |
| 2 | task_heartbeat handler + helper `_get_owned_task` + `_infer_author` | ~4 | 699 |
| 3 | task_comment handler | ~3 | 702 |
| 4 | task_artifacts handler + `_validate_artifact_path` + `_artifact_error_type` | ~7 | 709 |
| 5 | task_complete 扩展 + 现有 handler 重构走 `_get_owned_task` + 全测回归 | ~3 | 712 |

每个 task 一个 commit。最后 push origin/master。

---

## §8 未来扩展（本批不做）

- **block/unblock**：新增 status `blocked` + `block_reason` / `block_kind` 字段，配套 task_block / task_unblock 工具
- **link**：post-creation DAG edge 工具，含 cycle 检测
- **自动心跳桥**：在 agent 循环挂 `on_tool_call` 或 POST_TOOL_USE hook，runtime 活动每 60s 自动 heartbeat（不依赖 LLM 显式调用）
- **comment 编辑/删除**：软删除（标 `deleted: true`）
- **artifacts 多文件 batch 上限**：当前只限单文件 100MB，未来可加 batch 总大小限制
