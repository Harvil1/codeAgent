# Kanban Block/Unblock/Link + 自动心跳桥 设计

- **日期**：2026-07-17
- **状态**：用户授权直接推进
- **范围**：⑪b——Kanban 增强剩余部分。新 status `blocked` + `task_block`/`task_unblock`；`task_link` post-creation DAG edge（含 cycle 检测）；agent 循环自动心跳桥
- **依赖**：⑫ Worker 任务归属强制；⑪a heartbeat/comment/artifacts（刚完成）
- **参考**：`D:\project\hermes-agent-main\项目文档\04-多Agent-协调.md` §5.2 KANBAN_BLOCK / §5.5 自动心跳桥

---

## 摘要

给 TaskStore 加 `blocked` status + `block_reason` / `block_kind` 字段；加 3 个新工具 `task_block` / `task_unblock` / `task_link`（含 cycle 检测）；加自动心跳桥模块 `agent/team/auto_heartbeat.py`，挂到 POST_TOOL_USE hook，runtime 活动 每 60s 自动 bump `last_heartbeat_at`，不依赖 LLM 显式调用。

**不在范围**：dispatcher watchdog、claim TTL、Gateway notifier、`task_link` 之外的 DAG 操作（删除边、批量重排）。

---

## §1 问题与目标

### 问题

1. **⑪a 的 heartbeat 字段没人用**：依赖 LLM 显式调 `task_heartbeat` 才更新；现实是 LLM 经常忘调，字段长期为 None。需要 **自动心跳桥**：runtime 活动自动更新。
2. **Worker 卡住没法标"暂停"**：现有 status 只有 pending/in_progress/completed/deleted。worker 跑到一半发现缺凭证/等回复，只能硬着头皮 complete 或挂掉。需要 **task_block** 显式标记"卡住+原因"。
3. **任务依赖只能 create 时通过 `blocked_by` 设**：跑起来后想加新的依赖边（"task_A 跑完发现其实也依赖 task_C"）没办法。需要 **task_link** post-creation 加边。

### 目标

1. 新 status `blocked`，配套 `block_reason` / `block_kind` 字段（kind 仅作人类可读标签，4 种枚举）
2. `task_block` / `task_unblock` 工具，受 ownership 门控
3. `task_link(parent_id, child_id)` 工具：加 DAG 边，含 cycle 检测 + self-link 拒绝
4. `auto_heartbeat` 模块：每进程每 60s 一次自动 heartbeat（仅对 spawned worker 生效，主 agent no-op）
5. 失败静默：auto-heartbeat 失败只 log debug，不影响 agent 主循环

### 非目标

- 不实现 dispatcher watchdog（自动回收 blocked 任务等）
- 不实现 kind 'dependency' 的自动恢复（04 号文档里 dep 完成后任务自动回 ready；HarvilAgent 简化为手动 unblock）
- 不实现重复 block 升级 triage 的逻辑
- 不引入 kind='capability'/'transient'/'needs_input' 的差异化处理（kind 仅 metadata）

---

## §2 架构

### Block/Unblock/Link 数据流

```
LLM 调 task_block(id, reason="...", kind="needs_input")
    │
    ▼
_handle_task_block
    ├─ _get_owned_task(args, kwargs)   ← ownership 门控
    ├─ store.update(id, status="blocked", block_reason=reason, block_kind=kind)
    └─ 返回更新后的 task

LLM 调 task_unblock(id, new_status="pending")
    │
    ▼
_handle_task_unblock
    ├─ _get_owned_task(args, kwargs)
    ├─ 验证 new_status ∈ {"pending", "in_progress"}
    ├─ store.update(id, status=new_status, block_reason=None, block_kind=None)
    └─ 返回更新后的 task

LLM 调 task_link(parent_id, child_id)
    │
    ▼
_handle_task_link
    ├─ 对 parent_id 和 child_id 都过 _get_owned_task（双重门控）
    ├─ 拒绝 self-link：parent_id == child_id → error
    ├─ cycle 检测：DFS from parent_id along blocked_by，若遇 child_id → reject
    ├─ store.add_dependency(child_id, parent_id)  ← 新方法
    └─ 返回更新后的 child task
```

### 自动心跳桥数据流

```
AIAgent.run_conversation 主循环
    │
    ▼  每次工具调用结束
POST_TOOL_USE hook
    │
    ▼
auto_heartbeat.maybe_heartbeat()
    │
    ├─ tid = task_binding.get_bound_task_id()
    │  ├─ None（主 agent / legacy）→ return，no-op
    │  └─ 有值（spawned worker）→ 继续
    │
    ├─ rate-limit check：now - _last_attempt < 60s → return
    │
    ├─ _last_attempt = now
    │
    └─ try: store.heartbeat(tid)
       except Exception: logger.debug(...)  ← 静默失败
```

### 关键不变量

1. **Status 状态机扩展**：
   ```
   pending ⇄ in_progress → completed
       ↓        ↓
     blocked ←──┘
       ↓
   (unblock → pending 或 in_progress)
   ```
   `blocked` 是侧门状态，不影响已有 `can_start` / `find_ready`（它们查 pending，本来就跳过 blocked）。
2. **Cycle 检测必须在写入前**：边加进 `blocked_by` 后再检测就来不及了
3. **task_link 双重门控**：worker 不能用 task_link 给**别人的任务**加依赖（防 prompt 注入恶意把别人任务锁死）
4. **自动心跳桥静默**：任何失败都不能让 agent 主循环崩
5. **Rate-limit 是进程级**：用模块级全局变量 `_last_attempt`，60s 间隔

---

## §3 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/task_store.py` | ♻️ 改 | VALID_STATUSES 加 `"blocked"`；create 加 `block_reason`/`block_kind` 字段；加 `add_dependency(child, parent, validate_fn=None)` 方法（含 cycle 检测）；加 `has_path(start, target)` 工具方法（DFS 沿 blocked_by） |
| `tools/task_tools.py` | ♻️ 改 | 加 3 个 schema + 3 个 handler（task_block/task_unblock/task_link）；TASK_UPDATE_SCHEMA status enum 加 `"blocked"` |
| `agent/team/auto_heartbeat.py` | 🆕 新增 | `maybe_heartbeat()` 函数 + 模块级 `_last_attempt` + `_post_tool_use_hook(tool_name, args, result)` + `register(hooks_registry)` 注册到 POST_TOOL_USE |
| `cli.py` 或 `agent/__init__.py` | ♻️ 改 | 启动时注册 auto_heartbeat hook（具体看现有 hooks 注册在哪） |
| `tests/test_kanban_block_link.py` | 🆕 新增 | block/unblock/link + cycle 检测测试 |
| `tests/test_auto_heartbeat.py` | 🆕 新增 | 自动心跳桥单元测试（rate-limit、no-op、失败静默） |

---

## §4 组件设计

### §4.1 TaskStore 扩展

#### VALID_STATUSES 加 blocked

```python
VALID_STATUSES = {"pending", "in_progress", "completed", "deleted", "blocked"}
```

#### create 初始化新字段

```python
task = {
    # ... existing fields ...
    "last_heartbeat_at": None,
    "comments": [],
    "artifacts": [],
    # ↓ 新增 ↓
    "block_reason": None,
    "block_kind": None,
}
```

#### 新方法：has_path / add_dependency

```python
def has_path(self, start_id: str, target_id: str) -> bool:
    """DFS：从 start_id 沿 blocked_by 边走，能否到达 target_id？

    blocked_by 语义：A.blocked_by=[B] 表示 A 依赖 B（B 完成前 A 不能开始）。
    所以"沿 blocked_by 边走"= "查 start 依赖谁、间接依赖谁"。
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
    """
    child = self.get(child_id)
    if child is None:
        return None
    if parent_id == child_id:
        raise ValueError("self-link forbidden")
    if validate:
        # 加边后会不会成环？等价于：parent 是否已经依赖 child？
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

### §4.2 task_tools.py 加 3 个 handler

#### Schema

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

TASK_LINK_SCHEMA = {
    "name": "task_link",
    "description": (
        "post-creation 加依赖边：让 child 依赖 parent（parent 完成前 child 不能开始）。"
        "含 cycle 检测和 self-link 拒绝。"
        "parent_id 和 child_id 都得过 ownership 门控。"
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

#### Handler

```python
def _handle_task_block(args: dict, **kwargs) -> str:
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


def _handle_task_link(args: dict, **kwargs) -> str:
    parent_id = (args.get("parent_id") or "").strip()
    child_id = (args.get("child_id") or "").strip()
    if not parent_id or not child_id:
        return json.dumps(
            {"error": "parent_id 和 child_id 必需"}, ensure_ascii=False,
        )
    # 双重 ownership 门控
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
        error_type = "cycle_detected" if "cycle" in msg or "self-link" in msg else "invalid_args"
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

#### 注册

```python
registry.register(
    name="task_block", toolset="core",
    schema=TASK_BLOCK_SCHEMA, handler=_handle_task_block, emoji="⏸",
)
registry.register(
    name="task_unblock", toolset="core",
    schema=TASK_UNBLOCK_SCHEMA, handler=_handle_task_unblock, emoji="▶",
)
registry.register(
    name="task_link", toolset="core",
    schema=TASK_LINK_SCHEMA, handler=_handle_task_link, emoji="🔗",
)
```

#### TASK_UPDATE_SCHEMA status enum 加 blocked

```python
"status": {
    "type": "string",
    "enum": ["pending", "in_progress", "completed", "blocked"],
    "description": "新状态",
},
```

### §4.3 `agent/team/auto_heartbeat.py` 新增

```python
"""自动心跳桥：runtime 活动每 60s 自动 bump task.last_heartbeat_at。

防 spawned worker 跑长任务时 dispatcher watchdog 误回收（HarvilAgent 暂时
没有 watchdog，但字段值得维护，未来 dispatcher 接入即可用）。

逻辑：
  POST_TOOL_USE hook 触发 →
    读 HARVIL_KANBAN_TASK env →
      未设（主 agent / legacy）→ no-op
      已设 → rate-limit（60s/进程）→ TaskStore.heartbeat(tid)

所有失败静默（log debug），不能影响 agent 主循环。
"""
import logging
import time
from typing import Optional

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

### §4.4 集成：注册 hook 到 AIAgent

在 `AIAgent.__init__` 末尾加一行注册（`self.hooks_registry` 已在 `__init__` 第 149 行赋值）：

```python
# 自动心跳桥（POST_TOOL_USE hook）
from agent.team.auto_heartbeat import register as _register_auto_heartbeat
_register_auto_heartbeat(self.hooks_registry)
```

**注意**：
- 注册必须在 `self.hooks_registry = hooks_registry`（第 149 行）之后
- `register()` 内部已处理 `hooks_registry=None`（no-op）
- 主 agent 调用 register() 不会出问题（hook 内部 `get_bound_task_id()` 返 None → no-op）

---

## §5 错误处理

| 场景 | error_type | 说明 |
|---|---|---|
| task_block 跨任务 | `permission_denied` | ⑫ ownership 门控 |
| task_block reason 为空 | （无 error_type） | "reason 不能为空" |
| task_block 非法 kind | （无 error_type） | "非法 kind: X" |
| task_unblock 任务非 blocked | （无 error_type） | "任务不是 blocked 状态" |
| task_unblock 非法 new_status | （无 error_type） | "非法 new_status" |
| task_link self-link | `cycle_detected` | "self-link forbidden" |
| task_link 加边成环 | `cycle_detected` | "cycle detected: ..." |
| task_link 跨任务 | `permission_denied` | ownership 门控 |
| auto-heartbeat 失败 | （无返回错误） | 静默 log debug |

---

## §6 测试设计

### `tests/test_kanban_block_link.py`（新建，~12 个测试）

#### TaskStore 层
- `test_has_path_direct_dependency` — A.blocked_by=[B]，has_path(A, B) == True
- `test_has_path_indirect` — A→B→C，has_path(A, C) == True
- `test_has_path_no_path` — 无关任务，has_path(A, Z) == False
- `test_add_dependency_basic` — 加 A→B 边，child.blocked_by 含 parent
- `test_add_dependency_self_link_raises` — add_dependency(X, X) 抛 ValueError
- `test_add_dependency_cycle_rejected` — A→B 已存在，再加 B→A 抛 ValueError
- `test_add_dependency_idempotent` — 加同样的边两次，blocked_by 只一条

#### Handler 层
- `test_block_sets_status_and_reason` — task_block 后 status=blocked, block_reason=reason, block_kind=kind
- `test_block_rejects_empty_reason` — reason="" → error
- `test_block_rejects_invalid_kind` — kind="random" → error
- `test_unblock_resets_to_pending` — task_unblock 默认回 pending，清空 block_reason/kind
- `test_unblock_custom_status` — new_status="in_progress" → status=in_progress
- `test_unblock_non_blocked_rejected` — 对 pending 任务调 unblock → error
- `test_block_unblock_respect_ownership` — 跨任务操作被拒
- `test_link_adds_edge` — task_link 后 child.blocked_by 含 parent
- `test_link_self_rejected` — parent_id == child_id → cycle_detected
- `test_link_cycle_rejected` — 已有 A→B，再 link B→A → cycle_detected
- `test_link_respect_ownership` — parent 或 child 任一跨任务 → permission_denied

### `tests/test_auto_heartbeat.py`（新建，~5 个测试）

- `test_maybe_heartbeat_noop_without_env` — env 未设 → return False，不调 store
- `test_maybe_heartbeat_writes_with_env` — env 设了 + task 存在 → 写入 last_heartbeat_at
- `test_maybe_heartbeat_rate_limit` — 连续调两次 < 60s → 第二次 no-op
- `test_maybe_heartbeat_reset_for_test` — reset_for_test() 后立即再调能写入
- `test_maybe_heartbeat_silent_failure` — store.heartbeat 抛异常 → maybe_heartbeat 返 False 不上抛

**新增测试约 23 个**（TaskStore 7 + block/unblock handler 7 + link handler 4 + auto-heartbeat 5）。预期 712 + 23 = **735 测试通过**。

---

## §7 实现顺序（4 个 task）

| Task | 做什么 | 新增测试数 | 累计 |
|---|---|---|---|
| 1 | TaskStore 加 blocked status + 字段 + has_path + add_dependency | ~7 | 719 |
| 2 | task_block + task_unblock handler | ~7 | 726 |
| 3 | task_link handler（双重门控 + cycle） | ~4 | 730 |
| 4 | auto_heartbeat 模块 + 注册到 POST_TOOL_USE hook + 全测回归 push | ~5 | 735 |

每个 task 一个 commit。最后 push origin/master。

---

## §8 风险与权衡

1. **task_link 双重 ownership 门控**：worker 既不能给别人加依赖，也不能让别人给自己加依赖。代价：主 agent 想给两个 worker 任务加依赖时，要确认自己没被 env 绑定（spawned worker 才有 env，主 agent 没有，所以问题不大）。
2. **auto_heartbeat hook 集成点**：如果 AIAgent 没有现成的 hooks_manager，需要新建。本批假设 Phase 2a hooks 系统已就绪（已完成）。若集成复杂度超预期，Task 4 可能需要拆。
3. **Cycle 检测性能**：`has_path` 是 O(V+E) DFS，DAG 通常很小（几十个任务），不优化。
4. **`blocked` status 对现有逻辑影响**：`can_start` / `find_ready` 查 pending，自然跳过 blocked，无需改。`find_blocked` 现在返回"依赖未满足的 pending"，与新 blocked 语义不冲突（blocked 是另一回事）。

---

## §9 未来扩展

- dispatcher watchdog：扫描 last_heartbeat_at 超过 X 分钟的 in_progress 任务，自动 block
- block kind='dependency' 自动恢复：dep 完成后 status 自动回 pending
- 删 dependency 边工具 task_unlink
- batch link 工具
