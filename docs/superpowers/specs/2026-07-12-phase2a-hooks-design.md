# Phase 2a: Hooks 系统设计

- **作者**：Claude（经 brainstorming 流程产出）
- **日期**：2026-07-12
- **状态**：用户授权直接推进，跳过逐节确认
- **驱动**：生产场景，4 类用途全要（审计/日志、策略拦截、上下文增强、工作流自动化）
- **兼容性策略**：允许破坏性变更，但提供配置开关
- **范围**：仅 Phase 2a（Hooks 系统）。2b 后台任务 / 2c Cron 留给后续 Phase 2.2 / 2.3
- **对应 Spec**：`docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §6 Phase 2a 展开

---

## 摘要

新增 Hooks 系统，把"扩展 agent 主循环行为"从「改 `agent/__init__.py` 代码」变成「写回调函数 + 注册」。支持 4 种 event：UserPromptSubmit / PreToolUse / PostToolUse / Stop。同时支持程序式（Python API）和声明式（`~/.agent/.hooks/settings.json` + 子进程 JSON IPC）两种注册方式。所有 hook 失败 fail-open 默认（log + 跳过），PreToolUse 可选 fail_closed。

---

## §1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│ agent/__init__.py:run_conversation（主循环，4 个注入点）           │
│                                                                  │
│  while budget:                                                   │
│      ┌─── ① USER_PROMPT_SUBMIT（user_msg 入 history 前）          │
│      │        ↓ 返回可能修改过的 prompt                           │
│      │    messages.append(...)                                   │
│      │                                                           │
│      │    ② 组装 system_prompt + messages                         │
│      │    ③ 调 LLM                                                │
│      │    ④ 工具分发（model_tools.handle_function_call）          │
│      │        ┌─── ③ PRE_TOOL_USE（handler 调用前）                │
│      │        │     ↓ 返回 deny → 跳过 handler，返拒绝 JSON 给 LLM │
│      │        │   handler(...)                                   │
│      │        └─── ④ POST_TOOL_USE（handler 返回后）               │
│      │              ↓ 返回可能修改过的 result                     │
│      │                                                           │
│      └─── ⑤ STOP（stop_reason != tool_use，循环退出前）           │
│             ↓ 返回 force_continue msg → 不退出，作为新 user msg   │
└─────────────────────────────────────────────────────────────────┘
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/hooks.py` | 🆕 新增 | `HookEvent` 枚举 + `HookRegistry` + `Hook` / `HookScriptConfig` dataclass |
| `agent/hook_exec.py` | 🆕 新增 | 声明式 hook 的子进程执行 + JSON IPC |
| `agent/hook_loader.py` | 🆕 新增 | 从 `~/.agent/.hooks/settings.json` 加载声明式 hook |
| `agent/__init__.py` | ♻️ 改 | `run_conversation` 加 USER_PROMPT_SUBMIT + STOP 调用点；构造时加载 hooks |
| `model_tools.py` | ♻️ 改 | `handle_function_call` 加 PRE/POST_TOOL_USE 调用点 |
| `cli.py` | ♻️ 改 | RuntimeContext 持有 hooks_registry；启动时加载声明式 hooks |
| `config.py` | ♻️ 改 | 新增 `hooks` 块 |
| `tests/test_hooks.py` | 🆕 新增 | 4 种 event 程序式测试 |
| `tests/test_hook_exec.py` | 🆕 新增 | 子进程执行 + JSON IPC + 超时测试 |
| `tests/test_hook_loader.py` | 🆕 新增 | settings.json 解析 + 校验测试 |
| `tests/test_integration.py` | ♻️ 改 | 主循环集成测试 |

### 关键设计决策

1. **同步执行**：所有 hook 在主循环线程内同步跑（不引入 asyncio）。声明式 hook 的子进程有超时。
2. **失败隔离 fail-open 默认**：任何 hook 故障（异常/超时/坏 JSON/非零退出）只 log，不阻塞主循环。PreToolUse 声明式 hook 可配 `fail_closed: true`（故障 = 拒绝操作）。
3. **注册顺序明确**：同 event 内程序式先于声明式；同 kind 内按注册/数组顺序。
4. **PreToolUse deny 仍要返回 tool 消息**：保 tool_call 配对。返回 `{"error": "hook denied: <reason>", "error_type": "hook_deny"}`。
5. **Stop hook 防失控**：用 `_stop_fire_count` 每会话最多触发 `max_fires` 次（默认 3），避免循环永不退出。
6. **HookRegistry 由 RuntimeContext 持有**：注入 AIAgent，不用模块单例。

---

## §2 四种 Event 的协议

### 通用：payload 包络

所有 hook（程序式 + 声明式）都收到事件特定字段 + 公共元数据：

```json
{
  "event": "pre_tool_use",
  "session_id": "sess_xxx",
  "timestamp": "2026-07-12T15:30:22",
  "hook_name": "weekend-push-blocker",
  "...event-specific fields..."
}
```

### Event ① USER_PROMPT_SUBMIT

| 项 | 值 |
|---|---|
| 触发点 | `run_conversation` 顶部，`messages.append({"role":"user"...})` 之前 |
| Python 签名 | `Callable[[str], Optional[str]]`（prompt → 新 prompt 或 None） |
| IPC stdin | `{"event":"user_prompt_submit", "prompt": "...", ...}` |
| IPC stdout | `{}` 或 `{"action":"allow"}` → 不变；`{"prompt": "修改后"}` → 替换 |
| 组合 | **链式**：每个 hook 看到前一个的输出 |
| Python 返回 | `None` → 不变；`str` → 替换 prompt |

### Event ② PRE_TOOL_USE

| 项 | 值 |
|---|---|
| 触发点 | `handle_function_call` 内，`registry.dispatch(...)` 之前 |
| Python 签名 | `Callable[[str, dict], Optional[dict]]`（`tool_name, args` → denial/modify/None） |
| IPC stdin | `{"event":"pre_tool_use", "tool_name":"terminal", "args":{...}, ...}` |
| IPC stdout | `{}` 或 `{"action":"allow"}` → 放行；`{"action":"deny", "reason":"周末禁止 push"}` → 拒绝；`{"action":"modify", "args":{...}}` → 改 args |
| 组合 | **短路**：首个 `deny` 胜出。否则 `modify` 链式累积 |
| Python 返回 | `None` → 放行；`{"deny": "reason"}` → 拒绝；`{"modify_args": {...}}` → 替换 args |

**关键**：拒绝时 handler 不调，但仍返回 JSON 结果（保 tool_call 配对）：
```json
{"error": "hook denied: 周末禁止 push", "error_type": "hook_deny"}
```

### Event ③ POST_TOOL_USE

| 项 | 值 |
|---|---|
| 触发点 | `handle_function_call` 内，`registry.dispatch(...)` 返回后 |
| Python 签名 | `Callable[[str, dict, str], Optional[str]]`（`tool_name, args, result` → 新 result 或 None） |
| IPC stdin | `{"event":"post_tool_use", "tool_name":"...", "args":{...}, "result":"...", ...}` |
| IPC stdout | `{}` → 不变；`{"result": "修改后"}` → 替换 |
| 组合 | **链式** |
| Python 返回 | `None` → 不变；`str` → 替换 result |

### Event ④ STOP

| 项 | 值 |
|---|---|
| 触发点 | `run_conversation` 末尾，`stop_reason != "tool_use"` 准备 return 时 |
| Python 签名 | `Callable[[], Optional[str]]`（→ force_continue msg 或 None） |
| IPC stdin | `{"event":"stop", ...}` |
| IPC stdout | `{}` → 允许停止；`{"continue": "还有 task_2 没做完"}` → 强制续跑 |
| 组合 | **首个非 None 胜出**（后续不跑） |
| Python 返回 | `None` → 允许停止；`str` → 作为新 user 消息加入 history，循环继续 |

**防失控**：`_stop_fire_count` 每会话上限 `max_fires`（默认 3）。超限后忽略后续 Stop hook，强制退出。

### 统一错误处理（fail-open 默认）

| 故障 | 行为 |
|---|---|
| Hook 抛异常 | log warning + 视为 None（不变） |
| 子进程超时（>timeout） | kill 进程，log + 视为 None |
| stdout 非合法 JSON | log + 视为 None |
| stdout 缺字段 | log + 视为 None |
| 子进程 exit code != 0 | log + 视为 None |

**例外**：声明式 hook 配置 `"fail_closed": true`（仅 PreToolUse 有意义）—— 任何故障 = 拒绝操作。用于合规场景。

---

## §3 类设计与配置

### 3.1 核心类型（`agent/hooks.py`）

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any

class HookEvent(Enum):
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"

UserPromptSubmitFn = Callable[[str], Optional[str]]
PreToolUseFn = Callable[[str, dict], Optional[dict]]
PostToolUseFn = Callable[[str, dict, str], Optional[str]]
StopFn = Callable[[], Optional[str]]

@dataclass
class HookScriptConfig:
    """声明式 hook 的子进程配置。"""
    command: list[str]              # ["python", "./hooks/audit.py"]
    timeout: float = 10.0
    env: Optional[dict] = None

@dataclass
class Hook:
    """统一包装：程序式或声明式。"""
    name: str
    event: HookEvent
    kind: str                       # "programmatic" | "declarative"
    fn: Optional[Callable] = None
    script: Optional[HookScriptConfig] = None
    fail_closed: bool = False
```

### 3.2 HookRegistry

```python
class HookRegistry:
    def __init__(self):
        self._hooks: dict[HookEvent, list[Hook]] = {e: [] for e in HookEvent}
        self._stop_fire_count: int = 0

    # 注册（按 event 类型分，类型安全）
    def register_user_prompt_submit(self, fn, *, name=None): ...
    def register_pre_tool_use(self, fn, *, name=None, fail_closed=False): ...
    def register_post_tool_use(self, fn, *, name=None): ...
    def register_stop(self, fn, *, name=None): ...
    def register_declarative(self, hook: Hook): ...
    def clear(self, event: Optional[HookEvent] = None): ...

    # 执行（按 event 类型分，返回类型不同）
    def run_user_prompt_submit(self, prompt: str, *, session_id: str) -> str: ...
    def run_pre_tool_use(self, tool_name: str, args: dict, *,
                          session_id: str) -> tuple[Optional[str], Optional[dict]]: ...
    def run_post_tool_use(self, tool_name: str, args: dict, result: str,
                           *, session_id: str) -> str: ...
    def run_stop(self, *, session_id: str, max_fires: int = 3) -> Optional[str]: ...
```

**关键**：
- 注册方法按 event 类型分 → Python 类型检查能对上签名
- 执行方法也按 event 类型分 → 调用方拿到正确类型返回值
- 内部辅助 `_invoke_programmatic(hook, *args)` 和 `_invoke_declarative(hook, payload)` 统一处理失败隔离

### 3.3 子进程执行（`agent/hook_exec.py`）

```python
def run_script_hook(hook: Hook, payload: dict) -> Optional[dict]:
    """在子进程中执行声明式 hook。

    - payload JSON 写到 stdin
    - stdout 期望是合法 JSON
    - 超时 kill
    - 返回解析后的 dict 或 None（任何故障）
    """
    ...
```

### 3.4 加载声明式配置（`agent/hook_loader.py`）

```python
def load_declarative_hooks(
    registry: HookRegistry,
    settings_path: Path,
) -> int:
    """从 settings.json 加载声明式 hooks 到 registry。

    返回加载数量。文件不存在 = 静默返回 0。
    """
    ...
```

### 3.5 settings.json 格式

路径默认 `~/.agent/.hooks/settings.json`：

```json
{
  "hooks": {
    "user_prompt_submit": [
      {
        "name": "inject-project-context",
        "command": ["python", "~/.agent/.hooks/inject_project.py"],
        "timeout": 5.0
      }
    ],
    "pre_tool_use": [
      {
        "name": "weekend-push-blocker",
        "command": ["~/.agent/.hooks/block_weekend_push.sh"],
        "timeout": 3.0,
        "fail_closed": true
      }
    ],
    "post_tool_use": [
      {
        "name": "audit-logger",
        "command": ["~/.agent/.hooks/audit.sh"],
        "timeout": 10.0
      }
    ],
    "stop": []
  }
}
```

**校验规则**：
- 顶层必须有 `hooks` 字段（dict），否则 fail-fast
- 每个 event 名必须是 4 个合法值之一，否则 fail-fast
- 每个 hook 必须有 `name` 和 `command`，否则跳过该 hook + log warning
- `command` 必须是非空 list

### 3.6 config.py 新增块

```python
"hooks": {
    "enabled": True,                            # 全局开关；False 时跳过所有 hook
    "settings_path": None,                      # None → 默认 ~/.agent/.hooks/settings.json
    "script_timeout_default": 10.0,
    "stop_hook_max_fires": 3,
    "fail_closed_default": False,
},
```

### 3.7 注入到 RuntimeContext

`cli.py:RuntimeContext` 持有 `hooks_registry: HookRegistry` 实例。启动时：
1. 实例化空 registry
2. 若 `config.hooks.enabled`：调用 `hook_loader.load_declarative_hooks(registry, settings_path)` 加载用户配置
3. 把 registry 传给 `AIAgent(hooks_registry=...)`

`AIAgent.__init__` 接受 `hooks_registry=None`（None 时所有 hook 调用跳过，向后兼容）。

---

## §4 主循环注入点（具体代码位置）

### 4.1 USER_PROMPT_SUBMIT 注入

**位置**：`agent/__init__.py:run_conversation` 最开始（在 `self.conversation_history.append({"role":"user"...})` 之前）。

**改造**：
```python
def run_conversation(self, user_message: str) -> str:
    # === NEW: USER_PROMPT_SUBMIT hook ===
    if self.hooks_registry and self.config.get("hooks", {}).get("enabled", True):
        try:
            user_message = self.hooks_registry.run_user_prompt_submit(
                user_message, session_id=self.session_id or "",
            )
        except Exception as e:
            logger.warning("USER_PROMPT_SUBMIT 编排异常（不该发生但已兜底）: %s", e)

    # 原有逻辑：append 到 history
    self.conversation_history.append({"role": "user", "content": user_message})
    # ... 原有循环
```

### 4.2 PRE_TOOL_USE + POST_TOOL_USE 注入

**位置**：`model_tools.py:handle_function_call`，包裹 `registry.dispatch(...)`。

**改造**：
```python
def handle_function_call(
    function_name, function_args, *,
    hooks_registry=None,  # === NEW ===
    session_id=None,
    config=None,
    **other_kwargs,
):
    ensure_tools_discovered()
    function_args = _coerce_tool_args(function_name, function_args)

    # === NEW: PRE_TOOL_USE ===
    if hooks_registry and (config or {}).get("hooks", {}).get("enabled", True):
        deny_reason, modified_args = hooks_registry.run_pre_tool_use(
            function_name, function_args, session_id=session_id or "",
        )
        if deny_reason is not None:
            return json.dumps({
                "error": f"hook denied: {deny_reason}",
                "error_type": "hook_deny",
            }, ensure_ascii=False)
        if modified_args is not None:
            function_args = modified_args

    # 原有 dispatch
    result = registry.dispatch(function_name, function_args, **other_kwargs)

    # === NEW: POST_TOOL_USE ===
    if hooks_registry and (config or {}).get("hooks", {}).get("enabled", True):
        result = hooks_registry.run_post_tool_use(
            function_name, function_args, result, session_id=session_id or "",
        )

    return result
```

**caller 改造**：`agent/__init__.py` 调 `handle_function_call` 时多传 `hooks_registry=self.hooks_registry`。

### 4.3 STOP 注入

**位置**：`agent/__init__.py:run_conversation` 末尾，`return final_response` 之前。

**改造**：
```python
# 原有：stop_reason != "tool_use" → 准备 return
final_response = response.choices[0].message.content or ""

# === NEW: STOP hook ===
if (self.hooks_registry
        and self.config.get("hooks", {}).get("enabled", True)
        and self._stop_fire_count < self.config.get("hooks", {}).get(
            "stop_hook_max_fires", 3)):
    try:
        force_msg = self.hooks_registry.run_stop(
            session_id=self.session_id or "",
            max_fires=self.config.get("hooks", {}).get("stop_hook_max_fires", 3),
        )
    except Exception as e:
        logger.warning("STOP hook 编排异常: %s", e)
        force_msg = None

    if force_msg:
        self._stop_fire_count += 1
        # 把 force_msg 作为新 user 消息，继续循环
        self.conversation_history.append({
            "role": "user",
            "content": f"[stop_hook]: {force_msg}",
        })
        continue  # 跳回 while 顶部

return final_response
```

**关键**：`_stop_fire_count` 在 `AIAgent.__init__` 初始化为 0。

### 4.4 hooks_registry 注入 AIAgent

```python
# agent/__init__.py:AIAgent.__init__
def __init__(self, ..., hooks_registry=None, ...):
    # ...
    self.hooks_registry = hooks_registry  # None 时所有 hook 调用跳过
    self._stop_fire_count = 0
```

---

## §5 兼容性 / 迁移 / 测试

### 5.1 破坏性变更

| # | 变更 | 影响面 | 严重度 |
|---|---|---|---|
| ① | `agent/__init__.py:AIAgent.__init__` 新增 `hooks_registry=None` kwarg | 纯加法，默认 None | 低 |
| ② | `model_tools.py:handle_function_call` 新增 `hooks_registry=None` kwarg | 纯加法 | 低 |
| ③ | `cli.py:RuntimeContext` 新增 `hooks_registry` 属性 | 纯加法 | 低 |
| ④ | `config.py:DEFAULT_CONFIG` 新增 `hooks` 块 | 老 config.yaml 深合并，仍跑 | 低 |
| ⑤ | 新建 `~/.agent/.hooks/` 目录（懒创建） | 首次启动时 | 低 |

**所有变更都是「加法」**，无外部 API 改变。`hooks_registry=None` 时主循环行为完全等同于 Phase 1。

### 5.2 上线顺序（5 个有序 commit）

```
Commit 1  [新增]  agent/hooks.py + agent/hook_exec.py + agent/hook_loader.py + tests
                ↑ 纯加法，未挂到主循环

Commit 2  [配置]  config.py 新增 hooks 块
                ↑ 默认 enabled=True，但 registry 为空时无效果

Commit 3  [挂线]  agent/__init__.py + model_tools.py 加 hook 调用点 + cli.py 注入
                ↑ hooks_registry=None 时所有调用跳过（向后兼容）

Commit 4  [集成]  tests/test_integration.py 加端到端用例
                ↑ 程序式 hook 注册 + 4 种 event 触发验证

Commit 5  [可选] 创建示例 ~/.agent/.hooks/settings.json 模板
                ↑ 方便用户参考
```

**无双轨期**：Hooks 是纯加法，不需要 `use_new_hooks` 开关。全局开关 `config.hooks.enabled=False` 即可完全禁用。

### 5.3 错误处理矩阵

| 故障 | 行为 | 测试 |
|---|---|---|
| 程序式 hook 抛异常 | log warning，视为 None | ✓ |
| 声明式 hook 子进程超时 | kill 进程，log，视为 None | ✓ |
| 声明式 hook stdout 非合法 JSON | log，视为 None | ✓ |
| 声明式 hook exit code != 0 | log，视为 None | ✓ |
| 声明式 hook fail_closed=True 故障 | PreToolUse → 拒绝操作 | ✓ |
| Stop hook 超过 max_fires | 强制退出循环 | ✓ |
| settings.json 不存在 | 静默跳过加载 | ✓ |
| settings.json 格式错 | 启动时 fail-fast | ✓ |

### 5.4 测试矩阵

**新增测试文件**：
- `tests/test_hooks.py` —— HookRegistry 4 种 event 程序式 + 组合语义 + 失败隔离 + stop_fire_count
- `tests/test_hook_exec.py` —— 子进程执行 + JSON IPC + 超时 + 各种 stdout 形态
- `tests/test_hook_loader.py` —— settings.json 解析 + 校验 + 各种 malformed

**集成测试**：
- `tests/test_integration.py` 加用例：
  - 程序式 PreToolUse deny → tool 返回 hook_deny error
  - 程序式 UserPromptSubmit 修改 prompt → 实际入 history 的是新 prompt
  - 程序式 Stop force_continue → 循环不退出
  - hooks_registry=None 时主循环不受影响（回归）

**验证脚本**：
- `scripts/verify.py` 加一项 hooks 检查（注册一个程序式 hook，确认能被调用）

### 5.5 可观测性

```python
logger.info("hook %s executed (event=%s, kind=%s, action=%s)",
            hook.name, hook.event.value, hook.kind, action)
logger.warning("hook %s failed: %s", hook.name, error)
logger.warning("hook %s timed out after %ss", hook.name, timeout)
```

---

## §6 已知限制 / 非目标

1. **不实现 2b 后台任务 / 2c Cron**：本 spec 只做 Hooks 底座。后台任务和 Cron 留给 Phase 2.2 / 2.3。
2. **不引入 asyncio**：Hooks 同步执行。声明式 hook 阻塞主循环直到完成或超时。
3. **不实现 hook 链中断统计**：不记录"hook N 中断了多少次"。生产观察后若需要再加。
4. **不实现 hook 优先级**：注册顺序即执行顺序。需要优先级的场景用户新增注册时自行控制。
5. **声明式 hook 不支持 streaming stdout**：子进程必须一次性输出合法 JSON 后退出。需要 streaming 的场景等 Phase 2.x。
6. **Stop hook force_continue 消息直接进 history**：可能违反角色交替（如果上一条是 user）。同 Phase 1 的占位消息问题，Phase 2 不深入处理。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|---|---|---|---|
| 注册机制 | 程序式 + 声明式都支持 | 仅一种 | 用户要 4 类用途（含配置文件即可加 hook 的需求） |
| 同步 vs 异步 | 同步 | asyncio | 保持主循环简洁；声明式 hook 超时兜底 |
| fail-open 默认 | 是 | fail-closed 默认 | 避免一个 bug hook 让整个 agent 卡死；fail_closed 可配 |
| Hook 注入位置 | 主循环 4 点 + handle_function_call 包裹 | 单一中间件 | 类型安全，调用方拿正确返回类型 |
| 防失控机制 | stop_fire_count 上限 | 无 | Stop hook 可能让循环永不退出 |
| Registry 持有方 | RuntimeContext 注入 | 模块单例 | 与现有 memory_store / session_store 等模式一致 |
| IPC 协议 | JSON over stdin/stdout | 自定义二进制 / HTTP | 简单、语言无关、易调试 |
| PreToolUse deny 返回 | hook_deny error_type | 跳过 tool 消息 | 必须保 tool_call 配对（CLAUDE.md 铁律） |

## 附录 B: 与 Phase 1 协同

| Phase 1 机制 | 与 Hooks 的关系 |
|---|---|
| 上下文压缩（compress_if_needed） | 在 USER_PROMPT_SUBMIT 之后跑，hook 看到的是用户原始/修改后 prompt |
| 工具 offload（>30KB 落盘） | POST_TOOL_USE 在 offload 之前跑，hook 看到 handler 原始输出 |
| reactive_compact | 与 Hooks 独立，hook 不能拦截 reactive |
| use_new_pipeline 开关 | 与 config.hooks.enabled 独立 |

## 附录 C: 与现有 CLAUDE.md 原则的对齐

| 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰 | Hooks 把扩展点从主循环移到注册回调；agent/__init__.py 只加 4 个调用点 |
| Prompt Caching 神圣不可侵犯 | hooks 不改 system prompt；USER_PROMPT_SUBMIT 改的是 user 消息 |
| 完全可逆 | settings.json 不存在 = 无声明式 hook；程序式 hook 进程内重启消失 |
| 用户意图优先于算法 | fail_closed 是用户显式配置；fail-open 默认尊重"hook 是辅助不是关卡" |
| 发现 ≠ 可见 | hooks_registry 是发现；run_hooks 是可见；配置 enabled 控制总开关 |
| 安全默认 > 事后补救 | 声明式 hook 子进程默认 10s 超时；声明式脚本走 PermissionChecker 路径净化（command 是 list 形式，不经 shell）|
