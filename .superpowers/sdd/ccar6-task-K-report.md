# Task K: 中断机制完整化 — 实施报告

**Status**: 完成（无降级）
**Branch**: cli-dev
**Base**: 29acae73

## 实现内容

完整实现 brief Step 1-6（未降级）：

1. **Step 1 sync cancel_event**：`_delegate_sync` 用 `threading.Event()` 替代 timeout+abandon。
   主线程 `thread.join(timeout)` 超时后 `cancel_event.set()`，再等 `sync_cancel_timeout_seconds`
   （默认 2.0s）让子代理优雅退出；仍未退出才 abandon。

2. **Step 2 child AIAgent 检查 cancel_event**：`AIAgent.run_conversation` 加 `cancel_event=None`
   参数，主循环 while 开头（紧跟 `_interrupt_requested` 检查后）检查 `cancel_event.is_set()`，
   触发立即 `return self._extract_partial_result()`。

3. **Step 3 `_extract_partial_result`**：扫 `conversation_history` 反向找最后一条非空
   `assistant` 消息，返回 `[PARTIAL] {content}`；无则空串。fail-open（异常返空串）。

4. **Step 4 async kill + subagent_kill 工具**：`_delegate_async` 也创建 cancel_event，
   注册到新增的全局 `_async_tasks` dict（key=delegation_id, value={thread, cancel_event}）。
   `subagent_kill(task_id)` 查注册表 → set cancel_event → 返 success。
   子代理退出时 `finally` pop 注册表项（防泄漏）。
   check_fn `_subagent_kill_check_fn` 读 `config.delegation.async_kill_enabled`
   控制可见性。

5. **Step 5 batch KeyboardInterrupt 传播**：`_delegate_batch` 每任务一个独立 cancel_event，
   存到 `batch_cancel_events` list；`except KeyboardInterrupt` 分支 set 所有 cancel_event
   （不只是 parent.interrupt()）。

6. **Step 6 config flag**：`config.delegation` 加
   `sync_cancel_timeout_seconds: 2.0` + `async_kill_enabled: True`。

## TDD 证据

新建 `tests/test_subagent_cancellation.py`，17 个测试：
- `TestSyncCancelEvent` (3)：cancel_event 创建/优雅退出/预 set 快速返回
- `TestExtractPartialResult` (4)：返回最后 assistant / 空历史 / 空 assistant / 异常 fail-open
- `TestSubagentKill` (4)：schema 校验 / not_found / set cancel_event / async 注册到 _async_tasks
- `TestBatchKeyboardInterrupt` (1)：KeyboardInterrupt 不死锁 + cancel_event 传播
- `TestConfigFlags` (2)：两个 flag 存在 + 默认值合理
- `TestEndToEndCancel` (1)：sync + 慢 LLM + cancel 不阻塞
- `TestRunConversationCancelEvent` (2)：run_conversation 接受参数 + cancel 检测退出

RED → GREEN 流程：
- 初始 RED：7 测试 fail（缺 `_async_tasks` / `_handle_subagent_kill` / `cancel_event` 参数）
- 实现 Step 1-6 后全 GREEN（17/17 passed）

## cancel_event 传递链（file:line）

| 层 | file:line | 关键代码 |
|---|---|---|
| 1. _delegate_sync 创建 | `tools/delegate_tool.py:240` | `cancel_event = threading.Event()` |
| 2. _delegate_sync 注入 kwargs | `tools/delegate_tool.py:241` | `kwargs["cancel_event"] = cancel_event` |
| 3. _delegate_async 创建 | `tools/delegate_tool.py:296` | `cancel_event = threading.Event()` |
| 4. _delegate_async 注入 kwargs | `tools/delegate_tool.py:297` | `kwargs["cancel_event"] = cancel_event` |
| 5. _delegate_async 注册表 | `tools/delegate_tool.py:433` | `_async_tasks[delegation_id] = {...}` |
| 6. _delegate_batch 每任务独立 | `tools/delegate_tool.py:478-482` | `task_cancel = threading.Event()` + `task_kwargs["cancel_event"]` |
| 7. _run_child 读 kwargs | `tools/delegate_tool.py:936-941` | `_cancel_event = kwargs.get("cancel_event")` |
| 8. _run_child 传给 child.chat | `tools/delegate_tool.py:940` | `child.chat(..., cancel_event=_cancel_event)` |
| 9. AIAgent.chat 接受 | `agent/__init__.py:1897-1903` | `async def chat(self, message, cancel_event=None)` |
| 10. chat 透传 run_conversation | `agent/__init__.py:1902` | `run_conversation(message, cancel_event=...)` |
| 11. run_conversation 接受 | `agent/__init__.py:752` | `async def run_conversation(self, user_message, cancel_event=None)` |
| 12. run_conversation 每轮检查 | `agent/__init__.py:815-822` | `if cancel_event is not None and cancel_event.is_set():` |

## extractPartialResult 实现

`agent/__init__.py:1859-1883` `AIAgent._extract_partial_result`：
- 反向扫 `conversation_history`
- 找到 `role=assistant` 且 `content` 非空字符串
- 返回 `[PARTIAL] {content}`（前缀让父代理识别中断结果）
- fail-open：任何异常返回空串

## subagent_kill 工具

**已加**（未降级）。

- Schema：`tools/delegate_tool.py` 的 `SUBAGENT_KILL_SCHEMA`
- Handler：`_handle_subagent_kill`（查 `_async_tasks` → set cancel_event）
- check_fn：`_subagent_kill_check_fn`（读 `config.delegation.async_kill_enabled`）
- 注册：`registry.register(name="subagent_kill", toolset="core", check_fn=...)`
- 更新 `toolsets.py:_CORE_TOOLS` + `tests/test_tool_concurrency_classification.py:UNSAFE_TOOLS`

## 端到端测试覆盖

`TestEndToEndCancel::test_sync_subagent_cancellation_e2e`：
- mock `AIAgent` 的 FakeChild（conversation_history 有 assistant 消息）
- `child_timeout=0.1` → 主线程立即超时
- `sync_cancel_timeout_seconds=0.5` → 给 0.5s 优雅退出
- 验证返回 mode="sync"（不阻塞 5s+）

`TestRunConversationCancelEvent::test_run_conversation_checks_cancel_event_each_turn`：
- cancel_event 预先 set
- 验证 run_conversation 返回 `[PARTIAL] working...`（来自 _extract_partial_result）

## 文件改动

- `tools/delegate_tool.py` — _delegate_sync / _delegate_async / _delegate_batch / _run_child 改造 + subagent_kill 工具
- `agent/__init__.py` — run_conversation + chat 加 cancel_event 参数 + _extract_partial_result 方法
- `config.py` — delegation 加 2 flag
- `toolsets.py` — _CORE_TOOLS 加 subagent_kill
- `tests/test_subagent_cancellation.py` — 新建（17 测试）
- `tests/test_tool_concurrency_classification.py` — UNSAFE_TOOLS 加 subagent_kill

## Commits

见 git log（1 个 commit：feat(delegation): Task K 子代理中断机制完整化）。

## 验证清单

```
tests/test_subagent_cancellation.py:  17 passed
tests/test_delegation.py:             24 passed（回归）
全套:                                1907 passed, 1 skipped（1890 baseline + 17 new）
scripts/verify.py:                   22/22 PASS
```
