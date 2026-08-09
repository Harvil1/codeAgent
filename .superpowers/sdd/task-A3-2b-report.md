# Task A3 (Plan 2B) 报告：test_hooks*.py + test_hook_*.py

## 概览

| 文件 | 总测试数 | 改 async 数 | 保持 sync 数 |
|---|---|---|---|
| `tests/test_hooks.py` | 51 | 0 | 51 |
| `tests/test_hook_exec.py` | 8 | 0 | 8 |
| `tests/test_hook_loader.py` | 12 | 0 | 12 |
| `tests/test_hook_snapshot.py` | 5 | 0 | 5 |
| `tests/test_builtin_hooks.py` | 14 | 0 | 14 |
| **合计** | **90** | **0** | **90** |

## 改造模式总结

**零改造。** 5 个文件的所有测试保持 sync，因为 hook 系统的全部 API 都是同步函数：

- `agent/hooks.py`：`HookRegistry` 的全部 `register_*` / `run_*` 方法都是 sync（`def run_user_prompt_submit` / `def run_pre_tool_use` / `def register_post_tool_use_failure` 等 30+ 方法）
- `agent/hook_exec.py`：`dispatch_hook` / `run_script_hook` 都是 sync（`def dispatch_hook` / `def run_script_hook`）
- `agent/hook_loader.py`：`load_declarative_hooks` / `_parse_hook` / `get_snapshot` / `reset_snapshot` / `get_disk_version` 都是 sync
- `cli._handle_command`：sync（`def _handle_command`）

对规则 3 列出的所有 async API 关键词（`run_conversation` / `.chat(` / `registry.dispatch` / `handle_function_call` / `chat_completions` / `call_with_retry` / `compress_if_needed` / `_summarize_conversation` / `retrieve_relevant` / `_call_llm_streaming` / `_dispatch_tool_calls` / `AuxLLMRouter.chat_completions`）grep 5 个文件，**0 命中**。

本 task 与 T_A2 的 `test_sandbox*.py`（53 测试 0 改）情况一致：整个 hook 子系统的 API 是 sync 的，没有 async 改造空间。

## TDD before/after

### 5 文件独立（T_A3 范围）

```
Before: 0 failed / 100 passed / 0 skipped  （5 文件合计）
After:  0 failed / 100 passed / 0 skipped  （5 文件合计）
```

Before = After（零改造，测试原状跑通）。

## verify.py 结果

```
[ALL PASS] 总计 22:22 通过，0 失败
```

## 自检 + concerns

### 自检 PASS

- [x] 改造规则 1（签名）未触发：无 async API 调用
- [x] 改造规则 2（Mock 升级）未触发：无 AsyncMock 需求
- [x] 改造规则 3（保持 sync）正确识别 90 个测试无需改
- [x] 改造规则 4（复杂场景）未触发：无 async helper / async fixture / async raises
- [x] 5 文件 0 fail / verify.py 22/22
- [x] commit 在 cli-dev

### concerns

**无 concern。** 零代码改动，零风险。hook 子系统的设计决策（sync API + 子进程 IPC）使其天然不涉及 asyncio 改造。

### 设计观察（非 bug，不修）

hook 系统选择 sync API 是合理的：hook 本身的 I/O（子进程 spawn / HTTP POST）通过 `subprocess.run` / `requests.post` 同步阻塞，声明式 hook 的子进程 IPC 本就是同步的。如果未来 hook 需要 async I/O（如 async HTTP），可能需要 async 版本，但当前无此需求。
