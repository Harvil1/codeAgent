# Task P1.2 报告：L5 reactive_compact 响应式回压 feature flag

**分支**：cli-dev
**Base**：3d3bc0c0 (P1.1)
**日期**：2026-07-17

## 现有 _REACTIVE_RETRY 机制描述

Plan 2A 已在 `agent/__init__.py` 实现完整响应式回压路径：

1. **sentinel**：`AIAgent._REACTIVE_RETRY = object()`（类级哨兵，主循环据此识别"应重试本轮"）
2. **触发点**：`_call_llm_with_escalation()` 的 `except Exception` 分支
3. **检测条件**（字符串匹配 3 个 keyword，容错不同 provider）：
   - `"prompt_too_long"` / `"context_length"` / `"maximum context"`
4. **once-per-session**：`self._reacted: bool` 标记，构造函数显式初始化为 `False`（`agent/__init__.py:260`）
5. **动作**：调用 `agent.context_pipeline.reactive_compact(messages, ...)`，把历史截到 system + placeholder + 最近 N 条；更新 `self.conversation_history`；`invalidate_system_prompt()`；返回 `_REACTIVE_RETRY`
6. **主循环处理**：`run_conversation()` 看到 `_REACTIVE_RETRY` → 重新拿 system prompt + `continue`（重试本轮）

**问题**：上述路径**无任何开关**，行为默认常开。spec §7.1 要求"加 flag 开关即可"——与本 task 假设完全吻合。

## flag 接入位置 + 方式

**位置**：`agent/__init__.py` 的 `_call_llm_with_escalation` 异常分支

**方式**：在原有 `is_prompt_too_long and not self._reacted` 条件**前置** flag 检查（短路求值，flag OFF 时不进压缩路径）：

```python
from agent.feature_flags import is_feature_enabled
reactive_enabled = is_feature_enabled(self.config, "reactive_compact")
if (reactive_enabled
        and is_prompt_too_long
        and not getattr(self, "_reacted", False)):
    # ... 原有 reactive_compact 逻辑不变
```

flag 已在 P1.0 阶段注册到 `config.py:DEFAULT_CONFIG["features"]["reactive_compact"]`（默认 `enabled=False`），无需再加配置。

**为何这样接入**（不破坏现有路径）：
- flag ON 时，三个条件短路求值顺序与原条件等价（原是 `is_prompt_too_long and not _reacted`，现在前面加一个真值检查，逻辑合取不变）
- flag OFF 时直接跳过整个压缩块，落到下面 `logger.error + 返回 None` 的原有错误处理路径
- `_reacted` 状态机不变（flag OFF 时不会被改成 True，测试已覆盖）

## 改动行数

| 文件 | 改动 |
|---|---|
| `agent/__init__.py` | +8 / -1 |
| `tests/test_run_conversation_async.py` | +95 / 0（3 个新测试） |
| **合计** | **+103 / -1** |

## TDD RED/GREEN

### RED（flag gate 接入前）

```
tests/test_run_conversation_async.py::test_reactive_compact_flag_off_skips_retry FAILED
  AssertionError: reactive_compact flag OFF 时不应返回 _REACTIVE_RETRY，应返回 None
```

另 2 个测试（flag_on / once_per_session）当时已通过（旧逻辑常开），但仍保留作为回归保护。

### GREEN（flag gate 接入后）

```
tests/test_run_conversation_async.py::test_reactive_compact_flag_off_skips_retry PASSED
tests/test_run_conversation_async.py::test_reactive_compact_flag_on_triggers_retry PASSED
tests/test_run_conversation_async.py::test_reactive_compact_once_per_session PASSED
======================= 3 passed =======================
```

## verify.py 结果

```
[ALL PASS] 总计 22/22 通过，0 失败
```

## 测试套件总览

- **本 task 新增 3 测试**：全部 PASS
- **全量 pytest**：1509 passed, 1 skipped, 1 deselected
- **唯一失败**：`tests/test_tool_concurrency_classification.py::test_all_tools_classified`（已验证在 stash 干净树上同样偶发，是既存的全套运行时顺序污染，与本次改动无关；独立运行 PASS）

## 自检

- [x] 分支 cli-dev（commit 前再次 `git branch --show-current` 确认）
- [x] flag OFF → 返回 None（测试覆盖）
- [x] flag ON → 返回 _REACTIVE_RETRY + 压缩 history（测试覆盖）
- [x] flag ON + 已触发 → 返回 None（once-per-session 保持，测试覆盖）
- [x] flag 已在 DEFAULT_CONFIG 注册（P1.0 阶段已做）
- [x] feature_flags test 列表含 `reactive_compact`（`tests/test_feature_flags.py:121` 已存在）
- [x] 中文注释 + commit message
- [x] 不破坏 prompt cache（只动 messages + invalidate_system_prompt，与原逻辑一致）
- [x] verify.py 22/22

## Concerns

1. **fail-open 语义**：flag OFF 时 context_length_exceeded 走原错误分支（写 `[API 错误]` 进 history + 返回 None → 主循环 break），与 spec §5.4 "fail-safe 默认关" 一致。无新风险。
2. **flag OFF 行为变化**：原本 reactive 路径默认开，现在默认关。**这是设计意图**（spec §7.1 + Plan 3 batch3 明确要求 flag 化）。任何依赖该自动行为的用户需显式 `features.reactive_compact.enabled=true`。无过渡期 deprecation，因 batch3 整体处于 feature flag 灰度阶段。
3. **现有集成测试 `test_compression_pipeline_end_to_end`** 直接调用 `reactive_compact()` 函数（不经 `_call_llm_with_escalation`），不受 flag 影响，仍 PASS——确认 flag 只在 agent 主循环层生效，pipeline 函数本身保持可独立调用。
