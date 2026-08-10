# Task 3 报告：改造点 ② LLM 摘要质量提升（9 段式 + PTL 重试 + 熔断器）

## Status: DONE

## What Was Implemented

### 1. `agent/context_compressor.py`

- **`SUMMARIZE_PROMPT_9SECTION` 常量**：9 段结构化 prompt（Primary Request and Intent / Key Technical Concepts / Files and Code Sections / Errors and fixes / Problem Solving / All user messages / Pending Tasks / Current Work / Optional Next Step），强制逐字保留文件路径、命令、错误消息、用户原话
- **模块级熔断器状态**：`_consecutive_failures` / `MAX_CONSECUTIVE_FAILURES=3` / `_compact_circuit_open` / `MAX_PTL_RETRIES=3`
- **改造 `_summarize_conversation`**：
  1. 熔断器检查（开 → 直接走规则总结，不调 LLM）
  2. session_memory 优先（有预提取 → 直接返回，不调 LLM）
  3. 无客户端 → 走规则总结
  4. 9 段式 prompt
  5. PTL 重试（prompt_too_long → 丢 20% 旧消息，最多重试 3 次）
  6. 成功后剥离 `<analysis>` 草稿 + 重置熔断器
  7. 失败累加熔断器计数，达上限开启熔断
- **`_format_dialog_for_summary`**：从原 `_summarize_conversation` 内联格式化逻辑抽取成独立函数（tool 消息截断、assistant(tool_calls) 显示工具名、其他原样显示）
- **`_strip_analysis_draft`**：用 regex 剥离 `<analysis>...</analysis>` 草稿区
- **`reset_compact_circuit_breaker`**：会话开始时重置 `_consecutive_failures=0` + `_compact_circuit_open=False`

### 2. `agent/context_pipeline.py`

- `llm_compact` 函数签名加 `session_memory: Optional[str] = None` 参数
- 透传给 `_summarize_conversation(..., session_memory=session_memory)`

### 3. `agent/__init__.py`

- 在 `AIAgent.__init__` 加 `from agent.context_compressor import reset_compact_circuit_breaker`
- 在 `reset_offload_decisions()` 旁边调 `reset_compact_circuit_breaker()`

## TDD Evidence

- **RED**：测试文件 `tests/test_summarize_9section.py` 创建后运行，ImportError（`_strip_analysis_draft` 等不存在）
- **GREEN**：实现后 10/10 passed

## Files Changed

| 文件 | 改动 |
|---|---|
| `agent/context_compressor.py` | 加 `SUMMARIZE_PROMPT_9SECTION` / 熔断器常量 / 改造 `_summarize_conversation` / 加 `_format_dialog_for_summary` / `_strip_analysis_draft` / `reset_compact_circuit_breaker` |
| `agent/context_pipeline.py` | `llm_compact` 加 `session_memory` 参数透传 |
| `agent/__init__.py` | import + `AIAgent.__init__` 调 `reset_compact_circuit_breaker()` |
| `tests/test_summarize_9section.py` | 新文件：10 个测试（7 核心 + 1 单元 + 2 端到端） |

## 端到端测试覆盖（重点）

Task 1/2 reviewer 发现的问题：单元测试覆盖了函数但没覆盖生产链路。本任务加的端到端测试：

1. **`test_e2e_9section_prompt_via_compress_if_needed`**：构造大对话 → 通过 `compress_if_needed` → `llm_compact` → `_summarize_conversation` 完整链路触发，捕获实际发给 LLM 的 messages，验证含 "Primary Request and Intent" 等 9 段标题。这验证了 9 段式 prompt **在生产路径真的发出**，不只是单元测试里测一下。

2. **`test_e2e_circuit_breaker_via_compress_if_needed`**：连续 3 次 `compress_if_needed` 调用让 LLM 失败 → 熔断器累积到 MAX → 第 4 次 `compress_if_needed` 调用时验证 `mock.call_count` 不增加（熔断器在 `_summarize_conversation` 层拦截，LLM 根本没被调用）。这验证了熔断器 **跨调用持久** 且在生产路径真生效。

## SessionStore.get_memory_extract 决策

- **不存在**：grep `get_memory_extract` 全项目 0 匹配
- **软目标决策**：按 brief Step 3 指示，只在 `_summarize_conversation` 函数签名加 `session_memory` 参数（默认 None），`llm_compact` 加透传参数，**不实现 SessionStore 改造**（Phase 2 再做）
- **当前效果**：`session_memory` 永远 None，替代路径不触发；但参数已就绪，Phase 2 只需在 `compress_if_needed` 里读 SessionStore 并传入

## `_rule_based_summary` 抽取来源

- **原位置**：`agent/context_compressor.py` 第 86-99 行（Task 3 之前已存在的独立函数）
- **抽取方式**：**不是抽取，是复用**——该函数在 Task 3 之前就已经是独立函数。新代码的 `_summarize_conversation` 在所有失败路径（熔断器触发 / 非 PTL 异常 / PTL 重试耗尽 / 无客户端）都调用 `_rule_based_summary(messages)`，与原失败路径 `return _rule_based_summary(messages)` 语义完全一致
- **验证**：函数体未被修改，`test_rule_based_summary` 测试仍通过

## Self-Review Findings

- [x] 7 个核心测试 + 端到端都加了（10 个测试：7 核心 + 1 单元 + 2 端到端）
- [x] `_rule_based_summary` 复用现有函数（grep 验证，函数体未改）
- [x] 熔断器测试验证 mock 调用次数（`call_count_before/after` 比较，不只看返回值）
- [x] session_memory 测试验证 `call_count == 0`
- [x] `reset_compact_circuit_breaker()` 在 `AIAgent.__init__` 调了
- [x] 跟 Task 1+2 的 `reset_offload_decisions()` 在同一位置（相邻行）
- [x] 没动 Task 4 的代码（cache_monitor 不存在）
- [x] 没动 Task 1+2 已完成的代码（time_based_clear_old_tool_results / offload_decisions 等未改）

## Verification

- `uv run pytest tests/test_summarize_9section.py -v` → 10/10 passed
- `uv run pytest tests/ --tb=no -q` → 1662 passed, 1 skipped
- `uv run python scripts/verify.py` → 22/22 PASS

## Commits

（待 commit）
