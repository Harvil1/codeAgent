# CCAR4 Task A Report: cache 检测 12 维度 + per-tool hash + diff 文件 + TTL 分析

## What was implemented

把 `agent/cache_monitor.py` 从 5 维度扩展到完整 12 维度（对齐 claude-code-main promptCacheBreakDetection.ts），加 per-tool hash（指出哪个工具变了）、diff 文件落盘（break 后写 unified diff）、TTL 时长分析（5min / 1h 阈值）。

## TDD Evidence

### RED phase
```
17 failed, 33 passed  （新增 21 个测试，17 个先红）
```

### GREEN phase
```
50 passed  （全绿，包含 29 个老测试 + 21 个新测试）
```

## Files changed

| 文件 | 改动 |
|---|---|
| `agent/cache_monitor.py` | PromptState 加 7 维 + ToolHashEntry + tool_hashes 字段；record_prompt_state 接 12 维参数；check_cache_break 用 _diagnose_break（12 维对比 + TTL）+ _write_break_diff（diff 文件落盘）+ _enforce_diff_lru_limit（LRU 清理）；新增 _diff_tool_hashes / _state_system_text / set_diff_limit |
| `agent/__init__.py:_call_llm_with_escalation` | 调 record_prompt_state 时传 max_tokens / temperature / stream_mode / user_content_prefix / messages_count（从 config + messages 提取） |
| `agent/__init__.py:__init__` | 从 config 读 max_cache_break_diff_files 调 set_diff_limit |
| `cli.py:/cache-stats` | 加 diff_path 输出 |
| `config.py` | 加 max_cache_break_diff_files（默认 100） |
| `tests/test_cache_monitor.py` | 加 21 个测试（12 维 + per-tool hash 增删改 + diff 落盘 + TTL + 端到端 + fail-open + 回归） |

## 12 维度实现清单

| # | 维度 | 字段 | 端到端测试 |
|---|---|---|---|
| 1 | system_hash | PromptState.system_hash | test_check_cache_break_reports_all_12_dimensions |
| 2 | tools_hash | PromptState.tools_hash | test_check_cache_break_reports_all_12_dimensions |
| 3 | model | PromptState.model | test_check_cache_break_reports_all_12_dimensions |
| 4 | cache_strategy | PromptState.cache_strategy | test_check_cache_break_reports_all_12_dimensions |
| 5 | betas_hash | PromptState.betas_hash | test_check_cache_break_reports_all_12_dimensions |
| 6 | max_tokens | PromptState.max_tokens | test_check_cache_break_reports_all_12_dimensions + test_e2e_12_dimensions_captured_in_record |
| 7 | temperature | PromptState.temperature | test_check_cache_break_reports_all_12_dimensions |
| 8 | stream_mode | PromptState.stream_mode | test_check_cache_break_reports_all_12_dimensions |
| 9 | tool_choice | PromptState.tool_choice | test_check_cache_break_reports_all_12_dimensions |
| 10 | user_content_prefix | PromptState.user_content_prefix | test_check_cache_break_reports_all_12_dimensions |
| 11 | messages_count | PromptState.messages_count | test_check_cache_break_reports_all_12_dimensions |
| 12 | system_boundary | PromptState.system_boundary | test_check_cache_break_reports_all_12_dimensions + test_record_prompt_state_system_boundary_multi_block |

## per-tool hash 实现清单

| 能力 | 实现 | 测试 |
|---|---|---|
| ToolHashEntry dataclass | name + schema_hash | test_record_prompt_state_captures_12_dimensions |
| _diff_tool_hashes 增 | +N (names) | test_per_tool_hash_added |
| _diff_tool_hashes 删 | -N (names) | test_per_tool_hash_removed |
| _diff_tool_hashes 改 | ~N (names) | test_per_tool_hash_changed |

## diff 文件实现清单

| 能力 | 实现 | 测试 |
|---|---|---|
| _write_break_diff 落盘 | ~/.OmniMate/.cache-breaks/cache-break-*.diff | test_diff_file_written_on_system_change |
| system 变时写 system 段 | OLD/NEW hash + boundary | test_diff_file_written_on_system_change |
| tools 变时写 tools 段 | OLD/NEW tool names + diff | test_diff_file_written_on_tools_change |
| 非系统/tools 变不写 | model 变无 diff | test_diff_file_not_written_on_non_schema_break |
| LRU 100 上限 | _enforce_diff_lru_limit | test_diff_file_lru_cap |
| _break_history 含 diff_path | stats 可展示 | test_e2e_break_history_includes_diff_path |

## TTL 时长分析清单

| 场景 | 根因输出 | 测试 |
|---|---|---|
| 无字段变化 + elapsed < 5min | "server-side 或未知" | test_ttl_analysis_no_field_change_short_elapsed |
| 无字段变化 + 5min < elapsed < 1h | ">5min TTL 过期" | test_ttl_analysis_no_field_change_5min |
| 无字段变化 + elapsed > 1h | ">1h TTL 过期" | test_ttl_analysis_no_field_change_1h |

## _call_llm_streaming / 非流式实际传哪些参数

grep 结论：
- `_call_llm_streaming` 从 `self.config["model"]["max_tokens"]` 或 `self.config["llm"]["max_tokens"]` 读 max_tokens，通过 `**_extra` 传 `chat_completions_stream`
- 非流式路径通过 `call_with_retry(max_tokens=...)` 传 max_tokens（升级时用）
- `temperature` / `tool_choice` / `betas` 在当前代码**未显式传给 LLM**（靠 SDK 默认值）
- 但 `config["model"]["temperature"]` 在 DEFAULT_CONFIG 中存在（=0.7）

**实现策略**：record_prompt_state 的参数从 `self.config` 读（max_tokens / temperature），而非从 LLM 调用 kwargs 读——因为 kwargs 链路里这些参数不显式传递。stream_mode 从 `self._stream_callback is not None` 推断。user_content_prefix 从 messages 第一条 user 消息提取。messages_count 直接 len(messages)。

## Self-review findings

- [x] 12 维度都实现（system/tools/model/cache_strategy/betas/max_tokens/temperature/stream_mode/tool_choice/user_content_prefix/messages_count/system_boundary）
- [x] per-tool hash 实现（ToolHashEntry 列表 + _diff_tool_hashes）
- [x] diff 文件落盘（_write_break_diff，~/.OmniMate/.cache-breaks/）
- [x] TTL 时长分析（5min / 1h 阈值）
- [x] 端到端测试覆盖（_call_llm_with_escalation 真触发，3 个 e2e 测试）
- [x] fail-open 所有路径（try/except + logger.debug，含 _write_break_diff 独立 try/except）
- [x] 现有 29 个 cache_monitor 测试全过（回归）
- [x] /cache-stats 加 diff_path 输出

### 注意事项
- PromptState 只存 hash 不存原文（避免内存占用 + 保护 prompt 内容不落盘到 diff），所以 diff 文件展示 hash 对比而非原文 diff
- diff_path 在 _break_history 中可能是 None（非 system/tools 变化时）
- _last_baseline_at 每次 check_cache_break 更新 baseline 时刷新（不 break 时也更新）

## Verify

```
uv run pytest tests/test_cache_monitor.py -v    → 50 passed
uv run pytest tests/ --tb=no -q                 → 1706 passed, 1 skipped
uv run python scripts/verify.py                 → 22/22 ALL PASS
```

## Commits

- `be7a9a18` feat(cache): CCAR4 Task A 扩展 cache 检测到 12 维度 + per-tool hash + diff 文件
- `5b9ae67c` fix(cache): CCAR4 Task A Round 1 修 system delta + tool_choice/betas 接入 + diff 文件名唯一

## Review Fix Round 1

Reviewer 判定 "Approved with 3 Important + 3 Minor"，6 个 findings 全修。

### Important 1: system delta 总是 (+0 chars)

**根因**：`_state_system_text` 返回空字符串（PromptState 只存 hash 不存原文），导致 delta 永远是 0。

**Fix**：
- PromptState 加 `system_len: int = 0` 字段（只存长度不存原文）
- `record_prompt_state` 计算 system_prompt 长度填入（兼容 str 和 list of blocks）
- `_diagnose_break` 用 `cur.system_len - prev.system_len` 算 delta
- 删除 `_state_system_text` 函数（不再用）

**测试**：`test_system_delta_shows_real_char_count`（delta +50）+ `test_system_delta_shows_negative_char_count`（delta -120）+ `test_system_len_multi_block`（list of blocks）

### Important 2: tool_choice/betas 生产路径死维度

**根因**：`agent/__init__.py:_call_llm_with_escalation` 调 record_prompt_state 没传 tool_choice/betas，报告说"从 config 读兜底"但实际没读。

**Fix**：从 `config["model"]["tool_choice"]` 和 `config["model"]["betas"]` 读，传入 record_prompt_state。两者都为 None 时 hash 一致，不触发 break（恒定 baseline，OK 行为）。

**测试**：`test_e2e_tool_choice_betas_passed_from_config`（config → record 端到端验证）+ `test_tool_choice_none_does_not_trigger_break`（None 恒定 baseline）

### Important 3: diff 文件名同秒覆盖

**根因**：`datetime.now().strftime("%Y%m%d-%H%M%S")` 只到秒，同秒写多个 diff 覆盖。

**Fix**：加模块级 `_diff_counter` 递增计数器，文件名格式 `cache-break-{ts}-{counter:04d}.diff`。

**测试**：`test_diff_filename_unique_same_second`（同秒 5 次 break → 5 个唯一文件名）

### Minor 4: `_diff_limit` 定义顺序

`_diff_limit = 100` 从文件末尾移到 `_read_diff_limit` 函数之前（定义先于使用）。

### Minor 5: `_enforce_diff_lru_limit` 每次跑

改为只在 `diff_path` 非 None（真写了 diff）时才调 `_enforce_diff_lru_limit()`，避免无 diff 写入时的无谓 IO。

### Minor 6: `test_diff_file_lru_cap` 精确断言

`assert <= 100` 改为 `assert == 100`（105 次写入后 LRU 应精确到 100）。

### 测试结果

```
uv run pytest tests/test_cache_monitor.py -v    → 56 passed（原 50 + 新 6）
uv run pytest tests/ --tb=no -q                 → 1712 passed, 1 skipped
uv run python scripts/verify.py                 → 22/22 ALL PASS
```
