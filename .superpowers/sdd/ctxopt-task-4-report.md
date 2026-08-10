# Task 4 报告：改造点 ③ prompt cache 检测系统（12 维度监控 + break 根因定位）

## Status: DONE

## 实现内容

### 1. 新建 `agent/cache_monitor.py`
- `PromptState` dataclass：5 核心维度（system_hash / tools_hash / model / cache_strategy / betas_hash），12 维度剩余 7 个留 TODO
- `record_prompt_state(system_prompt, tools, model, **kwargs)` → `PromptState`：pre-call 快照
- `check_cache_break(current_state, cache_read_tokens, query_source)` → `Optional[str]`：post-call 检测，break 时返回根因字符串
- `notify_compaction()`：标记下次 cache 下降是预期的（compact 后调）
- `get_stats()`：返回 dict 给 `/cache-stats` 展示
- `reset_cache_monitor()`：会话开始时重置模块状态
- break 判定条件：`cache_read < _last_cache_read * 0.95` **且** `token_drop >= 2000`
- `_break_history` 限长 100 条（防长会话内存膨胀）
- 全部 fail-open（异常只 `logger.debug`）

### 2. `agent/__init__.py` 接入
- **`_extract_cache_read(usage)` 静态方法**：双路径兜底
  - dict 路径：`usage.get("prompt_cache_hit_tokens", 0) or usage.get("cache_read_input_tokens", 0)`
  - 对象路径：`getattr(usage, "prompt_cache_hit_tokens", 0) or getattr(usage, "cache_read_input_tokens", 0)`
  - DeepSeek 用 `prompt_cache_hit_tokens`，Anthropic 用 `cache_read_input_tokens`
- **`_call_llm_with_escalation` pre/post hook**：
  - pre-call：函数入口处 `record_prompt_state`（fail-open try/except）
  - post-call：`return response` 前 `check_cache_break`（fail-open try/except）
  - **流式 + 非流式都覆盖**：两个分支都在同一个 `try` 块内，post-call hook 在 `return response` 前（分支汇合点）
- **`AIAgent.__init__` 调 `reset_cache_monitor`**：第 204-209 行，在 `reset_compact_circuit_breaker` 旁边，fail-open

### 3. `agent/context_pipeline.py:llm_compact` 接入
- 在 `return new_messages, True` 前调 `notify_compaction()`（fail-open try/except）
- 只在实际发生压缩时调（`return messages, False` 路径不触发）

### 4. `cli.py` 新增 `/cache-stats` slash 命令（Step 5 已实现）
- `/cache-stats`：展示 total_breaks / last_break / last_cache_read
- fail-open（异常只打印错误）

## TDD 证据

### RED（实现前）
```
27 failed, 948 warnings in 2.10s
```
全部 ImportError（`agent.cache_monitor` 不存在）或 AttributeError（`_extract_cache_read` 不存在）。

### GREEN（实现后）
```
27 passed, 949 warnings in 4.10s
```

### 全套回归
```
1691 passed, 1 skipped, 157391 warnings in 67.63s
```
（之前 1625+，新增 27 个 cache_monitor 测试 + 其他 suite 增长）

### verify.py
```
[ALL PASS] 总计 22/22 通过，0 失败
```

## 文件改动清单

| 文件 | 改动 |
|---|---|
| `agent/cache_monitor.py` | 新建模块（178 行） |
| `agent/__init__.py` | `_extract_cache_read` 静态方法（第 420-443 行）；`_call_llm_with_escalation` pre/post hook（第 1242-1303 行）；`__init__` 调 `reset_cache_monitor`（第 204-209 行） |
| `agent/context_pipeline.py` | `llm_compact` 末尾调 `notify_compaction`（第 683-686 行） |
| `cli.py` | `/cache-stats` slash 命令（第 1401-1420 行） |
| `tests/test_cache_monitor.py` | 新建测试文件（27 个测试） |

## 流式 vs 非流式接入位置

**同一个函数 `_call_llm_with_escalation`（`agent/__init__.py:1238`）**：
- pre-call hook：第 1244-1253 行（函数入口，try 块之前，覆盖两个分支）
- 流式分支：第 1256-1260 行（`_call_llm_streaming`）
- 非流式分支：第 1261-1288 行（`call_with_retry` + max_tokens 升级）
- post-call hook：第 1290-1303 行（`return response` 前，分支汇合点，覆盖两个路径）

## usage 字段访问的实际形态

**DeepSeek/OpenAI 兼容路径**（`agent/llm_client.py:OpenAICompatClient`）：
- 非流式：`response.usage` 是 OpenAI SDK 的 `CompletionUsage` 对象（有 `prompt_cache_hit_tokens` 属性）
- 流式：`_call_llm_streaming` 末尾把 `final_usage` 包成 `SimpleNamespace`，同时塞 `prompt_cache_hit_tokens` 和 `cache_read_input_tokens`（两个都赋同值）

**Anthropic 路径**（`agent/llm_client.py:AnthropicClient`）：
- `cache_read_input_tokens` 字段名

**`_extract_cache_read` 兜底策略**：
```python
if isinstance(usage, dict):
    return usage.get("prompt_cache_hit_tokens", 0) or usage.get("cache_read_input_tokens", 0) or 0
return getattr(usage, "prompt_cache_hit_tokens", 0) or getattr(usage, "cache_read_input_tokens", 0) or 0
```
dict/对象双路径 + DeepSeek/Anthropic 双字段名 = 4 种组合全覆盖。

## 端到端测试覆盖清单

| 测试 | 验证点 |
|---|---|
| `test_e2e_run_conversation_triggers_cache_monitor` | 通过真实 `run_conversation` 跑一轮主循环，验证 `record_prompt_state` + `check_cache_break` 真的被调 |
| `test_e2e_call_llm_with_escalation_non_stream_triggers_hook` | 非流式 `_call_llm_with_escalation` 路径触发 hook |
| `test_e2e_call_llm_with_escalation_stream_triggers_hook` | 流式路径（真实 async generator mock）触发 hook |
| `test_e2e_fail_open_record_raises` | mock `record_prompt_state` 抛异常 → 主循环不崩 |
| `test_e2e_fail_open_check_raises` | mock `check_cache_break` 抛异常 → 主循环不崩 |
| `test_e2e_fail_open_extract_cache_read_raises` | response.usage=None → 主循环不崩 |
| `test_e2e_llm_compact_calls_notify_compaction` | `llm_compact` 实际压缩时调 `notify_compaction` |
| `test_e2e_compact_then_no_break` | compact 触发 `notify_compaction` 后，cache 大降不算 break |
| `test_reset_called_in_init` | `AIAgent.__init__` 真的调 `reset_cache_monitor`（状态归零） |

## Step 5 `/cache-stats`

已实现（`cli.py:1401-1420`）。输出格式：
```
本次会话 cache 累计 break 次数：N
最近 break：cache read FROM → TO（降 DROP tokens）
根因：CAUSE
最近一次 cache read：N tokens
```

## Self-Review 结果

| 检查项 | 状态 |
|---|---|
| cache_monitor.py 按 brief 骨架写 | YES |
| 流式 + 非流式分支都 hook | YES（同一函数，分支汇合点 post-call） |
| `_extract_cache_read` 双路径（dict/对象，DeepSeek/Anthropic 字段名） | YES |
| 所有接入点 try/except fail-open | YES（pre-call / post-call / notify_compaction / reset 全部 fail-open） |
| `llm_compact` 末尾调 `notify_compaction` | YES（第 683-686 行） |
| `AIAgent.__init__` 调 `reset_cache_monitor` | YES（第 204-209 行） |
| 端到端测试覆盖完整链路 | YES（9 个 e2e 测试） |
| fail-open 端到端测试覆盖 | YES（3 个 fail-open 测试） |
| 没动 Task 1+2+3 的代码 | YES（只在它们的 commit 基础上加新代码） |
| `_break_history` 限长 100 | YES（单元测试验证） |

## Commits

待提交。
