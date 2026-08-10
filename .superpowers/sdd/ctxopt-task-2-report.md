# Task 2 Report: 改造点 ① 大文件落盘精细化（per-tool + per-message + 决策冻结）

## What was implemented

将 L2 offload 从单阈值 10K 字符改造为 3 层精细化落盘：

1. **per-tool 阈值**（50K）：单条 tool result 超 50K 字符 → 落盘到磁盘
2. **per-message 聚合阈值**（200K）：一段连续 tool result（按 user 消息边界分组）总和超 200K → 按大小降序逐个落盘
3. **跨轮次决策冻结**：`_offload_decisions` 模块级 dict 记录已落盘决策，后续轮次 byte-identical 重放预览内容（保护 prompt cache，LRU 1000 上限防内存膨胀）

## TDD Evidence

### RED phase
- 新建 `tests/test_offload_refined.py`，9 个测试用例
- 首次运行：`ImportError: cannot import name 'reset_offload_decisions'` — 函数不存在

### GREEN phase
- 实现 `_offload_decisions` + `_OFFLOAD_DECISIONS_LIMIT` + `_record_decision` + `reset_offload_decisions` + `_enforce_per_message_budget`
- 重构 `offload_large_tool_results` 加 freeze + per-message 逻辑
- 集成到 `compress_if_needed` 生产链路（L2 micro + L2.6 预算 + freeze 预处理）
- 最终：9 passed

## Files changed

| 文件 | 改动 |
|---|---|
| `config.py` | `output_offload_threshold` 10K→50K；加 `message_offload_threshold: 200000` + `offload_decision_freeze: True` |
| `agent/context_pipeline.py` | 重构 `offload_large_tool_results`（加 freeze + per-message）；新增 `_offload_decisions` + `_OFFLOAD_DECISIONS_LIMIT` + `_record_decision` + `_enforce_per_message_budget` + `reset_offload_decisions`；`compress_if_needed` 加 freeze 预处理 + L2.6 记录决策 + threshold 默认 50K |
| `agent/__init__.py` | import `reset_offload_decisions`；`AIAgent.__init__` 调 `reset_offload_decisions()` |
| `tests/test_context_pipeline.py` | 加 autouse fixture `_reset_offload_decisions_per_test`（防跨测试 tool_call_id 冲突） |
| `tests/test_time_based_mc.py` | 加同样的 autouse fixture |
| `tests/test_offload_refined.py` | **新建**：9 个测试（7 核心用例 + E2E + AIAgent 重置） |

## 端到端测试如何覆盖完整链路

`test_e2e_offload_refined_full_chain` 构造真实的 AIAgent 实例，模拟生产调用链：

```
agent = AIAgent(...)                                    # 真实实例
agent.conversation_history = [5 × 60K tool results]    # 300K > 200K
messages = agent._assemble_turn_messages(...)          # Step 1：组装
messages, c1 = await compress_if_needed(messages, ...) # Step 2：compress（L2.6 落盘 + 记决策）
messages = strip_internal_fields(messages)              # Step 3：strip _timestamp

# 第 2 轮（模拟下一轮 LLM 调用）
agent.conversation_history = messages[1:]
messages_t2 = agent._assemble_turn_messages(...)
messages_t2, c2 = await compress_if_needed(messages_t2, ...)  # freeze 重放
messages_t2 = strip_internal_fields(messages_t2)

# 核心断言：
# 1. 第 1 轮落盘的 tool result content == 第 2 轮 content（byte-identical）
# 2. 落盘文件数不增加（第 2 轮不重复落盘）
```

这覆盖了 `_assemble_turn_messages → compress_if_needed → strip_internal_fields` 完整生产链路，防止 Task 1 同样的 "silent-dead-code" bug（测试绿但生产不生效）。

## 现有测试的更新清单

grep 找到基于 10K/30K 阈值的测试位置：

| 文件 | 原值 | 处理 |
|---|---|---|
| `tests/test_context_pipeline.py:388` | `threshold=10000` | **不动**（显式传 threshold，不走默认值） |
| `tests/test_context_pipeline.py:809` | `output_offload_threshold: 30000` | **不动**（显式传 config override） |
| `tests/test_context_pipeline.py:1052,1082` | `output_offload_threshold: 10**9` | **不动**（禁 L2 的测试配置） |
| `tests/test_integration.py:410,430,457` | `output_offload_threshold: 30000` | **不动**（显式传 config，不走默认值） |
| `tests/test_time_based_mc.py:354` | `output_offload_threshold: 10**9` | **不动**（禁 L2） |

结论：**所有现有测试都显式传 threshold/config override**，不依赖默认值，所以改默认值 10K→50K 不破坏任何现有测试。

新增的 autouse fixture（`_reset_offload_decisions_per_test`）解决了跨测试 `_offload_decisions` 泄漏问题——这是新引入的模块级状态，必须隔离。

## Self-review findings

### Completeness
- ✅ 7 个核心用例全覆盖（per-tool 60K / 30K / per-message 250K / 冻结 byte-identical / 跨会话清空 / LRU 1000 / 不跨 user 边界）
- ✅ 端到端测试覆盖完整链路（`_assemble_turn_messages → compress_if_needed → strip_internal_fields`）
- ✅ `AIAgent.__init__` 调 `reset_offload_decisions()`

### Quality
- ✅ `_enforce_per_message_budget` 按 user 消息边界分组（不跨 user 边界）
- ✅ LRU 1000 上限用 dict 保序 + `next(iter())` 淘汰（Py3.7+ 保序）
- ✅ 落盘文件用 `encoding="utf-8"` 写（`maybe_offload` 内部用 `atomic_write_text_lite`，已指定）
- ✅ Windows 路径兼容（`Path` 对象 + `/` 运算符）

### Discipline
- ✅ 没多加 feature flag（新字段直接在 config 的 context 节，无 features flag）
- ✅ 没动 Task 1 的代码（time_based_mc 部分）
- ✅ 没动 Task 3/4 的代码（context_compressor / cache_monitor）

### Testing
- ✅ 现有测试全跑得过（1650 passed, 1 skipped）
- ✅ 端到端测试用真实 AIAgent 实例
- ✅ verify.py 22/22

## Commits created

```
baee5bb2 feat(context): 改造点 ① 大文件落盘精细化（per-tool + per-message + 决策冻结）
```

## Key design decision

`offload_large_tool_results`（独立函数）和 `compress_if_needed` 编排器是两条独立路径：
- `offload_large_tool_results`：被测试直接调用（向后兼容）
- `compress_if_needed`：生产链路（L2 micro_compact + L2.6 总量预算）

决策冻结集成在**两条路径**里：
1. `offload_large_tool_results` 内部有 freeze 逻辑（per-tool 遍历时查 `_offload_decisions`）
2. `compress_if_needed` 有 freeze 预处理（L2 之前遍历重放）+ L2/L2.6 落盘后记录决策

这保证了：
- 直接调 `offload_large_tool_results` 的测试能验证 freeze
- 生产 `compress_if_needed` 链路里 freeze 真生效（E2E 测试覆盖）

## Review Fix Round 1

Commit: `00876465 fix(context): 改造点 ① Round 1 fix：接入 per-message 到生产路径 + L2.6 解耦 + 杂项`

### Important 1: `_enforce_per_message_budget` 死代码修复

**问题**：`_enforce_per_message_budget` 从未在 `compress_if_needed` 生产路径执行（只被 `offload_large_tool_results` 调用，但后者不被生产路径调）。

**Fix（选项 A）**：在 `compress_if_needed` 的 `micro_compact`（c2）后、L2.6 全局预算（c26）前插入 L2.5 per-message 聚合层（`c_per_msg`）。执行顺序变为：

```
time-based MC (c0) → L1 snip (c1) → freeze 预处理 (c_freeze)
→ L2 micro_compact (c2) → L2.5 per-message 聚合 (c_per_msg) [新]
→ L2.6 全局预算 (c26) → L3.5 contextCollapse (c35) → L4 llm (c4)
```

`changed = c0 or c1 or c_freeze or c2 or c_per_msg or c26 or c35 or c4`

**E2E 测试覆盖**：新增 2 个端到端测试（`test_offload_refined.py`）：
- `test_e2e_per_message_budget_runs_in_production`：10×25K（总 250K > 200K），per-tool 不触发 → 只有 per-message 聚合触发，验证 ≥2 条被落盘 + 决策被记录
- `test_e2e_per_message_respects_user_boundary_in_production`：两段 5×30K（每段 150K < 200K），per-message 按 user 边界独立判断 → 不触发

### Important 2: L2.6 budget 解耦

**问题**：`TOTAL_TOOL_BUDGET = config.get("message_offload_threshold", config.get("tool_result_total_budget", 200_000))` 让 L2.6 隐式 aliasing `message_offload_threshold`。

**Fix（选项 a）**：改为 `config.get("tool_result_total_budget", 200_000)`，不再 aliasing。两者语义不同——`message_offload_threshold` 是 per-segment 阈值（L2.5 用），`tool_result_total_budget` 是全局上限（L2.6 用）。

### Minor 3: E2E 注释修正

`test_offload_refined.py:339` 注释从 "L2.6 也会触发" 改为准确描述——micro_keep_recent=3 保护 3 条后，per-message 和 L2.6 实际都不触发，只有 L2 micro（per-tool 60K > 50K）触发。

### Minor 4: DEFAULT_THRESHOLD 对齐

`output_offload.py:DEFAULT_THRESHOLD` 从 30000 改为 50000，跟 `context_pipeline.py` 的 `offload_large_tool_results` 默认值和 `config.py` 的默认值一致。

**受影响测试**：
- `tests/test_output_offload.py`：3 个用 `maybe_offload` 不传 threshold 的测试改为显式传 `threshold=30000`（保持测试语义不变）。此文件被 `.gitignore` 排除，用 `git add -f` 强制纳入。
- `tests/test_integration.py`：`big` 从 50000 改为 60000（确保 > 新 DEFAULT_THRESHOLD 50000）。

### Minor 5: `_resolve_unique_path` 循环上限

`output_offload.py:_resolve_unique_path` 的 `while True` 改为 `for counter in range(1, 1001)`，超出时 raise `OSError`。正常场景下决策冻结机制保证同 tool_call_id 不重复 offload，这个分支几乎到不了，但加上限防病理情况。

### 测试结果

```
tests/test_offload_refined.py: 11 passed（原 9 + 新 2 E2E）
全套：1652 passed, 1 skipped
verify.py: 22/22 ALL PASS
```
