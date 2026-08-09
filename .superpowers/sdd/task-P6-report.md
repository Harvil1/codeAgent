# Task P6 Report — Plan Mode V2 多 Agent 并行（P6.1 + P6.2）

## 实现概要

实现 spec §7.6（Plan Mode V2 多 Agent 并行），feature flag `plan_mode_v2_parallel`（批次 1 已注册，默认 OFF）。

### _run_parallel_planners 实现

**位置**：`tools/plan_mode_tool.py:_run_parallel_planners`

```python
def _run_parallel_planners(subtasks: List[str], max_n: int, **kwargs) -> List[str]:
```

**设计决策**：
- 用 `ThreadPoolExecutor`（复用 `_delegate_batch` 已验证的模式，见 `tools/delegate_tool.py:290`），不用 `asyncio.gather`。原因：`_run_child` 是同步函数（内部自己 `asyncio.run`），`asyncio.gather` 要求 coroutine——硬塞要包一层 `run_in_executor`，多此一举。
- `_ABS_MAX_PARALLEL = 8` 硬上限：即使配置写 `max_parallel_agents=100` 也只放 8 个 worker，防爆资源。
- workers = `min(len(subtasks), min(max_n, 8))`。
- 每个子任务调 `_run_child(goal, context, "leaf", subagent_type="Plan")` —— 用 OmniMate 内置 Plan 子代理（`agent/builtin_agents/plan.md`，只读工具集 + maxTurns=30）。
- context 标明 `[Plan Mode V2 并行任务 i/N]`，让子代理知道自己在多 Agent 协作中。
- **兜底**：单个子代理失败被 `try/except` 捕获、log warning 后跳过，不阻塞其他。失败的子计划不出现在结果列表（用剩余合并）。

### plan_mode_tool 接入位置

**位置**：`tools/plan_mode_tool.py:handle_plan_mode_v2_dispatch` + 模块顶层 `registry.register("plan_mode_v2_dispatch", ...)`

新工具 `plan_mode_v2_dispatch` 注册到 `plan` 工具集（`toolsets.py` 的 `"plan"` 段加了一行 `"plan_mode_v2_dispatch"`）。

LLM 进入 plan mode 后就能看到这个工具的 schema。调用时：
1. flag OFF → handler 返回 `feature_disabled`，LLM 看提示走原单 Agent 路径（`/plan` + `exit_plan_mode`）
2. flag ON → handler 拆 N 个子任务并行调度、合并、返回 `merged_plan`
3. LLM 拿到 merged_plan 后可以审阅，按需调 `exit_plan_mode` 提交审批

### Plan 合并 prompt 模板

**位置**：`tools/plan_mode_tool.py:PLAN_MERGE_PROMPT_TEMPLATE` + `_merge_plans`

`_merge_plans(sub_plans, llm_client, model)`：
- 单份 → 直接返回原文（不调 LLM）
- 多份 → 调 LLM 合并，用 `PLAN_MERGE_PROMPT_TEMPLATE`（5 条规则：去重/互补/排顺序/冲突裁决/保留可执行性）
- LLM 失败 → 退化为拼接（fail-open，带 `### 子计划 i` 分隔标题，不阻塞）
- 单份子计划超 6000 字符截断（防 prompt 爆炸）
- 合并用父 agent 的 `llm_client`（避免子代理的轻量模型做要全局视角的合并活）

### feature flag 接入

**位置**：`tools/plan_mode_tool.py:handle_plan_mode_v2_dispatch` 开头

```python
from agent.feature_flags import is_feature_enabled, get_feature_config
if not is_feature_enabled(config, "plan_mode_v2_parallel"):
    return feature_disabled
flag_cfg = get_feature_config(config, "plan_mode_v2_parallel")
max_n = int(flag_cfg.get("max_parallel_agents", 3))
```

N 由 LLM 自决（决策 6）：`subtasks` 数组长度就是 LLM 想要的并发数，受 `max_parallel_agents` 钳制。

## 测试覆盖

**文件**：`tests/test_plan_mode.py`（追加 16 个 P6 测试，总 44 passed）

| 测试 | 覆盖场景 |
|---|---|
| `test_plan_mode_v2_dispatch_registered` | 工具注册到 plan toolset |
| `test_plan_mode_v2_dispatch_flag_off_returns_feature_disabled` | flag OFF 走原路径 |
| `test_plan_mode_v2_dispatch_empty_subtasks_invalid_args` | 参数校验 |
| `test_run_parallel_planners_n1_matches_single_behavior` | N=1 跟单 Agent 一致 |
| `test_run_parallel_planners_n3_concurrent_collects_all` | N=3 并发收集全 |
| `test_run_parallel_planners_max_n_caps_concurrency` | max_parallel_agents 上限生效（FakeExecutor 验证 workers=3） |
| `test_run_parallel_planners_subagent_failure_isolated` | 子代理失败兜底（B 崩 → A/C 保留） |
| `test_run_parallel_planners_empty_subtasks_returns_empty` | 空输入 |
| `test_merge_plans_single_returns_as_is` | 单份不调 LLM |
| `test_merge_plans_empty_returns_placeholder` | 空列表 |
| `test_merge_plans_multiple_calls_llm` | 多份调 LLM |
| `test_merge_plans_llm_failure_falls_back_to_concatenation` | LLM 失败退化拼接 |
| `test_plan_merge_prompt_template_format` | 模板可填充 |
| `test_plan_mode_v2_dispatch_full_flow_flag_on` | 端到端：flag ON + 2 子任务 + 合并 |
| `test_plan_mode_v2_dispatch_all_subagents_fail` | 全失败 → all_subagents_failed |

还更新了 2 个原有测试（`test_plan_toolset_defined` 和 `test_tool_concurrency_classification.py:test_all_tools_classified`）把 `plan_mode_v2_dispatch` 加进白名单。

## verify.py 结果

```
22/22 通过，0 失败
```

## 全量测试

`uv run pytest tests/ --tb=no -q`：**1624 passed, 1 skipped, 0 failed**（P6 后）。
- 相关模块单独跑：`test_plan_mode.py` 44 passed / `test_feature_flags.py + test_delegation.py + test_agent_defs.py` 56 passed

## 自检 + Concerns

### 自检
- [x] flag OFF 完全不影响原 plan mode 行为（handler 直接返回 feature_disabled，不进并行调度）
- [x] 默认 OFF（`config.py:282` 的 `enabled: False`），用户不显式开就不会触发
- [x] 工具 schema 恒定（进 plan toolset），不依赖 flag 切换可见性 —— 避免 prompt cache 失效
- [x] 合并用父 agent 的 LLM client（全局视角），不用子代理的轻量模型
- [x] 子代理失败兜底：用剩余结果合并，不整体崩
- [x] LLM 合并失败兜底：退化为拼接，不阻塞
- [x] 硬上限 `_ABS_MAX_PARALLEL=8` 防爆资源
- [x] KeyboardInterrupt 传播：cancel futures + `shutdown(wait=False, cancel_futures=True)`
- [x] `isConcurrencySafe=False`（串行）—— 起多个子代理 + 合并不应并发触发

### Concerns
1. **ThreadPoolExecutor vs asyncio.gather**：spec 原文写 `asyncio.gather`，但 `_run_child` 是同步的（内部 `asyncio.run`）。用 `ThreadPoolExecutor` 是更贴合现有 `_delegate_batch` 模式的选择，语义等价（N 个并行 + 等全完成）。如果后续 `_run_child` 改成原生 async，可以换 `asyncio.gather`。
2. **合并 LLM client 来源**：从 `kwargs["agent_ref"].llm_client` 拿。测试环境拿不到时（非标准调用路径）退化为拼接。这是 fail-open，不影响正确性。
3. **prompt cache**：`plan_mode_v2_dispatch` 在 plan toolset 里恒定可见（不随 flag 切换），flag 门控在 handler 层 —— 保护 prompt cache（设计原则 2）。
4. **未做真实 LLM 端到端**：所有测试都用 mock。真实多 Agent 并行需要 flag 开 + 真子代理跑，留 PD.1 手动验证阶段做。
