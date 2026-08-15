# CCAR12 Task 4 Report: Goal 工具化（含共享函数抽取）

> 注：本文件路径之前承载过 CCAR8/9/10/11 各轮报告，本份覆盖为 CCAR12 Task 4。

**Branch:** cli-dev
**Date:** 2026-08-15
**Status:** 完成

## 任务范围

1. `agent/goal.py` 新增共享函数 `start_goal_agent(agent, objective, token_budget, persist_path=None)` + 路径解析辅助 `goal_persist_path(agent)`——把 cli.py `_start_new_goal` 的核心三步（旧 goal pause / GoalState 构造挂 agent / 持久化）抽来，CLI `/goal` 与 LLM `goal_start` 工具同源。
2. `cli.py:_start_new_goal` 改调共享函数（行为不变，cli 层测试未动仍绿）。
3. `tools/goal_tool.py` 五工具：`goal_start` / `goal_status` / `goal_pause` / `goal_resume` / `goal_clear`（schema 全 "parameters" 键）。
4. 分类：UNSAFE +4（start/pause/resume/clear）、`goal_status` SAFE（expected_safe_count 15→16）；`_CORE_TOOLS` +5。
5. `tests/test_goal_tool.py` 30 个新测试。

## 实施步骤

### Step 1：读现状（brief 点名先 Read `_start_new_goal`）

- cli.py `_start_new_goal`（原 1873 行起）四步：① pause 旧 active goal + save ② 建新 GoalState + save + `set_goal_state` ③ aux_llm 拆解（asyncio.run + loop-running guard，fail-open）④ `[goal_start]` user 消息塞 `conversation_history` 末尾。全程穿插 console.print。
- AIAgent 已有 `_goal_state_path()`（`omnimate_home/.goal/current.json`）和 `set_goal_state`——共享函数直接复用。
- CCAR8 mailbox 模式：handler 从 `dispatch_kwargs["agent_ref"]` 拿 agent。

### Step 2：关键设计决策——共享函数**不碰 conversation_history**

原 cli 第 ④ 步不能进共享函数：CLI 在会话循环外追加 user 消息是安全的，但**工具路径**在 `_dispatch_tool_calls` 里 `assistant(tool_calls)` 已 append（行 1822）、tool result 由 `_merge_results_in_order` 事后回填——中间插 user 消息会破坏消息历史严格交替（API 400，CCAR5-6 修过的同款 bug）。工具路径也不需要它：goal 激活后主循环 goal-continue 分支（`run_conversation` 行 1126）自动多轮驱动。专门写了回归测试 `test_does_not_touch_conversation_history`。

### Step 3：共享函数（agent/goal.py）

- `start_goal_agent`：persist_path 显式参数优先（CLI 传 `_goal_state_path(rt)`，保证测试 FakeRT 的 tmp_path 隔离）→ `agent._goal_state_path()` → `agent.omnimate_home` → `get_omnimate_home()`（`goal_persist_path` 三级 fallback）。旧 goal 仅 active 才 pause（已 paused 的不覆盖 reason、不追加 notes）。`set_goal_state` 缺失时直接赋 `_goal_state` 属性（mock 兼容）。
- 签名比 brief 多了 `persist_path=None` 可选参数——默认三参调用即 brief 契约；加它是为了 CLI 路径的 `rt.home`（测试 tmp_path）不被 fallback 写到真实 `~/.OmniMate`。

### Step 4：cli.py 重构

`_start_new_goal` = 共享函数（核心三步）+ CLI-only 三件事：① 打印"已自动 pause 旧 goal"（调共享函数前先记 `will_pause_old`）② aux_llm 拆解（原样保留，含 loop-running guard）③ `[goal_start]` history 注入（原样保留，注释说明为何只能 CLI 做）。`tests/test_cli_commands.py` 的 6 个 /goal 测试**未改一行**仍绿。

### Step 5：tools/goal_tool.py（对齐 cron_tool 模式）

- 5 handler 全 `(args, **dispatch_kwargs)` 契约，返回 JSON 字符串，`ensure_ascii=False`。
- 错误分支：无 agent_ref → `not_configured`；缺 objective / 非法 budget → `invalid_args`；pause/resume/clear 无 goal → `no_active_goal`；异常兜底 `{"error", "error_type": type(e).__name__}`。
- `goal_start` 校验 token_budget 为正整数；`goal_status` 无 goal 返回 `{"status": "none"}`（含 task_count）；`goal_clear` = cancel + save + 摘挂 + unlink 持久化文件（对齐 CLI `/goal clear`）。
- 注册 5 个（toolset="core"，emoji 🎯）：`goal_status` isConcurrencySafe=True，其余 4 个 False。

### Step 6：分类 + 可见性

- `toolsets.py:_CORE_TOOLS` +5（发现 ≠ 可见）。
- `tests/test_tool_concurrency_classification.py`：SAFE_TOOLS + `goal_status`（expected_safe_count 15→16）、UNSAFE_TOOLS + 4 个写操作；注释同步。

## 验证

- 新测试：`uv run pytest tests/test_goal_tool.py -q` → 30 passed（TDD：先写测试确认 import 失败 red，再实现转 green）
- 定向回归：`test_goal_tool + test_tool_concurrency_classification + test_cli_commands + test_goal` → 114 passed
- 全套：`uv run pytest tests/ -q` → **2329 passed / 1 skipped / 0 failed**（Task 3 后 baseline 2299 passed + 1 skipped，净 +30）

## Concerns / Follow-up

1. **goal 工具未进 ASYNC_AGENT_DISALLOWED_TOOLS**：后台 async 子代理调 `goal_start` 会激活自己实例的 goal 循环（默认 200K budget），在 daemon 线程里持续烧 token。当前按 brief 未加（brief/plan 只点名分类两处同步）；若 reviewer 认同风险，可把 `goal_start`/`goal_clear` 加进 disallow 集合（对齐 cron_create/cron_delete 先例）。
2. **工具路径与 CLI 启动行为的两处差异**：① `goal_start` 工具不做 aux_llm 拆解（decompose 的 asyncio.run + loop guard 是 CLI sync 路径特有的；LLM 需要子任务可用 `task_create` 自建）；② 工具路径不注入 `[goal_start]` user 消息（见 Step 2），驱动完全靠 goal-continue 分支。无 task_ids → `evaluate_after_turn(all_tasks_done)` 恒 False，goal 不会自动 complete（budget 耗尽 pause 或 LLM 主动 `goal_clear`）。
3. **`persist_path` 可选参数是 brief 之外的扩展**（默认调用仍满足 brief 三参契约），reason 见 Step 3。
