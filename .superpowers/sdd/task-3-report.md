# CCAR13 Task 3 报告：subagent transcript 每轮 append（完整轨迹）

**Status**: DONE
**Date**: 2026-08-15

## 机制选择（第一步 grep 结论 + 理由）

| grep 项 | 结论 |
|---|---|
| `on_response` 触发点 | `agent/__init__.py:2112`，仅 `run_conversation` 末尾一次，只拿 `final_content` |
| `POST_LLM_CALL` payload | `agent/__init__.py:1083`（`_run_post_llm_call_hook`），主循环**每次 LLM 调用后**触发（流式/非流式汇合点之后），hook 收到完整 response 对象（`response.choices[0].message`） |
| `_run_child` 的 `hooks_registry` | **未传** → 子代理拿 `None`。**不共享主 agent**（无污染风险） |
| `mark_completed` 是否依赖 on_response | **不依赖**——`_run_child` try/finally 直接调（成功 `completed` / 异常 `failed`），删 on_response append 不破 completed 语义 |

**选择：POST_LLM_CALL 方案（brief 首选）**。理由：

1. grep 确认子代理 hooks_registry 不共享主 agent → brief 决策树允许注册；
2. `HookRegistry()` 构造是纯内存的（声明式 hook 由 cli.py 的 hook_loader 加载，registry 本身不自动扫盘）→ 新建空实例零副作用；
3. 不改 AIAgent 签名，复用既有扩展点（对齐"核心是窄腰"）。

**已知耦合（写进 CLAUDE.md 已知约束）**：走 `AIAgent._run_post_llm_call_hook`，受 `config["hooks"]["enabled"]` 门控（默认 True）。用户显式关 hooks 时轮级记录停摆（transcript 只剩开头 user 指令一条）。评估：hooks.enabled=False 本义是关 hook 系统，可接受；如未来要解耦，退化方案是 AIAgent 加 `on_turn` 参数（本次未做，避免不必要的核心改动）。

## 实现

### `tools/delegate_tool.py`（_run_child）

1. **user 指令开头 append（新增）**：transcript 初始化（write_metadata）后立刻 append `{"role":"user","content": goal 或 goal+"\n上下文: "+context}`。注：brief 说"既有"，实际 grep 发现旧版**没有** user append（transcript 只有一条最终 assistant 响应）——本轮补上，resume 才有对话起点。
2. **轮级 hook（替换 on_response）**：构造子代理前新建 `HookRegistry()` + 注册程序式 `POST_LLM_CALL` hook `_on_llm_turn`：
   - `_extract_turn_text`：None 安全提取 assistant 文本；Anthropic 风格 content blocks（list）只拼 `type=="text"` 块
   - 有文本才 append `{"role":"assistant","content":text,"_ts":...}`，fail-open（异常吞掉），返回 `None` 不修改 response
3. **子代理构造**：`on_response=_on_response_cb` → `hooks_registry=_child_hooks`（独立空实例）
4. **删除** `_on_response_cb`（每轮已记最终轮，避免双写）

### 关键语义决策：轨迹永不带 tool_calls

POST_LLM_CALL 拿得到 LLM 响应但拿不到 tool result。若轨迹记录带 tool_calls 的 assistant 消息，resume 时 `initial_messages` 会出现孤儿 tool_calls（无配对 tool result）→ API 400（项目史上踩过）。因此轮级记录**只存文本 content**，resume 的 initial_messages 是纯 user/assistant 文本流，配对天然完整。`subagent_resume_tool._run_resume` 的 clean 字段过滤（含 tool_calls）天然兼容——轨迹里根本没有该键。

### 文档同步

- `agent/subagent_persistence.py` 模块 docstring：删"⚠️ 只落盘最终响应 / Phase 2 计划"，改写每轮语义
- `CLAUDE.md`：更新 CCAR5-I 关键代码位置行 + CCAR10"轨迹边界"约束行（已过时的"中断代理无轨迹"改为新语义）

## 测试（tests/test_subagent_persistence.py，TDD 先红后绿）

新增 `TestPerTurnTranscript`（8 个）+ 公共 helper `_spawn_mock_child` / `_llm_resp`：

| 测试 | 断言 |
|---|---|
| test_transcript_records_multiple_turns | 2 轮 LLM 响应 → transcript ≥3 条（user + 2 assistant），顺序/内容正确 |
| test_user_directive_with_context | 带 context 时 user 指令含上下文 |
| test_tool_calls_only_turn_not_appended | content=None/"" 的纯 tool_calls 轮不 append；全轨迹无 tool_calls 键 |
| test_anthropic_list_content_blocks | content blocks 只拼 text 块 |
| test_hook_fail_open_on_append_error | append 抛异常 → hook 不崩、response 原样返回 |
| test_malformed_response_fail_open | choices 空/None/怪对象安全跳过 |
| test_final_response_no_double_write | on_response 已删；最终轮文本只出现 1 次 |
| test_hooks_registry_not_shared_with_parent | 独立 HookRegistry 实例，仅 1 个 POST_LLM_CALL hook |

改写 2 个既有测试：

- `test_transcript_persisted_on_success`：断言 on_response → 改为 hooks_registry 非 None + on_response 为空
- `test_on_response_appends_transcript`：改写为轮级版（拿 ctor kwargs 的 hooks_registry 模拟 2 轮 → 3 条）

## 验证

- `tests/test_subagent_persistence.py` + `tests/test_subagent_resume.py`：51 passed（resume 闭环：完整轨迹作 initial_messages 跑绿，completed 标记语义不破）
- 定向：delegation + hooks + persistence + resume = 132 passed
- **全套：2432 passed, 1 skipped（207s）**

## Concerns

1. **hooks.enabled 耦合**（上述，已写进 CLAUDE.md 约束；如需彻底解耦留 AIAgent.on_turn 退化方案）
2. **给子代理传 registry 的表面变化**：子代理从此 hooks_registry 非 None——空 registry 对其余 26 种事件全是 no-op；`auto_heartbeat` 会注册但其 OMNIMATE_KANBAN_TASK env 门控不满足时 no-op；trace/声明式 hook 均不加载。全套测试无回归佐证。
3. **多轮连续 assistant 消息**：resume 时 initial_messages 可能出现连续 assistant（中间轮无 user/tool 间隔），OpenAI 兼容 API 允许；Anthropic 格式也未报错（resume 测试全绿）。如未来 API 挑剔，可在 _run_resume 里合并相邻 assistant。
