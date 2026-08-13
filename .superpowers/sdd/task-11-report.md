# Task 11 报告：Goal 主循环集成 + 网络 pause

## 状态
**完成** — 30/30 测试 PASS（17 新 + 13 已有 goal 测试），全套 2082 passed / 1 skipped / 0 failed，verify 22/22 PASS。

## Commit
（待 commit）

## 1. ephemeral 机制实现方式

**复用现有 + 最小扩展**：

| 层 | 已有 | 新增 |
|---|---|---|
| 注入位置 | `_assemble_turn_messages` 末尾追加（bg/cron/team/delegation/post_compact_brief 已用） | 加 channel/mailbox/pending_ephemeral 注入点 |
| 持久化隔离 | `cli.py` 显式调 `session_store.append_message`（与 conversation_history 解耦） | 不变——ephemeral 进 history 内存，但 cli.py 不持久化 |
| 内部字段 strip | `strip_internal_fields` 去 `_timestamp` | 加 `_ephemeral` 到 `INTERNAL_KEYS`（防 LLM 看到） |
| 压缩时 strip | `_run_context_compression` / `reactive_compact` 用 `messages[1:]` 全量赋 history | 改为 `[m for m in ... if not m.get("_ephemeral")]` |

**关键设计：3 个 helper 纯函数**（可独立测试，agent/__init__.py 模块级）：
- `_build_goal_continue_message(goal_state)` → `<continue_goal>` ephemeral dict 或 None
- `_build_channel_injection(inbox)` → `<channel_push>` ephemeral dict 或 None（含 mark_consumed）
- `_build_mail_injection(mailbox, agent_name)` → `<mail>` ephemeral dict 或 None（含 mark_read）

**跨轮 ephemeral 暂存**：
- 主循环 goal continue 时把 ephemeral 塞 `self._pending_ephemeral_messages`（不进 history）
- 下一轮 `_assemble_turn_messages` 末尾消费并清空（让 LLM 看到）
- 这样 ephemeral 消息永远不进 `conversation_history`，但 LLM 能看到

## 2. 三个注入点（行号基于 commit 时版本）

| 注入点 | 文件:行 | 说明 |
|---|---|---|
| **goal continue** | `agent/__init__.py` 主循环最终响应分支末尾 | evaluate_after_turn 决策后塞 `_pending_ephemeral_messages` |
| **channel/mailbox** | `agent/__init__.py` `_assemble_turn_messages` 开头（history 之后、bg/cron 之前） | fail-open 注入 |
| **pending ephemeral 消费** | `agent/__init__.py` `_assemble_turn_messages` 末尾 | 消费 `_pending_ephemeral_messages` 并清空 |
| **network pause** | `agent/__init__.py` `_call_llm_with_escalation` 异常分支 | 关键词匹配后调 `goal.pause(reason="network")` |

## 3. prompt cache 保护（铁律遵守）

| 风险 | 防护措施 | 测试覆盖 |
|---|---|---|
| 修改 system prompt | 所有 ephemeral 走 user 角色；setter 不动 prompt_builder | `test_goal_continue_does_not_modify_system_prompt` |
| ephemeral 进持久化 history | `_pending_ephemeral_messages` 暂存，`_assemble_turn_messages` 末尾消费清空 | `test_goal_continue_message_not_in_persisted_history`（跑 200 轮触发压缩验证） |
| 压缩把 ephemeral 写回 history | `_run_context_compression` + `reactive_compact` 用列表推导过滤 `_ephemeral` | 间接覆盖（同上测试） |
| LLM 收到 `_ephemeral` 字段报 400 | `strip_internal_fields` 加 `_ephemeral` 到 INTERNAL_KEYS | helper 测试 + 集成测试验证 |

## 4. 测试覆盖

### 纯函数测试（7 个）
- `test_build_goal_continue_message_basic` — goal active 时构造正确 ephemeral dict
- `test_build_goal_continue_message_none_returns_none`
- `test_build_channel_injection_with_unconsumed` — 有消息时构造 + mark_consumed
- `test_build_channel_injection_empty_returns_none` / `_none_returns_none`
- `test_build_mail_injection_with_unread` — 有未读时构造 + mark_read
- `test_build_mail_injection_empty_returns_none` / `_none_returns_none`

### AIAgent 字段接线测试（3 个）
- `test_ai_agent_has_goal_state_field` — 构造默认 None + setter 生效
- `test_ai_agent_has_channel_inbox_field`
- `test_ai_agent_has_mailbox_field`（含 `_agent_name` 默认 "main"）

### 集成测试（5 个，跑真 AIAgent + mock LLM）
- `test_goal_continue_does_not_modify_system_prompt` — 关键：system prompt 在 goal active 时不变
- `test_goal_continue_message_not_in_persisted_history` — 关键：`<continue_goal>` 不进 conversation_history（200 轮跑完触发压缩验证）
- `test_channel_injection_in_assemble_turn_messages` — `_assemble_turn_messages` 注入 `<channel_push>`，不进 history
- `test_mail_injection_in_assemble_turn_messages` — 同上 `<mail>`
- `test_goal_auto_pause_on_network_error` — ConnectionError 触发 goal pause(reason="network")
- `test_goal_not_paused_on_non_network_error` — 非网络异常不 pause（防误触发）

## 5. 关键约束遵守

- [x] **prompt cache 保护**：所有 ephemeral 走 user 角色，不动 system prompt（关键测试 verify）
- [x] **ephemeral 不进持久化**：用 `_pending_ephemeral_messages` 暂存 + 压缩时过滤
- [x] **fail-open**：channel/mailbox 注入失败只 log warning（helper 内 try/except）
- [x] **现有测试全 PASS**：2082 passed / 1 skipped / 0 failed
- [x] **中文注释/commit**：所有新代码注释中文
- [x] **TDD**：先写 17 个失败测试 → 实现 → 全 PASS
- [x] **最小 invasive**：主循环只加约 39 行（goal continue 分支），不改既有逻辑

## 6. 文件改动清单

| 文件 | 改动 |
|---|---|
| `agent/__init__.py` | +4 构造参数 / +5 字段 / +3 setter / +1 helper（_extract_turn_tokens）/ +3 模块级 helper / +1 goal continue 分支 / +2 channel/mailbox 注入 / +1 pending ephemeral 消费 / +1 网络 pause 分支 / 2 处压缩 strip ephemeral |
| `agent/context_pipeline.py` | `strip_internal_fields` 加 `_ephemeral` 到 INTERNAL_KEYS |
| `tests/test_goal.py` | +17 新测试（含 7 纯函数 + 3 接线 + 5 集成 + 2 网络 pause） |
| `.superpowers/sdd/task-11-report.md` | 本报告 |

## 7. Concerns / Follow-up

1. **Task 12 集成预留**：`_check_all_goal_tasks_done` 当前返回 False（无 TaskStore 接入），Task 12 会 replace 为 `self.goal.check_all_tasks_done(self._goal_state)`
2. **goal continue 无 iteration 上限**：当前依赖 `max_iterations`（200）和 `token_budget_limit`（None=不限）。CLI `/goal` 命令（Task 12）应强制设 `token_budget_limit`，避免烧 token
3. **goal CLI 接入未做**：本 task 只做主循环集成，`/goal` slash 命令在 Task 12
4. **AIAgent 构造参数变多**：已 30+ 参数，未来可考虑用 kwargs dict 或 builder pattern（不在本 task 范围）
