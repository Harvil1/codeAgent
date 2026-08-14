# CCAR10 Task 5 报告：/resumable CLI 命令

**状态**：DONE
**Commit**：`1d9374c3 feat(cli): /resumable 列出/恢复可续跑子代理（CCAR10 Task 5）`
**测试**：18 passed（Task 4 原 8 + Task 5 新增 10）
**全套**：2216 passed（1 flaky stress test 预先存在，与本次改动无关）
**verify**：22/22 PASS

## 任务完成项

### 主任务：/resumable CLI 命令

- 实现 `_handle_resumable_command(args, rt)` in `cli.py`
  - **无参（列表）**：列出所有 status=running 的子代理（agent_id / agent_type / status / 消息数 / 时间）+ 提示用法 + 语义边界说明（"只有存了 transcript 的子代理可恢复，真正中断可能没有轨迹"）
  - **有参（恢复）**：调 `_run_resume(agent_id, instruction, agent_ref=..., config=..., memory_store=...)`，error 红色提示 / 成功绿色显示结果前 2000 字符（超长加截断标记）
  - 支持用户在 agent_id 后附带自定义续命指令（`/resumable <id> <instruction...>`）
- 接入 `_handle_command` 命令链（CCAR8 命令链模式，与 `/poor` / `/trace` 同款）
- 加 `/resumable` 到 `/help` 输出

### Task 4 follow-up 修复（顺手）

1. **`_spawn_resumed_agent` 透传 `memory_store`**
   - 新增 `memory_store` 关键字参数（None 默认，对齐 `_run_child` 模式）
   - AIAgent 构造传 `memory_store=memory_store`（让续跑子代理复用父记忆库）
2. **`_run_resume` 从 `agent_ref.memory_store` 提取**
   - 优先用 dispatch_kwargs 里的 memory_store，其次从 agent_ref 取
3. **测试隔离同款**：复用 Task 4 的 `isolated_sessions` fixture（monkeypatch `sp._sessions_dir`），Task 5 测试全部走这个 fixture

## 关键设计决策

### API 决策：handler 不接 `base_dir`
subagent_persistence 的 `list_resumable()` / `write_metadata()` / `load_transcript()` 等所有函数**不接 base_dir 参数**（走模块级 `_sessions_dir()`）。这与 Task 4 brief 描述的"带 base_dir"不一致，但实际接口如此。Task 5 的 handler 也保持不带 base_dir——隔离靠 monkeypatch `_sessions_dir`，与 Task 4 测试策略一致。Task 4 brief 写的"带 base_dir"是 spec 阶段的设想，实现时没加这个参数（也对齐 CCAR5-I 既有约定），不需要在 Task 5 改这个接口。

### Rich markup 转义坑（修过）
初版 handler 用 `f"  [{aid}] ..."` 输出 agent_id，Rich 把 `[sa_a001]` 解析成 style tag，静默吃掉了——列表输出完全看不到 agent_id！改用 `\\[{aid}]` 转义字面方括号。**教训**：任何动态内容要放 Rich `[xxx]` 里，必须 `\[` 转义。

### 测试断言避开 Rich 软换行干扰
长结果（5000 字符）测试用 `"Z" * 2000 in captured.out` 失败——Rich console 会软换行，把 Z 字符打散到多行。改用"数 Z 个数 == 2000"（不依赖连续性）+ 断言"截断提示出现"。

## 测试清单（10 新增）

1. `test_resumable_cli_list_empty` — 空列表分支提示"无可恢复"
2. `test_resumable_cli_list_with_items` — 两条 running 记录都显示 + 用法提示
3. `test_resumable_cli_list_filters_out_terminal_status` — completed/interrupted 不显示
4. `test_resumable_cli_list_notes_semantic_boundary` — 列表提示含语义边界说明
5. `test_resumable_cli_resume_success` — 成功路径绿色显示结果
6. `test_resumable_cli_resume_long_result_truncated` — 结果超 2000 字符截断
7. `test_resumable_cli_resume_error_red_output` — error JSON 红色提示
8. `test_resumable_cli_dispatched_from_handle_command` — `/resumable` 命令链分发
9. `test_spawn_resumed_agent_passes_memory_store` — spawn 透传 memory_store 给 AIAgent
10. `test_run_resume_passes_memory_store_from_agent_ref` — `_run_resume` 从 agent_ref.memory_store 提取并透传

## 修改文件

| 文件 | 改动 |
|---|---|
| `cli.py` | 新增 `_handle_resumable_command` + 命令链接入 + `/help` 加行 |
| `tools/subagent_resume_tool.py` | follow-up：`_spawn_resumed_agent` 加 `memory_store` 参数；`_run_resume` 从 agent_ref 取 memory_store 透传 |
| `tests/test_subagent_resume.py` | 新增 10 个测试 |

## Concerns / Follow-up

1. **列表消息数靠现场读 transcript**：meta 没存 message_count，每次列表要逐条 `load_transcript` 数（O(n × transcript_size)）。当前规模（少量子代理）无问题；规模大了可考虑在 meta 里缓存 message_count（append_message 时递增）。
2. **`/resumable <id> <instruction>` 格式**：用空格切分，instruction 可能含空格——已用 `" ".join(parts[1:])` 合并后续 token 兼容。
3. **flaky stress test**：`test_stress_skill_index_300_skills` 全套跑时挂（单独跑过），与本次改动无关，疑似其他测试污染了 `~/.OmniMate`。
