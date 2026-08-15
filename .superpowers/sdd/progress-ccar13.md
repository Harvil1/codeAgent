# SDD Progress Ledger — CCAR13

Plan: `docs/superpowers/plans/2026-08-15-ccar13-leftovers.md`
Spec: `docs/superpowers/specs/2026-08-15-ccar13-leftovers-design.md`
Started: 2026-08-15
Base commit: 7afa1b31
Branch: cli-dev

## Tasks
- [ ] Task 1: 快修四件（A1-A4）
- [ ] Task 2: 中件两件（pause 通知集中 / bg title）
- [ ] Task 3: subagent 完整轨迹
- [ ] Task 4: check_path 白名单（complete，见 Completion Log）
- [ ] Task 5: 收尾

## Completion Log
- Task 4: complete (commit 见 task-4-report.md) — check_path 闸门 3 恢复白名单语义（收回 dcec556b "其他全通过"）：write 走 default_allowed_roots()（workspace cwd + ~/.OmniMate + /add-dir extra roots，与 safe_path 同源），之外拒；闸门 1 受保护路径 / 闸门 2 项目代码写保护在前（加 home 根也写不了 ~/.ssh）；bypassPermissions 跳白名单不跳硬底线；write_file/str_replace 共用（改 permission 层一处，file_operations 零改动）；5 新测试 + 4 个旧测试补 extra root；全套 2438 passed / verify 22/22。
- Task 1: complete (commits 7afa1b31..88ae96d8, review Approved) — 快修四件（A1 spawn_depth 守卫 isinstance 防御 / A2 job=None 超时 kill / A3 reactive 提示 / A4 核对 7 键全有读取点无 dead key + 锁定测试）；6 新测试 + 2420 全绿；spec ✅ + 质量 Approved（A4 抽查 3 键全真）。
- Task 2: complete (commits 88ae96d8..c13fd037, review Approved) — B5 pause 通知集中 GoalState.pause()（三原因一处接 + 散接删防双发 + fail-open 不炸状态机）+ B6 bg title 带 task_id 前 8 位（不互吞 + 真实接线测试）；4 新测试 + 2424 全绿；spec ✅ + 质量 Approved。

## deferred Minor findings
- Task 2 (Minor): test_notifier.py 两个 CCAR11 旧"复刻接线"测试已不反映真实行为（留清理，真路径有新测试覆盖）
- Task 3: complete (commits c13fd037..b1a01b84, review Approved) — subagent 完整轨迹（独立 HookRegistry + POST_LLM_CALL 每轮 append + user 指令补记 + on_response 删防双写 + **轨迹永不带 tool_calls** 从源头防 API 400 孤儿）；8 新测试 + 2432 全绿；spec ✅ + 质量 Approved。CCAR5-I Phase 2 闭环：真正中断的子代理有轨迹可 resume。
- Task 3 (Minor): 轨迹不含工具调用概要（resume 知道"说过什么"不知道"做过什么"——未来增强）；连续 assistant 消息 resume 兼容（OpenAI 端验证过，止血点 _run_resume 合并）
- Task 3 (Minor): hooks.enabled 关闭时轨迹只剩 user 一条（文档化，on_turn 退化方案留 follow-up）
