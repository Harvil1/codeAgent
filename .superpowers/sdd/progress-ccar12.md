# SDD Progress Ledger — CCAR12

Plan: `docs/superpowers/plans/2026-08-15-ccar12-midvalue.md`
Spec: `docs/superpowers/specs/2026-08-15-ccar12-midvalue-design.md`
Started: 2026-08-15
Base commit: c984ca40
Branch: cli-dev

## 决策（用户已 ack）
- Windows 沙箱 = Job Objects 进程管控（诚实定位，文件防线靠既有层）
- Worktree = 会话级切换（长效 set ContextVar）
- config_set 白名单 7 键精确匹配

## Tasks
- [x] Task 1: win_job_object.py
- [x] Task 2: sandbox_runner + terminal 接入
- [x] Task 3: cron_tool
- [x] Task 4: goal_tool（共享函数抽取）
- [ ] Task 5: MCP Resources
- [ ] Task 6: worktree_tool（会话级切换）
- [ ] Task 7: config_tool
- [ ] Task 8: 收尾

## Completion Log
- Task 1: complete (commits c984ca40..105c5c4e, review Approved) — win_job_object.py（ctypes restype 显式防 x64 句柄问题 + 结构体 ABI 逐字段核过 + kill-on-close 真实进程验证）；4 测试；spec ✅ + 质量 Approved。

## deferred Minor findings
- Task 1 (Minor): 报告"GC 误杀"表述反了（裸 int 不会被 GC 关；真风险是句柄泄漏不清理）；OpenProcess 失败未 log last_error；_configure 失败仅 warning（KILL_ON_JOB_CLOSE 没设上 close 静默不清理）
- Task 2: complete (commits 105c5c4e..feb2f4db, review Approved) — sandbox 接入（Windows Popen 后 attach_job + finally 保活 close + 事件顺序硬断言 + argv 测试补 patch 防真机劫持）；偏离 availability_reason 契约保留（新增 sandbox_description）reviewer 判合理；2276 全绿。
- Task 2 follow-up: ① hook 沙箱（hook_exec._wrap_with_sandbox）接 Job Object（另开任务）；② job=None+超时时 shell 进程无人 kill（罕见降级态，低优先级）
- Task 3: complete (commits feb2f4db..16bfa18d, review Approved) — cron_tool 三工具 + CronScheduler 补 CRUD（add/remove/list 持锁 persist）；agent_ref.cron_scheduler 已存在零接线；ASYNC_AGENT_DISALLOWED_TOOLS 落实；24 新测试；spec ✅ + 质量 Approved。
- Task 3 (Minor): 报告测试数 36 实为 24；cron_list 只读可后续移 SAFE（改两处）
- Task 3 (Minor): _persist 失败 job 内存态仍返回成功（fail-open 一致，重启丢任务）
- Task 4: complete (commits 16bfa18d..c3eb26b3) — goal_tool 五工具 + start_goal_agent 共享函数抽取（CLI /goal 同源）；关键决策：共享函数不碰 conversation_history（工具路径插 user 消息破坏 tool 交替，驱动靠 goal-continue 分支）；goal_status SAFE（15→16）；30 新测试；全套 2329 passed。待 review。
- Task 4 concern: goal_start/goal_clear 未进 ASYNC_AGENT_DISALLOWED_TOOLS（后台子代理可在 daemon 线程激活 goal 循环烧 token，brief 未点名；reviewer 定夺）
