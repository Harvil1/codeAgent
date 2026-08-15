# CCAR12 Task 6 报告：Worktree 工具（会话级切换）

## 状态：完成

## 改动清单

| 文件 | 改动 |
|---|---|
| `agent/workspace_context.py` | 追加 `set_session_workspace_cwd(path)` / `clear_session_workspace_cwd()`（brief 原样代码）+ module-level `_session_token`。**并发局限已文档化**：① token 是模块级全局，只设计给主对话单线程使用（worktree_enter/exit 在主循环串行 dispatch）；② set/clear 必须同一 context（否则 ContextVar.reset 抛 "different Context"）；③ 子代理 `workspace_cwd_context` with 块不受影响（各自局部 token），但 enter 之后 spawn 的子代理**继承**会话 cwd（copy at spawn，符合"会话级切换"语义）；④ 重复 set 时旧 token 丢弃（视为切换，不恢复中间值） |
| `tools/worktree_tool.py` | 新建。`worktree_enter(name=None)` + `worktree_exit(keep=True)` 两工具 |
| `toolsets.py` | `_CORE_TOOLS` +2（worktree_enter / worktree_exit） |
| `tests/test_tool_concurrency_classification.py` | UNSAFE_TOOLS +2（expected_safe_count 16 不变） |
| `tests/test_worktree_tool.py` | 23 个测试（新增，`git add -f`） |

## 行为细节

- **worktree_enter(name)**：name 缺省 `wt-<时间戳>`；清洗非法字符（`[^A-Za-z0-9._-]` → `-` + 折叠 `..` + 去首尾 `.-`，防路径穿越/非法分支名——git 会拒绝 `..` 分支名，测试覆盖）。git 仓库 → `repo_root/.worktrees/<name>`（**存在则复用**，`reused: true`，复用时分支未知 → exit 不删分支）；新建走 `git worktree add -b omnimate/<name>/<short8>`（复用 CCAR5 `_create_git_worktree` 的命令形态）；git 失败或非 git → 降级系统临时目录（`workspace_type: "temp"`，CCAR5 语义）。成功后 `set_session_workspace_cwd` + CWD_CHANGED hook（`{session_id, old, new}`，照 delegate_tool 现有调用模式；agent_ref=None / 无 hooks_registry / 异常均 fail-open 跳过）。已在 worktree 中再 enter → `already_in_worktree` 错误（防状态覆盖泄漏）
- **worktree_exit(keep=True)**：keep 缺省 True（保守默认）。clear 恢复 cwd；keep=False 时 `has_worktree_changes`（CCAR5-G）检测——有改动保留 + `reason: "has_changes"` + hint，无改动 `cleanup_worktree_smart` 删目录 + 新建时一并 `git branch -D`（补 CCAR5-G 不删分支的分工：分支删除由有 branch 上下文的调用方负责）。不在 worktree 中 → `not_in_worktree`
- **会话状态**：`_SessionWorktree` module-level 单例（path/branch/workspace_type/reused），与 `_session_token` 同一并发边界；测试用 `_reset_session_worktree()` 清理
- **分类**：两工具 `isConcurrencySafe=False`（改会话级全局 cwd + 建/删 worktree，串行）+ core toolset
- **schema**：`"parameters"` 键（OpenAI 格式，CCAR11 契约测试覆盖）

## 测试

- 新增 `tests/test_worktree_tool.py`：23 个
  - workspace_context set/clear 3（切/恢复/幂等）
  - enter 8（建目录+切 cwd / 默认名 / **复用 reused** / already_in_worktree / name 清洗 / **CWD_CHANGED hook** / agent_ref=None 跳 hook / **非 git 降级 temp**）
  - exit 7（恢复 cwd / keep=True 保留 / keep 缺省 True / **keep=False 无改动清理+删分支** / **有改动保留** / not_in_worktree / temp 空目录清理）
  - 契约 2：handler `(args, **kwargs)` 签名 + schema "parameters" 键
  - 注册 2：registry core + isConcurrencySafe=False / `_CORE_TOOLS` 可见
- 全套 `uv run pytest tests/`：**2374 passed, 1 skipped**（无回归）
- `uv run python scripts/verify.py`：**22/22 PASS**

## Concerns（不阻塞）

1. **`.worktrees/` 在主 repo 树内会显示为 untracked**（`?? .worktrees/`）——CCAR5 delegate worktree 放 `repo_root.parent/.omnimate-worktrees`（repo 外）规避了这点；Task 6 按 brief 用 `repo_root/.worktrees/<name>`。与 worktree.py 事件流目录（`.worktrees/.events.jsonl`）同址。若嫌脏可建议用户 gitignore 或后续挪 repo 外
2. **asyncio 边界**：若未来 worktree_enter 在子 asyncio task 里 dispatch（context copy 语义），set 的效果不回透主循环 context。当前主循环工具 dispatch 同 task 调用，无问题；已文档化
3. **复用（reused）时不删分支**：复用的 worktree 可能是上次会话新建的（分支 `omnimate/<name>/<id>`），exit 清理目录后分支残留。改进方向：enter 复用时用 `git worktree list` 反查分支

---

## FIX（review Critical：to_thread context 拷贝，真实 dispatch 复现）

### 症状（reviewer 复现）

`registry.dispatch` 对 sync handler 走 `asyncio.to_thread`（**拷贝 context** 到 worker 线程）→ `_workspace_cwd.set()` 只改拷贝，不回透主循环：

1. **enter 静默失效**——工具自报成功但主循环 cwd 没切（恰是工具要防的污染主工作区）
2. `_session_worktree` 模块级全局置位后（线程间共享，会回写），再 enter 被 `already_in_worktree` 卡死
3. **exit 在拷贝 context 里 reset 主 token → ValueError → 永远 tool_exception**（状态机死锁）

### 修法

| 改动 | 内容 |
|---|---|
| `tools/worktree_tool.py` | 两个 handler 改 `async def`（dispatch 直接 await——同 task 同 context，set 生效 + token 同 context 可 reset；内部逻辑不变，worktree 创建/清理同步 IO 直接跑）+ 注册处 `is_async=True` + `_SessionWorktree` 存 `repo_root`（exit 删分支用它，不依赖恢复后的进程 cwd） |
| `tests/test_worktree_tool.py` | 原单元直调测试改 async/await + **新增 dispatch 端到端测试**（关键：单元直调绕过 to_thread，这就是漏检原因）`test_enter_exit_via_registry_dispatch`（dispatch 外、主 context 视角断言 cwd 真切/真恢复）+ `test_reenter_via_registry_dispatch_not_deadlocked`（enter→exit→再 enter 循环不被卡死）+ `test_handlers_are_async_not_threaded`（防改回 sync）——23→27 个 |
| `tests/test_tool_concurrency_classification.py` | 新增 `test_async_disallow_contains_worktree_enter` |
| `toolsets.py` | `ASYNC_AGENT_DISALLOWED_TOOLS` + `worktree_enter`（async 子代理污染模块级 `_session_worktree` 会把主对话卡 already_in_worktree；exit 留给自救——对齐 goal_start/goal_resume 处置） |
| `.gitignore` | + `.worktrees/`（清 Concerns #1） |

### 验证

- **测试前提验证**：临时注册 sync handler probe 走 dispatch → 主 context cwd 确实没切（`before == after`）——证实 to_thread 拷贝行为真实存在，端到端测试能捕获回归
- `uv run pytest tests/test_worktree_tool.py tests/test_tool_concurrency_classification.py -q`：**34 passed**
- 全套 `uv run pytest tests/`：**2378 passed, 1 skipped**（baseline 2374 + 4 新测试，0 failed）

### 教训（第 6 次"单元直调绕过 dispatch"）

handler 的 context 副作用（ContextVar set/reset）只有在真实 dispatch 路径才按生产方式执行——单元直调在测试自己的 context 里跑，set/reset 天然同 context 永远过。凡 handler 依赖 contextvars，必须有 `registry.dispatch` 端到端测试。
