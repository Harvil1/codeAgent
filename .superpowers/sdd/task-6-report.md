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
