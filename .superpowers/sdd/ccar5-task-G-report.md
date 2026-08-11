# Task G Report: worktree 变更检测 + 智能清理

## What was implemented

借鉴 Claude Code `hasWorktreeChanges` 检测机制：子代理在 `isolated_workspace=True` 时写的文件不再被 cleanup 丢失。

### 核心改动

1. **`has_worktree_changes(worktree_path)`** — 检测 worktree 是否有改动
   - git 目录：`git status --porcelain` 返回非空 = 有改动
   - 非 git 目录：`listdir()` 有文件 = 有改动（temp workspace 创建时是空的）
   - fail-open：异常返回 True（保守，避免误删）

2. **`cleanup_worktree_smart(worktree_path, force=False)`** — 独立智能清理函数
   - 有改动且 `force=False` → 保留（返回 False）
   - 无改动或 `force=True` → 清理（返回 True）
   - git 目录走 `git worktree remove` + `git branch -D`，非 git 走 `shutil.rmtree`

3. **`create_isolated_workspace` cleanup 闭包升级** — `_create_git_worktree` 和 `_create_temp_workspace` 的 cleanup 加了 `force` 参数
   - `cleanup()` 默认智能（有改动保留）
   - `cleanup(force=True)` 强制清理（旧行为）
   - `cleanup(keep=True)` 保留（向后兼容）
   - 返回值：True=已清理 / False=保留

4. **`_run_child` finally 块用智能 cleanup** — `delegate_tool.py:_run_child`
   - 默认走智能清理（有改动保留 + 日志记录路径）
   - `config.delegation.worktree_always_cleanup=True` 回到旧行为

5. **config 开关** — `config.py` 的 delegation 节加 `worktree_always_cleanup: False`

## TDD Evidence

```
RED:  ImportError: cannot import name 'has_worktree_changes' from 'tools.worktree'
GREEN: 17 passed
```

## has_worktree_changes 如何处理 git 和非 git

| 场景 | 检测方式 | 有改动 | 无改动 | 异常 |
|------|---------|--------|--------|------|
| git worktree | `git status --porcelain` | True | False | True (fail-open) |
| 非 git temp | `Path.iterdir()` | True (有文件) | False (空目录) | True (fail-open) |

## 端到端测试覆盖

- `test_e2e_delegate_with_file_changes_preserves_worktree` — 子代理写文件 → worktree 保留
- `test_e2e_delegate_no_changes_cleans_worktree` — 子代理不写 → worktree 清理
- `test_e2e_worktree_preserved_in_result` — mock 路径不崩

## 路径翻译提示

**未加**（Task brief 标注为可选）。原因：
1. 注入 user 消息可能影响 LLM 行为（不可预测）
2. 智能清理已解决核心问题（worktree 保留 = 用户能找到文件）
3. 保持改动最小化，降低风险

如需加可在后续迭代补：在 `_run_child` 的 `workspace_path` 非 None 时注入 `[WORKTREE NOTICE]` 消息。

## Files changed

| 文件 | 改动 |
|------|------|
| `tools/worktree.py` | +147 行：has_worktree_changes + cleanup_worktree_smart + cleanup 闭包加 force 参数 |
| `tools/delegate_tool.py` | +17 -1 行：_run_child finally 用智能 cleanup + config 开关 |
| `config.py` | +3 行：delegation.worktree_always_cleanup=False |
| `tests/test_worktree_changes.py` | 新建：17 个测试 |
| `tests/test_worktree.py` | 1 行改：test_temp_workspace_via_helper 加 force=True（适配智能清理） |

## Commits

- `54e7b363` feat(worktree): Task G worktree 变更检测 + 智能清理

## Verification

- `tests/test_worktree_changes.py`: 17/17 passed
- `tests/test_worktree.py + test_delegation.py`: 36/36 passed (回归)
- 全套: 1819 passed / 1 skipped
- `scripts/verify.py`: 22/22 ALL PASS

## Concerns

无重大问题。小注意点：
- 非 git temp 目录的检测用 iterdir（有文件 = 有改动），比 git status 粗粒度——但 temp workspace 创建时是空的，所以语义正确
- `cleanup_worktree_smart` 独立函数用 `git branch --list omnimate/*` 推断分支删除——如果未来分支命名变了要更新
