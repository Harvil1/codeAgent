# CCAR13 Task 4 报告：check_path 感知 extra_allowed_roots（C8）

> 注：本文件路径之前承载过 CCAR8-12 各轮报告，本份覆盖为 CCAR13 Task 4。

**Branch:** cli-dev
**Date:** 2026-08-15
**Status:** 完成

## 改动落点：permission 层（`agent/permission.py:PermissionChecker.check_path`），不是 file_operations

**理由**：
1. grep 确认全仓 `check_path` 调用方只有 `tools/file_operations.py` 两处（write_file L227 + str_replace L499）——判定收敛 permission 层一处，两个工具同时受益，未来新调用方自动继承；
2. 白名单源 `default_allowed_roots()`（workspace cwd + `~/.OmniMate` + `_EXTRA_ALLOWED_ROOTS`）本来就住在 permission.py——check_path 直接复用，与 safe_path **同源**，不会两层白名单漂移；
3. `tools/file_operations.py` 零代码改动（它的调用姿势已正确，缺的是被调方语义）。

## 现状判定（实现前，brief 第一步的结论）

`check_path` 闸门 3 在 default 模式下是"**其他全通过**"（dcec556b "放开路径限制" 遗留）——write_file 其实能写任何非受保护路径，/add-dir 的白名单对它毫无意义。spec C8 目标态明确要求"未 add 拒"，因此本任务实为**恢复白名单语义（收紧）**，不只是"感知 extra roots"。

## 实现（`agent/permission.py`）

`check_path` write 分支最终闸门改为：

```
闸门 1: is_protected_path → 硬拒（顺序铁律：先于白名单，加 home 根也写不了 ~/.ssh）
读:     此处放行（不受影响）
acceptEdits: cwd 内自动批（原逻辑保留）
闸门 2: is_write_protected_path（项目代码）→ 硬拒
闸门 3 (新): bypassPermissions → 直接放行（闸门 1/2 已守住，对齐既有 bypass 测试语义）
        其余模式（default/acceptEdits/autoDeny）→ default_allowed_roots() 白名单
        之内放行，之外 self._deny("写入路径不在白名单: ...", "protected")
```

- `allowed_roots` 参数恢复生效（显式传入覆盖默认白名单，此前是死参数）
- 拒绝统一走 `self._deny` → PERMISSION_DENIED 审计 hook 不遗漏

## 测试（tests/test_permission.py 追加 5 个，TDD 先红后绿；真实 permission 注册表 + finally clear）

| 测试 | 覆盖 |
|---|---|
| `test_write_file_respects_extra_allowed_roots` | add-dir 后 write_file 到该目录通过且真写盘；未 add 的兄弟目录拒（不落盘） |
| `test_write_file_protected_path_still_denied_after_adding_home_root` | 白名单加 home 根：home 普通文件放行、`~/.ssh/evil` 仍拒（保护先于白名单） |
| `test_str_replace_respects_extra_allowed_roots` | str_replace 同判定 |
| `test_write_file_denied_after_extra_root_removed` | clear 后同路径恢复拒 |
| `test_check_path_cwd_and_bypass_semantics` | cwd 内放行 + bypassPermissions 跳白名单 |

## 连带修正（4 个旧测试写 tmp_path 白名单外，新闸门下被拒）

- `tests/test_agent_def_extensions.py`：`test_file_changed_hook_fires_on_write_file` / `_on_str_replace` / `test_file_changed_no_hook_no_crash` → 补 `add_extra_allowed_root(tmp_path)` + finally clear
- `tests/test_integration.py`：`test_checkpoint_tracked_on_write_file` → 同上（`AIAgent(omnimate_home=tmp_path)` 不改 `constants.get_omnimate_home()`，白名单不含 tmp_path，只能靠 extra root 桥接）

## 验证

- `uv run pytest tests/` → **2438 passed / 0 failed**（Task 3 基线 2432 + 净 6：5 新增 + 1 个旧 skip-分支转真断言）
- `uv run python scripts/verify.py` → 22/22 PASS

## Concerns（给 reviewer）

1. **行为收紧，非纯增量**：收回 dcec556b 的"除项目代码外其他都可写"授权——write_file/str_replace 现在写 workspace cwd / `~/.OmniMate` / extra roots 之外会拒。这是 spec C8"未 add 拒"的直接推论，也与 CLAUDE.md 一直宣称的"write_file 默认走路径白名单"重新对齐；但要"到处可写"现在只能 bypassPermissions（仍守硬底线）或逐目录 /add-dir。**建议 reviewer 确认语义回摆符合预期**。
2. **check_path 无审批通道**：白名单外直接拒（不问 approval_callback），对齐 safe_path 无 callback 行为；将来要"白名单外写询问用户"需加 callback 通路（follow-up）。
3. **acceptEdits 分支仍先于闸门 2**（既有行为未动）：cwd 在项目 repo 内时 acceptEdits 可写项目代码（绕过写保护）。非本任务引入，收紧后该早期分支更显眼，建议单独议题。
4. **`AIAgent(omnimate_home=...)` 与 `constants.get_omnimate_home()` 不同源**：白名单用后者，构造参数不影响它（测试靠 extra root 桥接）——既有割裂，未动。
