# CCAR14 Task 3 报告：check_path 闸门 3 白名单外接审批通道

**状态**：完成
**日期**：2026-08-15

## 第一步：Read callback 现状（结论）

| 项 | 实际形态 |
|---|---|
| callback 属性名 | `PermissionChecker.approval_callback`（构造参数 `approval_callback`，非 `_approval_callback`） |
| 签名 | `Callable[[str], bool]` —— 单字符串参数，返回 bool。**不支持类型区分参数** |
| 接入方式 | `cli.py:_make_approval_callback()` 注入；内部按内容启发式分路径/命令分支（含 `/` 或 `\` 或 `~` 开头 → 路径分支，已有"即将写入路径(白名单外)"提示 + "同意后整个父目录不再询问"文案） |
| terminal 会话缓存 | `self._approved: set`（存完整命令字符串，`cmd_key in self._approved` 成员查询；批准后 `_approved.add(cmd_key)` + 持久化白名单 `_save_whitelist()`） |

类型区分：因签名只有一个 str 参数，按 brief 在消息文本里区分 —— callback 收到 `文件写入审批: <resolved path>`。cli.py 的启发式（含路径分隔符）会正确落入路径分支，前缀只影响显示文本。

## 实现（`agent/permission.py`）

1. `__init__` 加 `self._approved_write_roots: set = set()`（会话级，与 `_approved` 同区）
2. `check_path` 闸门 3 白名单循环后、最终 deny 前，四步：
   - **3.1 会话缓存命中**：`resolved.relative_to(approved_root)` 命中（含相等）→ 放行，gate="approval"
   - **3.2 autoDeny 短路**：不问直接拒（fail-closed，对齐 terminal 闸门 3 的 Task J 语义，用 `effective_mode` 判断支持 `mode_override`）
   - **3.3 default / acceptEdits（cwd 外落到此）+ callback 存在**：调 `approval_callback(f"文件写入审批: {resolved}")`；批准 → `resolved.parent` 进 `_approved_write_roots` + 放行；拒绝/异常 → 拒（fail-open 按拒绝）
   - **3.4 无 callback** → 拒（现状，消息含允许根）
3. `reset_cache()` 一并清空 `_approved_write_roots`（同为会话级缓存）
4. 闸门 1/2 未动：受保护路径 / 写保护路径仍在其前硬拒，不进审批通道
5. docstring 更新（闸门 3 描述补审批通道）

## 测试（`tests/test_permission.py` 追加 4 个）

| 测试 | 覆盖 |
|---|---|
| `test_check_path_approval_grant_caches_parent` | 批准后父目录进缓存；第二次同目录 callback 不再被调 + 放行；消息含"文件写入审批"前缀 |
| `test_check_path_approval_deny_not_cached` | 拒绝不缓存（每次重新问）；无 callback 保持拒（消息含"允许"）；callback 抛异常按拒绝 |
| `test_check_path_autoden_never_asks` | autoDeny 不调 callback 直接拒 |
| `test_check_path_gates_1_2_still_hard_before_approval` | ~/.ssh（闸门 1）+ 项目代码目录（闸门 2）均硬拒且 callback 一次不调 |

**测试修正说明**：brief 的 sketch `outside = tmp_path/"outside"` + monkeypatch cwd 到 tmp_path 会让 outside 落在白名单根（workspace cwd = tmp_path）内，直接"白名单内"放行、审批通道根本不触发（TDD 红灯暴露）。已改为 chdir 到 `tmp_path/"ws"`、目标目录用兄弟目录 `tmp_path/"outside"`，保证真正白名单外。

## 验证

- `uv run pytest tests/test_permission.py`：161 passed（原 157 + 新 4）
- `uv run pytest tests/`：**2450 passed, 0 failed**（3m34s）
- `uv run python scripts/verify.py`：22/22 ALL PASS

## Concerns

1. **callback 签名**：单 `str -> bool`，无类型参数。路径/命令区分靠 cli.py 的启发式（含分隔符 → 路径）。本次前缀 `文件写入审批: ` 不影响该启发式（路径仍含分隔符），但若未来前缀改成纯中文无路径形式会误判成命令分支。
2. **持久化未做**（按 brief 范围）：`_approved_write_roots` 仅会话级。构造参数里已有 `paths_whitelist_file` / `_approved_paths` 持久化机制（cli.py 已传 `approved_paths.json`），但 check_path 目前**不查询也不写入**它——留作 follow-up：批准时可同步 `_approved_paths.add(parent)` + `_save_paths_whitelist()`，实现跨会话不重复问（cli callback docstring 已宣称此语义但实际未接线）。
3. **hook / 桌面通知未接**：terminal 审批前有 PERMISSION_REQUEST hook + toast 通知；check_path 审批通道未加（brief 未要求）。用户不盯屏时文件写入审批可能被错过——建议 follow-up 同款 fail-open 接入。
4. **缓存粒度**：批准的是 `resolved.parent`（父目录级），与 cli 提示文案"同意后整个父目录不再询问"一致。若目标本身是目录（如 mkdir 深层路径），缓存的是其父目录——语义上"该目录所在处已批"，合理。

## FIX（reviewer 收尾补齐，2026-08-15）

1. **check_path 审批点接 PERMISSION_REQUEST hook + toast**（Concern 3 收口）：在闸门 3.3 调 `approval_callback` 之前，照抄 terminal 审批点（`check()` 闸门 2）的两段 fail-open——
   - `run_permission_request({"command": "文件写入审批: <path>", "reason": "写入路径不在白名单: <path>"})`（hook 异常 pass）
   - `notify("需要审批", "agent 请求写入白名单外路径")`（lazy import，异常 pass）
   - 位置在 `if self.approval_callback is not None:` 内部，无 callback / 缓存命中 / autoDeny 短路路径不触发（对齐 terminal 语义）
2. **cli.py `_make_approval_callback` docstring 止损**（Concern 2 的宣称不实部分）：原文"路径 → 加入 approved_paths.json 跨会话不再询问"改为如实描述——路径审批仅会话内有效（父目录进 `_approved_write_roots` 会话缓存），跨会话用 `/add-dir`。

### 测试（追加 1 个）

- `test_check_path_approval_triggers_hook_and_notify`：mock hooks_registry（MagicMock）+ `patch("agent.notifier.notify")`，断言审批放行前 `run_permission_request` 被调 1 次（payload 含"文件写入审批"）且 notify 被调 1 次（标题"需要审批"）

### 验证

- `uv run pytest tests/test_permission.py tests/test_cli_commands.py -q`：211 passed
- `uv run pytest tests/ -q`：**2451 passed, 0 failed**（baseline 2450 + 新 1）
