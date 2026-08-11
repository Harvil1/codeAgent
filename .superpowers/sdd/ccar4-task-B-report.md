# CCAR4 Task B Report: post-compact 主动恢复

## Status: DONE

## What I Implemented

### 新建模块 `agent/post_compact_recovery.py`
- `build_post_compact_brief(agent)`: 主入口，构建 recovery brief 字符串（fail-open）
- `_build_recent_files_brief(paths, max_files)`: 最近 N 个文件 + preview（每文件 ≤1K，走 safe_path 白名单）
- `_build_invoked_skills_brief(agent, max_skills, total_budget)`: invoked skills 正文重注入（每 skill ≤5K，总 ≤25K）
- 预算常量：`MAX_RECENT_FILES=5`, `RECENT_FILE_PREVIEW_CHARS=1000`, `MAX_INVOKED_SKILLS=5`, `SKILL_BUDGET_CHARS=25000`, `SKILL_PER_BUDGET_CHARS=5000`

### 改造 `agent/__init__.py:_run_context_compression`
- 替换原 `self._build_reinject_context()` 调用为 `build_post_compact_brief(self)`（委托到新模块）
- 双重 fail-open：外层 try/except + 模块内 fail-open

### Config 新增（`config.py` context 节）
- `post_compact_recovery_enabled: True`（开关，默认启用）
- `post_compact_recovery_max_files: 5`（最近文件数上限）
- `post_compact_recovery_max_skills: 5`（invoked skills 数上限）

### 关键发现：基础设施已存在
Task B brief 预期需要从零搭建追踪机制，但实际发现 OmniMate 已有完整的基础：
- `_recent_read_files` / `_recent_skills` 实例属性（`__init__.py:238-239`）
- `_record_recent(kind, key)` 方法（`__init__.py:1395`）
- 在 `_dispatch_tool_calls` 的 safe + unsafe 两路都调了 `_record_recent`（`__init__.py:1593-1596, 1644-1647`）
- `_load_skill_body(name)` 方法（`__init__.py:1406`）
- 旧的内联 `_build_reinject_context()` 方法（`__init__.py:1430`，保留向后兼容）

所以 Task B 的核心工作变成：**把内联逻辑提取成独立模块 + 加 config 开关 + 加 safe_path + 加全面测试**。

## TDD Evidence

### RED 阶段
13 个单元测试全部 ModuleNotFoundError（`agent/post_compact_recovery` 不存在）：
```
tests/test_post_compact_recovery.py::test_build_brief_empty_state FAILED
tests/test_post_compact_recovery.py::test_build_brief_only_files FAILED
... (13 FAILED)
```

### GREEN 阶段
创建模块 + 接入后全部通过：
```
19 passed, 670 warnings in 0.57s
```

## Files Changed

| 文件 | 改动 |
|---|---|
| `agent/post_compact_recovery.py` | 新建（170 行） |
| `agent/__init__.py` | `_run_context_compression` 替换 `_build_reinject_context()` → `build_post_compact_brief()` |
| `config.py` | context 节加 3 个 config flag |
| `tests/test_post_compact_recovery.py` | 新建（19 个测试） |
| `CLAUDE.md` | 关键代码位置表加 post-compact recovery 行 |

## load_skill / read_file handler 接入位置

**不需要新增接入**——现有代码已经在 `_dispatch_tool_calls` 里完整接入了追踪：

### read_file 追踪
- **safe 组**：`agent/__init__.py:1593-1594` — `_run_safe_group_concurrently` 里 pre-callback
- **unsafe 组**：`agent/__init__.py:1644-1645` — `_run_unsafe_tool_call` 里 pre-callback
- 两处都调 `self._record_recent("read", str(tool_args["path"]))`

### load_skill 追踪
- **safe 组**：`agent/__init__.py:1595-1596`
- **unsafe 组**：`agent/__init__.py:1646-1647`
- 两处都调 `self._record_recent("skill", str(tool_args["name"]))`

### 解决"handler 访问 agent 实例"问题
**不是问题**——追踪逻辑在 `_dispatch_tool_calls`（AIAgent 实例方法）里，天然能访问 `self`。
handler（`_handle_read_file` / `_handle_load_skill`）本身不需要知道 agent 实例。
追踪在 dispatch 层做，比在 handler 层做更干净（handler 保持无状态）。

## 端到端测试如何覆盖

3 个 E2E 测试通过 `_run_context_compression` 真触发：

1. **`test_e2e_recovery_injected_via_run_context_compression`**: mock `compress_if_needed` 返回 `changed=True`，验证 messages 末尾的 `<post_compress_brief>` 含最近文件内容
2. **`test_e2e_recovery_not_injected_when_disabled`**: config 关闭 recovery，验证 brief 不含文件内容
3. **`test_e2e_full_run_conversation_with_recovery`**: 完整 `run_conversation` 链路，验证 LLM 收到的 messages 含 recovery 段

2 个追踪 E2E 测试通过真实 `run_conversation` + mock LLM：
4. **`test_read_file_triggers_recent_tracking`**: read_file 工具调用后 `_recent_read_files` 有记录
5. **`test_load_skill_triggers_recent_tracking`**: load_skill 工具调用后 `_recent_skills` 有记录

## Self-Review Findings

- [x] `_recent_read_files` / `_recent_skills` 是 AIAgent 实例属性（会话级，新会话重置）
- [x] load_skill / read_file 真触发追踪（端到端测试覆盖，追踪在 dispatch 层非 handler 层）
- [x] build_post_compact_brief fail-open（双层：模块内 try/except + `_run_context_compression` 外层 try/except）
- [x] brief 注入在 `conversation_history = messages[1:]` **之后**（设计如此：brief 是临时消息不进 history，对齐 `test_brief_not_in_conversation_history`）
- [x] 端到端测试用真实 `_run_context_compression`（3 个 + 2 个追踪）
- [x] config flag 默认 True（启用）
- [x] safe_path 走白名单读文件（`_build_recent_files_brief` 调 `safe_path(path, write=False)`）
- [x] encoding="utf-8"（所有文件读）

### 设计偏差说明
Task B brief 说"brief 注入在 conversation_history = messages[1:] 之前"，但现有代码设计是**之后**注入（brief 是 ephemeral，不持久化）。这个设计被 `test_brief_not_in_conversation_history` 明确测试保护。我遵循了现有设计约定，不破坏已有测试。

## Test Summary
- 新增测试：19 个全 PASS
- 全套回归：1731 passed, 1 skipped, 0 failures
- verify.py: 22/22 ALL PASS

## Commits
- `feat(context): CCAR4 Task B post-compact 主动恢复（最近文件 + invoked skills）`
