# CCAR9 Task 4 Report: /init 命令生成 OMNIMATE.md

> 注：本文件路径与早期 task-4-report（CCAR8 TraceSink，commit 6573ce25；以及更早的 features flags 示例，commit c7cc5ea9）同名。
> 之前的报告已被本 CCAR9 Task 4 报告覆盖。
> 原 TraceSink 要点：`agent/trace.py` 新增 148 行（TraceSink 本地 sink，emit/query/summary）。
> 原 features flags 要点：`examples/settings-features.json` 10 个 feature flags 示例。

## 状态
**完成**（4/4 测试通过，全套 2164 passed / 1 skipped 无回归）

## 对标
Claude Code 的 `/init` 命令——收集项目信息（目录树/关键文件/类型统计）→ 主 LLM 四段式生成（项目本质/常用命令/架构/约定）→ 写 `<cwd>/OMNIMATE.md`。prompt_builder 已有 OMNIMATE.md 注入闭环（`prompt_builder.py:240` 扫 cwd 下的 OMNIMATE.md），**本 task 零改动**。

## 改动
| 文件 | 类型 | 说明 |
|---|---|---|
| `cli.py` | Modify | 加 `/init` 命令分发（`_handle_command` 内）+ `_handle_init_command` 函数 |
| `tests/test_init_command.py` | Test | 4 个新测试（正常/已存在不带 force/--force 覆盖/空目录 fail-open） |

## 接口
- **Consumes**: `rt.agent.llm_client.chat_completions`（主 LLM）/ `agent.workspace_context.get_workspace_cwd`
- **Produces**: `<cwd>/OMNIMATE.md`（下次会话由 prompt_builder 自动注入 system prompt）

## 实现要点
1. **fail-open 收集**：目录树扫描、关键文件读取、文件类型统计三段，每段独立 try/except，缺哪跳哪（空目录也能生成）
2. **--force 语义**：不带 `--force` 且文件已存在 → 提示并返回，不覆盖；带 `--force` 走完整流程覆盖
3. **asyncio.run 兜底**：参考 `_start_new_goal` 的同款模式——`asyncio.run` 在已有事件循环时 RuntimeError，走 try/except 兜底；进一步有 `loop.is_running()` 检测避免 `run_until_complete` 冲突
4. **中文 prompt + 四节固定结构**：要求 LLM 只写可从信息验证的事实，命令给出具体形式（如 `uv run pytest tests/`）
5. **局部 import**：`get_workspace_cwd` 在函数内局部 import（与 codebase 一致，agent_defs/prompt_builder/permission 等都这么干）
6. **LLM 返回空 → fail-open**：提示生成失败，不写空文件
7. **写入失败 → fail-open**：提示写入失败原因，不抛

## 测试
- `test_init_generates_file` — 正常流程（空目录+README → 生成 OMNIMATE.md 含 LLM 输出）
- `test_init_existing_without_force` — 已存在不带 `--force` 不覆盖
- `test_init_force_overwrites` — `--force` 覆盖重新生成
- `test_init_collection_failopen` — 空目录也能生成（fail-open）

测试 mock 了 `RuntimeContext` + `agent.llm_client`（`AsyncMock`），patch 源头 `agent.workspace_context.get_workspace_cwd`（局部 import 模式，patch `cli.get_workspace_cwd` 无效）。

## 遵循的约束（逐项确认）
- [x] 文件 I/O `encoding="utf-8"`：read_text/write_text 都显式指定
- [x] 中文注释/commit：全部注释 + commit message 中文
- [x] TDD：先写 4 失败测试 → ImportError 确认 → 实现 → 通过
- [x] fail-open：三段信息收集 + LLM 调用 + 写入，每段独立 try/except
- [x] `--force` 语义正确（test_init_force_overwrites + test_init_existing_without_force 验证）
- [x] 局部 import `get_workspace_cwd`（对齐 codebase 约定）
- [x] `asyncio.run` 在已有 loop 的兜底（对齐 `_start_new_goal` 模式）
- [x] `tests/` 在 `.gitignore`，用 `git add -f`

## 测试结果
- `uv run pytest tests/test_init_command.py -v` → **4 passed**
- `uv run pytest tests/ -q` → **2164 passed / 1 skipped / 0 failed**

## Commit
`<待填>` — `feat(init): /init 命令生成 OMNIMATE.md（CCAR9 Task 4，对标 /init）`

## Concerns / Follow-up
1. **Minor**：LLM 调用走 `chat_completions`（主 LLM）而非 aux_llm_router——对标 Claude Code 的 /init 用主 LLM 生成，语义一致。如果未来想做"轻量 init"（用 aux LLM 省成本），可以加 config 开关。
2. **Minor**：写 cwd 不走 safe_path 包装——target 是 `cwd / "OMNIMATE.md"`，cwd 本身在 safe_path 白名单内（默认允许 cwd + `~/.OmniMate`），所以不需要额外声明 allowed_roots。
3. **环境噪声**：全套测试时出现一个 `UnicodeDecodeError` warning（来自无关测试在后台线程用 gbk 读 PNG），不是本 task 引入的。
