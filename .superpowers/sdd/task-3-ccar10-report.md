# CCAR10 Task 3 Report: statusline

## 状态
**完成 / 全部测试通过**

## 交付内容

每轮 AI 响应完全输出后，主循环在末尾打一行 dim 紧凑状态：

```
⚡deepseek-chat │ 会话 12.3K tok │ goal:进行中#5 │ 项目:HermesAgent
```

四段（任意段缺省自动跳过）：model / 会话累计 token / goal 状态 / 项目名。

## 实现要点

### 1. token 来源
**优先用 `_llm_usage_stats`**（`agent/__init__.py:_record_llm_usage` 实时累加的字典，含 prompt + completion + cache tokens），`total_prompt_tokens + total_completion_tokens` 之和作为会话累计。fallback 到 `session_total_tokens` 字段（兼容 mock / 未来扩展）。

不加新字段——现有 `_llm_usage_stats` 已经在 `_record_llm_usage` 里每次 LLM 调用都累加，是天然的会话级累计。

### 2. 不用 rich.Live
约束关键：Windows 下 `rich.Live` 与 `input()` 冲突，且 Live 会刷掉滚动历史。改为**每轮 response 完整输出后**调 `console.print("[dim]...[/dim]")` 一次，简单可靠。

### 3. fail-open
`_render_statusline` 整体 try/except 返回 ""；`_statusline_project_key` 赋值也 try/except 兜底成 ""。statusline 永不影响主对话流程。

### 4. 接入点
`cli.py:run_interactive` 主循环里，response 保存到 session_store 之后、except 之前。**中断/异常路径都不打 statusline**（用户主动断开就不该再追加信息）。

### 5. project_key 赋值
`RuntimeContext.__init__` 加 `_statusline_project_key = None`；`initialize()` 末尾（`_fire_session_start` 前）调 `get_project_memory_key()` 赋值一次。整个会话复用，不每轮重算（canonical git root 启动时确定一次即可）。

## 文件改动

| 文件 | 改动 |
|---|---|
| `config.py` | DEFAULT_CONFIG 加 `"statusline": {"enabled": True}` |
| `cli.py` | 加 `_format_tokens` + `_render_statusline` 两个函数；`RuntimeContext.__init__` 加 `_statusline_project_key` 字段；`initialize()` 末尾赋值；`run_interactive` 主循环接入 |
| `tests/test_statusline.py` | 13 个测试覆盖：全段渲染 / 无 goal / paused / completed / cancelled / disabled / 无 config 默认 enabled / token 格式化 / 优先 stats / fallback 字段 / fail-open / 无项目键 / 无 model |

## 测试结果

```
tests/test_statusline.py: 13 passed
全套 pytest: 2199 passed, 1 skipped, 0 failed (188s)
verify.py: 22/22 ALL PASS
```

对比 CCAR9：2168 passed → 2199 passed（+31，含本 task 13 个）。

## Commit
见 `git log`（commit hash 在执行后填入）。

## Concerns / Follow-up

1. **goal 状态枚举**——目前只显式处理 `active`/`paused`/`completed`；`cancelled` 隐藏。如果未来 `GoalState` 加新状态（如 `failed`），需要同步更新 `_render_statusline`。
2. **token 口径**——当前用 `_llm_usage_stats` 的 `total_prompt_tokens + total_completion_tokens`，**不含 cache_read/creation tokens**。如果用户想看"实际计费 tokens"（含 cache），需要再加一段。当前口径是"模型消耗的逻辑 token"。
3. **不在中断/错误路径打 statusline**——这是有意为之，避免中断后又追加信息；但也意味着如果某轮 LLM 报错，statusline 不更新，累计 token 数字会"卡住"显示上一轮的值。这是正确行为（错误的轮次没产生 usage），但用户可能误解为 statusline 坏了。文档化即可。
4. **emoji ⚡**——在某些 Windows 终端（默认 cp437/cp936）可能显示为方框。已通过 fail-open 兜底；如果用户反馈视觉问题，可加 config 选项关闭 emoji（当前不加，YAGNI）。
