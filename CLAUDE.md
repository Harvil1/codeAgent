# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目本质

本项目后期会做成win电脑的软件安装包（现在不做）。
HarvilAgent 是基于 `D:\project\hermes-agent-main\replication-guide\` 复刻指南实现的自学习 AI Agent，并借鉴 Claude Code 的工程实践做了取长补短。**"越用越聪明" 不是一个营销词，它是由三个独立子系统 + 后台维护工人支撑的工程闭环**：

```
                Agent 核心（同步 while 循环 + 工具分发 + 消息历史）
       ┌──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
   技能系统           记忆系统            会话搜索
   (Skills)          (Memory)          (Sessions)
   "怎么做"          "用户是谁"        反查历史对话
   沉淀成 MD         沉淀成事实
       └──────────────────┘
                       ▼
              Curator 后台维护（7 天周期，状态机 + LLM 伞形合并）
```

横向还有一层 **安全与韧性基线**（借鉴 Claude Code）包裹整个核心：

```
   ┌──────────────────────────────────────────────────┐
   │  权限三道闸门 · 路径白名单 · 输出截断            │
   │  LLM 重试退避 · 备用模型切换 · TodoWrite reminder │
   │  MCP 外部工具 · 持久化任务图 · worktree 隔离     │
   └──────────────────────────────────────────────────┘
```

实现完成度：59 个 Python 文件，**294 个测试通过**，22 项检查清单全 PASS。

## 关键设计原则（修改代码前必读）

这六条贯穿整个项目，理解它们比理解任何单个模块重要：

1. **核心是窄腰，能力在边缘** —— 每个发给 LLM 的工具 schema 在每次 API 调用中重复传输，直接增加 token 成本。能做成技能（数据）就不要做成工具（代码）。扩展优先级：`现有代码 < CLI/技能 < check_fn 门控工具 < 插件 < MCP < 新核心工具`。
2. **Prompt Caching 神圣不可侵犯** —— system prompt 在**会话开始时构建一次**（`AIAgent._get_system_prompt` 缓存），中途任何修改都会让前缀缓存失效、成本翻倍。记忆写入立即落盘，但**下次会话才注入**。唯一例外是上下文压缩（`context_compressor`）。
3. **完全可逆** —— 自动归档系统**永不删除**，只能移到 `.archive/`。Task System 用软删除（`status="deleted"`），文件保留可恢复。这让 agent 敢于自动管理自己的知识库。
4. **用户意图优先于算法** —— `pinned` 的技能免疫所有 curator 自动转换。权限闸门里用户审批的命令进会话缓存不重复询问。算法可以错，用户的明确判断不能被覆盖。
5. **发现 ≠ 可见** —— 工具注册（登记到 `registry`）和暴露给 LLM（`resolve_toolset` → `get_definitions`）是两步。`check_fn` 让工具根据运行时环境动态出现/消失（MCP 工具用这个机制：server 断开时自动隐藏）。
6. **安全默认 > 事后补救** —— `terminal` 工具默认走三道权限闸门（黑名单 → 规则 → 审批），`write_file` 默认走路径白名单（cwd + `~/.agent`）。要"放开"必须显式配置，不能默认放开。

## 模块依赖链（从底层到上层）

```
constants.py            ← 无依赖，路径函数（AGENT_HOME 解析）
    ↑
agent/permission.py     ← 命令黑名单 + safe_path（被 tools 调用）
agent/llm_retry.py      ← LLM 调用重试/退避（被 agent 调用）
    ↑
tools/registry.py       ← 中央工具注册表（AST 自动发现 + check_fn TTL 缓存）
    ↑
tools/*.py              ← 每个文件 import 后在模块顶层 registry.register()
                           （terminal/file/memory/skill/delegate/todo/task）
agent/mcp_client.py     ← MCP 客户端（stdio transport）
tools/mcp_tool.py       ← 把 MCP 工具动态注册到 registry
    ↑
model_tools.py          ← handle_function_call + get_tool_definitions
                           （启用 mcp toolset 时动态发现 mcp__ 工具）
    ↑
agent/__init__.py       ← AIAgent 主类（run_conversation 同步循环 + TodoWrite reminder）
    ↑
cli.py                  ← RuntimeContext 聚合所有组件
```

**上层依赖下层，下层不知道上层存在。** `registry` 不知道 `agent` 是什么，只提供注册和分发。

## 消息历史的三种角色（铁律：必须严格交替）

```python
{"role": "system", "content": "..."}     # 仅一条，开头，会话中不变
{"role": "user", "content": "..."}
{"role": "assistant", "content": null, "tool_calls": [...]}
{"role": "tool", "tool_call_id": "call_xxx", "content": "<JSON 字符串>"}
```

工具 handler **必须返回 JSON 字符串**（不是 dict、不是裸字符串）。错误统一为 `{"error": "...", "error_type": "..."}`。权限拒绝用 `{"error_type": "permission_denied"}`。

## 技能 vs 工具 vs 记忆 vs 任务

|   | 工具 | 技能 | 记忆 | 任务 |
|---|---|---|---|---|
| 形态 | Python 代码 | Markdown 文件 | 短文本条目 | JSON 任务对象 |
| 创建者 | 开发者 | Agent + 用户 + 开发者 | Agent | Agent |
| 持久化 | 永久（代码） | `~/.agent/skills/` | `~/.agent/MEMORY.md` | `~/.agent/.tasks/` |
| 跨会话 | 是 | 是 | 是 | TodoWrite 否 / Task System 是 |
| 注入位置 | 工具 schema | user 消息（触发时） | system prompt | 工具调用 |

技能是数据不是代码——这让 LLM 能通过 `skill_manage` 工具自己创建和改进，是自学习的关键。

## 安全机制（三道闸门 + 路径白名单 + 输出截断）

修改命令执行/文件读写相关代码必读。详见 `agent/permission.py`。

```
terminal 命令执行：PermissionChecker.check() →
  闸门 1: 硬拒绝黑名单（rm -rf /、sudo、mkfs、fork bomb、强推主分支等 15+ 模式）
  闸门 2: 规则匹配（预留扩展）
  闸门 3: 用户审批 callback（会话内缓存已批准命令，不重复问）

read_file:  safe_path(write=False) → 拒绝 ~/.ssh、/etc、C:\Windows 等受保护路径
write_file: safe_path(write=True)  → 受保护路径 + 工作目录外写入全部拒绝
            默认白名单：cwd + get_agent_home()，可通过 allowed_roots 扩展

terminal 输出：超过 50000 字符截断，保留前后各一半 + 续写提示
```

⚠️ **不要为了"方便"绕过这些检查**。如果某工具确实需要写到 cwd 外，通过 `kwargs` 接收 `harvil_home` 并在 safe_path 的 `allowed_roots` 里显式声明。

## 韧性机制（重试 + 备用模型 + max_tokens 升级 + 529 早切）

`agent/llm_retry.py:call_with_retry` 被 `AIAgent.run_conversation` 用于每次 LLM 调用：

- 可重试错误（429 限流、5xx 服务器错误、连接/超时）：指数退避重试 5 次（1s → 2s → 4s → 8s → 16s），尊重 `Retry-After` header
- **退避加抖动（P0-1）**：每次 sleep = base + uniform(0, base × 0.25)，多实例并发遇到 429 时避免雷击。`jitter_ratio=0` 关闭抖动（向后兼容）
- **529 连续失败早切（P1-1）**：连续 `consecutive_529_threshold`（默认 3）次 529 立即切备用 client，不浪费剩余重试次数（Anthropic 过载通常持续一段时间）
- 不可重试错误（400 参数、401 认证、403 权限）：立即抛
- 主模型重试耗尽后，如果配置了 `fallback_model`，用备用模型再试一次
- AIAgent 构造参数：`AIAgent(..., fallback_model="deepseek-reasoner")`
- **max_tokens 升级（P0-3）**：LLM 返回 `finish_reason="length"`（max_tokens 截断）时，先升 `max_tokens` 到 32768 用非流式重试一次（不打断思路），升级后仍不够才让主循环走续写路径。`MaxTokensEscalator` 整个会话幂等（最多升 1 次）。入口：流式 `agent/__init__.py:_call_llm_streaming` 末尾 + 非流式 `agent/__init__.py:run_conversation` 非流式分支

## 任务追踪（两层）

| 层 | TodoWrite | Task System |
|---|---|---|
| 文件 | `agent/todo.py` + `tools/todo_tool.py` | `agent/task_store.py` + `tools/task_tools.py` |
| 持久化 | 内存（单会话） | `~/.agent/.tasks/{id}.json`（跨会话） |
| 依赖 | 无 | DAG（`blocked_by` + `can_start` + `find_ready`） |
| 约束 | 同时只能 1 个 in_progress | 状态机 pending→in_progress→completed |
| 工具 | `todo_write`（替换式） | `task_create` / `task_update` / `task_complete` / `task_list` |
| 提醒 | 3 轮未更新注入 `<todo_reminder>` | 无自动提醒（靠 LLM 调 task_list） |

修改 `AIAgent.run_conversation` 时记得：每轮 LLM 调用后必须 `self.todo_manager.increment_round()`，调用前检查 `should_remind()` 并临时注入 reminder 消息（**不进 conversation_history**，避免污染持久化）。

## 扩展机制（MCP + worktree + load_skill）

- **MCP**：`~/.agent/.mcp.json` 配置外部 server，启动时 `tools/mcp_tool.py:initialize_mcp` 连接 + 注册。工具以 `mcp__<server>__<tool>` 前缀暴露，`check_fn` 在 server 断开时自动隐藏。`enabled_toolsets` 要含 `"mcp"` 才对 LLM 可见。
- **worktree 隔离**：`tools/worktree.py:create_isolated_workspace`。`delegate_task(isolated_workspace=True)` 让子代理在独立 git worktree 或临时目录跑，互不干扰。
- **load_skill**：两级加载。system prompt 只放技能索引（名字+描述），LLM 按需调 `load_skill(name)` 获取完整正文（去 frontmatter）。区别于 `skill_view`（含 frontmatter，用户视角）。
- **summary_only**：`delegate_task(summary_only=True)`（默认）时，子代理结果超 500 字用 LLM 压缩成 300 字摘要，节省父代理 context。

## 常用命令

```bash
# 运行
uv run python main.py                   # 交互模式
uv run python main.py chat "你好"       # 一次性问答

# 测试
uv run pytest tests/ -v                 # 全部 294 个测试
uv run pytest tests/test_permission.py  # 单个模块（如权限）

# 复刻检查清单验证（原 22 项功能）
uv run python scripts/verify.py

# Curator 维护
uv run python -m curator_cli status
uv run python -m curator_cli run --dry-run
uv run python -m curator_cli pin <skill-name>

# 会话移交（handoff）
/handoff save "标题"        # 保存当前会话为 bundle
/handoff list               # 列出所有 bundle
/handoff load <id>          # 加载 bundle 覆盖当前会话
/handoff export <id> <path> # 导出 bundle 到文件（用于跨机迁移）
/handoff import <path>      # 从文件导入 bundle

# 依赖管理（必须用 uv，不要用 pip）
uv add <包名>                           # 添加运行时依赖
uv add --dev <包名>                     # 添加开发依赖
uv sync                                 # 同步已声明依赖
```

## 约定

- **语言**：所有注释、文档、commit message、计划、回复使用中文。代码标识符（变量、函数、类名）用英文。
- **依赖管理**：不要用 `pip install`，也不要手改 `pyproject.toml` 的 `dependencies`，统一 `uv add`。
- **文件 I/O**：必须指定 `encoding="utf-8"`（Windows 默认 cp1252 会乱码）。ruff 规则 `PLW1514` 强制。
- **命令执行安全**：不要绕过 `PermissionChecker` 和 `safe_path`。新增工具如果需要写文件，通过 `harvil_home` 参数 + `safe_path(write=True, allowed_roots=[...])` 显式声明允许的目录。
- **默认 provider**：DeepSeek（`base_url=https://api.deepseek.com/v1`，模型 `deepseek-chat`，env `DEEPSEEK_API_KEY`）。在 `config.yaml` / `.env` 切换其他 OpenAI 兼容 provider。
- **agent home**：默认 `~/.agent`，可用 `AGENT_HOME` 环境变量覆盖（profile 隔离机制）。
- **工具结果契约**：所有 handler 返回 JSON 字符串，错误用 `{"error": "...", "error_type": "..."}`。


## 关键代码位置

| 想修改什么 | 看这里 |
|---|---|
| 对话主循环 / 中断 / grace call | `agent/__init__.py:run_conversation` |
| system prompt 构建（记忆/技能索引/GUIDANCE） | `agent/prompt_builder.py:build_system_prompt` |
| 上下文压缩（唯一可改 system prompt 的场景） | `agent/context_pipeline.py:compress_if_needed` + 主循环 `agent/__init__.py` 压缩后注入 `<post_compress_brief>` |
| 命令权限闸门 | `agent/permission.py:PermissionChecker.check` |
| 路径白名单 | `agent/permission.py:safe_path` |
| LLM 重试/备用模型/退避抖动/529 早切 | `agent/llm_retry.py:call_with_retry` + `_compute_backoff`（抖动）+ 连续 529 计数 |
| max_tokens 升级（finish_reason=length 自动重试） | `agent/llm_retry.py:MaxTokensEscalator` + `detect_length_finish`；入口 `agent/__init__.py:_call_llm_streaming`（流式）和 `run_conversation` 非流式分支 |
| TodoWrite reminder 注入 | `agent/__init__.py`（搜 `should_remind`） |
| 工具注册模式（添加新工具看这个） | `tools/terminal_tool.py`（含权限集成） |
| 工具集可见性控制 | `toolsets.py:TOOLSETS` + `model_tools.py:get_tool_definitions` |
| MCP 外部工具接入 | `agent/mcp_client.py` + `tools/mcp_tool.py` |
| 持久化任务 + DAG 依赖 | `agent/task_store.py:TaskStore` |
| 子代理委托 + worktree + 摘要 | `tools/delegate_tool.py:_run_child` |
| 配置默认值（所有参数源头） | `config.py:DEFAULT_CONFIG` |
| 复刻指南（设计权衡详解） | `D:\project\hermes-agent-main\replication-guide\` |
| 会话移交 bundle | `agent/handoff.py:HandoffStore` |
| Vision/Image 工具 | `tools/image_tool.py`（image_analyze / image_ocr，复用 safe_path） |
| Plan Mode（计划模式 + 审批） | `agent/__init__.py`（plan_mode 字段 + 主循环三处分支）+ `tools/plan_mode_tool.py` + `cli.py`（/plan 命令） |
| Cron 调度（一次性 + 7 天过期 + catch_up 补偿） | `agent/cron.py:CronScheduler`（`_tick` 含过期/一次性；`_apply_catch_up` 启动时补跑错过触发） |
| snip 成对保护（L1 裁剪不拆散 tool_call/result） | `agent/context_pipeline.py:snip_compact`（`_has_tool_calls` / `_is_tool_result` 辅助） |
| 主动 output_offload（L2.5 大 tool 结果落盘） | `agent/context_pipeline.py:offload_large_tool_results`（`compress_if_needed` 编排里调） |
| 后台任务停滞看门狗（stall_timeout 秒无输出 → 通知） | `agent/background.py:BackgroundManager._watch_with_stall`（默认 45s，`config.bg_task.stall_timeout` 配置） |
| Hook 事件（11 种） | `agent/hooks.py:HookEvent`（核心 6 + SESSION_START/END、PRE/POST_COMPACT、CONFIG_CHANGE） |
| 子任务进度摘要（pendingToolUseSummary） | `agent/progress.py:ProgressReporter`（`tools/delegate_tool.py:_run_child` 接入） |
| 团队 request-response 协议 | `agent/team/bus.py:MessageBus.send_request`/`send_response`/`find_response`（response 强制配 request_id） |
| session fork | `agent/session_store.py:SessionStore.fork_session`（消息全复制到新 id） |
| MCP 多传输 + OAuth | `agent/mcp_client.py:MCPTransport` 抽象 + `StdioTransport`/`HTTPTransport`（含 OAuth refresh） |

## 已知约束（设计如此，不是 bug）

- **记忆写入后本会话不生效** —— 保护 prompt cache。`MemoryStore.snapshot_for_prompt()` 是 frozen 的。
- **首次 curator 运行被推迟** —— 种子化 `last_run_at`，等一个完整周期（避免新装就大改技能库）。
- **`use_count=0` 不是归档理由** —— 一个 "Kubernetes 故障处理" 技能可能 2 个月不触发，但仍有价值。按内容判断，不按计数。
- **子代理不继承对话历史** —— 独立 `AIAgent` 实例，只通过 `context` 参数传递必要信息。`summary_only=True`（默认）时连结果都被压缩。
- **权限审批缓存是会话级的** —— `PermissionChecker._approved` 集合。新会话重置，避免长期信任漂移。
- **TodoWrite 不持久化** —— 单会话用。跨会话任务用 Task System（`.tasks/{id}.json`）。
- **MCP 工具依赖外部进程** —— server 崩溃后工具自动隐藏（`check_fn` 返回 False），但不自动重启。

## 测试策略

- **按模块组织**：`tests/test_{basic,memory,skills,curator,sessions,context,delegation,config,integration,permission,todo,llm_retry,worktree,mcp,task_system}.py`
- **集成**：`tests/test_integration.py` 用 mock OpenAI client 跑完整对话流程（含工具调用、记忆注入、中断）
- **验证脚本**：`scripts/verify.py` 跑 11-scaffold.md 的 22 项检查清单，适合改完代码后快速回归（不含 P0-P3 新功能测试）
- **新增功能必加测试**：每个新模块（permission/todo/mcp/task_store 等）都有独立测试文件，改完跑 `uv run pytest tests/` 确认无回归
