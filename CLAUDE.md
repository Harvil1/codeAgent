# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目本质

本项目后期会做成win电脑的软件安装包（现在不做）。
OmniMate 是基于 `D:\project\hermes-agent-main\replication-guide\` 复刻指南实现的自学习 AI Agent，并借鉴 Claude Code 的工程实践做了取长补短。**"越用越聪明" 不是一个营销词，它是由三个独立子系统 + 后台维护工人支撑的工程闭环**：

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
   │  LLM 重试退避 · 备用模型切换 · 持久化任务图      │
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
6. **安全默认 > 事后补救** —— `terminal` 工具默认走三道权限闸门（黑名单 → 规则 → 审批），`write_file` 默认走路径白名单（cwd + `~/.OmniMate`）。要"放开"必须显式配置，不能默认放开。

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
                           （terminal/file/memory/skill/delegate/task）
agent/mcp_client.py     ← MCP 客户端（stdio transport）
tools/mcp_tool.py       ← 把 MCP 工具动态注册到 registry
    ↑
model_tools.py          ← handle_function_call + get_tool_definitions
                           （启用 mcp toolset 时动态发现 mcp__ 工具）
    ↑
agent/__init__.py       ← AIAgent 主类（run_conversation 同步循环）
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
| 持久化 | 永久（代码） | `~/.OmniMate/skills/` | `~/.OmniMate/MEMORY.md` | `~/.OmniMate/.tasks/` |
| 跨会话 | 是 | 是 | 是 | 是 |
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

⚠️ **不要为了"方便"绕过这些检查**。如果某工具确实需要写到 cwd 外，通过 `kwargs` 接收 `omnimate_home` 并在 safe_path 的 `allowed_roots` 里显式声明。

## OS 沙箱（对齐 Claude Code `/sandbox`）

terminal_tool 命令执行叠加 OS 内核级强制隔离，作为权限闸门之后的硬防线：

- **Linux**：Bubblewrap（bwrap）—— 用户命名空间 + bind mount
- **macOS**：Seatbelt（sandbox-exec）—— Apple 沙箱 profile
- **Windows**：不支持原生（用户走 WSL2/Docker），`is_available()` 返回 False

### 触发

```
/sandbox on       开启 OS 沙箱（命令包进 bwrap/sandbox-exec）
/sandbox off      关闭（默认）
/sandbox status   查看当前状态
```

### 关键文件

| 想修改什么 | 看这里 |
|---|---|
| 沙箱 wrapper 构造（跨平台） | `agent/sandbox_runner.py` |
| is_available / availability_reason | `agent/sandbox_runner.py` |
| terminal 注入 wrapper 的位置 | `tools/terminal_tool.py:_handle_terminal`（sandbox_mode 分支） |
| sandbox_mode 字段 + set_sandbox_mode | `agent/permission.py:PermissionChecker` |
| /sandbox 命令处理 | `cli.py:_handle_command`（name == "/sandbox"） |
| 配置默认值 | `config.py:DEFAULT_CONFIG["security"]` |

### 设计原则

- **叠加层**：不替换 PermissionChecker / safe_path / worktree，是新增层
- **fail-open**：不可用时警告并降级到原 `shell=True` 路径
- **GUI 命令跳过**：`start`/Chrome 等 GUI 程序在沙箱里启不来，强制走原路径
- **网络不隔离**：bwrap 不加 `--unshare-net`（用户决策）
- **只防写不防读**：第一版只挡写敏感路径（读保护留后续迭代）

### 可写目录范围

默认允许写入：`cwd` + `~/.OmniMate`，可通过 `config["security"]["sandbox_writable_roots"]` 扩展。

## 韧性机制（重试 + 备用模型 + max_tokens 升级 + 529 早切）

`agent/llm_retry.py:call_with_retry` 被 `AIAgent.run_conversation` 用于每次 LLM 调用：

- 可重试错误（429 限流、5xx 服务器错误、连接/超时）：指数退避重试 5 次（1s → 2s → 4s → 8s → 16s），尊重 `Retry-After` header
- **退避加抖动（P0-1）**：每次 sleep = base + uniform(0, base × 0.25)，多实例并发遇到 429 时避免雷击。`jitter_ratio=0` 关闭抖动（向后兼容）
- **529 连续失败早切（P1-1）**：连续 `consecutive_529_threshold`（默认 3）次 529 立即切备用 client，不浪费剩余重试次数（Anthropic 过载通常持续一段时间）
- 不可重试错误（400 参数、401 认证、403 权限）：立即抛
- 主模型重试耗尽后，如果配置了 `fallback_model`，用备用模型再试一次
- AIAgent 构造参数：`AIAgent(..., fallback_model="deepseek-reasoner")`
- **max_tokens 升级（P0-3）**：LLM 返回 `finish_reason="length"`（max_tokens 截断）时，先升 `max_tokens` 到 32768 用非流式重试一次（不打断思路），升级后仍不够才让主循环走续写路径。`MaxTokensEscalator` 整个会话幂等（最多升 1 次）。入口：流式 `agent/__init__.py:_call_llm_streaming` 末尾 + 非流式 `agent/__init__.py:run_conversation` 非流式分支

## 任务追踪（Task System）

| 层 | Task System |
|---|---|
| 文件 | `agent/task_store.py` + `tools/task_tools.py` |
| 持久化 | `~/.OmniMate/.tasks/{id}.json`（跨会话） |
| 依赖 | DAG（`blocked_by` + `can_start` + `find_ready`） |
| 约束 | 状态机 pending→in_progress→completed |
| 工具 | `task_create` / `task_update` / `task_complete` / `task_list` 等 |

## 扩展机制（MCP + worktree + load_skill）

- **MCP**：`~/.OmniMate/.mcp.json` 配置外部 server，启动时 `tools/mcp_tool.py:initialize_mcp` 连接 + 注册。工具以 `mcp__<server>__<tool>` 前缀暴露，`check_fn` 在 server 断开时自动隐藏。`enabled_toolsets` 要含 `"mcp"` 才对 LLM 可见。
- **worktree 隔离**：`tools/worktree.py:create_isolated_workspace`。`subagent(isolated_workspace=True)` 让子代理在独立 git worktree 或临时目录跑，互不干扰。
- **load_skill**：两级加载。system prompt 只放技能索引（名字+描述），LLM 按需调 `load_skill(name)` 获取完整正文（去 frontmatter）。区别于 `skill_view`（含 frontmatter，用户视角）。
- **summary_only**：`subagent(summary_only=True)`（默认）时，子代理结果超 500 字用 LLM 压缩成 300 字摘要，节省父代理 context。

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
- **命令执行安全**：不要绕过 `PermissionChecker` 和 `safe_path`。新增工具如果需要写文件，通过 `omnimate_home` 参数 + `safe_path(write=True, allowed_roots=[...])` 显式声明允许的目录。
- **默认 provider**：DeepSeek（`base_url=https://api.deepseek.com/v1`，模型 `deepseek-chat`，env `DEEPSEEK_API_KEY`）。在 `config.yaml` / `.env` 切换其他 OpenAI 兼容 provider。
- **agent home**：默认 `~/.OmniMate`，可用 `AGENT_HOME` 环境变量覆盖（profile 隔离机制）。
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
| 记忆三级粒度（L0/L1/L2） | `agent/memory_store.py:MemoryEntry.summary`（L1 摘要层）；索引行追加 summary，retriever 拿到的 index 自动含 L1 |
| 任务级反思引擎 | `agent/reflection.py:apply_reflection`（aux_llm 从轨迹提炼 user/feedback/project 三类经验，自动 memory_save）；入口 `agent/__init__.py:AIAgent._trigger_reflection_async`（run_conversation 末尾异步触发） |
| Memory Curator（记忆维护工人） | `agent/memory_curator.py:apply_automatic_transitions`（第 1 阶段确定性状态机）+ `run_memory_review`（第 2 阶段 LLM 合并/矛盾解决）+ `should_run_now_memory`（门控）；配置入口 `config["memory"]["curator"]`（enabled / interval_hours / llm_review_enabled / max_batch_size） |
| WebSearch（Tavily 网络搜索） | `tools/web_search_tool.py`（check_fn 门控：无 TAVILY_API_KEY 自动隐藏）；schema 在 `WEB_SEARCH_SCHEMA` |
| 自定义子代理 .md 定义 | `agent/agent_defs.py:scan_agent_defs`（扫描 `~/.OmniMate/agents/` + `<cwd>/.claude/agents/`，项目级覆盖用户级）；集成在 `tools/delegate_tool.py:_run_child`（subagent_type 传自定义名）+ cli.py `/agents` |
| 权限模式（default / bypassPermissions） | `agent/permission.py:PermissionChecker.mode`（bypass 跳过审批，但保留 fatal 底线 + 自我保护 + 受保护路径）；切换 `/permission` 命令或 `config.security.permission_mode` |
| Hooks 5 种 handler 类型 | `agent/hook_exec.py:dispatch_hook`（command/http/mcp_tool/prompt/agent）；声明式配置解析在 `agent/hook_loader.py:_parse_hook`；aux_router 注入 `set_aux_router_provider`（cli.py 接线） |
| acceptEdits 权限模式 | `agent/permission.py:PermissionChecker.check`（acceptEdits 分支 + `_is_safe_fs_in_cwd` + `_SHELL_OPS` 复合命令守卫）；自动批 cwd 内 safe-fs + 写入，守 fatal 底线 |
| 内置子代理 Explore/Plan | `agent/builtin_agents/{explore,plan}.md`（scan_agent_defs 默认加载，用户/项目可 override）；`toolsets["explore"]` 只读工具集 |
| 自定义子代理 memory/skills/mcpServers | `agent/agent_defs.py:AgentDefinition`（3 字段）+ `tools/delegate_tool.py:_run_child`（独立记忆目录 ~/.OmniMate/.agent-memory/<name>/ + 预装技能 + mcp_server_filter） |
| ToolSearch（MCP lazy schema） | `tools/tool_search_tool.py` + `tools/registry.py:get_catalog_entry` + `model_tools.py:get_tool_definitions`（拆 built-in 完整/mcp__ 精简目录） |
| Skills context:fork | `agent/skill_fork.py:run_skill_in_fork`（slash 触发 + load_skill 提示）；scan_skill_commands 读 frontmatter context 字段 |
| Hooks 18 种事件 | `agent/hooks.py:HookEvent`（11 核心 + round3 加 POST_TOOL_USE_FAILURE/SUBAGENT_START+STOP/TASK_CREATED+COMPLETED/PERMISSION_REQUEST+DENIED）|
| /rewind 4 模式 | `cli.py:_handle_rewind_command`（全恢复/只对话/只代码/从此压缩）|
| OS 沙箱（bwrap/Seatbelt） | `agent/sandbox_runner.py` + `tools/terminal_tool.py`（_handle_terminal sandbox 注入） |
| reflection reference 型 | `agent/reflection.py:REFLECTION_PROMPT_TEMPLATE`（4 类：user/feedback/project/reference）|
| time-based MC（60min 清旧 tool result） | `agent/context_pipeline.py:time_based_clear_old_tool_results`（在 `compress_if_needed` 最早跑，无 token 检查；`_timestamp` 在 `_assemble_turn_messages` 加，主循环 strip 后发给 LLM）|
| 落盘精细化（per-tool 50K + per-message 200K + 决策冻结） | `agent/context_pipeline.py:offload_large_tool_results` + `_enforce_per_message_budget` + `_offload_decisions`（LRU 1000）+ `reset_offload_decisions`（AIAgent.__init__ 调）|
| 9 段式 LLM 摘要（+ PTL 重试 + 熔断器 + session_memory） | `agent/context_compressor.py:SUMMARIZE_PROMPT_9SECTION` + `_summarize_conversation`（9 段 + PTL 重试 3 次 + 熔断 3 次失败）+ `reset_compact_circuit_breaker`（AIAgent.__init__ 调）|
| prompt cache 检测（12 维度 + break 根因） | `agent/cache_monitor.py:record_prompt_state` / `check_cache_break` / `notify_compaction`（llm_compact + reactive_compact 末尾调）/ `reset_cache_monitor`（AIAgent.__init__ 调）；hook 在 `_call_llm_with_escalation` 流式+非流式汇合点；`/cache-stats` 命令看统计 |
| async 子代理工具白名单（CCAR5-F） | `toolsets.py:ASYNC_AGENT_ALLOWED_TOOLSETS` + `ASYNC_AGENT_DISALLOWED_TOOLS`；应用在 `tools/delegate_tool.py:_delegate_async`（toolsets 取交集 + config.disabled_tools 注入）；config 开关 `delegation.async_tool_whitelist_enabled`（默认 True）；sync 模式不受影响 |
| worktree 变更检测 + 智能清理（CCAR5-G） | `tools/worktree.py:has_worktree_changes`（git status --porcelain；非 git 用 listdir）+ `cleanup_worktree_smart`（独立函数，有改动保留）+ `create_isolated_workspace` 的 cleanup 加 `force` 参数；`tools/delegate_tool.py:_run_child` finally 走智能 cleanup；config 开关 `delegation.worktree_always_cleanup`（默认 False=智能） |
| fork 子代理路径（CCAR5-H cache-identical） | `agent/fork_messages.py:build_forked_messages`（父 assistant turn + placeholder tool_result + directive）+ `build_forked_system_prompt`（父字节 + FORK MODE 标记）；接入 `tools/delegate_tool.py:_run_child` fork 分支（fail-open：构造异常 fallback 非 fork）；AIAgent 加 `initial_messages` 参数；subagent schema 加 `fork` 字段；config 开关 `delegation.fork_subagent_enabled`（默认 True）+ `fork_max_parent_turns`（默认 3） |
| post-compact 主动恢复（最近文件 + invoked skills） | `agent/post_compact_recovery.py:build_post_compact_brief`（在 `_run_context_compression` 末尾注入 ephemeral user 消息；fail-open；config 开关 `post_compact_recovery_enabled`）|
| partial compact（双向 from / up_to） | `agent/context_compressor.py:_summarize_conversation` 加 `from_idx/up_to_idx` 参数；`agent/context_pipeline.py:llm_compact` 拼装 head + [summary] + tail + `_fix_tool_call_pairs` 兜底；`compact` 工具 schema 加 2 字段 |
| reactive_compact 多次触发 + 冷却窗口 | `agent/context_pipeline.py:reactive_compact`（session_state.reactive_last_at + reactive_count；冷却 60s + 上限 5 次/会话；`reacted` 改 @property 向后兼容）；config `reactive_compact_cooldown_seconds` + `reactive_compact_max_per_session` |
| PTL tokenGap 精确算法 | `agent/context_compressor.py:_compute_ptl_drop_count` + `_get_model_max_tokens`（三格式正则 DeepSeek/Anthropic/OpenAI + 三层 fallback 到 20% 旧算法）|
| post-compact 主动恢复（最近文件 + invoked skills） | `agent/post_compact_recovery.py:build_post_compact_brief`（compact 末尾注入；走 safe_path 白名单 + fail-open）；追踪在 `_dispatch_tool_calls`（safe + unsafe 两路都调 `_record_recent`）；config `post_compact_recovery_enabled/max_files/max_skills` |
| 子代理 sidechain transcript 持久化（CCAR5-I） | `agent/subagent_persistence.py`（generate_agent_id / write_metadata / append_message / load_transcript / list_resumable / mark_completed / cleanup_old / cleanup_stale_subagents）；接入 `tools/delegate_tool.py:_run_child`（on_response 回调 + try/finally 标记 status）；cli.py 启动时清理 stale running + 过期 retention；config 开关 `delegation.subagent_persistence_enabled`（默认 True）+ `subagent_persistence_retention_days`（默认 7） |

## 已知约束（设计如此，不是 bug）

- **记忆写入后本会话不生效** —— 保护 prompt cache。`MemoryStore.snapshot_for_prompt()` 是 frozen 的。
- **首次 curator 运行被推迟** —— 种子化 `last_run_at`，等一个完整周期（避免新装就大改技能库）。
- **Memory Curator 默认推迟** —— 首次启动种子化 `last_run_at`，等一个完整周期（默认 7 天）才跑第一次，避免新装就大改记忆库。
- **`use_count=0` 不是归档理由** —— 一个 "Kubernetes 故障处理" 技能可能 2 个月不触发，但仍有价值。按内容判断，不按计数。
- **子代理不继承对话历史** —— 独立 `AIAgent` 实例，只通过 `context` 参数传递必要信息。`summary_only=True`（默认）时连结果都被压缩。
- **权限审批缓存是会话级的** —— `PermissionChecker._approved` 集合。新会话重置，避免长期信任漂移。
- **MCP 工具依赖外部进程** —— server 崩溃后工具自动隐藏（`check_fn` 返回 False），但不自动重启。
- **bypassPermissions 仍保留 fatal 底线** —— `rm -rf /` / `mkfs` / fork bomb / `dd` 覆盖磁盘在任何权限模式下都拒绝（`check_fatal_irreversible`）；bypass 只跳过审批，不是裸奔。
- **自定义子代理项目级覆盖用户级** —— `<cwd>/.claude/agents/` 同名定义覆盖 `~/.OmniMate/agents/`（与 skills 多目录优先级一致）。
- **`_skill_tool_scope` 会话内持久** —— load_skill 触发的 allowed/disabled tools 作用域当前无清除机制（技能切换覆盖语义），slash 注入路径暂未接入。
- **acceptEdits 守 cwd 边界 + fatal 底线** —— cwd 内 safe-fs 命令（mkdir/touch/mv/cp/rm/del）+ cwd 内写入自动批；shell 复合操作符（&&/||/;/|/反引号/$()）一律交原闸门；`rm -rf /` 等系统级破坏在任何模式都拒。
- **MCP schema 按需加载（ToolSearch）** —— mcp__ 工具默认只发精简目录条目（name+描述+hint），LLM 调 `tool_search(query)` 取详细参数；built-in 工具仍发完整 schema。
- **子代理 memory 独立目录** —— `memory: true` 时子代理记忆写到 `~/.OmniMate/.agent-memory/<name>/`，与主记忆库隔离，不参与 curator 维护。
- **mcp_server_filter 仅 schema 层** —— 自定义子代理 `mcpServers` 字段只过滤 LLM 可见 schema，registry 仍注册全部 MCP 工具（手动 dispatch 仍命中，对齐 Claude Code 语义）。
- **context:fork 同步等待** —— 技能子代理跑完才回主循环（对齐官方 `background:false`）；子代理用 minimal 工具集，spawn_depth+1 防递归。
- **Hooks 通知型事件 fail-open** —— round3 加的 7 个事件都是通知型，hook 异常只 log 不影响主流程。
- **子代理 transcript 落盘 fail-open** —— Task I 加的 sidechain transcript 持久化（`~/.OmniMate/.agent-sessions/`）所有操作 try/except，写盘失败不影响主流程；默认开（`delegation.subagent_persistence_enabled`），7 天 retention 清理；Phase 2 才做 `subagent_resume` 工具。

## 测试策略

- **按模块组织**：`tests/test_{basic,memory,skills,curator,sessions,context,delegation,config,integration,permission,llm_retry,worktree,mcp,task_system,agent_defs,web_search,hooks,time_based_mc,offload_refined,summarize_9section,cache_monitor,post_compact_recovery,partial_compact,reactive_compact,subagent_persistence}.py`
- **集成**：`tests/test_integration.py` 用 mock OpenAI client 跑完整对话流程（含工具调用、记忆注入、中断）
- **验证脚本**：`scripts/verify.py` 跑 11-scaffold.md 的 22 项检查清单，适合改完代码后快速回归（不含 P0-P3 新功能测试）
- **新增功能必加测试**：每个新模块（permission/mcp/task_store/agent_defs/web_search 等）都有独立测试文件，改完跑 `uv run pytest tests/` 确认无回归
