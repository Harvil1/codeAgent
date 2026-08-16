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

## 韧性机制（重试 + 备用模型 + max_tokens 升级 + 529 早切 + 看门狗 + 扣留恢复）

`agent/llm_retry.py:call_with_retry` 被 `AIAgent.run_conversation` 用于每次 LLM 调用：

- 可重试错误（429 限流、5xx 服务器错误、连接/超时）：指数退避重试 5 次（1s → 2s → 4s → 8s → 16s），尊重 `Retry-After` header
- **退避加抖动（P0-1）**：每次 sleep = base + uniform(0, base × 0.25)，多实例并发遇到 429 时避免雷击。`jitter_ratio=0` 关闭抖动（向后兼容）
- **529 连续失败早切（P1-1）**：连续 `consecutive_529_threshold`（默认 3）次 529 立即切备用 client，不浪费剩余重试次数（Anthropic 过载通常持续一段时间）
- 不可重试错误（400 参数、401 认证、403 权限）：立即抛（例外：R17 #13 的 400 溢出自适应）
- 主模型重试耗尽后，如果配置了 `fallback_model`，用备用模型再试一次
- AIAgent 构造参数：`AIAgent(..., fallback_model="deepseek-reasoner")`
- **max_tokens 升级（P0-3 / R17 #10）**：LLM 返回 `finish_reason="length"`（max_tokens 截断）时，先升 `max_tokens` 到 64000 用非流式重试一次（不打断思路）。`MaxTokensEscalator` 整个会话幂等（最多升 1 次）。入口：流式 `agent/__init__.py:_call_llm_streaming` 末尾 + 非流式 `agent/__init__.py:run_conversation` 非流式分支
- **续写恢复（R17 #10）**：升级后仍截断的纯文本响应 → `_recover_output_truncation` 把截断内容 + 续写 meta 追加到**局部请求视图**再调 LLM 拼接（最多 `llm.output_recovery_limit`=3 次）；成功以拼接完成的单条 assistant 消息入史
- **400 溢出自适应（R17 #13）**：400 报文含 `input + max_tokens > context limit` 数值 → 下调到 `limit - input - 1000`（下限 3000）立即重试（最多 2 次不耗正常计数）；输入本身太大照旧抛
- **unattended 退避帽（R17 #44）**：持久重试模式下 `_compute_backoff` 帽 5min（普通 60s）
- **流空闲看门狗（R17 #12）**：流式 90s 无 chunk → 中止流转 `LLMStreamIdleTimeout`（`llm.stream_idle_timeout_seconds` 配置，0 禁用）
- **扣留-恢复（R17 #9）**：流空闲超时 → 转非流式重试一次；PTL → 一律 reactive_compact 恢复（冷却/上限防循环）；恢复失败才透出（`_last_llm_error_kind` 分类）

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
| 路径白名单 | `agent/permission.py:safe_path`（直接调用方）+ `PermissionChecker.check_path` 闸门 3（write_file/str_replace，CCAR13 Task 4 起同源白名单） |
| LLM 重试/备用模型/退避抖动/529 早切/unattended 退避帽 | `agent/llm_retry.py:call_with_retry` + `_compute_backoff`（抖动 + `max_backoff` 参数：普通 60s / unattended `UNATTENDED_MAX_BACKOFF`=300s）+ 连续 529 计数 |
| max_tokens 升级（finish_reason=length 自动重试，64k） | `agent/llm_retry.py:MaxTokensEscalator`（`DEFAULT_ESCALATED_MAX_TOKENS`=64000）+ `detect_length_finish`；入口 `agent/__init__.py:_call_llm_streaming`（流式）和 `run_conversation` 非流式分支 |
| 400 溢出自适应（R17 #13） | `agent/llm_retry.py:parse_context_overflow`（CC 精确 + OpenAI 经典双正则）+ `compute_overflow_max_tokens`（缓冲 1k/下限 3000）+ `call_with_retry` 400 分支（最多 2 次不耗正常计数） |
| 续写恢复（R17 #10） | `agent/__init__.py:_recover_output_truncation`（升级后仍截断 → 局部视图续写拼接，`llm.output_recovery_limit`=3）+ `_merge_continuation_response`（拼接单条 assistant 入史） |
| 流空闲看门狗（R17 #12） | `agent/llm_client.py:_iterate_with_watchdog`（手工 `__anext__` + `wait_for`）+ `LLMStreamIdleTimeout`；两 client 流式循环接入；config `llm.stream_idle_timeout_seconds`（默认 90，0 禁用）经 cli 透传 |
| 扣留-恢复（R17 #9） | `agent/__init__.py:_call_llm_with_escalation` except 链（流空闲→非流式重试一次；PTL→一律 reactive_compact 不受 flag 门控）+ `_last_llm_error_kind` 分类字段 |
| 终止原因枚举（R17 #14） | `agent/__init__.py:LoopExitReason`（12 种，旧值保留）+ 主循环赋值点（预算/max_turns 分流、goal/idle/错误分类）+ `_handle_loop_exit`（按枚举生成消息 + PTL/流超时触发 STOP_FAILURE）+ `_emit_loop_exit_trace`（loop_exit 事件进 trace sink，含 completed） |
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
| curator 整理增强（R19 #22） | `agent/memory_curator.py:MEMORY_REVIEW_PROMPT_TEMPLATE`（五情况：+delete_falsified 删除被证伪/+normalize_dates 相对日期转绝对）+ `execute_action` 对应执行（均走软删除/原文备份改写，完全可逆） |
| 秘密扫描全链路（R19 #24） | `agent/secret_scanner.py`（11 类 gitleaks 规则，合并单正则命名组；命中只记规则 ID+50 字符截断）→ `memory_store.save/update` 拒绝写入 / `memory_curator.safe_rewrite_body` 拒绝改写 / `trace.emit` 值级 redact / `handoff._scan_for_secrets` 迁移共用 |
| 条件技能动态激活（R19 #25） | `agent/skill_commands.py:path_matches_skill_paths` + `find_conditional_skill_matches`（mtime+size 双因子缓存）+ `AIAgent._activate_conditional_skills`（ephemeral `<conditional_skills_ready>` 通知，会话级去重）；触发点 `_dispatch_tool_calls` 两处 pre-callback（read/write/str_replace 的 path）；与静态判定互补（paths 目录语义仍进索引）；glob 模式 YAML 里须加引号（* 是 alias 语法） |
| skillify 内置技能（R19 #28） | `skills/skillify/SKILL.md`（四步：分析会话→ask_user 访谈→skill_manage 保存→确认；用户纠正沉淀进"规则"段；纯 MD 零 Python） |
| 对话级记忆提取（R19 #21） | `agent/auto_extract.py:run_auto_extract`（增量轨迹→aux 单轮无工具提取→memory_store.save；与 reflection 共用 `build_memory_manifest` 防重复）+ `AIAgent._maybe_auto_extract`（游标始终推进 + 互斥 `_memory_touched_this_turn` + every_n_turns=3 节流 + spawn_depth==0）；config `memory.auto_extract`（默认关） |
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
| 自定义子代理 4 扩展字段（Task N） | `agent/agent_defs.py:AgentDefinition`（omit_claude_md / initial_prompt / required_mcp_servers / critical_reminder，frontmatter camelCase）+ `tools/delegate_tool.py:_run_child`（critical_reminder 拼 system_prompt；omit_claude_md 传 AIAgent 的 omit_project_memory；initial_prompt 前置首 user）+ `agent/prompt_builder.py:build_system_prompt_layers(omit_project_memory=True)` 跳过项目 OMNIMATE.md + `agent/__init__.py:AIAgent(omit_project_memory=...)` |
| Hooks 27 种事件（Task N） | `agent/hooks.py:HookEvent`（21 + Task N 加 FILE_CHANGED/CWD_CHANGED/INSTRUCTIONS_LOADED/SETUP/TEAMMATE_IDLE/ELICITATION_STARTED）；HookRegistry 加 6 对 register_/run_ 方法；FILE_CHANGED 真接入 `tools/file_operations.py:_trigger_file_changed`（write_file + str_replace 后），CWD_CHANGED 真接入 `tools/delegate_tool.py:_run_child` worktree 切换时；其余 4 事件加枚举留 follow-up |
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
| 压缩图片剥离（R18 #16） | `agent/context_compressor.py:strip_media_blocks`（content list 的 image/document 块 → `[image]`/`[document]` 文本标记，全文本合并为 str；纯 str 消息 no-op 不拷贝）；`_summarize_conversation` 与 fork 基底共用 |
| 压缩前缀复用 fork（R18 #15） | `agent/context_compressor.py:_summarize_conversation`（fork_prefix_messages/tools 参数：摘要请求 = 完整对话前缀 + 追加 9 段式指令，不传 model/max_tokens 保前缀缓存；失败降级独立路径；summary_model 配置时不用 fork）；`llm_compact`/`compress_if_needed` 透传 tools；AIAgent `_last_tool_schemas` |
| token 权威计数（R18 #17） | `agent/context_pipeline.py:estimate_tokens_hybrid`（锚点=上次主调用 usage 的 prompt+cache_read+cache_creation + 消息条数；增长时 权威值+增量粗估，回退/异常全量粗估）+ `_record_llm_usage(sent_message_count)` 记 `_last_usage_anchor`；L4 est_tokens 切换混合计数 |
| L4 触发熔断（R18 #18） | `agent/context_pipeline.py:MAX_CONSECUTIVE_L4_FAILURES=3` + `CompressionSessionState.llm_compact_failures`（降级产出算触发质量失败——`context_compressor._last_summary_degraded` 模块标志传递信号；连续 3 次降级本会话停触发 L4，真 LLM 摘要成功清零） |
| 批间摘要（R18 #19） | `agent/__init__.py:_maybe_start_tool_batch_summary`（fire-and-forget，config `context.tool_batch_summary_enabled` 默认关 + 仅主代理）+ `_generate_tool_batch_summary`（aux 一句话 ≤80 字）+ `_assemble_turn_messages` 消费注入 ephemeral `<tool_batch_summary>` |
| post-compact 主动恢复（最近文件 + invoked skills） | `agent/post_compact_recovery.py:build_post_compact_brief`（compact 末尾注入；走 safe_path 白名单 + fail-open）；追踪在 `_dispatch_tool_calls`（safe + unsafe 两路都调 `_record_recent`）；config `post_compact_recovery_enabled/max_files/max_skills` |
| 子代理 sidechain transcript 持久化（CCAR5-I） | `agent/subagent_persistence.py`（generate_agent_id / write_metadata / append_message / load_transcript / list_resumable / mark_completed / cleanup_old / cleanup_stale_subagents）；接入 `tools/delegate_tool.py:_run_child`（CCAR13 Task 3：user 指令开头 append + 独立 HookRegistry 的 POST_LLM_CALL 程序式 hook 每轮 append assistant 文本，on_response 最终响应 append 已删 + try/finally 标记 status）；cli.py 启动时清理 stale running + 过期 retention；config 开关 `delegation.subagent_persistence_enabled`（默认 True）+ `subagent_persistence_retention_days`（默认 7） |
| check_path 白名单语义（CCAR13 Task 4） | `agent/permission.py:PermissionChecker.check_path` 闸门 3 恢复白名单判定（write：workspace cwd / `~/.OmniMate` / `/add-dir` extra roots 之内放行、之外拒；白名单经 `default_allowed_roots()` 与 safe_path 同源）；顺序铁律：闸门 1 受保护路径 / 闸门 2 项目代码写保护在前（加 home 根也写不了 `~/.ssh`）；bypassPermissions 跳白名单不跳闸门 1/2；write_file + str_replace 共用此判定 |
| async 子代理默认拒审批（CCAR6-J） | `agent/permission.py:PermissionChecker`（permission_mode="autoDeny" 第 4 模式，闸门 2 短路 + 保留 fatal/safe-fs 底线）；`_delegate_async` 注入；config `async_auto_deny_permission`（默认 True）；`_run_child` 优先级链 custom_def > 注入 > default；`_common.py` mode 白名单含 autoDeny（关键：singleton checker 走 mode_override 不走 self.mode） |
| 子代理中断完整化（CCAR6-K） | `tools/delegate_tool.py:_delegate_sync`（threading.Event 替代 timeout+abandon）+ `_delegate_async`（cancel_event + `_async_tasks` 注册表）+ `_delegate_batch`（KeyboardInterrupt 传播）；`agent/__init__.py:run_conversation(cancel_event=)` 每轮检查 + `_extract_partial_result`（[PARTIAL] 前缀保留最后 assistant 消息）；`subagent_kill` 工具（core toolset）；config `sync_cancel_timeout_seconds` + `async_kill_enabled` |
| Goal 驱动系统（CCAR8，同步阻塞） | `agent/goal.py:GoalState`（状态机 + evaluate_after_turn + 原子持久化 `~/.OmniMate/.goal/current.json`）+ `agent/__init__.py`（goal continue 分支 + `_pending_ephemeral_messages` 暂存 + 网络 pause）；CLI `cli.py:/goal`（status/pause/resume/continue/clear/tasks）；config `goal.default_token_budget`（200K 必设防无限跑）+ `reflection_interval` |
| LocalMemoryRecall（CCAR8） | `tools/memory_recall_tool.py`（async handler，`agent_ref.aux_llm_router` 取 client + model=None 对齐 `_auto_recall_memory`）；不污染 snapshot_for_prompt（frozen） |
| Channels MCP 推送（CCAR8） | `agent/channel_inbox.py:ChannelInbox`（落盘 `~/.OmniMate/.inbox/` + format_digest）+ `agent/mcp_client.py:MCPTransport.set_notification_handler` + `StdioTransport._reader_loop`（daemon 线程懒启动：response→queue，notification→handler；lock 只护 stdin 写防死锁）；CLI `/inbox` |
| Poor Mode（CCAR8） | `agent/poor_mode.py:apply_poor_preset`（7 flag 全关：reflection/9段摘要/auto memory/cache监控/verification/curator/reactive_compact）+ `cli.py:/poor`；runtime 不写盘；不关 fatal 底线 |
| BriefTool（CCAR8） | `tools/brief_tool.py`（echo 型格式约定工具，core toolset，isConcurrencySafe=True） |
| crossProjectResume（CCAR8） | `agent/cross_project.py`（list_recent_bundles_across_projects + auto_save_current_session）+ `agent/handoff.py`（HandoffBundleMeta 加 source_cwd/auto_saved + save 加 2 kwargs，旧 bundle 向后兼容）；CLI `/resume_bundle`（/resume 被会话恢复占用） |
| teammateMailbox（CCAR8） | `agent/team/mailbox.py:Mailbox`（异步队列：send/check_unread/check_all/mark_read/clear，复用 bus._with_lock）+ `tools/mailbox_tool.py`（3 工具，`_resolve_mailbox_ctx` 从 agent_ref 取）；CLI `/mailbox`；与 MessageBus 分工：异步 fire-and-forget vs 同步 request-response |
| Trace 本地 sink（CCAR8） | `agent/trace.py:TraceSink`（daily jsonl `~/.OmniMate/.trace/` + emit/query/summary，fail-open）+ `_register_trace_hooks`（6 hook 点：pre/post_llm_call + post_tool_use/failure + subagent_start/stop）；AIAgent 构造参数 `trace_sink=`；CLI `/trace today|yesterday|<date>|tail N`；config `trace.enabled/retention_days` |
| 工具 handler dispatch 契约（CCAR8 教训） | `tools/registry.py:dispatch` 调 `handler(args, **dispatch_kwargs)`——工具参数从 `args` 取，命名上下文（memory_store/agent_ref/hooks_registry 等）从 kwargs 取；新工具签名必须 `(args, **kwargs)`，契约测试 `test_handler_signature_matches_dispatch_contract`（inspect.signature 验 VAR_KEYWORD）防 silent-dead-code |
| 记忆分层项目隔离（CCAR9） | `agent/project_scope.py:get_project_memory_key`（canonical git root 用 `--git-common-dir`，worktree 归一，非 git 退 cwd，fail-open）+ `agent/memory_store.py` 按 type 路由（user/feedback/other → 全局 `.memory/`；project/reference → `.memory/projects/<key>/`）；MEMORY.md 双节合并索引（全局 + 当前项目）；`_ensure_index_fresh` 含 cwd 切换感知（`_index_built_key`） |
| /init 生成 OMNIMATE.md（CCAR9） | `cli.py:_handle_init_command`（目录树/配置文件/类型统计 fail-open 收集 → 主 LLM 四段式生成：项目本质/常用命令/架构/约定 → 写 `<cwd>/OMNIMATE.md`；已存在不覆盖，`--force` 覆盖）；prompt_builder 递归扫注入闭环已有 |
| reflection 防重复（CCAR9） | `agent/reflection.py:run_reflection`（prompt 预注入已有记忆清单 manifest——前 100 条 name + description 前 60 字符 + "不要重复存储"提示，fail-open） |
| 检索式记忆注入（CCAR10，直接替代 snapshot） | `agent/memory_injection.py:build_relevant_memories_message`（aux_llm Top5 → `<relevant_memories>` ephemeral user；同轮缓存 LRU1 + `reset_injection_cache` 每轮）+ `_fallback_snapshot_message`（无 aux 降级，`_snapshot_injected` 一次性）+ 接入 `run_conversation` 开场（仅 spawn_depth==0）；prompt_builder/memory_manager 的 snapshot 注入已退役；旧路径 `_initial_memory_recall`/`_retrieve_relevant_memories` 已删 |
| statusline（CCAR10） | `cli.py:_render_statusline`（⚡model/会话 token（`_llm_usage_stats`）/goal:状态#轮次/项目名，每轮响应后 dim 一行）+ `_format_tokens`；RuntimeContext `_statusline_project_key`（initialize 赋一次）；config `statusline.enabled`（默认 true）；中断/异常路径不打 |
| subagent_resume（CCAR10，补 CCAR5-I Phase 2） | `tools/subagent_resume_tool.py:_run_resume`（load_transcript → `_spawn_resumed_agent`（initial_messages 重启 + spawn_depth+1 + minimal）→ append_message 续写同文件）+ CLI `cli.py:/resumable`（列表 + 恢复，Rich `\[id]` 转义）；core toolset + UNSAFE 分类 |
| Glob 工具（CCAR11） | `tools/glob_tool.py:_handle_glob(args, **kw)`（pathlib glob + mtime 降序 + 截断 + safe_path 读校验）；core toolset + SAFE 分类 |
| 缺口命令（CCAR11） | `cli.py:/compact`（确认 + 强制 L4 + 降级 snip）+ `/context`（token 分布表）+ `/status`（model/goal/MCP/项目）+ `/doctor`（6 项自诊断）+ `/diff`（checkpoint tracked_files）+ `/add-dir`（safe_path 白名单 + settings.json 持久化）+ `/paste`（PowerShell 剪贴板图片） |
| 桌面通知（CCAR11） | `agent/notifier.py:notify(title, msg)`（Windows toast 零依赖 + 30s 节流 + fail-open + config `notifications.enabled`）+ 3 触发点（bg 完成/权限审批/goal network pause） |
| 工具 schema OpenAI 格式契约（CCAR11 教训） | registry.get_definitions 直接 `{"type":"function","function":schema}` 塞 LLM——schema 键必须是 **"parameters"**（OpenAI）不是 "inputSchema"（Anthropic 风格）。契约测试 `test_all_builtin_schemas_use_openai_parameters_key` 遍历防回归（CCAR8-10 的 brief/mailbox×3/memory_recall/subagent_resume 曾用 inputSchema → 参数定义对 LLM 不可见） |
| Windows Job Objects 沙箱（CCAR12） | `agent/win_job_object.py:WinJobObject`（ctypes 零依赖：KILL_ON_JOB_CLOSE+禁逃逸+进程上限；**restype 必须显式 HANDLE** 防 x64 句柄问题）+ `sandbox_runner.attach_job` + `terminal_tool` Windows 分支（Popen 后挂 job，finally 保活 close）；诚实定位：进程管控非文件隔离，文件防线=safe_path 层 |
| cron/goal/config/worktree 工具化（CCAR12） | `tools/cron_tool.py`（CronScheduler 补 add/remove/list_jobs CRUD）+ `tools/goal_tool.py`（`agent/goal.py:start_goal_agent` 与 CLI 同源；共享函数不碰 conversation_history——goal-continue 分支自然驱动）+ `tools/config_tool.py`（白名单 7 键精确匹配 + next_session 语义键）+ `tools/worktree_tool.py`（会话级 `set_session_workspace_cwd`）|
| MCP Resources（CCAR12） | `agent/mcp_client.py` transport 基类 `list_resources/read_resource`（fail-open）+ Manager 透传 + `mcp__<server>__list_resources/read_resource` 动态注册（静态名在 mcp__ 前缀发现机制下不可见）|
| async 工具 context 契约（CCAR12 教训） | sync handler 经 `asyncio.to_thread` **拷贝 context**——handler 内 contextvar set 不回透主循环。需要跨 context 生效的工具（如切会话 cwd）必须 `async def`（dispatch 直接 await 同 task 同 context）。端到端测试必须在 dispatch 外断言（`test_enter_exit_via_registry_dispatch` 防回归）|
| skillLearning 行为学习管线（CCAR15） | `agent/skill_learning/`（store 置信度累积 → observer 四类启发信号 → evolver 簇进化 SKILL.md → llm_observer 可选后端）；主循环接线 `agent/__init__.py:_maybe_skill_learning`（轮末，仅 spawn_depth==0）；config `skill_learning` 4 键（默认关）；CLI `/skill-learning status\|start\|stop\|evolve\|prune` |
| preventSleep Windows 防休眠（CCAR15） | `agent/prevent_sleep.py`（ctypes SetThreadExecutionState + reason 引用计数 + atexit 兜底）；主循环每轮 `agent/__init__.py:_update_prevent_sleep`（goal active / bg running 判忙，转换守卫防计数无界）；config `security.prevent_sleep` 默认 True |
| skillLearning 进化门槛与隔离（CCAR15 裁决） | 只演化 global scope——项目约定类 instinct 落 project scope 仅存储不自动进化（生成到全局 skills 目录会跨项目泄漏 + 约定簇 trigger 恒同会撞 slug）；门槛簇平均 confidence ≥0.75 且 ≥3 条（config `evolve_threshold`/`evolve_min_cluster`）|
| 记忆检索年龄衰减（核心对齐 T4） | `agent/memory_store.py:full_index_text_with_age`（派生文本附 `[age: Nd]`，MEMORY.md 落盘不变）+ `agent/memory_retriever.py:annotate_index_with_age` + prompt"新记忆优先"规则；两个调用点（memory_injection / memory_recall_tool）已切换 |
| 写路径审批"总是允许"档（核心对齐 T5） | `agent/permission.py:check_path`（decision=="always" 分支：会话缓存 + 运行时白名单 + settings.json 持久化）+ `agent/settings.py:persist_extra_allowed_root/remove_extra_allowed_root` + CLI 回调 a 选项 + `/approved remove-root`；回调返回 True 仍是"本次允许" |
| 压缩边界保留段标注（核心对齐 T8） | `agent/context_pipeline.py:_build_compact_boundary`（时间/摘要覆盖/保留段三要素 + "保留段精确、摘要转述"，全量 + partial 两模式） |
| 压缩阈值增长预估（核心对齐 T1） | `agent/context_pipeline.py:estimate_turn_growth`（最近 window 轮最大单轮 token）+ `compress_if_needed` L4 判定 `est + growth >= threshold`；config `context.llm_compact_growth_window=3` / `llm_compact_growth_default=8000` |
| post-compact 恢复预算 + plan/async 状态（核心对齐 T2） | `agent/post_compact_recovery.py:build_post_compact_brief`（统一预算 `context.post_compact_recovery_budget=40000`，优先级 plan/async > 文件 > 技能）+ `_build_plan_async_state_brief`（`AIAgent._last_approved_plan` + `delegate_tool._async_tasks` 注册表补 goal/started_at） |
| 技能 files: 附件（核心对齐 T3） | `agent/skill_commands.py:read_skill_attachment_files/format_skill_attachments`（相对技能目录 + traversal 防护 + 单文件 8K（config `skills.file_attachment_max_chars`）+ 总数 5）；注入两处：`execute_skill`（slash）+ `load_skill` 返回 `attachments`；skill_view 不注入 |
| fork 全历史（核心对齐 T10） | `agent/fork_messages.py:build_forked_messages`（`full_history=True` 完整 user/assistant 流，tool result 换 placeholder，截到 `delegation.fork_full_history_max_turns=50`）；subagent `fork: true\|"full"`；T10 顺带修 fork 参数死接线（_handle_delegate_task 此前漏传 kwargs） |
| 工具可见性规则（核心对齐 T6） | `agent/tool_permissions.py`（settings.json `permissions.allow/deny`：精确名/`mcp__server__*`/整服务器；allow 豁免 deny；mtime+size 双因子缓存）；应用点 `model_tools.get_tool_definitions` + `registry.dispatch`（permission_denied） |
| 只读命令快速通道（核心对齐 T7） | `agent/permission.py:_is_readonly_command`（30+ 只读前缀表，复合命令逐段判定，重定向/`$()`/反引号即非只读；git branch/tag/remote 只收只读子形态，env 不进表）+ `check()` 闸门 1 后快速通道 + `_dispatch_tool_calls` 对只读 terminal 动态进并发组；config `security.readonly_fastpath_enabled=True` |
| plan 清上下文执行（核心对齐 T9） | `agent/__init__.py:_apply_post_plan_clear`（history 截断为 `<post_plan_brief>` + invalidate_system_prompt 重建 context 层；transcripts 落盘不动）+ 回调三元组协议 `(approved, feedback, clear_context)`（二元组向后兼容）+ CLI c 选项 |
| Windows 路径绕过检测（R16 #2） | `agent/permission.py:check_suspicious_path`（NTFS ADS 冒号（仅 win32）/8.3 短名/长路径前缀/尾点尾空格（只看最后段，豁免裸 . ..）/DOS 设备名/三连点段/UNC/波浪变体 ~user ~+ ~N/写路径禁 glob 元字符；全平台检测）+ `safe_path`/`check_path` 前置（gate=suspicious，任何模式先于保护表/白名单） |
| 双路径检查（R16 #5） | `agent/permission.py:_path_forms_for_check`（词法 normpath + realpath 双形式）+ `is_protected_path`/`is_write_protected_path` 双形式过保护表（防软链指向 ~/.ssh 等） |
| 危险删除路径判定（R16 #6） | `agent/permission.py:is_dangerous_removal_path` + `check_dangerous_removal`（rm/rmdir/del/erase/rd 目标为 裸 \*/\//家/根直接子目录/盘根(直接子目录) → 拒；接入点 bypass 之后、acceptEdits 之前；不算 fatal，不可审批解锁） |
| HTTP hook SSRF 防护（R16 #4） | `agent/ssrf_guard.py`（禁达段判定 + DNS 预检 + URL allowlist）+ `agent/hook_exec.py:run_http_hook`（环回放行/IP 直连校验/allow_redirects=False/环境代理跳过预检）；config `security.http_hook_allowed_urls`（None 不限/[] 全拒/非空 \* 通配） |
| Bash 注入面检查（R16 #1） | `agent/bash_injection.py:check_injection_surface`（$()/${}/$[]/反引号/进程替换 <() >() =(/zsh =cmd/IFS//proc/environ//dev/tcp/jq system()+危险 flag/zsh 危险 builtin+fc -e/控制字符/Unicode 空白/CR/换行分命令/反斜杠转义空白与操作符/词中 #/注释引号失步/花括号展开/flag 引号混淆；三引号视图 raw+with_dq+fully+keepq；quoted heredoc 体剥除）+ `check()` 闸门 1 后只读通道前命中升审批（gate=injection）+ `_approval_gate` 统一审批流（destructive 共用）+ `_SHELL_OPS` 追加 `${ <( >( =(（acceptEdits 不自动批）） |
| 内容级权限规则（R16 #3） | `agent/tool_permissions.py:check_command_rules`（`Bash(...)`/`Terminal(...)` 三形态：精确/x:\* 旧前缀词边界/x \* 通配（\\* 字面量、尾部单独 " \*" 匹配裸命令）；deny>ask>allow）+ `detect_shadowed_command_rules`（整级 deny/ask 遮蔽内容级 allow → 加载告警）+ `permission.py:check` 接入（闸门 0 后：deny 任何模式拒、ask 强制审批 bypass 不豁免、allow 闸门 1 后放行跳过注入面/破坏性审批，硬底线不受影响）；settings.json permissions 段新增 ask 列表 |

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
- **记忆分层项目隔离（CCAR9）** —— project/reference 类存 `.memory/projects/<canonical-git-root>/`，项目 A 的记忆在项目 B 物理不可见；user/feedback/other 全局共享。MEMORY.md 合并索引只含"当前项目"分区（切换项目后下次 rebuild 跟随，同实例有 `_index_built_key` 感知）。worktree 与主 repo 共享项目区。
- **/init 生成的是 cwd 的 OMNIMATE.md** —— 对齐 Claude Code /init；prompt_builder 递归扫注入（`_scan_project_memory_files`）是既有闭环。
- **检索式记忆注入每轮一次（CCAR10）** —— 主代理 only（spawn_depth==0）；检索结果走 ephemeral（不进 history/system prompt）；无 aux_llm_router 降级 snapshot（会话只注入一次）；memory_recall 工具仍可主动深查（含 L2 全文）。
- **subagent_resume 的轨迹语义（CCAR13 Task 3 改）** —— transcript = user 指令 + 每轮 assistant 文本（POST_LLM_CALL 每轮 append）；tool_calls / tool result 不落盘（无配对 result 会造孤儿消息 → API 400），resume 的 initial_messages 是纯 user/assistant 文本流；轮级记录受 `config["hooks"]["enabled"]` 门控（默认 True，关 hooks 只留 user 指令一条）。
- **/add-dir 持久化走 settings.json（CCAR11）** —— load_config 默认只读 settings.json（config.yaml 首启被迁走）；白名单写 config.yaml 会导致灌回 0 条。
- **check_path 闸门 3 恢复白名单语义（CCAR13 Task 4，收回 dcec556b 的放开）** —— write_file/str_replace 写 workspace cwd / `~/.OmniMate` / extra roots 之外现在会拒（之前"除项目代码外其他都可写"）；要放开走 `/add-dir` 或 bypassPermissions（后者仍守受保护路径 + 项目代码写保护底线）；受保护路径检查永远先于白名单。
- **/paste 只保存不分析（CCAR11）** —— PowerShell 读剪贴板存 `.paste/img_<ts>.png`，用户在消息中引用路径让 LLM 调 image_analyze。
- **notifier 仅 Windows（CCAR11）** —— 零依赖 PowerShell toast；非 Windows no-op；bg title 统一"后台任务"（30s 节流防刷屏）。
- **Windows 沙箱 = 进程管控（CCAR12）** —— Job Object 管子进程树（不逃逸+全树清理），不隔离文件系统；文件防线仍是 safe_path/白名单层。job 句柄必须保活到 Popen.wait 后（早关=子进程失去清理保证）。
- **goal_start/goal_resume/worktree_enter/cron_create/cron_delete 禁用于 async 子代理**（CCAR12）——goal-continue 无 spawn_depth 守卫、worktree 切换污染模块级状态；止损类（pause/clear/exit）保留自救。
- **config_set 白名单键必须有真实读取点**（CCAR12 教训）——dead key 写进黑洞还假报 runtime_applied=True 是最危险的静默失败；换键前 grep 消费方。
- **check_path 闸门 3 白名单语义（CCAR13 恢复）** —— write_file/str_replace 走 `default_allowed_roots()`（workspace cwd + ~/.OmniMate + /add-dir extra roots，与 safe_path 同源），白名单外拒（消息含允许根可引导 /add-dir）；bypassPermissions 跳白名单不跳硬底线；受保护路径/项目代码写保护先于白名单。
- **notifier bg title 带 task_id（CCAR13）** —— `后台任务:<id 前 8 位>`，30s 同标题节流不互吞；goal pause 通知集中在 GoalState.pause()（三原因一处接）。
- **check_path 闸门顺序 + 审批通道（CCAR14）** —— 顺序铁律：闸门 1（受保护）→ 闸门 2（agent 自身代码写保护，**acceptEdits 也不绕**）→ acceptEdits cwd 内放行 → bypass 短路 → 闸门 3 白名单；白名单外 default/acceptEdits 走审批 callback（批准→父目录进 `_approved_write_roots` 会话缓存；autoDeny 不问；无 callback 拒），审批前触发 PERMISSION_REQUEST hook + toast；跨会话持久化走 /add-dir。
- **hook 沙箱 Windows（CCAR14）** —— command 型 hook use_sandbox=True 在 Windows 走 Job Object（terminal 同款：不包装 Popen + attach + finally 保活）；Unix wrapper 保留；approved_paths.json 持久化机制存在但未接线（docstring 已止损）。
- **skillLearning 默认关 + 观察仅主代理（CCAR15）** —— config `skill_learning.enabled=False`，`/skill-learning start` 才开；观察/进化只在 spawn_depth==0 跑（防 feedback loop）；LLM 后端默认关（`observer="heuristic"`），熔断 3 次/冷却 30s/会话上限 20，任何失败回退启发式；整链 fail-open 不影响主对话。
- **preventSleep 引用计数语义（CCAR15）** —— `acquire(reason)/release(reason)` 按 reason 计数，归零才恢复系统休眠策略；主循环只在闲→忙/忙→闲转换时真正调（每轮无条件调会让计数无界）；中断/cancel 提前退出路径 held 残留到下轮或 atexit 兜底（保守方向：宁多醒不久睡）；非 Windows no-op。
- **L4 提前触发语义（核心对齐 T1）** —— 判定是 `est + growth >= threshold`（不是 `>`）：默认 growth 8000 意味着 ~92K 就压（100K 阈值）；增长预估取不到历史时回退保守默认，永不抛异常。
- **"总是允许"持久化走 settings.json（核心对齐 T5）** —— 与 /add-dir 同通道（`security.extra_allowed_roots`），不走 config.yaml（load_config 默认只读 settings.json，写 yaml 是断轨的）；审批回调返回 True 恒为"本次允许"（会话级），只有哨兵 `"always"` 才持久化。
- **permissions.allow 的唯一语义（核心对齐 T6）** —— allow 条目只在"豁免 deny"时生效（deny 整服务器 + allow 单工具），不是白名单模式（不在 allow 里的工具不受影响）；不搬 Bash(cmd:*) 子命令级（避免与权限闸门两套语义打架）。
- **只读表保守优先（核心对齐 T7）** —— 识别不了的形态一律不算只读（走原闸门）；`env` 不进表（`env VAR=x cmd` 可执行任意命令）、git branch/tag/remote 只收只读子形态、find 的 -delete/-exec 写形态 token 拦截。
- **plan 清上下文只清 LLM 上下文（核心对齐 T9）** —— 会话库是 append-only，恢复会话时调研消息仍在（可查可恢复，对齐"完全可逆"）；stable prompt 段保留（invalidate_system_prompt 只重建 context 层）。
- **技能附件不进 MEMORY/索引（核心对齐 T3）** —— files: 附件只在触发时注入（execute_skill / load_skill），skill_view（用户视角）与技能索引不含附件内容。
- **注入面命中是升审批不是拒（R16 #1）** —— 对齐 CC ask 语义：这些形态让"所见非所执行"但不一定是攻击，用户看到原文批准即可；与 CC 差异（无 shell-quote/tree-sitter 跳过 token 流检查、不拦普通重定向、heredoc 剥离简化、引号链状态机未全量）记录在 `agent/bash_injection.py` 模块头。
- **危险删除不算 fatal（R16 #6）** —— rm -rf /usr 这类在 default/acceptEdits/autoDeny 都拒且不可审批解锁，但 bypassPermissions 仍放行（区别于 rm -rf / 的 fatal 硬底线）。
- **SSRF 预检存在 DNS rebinding 窗口（R16 #4）** —— requests 无自定义 DNS lookup，校验（getaddrinfo）与连接之间理论上可被 rebinding 绕过（CC 用 axios lookup 把校验 IP 钉到 socket 消除了该窗口）；预检已挡配置型 hook 指向元数据/内网的绝大多数场景。环回 127/8 与 ::1 放行（本地 dev policy server 是 http hook 主流用法）。
- **内容级规则 bypass 边界（R16 #3）** —— deny 任何模式都拒（用户显式 deny 是最高意图）、ask 强制审批 bypass 不豁免；但 allow 只跳过审批类闸门，fatal/黑名单/危险删除硬底线不受影响。前缀匹配是词边界（`build:*` 不匹配 `build/`，对齐 CC）。
- **路径 suspicious 检查在任何模式都拒（R16 #2）** —— NTFS ADS/短名/尾点等形态即使 bypassPermissions 也拒（安全底线，对齐受保护路径语义）；裸 `.`/`..` 目录引用豁免尾点检查（glob 默认 path=. 不误伤）。
- **PTL 恢复不受 reactive_compact flag 门控（R17 #9 行为变化）** —— prompt_too_long 是可恢复错误，一律先走 reactive_compact 扣留恢复（冷却/上限防循环在函数内部生效）；Task P1.2 的 feature flag 语义废弃（韧性基线不是可选功能）。
- **升级 64k 依赖 400 自适应兜底（R17 #10/#13 联动）** —— 小输出上限 provider（DeepSeek 8K）对 max_tokens=64000 会报 400 溢出，`parse_context_overflow` 解析后动态下调重试；升级调用失败本身 fail-open 沿用截断响应。
- **续写恢复是局部请求视图（R17 #10）** —— 截断 assistant + 续写 meta 只进当次 API 请求，不进 conversation_history；成功后以拼接完成的**单条** assistant 消息入史（会话记录干净）。只处理纯文本截断；工具调用截断形态原样返回。
- **看门狗转非流式只重试一次（R17 #12/#9）** —— 流空闲 90s 中止后扣留转非流式 call_with_retry 一次，仍失败才透出（kind=stream_idle）；CC 的半超时 warning/停顿计数遥测无对应通道，只做超时 abort。
- **fork 摘要的前缀一致性边界（R18 #15）** —— fork 用上一轮 tool_schemas（压缩在轮边界、本轮 schema 未组装；plan_mode 切换轮 miss 一次可接受）；summary_model 显式配置时不 fork（专用小模型不同缓存空间，用户配置优先）；有图片消息时 strip 后 miss（避免压缩调用自身 PTL 优先）。
- **权威 token 锚点是保守偏高（R18 #17）** —— prompt+cache_read+cache_creation 之和在 OpenAI 语义下重复计 cache read（prompt_tokens 已含）——宁早压方向；锚点只在主调用记录（恢复链调用不更新）。
- **批间摘要默认关（R18 #19）** —— 注入改变发给 LLM 的消息内容（影响行为），保守默认；仅主代理（spawn_depth==0）；摘要只保留最新一批（后到覆盖）。
- **L4 触发熔断的失败=降级产出（R18 #18）** —— `_summarize_conversation` 永不抛异常（规则总结兜底），触发层失败信号走模块级 `_last_summary_degraded`（降级压缩可用但有损，连续 3 次停触发——对齐 CC autocompact 失败即停）。
- **记忆写入命中秘密即拒绝（R19 #24）** —— memory_store.save/update 是 fail-closed（ValueError，工具层转 error 返回）；curator 改写产物命中拒绝保留原文；trace 是 fail-open redact（日志通道不拒）。规则 ID 之外不记录命中值。
- **auto_extract 默认关 + 与主写入互斥（R19 #21）** —— 每回合 aux 调用有成本，`memory.auto_extract.enabled=False` 默认；本轮 LLM 调过 memory save/update → 跳过并推进游标（主 agent 优先）；游标始终推进——被互斥/节流跳过的回合不再回看。
- **条件技能 paths 的双语义（R19 #25）** —— 目录形态（`src/**`）匹配 cwd 时仍进静态索引（既有行为）；文件 glob（`*.py`）只走动态激活（ephemeral 通知）。glob 模式在 frontmatter 里必须加引号（YAML 的 `*` 是 alias 语法，不加引号解析成空串）。

## 测试策略

- **按模块组织**：`tests/test_{basic,memory,skills,curator,sessions,context,delegation,config,integration,permission,llm_retry,worktree,mcp,task_system,agent_defs,web_search,hooks,time_based_mc,offload_refined,summarize_9section,cache_monitor,post_compact_recovery,partial_compact,reactive_compact,subagent_persistence}.py`
- **集成**：`tests/test_integration.py` 用 mock OpenAI client 跑完整对话流程（含工具调用、记忆注入、中断）
- **验证脚本**：`scripts/verify.py` 跑 11-scaffold.md 的 22 项检查清单，适合改完代码后快速回归（不含 P0-P3 新功能测试）
- **新增功能必加测试**：每个新模块（permission/mcp/task_store/agent_defs/web_search 等）都有独立测试文件，改完跑 `uv run pytest tests/` 确认无回归
