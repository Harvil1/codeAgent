# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目本质

本项目后期会做成 win 电脑的软件安装包（现在不做）。
OmniMate 是基于 `D:\project\hermes-agent-main\replication-guide\` 复刻指南实现的自学习 AI Agent，并借鉴 Claude Code 的工程实践做了取长补短。**"越用越聪明" 不是一个营销词，它是由三个独立子系统 + 后台维护工人支撑的工程闭环**：

```
                Agent 核心（async 主循环 + 工具分发 + 消息历史）
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

横向还包裹一层**安全与韧性基线**：权限闸门 · 路径白名单 · 输出截断 · OS 沙箱 · LLM 重试退避/备用模型 · 持久化任务图 · MCP 外部工具 · worktree 隔离。

现状：262 个 Python 文件，2900+ 测试用例，22 项检查清单全 PASS。

## 关键设计原则（修改代码前必读）

这六条贯穿整个项目，理解它们比理解任何单个模块重要：

1. **核心是窄腰，能力在边缘** —— 每个发给 LLM 的工具 schema 在每次 API 调用中重复传输，直接增加 token 成本。能做成技能（数据）就不要做成工具（代码）。扩展优先级：`现有代码 < CLI/技能 < check_fn 门控工具 < 插件 < MCP < 新核心工具`。
2. **Prompt Caching 神圣不可侵犯** —— system prompt 在**会话开始时构建一次**（`AIAgent._get_system_prompt` 缓存），中途任何修改都会让前缀缓存失效、成本翻倍。记忆写入立即落盘，但**下次会话才注入**。唯一例外是上下文压缩（`context_compressor`）。
3. **完全可逆** —— 自动归档系统**永不删除**，只能移到 `.archive/`。Task System 用软删除（`status="deleted"`），文件保留可恢复。这让 agent 敢于自动管理自己的知识库。
4. **用户意图优先于算法** —— `pinned` 的技能免疫所有 curator 自动转换。权限闸门里用户审批的命令进会话缓存不重复询问。算法可以错，用户的明确判断不能被覆盖。
5. **发现 ≠ 可见** —— 工具注册（登记到 `registry`）和暴露给 LLM（`resolve_toolset` → `get_definitions`）是两步。`check_fn` 让工具根据运行时环境动态出现/消失（MCP 工具用这个机制：server 断开时自动隐藏）。
6. **安全默认 > 事后补救** —— `terminal` 工具默认走权限闸门（黑名单 → 规则 → 审批），`write_file` 默认走路径白名单（cwd + `~/.OmniMate`）。要"放开"必须显式配置，不能默认放开。

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
agent/mcp_client.py     ← MCP 客户端（stdio/HTTP 多传输）
tools/mcp_tool.py       ← 把 MCP 工具动态注册到 registry
    ↑
model_tools.py          ← handle_function_call + get_tool_definitions
    ↑
agent/__init__.py       ← AIAgent 主类（run_conversation async 主循环）
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
| 注入位置 | 工具 schema | user 消息（触发时） | 检索式 ephemeral 注入 | 工具调用 |

技能是数据不是代码——这让 LLM 能通过 `skill_manage` 工具自己创建和改进，是自学习的关键。

## 安全机制

修改命令执行/文件读写相关代码必读，详见 `agent/permission.py`。

```
terminal 命令执行：PermissionChecker.check() →
  内容级规则（deny>ask>allow，Bash(...)/Terminal(...) 三形态）
  闸门 1: 硬拒绝黑名单（rm -rf /、sudo、mkfs、fork bomb 等，任何模式都拒）
  闸门 1 后: 路径绕过检测（NTFS ADS/8.3 短名/尾点等）+ Bash 注入面检查（命中升审批）
  只读快速通道（30+ 只读前缀；识别不了的一律不算只读）
  破坏性/未知 → 用户审批 callback（会话内缓存已批准命令）

check_path 闸门顺序（铁律）：受保护路径 → agent 自身代码写保护（acceptEdits 也不绕）
  → acceptEdits cwd 内放行 → bypass 短路 → 白名单（cwd + ~/.OmniMate + /add-dir extra roots，
  与 safe_path 同源）；白名单外走审批 callback，持久化走 /add-dir

safe_path: 双路径形式（normpath + realpath）过保护表，防软链指向 ~/.ssh 等
危险删除判定：rm/rmdir/del 目标为裸 //、/、家、根直接子目录 → 拒（不算 fatal）
terminal 输出：超过 50000 字符截断
```

⚠️ **不要为了"方便"绕过这些检查**。如果某工具确实需要写到 cwd 外，通过 `kwargs` 接收 `omnimate_home` 并在 safe_path 的 `allowed_roots` 里显式声明。

### OS 沙箱（/sandbox on|off|status）

terminal 命令执行叠加的硬防线，**叠加层**不替换权限闸门：Linux 用 Bubblewrap，macOS 用 Seatbelt，Windows 用 Job Object（**进程管控非文件隔离**，文件防线仍是 safe_path 层；job 句柄须保活到 wait 后）。fail-open：不可用时警告降级；GUI 命令跳过；网络不隔离。默认可写 `cwd` + `~/.OmniMate`（`security.sandbox_writable_roots` 扩展）。实现在 `agent/sandbox_runner.py` + `agent/win_job_object.py`（ctypes，restype 必须显式 HANDLE）。

## 韧性机制

`agent/llm_retry.py:call_with_retry` 用于每次 LLM 调用：

- 可重试错误（429/5xx/连接超时）：指数退避 5 次（1s→16s）+ 抖动 + `Retry-After`；unattended 模式退避帽 5min（普通 60s）
- 529 连续 3 次早切备用模型（`fallback_model`）；主模型重试耗尽后备用模型再试一次
- `finish_reason=length` → max_tokens 升 64k 幂等重试；仍截断 → 续写恢复（局部请求视图拼接，最多 3 次，成功以单条 assistant 入史）。小输出上限 provider 会 400 溢出，靠 400 自适应（解析 context limit 下调 max_tokens）兜底
- 流式 90s 无 chunk → 看门狗中止转非流式重试一次（`llm.stream_idle_timeout_seconds`，0 禁用）
- PTL（prompt_too_long）→ 一律 reactive_compact 扣留恢复（冷却 60s + 上限 5 次/会话，不受 flag 门控）；恢复失败才透出（`_last_llm_error_kind` 分类）

## 任务追踪（Task System）

| 层 | Task System |
|---|---|
| 文件 | `agent/task_store.py` + `tools/task_tools.py` |
| 持久化 | `~/.OmniMate/.tasks/{id}.json`（跨会话） |
| 依赖 | DAG（`blocked_by` + `can_start` + `find_ready`） |
| 约束 | 状态机 pending→in_progress→completed |

## 扩展机制

- **MCP**：`~/.OmniMate/.mcp.json` 配置外部 server，`tools/mcp_tool.py:initialize_mcp` 连接注册。工具以 `mcp__<server>__<tool>` 前缀暴露，`check_fn` 在 server 断开时自动隐藏；默认只发精简目录，LLM 调 `tool_search` 取详细 schema。
- **worktree 隔离**：`tools/worktree.py:create_isolated_workspace`。`subagent(isolated_workspace=True)` 在独立 git worktree 或临时目录跑；清理是智能的（有改动保留）。
- **load_skill**：两级加载。system prompt 只放技能索引，LLM 按需调 `load_skill(name)` 取正文。区别于 `skill_view`（用户视角，含 frontmatter）。
- **summary_only**：`subagent(summary_only=True)`（默认）时结果超 500 字压缩成 300 字摘要。

## 常用命令

```bash
# 运行
uv run python main.py                   # 交互模式
uv run python main.py chat "你好"       # 一次性问答

# 测试
uv run pytest tests/ -v                 # 全部测试（2900+ 用例）
uv run pytest tests/test_permission.py  # 单个模块（如权限）

# 复刻检查清单验证（原 22 项功能）
uv run python scripts/verify.py

# Curator 维护
uv run python -m curator_cli status | run --dry-run | pin <skill-name>

# 会话移交
/handoff save "标题" | list | load <id> | export <id> <path> | import <path>

# 依赖管理（必须用 uv，不要用 pip）
uv add <包名> | uv add --dev <包名> | uv sync
```

## 约定

- **语言**：所有注释、文档、commit message、计划、回复使用中文。代码标识符用英文。
- **依赖管理**：不要用 `pip install`，也不要手改 `pyproject.toml` 的 `dependencies`，统一 `uv add`。
- **文件 I/O**：必须指定 `encoding="utf-8"`（Windows 默认 cp1252 会乱码）。ruff 规则 `PLW1514` 强制。
- **命令执行安全**：不要绕过 `PermissionChecker` 和 `safe_path`。新增工具需要写文件时，通过 `omnimate_home` 参数 + `safe_path(write=True, allowed_roots=[...])` 显式声明。
- **默认 provider**：DeepSeek（`base_url=https://api.deepseek.com/v1`，模型 `deepseek-chat`，env `DEEPSEEK_API_KEY`）。在 `config.yaml` / `.env` 切换其他 OpenAI 兼容 provider。
- **agent home**：默认 `~/.OmniMate`，可用 `OMNIMATE_HOME` 环境变量覆盖（profile 隔离机制）。
- **工具结果契约**：所有 handler 返回 JSON 字符串，错误用 `{"error": "...", "error_type": "..."}`。
- **工具 handler 签名必须是 `(args, **kwargs)`** —— 工具参数从 `args` 取，命名上下文（memory_store/agent_ref/hooks_registry 等）从 kwargs 取（`tools/registry.py:dispatch`）。签名不符 = silent dead code，有契约测试防回归。
- **工具 schema 参数键必须是 `"parameters"`**（OpenAI 格式，不是 Anthropic 的 `"inputSchema"`）——历史上踩坑 5 次（参数定义对 LLM 不可见），有契约测试防回归。
- **需要跨 context 生效的工具必须 `async def`** —— sync handler 经 `asyncio.to_thread` 拷贝 context，contextvar set 不回透主循环（如切会话 cwd）；dispatch 直接 await 同 task 同 context。
- **新工具必须同步登记 concurrency 分类清单**（safe/unsafe），否则并发分组不生效。

## 关键代码位置

详细实现细节（config 键/默认值/设计权衡）看各文件模块头注释 + `config.py:DEFAULT_CONFIG`；轮次出处看 git log。

| 想修改什么 | 看这里 |
|---|---|
| 对话主循环 / 中断 / grace call | `agent/__init__.py:run_conversation` |
| system prompt 构建（记忆/技能索引/GUIDANCE） | `agent/prompt_builder.py:build_system_prompt` |
| 上下文压缩管线（5 层 + 编排 + token 混合计数 + L4 熔断） | `agent/context_pipeline.py:compress_if_needed` |
| 9 段式 LLM 摘要 / partial compact / PTL tokenGap / fork 前缀复用 / 图片剥离 | `agent/context_compressor.py` |
| post-compact 主动恢复（文件/技能/plan/async 状态，统一预算） | `agent/post_compact_recovery.py:build_post_compact_brief` |
| prompt cache 检测（12 维 + break 根因，/cache-stats） | `agent/cache_monitor.py` |
| 流式并发执行（safe 预执行，默认关灰度） | `agent/streaming_executor.py:StreamingToolExecutor` |
| 命令权限闸门 / 权限模式（default/acceptEdits/bypass/autoDeny） | `agent/permission.py:PermissionChecker.check` |
| 路径白名单 / 路径绕过检测 / 双路径检查 / 危险删除 | `agent/permission.py:safe_path` + `check_path` + `check_suspicious_path` |
| bash AST 解析（bashlex wrapper） | `agent/bash_ast.py:parse_info` |
| 只读快速通道 / Bash 注入面检查 |`agent/permission.py:_is_readonly_command` + `agent/bash_injection.py` |
| 内容级权限规则 / 工具可见性（permissions.allow/deny） | `agent/tool_permissions.py` |
| SSRF 防护（http hook） | `agent/ssrf_guard.py` + `agent/hook_exec.py:run_http_hook` |
| LLM 重试 / 备用模型 / 退避 / 529 早切 | `agent/llm_retry.py:call_with_retry` |
| max_tokens 升级 + 400 溢出自适应 | `agent/llm_retry.py:MaxTokensEscalator` + `parse_context_overflow` |
| 续写恢复 / 扣留-恢复 / 终止原因枚举 | `agent/__init__.py:_recover_output_truncation` + `_call_llm_with_escalation` + `LoopExitReason` |
| 流空闲看门狗 | `agent/llm_client.py:_iterate_with_watchdog` |
| 工具注册模式（添加新工具看这个） | `tools/terminal_tool.py`（含权限集成） |
| 工具集可见性控制 / async 子代理白名单 | `toolsets.py:TOOLSETS` + `model_tools.py:get_tool_definitions` |
| 空结果保护（统一出口） | `model_tools.py:handle_function_call` |
| MCP 接入（多传输 + OAuth + Resources + Channels 推送） | `agent/mcp_client.py` + `tools/mcp_tool.py` + `agent/channel_inbox.py` |
| ToolSearch（MCP lazy schema） | `tools/tool_search_tool.py` + `tools/registry.py:get_catalog_entry` |
| 持久化任务 + DAG 依赖 | `agent/task_store.py:TaskStore` |
| 子代理委托（worktree/摘要/fork/内联 MCP/交接复审） | `tools/delegate_tool.py:_run_child` |
| 子代理中断完整化（cancel_event + subagent_kill） | `tools/delegate_tool.py` + `agent/__init__.py:_extract_partial_result` |
| 子代理 transcript 持久化 + resume | `agent/subagent_persistence.py` + `tools/subagent_resume_tool.py` |
| fork 子代理（cache-identical + 全历史） | `agent/fork_messages.py:build_forked_messages` |
| 自定义子代理 .md 定义（全部扩展字段） | `agent/agent_defs.py:AgentDefinition` + `scan_agent_defs` |
| 内置子代理（Explore/Plan/Coordinator） | `agent/builtin_agents/` |
| 配置默认值（所有参数源头） | `config.py:DEFAULT_CONFIG` |
| 复刻指南（设计权衡详解） | `D:\project\hermes-agent-main\replication-guide\` |
| 会话移交 bundle / 跨项目恢复 | `agent/handoff.py` + `agent/cross_project.py` |
| Vision / Glob / WebFetch / WebSearch / Read 双上限 / NotebookEdit | `tools/image_tool.py` / `glob_tool.py` / `web_fetch_tool.py` / `web_search_tool.py` / `file_operations.py` |
| LSP 符号导航（pylsp 门控） | `tools/lsp_tool.py` |
| 确定性工作流引擎（DSL/journal/预算） | `agent/workflow_engine.py` + `agent/workflow_journal.py` + `tools/workflow_tool.py` |
| workflow 脚本目录发现 | `agent/workflow_registry.py` |
| Plan Mode（计划 + 审批 + 清上下文执行） | `agent/__init__.py`（plan_mode 分支 + `_apply_post_plan_clear`）+ `tools/plan_mode_tool.py` |
| Cron 调度（一次性 + catch_up） | `agent/cron.py:CronScheduler` + `tools/cron_tool.py` |
| cron 任务模板发现 | `agent/templates.py` |
| Goal 驱动系统（状态机 + token 预算） | `agent/goal.py:GoalState` + `tools/goal_tool.py` |
| 后台任务 + monitor 监视器 + stall 看门狗 | `agent/background.py:BackgroundManager` |
| Hook 事件（27 种）+ 5 种 handler 类型 | `agent/hooks.py:HookEvent` + `agent/hook_exec.py:dispatch_hook` + `agent/hook_loader.py` |
| 子任务进度摘要 | `agent/progress.py:ProgressReporter` |
| 团队协作（同步 bus / 异步 mailbox） | `agent/team/bus.py` + `agent/team/mailbox.py` |
| session fork / resume 孤儿 tool 结果修复 | `agent/session_store.py:fork_session` + `cli.py:_fix_tool_call_pairs` |
| 记忆（L0/L1/L2 + 年龄衰减 + 项目隔离 + 路径安全） | `agent/memory_store.py` + `agent/project_scope.py` |
| 检索式记忆注入（直接替代 snapshot） | `agent/memory_injection.py` |
| 记忆检索工具（主动深查） | `tools/memory_recall_tool.py` |
| reflection（4 类经验 + 防重复 manifest） | `agent/reflection.py` |
| Memory Curator（状态机 + LLM review） | `agent/memory_curator.py` |
| 秘密扫描（memory/curator/trace/handoff 四处消费） | `agent/secret_scanner.py` |
| 对话级记忆提取（auto_extract） | `agent/auto_extract.py` |
| 技能（条件激活 + 动态目录发现 + files 附件 + slash 命令） | `agent/skill_commands.py` |
| 技能搜索（TF-IDF，中文分词） | `tools/skill_tools.py:_skill_search_rank` |
| skillify 内置技能（会话沉淀成技能） | `skills/skillify/SKILL.md` |
| skillLearning 行为学习管线 | `agent/skill_learning/` |
| Skills context:fork | `agent/skill_fork.py:run_skill_in_fork` |
| OS 沙箱（bwrap/Seatbelt/Win Job Object） | `agent/sandbox_runner.py` + `agent/win_job_object.py` |
| Poor Mode（7 flag 全关省 token） | `agent/poor_mode.py` |
| Trace 本地 sink（/trace） | `agent/trace.py:TraceSink` |
| 输入历史 + 粘贴引用协议 | `agent/input_history.py` |
| 队列命令消费（不打断当前响应） | `cli.py` 输入 daemon 线程 + `AIAgent._drain_queued_input` |
| statusline / CLI 命令（/rewind /compact /context /status /doctor /diff /add-dir /paste /history /init …） | `cli.py` |
| 桌面通知（Windows toast） | `agent/notifier.py` |
| preventSleep（Windows 防休眠） | `agent/prevent_sleep.py` |
| scratchpad 涂鸦区 + coordinator | `agent/scratchpad.py` + `agent/builtin_agents/coordinator.md` |
| brief（echo 型格式约定工具） | `tools/brief_tool.py` |
| R24 裁决不补 3 项 | #29 MCP 技能（`skill://` 无 server 生态）/ #43 jobs 模板（skill_bundle + Task System 已覆盖）/ #47 主会话后台化（与输入线程/流式预执行耦合风险大，bg_task + goal continue 已覆盖） |
| 审批前缀规则派生（curated 表） | `agent/command_prefix.py:derive_approved_prefix` |

## 已知约束（设计如此，不是 bug）

### 全局

- **记忆写入后本会话不生效** —— 保护 prompt cache。检索注入走 ephemeral 不进 history。
- **首次 curator 运行被推迟** —— 种子化 `last_run_at` 等一个完整周期（默认 7 天），避免新装就大改技能/记忆库。
- **`use_count=0` 不是归档理由** —— 按内容判断，不按计数（一个技能可能 2 个月不触发但仍有价值）。
- **子代理不继承对话历史** —— 独立 `AIAgent` 实例，只通过 `context` 参数传递必要信息。
- **权限审批缓存是会话级的** —— `PermissionChecker._approved`，新会话重置，避免长期信任漂移。
- **MCP 工具依赖外部进程** —— server 崩溃后工具自动隐藏但不自动重启。
- **bypassPermissions 仍保留 fatal 底线** —— `rm -rf /` / `mkfs` / fork bomb 在任何模式都拒；bypass 只跳过审批。
- **项目级覆盖用户级** —— `<cwd>/.claude/{agents,skills}/` 同名定义覆盖 `~/.OmniMate/`。
- **`_skill_tool_scope` 会话内持久** —— load_skill 触发的 allowed/disabled tools 无清除机制（技能切换覆盖语义）。
- **运行时可写配置的唯一通道是 settings.json** —— `load_config` 默认只读 settings.json（config.yaml 首启被迁走）；/add-dir、"总是允许"写路径都走 `security.extra_allowed_roots`，写 config.yaml 是断轨的（灌不回来）。
- **config_set 白名单键必须有真实读取点** —— dead key 写进黑洞还假报 runtime_applied=True 是最危险的静默失败；换键前 grep 消费方。

### 安全

- **acceptEdits 守 cwd 边界 + fatal 底线** —— cwd 内 safe-fs + 写入自动批；shell 复合操作符一律交原闸门。
- **危险删除不算 fatal** —— rm -rf /usr 这类不可审批解锁，但 bypassPermissions 仍放行（区别于 rm -rf / 的硬底线）。
- **注入面命中是升审批不是拒** —— 对齐 CC ask 语义（所见非所执行 ≠ 攻击）；与 CC 的实现差异记录在 `agent/bash_injection.py` 模块头。
- **内容级规则 bypass 边界** —— deny 任何模式都拒（用户显式 deny 是最高意图）、ask 强制审批 bypass 不豁免、allow 只跳审批类闸门。前缀匹配是词边界（`build:*` 不匹配 `build/`）。不搬 Bash(cmd:*) 子命令级（避免与权限闸门两套语义打架）。
- **路径 suspicious 检查在任何模式都拒** —— NTFS ADS/短名/尾点等；裸 `.`/`..` 豁免尾点检查。
- **bashlex AST 只收紧不放宽** —— deny/ask 逐段命中即命中（复合命令后半段拦得住，含命令替换体内的嵌套段）；allow 在复合命令上整串命中不生效（对齐 CC『allow 须覆盖全部段』的收紧语义）；只读正判的动词仍须在既有白名单表内；bashlex 解析失败一律回落现状正则（fail-open）。
- **SSRF 预检存在 DNS rebinding 窗口** —— requests 无自定义 DNS lookup（CC 用 axios lookup 钉死）；环回 127/8 与 ::1 放行。
- **permissions.allow 的唯一语义** —— allow 条目只在"豁免 deny"时生效（deny 整服务器 + allow 单工具），不是白名单模式。
- **只读表保守优先** —— 识别不了的形态一律不算只读；`env` 不进表；git branch/tag/remote 只收只读子形态。
- **分类器白名单剥离只影响闸门 4** —— 危险前缀白名单条目仍走正常 LLM 分类；连续 3/累计 20 拒绝本会话停用闸门 4 回落人工。
- **分类器三向 + nl_rules** —— verdict=ask 升审批不拒；confidence<0.7 一律 ask；`permissions.nl_rules` 是自然语言规则（分类器优先对照）。
- **Windows 沙箱 = 进程管控** —— Job Object 管子进程树，不隔离文件系统。
- **hook 沙箱 Windows 走 Job Object**（terminal 同款）；approved_paths.json 持久化机制存在但未接线。
- **goal_start/goal_resume/worktree_enter/cron_create/cron_delete 禁用于 async 子代理** —— 止损类（pause/clear/exit）保留自救。
- **审批前缀规则只从 curated 表派生** —— `agent/command_prefix.py` 白名单外的命令（含一切破坏性命令）保持 exact 匹配，宁可多问一次（批准 `git push origin x` 不会放行 `--force`）。
- **项目级 .mcp.json 首连审批 fail-closed** —— 未批准/无 callback（非交互）的 server 跳过不连接；批准持久化在 settings.json `mcp.approved_project_servers`（键格式 `<项目路径小写>::<server名>`）。
- **后台 LLM 调用遇 529 直接放弃** —— `call_with_retry(background=True)`（目前接线：子代理摘要）；前台语义不变。
- **inline mcpServers（agent .md）不走项目级审批** —— R24 #38 的 per-agent 内联连接与 R25 #3 的项目级 .mcp.json 威胁模型相同（clone 陌生 repo 带入），但绕过首连审批闸门（需 LLM 配合 spawn 才触发）；接线审批留 follow-up。

### 上下文与韧性

- **PTL 恢复不受 reactive_compact flag 门控** —— prompt_too_long 是可恢复错误，一律先 reactive_compact 扣留恢复（韧性基线不是可选功能）。
- **升级 64k 依赖 400 自适应兜底** —— 小输出上限 provider（DeepSeek 8K）对 64k 报 400 溢出，解析后动态下调重试。
- **续写恢复是局部请求视图** —— 截断 assistant + 续写 meta 只进当次 API 请求；成功后以拼接单条入史。只处理纯文本截断。
- **看门狗转非流式只重试一次** —— 流空闲 90s 中止后扣留转非流式一次，仍失败才透出。
- **L4 提前触发语义** —— 判定是 `est + growth >= threshold`（不是 `>`）：默认 growth 8000 意味着 ~92K 就压（100K 阈值）。
- **L4 触发熔断的失败=降级产出** —— `_summarize_conversation` 永不抛异常，连续 3 次降级本会话停触发 L4。
- **权威 token 锚点保守偏高** —— prompt+cache_read+cache_creation 在 OpenAI 语义下重复计 cache read（宁早压）。
- **批间摘要默认关** —— 注入改变发给 LLM 的消息内容，保守默认；仅主代理。
- **fork 摘要的前缀一致性边界** —— fork 用上一轮 tool_schemas（压缩在轮边界）；summary_model 显式配置时不 fork；有图片消息时 strip 后 miss。
- **边界裁剪只在有 `[COMPACT_BOUNDARY]` 标记时生效** —— 旧会话保守全量；裁剪取最后边界。
- **流式预执行只覆盖 safe 且默认关** —— `agent.streaming_tool_execution=False` 灰度；unsafe 不预执行（乱序副作用不可接受）。
- **plan 清上下文只清 LLM 上下文** —— 会话库 append-only 可恢复；stable prompt 段保留。
- **http hook env 插值默认关** —— `${VAR}` 只在 `security.http_hook_allowed_env_vars` 白名单内插值，非白名单保留原样并告警。
- **goal nudge 只在有预算限制时生效** —— `should_nudge` 需 `token_budget_limit` + 最近一轮工具成功；无预算 goal 不踢（对齐 CCB 收益递减门控的保守面）。

### 记忆与技能

- **记忆分层项目隔离** —— project/reference 类存 `.memory/projects/<canonical-git-root>/`，项目间物理不可见；user/feedback/other 全局共享；worktree 与主 repo 共享项目区。
- **记忆写入命中秘密即拒绝** —— memory_store fail-closed（ValueError）；curator 改写产物命中拒绝保留原文；trace 是 fail-open redact。
- **auto_extract 默认关 + 与主写入互斥** —— 本轮 LLM 调过 memory save/update 则跳过；游标始终推进不回看。
- **auto_extract 机械查证** —— 引用不存在相对路径的条目丢弃；绝对路径不验（保守放行）；完整带工具查证裁决不搬。
- **检索式记忆注入每轮一次** —— 主代理 only（spawn_depth==0）；无 aux_llm_router 降级 snapshot（会话一次）。
- **记忆检索 prefetch 是一次性消费** —— 结果 append 本轮 messages（天然 ephemeral），reactive_retry 不重复注入；失败 fail-open 静默。
- **技能 files 附件不进索引** —— 只在触发时注入（execute_skill / load_skill），skill_view 与索引不含。
- **条件技能 paths 双语义** —— 目录形态（`src/**`）进静态索引；文件 glob（`*.py`）只走动态激活。glob 在 frontmatter 必须加引号（YAML `*` 是 alias 语法）。
- **skillLearning 默认关 + 只演化 global scope** —— 项目约定类 instinct 落 project scope 仅存储不自动进化（跨项目泄漏 + slug 撞车）；观察/进化仅主代理（防 feedback loop）。
- **skillify 是纯 MD 零 Python** —— 四步访谈式沉淀，用户纠正进"规则"段。

### 子代理与工具

- **子代理 memory 独立目录** —— `memory: true` 时写 `~/.OmniMate/.agent-memory/<name>/`，不参与 curator。
- **mcp_server_filter 仅 schema 层** —— 只过滤 LLM 可见 schema，registry 仍注册全部（手动 dispatch 仍命中，对齐 CC）。
- **内联 MCP 不残留** —— 临时连接 finally 必断开；连接失败 fail-open。
- **context:fork 同步等待** —— 技能子代理跑完才回主循环；minimal 工具集 + spawn_depth+1 防递归。
- **transcript 落盘 fail-open** —— 轨迹 = user 指令 + 每轮 assistant 文本（tool_calls 不落盘，无配对 result 会造孤儿消息 → API 400）；默认开，7 天 retention。
- **async 子代理默认拒审批**（autoDeny 第 4 模式）—— 保留 fatal/safe-fs 底线。
- **monitor 豁免 stall 看门狗** —— tail -f/watch 安静是常态；一次性命令不要用 monitor。
- **LSP 工具依赖外部 pylsp** —— check_fn 门控自动隐藏；server 崩溃自动重建；不进项目依赖。
- **workflow 引擎的边界** —— goal=主循环多轮驱动，workflow=一次工具调用内编排（口径不同不双计：workflow 子代理 usage 不回累 goal）；journal 与 task_store 语义分离（task=待办，run=执行记录）；脚本 exec 是防误用不是安全边界（AST 白名单拦显式逃逸）；resume 只信 run 目录快照。
- **workflow 禁入 async 子代理** —— ASYNC_AGENT_DISALLOWED_TOOLS 含 workflow/subagent（防递归 spawn）；workflow 内的 agent() 一律 leaf 角色 + minimal 工具集。
- **workflow kill/cancel 是边界语义** —— kill 经 `_ACTIVE_RUNS` 事件只在 agent() 调用间隙生效（to_thread 里的单个子代理不可中断）；前台 run 阻塞主循环，kill 实际只能由另一会话/async 子代理发起。resume 预算是 per-run 重发（非累计），多次 resume 可累积总支出——预算硬顶是单次 run 口径。

### CLI 与桌面

- **/init 生成的是 cwd 的 OMNIMATE.md** —— 已存在不覆盖（--force 覆盖）。
- **/paste 只保存不分析** —— 存 `.paste/img_<ts>.png`，用户引用路径让 LLM 调 image_analyze。
- **notifier 仅 Windows** —— toast；bg title 带 task_id 前 8 位（30s 节流不互吞）。
- **preventSleep 引用计数语义** —— acquire/release 按 reason，只在忙闲转换时真正调（每轮调会计数无界）；中断路径残留到 atexit 兜底（宁多醒不久睡）。
- **排队输入不打断当前响应** —— 工具批结束 drain 回流 ephemeral；单用户 FIFO，不做三级优先级。
- **粘贴占位符 session 存占位、发送展开** —— 外存文件被清理时保留占位符 fail-open。
- **scratchpad 是「完全可逆」铁律的显式例外** —— 临时涂鸦区按 mtime 7 天清理（临时区非知识库）；白名单是运行时的不写 settings。
- **任务全清是标志不是删除** —— `all_done: true` 引导 LLM，不动任务状态（completed 持久可查）。
- **空结果保护** —— 空串/空 dict 注入 `(toolName completed with no output)`，防模型误判回合边界。
- **Read 超限是报错不是截断** —— 256KB/25K token 双上限，引导 offset/limit 分段。
- **web_fetch 提炼依赖 aux 注入** —— 子代理/无 aux 自动降级全文 fail-open。

## 测试策略

- **按模块组织**：`tests/test_<模块>.py`（120+ 个测试文件，2900+ 用例）
- **集成**：`tests/test_integration.py` 用 mock OpenAI client 跑完整对话流程（含工具调用、记忆注入、中断）
- **验证脚本**：`uv run python scripts/verify.py` 跑复刻指南 22 项检查清单，适合快速回归（不含新功能测试）
- **新增功能必加测试**：新模块配独立测试文件，改完跑 `uv run pytest tests/` 确认无回归
- **契约测试防回归**（历史踩坑的护栏）：handler 签名 `(args, **kwargs)`、schema 键 "parameters"、async 工具 contextvar 回透（dispatch 外断言）
