# HermesAgent 改进路线图 —— 取长补短 Claude Code

- **作者**：Claude（经 brainstorming 流程产出）
- **日期**：2026-07-12
- **状态**：设计已与用户逐节确认，待 spec 复核后进入 writing-plans
- **驱动**：生产场景（代码仓库维护、文档生成、数据 pipeline 等）
- **兼容性策略**：允许破坏性变更，但需提供清晰迁移路径与功能开关
- **交付粒度**：Phase 1 写到可实施级别；Phase 2-5 给接口契约 + 风险 + 验收

---

## 摘要

本设计文档对比 Claude Code 技术文档（`D:/project/learn-claude-code-main/docs/zh/`）与 HermesAgent 当前实现，识别出 6 个可改进的独立子系统，按依赖关系排出 5 个 Phase 的路线图，并对 Phase 1（核心韧性层）给出可实施的详细设计。

**核心结论**：HermesAgent 当前最大的工程债务是上下文管理——`context_compressor.py` 是单层 LLM 摘要，长会话下每次压缩花 1 次 API 调用且信息保真度差。Phase 1 用 4 层管线 + 大输出落盘 + transcript 归档把这个债务还掉，所有后续 Phase 的稳定性都依赖这一层。

---

## §1 总体路线图

```
Phase 1: 核心韧性（无新依赖，最先做）
   1a 分层压缩 L1/L2/L3/L4    ── 替换 context_compressor.py 单层摘要
   1b 大输出落盘               ── terminal/file 工具结果 >30KB 写文件
   1c pre_compress 快照        ── 压缩前完整 messages 落 .transcripts/
   1d Transcript 归档          ── 同上，独立工具支持回查

Phase 2: 扩展机制（依赖 Phase 1 稳定的循环）
   2a Hooks 系统               ── PreToolUse/PostToolUse/UserPromptSubmit/Stop
   2b 后台任务                 ── 守护线程 + 通知队列 + 每轮注入
   2c Cron 调度                ── minute_marker 去重，复用 Hooks 注册

Phase 3: 任务增强（独立小项，可随时插队）
   3a 任务级联解锁反馈          ── task_tools.py 完成时报告 unblocked
   3b Worktree 事件流审计       ── .worktrees/.events.jsonl

Phase 4: 多代理（最大改动，依赖 Phase 1+2 落地）
   4a Agent Teams              ── .team/inbox/*.jsonl 消息总线
   4b Autonomous Agent         ── WORK/IDLE/SHUTDOWN 三态

Phase 5: 长期维护
   5a 记忆整合 Dream           ── 文件数 ≥10 触发 LLM 合并去重
```

### 依赖关系

```
1a → 1c → 1d → (1b 独立可并行)
            │
            └──→ 2a (Hooks) ──→ 2b (BG)
                          └──→ 2c (Cron)
                                  │
                                  └──→ 4a (Teams) ──→ 4b (Auto)
3a, 3b, 5a 任意时机可插队
```

### 每 Phase 的产出与验收

| Phase | 产出 | 验收 |
|---|---|---|
| 1 | 新 `agent/context_pipeline.py` + `agent/transcript.py` + 改造的 terminal/file 工具 | 模拟 200 轮对话脚本，压缩 ≥3 次，token 成本对比当前下降 ≥40%，且 transcript 可重放 |
| 2 | `agent/hooks.py` + `tools/bg_task.py` + `agent/cron.py` + `settings.json` schema | PreToolUse hook 能拦截危险命令；后台任务可启动+查询+回流；Cron 每 N 分钟注入消息 |
| 3 | task_tools/worktree 小补丁 | 完成 task_1 自动报告 task_2 unblocked；worktree 操作有事件流 |
| 4 | `agent/team/` 子包 + `tools/team_tool.py` + autonomous loop 模式 | 2 个 agent 协作完成单个 task；某 agent 崩溃不影响其他 |
| 5 | `agent/dream.py` + curator 集成 | 记忆文件数从 15 合并到 ≤8，关键事实无丢失 |

### 路线图层级的三个风险

1. **Phase 1 阈值切换**：现有 `MESSAGES_BEFORE_COMPRESS=40` 切到分层阈值时需要 hysteresis，避免边界震荡。
2. **Phase 2 Hooks 必须失败不阻塞**：hook 抛异常只能 log，不能让 agent 整体崩。Claude Code 文档未明确这点。
3. **Phase 4 文件总线竞态**：多进程并发追加 JSONL 要用 `O_APPEND`（POSIX）或 `LockFileEx`（Windows），需 Phase 4 启动前 spike 验证。

---

## §2 Phase 1 架构总览

**核心思想**：把现有 `context_compressor.maybe_compress`（单层 LLM 摘要）拆成 4 层独立管线 + 1 个紧急通道 + 2 个旁路（落盘 + 快照），每层都更便宜，贵的那层（LLM 调用）只在前面都失败时才触发。

### 管线时序

```
┌──────────────────────────────────────────────────────────────────────┐
│ 工具执行阶段（每个 handler 返回前）                                    │
│   terminal/file handler                                               │
│        ↓ return content                                               │
│   output_offload.maybe_offload(content, tool_call_id)  ← 旁路 L3     │
│        ↓ if len > 30KB: 落盘 + 返回 2000 字符预览                       │
│   handler 返回 JSON                                                    │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 每轮 LLM 调用前（替换 agent/__init__.py:212-225 的 maybe_compress）    │
│                                                                       │
│   ① transcript.snapshot_if_needed(messages, force=False)              │
│        ↓ 接近阈值时落盘 .transcripts/transcript_{ts}.jsonl            │
│                                                                       │
│   ② pipeline.compress_if_needed(messages, attempt)                    │
│        ├─ L1 snip_compact   (0 API)  消息数 > 50 时裁中间             │
│        ├─ L2 micro_compact  (0 API)  旧 tool_result 换占位符           │
│        ├─ L4 llm_compact    (1 API)  仍超阈值才调 LLM 摘要             │
│        │      └─ force=True 前调 transcript.snapshot_if_needed        │
│        └─ 每层都过 _fix_tool_call_pairs 防止拆对                       │
│                                                                       │
│   ③ 如果任一层动了 → invalidate_system_prompt()                       │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│ LLM 调用后报错（reactive，新增）                                       │
│   if error_type == "prompt_too_long" and not already_reacted:         │
│       messages = reactive_compact(messages)                           │
│       只留最后 5 条 + 一条占位 user 消息说明发生了紧急压缩             │
│       already_reacted = True   # 防止循环                              │
│       retry                                                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/output_offload.py` | 🆕 新增 | `maybe_offload(content, call_id)` + 落盘到 `.task_outputs/tool-results/` |
| `agent/transcript.py` | 🆕 新增 | `snapshot_if_needed(messages)` + 落盘到 `.transcripts/` |
| `agent/context_pipeline.py` | 🆕 新增 | 4 层管线编排：`compress_if_needed()` + `reactive_compact()` |
| `agent/context_compressor.py` | ♻️ 重构 | 保留 `_fix_tool_call_pairs` 和 `_summarize_conversation` 作为 utility，对外接口 deprecate |
| `agent/memory_manager.py` | ♻️ 改 | 新增 `on_pre_compress(snapshot, messages)` 钩子（Phase 1 内 no-op） |
| `tools/terminal_tool.py` | ♻️ 改 | handler 返回前过 `output_offload.maybe_offload` |
| `tools/file_operations.py` | ♻️ 改 | 同上（read_file 大文件场景） |
| `agent/__init__.py` | ♻️ 改 | 第 212-225 行 `maybe_compress` 调用整段替换为 pipeline；新增 reactive_compact 分支 |
| `config.py:DEFAULT_CONFIG` | ♻️ 改 | 新增 `context.*` 配置块（阈值） |

### 关键决策

1. **L1/L2 是有损的（占位符替换不可逆）**，但因为 L3 已经把大输出落盘 + L1/L4 前已经写过 transcript，所以"信息没丢，只是不在 messages 数组里"——LLM 想看完整内容可以调 `read_file` 工具从落盘文件读回。
2. **memory 抽取不在 Phase 1 做**：HermesAgent 当前记忆模型是主动式（LLM 通过 `memory_tool` 自己决定写），强行加被动抽取会和现有模型冲突。只预留 `on_pre_compress` 钩子，实现留给 Phase 5。
3. **reactive_compact 只触发一次**（`already_reacted` 标志），避免死循环。
4. **`_fix_tool_call_pairs` 在每一层后都跑一遍**——L1 裁中间、L2 替换占位符都可能把 assistant(tool_calls) 和它的 tool 结果拆开。
5. **配置参数都有默认值**（向后兼容），但允许用户在 `config.yaml` 覆盖。

### 与现有契约衔接

- `maybe_compress` 返回 `(messages, bool)` 的契约不变，pipeline 沿用，最小化对 `agent/__init__.py` 的改动面。
- `invalidate_system_prompt()` 仍是唯一允许的 system prompt 失效入口（CLAUDE.md 关键设计原则 #2）。
- transcript 和 offload 文件都走 `safe_path`（CLAUDE.md 安全机制），通过 `harvil_home` 参数声明 `allowed_roots`。

---

## §3 分层压缩管线（阈值与算法）

### L3 — output_offload（旁路，工具结果产出时）

| 项 | 值 |
|---|---|
| 触发 | 单条 tool 结果字符数 > `output_offload_threshold`（默认 30000） |
| 动作 | 写入 `~/.agent/.task_outputs/tool-results/{tool_call_id}.txt`，messages 里只留 2000 字符预览 |
| 防抖 | 不需要（每条结果只判定一次） |
| 配对 | 不影响（占位消息 role 仍是 `tool`，`tool_call_id` 不变） |

替换后的 tool 消息 content 形如：

```json
{
  "truncated": true,
  "preview": "<前 2000 字符>",
  "full_at": "/abs/path/to/.task_outputs/tool-results/call_abc123.txt",
  "hint": "如需完整内容，调 read_file 读取 full_at"
}
```

**关键**：LLM 仍能通过 `read_file` 工具读回完整内容——这是把"压缩"做成"无损"的核心。

### L1 — snip_compact（每轮 LLM 前，0 API）

| 项 | 值 |
|---|---|
| 触发 | `len(conversation) > snip_message_threshold`（默认 50） |
| 动作 | 保留 system + 前 3 条 + 后 47 条 + 中间占位 user 消息 |
| 防抖 | 双阈值：触发 50 / 解除 30（避免边界震荡） |
| 配对 | 跑完后过 `_fix_tool_call_pairs` 补漏 |

```python
def snip_compact(messages, keep_first=3, keep_last=47):
    system, conv = split_system(messages)
    if len(conv) <= keep_first + keep_last:
        return messages, False
    head, tail = conv[:keep_first], conv[-keep_last:]
    omitted = len(conv) - keep_first - keep_last
    placeholder = {"role": "user", "content":
        f"[snip_compact: 中间 {omitted} 条已省略，"
        f"完整记录在 .transcripts/latest.jsonl]"}
    new = head + [placeholder] + tail
    return reassemble(system, new), True
```

### L2 — micro_compact（L1 之后，0 API）

| 项 | 值 |
|---|---|
| 触发 | conv 中 `role==tool` 的消息数 > `keep_recent_results`（默认 3） |
| 动作 | 把较旧的 tool 消息 content 替换为占位 JSON（保留 `role/tool_call_id/name`） |
| 防抖 | 幂等（已是占位的不动） |
| 配对 | 安全：只换 content，不删消息 |

```python
def micro_compact(messages, keep_recent=3):
    tool_idx = [i for i,m in enumerate(messages) if m.get("role")=="tool"]
    if len(tool_idx) <= keep_recent: return messages, False
    old = set(tool_idx[:-keep_recent])
    out = []
    for i, m in enumerate(messages):
        if i in old and not _already_placeheld(m):
            nm = dict(m); orig_len = len(str(m.get("content","")))
            nm["content"] = json.dumps({
                "micro_compacted": True,
                "orig_chars": orig_len,
                "hint": f"Tool {m.get('name','?')} 结果已折叠，"
                        f"完整内容见 .transcripts/latest.jsonl 或重跑工具"
            }, ensure_ascii=False)
            out.append(nm)
        else:
            out.append(m)
    return out, True
```

### L4 — llm_compact（L1+L2 之后仍超限，1 API）

| 项 | 值 |
|---|---|
| 触发 | `estimate_tokens(messages) > 100000` **或** `len(conv) > 100`（双触发） |
| 动作 | 复用现有 `_summarize_conversation`，总结较早的 `conv[:-10]`，注入为 user 占位 |
| 防抖 | 冷却 `COMPRESS_COOLDOWN_TURNS = 5`（沿用现有常量） |
| 配对 | 复用现有 `_fix_tool_call_pairs` |

```python
def llm_compact(messages, llm_client, model, keep_recent=10):
    if not _over_token_threshold(messages) and len(messages) <= 100:
        return messages, False
    system, conv = split_system(messages)
    if len(conv) <= keep_recent: return messages, False
    summary = _summarize_conversation(conv[:-keep_recent], llm_client, model=model)
    if not summary: return messages, False
    placeholder = {"role":"user","content":
        f"[之前的对话已自动总结]\n\n{summary}\n\n[以下是最近的对话]"}
    new = [placeholder] + conv[-keep_recent:]
    return reassemble(system, _fix_tool_call_pairs(new)), True
```

### Reactive — 紧急通道（API 报错时）

| 项 | 值 |
|---|---|
| 触发 | LLM 调用抛 `prompt_too_long` / `context_length_exceeded` |
| 动作 | 只留 system + 最后 5 条 + 占位说明 |
| 防抖 | `already_reacted` 会话级布尔，**只触发一次** |
| 配对 | 跑 `_fix_tool_call_pairs` |

```python
def reactive_compact(messages):
    system, conv = split_system(messages)
    keep = conv[-5:] if len(conv)>5 else conv[:]
    placeholder = {"role":"user","content":
        "[紧急上下文压缩：API 返回 prompt_too_long，"
        "已只保留最近 5 条消息。完整历史见 .transcripts/latest.jsonl]"}
    new = [placeholder] + keep
    return reassemble(system, _fix_tool_call_pairs(new))
```

### 编排：`compress_if_needed` 顶层入口

```python
def compress_if_needed(messages, *, attempt_count, llm_client, model,
                      config, session_state) -> tuple[list, bool]:
    # 1. L1
    messages, c1 = snip_compact(messages, **config.snip_args)
    # 2. L2（无论 L1 是否触发，都跑一次折叠）
    messages, c2 = micro_compact(messages, **config.micro_args)
    # 3. L4（受冷却限制）
    if attempt_count < MAX_COMPRESS_ATTEMPTS and _cooldown_ok(session_state):
        if _over_token_threshold(messages) or len(messages) > 100:
            # force=True：L4 是有损的，落盘保全尸
            transcript.snapshot_if_needed(messages, force=True, ...)
            messages, c4 = llm_compact(messages, llm_client, model,
                                        **config.llm_args)
            if c4: session_state.record_llm_compact()

    changed = c1 or c2 or c4
    if changed:
        messages = _fix_tool_call_pairs(messages)
    return messages, changed
```

### 默认阈值（写入 `config.py:DEFAULT_CONFIG["context"]`）

```python
"context": {
    # L3 offload
    "output_offload_threshold": 30000,
    "output_offload_preview": 2000,
    # L1 snip
    "snip_message_threshold": 50,
    "snip_release_threshold": 30,
    "snip_keep_first": 3,
    "snip_keep_last": 47,
    # L2 micro
    "micro_keep_recent_results": 3,
    # L4 llm
    "llm_compact_token_threshold": 100000,
    "llm_compact_message_threshold": 100,
    "llm_compact_keep_recent": 10,
    "llm_compact_cooldown_turns": 5,
    "max_compress_attempts": 3,
    # Reactive
    "reactive_keep_recent": 5,
    "reactive_once_per_session": True,
    # Transcript
    "transcript_enabled": True,
    "transcript_trigger": "pre_llm_compact",
    "transcript_retention": 20,
    # 功能开关（双轨期）
    "use_new_pipeline": False,   # Phase 1 Commit 4 加；Commit 6 改 True
}
```

**为什么这些阈值**：DeepSeek-Chat context 64K，按 3 字符/token 估，约 190K 字符上限。`snip` 在 50 消息时触发，对应 ~150K 字符，留出余量。所有阈值都是配置项；模型不同时（如 DeepSeek-Reasoner 32K）需要调小。

### 与现有 `maybe_compress` 的对照

| 维度 | 现有 | 新管线 |
|---|---|---|
| 触发 | 消息数 ≥40 | 分层：50/100/token 阈值 |
| 一次压缩成本 | 1 次 LLM API | 通常 0 API（L1+L2），最坏 1 API |
| 信息保真 | LLM 摘要必然丢 | L1/L2 无损（可从 transcript/offload 恢复） |
| 大输出处理 | 截断到 200 字符进摘要 | 落盘 + 2000 字符预览（LLM 可主动读回） |
| API 报错恢复 | 无 | reactive_compact 兜底 |

---

## §4 落盘文件格式、Transcript 接口、记忆保真

### 4.1 output_offload.py

**接口**：

```python
def maybe_offload(
    content: str,
    *,
    tool_call_id: str,
    agent_home: Path,
    config: dict,
) -> str:
    """工具 handler 调用。返回值直接作为 tool 消息 content 用。
    > threshold 时落盘，返回 JSON（含 preview + 指针）。
    ≤ threshold 时原样返回。
    """
```

**文件布局**：

```
~/.agent/.task_outputs/tool-results/
  ├── call_abc123.txt            # 原始内容，纯文本
  ├── call_def456.txt
  └── ...
```

**文件名规则**：用 `tool_call_id`（OpenAI 兼容协议里每轮工具调用唯一）。极端兜底：若已存在则追加 `_{counter}`。

**返回的占位 JSON**（tool 消息的 content）：

```json
{
  "truncated": true,
  "orig_chars": 52384,
  "preview": "<前 2000 字符>",
  "full_at": "C:/Users/.../.agent/.task_outputs/tool-results/call_abc123.txt",
  "hint": "完整结果已落盘，需要时调 read_file 读取 full_at"
}
```

**生命周期**：
- ❌ 不自动清理（生产场景下用户可能要复查昨天某次长输出）
- ✅ Phase 5 的 Dream 任务统一清理（按 mtime + 大小限额）
- ✅ 用户可手动删整个 `.task_outputs/` 目录

**safe_path 集成**（CLAUDE.md 安全机制）：
- `output_offload` 写入路径走 `safe_path(write=True, allowed_roots=[agent_home/".task_outputs"])`
- `read_file` 工具读回这些文件时天然走现有白名单（默认包含 agent_home）

### 4.2 transcript.py

**接口**：

```python
def snapshot_if_needed(
    messages: list,
    *,
    force: bool,
    agent_home: Path,
    config: dict,
    session_id: str,
) -> Optional[Path]:
    """force=True 时必落盘；否则按 trigger 配置判定（默认仅 L4 前）。"""
```

**触发时机**（`config.context.transcript_trigger`）：
- `"pre_llm_compact"` ← 默认。仅在 L4 真正要调用 LLM 摘要前 force=True。L1/L2 不落盘（无损，无需留全尸）。
- `"pre_any_compact"` ← 可选。L1 也落盘。
- `"disabled"` ← 可选。

**文件布局**：

```
~/.agent/.transcripts/
  ├── transcript_20260712_153022_a1b2.jsonl
  ├── transcript_20260712_161044_c3d4.jsonl
  ├── ...
  └── latest.jsonl   ← 符号链接（Windows 降级为文本指针 latest.txt）
```

**JSONL 格式**（每行一条消息，含 envelope）：

```json
{"seq":0,"role":"system","content":"...","ts":"2026-07-12T15:30:22"}
{"seq":1,"role":"user","content":"帮我分析...","ts":"..."}
{"seq":2,"role":"assistant","tool_calls":[{"id":"call_abc","function":{"name":"terminal","arguments":"..."}}],"ts":"..."}
{"seq":3,"role":"tool","tool_call_id":"call_abc","name":"terminal","content":"...","ts":"..."}
{"_meta":{"session_id":"sess_xxx","reason":"pre_llm_compact","orig_len":87,"kept_recent":10}}
```

最后一行 `_meta` 是元数据，便于事后排查为什么触发了压缩。

**生命周期**：
- 默认保留最近 20 个 transcript 文件（`config.context.transcript_retention`）
- 超出后删最旧的（按 mtime）
- 跨会话也可保留（每个 session 产生 0~N 个 transcript）

**Windows 注意**：`latest.jsonl` 符号链接在 Windows 需要管理员权限或开发者模式。降级方案：写入 `latest.txt` 文本文件，内容是当前最新的 transcript 路径。生产环境默认用降级方案。

### 4.3 pre_compress 快照与记忆抽取（务实降级）

**修订**：原 §2 草案提到"新增 `memory_manager.extract_from_snapshot()` 接口做 LLM 抽取"。深入设计后发现这会引入额外 LLM 调用 + 新 prompt 模板 + 测试矩阵，超出 Phase 1 边界。

**降级方案**：
- Phase 1 只做**快照存档**（4.2 的 transcript），不做主动 LLM 抽取。
- 给 `MemoryManager` 增加钩子点 `on_pre_compress(messages)`，签名稳定，但 Phase 1 内部留空实现。
- Phase 5（或独立的 Phase 1.5）再实现"扫 transcript 抽记忆"，复用 transcript 文件而不是当场 LLM 调用。

**理由**：
- HermesAgent 当前记忆模型是**主动式**（LLM 通过 `memory_tool` 自己决定写），和 Claude Code 的**被动抽取**不同。强行加被动抽取会和现有模型冲突。
- 当前最大的信息丢失来源是 **L4 LLM 摘要本身**，先把 transcript 落盘做了，事后回查能力就有了，已经解决 80% 的问题。
- 把"自动抽取"留到独立小阶段，避免 Phase 1 范围爆炸。

**接口预留**（spec 里写明，但 Phase 1 实现为 no-op）：

```python
# agent/memory_manager.py
class MemoryManager:
    def on_pre_compress(self, snapshot_path: Optional[Path], messages: list) -> None:
        """钩子：压缩前调用。Phase 1 留空，未来扩展。"""
        pass
```

### 4.4 恢复路径

| 场景 | LLM 怎么找回 |
|---|---|
| 单条工具结果被 L3 offload | 占位 JSON 里有 `full_at`，LLM 直接调 `read_file(full_at)` |
| 多条旧消息被 L1 snip / L2 micro | 占位 user 消息提示了 `.transcripts/latest.jsonl`，LLM 调 `read_file` |
| Reactive 紧急压缩丢了大批上下文 | 占位 user 消息明确说明发生紧急压缩，LLM 知道去查 transcript |

**system prompt 补一条提示**（改 `prompt_builder.py:TOOL_USAGE_GUIDANCE`）：

```
- 当你看到 "[snip_compact]" / "[micro_compacted]" / "[紧急上下文压缩]" 这类占位
  消息，且需要更早的上下文时，从占位消息里给的路径（.transcripts/ 或 .task_outputs/）
  用 read_file 读回。这些路径在 agent_home 下，默认安全。
```

### 4.5 落盘的并发安全（为 Phase 4 预留）

Phase 1 不解决并发（单线程同步循环），但要为后续留接口：

- `output_offload` 和 `transcript` 都用**带 pid + uuid 的文件名**，避免多 worker 撞名
- 写入用 `tempfile + os.replace` 原子替换（防半写）
- `latest.jsonl` 的更新用文件锁（`fcntl` Linux / `msvcrt` Windows），但 Phase 1 单线程可先跳过

---

## §5 迁移、破坏性变更、上线顺序

### 5.1 破坏性变更清单

| # | 变更 | 影响面 | 严重度 |
|---|---|---|---|
| ① | `agent/context_compressor.maybe_compress()` 废弃，由 `context_pipeline.compress_if_needed()` 取代 | 内部 API；外部只有 `agent/__init__.py:214` 调用 | 中（可加 shim 降级为低） |
| ② | `config.py:DEFAULT_CONFIG` 新增 `context` 块（含 13 个阈值） | 老的 `config.yaml` 仍跑（loader 深合并） | 低 |
| ③ | `tools/terminal_tool.py` / `tools/file_operations.py` 大输出改返回 offload JSON | LLM 看到的 tool 消息 content 形态变化 | 中（system prompt 加指导） |
| ④ | 新建目录 `~/.agent/.task_outputs/tool-results/` 和 `~/.agent/.transcripts/` | 首次运行懒创建；现有 agent_home 不冲突 | 低 |
| ⑤ | `agent/__init__.py:AIAgent` 新增内部状态：`_reacted`、`_compress_cooldown_state` | 仅内部，构造签名不变 | 低 |
| ⑥ | `agent/memory_manager.py` 新增 `on_pre_compress()` no-op 方法 | 纯加法 | 无 |
| ⑦ | `agent/prompt_builder.py:TOOL_USAGE_GUIDANCE` 加一段占位消息识别指南 | system prompt 字节级变化；下个 session 生效（不破缓存语义） | 低 |
| ⑧ | `tests/test_context.py` 重写为分层管线测试；新增 3 个测试文件 | CI 全跑；外部用户没碰到 | 低 |
| ⑨ | （可选）移除 `MAX_COMPRESS_ATTEMPTS`、`MESSAGES_BEFORE_COMPRESS` 等旧常量 | 若有人 import 会断 | 低 |

### 5.2 渐进上线顺序（每个 commit 都可发版）

为降低生产风险，采用**双轨期 + 功能开关**。共 7 个有序 commit：

```
Commit 1  [新增]  agent/output_offload.py + agent/transcript.py + tests
                ↑ 纯加法，不接 main loop，CI 跑新模块单测

Commit 2  [新增]  agent/context_pipeline.py（编排器，内部调 context_compressor
                  的 _summarize_conversation / _fix_tool_call_pairs）
                ↑ 纯加法，未挂到 agent/__init__.py

Commit 3  [配置]  config.py 增加 context 块 + use_new_pipeline 开关（默认 False）
                ↑ 老用户默认走旧管线

Commit 4  [挂线]  agent/__init__.py:212-225 改为：
                  if config.use_new_pipeline:
                      messages, compressed = context_pipeline.compress_if_needed(...)
                  else:
                      messages, compressed = maybe_compress(...)  # 旧路径保留
                ↑ 此时新管线已可用，但默认不开

Commit 5  [工具]  tools/terminal_tool.py + tools/file_operations.py 接 output_offload
                  受 config.use_new_pipeline 开关控制（保持一致）
                  system prompt 加占位消息识别指南
                ↑ 开关 False 时一切照旧

Commit 6  [切换]  config.use_new_pipeline 默认值改为 True
                  CHANGELOG 标记 v0.X：默认启用新管线，回滚设 False
                ↑ 用户若发现问题可一键回退

Commit 7  [清理]  下个版本（v0.X+1）：删除 maybe_compress 旧路径 + use_new_pipeline
                  开关 + 旧常量。CHANGELOG 标注完全移除日期。
```

**双轨期长度建议**：Commit 6 → Commit 7 之间至少 1 个 minor 版本（约 2 周），给生产用户时间反馈。

### 5.3 错误处理与降级

新管线在以下故障下不应让 agent 整体崩：

| 故障 | 降级行为 |
|---|---|
| 磁盘满，offload 写不进 | log warning，原 content 截断到 30KB + 注入 `{"error": "offload failed, truncated"}` |
| transcript 写不进 | log warning，跳过 snapshot，仍执行压缩（信息保真度下降但不阻塞） |
| L4 LLM 摘要调用失败 | 沿用现有 `_rule_based_summary` 降级 |
| Reactive 触发后仍 prompt_too_long | 抛 `ContextOverflowError`，由上层 try/except 接住，turn 结束并提示用户 |
| `_fix_tool_call_pairs` 自身异常 | log error 但不阻塞（极端情况，让 API 自己报错给 LLM） |

**总原则**：所有落盘和压缩操作都用 try/except 包住，失败只 log，不抛；只有"LLM API 自己报错"是真正不可恢复的，由 reactive 兜底。

### 5.4 可观测性（log 规范）

```python
logger.info("L1 snip_compact: conv %d → %d (omitted %d)",
            before, after, omitted)
logger.info("L2 micro_compact: folded %d old tool results", folded)
logger.info("L4 llm_compact: %d msgs summarized, %d chars → %d chars",
            n_summarized, orig_chars, summary_chars)
logger.warning("reactive_compact triggered: kept last %d", kept)
logger.info("offload: %s → %s (%d chars)", tool_call_id, path, orig_chars)
logger.info("transcript snapshot: %s (reason=%s, msgs=%d)",
            path, reason, n_msgs)
```

建议在 `config` 加 `context.log_level: "INFO"`（默认）便于切换 DEBUG 排查。

### 5.5 测试矩阵

**新增测试文件**：
- `tests/test_output_offload.py` —— 阈值判定、文件写入、JSON 形态、并发不撞名、磁盘满降级
- `tests/test_transcript.py` —— force=True/False、JSONL 格式、retention、latest 链接/降级
- `tests/test_context_pipeline.py` —— 每层独立、层组合、防抖、reactive once-per-session

**改造测试**：
- `tests/test_context.py` —— 改为断言新管线行为；保留 `_fix_tool_call_pairs` 和 `_summarize_conversation` 的现有用例

**集成测试**（关键）：`tests/test_integration.py` 加用例"**模拟 200 轮工具调用对话**"：
- mock OpenAI client 每 5 轮返回一次 tool_use
- 每 10 轮注入一个 50KB 的工具结果（触发 offload）
- 跑完后断言：
  - offload 文件数量 > 0
  - transcript 文件至少 1 个
  - L4 触发 ≥ 1 次
  - 全程无未捕获异常
  - 最终 messages 长度 < 起始阈值

**性能基准**（可选但推荐）：用 `pytest-benchmark` 测 `compress_if_needed` 在 200 条消息下的耗时。目标：L1+L2 路径 < 10ms（无 LLM），L4 路径与现有 `maybe_compress` 相当。

---

## §6 Phase 2-5 路线图概要

### Phase 2a — Hooks 系统

**为什么是 Phase 2 的核心**：Hooks 把"扩展循环行为"从「改 `agent/__init__.py` 代码」变成「写回调函数注册」。后续的后台任务通知注入、Cron 触发、审计日志都可以做成 hook，避免核心循环臃肿。

**接口契约**：

```python
# agent/hooks.py（新增）
class HookEvent(Enum):
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"

UserPromptSubmitHook = Callable[[str], Optional[str]]
PreToolUseHook = Callable[[str, dict], Optional[str]]         # 返回拒绝理由或 None
PostToolUseHook = Callable[[str, dict, str], Optional[str]]
StopHook = Callable[[], Optional[str]]

def register_hook(event: HookEvent, fn): ...
def run_hooks(event: HookEvent, *args) -> list: ...
```

**注册来源**：
- 程序式：`hooks.register(HookEvent.PRE_TOOL_USE, my_fn)`
- 声明式：`~/.agent/.hooks/settings.json`（脚本路径，由 hooks 模块 fork 执行，需走权限闸门）

**集成点**：
- `agent/__init__.py:run_conversation` 顶部插 USER_PROMPT_SUBMIT
- `model_tools.py:handle_function_call` 顶/尾插 PRE/POST_TOOL_USE
- `run_conversation` 退出前插 STOP

**失败隔离**：每个 hook try/except，异常只 log 不抛。

**风险**：
- PreToolUse 返回拒绝理由时，tool 消息仍要返回（不能跳过），否则破坏 tool_call 配对。
- 用户配的脚本 hook 需要走 PermissionChecker 避免配置成 `rm -rf /`。

**验收**：能注册 PreToolUse hook 拦截 `terminal` 危险命令；hook 抛异常时主循环不崩；4 种 event 各有测试。

### Phase 2b — 后台任务

**接口契约**：

```python
# tools/bg_task.py（新增）
def bg_start(command: str, *, cwd: str = None, timeout: int = 600) -> str
def bg_status(task_id: str) -> str        # JSON: status, pid, runtime
def bg_result(task_id: str) -> str        # JSON: exit_code, stdout, stderr (各 cap 5000)
def bg_list() -> str
def bg_stop(task_id: str) -> str
```

**内部架构**：
- `BackgroundManager` 单例（cli.py 注入）
- 每个 task 用 `subprocess.Popen` + 守护线程 read stdout
- 完成时 push 到 `_notifications: deque`（lock 保护）
- `run_conversation` 每轮 LLM 前 drain 队列，作为 `<task_notification>` 临时 user 消息注入（不进 conversation_history）

**风险**：
- 子进程生命周期：默认随主进程退出（同一 process group）；`detach=True` 才独立。
- Windows 上 subprocess + UTF-8 输出需 `encoding="utf-8"`。

**验收**：能跑 sleep 10 的 bg task，10 秒后在主对话收到完成通知。

### Phase 2c — Cron 调度

**接口契约**：

```python
# agent/cron.py（新增）
# 配置：~/.agent/.cron/jobs.json
# [
#   {"id": "daily_report", "cron": "0 9 * * *", "message": "生成今日报告", "enabled": true},
#   ...
# ]
class CronScheduler:
    def start(self): ...
    def stop(self): ...
    def drain_due(self) -> list[str]:
```

**核心算法**：
- 后台线程每 30s 调度一次
- `minute_marker = now.strftime("%Y-%m-%d %H:%M")`，每个 job 记 `_last_fired[id]`
- 同一 minute 内同 job 只触发一次（防抖）
- 错过补救：进程启动时检查 `catch_up` 字段（默认 false）

**注入**：作为 `USER_PROMPT_SUBMIT` hook 注入 `[Scheduled: daily_report] 生成今日报告`（依赖 Phase 2a）。

**风险**：
- Cron 触发但 agent 正忙 → 入队列等下一轮（不并发触发 LLM）
- 跨时区：用本地时间，`cron` 字段是标准 5 段格式

**验收**：配一个 `*/2 * * * *` 的 job，2 分钟内看到注入。

### Phase 3 — 任务增强（小项）

**3a 任务级联解锁反馈**：
- 改 `tools/task_tools.py:task_complete` handler
- 完成后扫所有 task，找出 `blockedBy` 刚被满足的
- 在返回 JSON 加 `"unblocked": [{"id":"task_X","subject":"..."}]`
- 风险：低

**3b Worktree 事件流**：
- 改 `tools/worktree.py`
- 新增 `_log_event(event_type, payload)` 写 `.worktrees/.events.jsonl`
- 事件：`create.before/after`、`remove.before/after`、`bind`、`unbind`
- 写入用 `tempfile + os.replace` 原子
- 风险：低

### Phase 4 — 多代理（最大改动，需要 spike）

⚠️ **路线图层面必须先做的事**：Phase 4 启动前先开一次独立 brainstorming，决定**并发模型**（多线程 vs 多进程 vs asyncio）。这个决策影响整个 Phase 4 设计。

**4a Agent Teams 接口契约**（待 spike 后细化）：

```python
# agent/team/bus.py（新增）
class MessageBus:
    def send(self, sender: str, to: str, content: str, msg_type: str) -> str
    def read_inbox(self, name: str) -> list[dict]    # 消费式：读后清空
    def send_request(self, sender: str, to: str, content: str) -> str
    def respond(self, request_id: str, content: str) -> None

# tools/team_tool.py
def team_send(to, content, msg_type="message"|"request"|"response") -> str
def team_inbox() -> str
def team_members() -> str
```

**4b Autonomous Agent 接口契约**：

```python
# agent/autonomous.py（新增）
class AutonomousLifecycle:
    WORK_ROUNDS_LIMIT = 50
    IDLE_TIMEOUT_SECONDS = 60
    POLL_INTERVAL = 5

    def run(self, name: str):
        # WORK -> IDLE -> SHUTDOWN 三态循环
```

**风险**（路线图层）：
- 🔴 **文件锁跨进程**：JSONL 收件箱多进程并发 append 必须原子。POSIX 用 `O_APPEND`，Windows 用 `LockFileEx`。spike 中验证。
- 🔴 **任务认领 race**：两个 agent 同时看到 `pending` 任务都尝试认领。需要文件锁 + 状态检查原子化。
- 🔴 **死循环派生**：autonomous agent 自己 spawn 新 autonomous → 指数爆炸。需要"代数限制"（depth=2）或禁止递归派生。
- 🟡 **资源耗尽**：每个 agent 一个进程 + 一份 system prompt cache，5 个 agent 内存翻 5 倍。需要预算限制。

**验收门槛**（spike 通过的标准）：
- 2 个 agent 进程能通过 JSONL 收件箱完成一次 request-response
- 同一任务被同时认领时只有一个成功
- autonomous agent 跑完任务能正确进入 SHUTDOWN 而不是无限轮询

### Phase 5 — 记忆整合 Dream

**接口契约**：

```python
# agent/dream.py（新增）
def maybe_dream(memory_store, llm_client, config) -> bool:
    """文件数 ≥ threshold（默认 10）时触发。
    读所有未 pinned 的 memory，让 LLM 合并去重，原子写回。
    """
```

**集成**：作为 curator 周期任务的一个 step（复用现有 `agent/curator.py`）。

**风险**：🔴 **不可逆**——LLM 合并可能丢用户关心的事实。必须：
- Dream 前先把所有 memory 文件备份到 `.archive/dream-{ts}/`
- Pinned 的 memory 完全跳过
- Dream 后用对比测试验证关键事实不丢（写测试用例）

**验收**：15 个 memory 文件 → dream 后 ≤ 8 个，所有 pinned 保留，备份可恢复。

### 跨 Phase 测试策略

| 类别 | 策略 |
|---|---|
| 单元测试 | 每个新模块独立测试文件；目标 Phase 1 完成后总测试数从 294 → ~360 |
| 集成测试 | `tests/test_integration.py` 持续扩展；每个 Phase 完成加 1-2 个端到端用例 |
| 回归脚本 | `scripts/verify.py` 扩展，加 Phase 1~5 各 1 项验收检查 |
| 性能基准 | `tests/bench/` 新目录；用 `pytest-benchmark` 测压缩管线 + hooks 链路 + 多代理并发吞吐 |
| 故障注入 | Phase 4 必加：杀 agent 进程后看消息总线一致性；磁盘满时压缩降级；网络断时 reactive 触发 |
| 复刻检查清单 | 原 22 项继续过；每个 Phase 完成后加自己的检查清单 |

### 跨 Phase 配置 schema 演进

到 Phase 5 结束，`config.yaml` 大约会扩到这些块：

```yaml
context:     # Phase 1
hooks:       # Phase 2a
bg_task:     # Phase 2b
cron:        # Phase 2c
team:        # Phase 4a
autonomous:  # Phase 4b
dream:       # Phase 5a
```

**建议**：Phase 1 落地后写 `docs/CONFIG_SCHEMA.md`，每加一块同步更新。Phase 4 启动前用 jsonschema 验证 config，避免 typo 导致静默失败。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|---|---|---|---|
| 分层压缩 | L1/L2/L3/L4 + reactive | 沿用单层 LLM 摘要 | 生产场景下 token 成本和信息保真是核心痛点 |
| 记忆抽取 | 推迟到 Phase 5 | Phase 1 内做 LLM 抽取 | 与现有主动式记忆模型冲突；范围爆炸风险 |
| 功能开关 | `use_new_pipeline` 双轨期 | 直接替换 | 生产用户需可一键回滚 |
| L4 前落盘 | 仅 L4 force=True | L1 也落盘 | L1/L2 无损，无需留全尸；省磁盘 |
| 默认阈值 | 50 消息触发 L1 | 沿用 40 | 新管线 L1 是无损的，可以晚一点触发 |
| 多代理并发 | 路线图标红，需 spike | 直接定多线程 | 跨进程文件锁是高风险，需先验证 |
| Reactive 次数 | once-per-session | 多次重试 | 避免压缩→超限→压缩死循环 |

## 附录 B: 与现有 CLAUDE.md 原则的对齐

| CLAUDE.md 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰，能力在边缘 | Hooks（Phase 2）把扩展点从循环体内移到体外；新管线（Phase 1）替换 `maybe_compress` 单点 |
| Prompt Caching 神圣不可侵犯 | system prompt 仍只在会话开始构建一次；新增的 TOOL_USAGE_GUIDANCE 是首次构建时拼入；压缩仍走 invalidate 重建 |
| 完全可逆 | offload/transcript 文件永不自动删；Dream（Phase 5）必须备份；Task System 软删除沿用 |
| 用户意图优先于算法 | 所有阈值都可 config.yaml 覆盖；`use_new_pipeline` 允许完全关闭新管线 |
| 发现 ≠ 可见 | Hooks 注册和触发分离；Cron 启用/禁用独立字段 |
| 安全默认 > 事后补救 | offload/transcript 走 safe_path + agent_home 白名单；脚本 hook 走 PermissionChecker |

---

## 已知限制（Phase 2 待解决）

### I1: 占位消息 `role:"user"` 可能违反严格交替

**问题**：L1 snip_compact、L4 llm_compact、reactive_compact 三处占位消息都使用 `role:"user"`。当插入点前后也是 `role:"user"` 消息时（例如连续两条用户消息），会产生连续 user 消息。这在严格校验消息交替的 OpenAI 兼容 API（如 DeepSeek）上可能被拒绝。

**当前缓解**：
- `_fix_tool_call_pairs` 在每层压缩后统一跑一遍，能处理大部分 tool_call 配对问题
- 实际场景中，占位消息通常插入在 user/assistant 之间（因 LLM 回复后才会触发下一轮压缩），所以连续 user 的情况较少
- reactive_compact 和 llm_compact 都以占位 user 开头后接 keep_recent 消息，而 keep_recent 通常是 assistant+tool 交替

**Phase 2 修复方向**：
- 方案 A：把占位消息的 role 改为 `"system"`（部分 API 不支持 system 在非首位）
- 方案 B：在占位消息前插入一条 `role:"assistant"` 空消息（多 1 条开销但保证交替）
- 方案 C：在 `_fix_tool_call_pairs` 中增加连续 user 检测和修复逻辑
- 建议 Phase 2 做实际压力测试后选择方案

