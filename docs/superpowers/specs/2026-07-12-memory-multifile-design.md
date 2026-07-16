# 记忆系统重构：Claude Code 多文件模式

- **日期**：2026-07-12
- **状态**：用户授权直接推进
- **范围**：把现有 2-文件硬上限记忆系统改造为 Claude Code 风格的多文件 + 索引 + 按需检索
- **对应决策**：迁移=C（不迁移）/ 检索=B（LLM 每轮选 5 条）/ 分类=A（统一池 + frontmatter type）
- **替代**：spec §6 原 Phase 5a Dream（基于错误假设，已废弃）

---

## 摘要

把 `~/.agent/MEMORY.md` + `USER.md` 双文件硬上限模型，重构为 `~/.agent/.memory/{ulid}.md` 多文件 + `MEMORY.md` 索引 + 每轮 LLM 检索 top-5 的 Claude Code 模式。容量从 ~3500 字符 → 基本无限。每轮主 LLM 调用前 +1 次检索 LLM 调用。

**老数据按决策 C 不迁移**——直接清空。用户需自行备份旧 `MEMORY.md` / `USER.md`。

---

## §1 架构

```
~/.agent/
├── MEMORY.md                    # 索引（每条 1 行 + 链接），system prompt 注入
├── .memory/
│   ├── 01JABC123.md             # 单条记忆，frontmatter + body
│   ├── 01JABC456.md
│   └── ...
├── .archive/
│   └── memory-{ts}/             # 删除的记忆软删除到这里
└── ...

每轮主 LLM 调用前：
  ① 读 MEMORY.md 索引（已缓存）
  ② 调检索 LLM：input=user_msg + index，output=top-5 memory_id
  ③ 读这 5 个 .memory/{id}.md 的 body
  ④ 注入 user 消息："<relevant_memories>...</relevant_memories>"
  ⑤ 主 LLM 调用
```

### 文件边界

| 文件 | 状态 | 职责 |
|---|---|---|
| `agent/memory_store.py` | ♻️ 重写 | 多文件 + 索引 + frontmatter 解析 |
| `agent/memory_retriever.py` | 🆕 新增 | `retrieve_relevant(query, index, llm_client) -> list[str]` |
| `tools/memory_tool.py` | ♻️ 重写 | 新 schema：save / update / delete / load / list |
| `agent/prompt_builder.py` | ♻️ 改 | system prompt 改用新索引（不再调 `format_for_system_prompt`） |
| `agent/__init__.py` | ♻️ 改 | 主循环每轮调 retriever + 注入 `<relevant_memories>` |
| `cli.py` | ♻️ 改 | RuntimeContext 装配（retriever 用主 client） |
| `config.py` | ♻️ 改 | `memory` 块新增字段 |
| `tests/test_memory.py` | ♻️ 重写 | 全部记忆系统测试 |

### 关键决策

1. **老数据不迁移**：启动时检测旧 `MEMORY.md` 格式（无 frontmatter），备份到 `.archive/legacy-memory-{ts}/`，新建空索引。
2. **每轮 +1 LLM 调用做检索**：成本约 $0.0001/轮，可接受。检索用主 client（用户可在 config 配置改用便宜模型）。
3. **写入立即落盘，索引下次会话生效**：保护 prompt cache（同 Phase 1 原则）。检索 LLM 读最新磁盘状态，新写入的 memory 本会话内可被检索到，但不进 system prompt 索引。
4. **5 类 type**：`user` / `feedback` / `project` / `reference` / `other`。
5. **软删除**：删除移到 `.archive/memory-{ts}/{id}.md`，可恢复。

---

## §2 文件格式

### 单条记忆 `~/.agent/.memory/{ulid}.md`

```markdown
---
name: 用户偏好简洁回复
description: 用尽量少的字数回答，不重复用户已知信息
type: user
created_at: 2026-07-12T15:30:00
updated_at: 2026-07-12T15:30:00
---

用户明确表示过 3 次，希望回复简短直接。多次追问后会抱怨啰嗦。
适用场景：所有对话回复。
```

字段：
- `name`（必需）：唯一短标识（≤30 字符）
- `description`（必需）：一行索引描述（≤80 字符）
- `type`（必需）：`user` / `feedback` / `project` / `reference` / `other`
- `created_at` / `updated_at`（必需）：ISO 时间戳
- body（可选）：完整正文

文件名：`{ulid}.md`（时间排序 + 唯一），不用 name（避免命名冲突 + 特殊字符）。

### 索引文件 `~/.agent/MEMORY.md`

```markdown
# Memory Index

自动生成，请勿手动编辑。每行格式：`- [name](.memory/{id}.md) — description`

- [用户偏好简洁回复](.memory/01JABC123.md) — 用尽量少的字数回答，不重复用户已知信息
- [项目用 pytest](.memory/01JABC456.md) — HermesAgent 测试用 pytest + xdist
- ...
```

启动时从 `.memory/*.md` 扫描重建（用户手改会被覆盖）。

---

## §3 数据结构

```python
@dataclass
class MemoryEntry:
    id: str               # ulid，如 "01JABC123"
    name: str
    description: str
    type: str             # user/feedback/project/reference/other
    body: str             # 完整正文
    created_at: datetime
    updated_at: datetime
```

```python
class MemoryStore:
    """多文件记忆存储。每次写都落盘。"""

    def __init__(self, *, harvil_home: Path): ...

    # ---- 读 ----
    def list_all(self) -> list[MemoryEntry]: ...
    def get(self, memory_id: str) -> Optional[MemoryEntry]: ...
    def build_index_text(self) -> str:
        """生成 MEMORY.md 索引内容（启动时用）。"""

    def snapshot_for_prompt(self) -> str:
        """返回索引文本，供 system prompt 注入（frozen，本会话不变）。"""

    # ---- 写 ----
    def save(self, *, name: str, description: str, type: str,
             body: str) -> str:
        """创建新记忆。返回 memory_id。立即落盘 + 更新索引文件。"""

    def update(self, memory_id: str, *, name=None, description=None,
               type=None, body=None) -> MemoryEntry:
        """更新已有记忆字段。立即落盘 + 重建索引。"""

    def delete(self, memory_id: str) -> bool:
        """软删除：移到 .archive/memory-{ts}/。"""

    def load_body(self, memory_id: str) -> Optional[str]:
        """读 body。memory_load 工具用。"""
```

---

## §4 检索器（`agent/memory_retriever.py`）

```python
def retrieve_relevant(
    query: str,
    index_text: str,
    *,
    llm_client,
    model: str,
    max_results: int = 5,
) -> list[str]:
    """调 LLM 从索引中选 top-N 最相关 memory_id。

    流程：
    1. 构造 prompt：query + index + 输出格式说明
    2. 调 llm_client.chat_completions
    3. 解析输出为 list[memory_id]
    4. 失败（解析错/超时/异常）→ 返回 []（fail-open）

    返回：list[memory_id]，最多 max_results 个
    """
```

### 检索 prompt 模板

```
你是记忆检索助手。当前用户消息：

<query>
{query}
</query>

可用记忆索引（每行一条）：

<index>
{index_text}
</index>

返回最多 {max_results} 条与当前 query 最相关的记忆 ID。
格式：JSON 数组，元素是 ID 字符串（如 ["01JABC123", "01JABC456"]）。
只返回 JSON，不要其他文本。
若没有相关的，返回 []。
```

---

## §5 主循环集成（`agent/__init__.py`）

`run_conversation` 顶部（在 USER_PROMPT_SUBMIT hook 之后、drain notifications 之后）：

```python
# === NEW: 相关记忆检索 ===
relevant_memories_text = ""
if self.memory_retriever and self.memory_store:
    try:
        query = user_message  # 当前用户消息
        index_text = self._cached_memory_index  # 会话级 frozen 缓存
        relevant_ids = self.memory_retriever.retrieve_relevant(
            query=query,
            index_text=index_text,
            llm_client=self.llm_client,
            model=self.model,
            max_results=5,
        )
        if relevant_ids:
            bodies = []
            for mid in relevant_ids:
                body = self.memory_store.load_body(mid)
                if body:
                    bodies.append(f"[{mid}]\n{body}")
            if bodies:
                relevant_memories_text = "\n\n".join(bodies)
    except Exception as e:
        logger.warning("memory retrieval 失败（fail-open）: %s", e)
        relevant_memories_text = ""
```

主循环 messages 组装时，注入到 user_message（不是 system，保护 cache）：

```python
user_content = user_message
if relevant_memories_text:
    user_content = (
        f"<relevant_memories>\n{relevant_memories_text}\n</relevant_memories>\n\n"
        f"{user_message}"
    )
self.conversation_history.append({"role": "user", "content": user_content})
```

⚠️ **关键差异**：`<relevant_memories>` 是**持久**进入 conversation_history 的（不像 bg_task 通知是临时）。原因：相关记忆是上下文的一部分，不是状态通知。压缩时和普通 user 消息一起处理。

`_cached_memory_index` 在 `__init__` 时调用 `memory_store.snapshot_for_prompt()` 缓存，会话内不变。

---

## §6 工具表面（`tools/memory_tool.py` 重写）

5 个 action（替换原 add/replace/remove）：

```python
MEMORY_SCHEMA = {
    "name": "memory",
    "description": "管理持久化记忆...",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "update", "delete", "load", "list"],
            },
            "id": {"type": "string", "description": "update/delete/load 时必需"},
            "name": {"type": "string", "description": "save 时必需；update 可选"},
            "description": {"type": "string", "description": "save 时必需；update 可选"},
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference", "other"],
                "description": "save 时必需；update 可选",
            },
            "body": {"type": "string", "description": "save/update 时可选"},
        },
        "required": ["action"],
    },
}
```

返回 JSON 字符串。每个 action 单独 handler 内分支。

---

## §7 prompt_builder 改造

`agent/prompt_builder.py:build_system_prompt` 现在：

```python
# 5. 记忆快照（frozen）
if memory_store:
    try:
        mem_block = memory_store.format_for_system_prompt("memory")
        user_block = memory_store.format_for_system_prompt("user")
        if mem_block:
            parts.append(mem_block)
        if user_block:
            parts.append(user_block)
    except Exception:
        pass
```

改为：

```python
# 5. 记忆索引（frozen，多文件模式）
if memory_store:
    try:
        index_block = memory_store.snapshot_for_prompt()
        if index_block:
            parts.append(f"## 记忆索引\n{index_block}")
    except Exception:
        pass
```

---

## §8 配置（`config.py`）

```python
"memory": {
    "enabled": True,
    "provider": None,
    # 新增字段（Phase 5）
    "multifile_enabled": True,                # False 时回退到旧 2-文件模式（向后兼容）
    "memory_dir": None,                       # 默认 ~/.agent/.memory/
    "retrieval_enabled": True,                # False 时跳过每轮检索（节省 LLM 调用）
    "retrieval_max_results": 5,
    "retrieval_model": None,                  # None → 用主 model
    # 沿用字段（旧模式 / 历史兼容）
    "memory_char_limit": 2200,                # 仅 multifile_enabled=False 时生效
    "user_char_limit": 1375,
},
```

⚠️ **双轨期**：`multifile_enabled=False` 时回退到旧模式。用户从此前版本升级默认 `True`（按决策 C 老数据不迁移，直接进新模式）。

---

## §9 失败处理

| 故障 | 行为 |
|---|---|
| 检索 LLM 超时/异常 | log warning，relevant_memories_text = ""，主循环不阻塞 |
| 检索 LLM 返回非合法 JSON | log warning，返回 [] |
| 检索返回的 memory_id 不存在 | 跳过该 id，log debug |
| 读 memory body 失败 | 跳过该条 |
| 索引文件不存在 | 启动时新建空索引 |
| 单条 memory 文件 frontmatter 解析失败 | log warning，跳过该文件 |
| 写新 memory 时磁盘满 | raise OSError，工具层捕获返 error JSON |

---

## §10 测试矩阵

### `tests/test_memory.py`（重写，~15 个）

- `test_save_creates_file_and_updates_index`
- `test_save_minimal_fields`
- `test_save_invalid_type_raises`
- `test_get_returns_entry`
- `test_get_unknown_returns_none`
- `test_list_all_returns_entries`
- `test_update_modifies_fields`
- `test_update_unknown_raises`
- `test_delete_moves_to_archive`
- `test_delete_unknown_returns_false`
- `test_load_body_returns_content`
- `test_load_body_unknown_returns_none`
- `test_snapshot_for_prompt_returns_index_text`
- `test_index_rebuilt_on_startup`
- `test_legacy_memory_files_archived_on_startup`（决策 C：旧 MEMORY.md → .archive/）

### `tests/test_memory_retriever.py`（新增，~6 个）

- `test_retrieve_relevant_returns_ids`
- `test_retrieve_relevant_handles_empty_index`
- `test_retrieve_relevant_handles_llm_failure_returns_empty`
- `test_retrieve_relevant_handles_malformed_json_returns_empty`
- `test_retrieve_relevant_respects_max_results`
- `test_retrieve_relevant_uses_correct_prompt`

### `tests/test_integration.py`（~3 个新增）

- `test_aiagent_injects_relevant_memories_into_user_msg`
- `test_aiagent_no_memory_retriever_backward_compat`
- `test_retrieval_failure_does_not_break_main_loop`

---

## §11 已知限制 / 非目标

1. **不实现语义向量检索**：用 LLM 选 top-5 而非 embedding。性能足够，零基础设施依赖。
2. **不实现记忆重要性评分**：所有 memory 平等。未来可加 `use_count` / `last_accessed` 字段。
3. **不实现 TTL 自动过期**：用户显式 delete 才清理。未来可加 `expires_at` 字段。
4. **不实现跨 agent 共享**：所有 memory 在单个 `~/.agent/` 下。多 agent 实例（如 Phase 4 Teams）需要扩展。
5. **检索 LLM 调用不可省**：除非 `config.memory.retrieval_enabled=False`。每轮固定 +1 次 LLM 调用。
6. **首次会话不触发检索**：因为索引为空。`memory_save` 第一次调后下次会话才在索引里。

---

## 附录 A: 决策记录

| 决策 | 选择 | 替代 | 理由 |
|---|---|---|---|
| 迁移 | 不迁移（C） | 自动迁移（A） | 用户决策；简化首版实现 |
| 检索方式 | LLM 每轮选 5（B） | system prompt 全索引（A） | 用户决策；更接近 Claude Code；索引大时不撑爆 system prompt |
| 分类 | 统一池 + type 字段（A） | 双池 MEMORY/USER（B） | 用户决策；更符合 Claude Code 模式 |
| 文件名 | ULID | name slug | 避免特殊字符 + 时间排序 |
| 索引位置 | system prompt（frozen） | user turn | 会话内稳定，保护 prompt cache |
| 相关记忆注入位置 | user turn（持久进 history） | 临时消息 | 是上下文，不是状态通知 |
| 删除 | 软删除（移到 .archive） | 硬删除 | CLAUDE.md "完全可逆" 原则 |
| 检索模型 | 主 model（可配） | 强制便宜模型 | 简单；用户可在 config 切换 |

## 附录 B: 与现有原则对齐

| 原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰 | memory 工具是数据驱动；retriever 是独立模块 |
| Prompt Caching 神圣 | 索引 frozen 缓存在 system prompt；新写入下次会话才生效 |
| 完全可逆 | 软删除到 .archive；旧 MEMORY.md 备份到 .archive/legacy-memory-{ts}/ |
| 用户意图优先 | retrieval_enabled 开关；用户可显式 delete |
| 安全默认 | 检索失败 fail-open；写入失败返 error |
