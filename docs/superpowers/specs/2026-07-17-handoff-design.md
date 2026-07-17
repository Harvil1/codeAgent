# 会话移交（Handoff Bundle MVP）设计文档

- **日期**：2026-07-17
- **批次**：第 8 批改进（⑮ 会话移交）
- **范围**：中等（3-4 Task）
- **状态**：设计阶段

---

## 1. 背景与目标

### 1.1 Hermes 参考设计

`D:\project\hermes-agent-main\项目文档\06-Gateway-多平台接入.md` 的「六、会话移交」描述：CLI/Desktop 会话可通过 `/handoff <platform>` 转移到 Telegram/Discord 等平台，由 GatewayRunner、SessionEntry（含 `handoff_platform`/`handoff_state`/`handoff_error` 三字段）、双消息守卫、20+ 平台适配器协同完成。

### 1.2 HarvilAgent 现状

- 已有 `SessionStore`（SQLite + FTS5，`agent/session_store.py`）
- 已有 `RuntimeContext.new_session()` / `resume_session(id)`（`cli.py:305-357`）
- 已有 slash 命令分发器（`cli.py:431`）和 `/resume` 命令
- **没有 Gateway / 平台适配器**（⑬ 是独立大项目，未启动）

### 1.3 范围裁剪决策

强做「真·平台 handoff」需要 8-10 Task（超出 ⑬ 大项目边界），故本批次聚焦**会话状态可移植核心能力**：

- 把当前会话打包成自包含 JSON bundle（transcript + 元数据 + 引用）
- bundle 可在本机/异机 HarvilAgent 实例间导入导出
- 当未来 Gateway 上线时，bundle 即为发给平台 bot 的载荷格式

### 1.4 用例

1. 跨机迁移：机器 A 上做一半的调试 → `/handoff save` → 拷贝 JSON → 机器 B `/handoff import` 继续
2. 快照归档：项目交付、PR 提交时冻结对话状态
3. 调试复现：把出问题的 transcript 移到隔离实例重放
4. 为 ⑬ Gateway 铺路：bundle 格式即平台载荷格式

### 1.5 非目标（YAGNI）

- 平台适配器（Telegram/Discord 等）
- 跨进程实时消息桥
- Bundle 加密（依赖文件系统权限）
- Agent 自主调用（保持窄腰，仅用户主动触发）

---

## 2. Bundle 文件格式

### 2.1 存储位置

- 活跃 bundle：`~/.agent/.handoff/<ulid>.json`
- 归档（软删除）：`~/.agent/.handoff/.archive/<ulid>.json`

### 2.2 JSON Schema

```json
{
  "format_version": "1",
  "bundle_id": "01HXY8JQ4NAGSRF9K3EPZQWHM2",
  "created_at": "2026-07-17T12:34:56.789Z",
  "title": "调试 browser_cdp 工具",
  "source_session_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_platform": "cli",
  "model": {
    "name": "deepseek-chat",
    "provider": "deepseek"
  },
  "transcript": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_abc", "type": "function",
       "function": {"name": "terminal", "arguments": "{\"cmd\":\"ls\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "{...}"}
  ],
  "memory_pointers": ["01HXY8JQ4N", "01HXY8JQ4P"],
  "skill_states": {
    "pinned": ["k8s-troubleshooting"]
  },
  "todo_state": null,
  "task_pointers": ["01HXY8TASK1"],
  "handoff_state": "pending",
  "notes": "可选用户备注",
  "schema_checksum": "sha256:abcdef..."
}
```

### 2.3 字段规约

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `format_version` | string | 是 | 当前固定 `"1"`，未来兼容性判断 |
| `bundle_id` | string | 是 | ULID（26 字符），跨 bundle 全局唯一 |
| `created_at` | string | 是 | ISO8601 with milliseconds + `Z` |
| `title` | string | 否 | 用户给定或自动生成（首条 user 消息前 60 字符） |
| `source_session_id` | string | 否 | 原 SQLite session_id（可空） |
| `source_platform` | string | 是 | 当前固定 `"cli"`，未来扩展 `"telegram"` 等 |
| `model.name` | string | 是 | 保存时使用的模型名 |
| `model.provider` | string | 是 | provider 标识 |
| `transcript` | array | 是 | OpenAI Chat Completions 格式消息数组（含 tool_calls） |
| `memory_pointers` | array | 否 | memory ID（ULID）列表，不内联 body |
| `skill_states.pinned` | array | 否 | pinned 技能名列表 |
| `todo_state` | object \| null | 否 | TodoWrite 当前状态（如非空） |
| `task_pointers` | array | 否 | 跨会话 Task System 任务 ID |
| `handoff_state` | string | 是 | `pending` / `completed`（未来扩展 `in_progress`/`failed`） |
| `notes` | string | 否 | 用户备注（API 层参数，本期 CLI 不暴露 flag） |
| `schema_checksum` | string | 是 | transcript 序列化后的 SHA256，前缀 `sha256:` |

### 2.4 transcript 序列化规约

为让 `schema_checksum` 稳定：

```python
transcript_bytes = json.dumps(
    transcript,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
checksum = "sha256:" + hashlib.sha256(transcript_bytes).hexdigest()
```

---

## 3. CLI 命令

### 3.1 命令清单

| 命令 | 作用 | 别名/简写 |
|---|---|---|
| `/handoff save [title]` | 从当前会话生成 bundle 写入 `.handoff/` | `/handoff` 等价于 `/handoff save` |
| `/handoff list` | 列出所有 bundle | `/handoff ls` |
| `/handoff load <id\|index>` | 加载 bundle 覆盖当前会话 | `/handoff ld` |
| `/handoff show <id\|index>` | 显示 bundle 元信息 + 最近 6 条预览 | `/handoff sh` |
| `/handoff delete <id\|index>` | 软删除（移到 `.archive/`） | `/handoff rm` |
| `/handoff export <id\|index> <path>` | 拷贝 bundle 到任意路径 | — |
| `/handoff import <path>` | 从外部路径导入 bundle | — |
| `/handoff help` | 显示子命令帮助 | — |

### 3.2 ID 解析规则

`<id|index>` 参数支持：

1. **完整 ULID**（26 字符）：精确匹配
2. **ULID 前缀**：≥4 字符；歧义时列出所有匹配让用户重选
3. **`list` 中的序号**：`0` 是最新；解析时查当前 list 顺序

⚠️ **序号不稳定**：`/handoff save` 或 `/handoff delete` 后 list 顺序会变。脚本化场景应使用 ULID 前缀而非序号。

### 3.3 命令分发

`/handoff` 在 `_handle_command`（`cli.py:431`）加分支，转发到新函数 `_handle_handoff_command(sub, args, rt)`。

### 3.4 与现有 `/resume` 的关系

- `/resume` 从 SQLite 会话恢复（本机原会话）
- `/handoff load` 从 bundle 恢复（跨机或归档）
- 两者独立，不互斥；bundle load 后会落 SQLite 留痕（新建 session_id）

---

## 4. 模块结构

```
agent/handoff.py            ← HandoffBundle + HandoffStore（业务逻辑）
cli.py                      ← _handle_handoff_command 分发（薄层）
tests/test_handoff.py       ← 单元 + 集成测试
```

### 4.1 HandoffStore API

```python
@dataclass
class HandoffBundleMeta:
    bundle_id: str
    created_at: datetime
    title: Optional[str]
    message_count: int
    handoff_state: str
    file_size: int

@dataclass
class HandoffBundle:
    format_version: str
    bundle_id: str
    created_at: datetime
    title: Optional[str]
    source_session_id: Optional[str]
    source_platform: str
    model: Dict[str, str]
    transcript: List[dict]
    memory_pointers: List[str]
    skill_states: Dict[str, Any]
    todo_state: Optional[Dict[str, Any]]
    task_pointers: List[str]
    handoff_state: str
    notes: Optional[str]
    schema_checksum: str

class HandoffStore:
    MAX_BUNDLE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, handoff_dir: Path): ...

    def save(self, *, transcript: List[dict],
             source_session_id: Optional[str],
             model: Dict[str, str],
             title: Optional[str] = None,
             memory_pointers: Optional[List[str]] = None,
             skill_states: Optional[Dict[str, Any]] = None,
             todo_state: Optional[Dict[str, Any]] = None,
             task_pointers: Optional[List[str]] = None,
             notes: Optional[str] = None,
             source_platform: str = "cli",
             allow_secrets: bool = False) -> str:
        """生成 bundle 写入磁盘，返回 bundle_id。
        - 扫描 transcript 找密钥模式（除非 allow_secrets=True）
        - 检查大小 ≤ MAX_BUNDLE_SIZE_BYTES
        - 计算 schema_checksum
        - 原子写入（先写 .tmp，再 rename）
        """

    def load(self, bundle_id_or_index: str) -> HandoffBundle:
        """加载并返回 HandoffBundle。
        - 校验 format_version
        - 校验 schema_checksum（不匹配警告但不崩）
        - 找不到则抛 FileNotFoundError
        """

    def list_bundles(self) -> List[HandoffBundleMeta]:
        """按 created_at 倒序返回（不含 .archive/）。"""

    def resolve_id(self, query: str) -> str:
        """把 ULID 前缀或 list 序号解析为完整 bundle_id。
        - 歧义抛 ValueError 含所有匹配
        """

    def delete(self, bundle_id: str) -> Path:
        """软删除：移到 .archive/<bundle_id>.json，返回新路径。"""

    def export_to(self, bundle_id: str, dest_path: Path) -> Path:
        """拷贝 bundle 到任意路径（不改 bundle_id）。"""

    def import_from(self, src_path: Path) -> str:
        """从外部路径导入 bundle，返回 bundle_id。
        - 如果 bundle_id 已存在，生成新 ULID（其他字段保留，包括 source_platform）
        - 校验 format_version + schema_checksum
        - 不改 source_platform（机器 B 导入机器 A 的 bundle 后仍标记原平台）
        """

    def mark_completed(self, bundle_id: str) -> None:
        """把 handoff_state 改为 'completed'（load 成功后调用）。"""
```

### 4.2 HandoffError 异常层次

```python
class HandoffError(Exception):
    """所有 handoff 错误的基类。"""

class BundleNotFoundError(HandoffError):
    """bundle_id 不存在。"""

class BundleCorruptedError(HandoffError):
    """format_version 不支持 或 schema_checksum 不匹配。"""

class BundleTooLargeError(HandoffError):
    """bundle 超过 MAX_BUNDLE_SIZE_BYTES。"""

class SecretDetectedError(HandoffError):
    """transcript 含密钥模式（默认拒绝保存）。"""

class AmbiguousBundleIDError(HandoffError):
    """ULID 前缀匹配多个 bundle，含 candidates 字段。"""
```

---

## 5. 安全与韧性

### 5.1 密钥扫描

`/handoff save` 默认拒绝含以下模式的 transcript（防止把 API key 跟着 bundle 带出去）：

```python
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI/DeepSeek
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    re.compile(r"api_key[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"),
    re.compile(r"token[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]
```

扫描所有 `user`/`assistant`/`tool` 消息的 content。命中即报 `SecretDetectedError`，提示用户：
- 改写消息后再保存
- 或显式 `/handoff save --allow-secrets`（不推荐，CLI 不暴露此 flag，需通过 API）

### 5.2 大小上限

`MAX_BUNDLE_SIZE_BYTES = 10 MB`。超过则报 `BundleTooLargeError`，提示用户：
- `/compress` 压缩上下文后再保存
- 或仅保存最近 N 轮（未来扩展，本期不做）

### 5.3 完整性校验

- 保存时计算 `schema_checksum`
- 加载时重算并对比：不匹配警告但允许加载（用户自负）
- `format_version` 不为 `"1"`：拒绝加载

### 5.4 原子写入

```python
tmp_path = target.with_suffix(".json.tmp")
tmp_path.write_text(json_str, encoding="utf-8")
tmp_path.replace(target)  # 原子 rename
```

避免中途崩溃留下半写文件。

### 5.5 软删除

`delete` 把 bundle 移到 `.archive/` 子目录，不硬删（设计原则 #3）。未来可加清理策略（如 30 天后真删），本期不做。

---

## 6. 数据流示例

### 6.1 save → export → import → load 完整往返

```
[机器 A]
/handoff save "调试 browser_cdp"
  → HandoffStore.save(transcript=agent.conversation_history, ...)
  → 写入 ~/.agent/.handoff/01HXY8JQ4N...json
  → 输出 "✓ bundle 01HXY8JQ4N 已保存（12 条消息，3.2KB）"

/handoff export 0 ~/Downloads/debug-bundle.json
  → HandoffStore.export_to("01HXY8JQ4N", Path("~/Downloads/debug-bundle.json"))
  → 拷贝文件

[拷贝 debug-bundle.json 到机器 B]

[机器 B]
/handoff import ~/Downloads/debug-bundle.json
  → HandoffStore.import_from(src_path)
  → 校验 format_version + schema_checksum
  → bundle_id 已存在则重新生成 ULID
  → 拷贝到 ~/.agent/.handoff/<ulid>.json
  → 输出 "✓ 已导入 bundle <ulid>"

/handoff load 0
  → HandoffStore.load("0") → HandoffBundle
  → 检查 memory_pointers：5 个中 2 个本机不存在 → 警告
  → 提示 "将覆盖当前会话（当前 8 条消息）? (y/N)"
  → 用户 y
  → agent.conversation_history = bundle.transcript
  → agent.invalidate_system_prompt()
  → session_store.create_session(...)  # 新建 SQLite 会话留痕
  → agent.session_id = new_session_id
  → HandoffStore.mark_completed(bundle.bundle_id)
  → 输出 "✓ 已加载（12 条消息，标题：调试 browser_cdp）"
```

---

## 7. 错误处理矩阵

| 场景 | 错误类 | CLI 表现 |
|---|---|---|
| bundle_id 不存在 | `BundleNotFoundError` | `[red]未找到 bundle: ...[/red]` |
| ULID 前缀歧义 | `AmbiguousBundleIDError` | 列出候选让用户重选 |
| format_version 未知 | `BundleCorruptedError` | `[red]不支持的 format_version: 2[/red]` |
| schema_checksum 不匹配 | 警告（不抛） | `[yellow]⚠️ checksum 不匹配，文件可能损坏，仍尝试加载[/yellow]` |
| transcript 含密钥 | `SecretDetectedError` | `[red]检测到密钥模式，拒绝保存[/red]` + 命中行号 |
| bundle > 10 MB | `BundleTooLargeError` | `[red]bundle 过大（X MB），上限 10MB。建议先 /compress[/red]` |
| load 时 memory_pointers 部分缺失 | 警告（不抛） | `[yellow]⚠️ 5 个 memory 引用中 2 个本机不存在[/yellow]` |
| 当前会话非空且用户在 load | 确认提示 | `将覆盖当前 X 条消息，是否先 /handoff save? (y/N)` |
| import 路径不存在 | `FileNotFoundError` | `[red]文件不存在: ...[/red]` |
| JSON 解析失败 | `json.JSONDecodeError` | `[red]bundle 文件损坏：JSON 解析失败[/red]` |

---

## 8. 测试策略

`tests/test_handoff.py` 新增 ~18 个测试：

### 8.1 HandoffStore 单元测试

1. `test_save_creates_valid_bundle` — 保存后文件存在、format_version="1"、ULID 合法、checksum 正确
2. `test_load_preserves_transcript_byte_for_byte` — save 后立即 load，transcript 逐字段相等
3. `test_list_bundles_sorted_by_created_at_desc` — 多个 bundle 按时间倒序
4. `test_delete_is_soft_to_archive` — 删除后文件出现在 `.archive/`，能从 `.archive/` 恢复
5. `test_export_import_roundtrip` — export 后 import 到新 store，内容一致
6. `test_import_with_existing_id_regenerates_ulid` — 同 bundle_id 二次导入得到新 ID
7. `test_resolve_id_full_ulid` — 完整 ULID 精确匹配
8. `test_resolve_id_prefix_unique` — 4 字符前缀唯一时正确解析
9. `test_resolve_id_prefix_ambiguous` — 歧义抛 `AmbiguousBundleIDError` 含候选列表
10. `test_resolve_id_list_index` — 序号 0 是最新
11. `test_secret_detection_rejects_save` — 含 `sk-xxx` 的 user 消息触发 `SecretDetectedError`
12. `test_secret_detection_multiple_patterns` — Bearer / api_key / token / PEM 私钥 各命中一次
13. `test_size_limit_rejects_large_bundle` — 构造 11MB transcript 触发 `BundleTooLargeError`
14. `test_unknown_format_version_rejected` — 手工构造 format_version="2" 的 bundle 加载失败
15. `test_checksum_mismatch_warns_but_loads` — 改 transcript 后 load 不抛但记 warning
16. `test_atomic_write_no_partial_file_on_error` — 模拟写入中途失败，无 `.json.tmp` 残留
17. `test_mark_completed_updates_state` — load 后 `handoff_state == "completed"`

### 8.2 CLI 集成测试

18. `test_handoff_save_load_e2e` — mock AIAgent + RuntimeContext，验证 `/handoff save` → `/handoff list` → `/handoff load` 完整流程

### 8.3 测试基础设施

- 复用 `tests/conftest.py` 的 tmp_path fixture（如果存在）；否则用 `tmp_path` 内联
- Mock `agent.conversation_history` 用固定示例消息
- 不调真实 LLM；不依赖 `~/.agent` 真实目录

---

## 9. 与现有架构的契合

| 设计原则（CLAUDE.md） | 本设计如何遵守 |
|---|---|
| 核心是窄腰（#1） | 不新增 agent 工具，逻辑在 `agent/handoff.py`，CLI 薄层分发 |
| Prompt Cache 神圣不可侵犯（#2） | load 时调 `invalidate_system_prompt()`，不破坏缓存契约 |
| 完全可逆（#3） | delete 走 `.archive/`，永不硬删 |
| 用户意图优先（#4） | load 前确认覆盖，导出/删除均有交互提示 |
| 发现 ≠ 可见（#5） | 不暴露 agent 工具，所以不涉及 `check_fn` |
| 安全默认（#6） | 密钥扫描默认开启、大小上限默认 10MB、原子写入 |

---

## 10. 留给未来的接口

### 10.1 Gateway 载荷格式

当 ⑬ Gateway 实现时：
- bundle JSON 即平台 bot 收到的载荷
- `source_platform` 字段从 `"cli"` 扩展为 `"telegram"` 等
- `handoff_state` 状态机扩展 `in_progress` / `failed`

### 10.2 Bundle 压缩（未来）

如需压缩（10MB 不够），未来可加 `format_version: "2"` 支持 gzip 压缩 transcript。本期不做。

### 10.3 增量 handoff（未来）

未来可加 `parent_bundle_id` 字段支持「派生 bundle」链。本期不做。

---

## 11. 实施任务分解（供 writing-plans 参考）

预计 4 个 Task：

1. **Task 1**：`agent/handoff.py` 核心 — `HandoffBundle`、`HandoffBundleMeta`、`HandoffStore`（save/load/list/resolve_id）+ 异常层次
2. **Task 2**：`agent/handoff.py` 扩展 — delete（软删除）+ export_to + import_from + mark_completed + 密钥扫描 + 大小上限 + 原子写入
3. **Task 3**：`cli.py` 集成 — `_handle_handoff_command` + 7 个子命令 + `_handle_command` 分发分支 + `/help` 更新
4. **Task 4**：`tests/test_handoff.py` — 18 个测试 + 跑 `uv run pytest tests/` 确认无回归 + commit + push

---

## 12. 验收标准

- [ ] `agent/handoff.py` 实现完整 HandoffStore API
- [ ] CLI 7 个子命令全部可用
- [ ] `tests/test_handoff.py` ≥18 个测试全部通过
- [ ] `uv run pytest tests/` 总数 ≥805（当前 787 + 18）
- [ ] `uv run python scripts/verify.py` 22 项全 PASS（无回归）
- [ ] 手工验证：save → export → import → load 完整往返
- [ ] commit + push 到 origin/master
