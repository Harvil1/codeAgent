# 记忆系统重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** 把 2-文件硬上限记忆系统改造为 Claude Code 多文件模式（`.memory/{ulid}.md` + 索引 + 每轮 LLM 检索 top-5）。

**Architecture:** 重写 `agent/memory_store.py` 为多文件 + frontmatter；新增 `agent/memory_retriever.py`；改 `tools/memory_tool.py` 5-action；改 `agent/__init__.py` 主循环注入 `<relevant_memories>`；改 `prompt_builder.py` 用索引代替 frozen entries。

**Tech Stack:** Python 3.11+、ulid（或自实现）、PyYAML、uv、pytest。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-memory-multifile-design.md`

## Global Constraints

- 文件 I/O 必须 `encoding="utf-8"`
- 用 `uv`，不要 `pip install`；新增依赖用 `uv add`
- 中文注释/commit；英文标识符
- 不要 import 用不到的模块（ruff F401）
- 工具 handler 返回 JSON 字符串
- 507 测试不得回归
- Subagent 不 commit；controller 统一 commit
- 老数据不迁移（决策 C），现有 MEMORY.md / USER.md 启动时备份到 `.archive/legacy-memory-{ts}/`
- 写入立即落盘，但 system prompt 索引下次会话才生效（保护 prompt cache）
- 检索失败 fail-open（log + 空注入）

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/memory_store.py` | T1 重写 | 多文件 MemoryStore + frontmatter 解析 + ULID 文件名 + 索引重建 |
| `agent/memory_retriever.py` | T2 新增 | LLM 检索 top-5 |
| `tools/memory_tool.py` | T3 重写 | 5-action: save / update / delete / load / list |
| `agent/prompt_builder.py` | T4 改 | system prompt 用 snapshot_for_prompt() 索引代替旧 format_for_system_prompt |
| `agent/__init__.py` | T5 改 | AIAgent.cached_memory_index + 主循环 retriever 注入 |
| `config.py` | T6 改 | memory 块新增字段 |
| `cli.py` | T7 改 | RuntimeContext 装配 memory_retriever |
| `tests/test_memory.py` | T1 重写 | MemoryStore 单元测试 |
| `tests/test_memory_retriever.py` | T2 新增 | retriever 单元测试 |
| `tests/test_integration.py` | T5/T8 改 | 主循环注入 + e2e |

---

## Task 1: agent/memory_store.py 多文件重写

**Files:**
- Modify: `agent/memory_store.py`（整文件重写）
- Modify: `tests/test_memory.py`（整文件重写）

**Interfaces:**
- Consumes: `pathlib`、`yaml`（用 `uv add pyyaml` 加）、`datetime`、`logging`、`typing`
- Produces: `MemoryEntry` dataclass、`MemoryStore` 类（save / update / delete / load_body / get / list_all / build_index_text / snapshot_for_prompt / migrate_legacy_if_any）

- [ ] **Step 1: 加依赖**

Run: `uv add pyyaml`
（PyYAML 用于解析 frontmatter；保持与现有 SKILL.md frontmatter 解析一致）

- [ ] **Step 2: 写失败测试**

```python
# tests/test_memory.py （整文件重写）
"""多文件记忆系统测试。"""
import time
from pathlib import Path

import pytest

from agent.memory_store import MemoryStore, MemoryEntry


def test_save_creates_file_and_updates_index(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="用户偏好简洁回复",
        description="用尽量少的字数回答",
        type="user",
        body="用户多次要求简短直接回复",
    )
    assert mid  # non-empty string
    # 文件应存在
    mem_file = tmp_path / ".memory" / f"{mid}.md"
    assert mem_file.exists()
    # 索引文件 MEMORY.md 也应被更新
    index_text = tmp_path / "MEMORY.md"
    assert index_text.exists()
    assert mid in index_text.read_text(encoding="utf-8")


def test_save_minimal_fields(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="x", description="y", type="other",
    )  # 无 body
    entry = store.get(mid)
    assert entry.body == ""


def test_save_invalid_type_raises(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    with pytest.raises(ValueError):
        store.save(name="x", description="y", type="invalid_kind", body="")


def test_get_returns_entry(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b")
    entry = store.get(mid)
    assert entry.id == mid
    assert entry.name == "t1"
    assert entry.type == "user"


def test_get_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.get("nonexistent") is None


def test_list_all_returns_entries(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    store.save(name="t2", description="d", type="project", body="")
    all_entries = store.list_all()
    assert len(all_entries) == 2


def test_update_modifies_fields(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b1")
    store.update(mid, name="t1-new", body="b2")
    entry = store.get(mid)
    assert entry.name == "t1-new"
    assert entry.body == "b2"
    assert entry.description == "d"  # 未改


def test_update_unknown_raises(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    with pytest.raises(KeyError):
        store.update("nonexistent", body="x")


def test_delete_moves_to_archive(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t1", description="d", type="user", body="b")
    ok = store.delete(mid)
    assert ok is True
    # 主目录文件不存在
    assert not (tmp_path / ".memory" / f"{mid}.md").exists()
    # archive 下能找到
    archives = list((tmp_path / ".archive").glob("memory-*/" + f"{mid}.md"))
    assert len(archives) >= 1


def test_delete_unknown_returns_false(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.delete("nonexistent") is False


def test_load_body_returns_content(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="完整内容")
    assert store.load_body(mid) == "完整内容"


def test_load_body_unknown_returns_none(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    assert store.load_body("nonexistent") is None


def test_snapshot_for_prompt_returns_index_text(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="描述1", type="user", body="")
    store.save(name="t2", description="描述2", type="project", body="")
    snap = store.snapshot_for_prompt()
    assert "t1" in snap
    assert "描述1" in snap
    assert "t2" in snap


def test_index_rebuilt_on_startup(tmp_path: Path):
    """新建 store 时扫描 .memory/ 重建索引。"""
    # 先用 store1 写两个 memory
    store1 = MemoryStore(harvil_home=tmp_path)
    mid1 = store1.save(name="t1", description="d", type="user", body="")
    mid2 = store1.save(name="t2", description="d", type="project", body="")
    # 再开一个 store（模拟下次会话），应能看到两条
    store2 = MemoryStore(harvil_home=tmp_path)
    all_entries = store2.list_all()
    assert len(all_entries) == 2
    assert {e.id for e in all_entries} == {mid1, mid2}


def test_migrate_legacy_archives_old_files(tmp_path: Path):
    """启动时检测旧 MEMORY.md / USER.md 格式（无 frontmatter），备份到 .archive/。"""
    # 写一个旧格式 MEMORY.md
    (tmp_path / "MEMORY.md").write_text(
        "# Agent Memory\n\n- 老记忆 1\n- 老记忆 2\n",
        encoding="utf-8",
    )
    (tmp_path / "USER.md").write_text(
        "# User Profile\n\n- 老用户画像\n",
        encoding="utf-8",
    )

    # 启动 store 应备份老文件 + 新建空索引
    store = MemoryStore(harvil_home=tmp_path)
    # 老文件已被新（空）索引覆盖
    new_index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "老记忆" not in new_index
    # archive 里有备份
    archives = list((tmp_path / ".archive").glob("legacy-memory-*/MEMORY.md"))
    assert len(archives) >= 1
    assert "老记忆" in archives[0].read_text(encoding="utf-8")


def test_malformed_frontmatter_skipped(tmp_path: Path, caplog):
    """frontmatter 解析失败的文件跳过 + log warning。"""
    store = MemoryStore(harvil_home=tmp_path)
    # 写一个合法的
    good_mid = store.save(name="good", description="d", type="user", body="")
    # 直接写一个坏文件到 .memory/
    bad_file = tmp_path / ".memory" / "bad_mid.md"
    bad_file.write_text(
        "---\ninvalid: yaml: content\n---\nbody",
        encoding="utf-8",
    )
    # 重新加载（模拟下次会话）
    store2 = MemoryStore(harvil_home=tmp_path)
    all_ids = {e.id for e in store2.list_all()}
    assert good_mid in all_ids
    assert "bad_mid" not in all_ids  # 被跳过
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL — `MemoryEntry` 不存在 / `MemoryStore.save` 不存在

- [ ] **Step 4: 写实现**

```python
# agent/memory_store.py （整文件重写）
"""多文件记忆存储：Claude Code 风格。

存储结构：
- ~/.agent/.memory/{ulid}.md：单条记忆，YAML frontmatter + body
- ~/.agent/MEMORY.md：索引（自动生成，每次写后重建）

原则：
- 写入立即落盘 + 重建索引
- snapshot_for_prompt() 返回索引文本，会话内 frozen
- 老格式（无 frontmatter）启动时备份到 .archive/legacy-memory-{ts}/
- 删除软删除到 .archive/memory-{ts}/{id}.md
"""
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

VALID_TYPES = {"user", "feedback", "project", "reference", "other"}


@dataclass
class MemoryEntry:
    """单条记忆。"""
    id: str
    name: str
    description: str
    type: str
    body: str
    created_at: datetime
    updated_at: datetime


def _generate_id() -> str:
    """生成时间排序的唯一 ID（不用 ulid 库，简化为 uuid 拼时间戳）。"""
    ts = int(datetime.now().timestamp() * 1000)
    short_uuid = uuid.uuid4().hex[:6]
    return f"{ts}{short_uuid}"  # 如 "1720870000000a1b2c3"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """解析 `---\\n...yaml...\\n---\\nbody` 格式。返回 (meta, body) 或 (None, text)。"""
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        meta = yaml.safe_load(parts[1])
        if not isinstance(meta, dict):
            return None, text
        return meta, parts[2].lstrip("\n")
    except yaml.YAMLError as e:
        logger.warning("frontmatter 解析失败: %s", e)
        return None, text


def _format_frontmatter(meta: dict) -> str:
    """格式化 dict 为 frontmatter 字符串。"""
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip() + "\n---\n\n"


class MemoryStore:
    """多文件记忆存储。"""

    def __init__(self, *, harvil_home: Path):
        self._home = Path(harvil_home)
        self._memory_dir = self._home / ".memory"
        self._index_path = self._home / "MEMORY.md"
        self._lock = threading.Lock()
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        # 启动时迁移老格式（若存在）
        self._migrate_legacy_if_any()
        # 重建索引（保证一致）
        self._rebuild_index()

    # ---- 内部：写 ----
    def _write_entry_file(self, entry: MemoryEntry) -> None:
        """写单条 .md 文件。"""
        meta = {
            "name": entry.name,
            "description": entry.description,
            "type": entry.type,
            "created_at": entry.created_at.isoformat(timespec="seconds"),
            "updated_at": entry.updated_at.isoformat(timespec="seconds"),
        }
        content = _format_frontmatter(meta) + entry.body
        path = self._memory_dir / f"{entry.id}.md"
        path.write_text(content, encoding="utf-8")

    def _rebuild_index(self) -> None:
        """扫描 .memory/ 重建 MEMORY.md。"""
        lines = ["# Memory Index", ""]
        lines.append("自动生成，请勿手动编辑。每行：`- [name](.memory/{id}.md) — description`")
        lines.append("")
        for entry in self._scan_all_entries():
            lines.append(
                f"- [{entry.name}](.memory/{entry.id}.md) — {entry.description}"
            )
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _scan_all_entries(self) -> List[MemoryEntry]:
        """扫描 .memory/ 下所有 .md，解析为 MemoryEntry。失败的跳过。"""
        entries = []
        for path in sorted(self._memory_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            if meta is None:
                logger.warning("跳过无 frontmatter 的 memory 文件: %s", path)
                continue
            try:
                entry = MemoryEntry(
                    id=path.stem,
                    name=meta.get("name", ""),
                    description=meta.get("description", ""),
                    type=meta.get("type", "other"),
                    body=body,
                    created_at=datetime.fromisoformat(meta.get("created_at", _now_iso())),
                    updated_at=datetime.fromisoformat(meta.get("updated_at", _now_iso())),
                )
                entries.append(entry)
            except (ValueError, TypeError) as e:
                logger.warning("memory 文件字段解析失败 %s: %s", path, e)
        return entries

    def _migrate_legacy_if_any(self) -> None:
        """检测旧格式 MEMORY.md / USER.md，备份到 .archive/legacy-memory-{ts}/。

        判断标准：文件存在 + 内容不以 `---` 开头（无 frontmatter）。
        新建的多文件 MEMORY.md 索引以 `# Memory Index` 开头，不会被误判。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_any = False
        archive_dir = self._home / ".archive" / f"legacy-memory-{ts}"
        for filename in ("MEMORY.md", "USER.md"):
            path = self._home / filename
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            if content.startswith("---") or content.startswith("# Memory Index"):
                continue  # 新格式，不动
            # 老格式 → 备份
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive_dir / filename)
            logger.info("旧 %s 已备份到 %s/", filename, archive_dir)
            path.unlink()
            archived_any = True
        if archived_any:
            logger.info("老格式记忆文件已迁移。新格式 MEMORY.md 索引将在下一步重建。")

    # ---- 公开：读 ----
    def list_all(self) -> List[MemoryEntry]:
        with self._lock:
            return self._scan_all_entries()

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            path = self._memory_dir / f"{memory_id}.md"
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            if meta is None:
                return None
            return MemoryEntry(
                id=memory_id,
                name=meta.get("name", ""),
                description=meta.get("description", ""),
                type=meta.get("type", "other"),
                body=body,
                created_at=datetime.fromisoformat(meta.get("created_at", _now_iso())),
                updated_at=datetime.fromisoformat(meta.get("updated_at", _now_iso())),
            )

    def load_body(self, memory_id: str) -> Optional[str]:
        entry = self.get(memory_id)
        return entry.body if entry else None

    def snapshot_for_prompt(self) -> str:
        """返回索引文本（frozen，会话内不变）。"""
        # 读当前 MEMORY.md 内容（仅索引行，跳过头部 3 行）
        if not self._index_path.exists():
            return ""
        text = self._index_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # 跳过前 3 行（标题 + 空 + 说明）和第 4 行空行
        return "\n".join(lines[4:]) if len(lines) > 4 else ""

    def build_index_text(self) -> str:
        """重建并返回索引（启动时用）。"""
        with self._lock:
            self._rebuild_index()
        return self.snapshot_for_prompt()

    # ---- 公开：写 ----
    def save(
        self,
        *,
        name: str,
        description: str,
        type: str,
        body: str = "",
    ) -> str:
        """创建新记忆。返回 memory_id。"""
        if not name or not description:
            raise ValueError("name 和 description 必需")
        if type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一，实际: {type}")
        with self._lock:
            now = datetime.now()
            mid = _generate_id()
            entry = MemoryEntry(
                id=mid, name=name, description=description,
                type=type, body=body,
                created_at=now, updated_at=now,
            )
            self._write_entry_file(entry)
            self._rebuild_index()
        return mid

    def update(
        self,
        memory_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
    ) -> MemoryEntry:
        """更新字段。不存在的 id 抛 KeyError。"""
        if type is not None and type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一")
        with self._lock:
            entry = self.get(memory_id)
            if entry is None:
                raise KeyError(f"memory not found: {memory_id}")
            if name is not None:
                entry.name = name
            if description is not None:
                entry.description = description
            if type is not None:
                entry.type = type
            if body is not None:
                entry.body = body
            entry.updated_at = datetime.now()
            self._write_entry_file(entry)
            self._rebuild_index()
        return entry

    def delete(self, memory_id: str) -> bool:
        """软删除：移到 .archive/memory-{ts}/{id}.md。"""
        with self._lock:
            path = self._memory_dir / f"{memory_id}.md"
            if not path.exists():
                return False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self._home / ".archive" / f"memory-{ts}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive_dir / path.name))
            self._rebuild_index()
        return True

    # ---- 兼容旧接口（被 prompt_builder 等调用，留 stub 避免破坏） ----
    def format_for_system_prompt(self, target: str) -> str:
        """旧接口兼容：返回索引（忽略 target 参数）。"""
        return self.snapshot_for_prompt()

    def add(self, target: str, content: str) -> bool:
        """旧接口兼容：等价于 save（target 当 type 用）。"""
        try:
            t = target if target in VALID_TYPES else "other"
            self.save(name=content[:30], description=content, type=t, body=content)
            return True
        except Exception as e:
            logger.warning("旧 add() 兼容失败: %s", e)
            return False

    def modify(self, action: str, target: str, content: str, old_content: str = "") -> bool:
        """旧接口兼容：粗略映射到 save（行为不完全等价）。"""
        if action == "add":
            return self.add(target, content)
        # replace / remove 在旧接口下行为模糊，旧数据已弃用，直接返 False 提示用户用新工具
        logger.warning("旧 modify(action=%s) 不再支持，请用 memory 工具新 action", action)
        return False
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_memory.py -v`
Expected: PASS（15 tests）

- [ ] **Step 6: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 507 减去被替换的旧 memory 测试 + 15 新 = 取决于旧测试数；总之无回归）

- [ ] **Step 7: 不 commit**

---

## Task 2: agent/memory_retriever.py

**Files:**
- Create: `agent/memory_retriever.py`
- Create: `tests/test_memory_retriever.py`

**Interfaces:**
- Consumes: LLM client（OpenAI 兼容）
- Produces: `retrieve_relevant(query, index_text, *, llm_client, model, max_results=5) -> list[str]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_memory_retriever.py
"""memory_retriever 测试。"""
import json
from unittest.mock import MagicMock

import pytest

from agent.memory_retriever import retrieve_relevant


def _mock_llm(returned_text: str):
    m = MagicMock()
    m.chat_completions.return_value.choices = [
        MagicMock(message=MagicMock(content=returned_text))
    ]
    return m


def test_retrieve_relevant_returns_ids():
    llm = _mock_llm('["id1", "id2"]')
    result = retrieve_relevant(
        query="如何配置 pytest",
        index_text="- [pytest](.memory/id1.md) — pytest 配置\n- [git](.memory/id2.md) — git 用法",
        llm_client=llm, model="test-model",
    )
    assert result == ["id1", "id2"]


def test_retrieve_relevant_handles_empty_index():
    llm = _mock_llm('[]')
    result = retrieve_relevant(
        query="x", index_text="",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_handles_llm_failure_returns_empty():
    """LLM 抛异常 → 返回空 list（fail-open）。"""
    llm = MagicMock()
    llm.chat_completions.side_effect = RuntimeError("API down")
    result = retrieve_relevant(
        query="x", index_text="some index",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_handles_malformed_json_returns_empty():
    llm = _mock_llm("not a json")
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_respects_max_results():
    llm = _mock_llm('["a", "b", "c", "d", "e", "f", "g"]')
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m", max_results=3,
    )
    assert len(result) == 3


def test_retrieve_relevant_handles_non_list_json():
    llm = _mock_llm('{"not": "a list"}')
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m",
    )
    assert result == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_memory_retriever.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 写实现**

```python
# agent/memory_retriever.py
"""相关记忆检索器：每轮主 LLM 调用前调一次。

输入：当前 user message + memory 索引
输出：top-N 最相关的 memory_id 列表

失败 fail-open：任何异常（LLM 超时/坏 JSON/空响应）返回空 list。
"""
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


RETRIEVAL_PROMPT_TEMPLATE = """你是记忆检索助手。当前用户消息：

<query>
{query}
</query>

可用记忆索引（每行一条）：

<index>
{index_text}
</index>

返回最多 {max_results} 条与当前 query 最相关的记忆 ID（从索引的 .memory/{id}.md 路径中提取 {id} 部分）。
格式：JSON 数组，元素是 ID 字符串。例如：["1720870000000a1b2c3", "1720870000000d4e5f6"]
只返回 JSON 数组，不要其他文本。若无相关的，返回 []。
"""


def retrieve_relevant(
    *,
    query: str,
    index_text: str,
    llm_client,
    model: str,
    max_results: int = 5,
) -> List[str]:
    """调 LLM 选 top-N 相关 memory_id。失败返回 []。"""
    if not query.strip() or not index_text.strip():
        return []

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        query=query[:1000],  # 防止 query 太长
        index_text=index_text[:5000],  # 防止 index 太长
        max_results=max_results,
    )

    try:
        response = llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("memory retrieval LLM 调用失败（fail-open）: %s", e)
        return []

    # 提取 JSON 数组（容忍模型输出多余文本）
    try:
        # 尝试直接 parse
        result = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 [ ... ] 子串
        import re
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not match:
            logger.warning("memory retrieval 输出非合法 JSON: %s", content[:200])
            return []
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(result, list):
        return []
    # 只保留字符串元素 + 截断到 max_results
    return [str(x) for x in result if isinstance(x, str)][:max_results]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_memory_retriever.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 不 commit**

---

## Task 3: tools/memory_tool.py 重写（5 action）

**Files:**
- Modify: `tools/memory_tool.py`（整文件重写）

**Interfaces:**
- Consumes: `MemoryStore`（T1）
- Produces: 5 个 action handler 在同一函数内分支

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_memory.py（或新建 tests/test_memory_tool.py）
import json
from pathlib import Path
from tools.memory_tool import _handle_memory
from agent.memory_store import MemoryStore


def _run(action, store, **kwargs):
    args = {"action": action, **kwargs}
    result_str = _handle_memory(args, memory_store=store)
    return json.loads(result_str)


def test_memory_tool_save(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("save", store, name="t1", description="d",
                   type="user", body="b")
    assert parsed["success"] is True
    assert "id" in parsed


def test_memory_tool_list(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    store.save(name="t1", description="d", type="user", body="")
    parsed = _run("list", store)
    assert parsed["success"] is True
    assert parsed["count"] == 1


def test_memory_tool_load(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="full body")
    parsed = _run("load", store, id=mid)
    assert parsed["success"] is True
    assert parsed["body"] == "full body"


def test_memory_tool_update(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="b1")
    parsed = _run("update", store, id=mid, body="b2")
    assert parsed["success"] is True
    assert store.get(mid).body == "b2"


def test_memory_tool_delete(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(name="t", description="d", type="user", body="")
    parsed = _run("delete", store, id=mid)
    assert parsed["success"] is True


def test_memory_tool_load_unknown(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("load", store, id="nonexistent")
    assert parsed["success"] is False
    assert "error" in parsed


def test_memory_tool_save_invalid_type(tmp_path: Path):
    store = MemoryStore(harvil_home=tmp_path)
    parsed = _run("save", store, name="t", description="d",
                   type="invalid_kind", body="")
    assert parsed["success"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_memory.py -k memory_tool -v`
Expected: FAIL（旧 _handle_memory 不接 5 action）

- [ ] **Step 3: 写实现**

```python
# tools/memory_tool.py （整文件重写）
"""记忆工具：管理持久化多文件记忆。

action:
  - save: 创建新记忆（必需 name/description/type）
  - update: 更新已有记忆字段
  - delete: 软删除（移到 .archive/）
  - load: 读 body
  - list: 列出所有记忆
"""
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "管理持久化记忆（跨会话保存）。每条记忆是一个独立文件，含 frontmatter + body。\n"
        "写入立即落盘，但索引下次会话才注入到 system prompt（保护 prompt cache）。\n\n"
        "action:\n"
        "  - save: 创建新记忆（必需 name/description/type）\n"
        "  - update: 更新字段（必需 id）\n"
        "  - delete: 软删除（必需 id）\n"
        "  - load: 读完整 body（必需 id）\n"
        "  - list: 列出所有记忆\n\n"
        "type 可选值: user / feedback / project / reference / other"
    ),
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


def _handle_memory(args: dict, **kwargs) -> str:
    action = args.get("action")
    store = kwargs.get("memory_store")

    if store is None:
        return json.dumps({
            "success": False, "error": "记忆系统未初始化",
        }, ensure_ascii=False)

    try:
        if action == "save":
            mid = store.save(
                name=args.get("name", ""),
                description=args.get("description", ""),
                type=args.get("type", "other"),
                body=args.get("body", ""),
            )
            return json.dumps({
                "success": True, "action": "save", "id": mid,
                "message": "已保存（索引下次会话生效）",
            }, ensure_ascii=False)

        if action == "update":
            mid = args.get("id", "")
            entry = store.update(
                mid,
                name=args.get("name"),
                description=args.get("description"),
                type=args.get("type"),
                body=args.get("body"),
            )
            return json.dumps({
                "success": True, "action": "update", "id": mid,
                "entry": {
                    "name": entry.name, "description": entry.description,
                    "type": entry.type,
                },
            }, ensure_ascii=False)

        if action == "delete":
            mid = args.get("id", "")
            ok = store.delete(mid)
            if not ok:
                return json.dumps({
                    "success": False, "error": f"未找到: {mid}",
                }, ensure_ascii=False)
            return json.dumps({
                "success": True, "action": "delete", "id": mid,
                "message": "已软删除到 .archive/",
            }, ensure_ascii=False)

        if action == "load":
            mid = args.get("id", "")
            entry = store.get(mid)
            if entry is None:
                return json.dumps({
                    "success": False, "error": f"未找到: {mid}",
                }, ensure_ascii=False)
            return json.dumps({
                "success": True, "id": mid,
                "name": entry.name, "description": entry.description,
                "type": entry.type, "body": entry.body,
                "created_at": entry.created_at.isoformat(timespec="seconds"),
                "updated_at": entry.updated_at.isoformat(timespec="seconds"),
            }, ensure_ascii=False)

        if action == "list":
            entries = store.list_all()
            summaries = [
                {"id": e.id, "name": e.name, "description": e.description, "type": e.type}
                for e in entries
            ]
            return json.dumps({
                "success": True, "count": len(entries), "memories": summaries,
            }, ensure_ascii=False)

        return json.dumps({
            "success": False, "error": f"未知 action: {action}",
        }, ensure_ascii=False)

    except (ValueError, KeyError) as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.exception("memory 工具异常")
        return json.dumps({
            "success": False, "error": f"内部错误: {e}",
        }, ensure_ascii=False)


registry.register(
    name="memory",
    toolset="core",
    schema=MEMORY_SCHEMA,
    handler=_handle_memory,
    emoji="🧠",
)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_memory.py -v`
Expected: PASS（含 T1 + T3 测试）

- [ ] **Step 5: 不 commit**

---

## Task 4: prompt_builder.py 改用索引

**Files:**
- Modify: `agent/prompt_builder.py:build_system_prompt`

**Interfaces:**
- Consumes: T1 `MemoryStore.snapshot_for_prompt()`
- Produces: system prompt 包含 `## 记忆索引` 段

- [ ] **Step 1: 改 prompt_builder.py**

找到 `build_system_prompt` 中现有「5. 记忆快照（frozen）」段，替换为：

```python
    # 5. 记忆索引（多文件模式，Phase 5）
    if memory_store:
        try:
            index_block = memory_store.snapshot_for_prompt()
            if index_block:
                parts.append(f"## 记忆索引\n{index_block}")
        except Exception as e:
            logger.warning("读取记忆索引失败: %s", e)
```

注意：移除原 `format_for_system_prompt("memory")` 和 `format_for_system_prompt("user")` 双调用（旧 API 已合并为单 `snapshot_for_prompt`）。`MemoryStore` 留了 stub `format_for_system_prompt` 但不再被调用。

- [ ] **Step 2: 跑回归**

Run: `uv run pytest tests/test_context.py -v`
Expected: PASS（如果 test_build_system_prompt_with_memory 失败，更新断言为新格式）

Run: `uv run pytest tests/ -v`
Expected: PASS（取決于 test_context 是否硬编码旧格式）

- [ ] **Step 3: 不 commit**

---

## Task 5: agent/__init__.py 主循环 retriever 注入

**Files:**
- Modify: `agent/__init__.py:AIAgent.__init__` + `run_conversation`
- Modify: `tests/test_integration.py`

**Interfaces:**
- Consumes: T2 retriever + T1 store
- Produces: AIAgent `memory_retriever=None` kwarg + `_cached_memory_index`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py
def test_aiagent_accepts_memory_retriever_kwarg():
    agent = _make_test_agent()
    assert agent.memory_retriever is None


def test_aiagent_injects_relevant_memories_into_user_msg(tmp_path):
    """retriever 返回 id → store 读 body → 注入 <relevant_memories>。"""
    from unittest.mock import MagicMock, patch
    from agent import AIAgent
    from agent.memory_store import MemoryStore

    store = MemoryStore(harvil_home=tmp_path)
    mid = store.save(
        name="pytest 配置",
        description="项目用 pytest",
        type="project",
        body="运行测试用 uv run pytest tests/ -v",
    )

    # mock retriever 返回 [mid]
    fake_retriever = MagicMock()
    fake_retriever.retrieve_relevant.return_value = [mid]

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        memory_store=store, memory_retriever=fake_retriever,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("怎么跑测试")

    # conversation_history[0] 应含 <relevant_memories> + mid body
    first_user = agent.conversation_history[0]["content"]
    assert "<relevant_memories>" in first_user
    assert "uv run pytest" in first_user
    # 原始 user message 也应在
    assert "怎么跑测试" in first_user


def test_aiagent_no_memory_retriever_backward_compat(tmp_path):
    """memory_retriever=None 时不抛，user_message 原样入 history。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


def test_retrieval_failure_does_not_break_main_loop(tmp_path):
    """retriever 抛异常时主循环不崩。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.memory_store import MemoryStore

    store = MemoryStore(harvil_home=tmp_path)
    bad_retriever = MagicMock()
    bad_retriever.retrieve_relevant.side_effect = RuntimeError("boom")

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        memory_store=store, memory_retriever=bad_retriever,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    agent.run_conversation("hi")
    # 不抛 + user_message 原样入 history
    assert agent.conversation_history[0]["content"] == "hi"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_aiagent_accepts_memory_retriever_kwarg -v`
Expected: FAIL — unexpected kwarg

- [ ] **Step 3: 改 agent/__init__.py**

`AIAgent.__init__` 加 `memory_retriever=None` + 初始化 `_cached_memory_index`：

```python
    def __init__(
        self,
        *,
        # ... 原有参数 ...
        cron_scheduler=None,
        memory_retriever=None,  # === NEW ===
    ):
        # ...
        self.memory_retriever = memory_retriever
        # 缓存 memory 索引（会话内 frozen，保护 prompt cache）
        self._cached_memory_index = ""
        if self.memory_store:
            try:
                self._cached_memory_index = self.memory_store.snapshot_for_prompt()
            except Exception as e:
                logger.warning("缓存 memory 索引失败: %s", e)
```

`run_conversation` 顶部（USER_PROMPT_SUBMIT hook 之后、append user 到 history 之前）插入：

```python
        # === NEW: memory 检索 + 注入 ===
        relevant_memories_text = ""
        if (self.memory_retriever and self.memory_store
                and self._cached_memory_index):
            try:
                relevant_ids = self.memory_retriever.retrieve_relevant(
                    query=user_message,
                    index_text=self._cached_memory_index,
                    llm_client=self.llm_client,
                    model=self.model,
                    max_results=5,
                )
                if relevant_ids:
                    bodies = []
                    for mid in relevant_ids:
                        body = self.memory_store.load_body(mid)
                        if body:
                            bodies.append(f"[memory:{mid}]\n{body}")
                    if bodies:
                        relevant_memories_text = "\n\n".join(bodies)
            except Exception as e:
                logger.warning("memory retrieval 失败（fail-open）: %s", e)
                relevant_memories_text = ""

        # 组装实际入 history 的 user_content
        if relevant_memories_text:
            user_message_for_history = (
                f"<relevant_memories>\n{relevant_memories_text}\n</relevant_memories>\n\n"
                f"{user_message}"
            )
        else:
            user_message_for_history = user_message

        self.conversation_history.append({
            "role": "user", "content": user_message_for_history,
        })
```

注意：把原本的 `self.conversation_history.append({"role":"user","content":user_message})` 替换为上述。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py -k memory_retriever -v`
Expected: PASS（4 个新测试）

- [ ] **Step 5: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: 不 commit**

---

## Task 6: config.py memory 块新增字段

**Files:**
- Modify: `config.py:DEFAULT_CONFIG["memory"]`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py
def test_default_config_memory_multifile_fields():
    from config import DEFAULT_CONFIG
    m = DEFAULT_CONFIG["memory"]
    for key in ("multifile_enabled", "memory_dir", "retrieval_enabled",
                "retrieval_max_results", "retrieval_model"):
        assert key in m, f"缺 {key}"


def test_default_config_memory_multifile_defaults():
    from config import DEFAULT_CONFIG
    m = DEFAULT_CONFIG["memory"]
    assert m["multifile_enabled"] is True
    assert m["memory_dir"] is None
    assert m["retrieval_enabled"] is True
    assert m["retrieval_max_results"] == 5
    assert m["retrieval_model"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_default_config_memory_multifile_fields -v`
Expected: FAIL

- [ ] **Step 3: 改 config.py**

在 `DEFAULT_CONFIG["memory"]` 块加新字段：

```python
    "memory": {
        "enabled": True,
        "provider": None,
        # 新增（Phase 5 多文件模式）
        "multifile_enabled": True,                # False 时回退（v1 总是 True）
        "memory_dir": None,                       # 默认 ~/.agent/.memory/
        "retrieval_enabled": True,                # False 时跳过每轮检索
        "retrieval_max_results": 5,
        "retrieval_model": None,                  # None → 用主 model
        # 沿用（旧模式 / 历史兼容）
        "memory_char_limit": 2200,
        "user_char_limit": 1375,
    },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Task 7: cli.py RuntimeContext 装配 retriever

**Files:**
- Modify: `cli.py:RuntimeContext.__init__` + `_create_agent`

- [ ] **Step 1: 改 cli.py**

在 `RuntimeContext.__init__` 加（紧邻 memory_store 实例化之后）：

```python
        # === NEW: memory retriever（多文件模式） ===
        self.memory_retriever = None
        mem_cfg = self.config.get("memory", {})
        if (mem_cfg.get("enabled", True) and
                mem_cfg.get("retrieval_enabled", True) and
                mem_cfg.get("multifile_enabled", True)):
            from agent.memory_retriever import retrieve_relevant
            # 用 functools.partial 包装成可调用对象
            import functools
            retrieval_model = mem_cfg.get("retrieval_model") or self.config.get("model", {}).get("name")
            self.memory_retriever = functools.partial(
                retrieve_relevant, model=retrieval_model,
            )
```

⚠️ 实际 retriever 接收 llm_client 参数；如果 retriever 是 partial(model=...)，则 caller 还要传 llm_client。或者把 retriever 包装为类。**简化**：直接传函数引用 `retrieve_relevant`，AIAgent 在调用时传 model：

```python
self.memory_retriever = retrieve_relevant  # 直接传函数引用
```

然后在 `_create_agent` 传 `memory_retriever=self.memory_retriever`：

```python
agent = AIAgent(
    # ... 原有 ...
    memory_retriever=self.memory_retriever,
)
```

- [ ] **Step 2: 跑回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: 不 commit**

---

## Task 8: e2e + 旧测试清理

**Files:**
- Modify: `tests/test_integration.py`
- Modify: 可能要改 `tests/test_context.py`（旧 memory 测试）

- [ ] **Step 1: 加 e2e 测试**

```python
# 追加到 tests/test_integration.py
def test_e2e_memory_save_then_retrieve_next_session(tmp_path):
    """端到端：会话 1 save → 会话 2 检索 + 注入。

    模拟两个会话（两次 AIAgent 实例化）。
    """
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.memory_store import MemoryStore
    from agent.memory_retriever import retrieve_relevant

    # === 会话 1：保存记忆 ===
    store1 = MemoryStore(harvil_home=tmp_path)
    mid = store1.save(
        name="pytest 命令",
        description="项目用 pytest 跑测试",
        type="project",
        body="uv run pytest tests/ -v",
    )

    # === 会话 2：索引应能看到，retriever 应能选到 ===
    store2 = MemoryStore(harvil_home=tmp_path)  # 重建索引
    # mock LLM：retriever 调用返回 [mid]，主 LLM 返回 ok
    main_llm = MagicMock()
    call_count = [0]
    def side_effect(msgs, **kw):
        call_count[0] += 1
        # 第 1 次调用是 retriever（query+index）
        if call_count[0] == 1:
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content=f'["{mid}"]'))]
            return resp
        # 主 LLM 调用
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
        resp.finish_reason = "stop"
        return resp
    main_llm.chat_completions.side_effect = side_effect

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], harvil_home=str(tmp_path),
        memory_store=store2, memory_retriever=retrieve_relevant,
    )
    agent.llm_client = main_llm
    agent.run_conversation("怎么跑测试")

    # 第一次入 history 的 user 消息应含 <relevant_memories> + memory body
    first_user = agent.conversation_history[0]["content"]
    assert "<relevant_memories>" in first_user
    assert "uv run pytest" in first_user
    assert "怎么跑测试" in first_user
```

- [ ] **Step 2: 跑 e2e**

Run: `uv run pytest tests/test_integration.py::test_e2e_memory_save_then_retrieve_next_session -v`
Expected: PASS

- [ ] **Step 3: 修旧测试**

如果 `tests/test_context.py` 或其他测试断言旧 `format_for_system_prompt` 输出格式，更新断言或删除该测试。Run: `uv run pytest tests/ -v` 看哪里挂了。

- [ ] **Step 4: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: 不 commit**

---

## Self-Review

**Spec 覆盖**：
- ✅ §1 架构 → T1-T7
- ✅ §2 文件格式（frontmatter + ULID 文件名 + 索引） → T1
- ✅ §3 数据结构（MemoryEntry + MemoryStore） → T1
- ✅ §4 retriever + prompt 模板 → T2
- ✅ §5 主循环注入 `<relevant_memories>` → T5
- ✅ §6 工具 5-action → T3
- ✅ §7 prompt_builder 改 → T4
- ✅ §8 config 新字段 → T6
- ✅ §8 cli.py 装配 → T7
- ✅ §9 失败处理（retriever 异常 / 索引不存在 / frontmatter 解析失败） → T1+T2+T5 测试覆盖
- ✅ §10 测试矩阵 → T1 (15) + T2 (6) + T3 (7) + T5 (4) + T8 (1)
- ✅ §11 已知限制 → 全遵守

**Placeholder 扫描**：无 TBD/TODO。

**类型一致性**：
- `MemoryEntry` 字段 T1 定义，T2/T3 引用 ✓
- `MemoryStore.save/update/delete/load_body/get/list_all/snapshot_for_prompt` 签名 T1 → T3/T5 一致 ✓
- `retrieve_relevant(query, index_text, *, llm_client, model, max_results)` T2 定义，T5 调用 ✓
- `_cached_memory_index` 属性 T5 init 设置，主循环读 ✓

---

## Execution Handoff

按用户授权直接进 SDD。
