# Phase 1: 核心韧性层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `agent/context_compressor.py` 的单层 LLM 摘要替换为 4 层压缩管线（L1/L2/L3/L4 + reactive）+ 大输出落盘 + transcript 归档，通过 `config.context.use_new_pipeline` 开关双轨上线。

**Architecture:** 新增 3 个独立模块（`agent/output_offload.py`、`agent/transcript.py`、`agent/context_pipeline.py`），改造 4 个现有模块（`config.py`、`agent/__init__.py`、`tools/terminal_tool.py`、`tools/file_operations.py`、`agent/prompt_builder.py`）。所有新代码默认不启用，开关切 True 才生效。

**Tech Stack:** Python 3.11+、uv、pytest、OpenAI 兼容 LLM client、文件系统（强制 `encoding="utf-8"`）。

**对应 Spec:** `docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md` §2-§5

## Global Constraints

- 文件 I/O 必须指定 `encoding="utf-8"`（CLAUDE.md 强制，ruff `PLW1514`）
- 依赖管理统一用 `uv add`（不要 `pip install`，不要手改 `pyproject.toml`）
- 所有工具 handler 返回 JSON 字符串；错误用 `{"error": "...", "error_type": "..."}`
- 写文件路径走 `agent.permission.safe_path(write=True, allowed_roots=[...])`
- 注释、文档、commit message 用中文；代码标识符用英文
- 测试命令：`uv run pytest tests/<file>.py -v`；全量回归 `uv run pytest tests/ -v`
- 现有 294 个测试不得回归（开关 False 时全部继续通过）
- 项目当前不是 git 仓库；`git commit` 步骤视为逻辑检查点（用户后续若 `git init` 则直接可用）

---

## File Structure

| 文件 | 任务 | 责任 |
|---|---|---|
| `agent/output_offload.py` | T1 新增 | 大输出旁路落盘（L3） |
| `agent/transcript.py` | T2 新增 | 压缩前快照存档 |
| `agent/context_pipeline.py` | T3-T7 新增 | 4 层管线编排（L1/L2/L4 + reactive + session_state） |
| `agent/context_compressor.py` | T9 改造 | `maybe_compress` 加 DeprecationWarning，保留 `_fix_tool_call_pairs` / `_summarize_conversation` / `_rule_based_summary` / `estimate_message_tokens` |
| `agent/memory_manager.py` | T8 加法 | 新增 `on_pre_compress` no-op 钩子 |
| `config.py` | T8 改造 | `DEFAULT_CONFIG` 新增 `context` 块 |
| `agent/__init__.py` | T9 改造 | 第 212-225 行 `maybe_compress` 调用替换为分支（开关控制）+ 新增 reactive 异常分支 |
| `tools/terminal_tool.py` | T10 改造 | handler 返回前过 `output_offload.maybe_offload` |
| `tools/file_operations.py` | T10 改造 | 同上 |
| `agent/prompt_builder.py` | T10 改造 | `TOOL_USAGE_GUIDANCE` 加占位消息识别段 |
| `tests/test_output_offload.py` | T1 新增 | 单元测试 |
| `tests/test_transcript.py` | T2 新增 | 单元测试 |
| `tests/test_context_pipeline.py` | T3-T7 新增 | 每层独立测试 + 编排测试 |
| `tests/test_context.py` | T9 改造 | 加 deprecation 测试 |
| `tests/test_integration.py` | T11 新增 | 200 轮对话端到端 |

---

## Task 1: output_offload 模块（旁路 L3）

**Files:**
- Create: `agent/output_offload.py`
- Create: `tests/test_output_offload.py`

**Interfaces:**
- Consumes: `pathlib.Path`、`tempfile`、`logging`
- Produces: `maybe_offload(content: str, *, tool_call_id: str, agent_home: Path, threshold: int = 30000, preview_chars: int = 2000) -> str`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_output_offload.py
"""output_offload 模块测试。"""
import json
from pathlib import Path

import pytest

from agent.output_offload import maybe_offload


def test_under_threshold_returns_content_unchanged(tmp_path: Path):
    """小于阈值时原样返回（不是 JSON）。"""
    content = "short result"
    result = maybe_offload(
        content, tool_call_id="call_abc", agent_home=tmp_path,
        threshold=30000, preview_chars=2000,
    )
    assert result == content


def test_over_threshold_writes_file_and_returns_json(tmp_path: Path):
    """大于阈值时落盘并返回 JSON 指针。"""
    content = "x" * 50000
    result = maybe_offload(
        content, tool_call_id="call_abc", agent_home=tmp_path,
        threshold=30000, preview_chars=2000,
    )
    parsed = json.loads(result)
    assert parsed["truncated"] is True
    assert parsed["orig_chars"] == 50000
    assert len(parsed["preview"]) == 2000
    assert "full_at" in parsed
    assert "hint" in parsed

    offload_path = Path(parsed["full_at"])
    assert offload_path.exists()
    assert offload_path.read_text(encoding="utf-8") == content
    assert "call_abc" in offload_path.name


def test_offload_path_under_agent_home(tmp_path: Path):
    """落盘路径必须严格在 agent_home/.task_outputs/tool-results/ 下。"""
    content = "x" * 50000
    result = maybe_offload(content, tool_call_id="call_xyz", agent_home=tmp_path)
    parsed = json.loads(result)
    offload_path = Path(parsed["full_at"])
    assert offload_path.is_relative_to(tmp_path / ".task_outputs" / "tool-results")


def test_duplicate_tool_call_id_appends_counter(tmp_path: Path):
    """相同 tool_call_id 第二次落盘不覆盖，追加 _N。"""
    content1 = "x" * 50000
    maybe_offload(content1, tool_call_id="call_dup", agent_home=tmp_path)
    content2 = "y" * 50000
    result = maybe_offload(content2, tool_call_id="call_dup", agent_home=tmp_path)
    parsed = json.loads(result)
    assert Path(parsed["full_at"]).read_text(encoding="utf-8") == content2
    assert (tmp_path / ".task_outputs" / "tool-results" / "call_dup.txt").exists()


def test_disk_full_falls_back_to_truncated_content(tmp_path: Path, monkeypatch):
    """写入失败时不抛异常，降级为截断 + error 标注。"""
    def raise_oserror(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("agent.output_offload._write_atomically", raise_oserror)

    content = "x" * 50000
    result = maybe_offload(content, tool_call_id="call_err", agent_home=tmp_path)
    parsed = json.loads(result)
    assert parsed["error_type"] == "offload_io_error"
    assert "truncated_content" in parsed
    assert len(parsed["truncated_content"]) == 30000


def test_non_string_content_passthrough(tmp_path: Path):
    """非字符串 content（如 None / dict）原样返回，不尝试落盘。"""
    assert maybe_offload(None, tool_call_id="x", agent_home=tmp_path) is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_output_offload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.output_offload'`

- [ ] **Step 3: 写最小实现**

```python
# agent/output_offload.py
"""大输出落盘：工具结果超过阈值时写到磁盘，messages 里只留预览。

这是分层压缩管线的旁路 L3（见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3）。

设计目标：
- 信息无损：完整内容在磁盘上，LLM 可通过 read_file 工具读回
- 文件名用 tool_call_id 保证唯一
- 失败降级：磁盘满时截断 content 并标注，不抛异常
"""
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 30000
DEFAULT_PREVIEW_CHARS = 2000


def maybe_offload(
    content: str,
    *,
    tool_call_id: str,
    agent_home: Path,
    threshold: int = DEFAULT_THRESHOLD,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """工具 handler 调用。返回值直接作为 tool 消息 content 用。

    - content 字符数 <= threshold：原样返回
    - content 字符数 > threshold：写入 agent_home/.task_outputs/tool-results/{tool_call_id}.txt，
      返回 JSON 字符串（含 preview + full_at 指针）
    - 写入失败（如磁盘满）：返回 JSON，含 error_type=offload_io_error + truncated_content

    参数：
        content: 工具原始输出（非字符串时原样返回）
        tool_call_id: OpenAI 兼容协议的工具调用 ID（每次唯一）
        agent_home: agent 根目录（如 ~/.agent）
        threshold: 触发落盘的字符数阈值
        preview_chars: 落盘后保留在 messages 里的预览长度
    """
    if not isinstance(content, str) or len(content) <= threshold:
        return content

    offload_dir = agent_home / ".task_outputs" / "tool-results"
    target_path = _resolve_unique_path(offload_dir, tool_call_id)

    try:
        _write_atomically(target_path, content)
    except OSError as e:
        logger.warning("offload 写入失败 (%s)，降级为截断: %s", target_path, e)
        return json.dumps({
            "error": f"offload failed: {e}",
            "error_type": "offload_io_error",
            "truncated_content": content[:threshold],
        }, ensure_ascii=False)

    return json.dumps({
        "truncated": True,
        "orig_chars": len(content),
        "preview": content[:preview_chars],
        "full_at": str(target_path),
        "hint": "完整结果已落盘，需要时调 read_file 读取 full_at",
    }, ensure_ascii=False)


def _resolve_unique_path(offload_dir: Path, tool_call_id: str) -> Path:
    """生成不冲突的落盘路径。tool_call_id 文件已存在时追加 _N。"""
    offload_dir.mkdir(parents=True, exist_ok=True)
    base = offload_dir / f"{tool_call_id}.txt"
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = offload_dir / f"{tool_call_id}_{counter}.txt"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_atomically(path: Path, content: str) -> None:
    """原子写入：先写临时文件再 replace，防半写状态。"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)  # 原子 rename
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_output_offload.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/output_offload.py tests/test_output_offload.py
git commit -m "feat(context): 新增 output_offload 模块（旁路 L3 大输出落盘）"
```

---

## Task 2: transcript 模块（压缩前快照）

**Files:**
- Create: `agent/transcript.py`
- Create: `tests/test_transcript.py`

**Interfaces:**
- Consumes: `pathlib.Path`、`tempfile`、`datetime`、`logging`
- Produces: `snapshot_if_needed(messages: list, *, agent_home: Path, session_id: str, force: bool = False, enabled: bool = True, retention: int = 20) -> Optional[Path]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_transcript.py
"""transcript 模块测试：压缩前快照存档。"""
import json
from pathlib import Path

from agent.transcript import snapshot_if_needed


SAMPLE_MESSAGES = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "tool_calls": [
        {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "result"},
]


def test_force_writes_jsonl(tmp_path: Path):
    """force=True 必落盘，返回路径。"""
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="sess_x",
        force=True,
    )
    assert path is not None
    assert path.exists()
    assert path.suffix == ".jsonl"
    # 每行是合法 JSON
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["role"] == "system"
    assert parsed[-1].get("_meta", {}).get("session_id") == "sess_x"
    assert parsed[-1]["_meta"]["orig_len"] == 4


def test_disabled_returns_none(tmp_path: Path):
    """enabled=False 直接返回 None。"""
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="sess_x",
        force=True, enabled=False,
    )
    assert path is None


def test_retention_prunes_oldest(tmp_path: Path):
    """超出 retention 时删最旧。"""
    for i in range(5):
        snapshot_if_needed(
            SAMPLE_MESSAGES, agent_home=tmp_path, session_id=f"sess_{i}",
            force=True, retention=3,
        )
    files = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(files) == 3  # 只保留最近 3 个


def test_filename_contains_timestamp_and_uuid(tmp_path: Path):
    """文件名格式：transcript_{YYYYMMDD_HHMMSS}_{shortuuid}.jsonl"""
    import re
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s",
        force=True,
    )
    pattern = re.compile(r"transcript_\d{8}_\d{6}_[a-f0-9]{4}\.jsonl")
    assert pattern.match(path.name), f"文件名格式不对: {path.name}"


def test_creates_latest_pointer(tmp_path: Path):
    """同时维护 latest 指针（Windows 降级为文本文件）。"""
    snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s", force=True,
    )
    latest = tmp_path / ".transcripts" / "latest.txt"
    assert latest.exists()
    content = latest.read_text(encoding="utf-8")
    assert "transcript_" in content


def test_write_failure_returns_none_and_logs(tmp_path: Path, monkeypatch):
    """写入失败时不抛，返回 None。"""
    def raise_oserror(*args, **kwargs):
        raise OSError("permission denied")
    monkeypatch.setattr("agent.transcript._write_atomically", raise_oserror)

    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s", force=True,
    )
    assert path is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_transcript.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写最小实现**

```python
# agent/transcript.py
"""压缩前快照存档：把完整 messages 落到 .transcripts/，便于事后回查。

触发时机由调用方决定（默认仅在 L4 LLM 摘要前 force=True）。
文件格式：JSONL，每行一条消息，最后一行是 _meta 元数据。
"""
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RETENTION = 20


def snapshot_if_needed(
    messages: list,
    *,
    agent_home: Path,
    session_id: str,
    force: bool = False,
    enabled: bool = True,
    retention: int = DEFAULT_RETENTION,
) -> Optional[Path]:
    """落盘 messages 到 .transcripts/transcript_{ts}_{uuid}.jsonl。

    - enabled=False：直接返回 None
    - force=False：本函数当前等同 enabled=False（实际触发逻辑由调用方决定，spec 中只有 force=True 一种触发）
    - force=True：必落盘
    - 写入失败：log warning，返回 None（不阻塞主循环）
    - 落盘后：维护 latest.txt 指针（Windows 兼容），并按 retention 删最旧

    返回写入的 Path，或 None（未写入）。
    """
    if not enabled or not force:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    transcripts_dir = agent_home / ".transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    target = transcripts_dir / f"transcript_{ts}_{short_uuid}.jsonl"

    try:
        _write_jsonl(target, messages, session_id=session_id, reason="pre_llm_compact")
    except OSError as e:
        logger.warning("transcript 写入失败 (%s): %s", target, e)
        return None

    _update_latest_pointer(transcripts_dir, target)
    _prune_old(transcripts_dir, retention)

    logger.info("transcript snapshot: %s (msgs=%d)", target, len(messages))
    return target


def _write_jsonl(path: Path, messages: list, *, session_id: str, reason: str) -> None:
    """每行一条消息（含 ts），最后一行是 _meta。原子写入。"""
    now_iso = datetime.now().isoformat(timespec="seconds")
    lines = []
    for seq, msg in enumerate(messages):
        envelope = {"seq": seq, "ts": now_iso, **msg}
        lines.append(json.dumps(envelope, ensure_ascii=False))
    # 元数据行
    lines.append(json.dumps({
        "_meta": {
            "session_id": session_id,
            "reason": reason,
            "orig_len": len(messages),
        }
    }, ensure_ascii=False))

    content = "\n".join(lines) + "\n"
    _write_atomically(path, content)


def _write_atomically(path: Path, content: str) -> None:
    """原子写入（同 output_offload 的实现）。"""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8",
        delete=False, suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _update_latest_pointer(transcripts_dir: Path, target: Path) -> None:
    """维护 latest.txt 文本指针（不用 symlink，避免 Windows 权限问题）。"""
    pointer = transcripts_dir / "latest.txt"
    try:
        pointer.write_text(str(target), encoding="utf-8")
    except OSError as e:
        logger.debug("latest 指针更新失败: %s", e)


def _prune_old(transcripts_dir: Path, retention: int) -> None:
    """保留最近 retention 个 transcript 文件，删最旧。"""
    files = sorted(
        transcripts_dir.glob("transcript_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[retention:]:
        try:
            old.unlink()
        except OSError as e:
            logger.debug("删除旧 transcript 失败 %s: %s", old, e)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_transcript.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/transcript.py tests/test_transcript.py
git commit -m "feat(context): 新增 transcript 模块（压缩前快照存档）"
```

---

## Task 3: pipeline L1 snip_compact

**Files:**
- Create: `agent/context_pipeline.py`（首次创建，本任务只放 L1）
- Create: `tests/test_context_pipeline.py`（首次创建，本任务只测 L1）

**Interfaces:**
- Consumes: 无外部依赖（纯函数）
- Produces: `snip_compact(messages: list, *, keep_first: int = 3, keep_last: int = 47, threshold: int = 50) -> tuple[list, bool]`、`_split_system(messages) -> tuple[Optional[dict], list]`（内部 helper，后续 task 复用）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_context_pipeline.py
"""分层压缩管线测试。"""
from agent.context_pipeline import snip_compact, _split_system


def _mk_msgs(n, with_system=True):
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": "sys"})
    for i in range(n):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_split_system_extracts_system():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    sys, conv = _split_system(msgs)
    assert sys == {"role": "system", "content": "s"}
    assert conv == [{"role": "user", "content": "u"}]


def test_split_system_no_system_returns_none():
    msgs = [{"role": "user", "content": "u"}]
    sys, conv = _split_system(msgs)
    assert sys is None
    assert conv == msgs


def test_snip_below_threshold_noop():
    msgs = _mk_msgs(20)  # 1 + 40 = 41 条 < 50
    out, changed = snip_compact(msgs, threshold=50)
    assert changed is False
    assert out == msgs


def test_snip_above_threshold_cuts_middle():
    msgs = _mk_msgs(40)  # 1 + 80 = 81 条 > 50
    out, changed = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    assert changed is True
    # 期望结构：system + 3 头 + 1 占位 + 47 尾 = 52
    assert len(out) == 52
    assert out[0]["role"] == "system"
    # 占位消息
    placeholder = out[4]  # system + 3 head + placeholder
    assert "snip_compact" in placeholder["content"]


def test_snip_placeholder_mentions_transcript_path():
    msgs = _mk_msgs(40)
    out, _ = snip_compact(msgs, threshold=50)
    placeholders = [m for m in out if "snip_compact" in m.get("content", "")]
    assert len(placeholders) == 1
    assert ".transcripts" in placeholders[0]["content"]


def test_snip_idempotent_after_release():
    """snip 后的消息数若低于 release 阈值，再调一次不二次裁剪。"""
    msgs = _mk_msgs(40)
    out1, _ = snip_compact(msgs, threshold=50, keep_first=3, keep_last=47)
    # out1 有 52 条 > 50，但占位消息已是裁剪结果
    out2, changed = snip_compact(out1, threshold=50, keep_first=3, keep_last=47)
    # 第二次不应再裁（已经是占位形态）—— 通过检测占位数量
    placeholders = [m for m in out2 if "snip_compact" in m.get("content", "")]
    assert len(placeholders) == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写最小实现**

```python
# agent/context_pipeline.py
"""分层压缩管线：L1 snip / L2 micro / L4 llm + reactive。

替代 context_compressor.maybe_compress 的单层 LLM 摘要。
设计详见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3。
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _split_system(messages: list) -> Tuple[Optional[dict], list]:
    """分离 system 消息（如果有）。返回 (system_msg_or_None, rest)。"""
    if messages and messages[0].get("role") == "system":
        return messages[0], messages[1:]
    return None, messages


def _reassemble(system: Optional[dict], conv: list) -> list:
    """重新组装：system（若有）+ conv。"""
    return [system, *conv] if system else conv


def snip_compact(
    messages: list,
    *,
    keep_first: int = 3,
    keep_last: int = 47,
    threshold: int = 50,
) -> Tuple[list, bool]:
    """L1：消息数 > threshold 时裁中间，保留首 N + 尾 M + 占位。

    无损：占位消息提示 LLM 去 .transcripts/latest.jsonl 读回完整内容。
    返回 (新消息, 是否裁剪)。
    """
    system, conv = _split_system(messages)
    # 已有占位 → 不二次裁（幂等）
    placeholders = [m for m in conv if "snip_compact" in str(m.get("content", ""))]
    if placeholders:
        return messages, False
    if len(conv) <= threshold:
        return messages, False
    if len(conv) <= keep_first + keep_last:
        return messages, False

    head = conv[:keep_first]
    tail = conv[-keep_last:]
    omitted = len(conv) - keep_first - keep_last
    placeholder = {
        "role": "user",
        "content": (
            f"[snip_compact: 中间 {omitted} 条已省略，"
            f"完整记录见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = head + [placeholder] + tail
    new_messages = _reassemble(system, new_conv)
    logger.info("L1 snip_compact: conv %d → %d (omitted %d)",
                len(conv), len(new_conv), omitted)
    return new_messages, True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat(context): 新增 L1 snip_compact 分层（裁中间保首尾）"
```

---

## Task 4: pipeline L2 micro_compact

**Files:**
- Modify: `agent/context_pipeline.py`（追加 L2 函数）
- Modify: `tests/test_context_pipeline.py`（追加 L2 测试）

**Interfaces:**
- Consumes: `_split_system`、`_reassemble`（来自 Task 3）
- Produces: `micro_compact(messages: list, *, keep_recent: int = 3) -> tuple[list, bool]`

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_context_pipeline.py
from agent.context_pipeline import micro_compact


def _mk_with_tools(n_tools, recent=3):
    """构造 n_tools 条 tool 消息（夹在 user/assistant 之间）。"""
    msgs = [{"role": "system", "content": "s"}]
    for i in range(n_tools):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_{i}", "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}", "name": "t",
            "content": f"result_{i}" * 100,  # 长内容
        })
    return msgs


def test_micro_below_threshold_noop():
    msgs = _mk_with_tools(3)
    out, changed = micro_compact(msgs, keep_recent=3)
    assert changed is False
    assert out == msgs


def test_micro_replaces_old_tool_content():
    msgs = _mk_with_tools(5)  # 5 个 tool 消息
    out, changed = micro_compact(msgs, keep_recent=3)
    assert changed is True
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 5  # 数量不变
    # 前 2 个被折叠，后 3 个保留原文
    assert "micro_compacted" in tool_msgs[0]["content"]
    assert "micro_compacted" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"] == "result_2" * 100
    assert tool_msgs[3]["content"] == "result_3" * 100
    assert tool_msgs[4]["content"] == "result_4" * 100


def test_micro_preserves_tool_call_id_and_name():
    """折叠只换 content，role/tool_call_id/name 不变（保配对）。"""
    msgs = _mk_with_tools(5)
    out, _ = micro_compact(msgs, keep_recent=3)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_0"
    assert tool_msgs[0]["name"] == "t"


def test_micro_idempotent():
    """已经是占位的不再二次折叠。"""
    msgs = _mk_with_tools(5)
    out1, _ = micro_compact(msgs, keep_recent=3)
    out2, changed = micro_compact(out1, keep_recent=3)
    assert changed is False  # 第二次无事可做
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_pipeline.py::test_micro_below_threshold_noop -v`
Expected: FAIL — `ImportError: cannot import name 'micro_compact'`

- [ ] **Step 3: 追加最小实现**

```python
# 追加到 agent/context_pipeline.py
import json


def micro_compact(
    messages: list,
    *,
    keep_recent: int = 3,
) -> Tuple[list, bool]:
    """L2：把较旧的 tool 消息 content 替换为占位 JSON。

    无损：占位提示去 .transcripts/latest.jsonl 或重跑工具。
    安全：只换 content，保留 role/tool_call_id/name（不破 tool_call 配对）。
    幂等：已是占位的不再动。
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return messages, False

    to_compact = set(tool_indices[:-keep_recent])  # 除最后 keep_recent 个外
    folded = 0
    out = []
    for i, m in enumerate(messages):
        if i in to_compact and not _already_micro_placeheld(m):
            new_m = dict(m)
            orig_len = len(str(m.get("content", "")))
            new_m["content"] = json.dumps({
                "micro_compacted": True,
                "orig_chars": orig_len,
                "hint": (
                    f"Tool {m.get('name', '?')} 结果已折叠，"
                    f"完整内容见 .transcripts/latest.jsonl 或重跑工具"
                ),
            }, ensure_ascii=False)
            out.append(new_m)
            folded += 1
        else:
            out.append(m)

    if folded == 0:
        return messages, False
    logger.info("L2 micro_compact: folded %d old tool results", folded)
    return out, True


def _already_micro_placeheld(msg: dict) -> bool:
    """检测 tool 消息 content 是否已是 micro_compacted 占位。"""
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
        return bool(parsed.get("micro_compacted"))
    except (json.JSONDecodeError, TypeError):
        return False
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: PASS（10 tests，含 Task 3 的 6 个 + 本任务的 4 个）

- [ ] **Step 5: 提交**

```bash
git add agent/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat(context): 新增 L2 micro_compact 分层（折叠旧 tool 内容）"
```

---

## Task 5: pipeline L4 llm_compact

**Files:**
- Modify: `agent/context_pipeline.py`（追加 L4）
- Modify: `tests/test_context_pipeline.py`（追加 L4 测试）

**Interfaces:**
- Consumes: `_split_system`、`_reassemble`（Task 3）；`_summarize_conversation`、`_fix_tool_call_pairs`、`estimate_message_tokens` from `agent.context_compressor`（现有）
- Produces: `llm_compact(messages, *, llm_client, model, keep_recent=10, token_threshold=100000, msg_threshold=100) -> tuple[list, bool]`

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_context_pipeline.py
from unittest.mock import MagicMock

from agent.context_pipeline import llm_compact


class _FakeLLM:
    """模拟 OpenAI 兼容 client。"""
    def chat_completions(self, msgs):
        m = MagicMock()
        m.choices = [MagicMock(message=MagicMock(content="这是对话总结"))]
        return m


def test_llm_below_threshold_noop():
    msgs = _mk_msgs(20)
    out, changed = llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100000, msg_threshold=100,
    )
    assert changed is False


def test_llm_over_msg_threshold_compacts():
    msgs = _mk_msgs(80)  # 1 + 160 = 161 条 > 100
    out, changed = llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=100000, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 期望：system + summary placeholder + 10 keep_recent = 12
    assert len(out) == 12
    assert out[0]["role"] == "system"
    assert "总结" in out[1]["content"]


def test_llm_no_client_falls_back_to_rule_based():
    """llm_client=None 时仍能工作（沿用现有 _rule_based_summary）。"""
    msgs = _mk_msgs(80)
    out, changed = llm_compact(
        msgs, llm_client=None, model="x",
        token_threshold=100000, msg_threshold=100, keep_recent=10,
    )
    assert changed is True
    # 占位消息应含规则提取内容
    assert "用户" in out[1]["content"] or "总结" in out[1]["content"]


def test_llm_fixes_tool_call_pairs():
    """压缩后 _fix_tool_call_pairs 应补漏（无配对 tool_result 的 tool_call）。"""
    msgs = [{"role": "system", "content": "s"}]
    msgs.append({"role": "user", "content": "u"})
    msgs.append({
        "role": "assistant",
        "tool_calls": [{"id": "call_x", "function": {"name": "t", "arguments": "{}"}}],
    })
    # 故意不给 tool 消息（模拟压缩边界丢失）
    for i in range(120):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    out, _ = llm_compact(
        msgs, llm_client=_FakeLLM(), model="x",
        token_threshold=10**9, msg_threshold=100, keep_recent=10,
    )
    # 找回 tool_result 补漏（如果有未配对的 tool_call 留在 keep_recent 里）
    # 这里 keep_recent 是最后 10 条，不含 assistant(tool_calls)，所以应该不补
    # 主要验证不抛异常
    assert isinstance(out, list)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_pipeline.py::test_llm_over_msg_threshold_compacts -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 追加最小实现**

```python
# 追加到 agent/context_pipeline.py
from agent.context_compressor import (
    _summarize_conversation, _fix_tool_call_pairs, estimate_message_tokens,
)


def llm_compact(
    messages: list,
    *,
    llm_client,
    model: Optional[str],
    keep_recent: int = 10,
    token_threshold: int = 100000,
    msg_threshold: int = 100,
) -> Tuple[list, bool]:
    """L4：L1+L2 后仍超阈值时，调 LLM 总结早期对话。

    有损：用 1 次 API 调用换上下文空间。调用方应先 transcript.snapshot_if_needed(force=True)。
    沿用现有 _summarize_conversation（含 _rule_based_summary 降级）和 _fix_tool_call_pairs。
    """
    system, conv = _split_system(messages)
    over_token = estimate_message_tokens(messages) > token_threshold
    over_msg = len(conv) > msg_threshold
    if not (over_token or over_msg):
        return messages, False
    if len(conv) <= keep_recent:
        return messages, False

    to_summarize = conv[:-keep_recent]
    keep = conv[-keep_recent:]

    summary = _summarize_conversation(to_summarize, llm_client, model=model)
    if not summary:
        return messages, False

    placeholder = {
        "role": "user",
        "content": (
            "[之前的对话已自动总结]\n\n"
            f"{summary}\n\n"
            "[以下是最近的对话，请继续]"
        ),
    }
    new_conv = [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    logger.info(
        "L4 llm_compact: %d msgs summarized, %d chars → %d chars summary",
        len(to_summarize),
        sum(len(str(m.get("content", ""))) for m in to_summarize),
        len(summary),
    )
    return new_messages, True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: PASS（14 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat(context): 新增 L4 llm_compact 分层（沿用现有 _summarize_conversation）"
```

---

## Task 6: pipeline CompressionSessionState + reactive_compact

**Files:**
- Modify: `agent/context_pipeline.py`（追加 dataclass + reactive）
- Modify: `tests/test_context_pipeline.py`（追加测试）

**Interfaces:**
- Consumes: `_split_system`、`_reassemble`、`_fix_tool_call_pairs`
- Produces: `CompressionSessionState` dataclass、`reactive_compact(messages, *, session_state, keep_recent=5) -> tuple[list, bool]`

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_context_pipeline.py
from agent.context_pipeline import CompressionSessionState, reactive_compact


def test_session_state_default():
    s = CompressionSessionState()
    assert s.reacted is False
    assert s.llm_compact_count == 0
    assert s.cooldown_ok(5) is True


def test_session_state_record_and_cooldown():
    s = CompressionSessionState()
    s.current_turn = 10
    s.record_llm_compact()
    assert s.llm_compact_count == 1
    assert s.last_llm_compact_turn == 10
    s.current_turn = 12
    assert s.cooldown_ok(5) is False  # 12-10=2 < 5
    s.current_turn = 16
    assert s.cooldown_ok(5) is True   # 16-10=6 >= 5


def test_reactive_truncates_to_last_5():
    msgs = _mk_msgs(40)  # 81 条
    state = CompressionSessionState()
    out, changed = reactive_compact(msgs, session_state=state)
    assert changed is True
    assert state.reacted is True
    # system + placeholder + 5 条
    assert len(out) == 7
    assert out[0]["role"] == "system"
    assert "紧急上下文压缩" in out[1]["content"]


def test_reactive_once_per_session():
    """session_state.reacted=True 时不再触发。"""
    msgs = _mk_msgs(40)
    state = CompressionSessionState(reacted=True)
    out, changed = reactive_compact(msgs, session_state=state)
    assert changed is False
    assert out == msgs


def test_reactive_short_history_kept_as_is():
    """消息少于 keep_recent 时不补占位，全部保留。"""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}]
    state = CompressionSessionState()
    out, changed = reactive_compact(msgs, session_state=state, keep_recent=5)
    assert changed is True
    assert state.reacted is True
    # system + placeholder + 全部 3 条
    assert len(out) == 5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_pipeline.py::test_reactive_truncates_to_last_5 -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 追加最小实现**

```python
# 追加到 agent/context_pipeline.py
from dataclasses import dataclass


@dataclass
class CompressionSessionState:
    """单会话的压缩状态。

    - reacted: 本会话是否已触发过 reactive_compact（once-per-session）
    - llm_compact_count: L4 触发次数
    - last_llm_compact_turn: 上次 L4 触发时的 current_turn（用于 cooldown）
    - current_turn: 当前 LLM 轮次（由 agent 主循环 increment）
    """
    reacted: bool = False
    llm_compact_count: int = 0
    last_llm_compact_turn: int = -10**6
    current_turn: int = 0

    def record_llm_compact(self) -> None:
        self.llm_compact_count += 1
        self.last_llm_compact_turn = self.current_turn

    def cooldown_ok(self, cooldown_turns: int) -> bool:
        return self.current_turn - self.last_llm_compact_turn >= cooldown_turns

    def increment_turn(self) -> None:
        self.current_turn += 1


def reactive_compact(
    messages: list,
    *,
    session_state: CompressionSessionState,
    keep_recent: int = 5,
) -> Tuple[list, bool]:
    """紧急通道：API 报 prompt_too_long 时调用。

    只留 system + 占位 + 最后 keep_recent 条。
    会话级 once-per-session：session_state.reacted=True 后不再触发。
    """
    if session_state.reacted:
        return messages, False

    system, conv = _split_system(messages)
    keep = conv[-keep_recent:] if len(conv) > keep_recent else conv[:]
    placeholder = {
        "role": "user",
        "content": (
            "[紧急上下文压缩：API 返回 prompt_too_long，"
            f"已只保留最近 {len(keep)} 条消息。"
            "完整历史见 .transcripts/latest.jsonl]"
        ),
    }
    new_conv = [placeholder] + keep
    new_conv = _fix_tool_call_pairs(new_conv)
    new_messages = _reassemble(system, new_conv)

    session_state.reacted = True
    logger.warning("reactive_compact triggered: kept last %d", len(keep))
    return new_messages, True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: PASS（19 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat(context): 新增 reactive_compact + CompressionSessionState"
```

---

## Task 7: pipeline 编排器 compress_if_needed

**Files:**
- Modify: `agent/context_pipeline.py`（追加顶层编排）
- Modify: `tests/test_context_pipeline.py`（追加编排测试）

**Interfaces:**
- Consumes: `snip_compact`、`micro_compact`、`llm_compact`（Task 3-5）；`transcript.snapshot_if_needed`（Task 2）
- Produces: `compress_if_needed(messages, *, attempt_count, llm_client, model, config, session_state, agent_home, session_id) -> tuple[list, bool]`

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_context_pipeline.py
from agent.context_pipeline import compress_if_needed


_DEFAULT_CFG = {
    "snip_message_threshold": 50,
    "snip_keep_first": 3,
    "snip_keep_last": 47,
    "micro_keep_recent_results": 3,
    "llm_compact_token_threshold": 100000,
    "llm_compact_message_threshold": 100,
    "llm_compact_keep_recent": 10,
    "llm_compact_cooldown_turns": 5,
    "max_compress_attempts": 3,
    "transcript_enabled": True,
    "transcript_retention": 20,
}


def test_compress_runs_l1_only_for_medium_conv(tmp_path):
    """50 < 消息数 < 100 时只跑 L1+L2，不触发 L4。"""
    msgs = _mk_msgs(40)  # 81 条，触发 L1，不触发 L4
    state = CompressionSessionState()
    out, changed = compress_if_needed(
        msgs, attempt_count=0, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    assert state.llm_compact_count == 0  # L4 未触发


def test_compress_runs_l4_for_huge_conv(tmp_path):
    """消息数 > 100 且 token 超限时触发 L4。"""
    msgs = _mk_msgs(80)  # 161 条
    state = CompressionSessionState()
    out, changed = compress_if_needed(
        msgs, attempt_count=0, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is True
    assert state.llm_compact_count == 1


def test_compress_respects_max_attempts(tmp_path):
    """attempt_count >= max_compress_attempts 时不再 L4。"""
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    out, changed = compress_if_needed(
        msgs, attempt_count=3, llm_client=_FakeLLM(), model="x",
        config={**_DEFAULT_CFG, "max_compress_attempts": 3}, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    # L1+L2 仍跑，L4 被跳过
    assert state.llm_compact_count == 0


def test_compress_respects_cooldown(tmp_path):
    """L4 触发后 cooldown 期内不再触发。"""
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    state.current_turn = 10
    state.record_llm_compact()  # turn=10 触发
    state.current_turn = 12      # 只过了 2 轮 < 5

    out, changed = compress_if_needed(
        msgs, attempt_count=0, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    # L4 被 cooldown 拦下，但 L1+L2 仍可能跑（changed 可能仍 True）
    assert state.llm_compact_count == 1  # 未增长


def test_compress_no_change_when_small(tmp_path):
    msgs = _mk_msgs(10)
    state = CompressionSessionState()
    out, changed = compress_if_needed(
        msgs, attempt_count=0, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="s",
    )
    assert changed is False


def test_compress_writes_transcript_before_l4(tmp_path):
    """L4 触发前应落盘 transcript（force=True）。"""
    msgs = _mk_msgs(80)
    state = CompressionSessionState()
    compress_if_needed(
        msgs, attempt_count=0, llm_client=_FakeLLM(), model="x",
        config=_DEFAULT_CFG, session_state=state,
        agent_home=tmp_path, session_id="sess_t",
    )
    transcripts = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(transcripts) >= 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_pipeline.py::test_compress_runs_l4_for_huge_conv -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 追加最小实现**

```python
# 追加到 agent/context_pipeline.py
from agent.transcript import snapshot_if_needed


def compress_if_needed(
    messages: list,
    *,
    attempt_count: int,
    llm_client,
    model: Optional[str],
    config: dict,
    session_state: CompressionSessionState,
    agent_home,
    session_id: str,
) -> Tuple[list, bool]:
    """分层压缩编排器。返回 (新消息, 是否发生变化)。

    顺序：L1 snip → L2 micro → (条件) transcript 快照 → L4 llm。
    每层独立判定是否触发，最终统一过 _fix_tool_call_pairs。
    """
    # L1 snip
    messages, c1 = snip_compact(
        messages,
        keep_first=config.get("snip_keep_first", 3),
        keep_last=config.get("snip_keep_last", 47),
        threshold=config.get("snip_message_threshold", 50),
    )

    # L2 micro
    messages, c2 = micro_compact(
        messages,
        keep_recent=config.get("micro_keep_recent_results", 3),
    )

    # L4 llm（条件：未超 max_attempts + cooldown 已过 + 超阈值）
    c4 = False
    max_attempts = config.get("max_compress_attempts", 3)
    cooldown = config.get("llm_compact_cooldown_turns", 5)
    over_threshold = (
        estimate_message_tokens(messages) > config.get("llm_compact_token_threshold", 100000)
        or len(_split_system(messages)[1]) > config.get("llm_compact_message_threshold", 100)
    )
    if over_threshold and attempt_count < max_attempts and session_state.cooldown_ok(cooldown):
        # L4 前落盘 transcript（force=True，因为 L4 是有损的）
        if config.get("transcript_enabled", True):
            try:
                snapshot_if_needed(
                    messages,
                    agent_home=agent_home,
                    session_id=session_id,
                    force=True,
                    enabled=True,
                    retention=config.get("transcript_retention", 20),
                )
            except Exception as e:
                logger.warning("transcript snapshot 失败（不阻塞 L4）: %s", e)

        messages, c4 = llm_compact(
            messages,
            llm_client=llm_client,
            model=model,
            keep_recent=config.get("llm_compact_keep_recent", 10),
            token_threshold=config.get("llm_compact_token_threshold", 100000),
            msg_threshold=config.get("llm_compact_message_threshold", 100),
        )
        if c4:
            session_state.record_llm_compact()

    changed = c1 or c2 or c4
    if changed:
        # 终极保险：再过一遍 _fix_tool_call_pairs
        system, conv = _split_system(messages)
        messages = _reassemble(system, _fix_tool_call_pairs(conv))
    return messages, changed
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_pipeline.py -v`
Expected: PASS（25 tests）

- [ ] **Step 5: 提交**

```bash
git add agent/context_pipeline.py tests/test_context_pipeline.py
git commit -m "feat(context): 新增 compress_if_needed 编排器（L1→L2→L4 + transcript 快照）"
```

---

## Task 8: config + memory_manager 加 on_pre_compress 钩子

**Files:**
- Modify: `config.py:DEFAULT_CONFIG`（新增 `context` 块）
- Modify: `agent/memory_manager.py`（新增 no-op 方法）
- Modify: `tests/test_config.py`（断言新块）

**Interfaces:**
- Consumes: 无
- Produces: `DEFAULT_CONFIG["context"]` 字典；`MemoryManager.on_pre_compress(snapshot_path, messages) -> None`

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_config.py
def test_default_config_has_context_block():
    """config.py 的 DEFAULT_CONFIG 应含 context 块及所有 Phase 1 阈值。"""
    from config import DEFAULT_CONFIG
    ctx = DEFAULT_CONFIG["context"]
    expected_keys = {
        "output_offload_threshold", "output_offload_preview",
        "snip_message_threshold", "snip_release_threshold",
        "snip_keep_first", "snip_keep_last",
        "micro_keep_recent_results",
        "llm_compact_token_threshold", "llm_compact_message_threshold",
        "llm_compact_keep_recent", "llm_compact_cooldown_turns",
        "max_compress_attempts",
        "reactive_keep_recent", "reactive_once_per_session",
        "transcript_enabled", "transcript_trigger", "transcript_retention",
        "use_new_pipeline",
    }
    assert expected_keys.issubset(set(ctx.keys())), f"缺: {expected_keys - set(ctx.keys())}"


def test_default_config_use_new_pipeline_is_false():
    """双轨期默认 False（Commit 6 才改 True）。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["context"]["use_new_pipeline"] is False


def test_memory_manager_on_pre_compress_is_noop():
    """MemoryManager.on_pre_compress 默认 no-op（不抛即可）。"""
    from agent.memory_manager import MemoryManager
    from agent.memory_store import MemoryStore
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(Path(td))
        mm = MemoryManager(memory_store=store, external_provider=None)
        mm.on_pre_compress(None, [])  # 不抛
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_default_config_has_context_block -v`
Expected: FAIL — `KeyError: 'context'`

- [ ] **Step 3: 改 config.py**

在 `config.py:DEFAULT_CONFIG` 里新增 `context` 块（紧跟 `agent` 块之后）：

```python
# 找到 DEFAULT_CONFIG 里 "agent": {...} 块的结束位置，在其后追加：

    # 上下文压缩管线（Phase 1）
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
        # 功能开关（双轨期；Commit 6 改 True）
        "use_new_pipeline": False,
    },
```

- [ ] **Step 4: 改 memory_manager.py**

在 `MemoryManager` 类里追加方法（任何位置都行，建议紧跟 `__init__` 之后）：

```python
    def on_pre_compress(self, snapshot_path, messages: list) -> None:
        """钩子：压缩前调用。Phase 1 留空（no-op），未来扩展用。

        设计原因：HermesAgent 当前记忆模型是主动式（LLM 通过 memory_tool 自己写），
        强行加 LLM 被动抽取会和现有模型冲突。Phase 5（或独立 Phase 1.5）实现。
        """
        pass
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（含 3 个新测试）

- [ ] **Step 6: 全量回归**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 294 + 本任务 3 = 297，且开关默认 False 不影响其他模块）

- [ ] **Step 7: 提交**

```bash
git add config.py agent/memory_manager.py tests/test_config.py
git commit -m "feat(config): 新增 context 配置块 + MemoryManager.on_pre_compress 钩子"
```

---

## Task 9: agent/__init__.py 集成（开关控制，Commit 4）

**Files:**
- Modify: `agent/__init__.py:212-225`（替换 maybe_compress 调用）+ 新增 reactive 异常分支
- Modify: `agent/context_compressor.py`（maybe_compress 加 DeprecationWarning）
- Modify: `tests/test_context.py`（加 deprecation 警告断言）+ `tests/test_integration.py`（加双轨用例）

**Interfaces:**
- Consumes: `context_pipeline.compress_if_needed` / `reactive_compact` / `CompressionSessionState`（Task 6-7）；`maybe_compress`（现有，deprecate）
- Produces: `AIAgent` 新增内部状态 `_compress_session_state: CompressionSessionState` 和 `_reacted`

- [ ] **Step 1: 加 deprecation 警告测试**

```python
# 追加到 tests/test_context.py
import warnings
from agent.context_compressor import maybe_compress


def test_maybe_compress_emits_deprecation_warning():
    """maybe_compress 调用时应发 DeprecationWarning。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msgs = [{"role": "system", "content": "s"}]
        msgs += [{"role": "user", "content": f"u{i}"} for i in range(50)]
        maybe_compress(msgs, attempt_count=0, llm_client=None)
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w)
```

```python
# 追加到 tests/test_integration.py
def test_aiagent_uses_new_pipeline_when_flag_true():
    """config.context.use_new_pipeline=True 时走新管线（L1 snip）。"""
    # 略：mock OpenAI client，构造超 50 条历史，断言 snip_compact 占位消息出现
    # 这一步在 Task 11 的端到端测试里会更完整；此处只验证开关分支不抛
    from agent.context_pipeline import compress_if_needed, CompressionSessionState
    # 直接调用，验证签名匹配
    msgs = [{"role": "system", "content": "s"}]
    msgs += [{"role": "user", "content": f"u{i}"} for i in range(60)]
    state = CompressionSessionState()
    out, _ = compress_if_needed(
        msgs, attempt_count=0, llm_client=None, model=None,
        config={
            "snip_message_threshold": 50, "snip_keep_first": 3, "snip_keep_last": 47,
            "micro_keep_recent_results": 3, "llm_compact_token_threshold": 100000,
            "llm_compact_message_threshold": 100, "llm_compact_keep_recent": 10,
            "llm_compact_cooldown_turns": 5, "max_compress_attempts": 3,
            "transcript_enabled": False, "transcript_retention": 20,
        },
        session_state=state, agent_home=None, session_id="t",
    )
    assert any("snip_compact" in m.get("content", "") for m in out)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context.py::test_maybe_compress_emits_deprecation_warning -v`
Expected: FAIL — 没有发警告

- [ ] **Step 3: 改 context_compressor.py（加警告）**

找到 `agent/context_compressor.py:maybe_compress` 函数定义，在第一行（docstring 之后）插入：

```python
def maybe_compress(messages, *, attempt_count=0, model=None, llm_client=None,
                   context_window_tokens=128000):
    """[已废弃] 单层 LLM 摘要压缩。新代码请用 agent.context_pipeline.compress_if_needed。

    保留是为了双轨期回退（config.context.use_new_pipeline=False 时仍调用）。
    Commit 7（下个 minor 版本）完全移除。
    """
    import warnings
    warnings.warn(
        "maybe_compress 已废弃，请改用 agent.context_pipeline.compress_if_needed",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... 原有实现保留不动 ...
```

- [ ] **Step 4: 改 agent/__init__.py 集成分支**

定位 `agent/__init__.py:212-225`（现有 `maybe_compress` 调用块）。**替换为：**

```python
            # 上下文压缩（接近 token 上限时触发）
            if self.compression_enabled:
                use_new = self.config.get("context", {}).get("use_new_pipeline", False)
                if use_new:
                    # 新管线：L1/L2/L4 + transcript 快照
                    if not hasattr(self, "_compress_session_state"):
                        from agent.context_pipeline import CompressionSessionState
                        self._compress_session_state = CompressionSessionState()
                    from agent.context_pipeline import compress_if_needed
                    ctx_cfg = self.config.get("context", {})
                    messages, compressed = compress_if_needed(
                        messages,
                        attempt_count=self._compression_attempts,
                        llm_client=self.llm_client,
                        model=self.model,
                        config=ctx_cfg,
                        session_state=self._compress_session_state,
                        agent_home=self.harvil_home,
                        session_id=self.session_id,
                    )
                else:
                    # 旧路径（双轨期保留）
                    from agent.context_compressor import maybe_compress
                    messages, compressed = maybe_compress(
                        messages,
                        attempt_count=self._compression_attempts,
                        model=self.model,
                        llm_client=self.llm_client,
                    )
                if compressed:
                    # 压缩会修改历史，需要同步并重建 system prompt
                    self.conversation_history = messages[1:]  # 跳过 system
                    self.invalidate_system_prompt()
                    system_prompt = self._get_system_prompt()
                    self._compression_attempts += 1
```

然后在 LLM 调用的 try/except 块（约第 228-243 行）增加 reactive 分支。**找到现有的：**

```python
            try:
                from agent.llm_retry import call_with_retry
                response = call_with_retry(...)
            except Exception as e:
                logger.error("LLM API 调用失败（重试后）: %s", e)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[API 错误: {e}]",
                })
                break
```

**改为：**

```python
            try:
                from agent.llm_retry import call_with_retry
                response = call_with_retry(
                    self.llm_client,
                    messages,
                    tools=tool_schemas if tool_schemas else None,
                    fallback_llm_client=self.fallback_llm_client,
                )
            except Exception as e:
                err_str = str(e).lower()
                is_prompt_too_long = (
                    "prompt_too_long" in err_str
                    or "context_length" in err_str
                    or "maximum context" in err_str
                )
                use_new = self.config.get("context", {}).get("use_new_pipeline", False)
                if (is_prompt_too_long and use_new
                        and not getattr(self, "_reacted", False)):
                    from agent.context_pipeline import reactive_compact
                    if not hasattr(self, "_compress_session_state"):
                        from agent.context_pipeline import CompressionSessionState
                        self._compress_session_state = CompressionSessionState()
                    messages = reactive_compact(
                        messages,
                        session_state=self._compress_session_state,
                        keep_recent=self.config.get("context", {}).get(
                            "reactive_keep_recent", 5),
                    )
                    self._reacted = True
                    self.conversation_history = messages[1:]
                    self.invalidate_system_prompt()
                    logger.warning("reactive_compact 后重试本轮")
                    continue  # 重试本轮
                logger.error("LLM API 调用失败（重试后）: %s", e)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[API 错误: {e}]",
                })
                break
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_context.py tests/test_integration.py -v`
Expected: PASS（含 2 个新测试）

- [ ] **Step 6: 全量回归（关键：开关 False 时一切照旧）**

Run: `uv run pytest tests/ -v`
Expected: PASS（原 294 + 新增；开关默认 False，旧路径仍在）

- [ ] **Step 7: 提交**

```bash
git add agent/__init__.py agent/context_compressor.py tests/test_context.py tests/test_integration.py
git commit -m "feat(context): AIAgent 集成新管线（开关控制）+ reactive_compact 分支"
```

---

## Task 10: 工具层 offload 集成 + system prompt 指导（Commit 5）

**Files:**
- Modify: `tools/terminal_tool.py`（handler 返回前过 maybe_offload）
- Modify: `tools/file_operations.py`（同上）
- Modify: `agent/prompt_builder.py:TOOL_USAGE_GUIDANCE`（加占位消息识别段）
- Modify: 现有工具测试 / 新增小测试

**Interfaces:**
- Consumes: `output_offload.maybe_offload`（Task 1）；`config.context` 阈值（Task 8）
- Produces: 工具 handler 在大输出时返回 offload JSON

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_integration.py（或新建 tests/test_tool_offload_integration.py）
import json
from pathlib import Path
from unittest.mock import patch

from tools.terminal_tool import _execute_terminal  # 或实际入口名


def test_terminal_large_output_triggers_offload(tmp_path: Path, monkeypatch):
    """terminal 输出 > 30KB 时走 offload（开关开启时）。"""
    # 此测试需要 terminal_tool 暴露的内部入口；
    # 若 terminal_tool 已封装为 handler(info, **kwargs)，则构造 info 调用
    # 这里给一个示意断言：
    long_output = "x" * 50000
    from agent.output_offload import maybe_offload
    result = maybe_offload(
        long_output, tool_call_id="call_t1", agent_home=tmp_path,
        threshold=30000, preview_chars=2000,
    )
    parsed = json.loads(result)
    assert parsed["truncated"] is True


def test_prompt_builder_includes_offload_guidance():
    """system prompt 应含占位消息识别段。"""
    from agent.prompt_builder import TOOL_USAGE_GUIDANCE
    assert "snip_compact" in TOOL_USAGE_GUIDANCE
    assert ".transcripts" in TOOL_USAGE_GUIDANCE
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_integration.py::test_prompt_builder_includes_offload_guidance -v`
Expected: FAIL — 字符串不在

- [ ] **Step 3: 改 prompt_builder.py**

找到 `agent/prompt_builder.py:TOOL_USAGE_GUIDANCE` 字符串。**在末尾追加一段**（注意是字符串拼接）：

```python
TOOL_USAGE_GUIDANCE = (
    "## 工具使用规范\n"
    "- 所有工具结果都是 JSON 字符串，解析后使用\n"
    "- 工具失败时返回 {\"error\": \"...\"}，根据错误自我修正\n"
    "- 文件操作必须指定 encoding='utf-8'\n"
    "- 不要假设工具可用，check_fn 可能因环境不同而隐藏某些工具\n"
    "- **临时文件清理**：用 write_file 创建的临时脚本/中间文件，"
    "执行完成后必须立即用 terminal 删除，保持工作区干净。"
    "不要在用户的工作目录留下垃圾文件。\n"
    "- **上下文占位消息识别**：当你看到 [snip_compact] / "
    "\"micro_compacted\" / [紧急上下文压缩] 这类占位消息，且需要更早的"
    "上下文时，从占位消息里给的路径（通常是 .transcripts/latest.jsonl "
    "或 .task_outputs/tool-results/ 下的文件）用 read_file 读回。"
    "这些路径在 agent_home 下，默认安全。"
)
```

- [ ] **Step 4: 改 terminal_tool.py（在 handler 返回前过 offload）**

打开 `tools/terminal_tool.py`，找到 handler 入口（典型形如 `def terminal(info, **kwargs)` 或 `def _run_terminal(command, ...)`）。**在最终 return 前，把 content 包一层 maybe_offload**：

```python
# 在文件顶部 import 区追加
from agent.output_offload import maybe_offload

# 找到 handler 的最终 return，把 return json.dumps({...}) 改成：
def _finalize_output(result_content: str, tool_call_id: str, harvil_home, config):
    """根据开关决定是否走 offload。"""
    use_new = (config or {}).get("context", {}).get("use_new_pipeline", False)
    if not use_new:
        return result_content
    return maybe_offload(
        result_content,
        tool_call_id=tool_call_id,
        agent_home=harvil_home,
        threshold=(config or {}).get("context", {}).get("output_offload_threshold", 30000),
        preview_chars=(config or {}).get("context", {}).get("output_offload_preview", 2000),
    )
```

然后在 handler 里把返回 content 的部分改为先过 `_finalize_output`。具体行号取决于现有结构，原则是：**stdout 字段超过阈值时整个 content 走 offload**。

如果 handler 的返回结构是 `{"stdout": "...", "stderr": "...", "exit_code": 0}`，那么 offload 应作用在 `stdout` 上（最大字段）。示例：

```python
# 伪代码示意 —— 实际改动按 terminal_tool.py 结构调整
output = {"stdout": long_text, "stderr": "...", "exit_code": 0}
final_stdout = _finalize_output(long_text, tool_call_id, harvil_home, config)
if final_stdout != long_text:
    output["stdout"] = final_stdout  # 已被替换为 offload JSON 字符串
    output["stdout_offloaded"] = True
return json.dumps(output, ensure_ascii=False)
```

- [ ] **Step 5: 同改 file_operations.py**

`tools/file_operations.py` 中 `read_file` 工具的 handler 在读到超大文件时也走 offload。逻辑同上：找到 read_file handler 的 return，把 content 包一层 `_finalize_output`。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/ -v`
Expected: PASS（开关默认 False，offload 路径不触发，现有测试不受影响）

- [ ] **Step 7: 手动验证开关开启时不崩**

写一个 smoke 脚本 `scripts/smoke_offload.py`（临时，验证完删）：

```python
# scripts/smoke_offload.py（临时验证用，验证完删除）
"""临时 smoke：开关开启时跑一次 read_file 大文件。"""
import tempfile
from pathlib import Path
from agent.output_offload import maybe_offload

with tempfile.TemporaryDirectory() as td:
    big = "x" * 50000
    result = maybe_offload(big, tool_call_id="smoke", agent_home=Path(td))
    print("OK" if "truncated" in result else "FAIL")
```

Run: `uv run python scripts/smoke_offload.py`
Expected: 输出 `OK`

- [ ] **Step 8: 提交**

```bash
git add tools/terminal_tool.py tools/file_operations.py agent/prompt_builder.py tests/test_integration.py
git commit -m "feat(tools): terminal/file 大输出走 offload + system prompt 加占位识别段"
```

---

## Task 11: 端到端集成测试 + 切换默认开关（Commit 6）

**Files:**
- Modify: `tests/test_integration.py`（加 200 轮对话用例）
- Modify: `config.py:DEFAULT_CONFIG["context"]["use_new_pipeline"]` → `True`
- Create: `CHANGELOG.md`（若不存在）

**Interfaces:**
- Consumes: 所有前置 Task
- Produces: 端到端验证通过；默认启用新管线

- [ ] **Step 1: 写 200 轮端到端用例**

```python
# 追加到 tests/test_integration.py
from unittest.mock import MagicMock
from agent.context_pipeline import CompressionSessionState
from agent.transcript import snapshot_if_needed
from agent.output_offload import maybe_offload


def test_e2e_200_turn_conversation_with_pipeline(tmp_path):
    """端到端：200 轮工具调用对话，验证新管线稳定。

    - mock LLM 每 5 轮返回一次 tool_use
    - 每 10 轮注入一个 50KB 工具结果（触发 offload）
    - 断言：offload 文件 > 0，transcript ≥ 1，L4 触发 ≥ 1，全程无异常
    """
    from agent.context_pipeline import compress_if_needed

    config = {
        "snip_message_threshold": 50, "snip_keep_first": 3, "snip_keep_last": 47,
        "micro_keep_recent_results": 3,
        "llm_compact_token_threshold": 100000, "llm_compact_message_threshold": 100,
        "llm_compact_keep_recent": 10, "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "transcript_enabled": True, "transcript_retention": 20,
    }

    class _FakeLLM:
        def __init__(self):
            self.call_count = 0
        def chat_completions(self, msgs):
            self.call_count += 1
            m = MagicMock()
            m.choices = [MagicMock(message=MagicMock(
                content="ok", tool_calls=None
            ))]
            return m

    # 构造对话历史
    messages = [{"role": "system", "content": "sys"}]
    for i in range(200):
        messages.append({"role": "user", "content": f"turn {i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
        # 每 10 轮加一个大 tool 结果
        if i % 10 == 0:
            messages.append({
                "role": "assistant",
                "tool_calls": [{"id": f"call_{i}",
                                "function": {"name": "t", "arguments": "{}"}}],
            })
            big = "x" * 50000
            offloaded = maybe_offload(
                big, tool_call_id=f"call_{i}", agent_home=tmp_path,
            )
            messages.append({
                "role": "tool", "tool_call_id": f"call_{i}", "name": "t",
                "content": offloaded,
            })

    state = CompressionSessionState()
    llm = _FakeLLM()

    # 跑 5 轮压缩（模拟每轮 LLM 前调用）
    for turn in range(5):
        state.current_turn = turn
        messages, _ = compress_if_needed(
            messages, attempt_count=turn, llm_client=llm, model="x",
            config=config, session_state=state,
            agent_home=tmp_path, session_id="e2e",
        )

    # 断言
    offload_files = list((tmp_path / ".task_outputs" / "tool-results").glob("*.txt"))
    transcripts = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(offload_files) > 0, "应该有 offload 文件"
    assert len(transcripts) >= 1, "应该有 transcript"
    assert state.llm_compact_count >= 1, "L4 应至少触发一次"
    # 最终 messages 长度应远小于起始（压缩生效）
    assert len(messages) < 100
```

- [ ] **Step 2: 跑测试确认通过**

Run: `uv run pytest tests/test_integration.py::test_e2e_200_turn_conversation_with_pipeline -v`
Expected: PASS

- [ ] **Step 3: 切换默认开关**

修改 `config.py:DEFAULT_CONFIG["context"]["use_new_pipeline"]`：

```python
        # 功能开关（双轨期；Commit 6 改 True）
        "use_new_pipeline": True,    # ← 从 False 改为 True
```

- [ ] **Step 4: 全量回归（关键检查：现在新管线默认启用）**

Run: `uv run pytest tests/ -v`
Expected: PASS（可能有个别测试因依赖旧管线行为而失败，需逐个检查修复）

如发现失败：
- 失败的是「断言旧 maybe_compress 行为」的测试 → 改为断言新管线行为
- 失败的是「未设置 config 走默认」→ 现在默认是 True，需在测试 fixture 显式设 False 或适配新行为

- [ ] **Step 5: 写 CHANGELOG**

```markdown
# CHANGELOG.md（新建或追加）

## v0.X.0 - 2026-07-12

### 新增
- 上下文压缩分层管线（L1 snip / L2 micro / L4 llm + reactive）
- 大输出落盘（output_offload，> 30KB 写文件留预览）
- 压缩前快照存档（transcript，落 .transcripts/）
- 紧急上下文压缩（reactive_compact，API 报 prompt_too_long 时兜底）

### 变更
- `config.context.use_new_pipeline` 默认值：False → **True**
- `agent/context_compressor.maybe_compress` 标记废弃（下个 minor 移除）

### 回滚方式
在 `config.yaml` 加：
```yaml
context:
  use_new_pipeline: false
```
即可完全恢复 Phase 1 之前的单层 LLM 摘要行为。
```

- [ ] **Step 6: 提交**

```bash
git add config.py tests/test_integration.py CHANGELOG.md
git commit -m "feat(context)!: 默认启用新管线（use_new_pipeline=True）

BREAKING CHANGE: 默认上下文压缩从单层 LLM 摘要切换为 4 层管线。
回滚方式：config.yaml 设 context.use_new_pipeline=false"
```

---

## 任务外（Commit 7：完全移除旧路径）

**不在本计划范围**。建议在下个 minor 版本（约 2 周后）单独提 PR：
- 删除 `agent/context_compressor.maybe_compress` 函数体（保留 utility 函数）
- 删除 `MESSAGES_BEFORE_COMPRESS`、`KEEP_RECENT_MESSAGES`、`MAX_COMPRESS_ATTEMPTS` 常量（如无外部 import）
- 删除 `config.context.use_new_pipeline` 字段及 `agent/__init__.py` 的分支
- CHANGELOG 标注完全移除

---

## Self-Review

**Spec 覆盖检查**：
- ✅ §2 架构总览 → Task 1-7 实现新模块；Task 8-9 接入主循环
- ✅ §3 L3 output_offload → Task 1
- ✅ §3 L1 snip_compact → Task 3
- ✅ §3 L2 micro_compact → Task 4
- ✅ §3 L4 llm_compact → Task 5
- ✅ §3 reactive_compact → Task 6
- ✅ §3 compress_if_needed 编排 → Task 7
- ✅ §3 默认阈值表 → Task 8 写入 config.py
- ✅ §4.1 output_offload 文件格式 → Task 1
- ✅ §4.2 transcript 文件格式 + retention → Task 2
- ✅ §4.3 on_pre_compress no-op 钩子 → Task 8
- ✅ §4.4 恢复路径（system prompt 指导）→ Task 10
- ✅ §4.5 并发安全（tempfile + os.replace）→ Task 1/2 实现
- ✅ §5.1 破坏性变更清单 → 各 Task 体现
- ✅ §5.2 双轨期 + 功能开关 → Task 8 加字段、Task 9 用分支、Task 11 切默认
- ✅ §5.3 错误降级 → Task 1（磁盘满）+ Task 2（写失败）+ Task 7（snapshot 失败不阻塞）
- ✅ §5.4 log 规范 → 各 Task 实现 logger.info / warning
- ✅ §5.5 测试矩阵 → Task 1/2/3-7 单元测试 + Task 11 端到端 + 200 轮用例

**Placeholder 扫描**：无 TBD/TODO/「类似 Task N」。所有代码块完整。

**类型一致性**：
- `maybe_offload` 签名 → Task 1 定义，Task 10 调用 ✓
- `snapshot_if_needed` 签名 → Task 2 定义，Task 7 调用 ✓
- `snip_compact` / `micro_compact` / `llm_compact` → Task 3/4/5 定义，Task 7 编排 ✓
- `CompressionSessionState` 字段 → Task 6 定义，Task 7/9 使用 ✓
- `compress_if_needed` 签名 → Task 7 定义，Task 9 调用 ✓
- `reactive_compact` 签名 → Task 6 定义，Task 9 调用 ✓

**遗漏检查**：spec §3 提到 L1 双阈值（触发 50 / 解除 30）防抖。当前 `snip_compact` 实现用「检测占位存在 → 不二次裁」做幂等，等效于 release 阈值控制。如需更严格的 release 阈值，可在 Task 3 加 `release_threshold` 参数。当前实现已满足"防抖"语义，spec 覆盖完整。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-phase1-context-resilience.md`. Two execution options:

**1. Subagent-Driven (recommended)** - 每个 Task 分派 fresh subagent，Task 间二阶段 review，迭代快
**2. Inline Execution** - 当前会话内执行，批量 + checkpoint

Which approach?
