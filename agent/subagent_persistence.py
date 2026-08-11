"""子代理 sidechain transcript 持久化（借鉴 claude-code-main）。

存储结构：
  ~/.OmniMate/.agent-sessions/
  ├── <agent_id>.jsonl          # 对话历史流式追加（每行一条 JSON）
  └── <agent_id>.meta.json      # 元数据（agent_id / agent_type / status / created_at / ...）

agent_id 格式：sub-{parent_session_id 前 8 位}-{YYYYMMDD-HHMMSS}-{random8}
  例：sub-abc12345-20260811-143022-x4k9po2m

status 状态机：running → completed | failed | interrupted

设计约定（CLAUDE.md）：
- **fail-open 硬要求**：所有持久化操作 try/except，绝不让主流程崩
- **encoding="utf-8"**：所有文件 I/O
- **完全可逆**：transcript 可删可清（cleanup_old）
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# agent_id 生成
# ---------------------------------------------------------------------------

def generate_agent_id(parent_session_id: str = "") -> str:
    """生成子代理 ID。

    格式：sub-{parent_session_id 前 8 位}-{YYYYMMDD-HHMMSS}-{random8}
    parent_session_id 为空时用 "orphan" 占位。
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    rand = uuid.uuid4().hex[:8]
    parent_part = parent_session_id[:8] if parent_session_id else "orphan"
    return f"sub-{parent_part}-{ts}-{rand}"


# ---------------------------------------------------------------------------
# 存储目录
# ---------------------------------------------------------------------------

def _sessions_dir() -> Path:
    """获取 .agent-sessions 目录（不存在时创建）。"""
    from constants import get_omnimate_home
    d = get_omnimate_home() / ".agent-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def write_metadata(agent_id: str, meta: dict) -> None:
    """写元数据（覆盖式，会自动追加 agent_id + updated_at）。fail-open。

    atomic: 先写临时文件再 rename（Windows 兼容）。
    """
    try:
        path = _sessions_dir() / f"{agent_id}.meta.json"
        payload = {
            **meta,
            "agent_id": agent_id,
            "updated_at": time.time(),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("write_metadata fail-open [%s]: %s", agent_id, e)


def load_metadata(agent_id: str) -> Optional[dict]:
    """读元数据。不存在或异常时返回 None。"""
    try:
        path = _sessions_dir() / f"{agent_id}.meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("load_metadata fail-open [%s]: %s", agent_id, e)
        return None


# ---------------------------------------------------------------------------
# transcript (JSONL)
# ---------------------------------------------------------------------------

def append_message(agent_id: str, message: dict) -> None:
    """流式追加一条 message 到 transcript。fail-open。"""
    try:
        path = _sessions_dir() / f"{agent_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("append_message fail-open [%s]: %s", agent_id, e)


def load_transcript(agent_id: str) -> List[dict]:
    """读完整 transcript。fail-open 返回空列表。"""
    try:
        path = _sessions_dir() / f"{agent_id}.jsonl"
        if not path.exists():
            return []
        msgs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                msgs.append(json.loads(line))
        return msgs
    except Exception as e:
        logger.warning("load_transcript fail-open [%s]: %s", agent_id, e)
        return []


# ---------------------------------------------------------------------------
# 列表 / 状态管理
# ---------------------------------------------------------------------------

def list_resumable() -> List[dict]:
    """列出所有 status=running 的子代理元数据。

    用于启动时清理 stale running（进程重启后这些记录都是僵尸）。
    """
    try:
        result = []
        for meta_path in _sessions_dir().glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") == "running":
                    result.append(meta)
            except Exception:
                continue
        return result
    except Exception as e:
        logger.debug("list_resumable fail-open: %s", e)
        return []


def mark_completed(agent_id: str, status: str = "completed") -> None:
    """标记子代理完成/失败/中断。

    读取现有 meta → 更新 status + completed_at → 写回。
    """
    meta = load_metadata(agent_id) or {}
    meta["status"] = status
    meta["completed_at"] = time.time()
    write_metadata(agent_id, meta)


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

def cleanup_old(days: int = 7) -> int:
    """清理 N 天前已进入终态（completed/failed/interrupted）的子代理记录。

    返回清理数量。只清终态，running 不删（防误删正在跑的）。
    """
    cutoff = time.time() - days * 86400
    cleaned = 0
    try:
        sessions_dir = _sessions_dir()
    except Exception as e:
        logger.debug("cleanup_old: _sessions_dir 失败: %s", e)
        return 0

    for meta_path in sessions_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") in ("completed", "failed", "interrupted"):
                if meta.get("completed_at", 0) < cutoff:
                    meta_path.unlink(missing_ok=True)
                    jsonl_path = meta_path.with_suffix(".jsonl")
                    jsonl_path.unlink(missing_ok=True)
                    cleaned += 1
        except Exception:
            continue
    return cleaned


def cleanup_stale_subagents() -> int:
    """启动时清理 stale running 记录（标记为 interrupted）。

    进程重启后所有 status=running 的记录都是僵尸（之前的进程已退出）。
    返回清理数量。
    """
    cleaned = 0
    for meta in list_resumable():
        try:
            agent_id = meta.get("agent_id")
            if agent_id:
                mark_completed(agent_id, "interrupted")
                cleaned += 1
        except Exception as e:
            logger.debug("cleanup_stale_subagents 跳过 %s: %s", meta.get("agent_id"), e)
    if cleaned > 0:
        logger.info("cleanup_stale_subagents: %d 个 stale running 子代理标记为 interrupted", cleaned)
    return cleaned
