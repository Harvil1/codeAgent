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
