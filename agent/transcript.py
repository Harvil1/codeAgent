"""压缩前快照存档：在对话被压缩（摘要替换原文）之前，把完整消息原样备份到 .transcripts/，方便事后翻旧账。

上下文压缩是有损的——原始对话一旦被摘要替代就找不回来了，
所以每次压缩前先留一份全量底稿。什么时候备份由调用方决定
（默认只在 L4 LLM 摘要那一步 force=True）。
文件格式：JSONL，一行一条消息，最后一行是 _meta 元数据。
"""
import json
import logging
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
    """把当前完整 messages 落盘成 .transcripts/transcript_{时间戳}_{短uuid}.jsonl。

    参数：
        messages：要备份的完整消息列表
        agent_home：OmniMate 主目录（~/.OmniMate），备份存在它下面的 .transcripts/
        session_id：当前会话 id（写进元数据）
        force：True 才真落盘（spec 里只有 force=True 一种触发场景，False 等于关）
        enabled：总开关，False 直接返回 None
        retention：最多保留几份备份文件，超了删最旧的

    返回：
        写入的文件 Path；没写（开关关着）或写失败（只打 warning，绝不阻塞主循环）返回 None。

    落盘成功后还会顺手做两件事：更新 latest.txt 指针（指向最新一份备份，用普通
    文本文件而不是软链接——Windows 上建软链常要管理员权限）和按 retention 清理旧文件。
    """
    if not enabled or not force:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    transcripts_dir = agent_home / ".transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    target = transcripts_dir / f"transcript_{ts}_{short_uuid}.jsonl"

    try:
        # 写文件前先过 safe_path 路径安全检查——不能绕过全局路径闸门
        from agent.permission import safe_path
        perm = safe_path(target, write=True, allowed_roots=[transcripts_dir.resolve()])
        if not perm.allowed:
            raise OSError(f"safe_path 拒绝: {perm.reason}")
        _write_jsonl(target, messages, session_id=session_id, reason="pre_llm_compact")
    except OSError as e:
        logger.warning("transcript 写入失败 (%s): %s", target, e)
        return None

    _update_latest_pointer(transcripts_dir, target)
    _prune_old(transcripts_dir, retention)

    logger.info("transcript snapshot: %s (msgs=%d)", target, len(messages))
    return target


def _write_jsonl(path: Path, messages: list, *, session_id: str, reason: str) -> None:
    """把 messages 写成 JSONL 文件：一行一条消息（带序号和时间戳），末行是元数据。

    参数：
        path：目标文件路径
        messages：消息列表
        session_id：会话 id（进元数据）
        reason：这次备份的原因标记（进元数据）

    返回：无。写法是原子写（先写临时文件再替换），读者看不到半截文件。
    """
    now_iso = datetime.now().isoformat(timespec="seconds")
    lines = []
    for seq, msg in enumerate(messages):
        envelope = {"seq": seq, "ts": now_iso, **msg}
        lines.append(json.dumps(envelope, ensure_ascii=False))
    # 末行放元数据（会话 id / 原因 / 原始条数）
    lines.append(json.dumps({
        "_meta": {
            "session_id": session_id,
            "reason": reason,
            "orig_len": len(messages),
        }
    }, ensure_ascii=False))

    content = "\n".join(lines) + "\n"
    from agent.atomic_io import atomic_write_text_lite
    atomic_write_text_lite(path, content)


def _update_latest_pointer(transcripts_dir: Path, target: Path) -> None:
    """更新 latest.txt 指针，让它指向最新一份备份文件。

    为什么用普通文本文件而不用软链接（symlink）：Windows 上创建软链接
    默认需要管理员权限，普通用户会失败。

    参数：
        transcripts_dir：备份目录
        target：最新写入的备份文件

    返回：无。写失败只打 debug 日志（指针丢了不影响备份本身）。
    """
    pointer = transcripts_dir / "latest.txt"
    try:
        pointer.write_text(str(target), encoding="utf-8")
    except OSError as e:
        logger.debug("latest 指针更新失败: %s", e)


def _prune_old(transcripts_dir: Path, retention: int) -> None:
    """只保留最近 retention 份备份文件，更旧的删掉（防止备份无限堆积）。

    参数：
        transcripts_dir：备份目录
        retention：保留份数上限

    返回：无。单个文件删失败只打 debug 日志，继续删下一个。
    """
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
