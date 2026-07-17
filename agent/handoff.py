"""会话移交 bundle：把当前会话打包成自包含 JSON，可跨机导入导出。

存储：
- ~/.agent/.handoff/<bundle_id>.json    活跃 bundle
- ~/.agent/.handoff/.archive/<id>.json   软删除

格式详见 docs/superpowers/specs/2026-07-17-handoff-design.md
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------

class HandoffError(Exception):
    """所有 handoff 错误的基类。"""


class BundleNotFoundError(HandoffError):
    """bundle_id 不存在。"""


class BundleCorruptedError(HandoffError):
    """format_version 不支持 或 JSON 解析失败。"""


class BundleTooLargeError(HandoffError):
    """bundle 超过 MAX_BUNDLE_SIZE_BYTES。"""


class SecretDetectedError(HandoffError):
    """transcript 含密钥模式（默认拒绝保存）。"""

    def __init__(self, message: str, matches: List[Dict[str, Any]]):
        super().__init__(message)
        self.matches = matches


class AmbiguousBundleIDError(HandoffError):
    """ULID 前缀匹配多个 bundle。"""

    def __init__(self, message: str, candidates: List[str]):
        super().__init__(message)
        self.candidates = candidates


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HandoffBundle:
    """完整 bundle（加载后的内存表示）。"""
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


@dataclass
class HandoffBundleMeta:
    """bundle 元信息（list 用，不含 transcript）。"""
    bundle_id: str
    created_at: datetime
    title: Optional[str]
    message_count: int
    handoff_state: str
    file_size: int


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _generate_id() -> str:
    """生成时间排序的唯一 ID（沿用 memory_store.py 的模式）。"""
    ts = int(datetime.now().timestamp() * 1000)
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}{short_uuid}"


def _now_iso() -> str:
    """ISO8601 with milliseconds + Z。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") \
        + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _parse_iso(s: str) -> datetime:
    """解析 ISO8601（容错：处理带/不带 Z、毫秒）。"""
    s2 = s.rstrip("Z")
    try:
        return datetime.fromisoformat(s2)
    except ValueError:
        # 退而求其次
        return datetime.utcnow()


def _compute_checksum(transcript: List[dict]) -> str:
    """对 transcript 稳定序列化后计算 SHA256。"""
    transcript_bytes = json.dumps(
        transcript,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(transcript_bytes).hexdigest()


def _atomic_write(path: Path, json_str: str) -> None:
    """先写 .tmp 再 rename，避免半写文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json_str, encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# HandoffStore
# ---------------------------------------------------------------------------

class HandoffStore:
    """会话移交存储。

    所有方法同步阻塞。文件 I/O 强制 UTF-8。
    """

    MAX_BUNDLE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
    SUPPORTED_FORMAT_VERSION = "1"

    def __init__(self, handoff_dir: Path):
        self._handoff_dir = Path(handoff_dir)
        self._handoff_dir.mkdir(parents=True, exist_ok=True)
        (self._handoff_dir / ".archive").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        transcript: List[dict],
        source_session_id: Optional[str],
        model: Dict[str, str],
        title: Optional[str] = None,
        memory_pointers: Optional[List[str]] = None,
        skill_states: Optional[Dict[str, Any]] = None,
        todo_state: Optional[Dict[str, Any]] = None,
        task_pointers: Optional[List[str]] = None,
        notes: Optional[str] = None,
        source_platform: str = "cli",
        allow_secrets: bool = False,
    ) -> str:
        """生成 bundle 写入磁盘，返回 bundle_id。"""
        # 密钥扫描（Task 3 会真正实现，此处先 stub 通过）
        # 大小检查（Task 3 会真正实现）
        bundle_id = _generate_id()
        created_at = _now_iso()
        checksum = _compute_checksum(transcript)

        bundle_dict = {
            "format_version": self.SUPPORTED_FORMAT_VERSION,
            "bundle_id": bundle_id,
            "created_at": created_at,
            "title": title,
            "source_session_id": source_session_id,
            "source_platform": source_platform,
            "model": model,
            "transcript": transcript,
            "memory_pointers": memory_pointers or [],
            "skill_states": skill_states or {},
            "todo_state": todo_state,
            "task_pointers": task_pointers or [],
            "handoff_state": "pending",
            "notes": notes,
            "schema_checksum": checksum,
        }

        # 大小预检（基于序列化后字节数）
        json_str = json.dumps(bundle_dict, ensure_ascii=False, indent=2)
        if len(json_str.encode("utf-8")) > self.MAX_BUNDLE_SIZE_BYTES:
            raise BundleTooLargeError(
                f"bundle 过大（{len(json_str.encode('utf-8'))} bytes），"
                f"上限 {self.MAX_BUNDLE_SIZE_BYTES} bytes"
            )

        bundle_path = self._handoff_dir / f"{bundle_id}.json"
        _atomic_write(bundle_path, json_str)
        logger.info("handoff bundle 已保存: %s", bundle_id)
        return bundle_id

    def load(self, bundle_id_or_index: str) -> HandoffBundle:
        """加载 bundle。校验 format_version 和 checksum（不匹配警告不抛）。"""
        bundle_id = self._resolve_id(bundle_id_or_index)
        bundle_path = self._handoff_dir / f"{bundle_id}.json"
        if not bundle_path.exists():
            raise BundleNotFoundError(f"未找到 bundle: {bundle_id}")

        try:
            data = json.loads(bundle_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BundleCorruptedError(f"JSON 解析失败: {e}") from e

        # format_version 校验
        if data.get("format_version") != self.SUPPORTED_FORMAT_VERSION:
            raise BundleCorruptedError(
                f"不支持的 format_version: {data.get('format_version')}"
            )

        # checksum 校验（警告不抛）
        expected = data.get("schema_checksum", "")
        actual = _compute_checksum(data.get("transcript", []))
        if expected != actual:
            logger.warning(
                "bundle %s checksum 不匹配（expected=%s, actual=%s），"
                "文件可能损坏",
                bundle_id, expected, actual,
            )

        return HandoffBundle(
            format_version=data["format_version"],
            bundle_id=data["bundle_id"],
            created_at=_parse_iso(data["created_at"]),
            title=data.get("title"),
            source_session_id=data.get("source_session_id"),
            source_platform=data.get("source_platform", "cli"),
            model=data.get("model", {}),
            transcript=data.get("transcript", []),
            memory_pointers=data.get("memory_pointers", []),
            skill_states=data.get("skill_states", {}),
            todo_state=data.get("todo_state"),
            task_pointers=data.get("task_pointers", []),
            handoff_state=data.get("handoff_state", "pending"),
            notes=data.get("notes"),
            schema_checksum=data.get("schema_checksum", ""),
        )

    def _resolve_id(self, query: str) -> str:
        """内部：把 query 解析为完整 bundle_id（不做歧义处理，Task 2 会扩展）。

        Task 1 简化版：
        - 完整 ID（文件存在）→ 直接返回
        - 否则抛 BundleNotFoundError（Task 2 加前缀/index）
        """
        candidate = self._handoff_dir / f"{query}.json"
        if candidate.exists():
            return query
        raise BundleNotFoundError(f"未找到 bundle: {query}")
