"""会话移交 bundle：把当前会话打包成自包含 JSON，可跨机导入导出。

存储：
- ~/.OmniMate/.handoff/<bundle_id>.json    活跃 bundle
- ~/.OmniMate/.handoff/.archive/<id>.json   软删除

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

from agent.atomic_io import atomic_write_text

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

# 进程级单调计数器：Windows 时钟分辨率约 16ms，连续保存时 _generate_id 与
# _now_iso 可能产生相同时间戳，导致 list_bundles 排序不稳定。当检测到时间戳
# 未前进时，自增计数器附加到毫秒位，保证后保存的 bundle 严格大于先保存的。
_last_ts_ms: int = 0
_collision_counter: int = 0


def _monotonic_ts_ms() -> int:
    """返回单调递增的毫秒时间戳（同一真实毫秒内用计数器兜底）。"""
    global _last_ts_ms, _collision_counter
    now_ms = int(datetime.now().timestamp() * 1000)
    if now_ms <= _last_ts_ms:
        _collision_counter += 1
        return _last_ts_ms + _collision_counter
    _last_ts_ms = now_ms
    _collision_counter = 0
    return now_ms


def _generate_id() -> str:
    """生成时间排序的唯一 ID（沿用 memory_store.py 的模式）。"""
    ts = _monotonic_ts_ms()
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}{short_uuid}"


def _now_iso() -> str:
    """ISO8601 with milliseconds + Z。与 _generate_id 共享单调时间戳源。"""
    ts_ms = _monotonic_ts_ms()
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{(ts_ms % 1000):03d}Z"


def _parse_iso(s: str) -> datetime:
    """解析 ISO8601(委托给 agent.utils.parse_iso,失败时返回当前 UTC 时间)。"""
    from agent.utils import parse_iso
    return parse_iso(s, failure_factory=lambda: datetime.now(timezone.utc))


def _compute_checksum(transcript: List[dict]) -> str:
    """对 transcript 稳定序列化后计算 SHA256。"""
    transcript_bytes = json.dumps(
        transcript,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(transcript_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 密钥扫描（Task 3）
# ---------------------------------------------------------------------------

# 合并为单个正则 + 命名捕获组：一次扫描即可覆盖全部 5 类密钥模式，
# 性能从 5×N 次降到 N 次（N = 消息数 × 内容长度）。
# 命名组同时让错误信息更可读（"命中：<openai>" 而不是裸正则）。
SECRET_PATTERN = re.compile(
    r"(?P<openai>sk-[A-Za-z0-9_\-]{20,})"
    r"|(?P<bearer>Bearer\s+[A-Za-z0-9_\-\.]{20,})"
    r"|(?P<api_key>api_key[\"\s:=]+[\"']?[A-Za-z0-9]{16,})"
    r"|(?P<token>token[\"\s:=]+[\"']?[A-Za-z0-9]{16,})"
    r"|(?P<pem>-----BEGIN [A-Z ]+PRIVATE KEY-----)"
)


def _scan_for_secrets(transcript: List[dict]) -> List[Dict[str, Any]]:
    """扫描 transcript 找密钥模式。返回命中列表。

    用合并正则一次扫描；命中后通过命名组反查模式类型。
    """
    matches: List[Dict[str, Any]] = []
    for idx, msg in enumerate(transcript):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for m in SECRET_PATTERN.finditer(content):
            # 反查命中的命名组（key→value 非空的那组）
            kind = next(
                (k for k, v in m.groupdict().items() if v),
                "unknown",
            )
            matches.append({
                "message_index": idx,
                "role": msg.get("role", "?"),
                "pattern": f"<{kind}>",
                "snippet": m.group(0)[:50],  # 截断防再次暴露
            })
    return matches


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
        # 密钥扫描（除非显式 allow_secrets）
        if not allow_secrets:
            matches = _scan_for_secrets(transcript)
            if matches:
                raise SecretDetectedError(
                    f"检测到 {len(matches)} 处疑似密钥，拒绝保存。"
                    f"命中：{matches[0]['pattern']} @ msg#{matches[0]['message_index']}",
                    matches=matches,
                )

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
        atomic_write_text(bundle_path, json_str)
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

    # ------------------------------------------------------------------
    # list / resolve / delete
    # ------------------------------------------------------------------

    def list_bundles(self) -> List[HandoffBundleMeta]:
        """按 created_at 倒序返回活跃 bundle（不含 .archive/）。"""
        metas: List[HandoffBundleMeta] = []
        for path in self._handoff_dir.glob("*.json"):
            if path.parent.name == ".archive":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metas.append(HandoffBundleMeta(
                    bundle_id=data["bundle_id"],
                    created_at=_parse_iso(data["created_at"]),
                    title=data.get("title"),
                    message_count=len(data.get("transcript", [])),
                    handoff_state=data.get("handoff_state", "pending"),
                    file_size=path.stat().st_size,
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("跳过损坏的 bundle %s: %s", path, e)
                continue
        # 倒序：created_at 大的在前（_now_iso 单调递增，无 tie）
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def resolve_id(self, query: str) -> str:
        """把 ULID/前缀/序号 解析为完整 bundle_id（public 接口）。"""
        return self._resolve_id(query)

    def _resolve_id(self, query: str) -> str:
        """内部：ULID 前缀（≥4 字符）或 list 序号 解析为完整 bundle_id。"""
        # 0. 路径消毒：拒绝含分隔符或 .. 的 query（防 glob/path 注入）
        if not query or "/" in query or "\\" in query or ".." in query:
            raise BundleNotFoundError(
                f"无效 bundle 标识: {query!r}（含路径分隔符或 .. ）"
            )

        # 1. 完整 ID 直接命中
        candidate = self._handoff_dir / f"{query}.json"
        if candidate.exists():
            return query

        # 2. 前缀匹配（≥4 字符）——先于序号判断，避免纯数字前缀（如
        #    时间戳部分）被误当成 list index
        if len(query) >= 4:
            matches = [
                p.stem for p in self._handoff_dir.glob(f"{query}*.json")
                if p.parent.name != ".archive"
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise AmbiguousBundleIDError(
                    f"前缀 '{query}' 匹配多个 bundle: {matches}",
                    candidates=matches,
                )
            raise BundleNotFoundError(f"未找到 bundle: {query}")

        # 3. 纯数字：当作 list 序号（仅短 query 走到此分支）
        if query.isdigit():
            metas = self.list_bundles()
            idx = int(query)
            if 0 <= idx < len(metas):
                return metas[idx].bundle_id
            raise BundleNotFoundError(
                f"序号 {idx} 超出范围（共 {len(metas)} 个 bundle）"
            )

        # 4. 前缀太短且未命中
        raise BundleNotFoundError(f"未找到 bundle: {query}（前缀至少 4 字符）")

    def delete(self, bundle_id: str) -> Path:
        """软删除：移到 .archive/<bundle_id>.json。返回归档路径。"""
        full_id = self._resolve_id(bundle_id)
        src = self._handoff_dir / f"{full_id}.json"
        if not src.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        archive_dir = self._handoff_dir / ".archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / f"{full_id}.json"
        src.replace(dest)  # 原子 rename
        logger.info("bundle %s 已软删除到 %s", full_id, dest)
        return dest

    # ------------------------------------------------------------------
    # export / import / mark_completed
    # ------------------------------------------------------------------

    def export_to(self, bundle_id: str, dest_path: Path) -> Path:
        """拷贝 bundle 到任意路径（不改 bundle_id）。"""
        full_id = self._resolve_id(bundle_id)
        src = self._handoff_dir / f"{full_id}.json"
        if not src.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())  # 二进制拷贝避免编码问题
        logger.info("bundle %s 已导出到 %s", full_id, dest)
        return dest

    def import_from(self, src_path: Path) -> str:
        """从外部路径导入 bundle。返回 bundle_id。

        - 如果 bundle_id 已存在，生成新 ULID（其他字段保留）
        - 校验 format_version + JSON 完整性
        """
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"文件不存在: {src}")

        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BundleCorruptedError(f"JSON 解析失败: {e}") from e

        if data.get("format_version") != self.SUPPORTED_FORMAT_VERSION:
            raise BundleCorruptedError(
                f"不支持的 format_version: {data.get('format_version')}"
            )

        existing_id = data.get("bundle_id", "")
        # 如果 ID 已被占用，重新生成
        if (self._handoff_dir / f"{existing_id}.json").exists():
            new_id = _generate_id()
            data["bundle_id"] = new_id
        else:
            new_id = existing_id

        # 写入（带 checksum 重算，确保一致）
        if "schema_checksum" not in data or not data.get("schema_checksum"):
            data["schema_checksum"] = _compute_checksum(data.get("transcript", []))
        # 如果重新生成了 ID，需要重写 created_at？不，保留原 created_at 让用户知道源时间。

        bundle_path = self._handoff_dir / f"{new_id}.json"
        atomic_write_text(bundle_path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("bundle 已导入: %s (源: %s)", new_id, src)
        return new_id

    def mark_completed(self, bundle_id: str) -> None:
        """把 handoff_state 改为 'completed'。"""
        full_id = self._resolve_id(bundle_id)
        path = self._handoff_dir / f"{full_id}.json"
        if not path.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        data = json.loads(path.read_text(encoding="utf-8"))
        data["handoff_state"] = "completed"
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("bundle %s 标记为 completed", full_id)
