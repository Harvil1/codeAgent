"""会话移交（handoff——把当前对话打包存档，换个会话甚至换台电脑再原样恢复）。

这个文件实现"打包箱"：把当前会话完整装进一个自包含的 JSON 文件（bundle，
一个独立完整的包裹），可以导出成文件拷到别的机器再导入，恢复出一样的对话。

在项目里的位置：被 cli.py 的 /handoff 命令和 agent/cross_project.py（跨项目
恢复）调用；落盘复用 agent/atomic_io 的原子写入，密钥扫描复用
agent/secret_scanner。

存放在哪：
- ~/.codeAgent/.handoff/<bundle_id>.json    正常的 bundle
- ~/.codeAgent/.handoff/.archive/<id>.json  软删除（移进来但没真删，可翻回来）

bundle 的字段格式详见 docs/superpowers/specs/2026-07-17-handoff-design.md
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
    """所有 handoff 相关错误的共同祖先，方便调用方一次 except 捕获全部。"""


class BundleNotFoundError(HandoffError):
    """找不到指定的 bundle（ID 不存在或前缀/序号没匹配上）。"""


class BundleCorruptedError(HandoffError):
    """bundle 文件坏了：JSON 解析失败，或 format_version（文件格式版本号）是本代码不认识的。"""


class BundleTooLargeError(HandoffError):
    """bundle 体积超过 MAX_BUNDLE_SIZE_BYTES（10MB 上限），不许保存。"""


class SecretDetectedError(HandoffError):
    """对话记录（transcript——完整消息列表）里扫出了疑似密钥/密码的文本，默认拒绝保存，防止把钥匙打包带走。"""

    def __init__(self, message: str, matches: List[Dict[str, Any]]):
        super().__init__(message)
        self.matches = matches


class AmbiguousBundleIDError(HandoffError):
    """用户给的前缀太短，同时匹配到多个 bundle，不知道指哪个。"""

    def __init__(self, message: str, candidates: List[str]):
        super().__init__(message)
        self.candidates = candidates


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HandoffBundle:
    """从磁盘加载后的完整 bundle 内存对象。

    和 HandoffBundleMeta 的区别：这个含全部字段（含整份 transcript），load 时用。
    """
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
    """bundle 的"目录卡片"（元信息）：只有标题、条数这些摘要，不含整份对话内容。

    为什么要有它：列表展示（/handoff list）时不需要把几十万字的 transcript
    也读进内存，只要一张卡片就够了。
    """
    bundle_id: str
    created_at: datetime
    title: Optional[str]
    message_count: int
    handoff_state: str
    file_size: int
    # 向后兼容字段（旧 bundle 没这两项，就取默认值 None/False）
    source_cwd: Optional[str] = None
    auto_saved: bool = False


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

# 进程级单调计数器。为什么需要：Windows 的时钟精度只有约 16ms，连续保存两个
# bundle 时时间戳可能一模一样，排序就会抖动（先保存的排后面）。所以发现时间
# 没往前走时，就用计数器在毫秒位上加一点，保证"后保存的时间戳一定更大"。
_last_ts_ms: int = 0
_collision_counter: int = 0


def _monotonic_ts_ms() -> int:
    """拿到一个只增不减的毫秒时间戳；真实时间撞在同一毫秒内时用计数器补位区分。"""
    global _last_ts_ms, _collision_counter
    now_ms = int(datetime.now().timestamp() * 1000)
    if now_ms <= _last_ts_ms:
        _collision_counter += 1
        return _last_ts_ms + _collision_counter
    _last_ts_ms = now_ms
    _collision_counter = 0
    return now_ms


def _generate_id() -> str:
    """生成一个"按时间排序自然有序"的唯一 ID：毫秒时间戳 + 8 位随机短码。

    为什么这么做：ID 前缀就是保存时间，文件名排序 = 时间排序；做法和
    memory_store.py 保持同一套模式。
    """
    ts = _monotonic_ts_ms()
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}{short_uuid}"


def _now_iso() -> str:
    """生成带毫秒的 ISO8601 时间字符串（UTC，结尾 Z）。

    和 _generate_id 用同一个单调时间戳源——这样 ID 和创建时间永远一致，
    不会出现"ID 比 created_at 大"的怪事。
    """
    ts_ms = _monotonic_ts_ms()
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{(ts_ms % 1000):03d}Z"


def _parse_iso(s: str) -> datetime:
    """把 ISO8601 时间字符串解析成 datetime 对象。

    参数：
        s：时间字符串。
    解析工作交给 agent.utils.parse_iso；字符串格式不对时不抛异常，返回当前
    UTC 时间兜底（宁可时间不准也别让整个加载失败）。
    """
    from agent.utils import parse_iso
    return parse_iso(s, failure_factory=lambda: datetime.now(timezone.utc))


def _compute_checksum(transcript: List[dict]) -> str:
    """给对话记录算一个"指纹"（SHA256 哈希），用于检测文件是否被改过/损坏。

    参数：
        transcript：消息列表。
    先按固定规则序列化（键排序、紧凑分隔符，保证同样内容每次算出的字符串一样），
    再对字节做 SHA256。
    """
    transcript_bytes = json.dumps(
        transcript,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(transcript_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 密钥扫描（识别规则在公共扫描器 agent/secret_scanner）
# ---------------------------------------------------------------------------

# 密钥识别规则统一在 agent/secret_scanner（覆盖
# gitleaks 扩展：github-pat/aws/google/slack/jwt/anthropic 等密钥形态）。
# 这里只留一个向后兼容的别名，别的模块老代码 import 这个名字还能用。
from agent.secret_scanner import SECRET_RULES_RE as SECRET_PATTERN  # noqa: F401


def _scan_for_secrets(transcript: List[dict]) -> List[Dict[str, Any]]:
    """在对话记录里逐条消息找疑似密钥的文本，返回命中清单。

    参数：
        transcript：消息列表（ [{"role": ..., "content": ...}, ...] ）。
    实际识别走公共扫描器 agent/secret_scanner；这里只是把
    结果包装成 transcript 视角——标明命中发生在第几条消息、什么角色。
    """
    from agent.secret_scanner import scan_text
    matches: List[Dict[str, Any]] = []
    for idx, msg in enumerate(transcript):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for hit in scan_text(content):
            matches.append({
                "message_index": idx,
                "role": msg.get("role", "?"),
                "pattern": f"<{hit['rule']}>",
                "snippet": hit["snippet"],  # 只留截断片段：报命中时别把整把"钥匙"再打印一遍
            })
    return matches


# ---------------------------------------------------------------------------
# HandoffStore
# ---------------------------------------------------------------------------

class HandoffStore:
    """会话移交的"仓库管理员"：负责 bundle 的保存/加载/列表/删除/导入导出。

    所有方法都是同步阻塞的（调用时要等磁盘操作做完）。所有文件读写强制
    UTF-8 编码（Windows 默认编码不是 UTF-8，不指定会乱码）。
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
        # 跨项目恢复相关：
        source_cwd: Optional[str] = None,
        auto_saved: bool = False,
    ) -> str:
        """把当前会话打包成一个 bundle 文件写进磁盘，返回新 bundle 的 ID。

        参数：
            transcript：完整的消息历史（要打包带走的核心内容）。
            source_session_id：来源会话的 ID（可空，方便追溯是哪次会话存的）。
            model：用的什么模型，如 {"main": "deepseek-chat"}。
            title：bundle 标题，列表展示用（可空）。
            memory_pointers：关联的记忆条目指针列表（默认空）。
            skill_states：技能状态快照（默认空 dict）。
            todo_state：当时的待办事项状态（可空）。
            task_pointers：关联的持久化任务指针（默认空）。
            notes：用户备注（可空）。
            source_platform：来源平台标识，默认 "cli"。
            allow_secrets：设 True 才允许跳过密钥扫描强行保存；默认 False
                （扫出密钥直接拒收，防止把 API 钥匙打包带走）。
            source_cwd：来源项目目录（跨项目恢复时用）。
            auto_saved：是否系统自动保存的标记（区别于用户手动 /handoff save）。

        返回：
            新生成的 bundle_id（字符串）。
        """
        # 先过密钥扫描这道安检门，除非用户显式说 allow_secrets=True
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
            # 跨项目恢复相关：
            "source_cwd": source_cwd,
            "auto_saved": auto_saved,
        }

        # 先序列化成字符串量一下体积再写盘——超 10MB 直接拒收，别写一半才发现超大
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
        """按 ID（支持前缀/序号）把 bundle 从磁盘读回来，做成内存对象。

        参数：
            bundle_id_or_index：bundle 的完整 ID、足够长的前缀，或列表序号。

        返回：
            HandoffBundle：加载好的完整 bundle。

        会做两道校验：format_version 不认识就抛 BundleCorruptedError；
        checksum（内容指纹）对不上只打警告不抛——文件可能被改过/损坏，但
        内容还能读，交给调用方自己判断要不要用。
        """
        bundle_id = self._resolve_id(bundle_id_or_index)
        bundle_path = self._handoff_dir / f"{bundle_id}.json"
        if not bundle_path.exists():
            raise BundleNotFoundError(f"未找到 bundle: {bundle_id}")

        try:
            data = json.loads(bundle_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise BundleCorruptedError(f"JSON 解析失败: {e}") from e

        # 格式版本不认识就拒收（老代码读新格式可能丢字段）
        if data.get("format_version") != self.SUPPORTED_FORMAT_VERSION:
            raise BundleCorruptedError(
                f"不支持的 format_version: {data.get('format_version')}"
            )

        # 指纹对不上说明内容被动过——但只警告不拦截（宁可用，别一坏就全废）
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
        """列出所有还在服役的 bundle 的"目录卡片"，最新保存的排最前。

        返回：
            HandoffBundleMeta 列表（只有摘要信息，不含对话正文）。
        已软删除到 .archive/ 的不算；单个文件坏了只跳过它，不影响列表其他项。
        """
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
                    # 兼容字段：旧 bundle 文件里没有，取默认值即可
                    source_cwd=data.get("source_cwd"),
                    auto_saved=data.get("auto_saved", False),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("跳过损坏的 bundle %s: %s", path, e)
                continue
        # 倒序：新的在前。因为 _now_iso 单调递增，不存在时间相同的并列，排序稳定
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def resolve_id(self, query: str) -> str:
        """把用户输入的"完整 ID / ID 前缀 / 列表序号"翻译成完整 bundle_id。

        参数：
            query：用户给的标识（完整 ID、至少 4 字符的前缀、或序号数字）。

        返回：
            完整的 bundle_id 字符串。
        这是给外部调用的公开接口，内部逻辑在 _resolve_id。
        """
        return self._resolve_id(query)

    def _resolve_id(self, query: str) -> str:
        """resolve_id 的实际实现：按"完整 ID → 前缀 → 序号"的顺序逐步解析。"""
        # 0. 先给输入消毒：含路径分隔符或 .. 的一律拒收，防止拿它拼出
        #    越界路径（比如 "../../etc/passwd" 这种花活）
        if not query or "/" in query or "\\" in query or ".." in query:
            raise BundleNotFoundError(
                f"无效 bundle 标识: {query!r}（含路径分隔符或 .. ）"
            )

        # 1. 如果本来就是完整 ID（文件存在），直接用
        candidate = self._handoff_dir / f"{query}.json"
        if candidate.exists():
            return query

        # 2. 当作前缀匹配（至少 4 个字符）。为什么在前缀要在序号之前判断：
        #    bundle ID 以时间戳开头，很容易出现纯数字前缀，先查序号会把它
        #    误当成列表下标
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

        # 3. 短的纯数字串：当作列表序号（第几条）。只有短的 query 才会走到这里
        if query.isdigit():
            metas = self.list_bundles()
            idx = int(query)
            if 0 <= idx < len(metas):
                return metas[idx].bundle_id
            raise BundleNotFoundError(
                f"序号 {idx} 超出范围（共 {len(metas)} 个 bundle）"
            )

        # 4. 前缀太短又没匹配上：提示用户前缀至少要 4 个字符
        raise BundleNotFoundError(f"未找到 bundle: {query}（前缀至少 4 字符）")

    def delete(self, bundle_id: str) -> Path:
        """"删除"一个 bundle：其实只是把它搬进 .archive/ 目录（软删除，可翻回来）。

        参数：
            bundle_id：完整 ID / 前缀 / 序号均可。

        返回：
            归档后的文件路径。
        这是项目"完全可逆"铁律的体现——永不真删，只挪窝。
        """
        full_id = self._resolve_id(bundle_id)
        src = self._handoff_dir / f"{full_id}.json"
        if not src.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        archive_dir = self._handoff_dir / ".archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / f"{full_id}.json"
        src.replace(dest)  # 原子改名：要么整个搬成功，要么没动，不会出现半搬状态
        logger.info("bundle %s 已软删除到 %s", full_id, dest)
        return dest

    # ------------------------------------------------------------------
    # export / import / mark_completed
    # ------------------------------------------------------------------

    def export_to(self, bundle_id: str, dest_path: Path) -> Path:
        """把 bundle 复制一份到用户指定的任意路径（比如 U 盘，拿去别的机器导入）。

        参数：
            bundle_id：完整 ID / 前缀 / 序号均可。
            dest_path：目标文件路径（父目录不存在会自动创建）。

        返回：
            导出文件的路径。bundle_id 原样保留，方便导入后对得上号。
        """
        full_id = self._resolve_id(bundle_id)
        src = self._handoff_dir / f"{full_id}.json"
        if not src.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())  # 按字节原样复制：绕开文本编码转换，文件不会有半个字符的差别
        logger.info("bundle %s 已导出到 %s", full_id, dest)
        return dest

    def import_from(self, src_path: Path) -> str:
        """从外部文件（比如别的机器导出的）把 bundle 收进本机仓库。

        参数：
            src_path：外部 bundle 文件路径。

        返回：
            落库后的 bundle_id。

        规则：
        - 先校验：JSON 必须能解析、format_version 必须认识，否则当坏件拒收
        - 如果文件里的 bundle_id 在本机已存在，就重新生成一个新 ID（内容保留，
          避免覆盖本地已有的同名 bundle）
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
        # 同 ID 撞车就换个新 ID，别覆盖本地已有的
        if (self._handoff_dir / f"{existing_id}.json").exists():
            new_id = _generate_id()
            data["bundle_id"] = new_id
        else:
            new_id = existing_id

        # 缺指纹的补算一份，保证落库的 bundle 校验字段齐全
        if "schema_checksum" not in data or not data.get("schema_checksum"):
            data["schema_checksum"] = _compute_checksum(data.get("transcript", []))
        # 注意：换了新 ID 也不改 created_at——保留原始创建时间，用户能看出这是从别处搬来的旧会话

        bundle_path = self._handoff_dir / f"{new_id}.json"
        atomic_write_text(bundle_path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("bundle 已导入: %s (源: %s)", new_id, src)
        return new_id

    def mark_completed(self, bundle_id: str) -> None:
        """给 bundle 盖"已完成移交"的章：把状态字段 handoff_state 改成 'completed'。

        参数：
            bundle_id：完整 ID / 前缀 / 序号均可。

        返回：
            无。用在恢复完成后打标，列表里一眼能看出哪些 bundle 还没处理。
        """
        full_id = self._resolve_id(bundle_id)
        path = self._handoff_dir / f"{full_id}.json"
        if not path.exists():
            raise BundleNotFoundError(f"未找到 bundle: {full_id}")

        data = json.loads(path.read_text(encoding="utf-8"))
        data["handoff_state"] = "completed"
        atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("bundle %s 标记为 completed", full_id)
