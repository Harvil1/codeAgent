"""多文件记忆存储：记忆（AI 对用户和项目沉淀下来的事实条目，跨会话保留）按主题分文件存放。

存储结构：
- ~/.codeAgent/.memory/{topic}.jsonl：一个主题一个文件，文件里每行一条记忆（JSON）
- ~/.codeAgent/MEMORY.md：总目录/索引（自动生成，按主题分组，超 200 行或 25KB 截断）

几条核心规则：
- 写入即维护：往同一主题写同名（name 相同）的记忆是更新旧条目，不是无限堆积
- 会话内注入走检索式临时注入（memory_injection + memory_retriever，
  每轮挑最相关的 5 条放进一次性的 user 消息，不进 system prompt）；
  snapshot_for_prompt() 的截断索引只是没有 aux_llm（辅助小模型）时的兜底，
  检索器用的是 full_index_text() 的完整版
- 旧格式（每条记忆一个 .md 文件）在启动时自动搬家到主题 jsonl
- 删除是软删除：挪到 .archive/memory-{时间戳}/，随时可找回
"""
import json
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from agent.atomic_io import atomic_write_text

# 索引快照的截断上限（最多 200 行或 25KB，哪个先超按哪个截）
_INDEX_MAX_LINES = 200
_INDEX_MAX_BYTES = 25000

logger = logging.getLogger(__name__)

VALID_TYPES = {"user", "feedback", "project", "reference", "other"}

# 分区设计：project/reference 两类按项目分家存放，
# user/feedback 等其余类型全局共享——项目经验不串门，用户偏好处处生效
_PROJECT_TYPES = ("project", "reference")


def _is_project_type(mtype: str) -> bool:
    """判断这条记忆是否属于"按项目分区"的类型（project/reference 存项目专区，其余存全局区）。

    参数：
    - mtype：记忆类型字符串
    返回：属于项目分区类型返回 True，否则 False。
    """
    return mtype in _PROJECT_TYPES


@dataclass
class MemoryEntry:
    """一条记忆的数据结构：一条跨会话保留的事实（名字、内容、类型、时间、置信度等字段）。"""
    id: str  # 对外的完整 id，格式是 {主题}#{短id}
    name: str
    description: str
    type: str
    body: str
    created_at: datetime
    updated_at: datetime
    # 主题：这条记忆归哪个话题，对应 .memory/{topic}.jsonl 文件
    topic: str = "general"
    # L1 摘要层——给记忆配的一句摘要
    summary: str = ""
    confidence: float = 1.0
    expected_valid_days: int = 365
    source_session_id: str = ""
    state: str = "active"
    last_reviewed_at: str = ""


def _generate_id() -> str:
    """造一个唯一短 id（毫秒时间戳 + 6 位随机后缀），在主题文件内区分每条记忆。

    时间戳开头让它天然按时间排序，随机后缀保证同一毫秒也不重号。
    """
    ts = int(datetime.now().timestamp() * 1000)
    short_uuid = uuid.uuid4().hex[:6]
    return f"{ts}{short_uuid}"


def _now_iso() -> str:
    # 必须用 UTC——本地时间和 curator 那边的 UTC 比较会差 8 小时
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_entry_id(entry_id: str) -> Tuple[str, str]:
    """把对外 id（{topic}#{uid}）拆成 (主题, 短id) 两段。id 里没有 # 时兜底按 general 主题处理。"""
    if "#" in entry_id:
        topic, _, uid = entry_id.rpartition("#")
        return topic or "general", uid
    return "general", entry_id


def _parse_frontmatter(text: str) -> tuple[Optional[dict], str]:
    """解析 `---\\n...yaml...\\n---\\n正文` 这种带 YAML 头的文本（老格式搬家时用）。

    参数：
    - text：文件全文
    返回：(YAML 头字典, 正文) 二元组；格式不对或 YAML 解析失败返回 (None, 原文)。
    """
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
    """把字典序列化成 frontmatter 文本（`---\\n...yaml...\\n---\\n`）。

    和 `_parse_frontmatter` 是一对（一个拆一个装）。主要给测试用：造老格式的
    .md 文件来测"老格式搬家到 jsonl"的逻辑，另外任何要写 frontmatter 的
    工具脚本也能用。meta 是空字典时返回空串（不输出 frontmatter）。
    """
    if not meta:
        return ""
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n"


def validate_memory_dir(memory_dir, codeagent_home) -> Optional[str]:
    """memory 目录的安全校验。

    memory 目录的位置来自配置/环境变量，可能被指到敏感位置（比如 ~/.ssh）
    ——记忆系统会因此拿到敏感目录的写权限，必须先把关。

    返回：通过返回 None；不通过返回中文的拒绝原因。两条规则：
    1. **受保护路径拒绝**：memory_dir（解析软链后的真实路径）落在 ~/.ssh、
       /etc、C:\\Windows 这类敏感位置下——挡住"改 AGENT_HOME 环境变量或
       memory_dir 配置指向敏感位置换取写权限"的路
    2. **软链逃逸检测**：memory_dir 解析成真实路径后必须仍在 codeagent_home
       的真实路径之下——挡住"home/.memory 是个指向 ~/.ssh 的软链"这类把戏

    说明：CodeAgent 的配置只从用户级 settings.json 读（项目配置没有注入面），
    来源校验天然满足，这里只补路径内容校验。

    参数：
    - memory_dir：待校验的 memory 目录
    - codeagent_home：agent 的家目录（~/.codeAgent）
    """
    try:
        memory_dir = Path(memory_dir)
        codeagent_home = Path(codeagent_home)
    except TypeError:
        return "路径类型异常"

    # 规则 1：受保护路径（写入路径和解析后的真实路径都查——哪怕名字上只是指向软链也算命中）
    from agent.permission import is_protected_path
    prot = is_protected_path(memory_dir)
    if prot:
        return f"落在受保护路径（{prot}）"
    try:
        real_mem = memory_dir.resolve()
    except (OSError, RuntimeError) as e:
        return f"realpath 解析失败: {e}"
    prot2 = is_protected_path(real_mem)
    if prot2:
        return f"realpath 落在受保护路径（{prot2}）——符号链接逃逸"

    # 规则 2：解析后的真实路径必须还在 home 的真实路径之下
    try:
        real_home = codeagent_home.resolve()
        real_mem.relative_to(real_home)
    except (ValueError, OSError, RuntimeError):
        return (
            f"realpath {real_mem} 不在 agent home {real_home} 之下"
            "（符号链接逃逸或配置越界）"
        )
    return None


class MemoryStore:
    """按主题分文件的记忆存储管家：读写、缓存、索引、软删除都从这走。"""

    def __init__(self, *, codeagent_home: Path, memory_dir=None):
        """初始化存储：校验目录安全、建目录、搬家老格式、生成索引。

        参数：
        - codeagent_home：agent 家目录（~/.codeAgent）
        - memory_dir：记忆目录，不传就用 home 下的 .memory/
        """
        self._home = Path(codeagent_home)
        self._memory_dir = Path(memory_dir) if memory_dir else self._home / ".memory"
        # 先做目录安全校验（受保护路径 + 软链逃逸）。
        # 宁可启动失败也绝不悄悄写到敏感位置（fail-closed）。
        violation = validate_memory_dir(self._memory_dir, self._home)
        if violation:
            raise ValueError(f"memory 目录安全校验失败: {violation}")
        self._index_path = self._home / "MEMORY.md"
        self._lock = threading.Lock()
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._cached_snapshot: str = ""
        # 主题文件的内存缓存（只在锁内访问）：读的时候不用每次都读硬盘；
        # 更新/删除这种低频操作才整文件重写，新建直接往文件尾追加（最快）。
        # 缓存靠文件修改时间失效：别的实例（比如 curator 自建的）改了文件会自动重读。
        self._rows_cache: dict = {}  # topic -> (mtime_at_load, rows)
        # 惰性索引重建：写入时只做"索引脏了"标记，真要读索引时才重建
        # （每存一条就全量重扫重写的话，批量存 n 条是 n² 开销）。
        # 不改变语义：记忆本来就要等下次会话才注入（保护 prompt cache 的设计），
        # 下次会话构造时会调 build_index_text 强制刷新索引。
        self._index_dirty = False
        # 记住上次重建索引用的项目键，
        # _ensure_index_fresh 靠它发现"切换了项目"——没有新写入也要重建索引。
        # 初始为 None，第一次 _rebuild_index() 会填上实际值。
        self._index_built_key: Optional[str] = None
        # 启动时把老格式（一条记忆一个 .md）搬家到主题 jsonl
        self._migrate_legacy_if_any()
        self._rebuild_index()

    # ------------------------------------------------------------------
    # 主题文件读写（分区机制：全局区/项目区）
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(zone_dir: Optional[Path], topic: str) -> tuple:
        """算 _rows_cache 缓存的键——必须把分区（zone）编进去，否则全局区和项目区会互相串数据。

        zone_dir 为 None 表示全局区，否则是项目区目录。键是 (zone_str, topic)
        二元组：zone_str 用空串代表全局区，否则用目录的字符串形式。

        参数：
        - zone_dir：分区目录（None=全局区）
        - topic：主题名
        """
        return (str(zone_dir) if zone_dir else "", topic)

    def _zone_base_dir(self, zone_dir: Optional[Path]) -> Path:
        """zone_dir 为 None 时返回全局 .memory/ 目录，否则原样返回项目区目录。

        调用方自己负责保证项目区目录存在（项目区是首次写入时才按需创建的）。

        参数：
        - zone_dir：分区目录（None=全局区）
        """
        return Path(zone_dir) if zone_dir else self._memory_dir

    def _topic_path(self, topic: str, *, zone_dir: Optional[Path] = None) -> Path:
        """算某个主题的 jsonl 文件路径（主题名会先清洗成安全文件名）。

        zone_dir 为 None 走全局 .memory/（默认），否则走 zone_dir 指向的项目区。

        参数：
        - topic：主题名
        - zone_dir：分区目录（None=全局区）
        """
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", (topic or "general"))
        return self._zone_base_dir(zone_dir) / f"{safe}.jsonl"

    def _topic_mtime(self, topic: str, *, zone_dir: Optional[Path] = None) -> tuple:
        """拿主题文件的 (修改时间, 文件大小)，当缓存的"版本号"用；文件不存在返回 (-1.0, -1)。

        为什么要两个一起看：Windows 上修改时间精度只有约 15 毫秒，刚写完马上又写
        可能时间没变，单看时间会误判"缓存还有效"，加上文件大小就稳了。

        参数：
        - topic：主题名
        - zone_dir：分区目录（None=全局区）
        """
        try:
            st = self._topic_path(topic, zone_dir=zone_dir).stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return (-1.0, -1)

    def _read_topic_rows(
        self, topic: str, *, zone_dir: Optional[Path] = None,
    ) -> List[dict]:
        """读一个主题文件的全部行（带内存缓存；缓存失效才真读一次盘）。

        返回的是缓存里那个 list 本身——调用方（在锁内）直接改它，改动会同步
        反映到缓存，这是故意的（save/update 就是就地改 rows 再整体写回）。
        别的实例改了文件（修改时间变了）会自动重新读盘。

        参数：
        - topic：主题名
        - zone_dir：分区目录（None=全局区，否则读指定项目区）
        """
        key = self._cache_key(zone_dir, topic)
        mtime = self._topic_mtime(topic, zone_dir=zone_dir)
        cached = self._rows_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        path = self._topic_path(topic, zone_dir=zone_dir)
        rows = []
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("topic 文件 %s 有损坏行，跳过", path)
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            except Exception as e:
                logger.warning("读 topic 文件失败 %s: %s", path, e)
        self._rows_cache[key] = (mtime, rows)
        return rows

    def _write_topic_rows(
        self, topic: str, rows: List[dict], *, zone_dir: Optional[Path] = None,
    ) -> None:
        """把整个主题文件原子重写（JSONL 格式），并同步内存缓存。更新/删除路径用。

        参数：
        - topic：主题名
        - rows：要写回的全部行（dict 列表）
        - zone_dir：分区目录（None=全局区）
        """
        # 项目区按需创建（首次写入时才建目录，和全局区在 __init__ 就建好不一样）
        if zone_dir is not None:
            zone_dir.mkdir(parents=True, exist_ok=True)
        path = self._topic_path(topic, zone_dir=zone_dir)
        text = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in rows
        ) + ("\n" if rows else "")
        atomic_write_text(path, text)
        key = self._cache_key(zone_dir, topic)
        self._rows_cache[key] = (self._topic_mtime(topic, zone_dir=zone_dir), rows)

    def _append_topic_row(
        self, topic: str, row: dict, *, zone_dir: Optional[Path] = None,
    ) -> None:
        """新建记忆的专用快路径：缓存里追加 + 文件末尾追加，不用重写整个文件。

        参数：
        - topic：主题名
        - row：新记忆那一行（dict）
        - zone_dir：分区目录（None=全局区）
        """
        if zone_dir is not None:
            zone_dir.mkdir(parents=True, exist_ok=True)
        rows = self._read_topic_rows(topic, zone_dir=zone_dir)  # 先读一次，保证缓存已加载
        rows.append(row)
        path = self._topic_path(topic, zone_dir=zone_dir)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("append topic 行失败 %s: %s", path, e)
        key = self._cache_key(zone_dir, topic)
        self._rows_cache[key] = (self._topic_mtime(topic, zone_dir=zone_dir), rows)

    def _find_row(
        self, topic: str, uid: str, *, zone_dir: Optional[Path] = None,
    ) -> Optional[dict]:
        """在主题文件里按短 id 找那一行。找不到返回 None。

        参数：
        - topic：主题名
        - uid：短 id（不带主题前缀）
        - zone_dir：分区目录（None=全局区）
        """
        for r in self._read_topic_rows(topic, zone_dir=zone_dir):
            if r.get("id") == uid:
                return r
        return None

    # ------------------------------------------------------------------
    # 按类型路由到分区 + 跨分区查找条目
    # ------------------------------------------------------------------

    def _resolve_zone(self, mtype: str) -> Optional[Path]:
        """按记忆类型决定写到哪个分区。

        project/reference → 项目区目录（按当前工作目录动态算出来）。
        user/feedback/other → None（全局区）。

        每次调用都现算项目键（不缓存在实例字段里），这样子代理通过 contextvars
        切了工作目录的场景也能算对。

        参数：
        - mtype：记忆类型
        返回：项目区目录 Path 或 None（全局区）。
        """
        if not _is_project_type(mtype):
            return None
        # 函数内 import，防止模块加载时互相依赖成环（实际上没有反向依赖，保险起见）
        from agent.project_scope import get_project_memory_dir
        return get_project_memory_dir(self._home)

    def _current_project_zone(self) -> Optional[Path]:
        """返回当前工作目录对应的项目区目录（不管类型，单纯要查项目区时用）。

        get/update/delete 跨区查找时用——项目区归哪个项目，由当前工作目录说了算。
        """
        from agent.project_scope import get_project_memory_dir
        return get_project_memory_dir(self._home)

    def _current_project_key_safe(self) -> Optional[str]:
        """算当前的项目键，给 _ensure_index_fresh 用来比较"是不是切了项目"。

        出错时返回 None 当作没有项目区（fail-open：这里坏了不该挡住主流程）。
        """
        try:
            from agent.project_scope import get_project_memory_key
            return get_project_memory_key()
        except Exception:
            return None

    def _locate_entry(self, topic: str, uid: str) -> Optional[tuple]:
        """在全局区和当前项目区里找指定条目。

        返回 (所在分区目录, 该主题全部行, 命中的行, 命中行的下标)：
        - 分区目录为 None 表示全局区，Path 表示项目区
        - 全部行拿来就地改完后整体写回
        找不到返回 None。查找顺序：先全局区，再项目区。

        参数：
        - topic：主题名
        - uid：短 id
        """
        # 1. 先查全局区
        rows_global = self._read_topic_rows(topic, zone_dir=None)
        for i, r in enumerate(rows_global):
            if r.get("id") == uid:
                return (None, rows_global, r, i)
        # 2. 再查当前项目区
        zone = self._current_project_zone()
        if zone is not None:
            rows_proj = self._read_topic_rows(topic, zone_dir=zone)
            for i, r in enumerate(rows_proj):
                if r.get("id") == uid:
                    return (zone, rows_proj, r, i)
        return None

    def _row_to_entry(
        self, topic: str, row: dict,
        *, zone_dir: Optional[Path] = None,
    ) -> MemoryEntry:
        """把文件里的一行（dict）转成 MemoryEntry 对象。

        zone_dir 记下这条记忆住在哪个分区（None=全局区），用来生成正确的索引链接。
        注意：链接路径必须区分全局区（.memory/）和项目区
        （.memory/projects/<项目键>/），否则项目条目的链接会指错地方。

        参数：
        - topic：主题名
        - row：文件里的一行（dict）
        - zone_dir：这条记忆所在分区（None=全局区）
        """
        entry = MemoryEntry(
            id=f"{topic}#{row.get('id', '')}",
            name=row.get("name", ""),
            description=row.get("description", ""),
            type=row.get("type", "other"),
            body=row.get("body", ""),
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
            topic=topic,
            summary=row.get("summary", "") or "",
            confidence=float(row.get("confidence", 1.0) or 1.0),
            expected_valid_days=int(row.get("expected_valid_days", 365) or 365),
            source_session_id=row.get("source_session_id", "") or "",
            state=row.get("state", "active") or "active",
            last_reviewed_at=row.get("last_reviewed_at", "") or "",
        )
        # 顺手带上分区信息（不做成 MemoryEntry 正式字段——临时属性，只有 _rebuild_index 用）
        entry._zone_dir = zone_dir  # type: ignore[attr-defined]
        return entry

    def _entry_link(self, entry: MemoryEntry) -> str:
        """生成 MEMORY.md 索引里那条 markdown 链接的路径。

        全局区：.memory/{topic}.jsonl#{uid}
        项目区：.memory/projects/{项目键}/{topic}.jsonl#{uid}
        """
        topic = entry.topic
        uid = entry.id.split("#")[-1]
        zone_dir = getattr(entry, "_zone_dir", None)
        if zone_dir is None:
            return f".memory/{topic}.jsonl#{uid}"
        # 项目区：从分区目录名里取出项目键（最后一级目录名）
        try:
            proj_key = zone_dir.name
            return f".memory/projects/{proj_key}/{topic}.jsonl#{uid}"
        except Exception:
            return f".memory/{topic}.jsonl#{uid}"

    def _scan_all_entries(self) -> List[MemoryEntry]:
        """扫全局区 + 当前项目区的所有主题文件，解析出全部记忆。

        list_all 和 _rebuild_index 用——把两个区合在一起。每条记忆带上 _zone_dir
        标记来自哪个区，_rebuild_index 靠它分区出章节、生成正确链接。
        当前项目区由工作目录决定（子代理场景也算得对）。
        """
        entries = []
        # 1. 先扫全局区
        for path in sorted(self._memory_dir.glob("*.jsonl")):
            topic = path.stem
            for row in self._read_topic_rows(topic, zone_dir=None):
                try:
                    entries.append(self._row_to_entry(topic, row, zone_dir=None))
                except (ValueError, TypeError) as e:
                    logger.warning("memory 行解析失败 %s: %s", path, e)
        # 2. 再扫当前项目区（如果存在）
        zone = self._current_project_zone()
        if zone is not None and zone.exists():
            for path in sorted(zone.glob("*.jsonl")):
                topic = path.stem
                for row in self._read_topic_rows(topic, zone_dir=zone):
                    try:
                        entries.append(
                            self._row_to_entry(topic, row, zone_dir=zone)
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning("memory 行解析失败 %s: %s", path, e)
        return entries

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """重建 MEMORY.md 索引：全局记忆 + 当前项目记忆，两个分区各成一节。

        排序规则没变（类型优先级 → 置信度 → 更新时间倒序），但各分区内自己排，
        再各自按主题分组。最终长这样：
            # Memory Index
            （头部说明）

            ## 全局记忆
            ### 主题：{topic}
            - [name](.memory/{topic}.jsonl#{uid}) — desc

            ## 当前项目记忆（{project_key}）
            ### 主题：{topic}
            - [name](.memory/projects/{key}/{topic}.jsonl#{uid}) — desc

        项目区是空的（没有 project 类条目）就不输出"当前项目记忆"这一节。
        """
        type_priority = {"feedback": 0, "user": 1, "project": 2, "reference": 3, "other": 4}
        entries = self._scan_all_entries()
        entries = [e for e in entries if e.state != "archived"]
        # 三层排序：连排三次稳定排序，最后按类型优先级定大局
        entries.sort(key=lambda e: str(e.updated_at), reverse=True)
        entries.sort(key=lambda e: e.confidence, reverse=True)
        entries.sort(key=lambda e: type_priority.get(e.type, 99))

        # 按分区拆开：没带 _zone_dir 的是全局区，带的是项目区
        global_entries = [e for e in entries if getattr(e, "_zone_dir", None) is None]
        proj_entries = [e for e in entries if getattr(e, "_zone_dir", None) is not None]

        lines = [
            "# Memory Index",
            "",
            "自动生成，请勿手动编辑。⭐ 表示 feedback 类(用户纠正过的),永远优先显示。",
            "全局区记忆跨项目共享；项目区记忆仅当前项目可见。",
            "",
        ]

        def _emit_section(
            section_title: str, section_entries: List[MemoryEntry],
        ) -> None:
            """把一组条目按主题分组，写进 lines（内部小工具）。"""
            if not section_entries:
                return
            lines.append(section_title)
            lines.append("")
            by_topic: Dict[str, list] = {}
            for e in section_entries:
                by_topic.setdefault(e.topic, []).append(e)
            for topic in sorted(by_topic.keys()):
                lines.append(f"### 主题：{topic}")
                for e in by_topic[topic]:
                    marker = "⭐ " if e.type == "feedback" else ""
                    link = self._entry_link(e)
                    if e.summary:
                        lines.append(
                            f"- {marker}[{e.name}]({link}) — "
                            f"{e.description} | 摘要：{e.summary}"
                        )
                    else:
                        lines.append(
                            f"- {marker}[{e.name}]({link}) — {e.description}"
                        )
                lines.append("")

        # 全局节（始终输出这一节，即使为空——顶部说明总得有）
        _emit_section("## 全局记忆", global_entries)

        # 项目节（项目区有条目时才输出这一节）
        if proj_entries:
            from agent.project_scope import get_project_memory_key
            proj_key = get_project_memory_key()
            _emit_section(
                f"## 当前项目记忆（{proj_key}）", proj_entries,
            )

        atomic_write_text(self._index_path, "\n".join(lines) + "\n")
        # 缓存 snapshot 时去掉头部说明（标题、空行、说明文字不算正文）：
        # 找到第一个 "## " 开头的行、从那里开始截
        # （不按固定行数跳——头部行数变了会错位）。
        head_end = 0
        for i, ln in enumerate(lines):
            if ln.startswith("## "):
                head_end = i
                break
        self._cached_snapshot = "\n".join(lines[head_end:]) if head_end > 0 else ""
        self._index_dirty = False
        # 记下本次重建索引用的项目键，
        # _ensure_index_fresh 靠它发现"切换了项目"——没有新写入也要重建索引
        self._index_built_key = self._current_project_key_safe()

    def snapshot_for_prompt(self) -> str:
        """返回截断版索引（最多 200 行 / 25KB，哪个先超按哪个截）。

        用途：system prompt 常驻注入（prompt_builder 会话级拼一次）——
        让主对话模型"知道已经有什么记忆"；详情仍走每轮检索式注入
        （见 memory_injection.py）。
        """
        self._ensure_index_fresh()
        snap = self._cached_snapshot
        lines = snap.splitlines()
        if len(lines) > _INDEX_MAX_LINES:
            snap = "\n".join(lines[:_INDEX_MAX_LINES]) + (
                "\n... [记忆索引超出行数上限，其余按需检索]"
            )
        if len(snap) > _INDEX_MAX_BYTES:
            snap = snap[:_INDEX_MAX_BYTES] + "\n... [记忆索引超出字节上限，其余按需检索]"
        return snap

    def full_index_text(self) -> str:
        """返回完整索引（给 memory_retriever 按需检索用，不受注入截断的影响）。"""
        self._ensure_index_fresh()
        return self._cached_snapshot

    def full_index_text_with_age(self) -> str:
        """带"年龄"标注的检索索引（防止召回太久没更新的旧记忆）。

        在 full_index_text 的基础上给每行附上 `[age: N天]`（最后更新距今天数），
        喂给辅助模型检索时，配合提示词里"新记忆优先"的规则做年龄打折扣。
        纯粹是派生文本：MEMORY.md 落盘格式和存储都不变。出错时原样返回（fail-open）。
        """
        base = self.full_index_text()
        if not base:
            return ""
        now = datetime.now(timezone.utc)
        link_age: dict = {}
        for e in self._scan_all_entries():
            if e.state == "archived":
                continue
            try:
                updated = e.updated_at
                # 不带时区的历史时间戳先补上 UTC 再比，避免带/不带时区的时间相减报错
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                days = max(0, (now - updated).days)
            except Exception:
                days = None
            link_age[self._entry_link(e)] = days
        try:
            from agent.memory_retriever import annotate_index_with_age
            return annotate_index_with_age(base, link_age)
        except Exception as e:
            logger.warning("索引年龄标注失败（fail-open 返回原索引）: %s", e)
            return base

    def _mark_index_dirty(self) -> None:
        """写路径调用：给索引盖个"待重建"的章（不马上重建，防止批量写入时反复全量重建、开销翻着倍涨）。"""
        self._index_dirty = True

    def _ensure_index_fresh(self) -> None:
        """读路径调用：发现索引"脏了"或切换了项目目录才重建（惰性）。线程安全（拿锁）。

        除了脏标记，还要比对当前项目键（存在 self._index_built_key，重建时记下），
        键变了也要重建。场景：同一个实例把工作目录切到另一个项目（没有新写入），
        snapshot 也应该显示新项目的记忆。
        """
        current_key = self._current_project_key_safe()
        if self._index_dirty or current_key != self._index_built_key:
            with self._lock:
                # 双重检查：拿到锁后再确认一次（防止多个线程抢着重复重建）
                current_key = self._current_project_key_safe()
                if self._index_dirty or current_key != self._index_built_key:
                    self._rebuild_index()

    def build_index_text(self) -> str:
        """强制重建索引并返回截断版（启动时用）。"""
        with self._lock:
            self._rebuild_index()
        return self.snapshot_for_prompt()

    # ------------------------------------------------------------------
    # 公开：读
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """按完整 id 取一条记忆。找不到返回 None。

        参数：
        - memory_id：完整 id（{主题}#{短id}）
        """
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            located = self._locate_entry(topic, uid)
            if located is None:
                return None
            zone_dir, _rows, target, _i = located
            return self._row_to_entry(topic, target)

    def load_body(self, memory_id: str) -> Optional[str]:
        """按 id 取记忆的正文 body。找不到返回 None。

        参数：
        - memory_id：完整 id
        """
        entry = self.get(memory_id)
        return entry.body if entry else None

    def list_all(self) -> List[MemoryEntry]:
        """列出全部记忆（全局区 + 当前项目区合在一起）。"""
        with self._lock:
            return self._scan_all_entries()

    def find_by_topic_name(
        self, topic: str, name: str, *, type: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """同一主题里按 name 查重（"写入即维护"的查重入口）。

        注意：给了 type 时只查 save(type=...) 实际会写进去的
        那个分区——和 save 的单区查重口径一致（上层靠它判断"这次是更新还是
        新建"，跨区查会让标签谎报）。type 不给时跨区查
        （先全局后当前项目区）。跨区同名允许共存（物理隔离是特性：
        项目区的条目别的项目看不见）。

        参数：
        - topic：主题名
        - name：记忆名
        - type：记忆类型（可选，给了就只查目标分区）
        返回：命中的记忆，没有则 None。
        """
        if not name:
            return None
        topic = topic or "general"
        with self._lock:
            if type is not None:
                zone = self._resolve_zone(type)
                for row in self._read_topic_rows(topic, zone_dir=zone):
                    if row.get("name") == name and row.get("state", "active") != "archived":
                        return self._row_to_entry(topic, row, zone_dir=zone)
                return None
            # 跨区查找：先全局区，再当前项目区
            for row in self._read_topic_rows(topic, zone_dir=None):
                if row.get("name") == name and row.get("state", "active") != "archived":
                    return self._row_to_entry(topic, row)
            pz = self._current_project_zone()
            if pz is not None:
                for row in self._read_topic_rows(topic, zone_dir=pz):
                    if row.get("name") == name and row.get("state", "active") != "archived":
                        return self._row_to_entry(topic, row, zone_dir=pz)
            return None

    # ------------------------------------------------------------------
    # 公开：写
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        name: str,
        description: str,
        type: str,
        body: str = "",
        summary: str = "",
        confidence: float = 1.0,
        expected_valid_days: int = 365,
        source_session_id: str = "",
        topic: str = "general",
    ) -> str:
        """创建或更新一条记忆（写入即维护：同主题同名 → 更新旧条目）。返回记忆 id。

        按类型路由分区：
        - user/feedback/other → 全局区（~/.codeAgent/.memory/）
        - project/reference → 项目区（~/.codeAgent/.memory/projects/<项目键>/）

        查重只查目标分区——不同分区的同名记忆算不同条目，这是分层隔离的语义，
        不能误判成重复。

        参数：
        - name：记忆名（必填）
        - description：一句话描述（必填）
        - type：类型（user/feedback/project/reference/other 之一）
        - body：正文
        - summary：摘要
        - confidence：置信度（0-1）
        - expected_valid_days：预期多少天内有效（超龄会被 curator 标旧/归档）
        - source_session_id：来源会话 id
        - topic：主题（默认 general）
        """
        if not name or not description:
            raise ValueError("name 和 description 必需")
        if type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一，实际: {type}")
        # 写入前先扫秘密——疑似密钥直接拒写（秘密不该进记忆库；只报规则 ID 不回显内容）
        from agent.secret_scanner import find_secrets_in
        secret_hits = find_secrets_in(name, description, summary, body)
        if secret_hits:
            raise ValueError(
                f"记忆内容疑似含密钥（规则: {secret_hits[0]['rule']}），拒绝写入"
            )
        # 注意：save 里自己内联做查重/更新，避免嵌套拿锁（threading.Lock 不可重入，重复拿会自己锁死自己）
        with self._lock:
            # 按类型路由到对应分区
            zone_dir = self._resolve_zone(type)
            rows = self._read_topic_rows(topic, zone_dir=zone_dir)
            existing = next(
                (r for r in rows
                 if r.get("name") == name and r.get("state", "active") != "archived"),
                None,
            )
            if existing is not None:
                # 写入即维护：同主题同名 → 更新而不是新建
                existing.update({
                    "description": description, "type": type, "body": body,
                    "summary": summary, "confidence": confidence,
                    "expected_valid_days": expected_valid_days,
                    "source_session_id": source_session_id,
                    "updated_at": _now_iso(),
                })
                self._write_topic_rows(topic, rows, zone_dir=zone_dir)
                self._mark_index_dirty()
                return f"{topic}#{existing['id']}"

            # 新建也必须用 _now_iso()（UTC）——新建用
            # 本地时间、更新用 UTC 混着来，东八区新记忆的年龄会虚大 8 小时
            # （curator 的年龄判定跟着一起错）
            now = _now_iso()
            uid = _generate_id()
            row = {
                "id": uid, "name": name, "description": description,
                "type": type, "body": body, "summary": summary,
                "created_at": now,
                "updated_at": now,
                "confidence": confidence,
                "expected_valid_days": expected_valid_days,
                "source_session_id": source_session_id,
                "state": "active",
            }
            self._append_topic_row(topic, row, zone_dir=zone_dir)
            self._mark_index_dirty()
            return f"{topic}#{uid}"

    def update(
        self,
        memory_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
        summary: Optional[str] = None,
        confidence: Optional[float] = None,
        expected_valid_days: Optional[int] = None,
        source_session_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> MemoryEntry:
        """更新一条记忆的部分字段。id 不存在抛 KeyError。

        跨分区查找条目（先全局后项目区），但条目留在原分区——即使改了 type
        字段也写回原分区，不重新路由（想跨区搬家要 delete + save）。

        state 字段（"active"/"stale"，归档走 delete）：
        **只改 state 不刷新 updated_at**——curator 的年龄判定按内容年龄算，
        如果标 stale 时把 updated_at 刷新了，年龄归零会把这条 stale 记忆
        误判成新内容，在 stale 和 active 之间反复翻转（死循环）。

        参数：
        - memory_id：完整 id（必填）
        - name/description/type/body/summary：内容字段，传 None 表示不改
        - confidence/expected_valid_days/source_session_id：元信息字段，None 不改
        - state：状态（active/stale），None 不改
        """
        if type is not None and type not in VALID_TYPES:
            raise ValueError(f"type 必须是 {VALID_TYPES} 之一")
        if state is not None and state not in ("active", "stale"):
            raise ValueError("state 只接受 active/stale（archived 走 delete）")
        # 秘密扫描只查新传入的值，命中拒绝更新
        from agent.secret_scanner import find_secrets_in
        secret_hits = find_secrets_in(name, description, summary, body)
        if secret_hits:
            raise ValueError(
                f"更新内容疑似含密钥（规则: {secret_hits[0]['rule']}），拒绝写入"
            )
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            located = self._locate_entry(topic, uid)
            if located is None:
                raise KeyError(f"memory not found: {memory_id}")
            zone_dir, rows, target, _i = located
            if name is not None:
                target["name"] = name
            if description is not None:
                target["description"] = description
            if type is not None:
                target["type"] = type
            if body is not None:
                target["body"] = body
            if summary is not None:
                target["summary"] = summary
            if confidence is not None:
                target["confidence"] = confidence
            if expected_valid_days is not None:
                target["expected_valid_days"] = expected_valid_days
            if source_session_id is not None:
                target["source_session_id"] = source_session_id
            if state is not None:
                target["state"] = state
            # 只有内容字段真的变了才刷新 updated_at（只改 state 不刷新，
            # 理由见上面 docstring——curator 年龄判定按内容年龄算，防 stale↔active 反复翻转）
            content_changed = any(v is not None for v in (
                name, description, type, body, summary,
                confidence, expected_valid_days, source_session_id,
            ))
            if content_changed:
                target["updated_at"] = _now_iso()
            # 写回原分区（即使改了 type 也写回条目所在的原分区——不支持跨区搬家）
            self._write_topic_rows(topic, rows, zone_dir=zone_dir)
            self._mark_index_dirty()
            return self._row_to_entry(topic, target)

    def delete(self, memory_id: str) -> bool:
        """软删除：先把条目副本存进 .archive/，再从主题文件里移除。

        跨分区查找条目（先全局后项目区），按实际所在分区删除。

        参数：
        - memory_id：完整 id
        返回：删成功 True，找不到 False。
        """
        with self._lock:
            topic, uid = _split_entry_id(memory_id)
            located = self._locate_entry(topic, uid)
            if located is None:
                return False
            zone_dir, rows, target, _i = located
            # 软删除：先把一份副本存进归档目录
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_dir = self._home / ".archive" / f"memory-{ts}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                (archive_dir / f"{topic}-{uid}.json").write_text(
                    json.dumps(target, ensure_ascii=False), encoding="utf-8",
                )
            except Exception as e:
                logger.warning("记忆软删除存档失败: %s", e)
            # 再从主题文件里移除（写回原分区）
            rows = [r for r in rows if r.get("id") != uid]
            self._write_topic_rows(topic, rows, zone_dir=zone_dir)
            self._mark_index_dirty()
            return True

    def clear_all(self) -> int:
        """软删除全部记忆（主题文件整个挪进 .archive，可恢复）。返回删掉的条数。

        同时清全局区和当前项目区。
        """
        with self._lock:
            total = 0
            # 1. 先清全局区
            total += self._clear_zone(None)
            # 2. 再清当前项目区
            zone = self._current_project_zone()
            if zone is not None and zone.exists():
                total += self._clear_zone(zone)
            self._mark_index_dirty()
            return total

    def _clear_zone(self, zone_dir: Optional[Path]) -> int:
        """清空指定分区的全部主题文件（软删除挪到 .archive）。返回清掉的条数。

        参数：
        - zone_dir：分区目录（None=全局区）
        """
        base = self._zone_base_dir(zone_dir)
        topics = sorted(p.stem for p in base.glob("*.jsonl"))
        total = 0
        for topic in topics:
            rows = self._read_topic_rows(topic, zone_dir=zone_dir)
            total += len(rows)
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_dir = self._home / ".archive" / f"memory-{ts}"
                archive_dir.mkdir(parents=True, exist_ok=True)
                src = self._topic_path(topic, zone_dir=zone_dir)
                dst = archive_dir / f"{topic}.jsonl"
                # 防止覆盖同名的归档文件——重名就加序号
                n = 1
                while dst.exists():
                    dst = archive_dir / f"{topic}-{n}.jsonl"
                    n += 1
                shutil.move(str(src), str(dst))
            except Exception as e:
                logger.warning("清空 topic %s 失败: %s", topic, e)
                continue
        return total

    # ------------------------------------------------------------------
    # 老格式搬家（一条记忆一个 .md → 主题 jsonl）
    # ------------------------------------------------------------------

    def _migrate_legacy_if_any(self) -> None:
        """启动时把老格式的 `.memory/{id}.md`（frontmatter 单条文件）搬家到主题 jsonl。"""
        legacy_mds = [p for p in self._memory_dir.glob("*.md") if p.name != "latest.md"]
        if not legacy_mds:
            return
        logger.info("发现 %d 个旧格式记忆文件，迁移到主题组织", len(legacy_mds))
        for path in legacy_mds:
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text)
                if meta is None:
                    continue
                topic = meta.get("topic") or _infer_topic(meta.get("name", ""), meta.get("type", "other"))
                row = {
                    "id": path.stem, "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "type": meta.get("type", "other"),
                    "body": body,
                    "created_at": meta.get("created_at", _now_iso()),
                    "updated_at": meta.get("updated_at", _now_iso()),
                    "summary": meta.get("summary", "") or "",
                    "confidence": float(meta.get("confidence", 1.0) or 1.0),
                    "expected_valid_days": int(meta.get("expected_valid_days", 365) or 365),
                    "source_session_id": meta.get("source_session_id", "") or "",
                    "state": meta.get("state", "active") or "active",
                }
                rows = self._read_topic_rows(topic)
                rows.append(row)
                self._write_topic_rows(topic, rows)
                # 老文件挪去归档
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_dir = self._home / ".archive" / f"legacy-memory-{ts}"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), archive_dir / path.name)
                except Exception as e:
                    logger.warning("归档旧记忆 %s 失败: %s", path, e)
            except Exception as e:
                logger.warning("迁移旧记忆 %s 失败: %s", path, e)

    # ------------------------------------------------------------------
    # 老接口兼容（给旧调用方垫的一层）
    # ------------------------------------------------------------------

    def format_for_system_prompt(self, target: str) -> str:
        """老接口兼容：直接返回索引（target 参数已废弃不用）。"""
        return self.snapshot_for_prompt()

    def add(self, target: str, content: str) -> bool:
        """老接口兼容：相当于 save（target 当 type 用）。成功 True，失败 False。"""
        try:
            t = target if target in VALID_TYPES else "other"
            self.save(name=content[:30], description=content, type=t, body=content)
            return True
        except Exception as e:
            logger.warning("旧 add() 兼容失败: %s", e)
            return False

    def modify(self, action: str, target: str, content: str, old_content: str = "") -> bool:
        """老接口兼容：粗略映射到 save。只支持 add，其他 action 不再支持。"""
        if action == "add":
            return self.add(target, content)
        logger.warning("旧 modify(action=%s) 不再支持，请用 memory 工具新 action", action)
        return False


def _parse_dt(value) -> datetime:
    """解析 ISO 格式的时间字符串。空值/解析失败时返回当前时间。"""
    try:
        return datetime.fromisoformat(str(value or _now_iso()))
    except (ValueError, TypeError):
        return datetime.now()


def _infer_topic(name: str, type: str) -> str:
    """按 name/type 猜一个主题名（老格式搬家用，实在猜不出用 general）。"""
    if type == "feedback":
        return "feedback"
    if type == "project":
        return "project"
    first = (name or "").strip().split()[0] if (name or "").strip() else ""
    return first[:12] if first else "general"
