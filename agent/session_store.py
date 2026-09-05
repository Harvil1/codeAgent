"""会话存储：用 JSONL 文件存聊天记录（不用数据库 SQLite）。

这个文件管的是"历史会话库"——每个会话的标题、时间、消息记录，支持
/resume 恢复、/search 搜索、fork 克隆。给 cli.py 和 agent/__init__.py
这些上层调用方用。

文件布局（好比一个档案柜）：
  ~/.codeAgent/.sessions/
  ├── index.json              # 目录卡片：所有会话的元数据列表
  ├── <session_id>.jsonl      # 每个会话一个档案袋：消息历史，每行一条 JSON
  └── <session_id>.jsonl.bak  # 删除会话时只是改名为 .bak 备份（可恢复，不真删）

设计权衡（为啥不用 SQLite 数据库）：
  - Windows 上 SQLite 的文件锁出过问题
  - 符合项目"文件优先"的哲学（memory/tasks 也都是纯文件）
  - 用户可以直接用记事本打开看/改 JSONL
  - 跨平台行为一致
  - 代价：全文搜索退化为 Python re 正则扫描（单用户小规模够用）

对外接口是通用的存取语义——cli.py / agent/__init__.py 等
调用方不依赖具体存储实现。
"""

import atexit
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 兼容老接口的存根（老函数名，别的模块可能还在 import，留着占位）
# ---------------------------------------------------------------------------

def is_fts5_available() -> bool:
    """兼容老接口：查 FTS5 全文搜索扩展是否可用——JSONL 存储用不上，恒返回 False。"""
    return False


def is_trigram_available() -> bool:
    """兼容老接口：查 trigram 索引是否可用——JSONL 存储用不上，恒返回 False。"""
    return False


def _contains_cjk(s: str) -> bool:
    """兼容老接口：判断字符串里有没有中日韩字符（原供分词用，现在留着防 import 报错）。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


# 目录卡片（index.json）去抖写盘阈值：累积 N 条或距上次写盘 T 秒才真正落盘。
# 卡片只是目录册（消息正文在 .jsonl 里永不丢），落后几条可接受；
# 每条消息都原子重写整个 index.json 在长会话里是 IO 热点。
_INDEX_FLUSH_COUNT = 50
_INDEX_FLUSH_SECONDS = 2.0

# 消息缓存最多同时驻留几个会话（LRU）：/search 一次能扫上千个会话，
# 全部解析结果常驻内存会无限涨；16 个已覆盖「当前会话 + 热浏览」需求。
_MSGS_CACHE_CAP = 16


class SessionStore:
    """会话存储管理器（JSONL 文件实现）：负责会话的增删查改、消息追加、搜索、统计、fork。

    __init__ 接受 db_path 是兼容老接口，实际当目录用：
    - 传的是文件路径（如 sessions.db）→ 自动改用它旁边的 .sessions/ 目录
    - 传的是目录路径 → 直接用

    线程安全：写 index.json 和 .jsonl 都要先拿 _lock 锁，防并发写坏文件。
    """

    def __init__(self, db_path):
        """初始化：确定存储目录、建目录、准备缓存，必要时迁移老的 SQLite 库。

        参数：
            db_path：老接口传法是 sessions.db 文件路径；实际按目录用——
            带 .db 后缀的文件路径会自动转成它所在目录下的 ".sessions"
            子目录；传目录则直接使用。

        返回：无（构造函数）。
        """
        db_path = Path(db_path)
        # 兼容：传的是 .db 文件路径时，改用同目录下的 .sessions 目录
        if db_path.suffix == ".db":
            self._sessions_dir = db_path.parent / ".sessions"
        else:
            self._sessions_dir = db_path
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._sessions_dir / "index.json"
        self._lock = threading.Lock()
        self._index_cache: Optional[List[dict]] = None  # 目录卡片的内存缓存（第一次访问时加载）
        # 消息缓存（热路径优化）：session_id -> ((mtime, size), msgs)。
        # 为什么要缓存：search/get_messages/get_stats 是热路径，每次都
        # 读盘 + JSON 解析太慢；文件一变（mtime/size 变）缓存自动失效。
        # 为什么键用 mtime+size 双因子：Windows 的 mtime 精度只有 ~15ms，
        # 同一时间窗内 append 前后 mtime 可能一样，单看 mtime 会误判
        # "文件没变"而漏读新消息。
        self._msgs_cache: "OrderedDict[str, tuple]" = OrderedDict()
        # 轮次号内存表：session_id -> max(turn_index)。旧版每条 append 都
        # 全量重读会话文件算 max(turn_index)（N 条消息 O(N²) 读盘），现在
        # 首见会话读一次文件引导、之后纯内存加法。
        self._turn_state: dict = {}
        # 目录卡片去抖记账：还有几条卡片更新没写盘 / 上次写盘的 monotonic 时间
        self._index_dirty_count = 0
        self._index_last_flush = 0.0
        # 进程退出前把没落盘的卡片更新补写进去（注册了实例方法引用，
        # store 是进程级单例，不存在提前回收问题）
        atexit.register(self.flush_index)
        # 如果发现老的 SQLite 库就自动迁移
        self._maybe_migrate_sqlite()

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    def _load_index(self) -> List[dict]:
        """（内部）读目录卡片 index.json，读一次后缓存在内存里反复用。

        返回：会话元数据的 dict 列表（文件不存在或坏了就当空列表）。
        """
        if self._index_cache is not None:
            return self._index_cache
        if not self._index_path.exists():
            self._index_cache = []
            return self._index_cache
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._index_cache = data.get("sessions", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取 sessions index 失败，重置为空: %s", e)
            self._index_cache = []
        return self._index_cache

    def _save_index(self) -> None:
        """（内部）把内存里的目录卡片写回 index.json。

        为什么用 atomic_write_text（先写临时文件再改名）：保证写一半
        断电/崩溃也不会留下半个坏文件，跨平台行为一致。
        """
        from agent.atomic_io import atomic_write_text
        if self._index_cache is None:
            return
        data = {"sessions": self._index_cache}
        atomic_write_text(
            self._index_path,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def _session_file(self, session_id: str) -> Path:
        """（内部）算出某个会话的 .jsonl 档案袋文件完整路径。

        参数：
            session_id：会话 ID

        返回：Path 对象，如 <目录>/<session_id>.jsonl
        """
        return self._sessions_dir / f"{session_id}.jsonl"

    def _read_session_msgs(self, session_id: str) -> List[dict]:
        """（内部）读某个会话 .jsonl 里的全部消息（只读，不动 index）。带 (mtime, size) 缓存。

        search/get_messages/get_stats/fork 是热路径（上万条消息全量读盘
        一次要 100ms 以上），文件没变就直接用上次解析好的结果；调用方
        拿到后只做只读遍历，不会互相污染。
        文件不存在时返回空列表；个别行 JSON 坏了就跳过那行。

        参数：
            session_id：会话 ID

        返回：原始消息 dict 列表（含 id/timestamp 等内部字段）。
        """
        path = self._session_file(session_id)
        try:
            st = path.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            return []
        cached = self._msgs_cache.get(session_id)
        if cached is not None and cached[0] == cache_key:
            self._msgs_cache.move_to_end(session_id)  # LRU：命中挪到最热端
            return cached[1]
        msgs = []
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._msgs_cache[session_id] = (cache_key, msgs)
        # LRU 淘汰：超容量就丢最冷端的会话（当前活跃会话天然在最热端）
        while len(self._msgs_cache) > _MSGS_CACHE_CAP:
            self._msgs_cache.popitem(last=False)
        return msgs

    def _compute_turn_index(self, session_id: str, role: str) -> int:
        """（内部）算新消息的轮次编号：每来一条 user 消息就开一个新轮次（一问多答算一轮）。

        旧版每次 append 都全量读+解析整个会话文件算 max(turn_index)——
        N 条消息就是 O(N²) 次读盘，长会话每条消息都把事件循环卡一下。
        现在内存递增：首见会话读一次文件引导（fork/重启后自动对齐），
        之后纯内存查表。append 成功后调用方把返回值回写进内存表。

        参数：
            session_id：会话 ID
            role：新消息的角色（"user" 开新轮，其他角色沿用当前轮）

        返回：int 轮次号（空会话的第一条 user 消息是第 1 轮）。
        """
        max_turn = self._turn_state.get(session_id)
        if max_turn is None:
            msgs = self._read_session_msgs(session_id)
            max_turn = max((m.get("turn_index", 0) for m in msgs), default=0)
            self._turn_state[session_id] = max_turn
        return max_turn + 1 if role == "user" else max_turn

    # ------------------------------------------------------------------
    # 自动迁移老的 SQLite 库
    # ------------------------------------------------------------------

    def _maybe_migrate_sqlite(self) -> None:
        """检测有没有老的 sessions.db（SQLite 格式），有且 index 为空时一次性迁移到 JSONL。

        参数：无。

        返回：无（迁移结果只写日志）。
        """
        # sessions.db 一般放在 sessions_dir 的上一级，两个位置都探一下
        candidates = [
            self._sessions_dir.parent / "sessions.db",
            self._sessions_dir / "sessions.db",
        ]
        old_db = next((p for p in candidates if p.exists()), None)
        if old_db is None:
            return
        # 已迁移过（旁边留了 .bak）→ 不重复搬
        if old_db.with_suffix(".db.bak").exists():
            return
        # 只在 index 还是空的时候迁移，避免覆盖新数据
        if self._load_index():
            return
        logger.info("检测到老 SQLite %s，开始迁移到 JSONL...", old_db)
        try:
            self._migrate_from_sqlite(old_db)
            old_db.rename(old_db.with_suffix(".db.bak"))
            logger.info("迁移完成，老 db 改名为 %s", old_db.with_suffix(".db.bak").name)
        except Exception as e:
            logger.error("迁移失败（保留 sessions.db）: %s", e)

    def _migrate_from_sqlite(self, db_path: Path) -> None:
        """真正干活：把 SQLite 库里的会话和消息逐条搬进 JSONL 文件。

        参数：
            db_path：老的 sessions.db 文件路径

        返回：无（搬完顺手更新内存 index 并写盘）。
        """
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            sessions = conn.execute(
                "SELECT id, title, created_at, updated_at, message_count, "
                "model, provider FROM sessions"
            ).fetchall()
            new_index = []
            for s in sessions:
                sid = s["id"]
                new_index.append({
                    "id": sid,
                    "title": s["title"],
                    "created_at": s["created_at"],
                    "updated_at": s["updated_at"],
                    "message_count": s["message_count"] or 0,
                    "model": s["model"],
                    "provider": s["provider"],
                })
                msgs = conn.execute(
                    "SELECT id, role, content, tool_calls, tool_call_id, name, "
                    "timestamp, turn_index FROM messages "
                    "WHERE session_id = ? ORDER BY rowid",
                    (sid,),
                ).fetchall()
                lines = []
                for m in msgs:
                    line_obj = {
                        "id": m["id"],
                        "role": m["role"],
                        "content": m["content"],
                        "tool_calls": (json.loads(m["tool_calls"])
                                       if m["tool_calls"] else None),
                        "tool_call_id": m["tool_call_id"],
                        "name": m["name"],
                        "timestamp": m["timestamp"],
                        "turn_index": m["turn_index"],
                    }
                    lines.append(json.dumps(line_obj, ensure_ascii=False))
                if lines:
                    from agent.atomic_io import atomic_write_text
                    atomic_write_text(
                        self._session_file(sid), "\n".join(lines) + "\n"
                    )
            self._index_cache = new_index
            self._save_index()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 会话生命周期（建/删/查）
    # ------------------------------------------------------------------

    def close(self) -> None:
        """兼容老接口的空操作：JSONL 存储没有连接要关，什么都不做。"""
        pass

    def __del__(self):
        # 兼容老接口：现在没有连接要关
        try:
            pass
        except Exception:
            pass

    def create_session(
        self,
        *,
        title: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """新建一个会话（登记到目录卡片 + 建空的 .jsonl 档案袋）。

        参数：
            title：会话标题，可不填
            model：创建时用的模型名，可不填
            provider：模型提供方（如 deepseek），可不填

        返回：新生成的 session_id（UUID 字符串）。
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            index = self._load_index()
            index.append({
                "id": session_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "model": model,
                "provider": provider,
            })
            self._save_index()
        # 建一个空的 .jsonl 档案袋占位
        self._session_file(session_id).touch()
        return session_id

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """往会话末尾追加一条消息（像在档案袋里再加一张纸条）——主循环每产生一条消息都落盘，崩溃/重启后才能完整恢复对话。

        参数：
            session_id：往哪个会话追加
            role：消息角色（"user"/"assistant"/"tool" 等）
            content：消息正文
            tool_calls：assistant 消息携带的工具调用列表（可不填）
            tool_call_id：tool 消息对应的调用 ID，用于配对（可不填）
            name：tool 消息的工具名（会话恢复格式用）

        返回：这条消息自己的 msg_id（UUID 字符串）。
        """
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        turn_index = self._compute_turn_index(session_id, role)
        line_obj = {
            "id": msg_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "name": name,
            "timestamp": now,
            "turn_index": turn_index,
        }
        with self._lock:
            # 往 .jsonl 尾部追加一行（单行小写入，在 POSIX 上小于
            # PIPE_BUF 时天然原子，不会被别的进程写穿插）
            with self._session_file(session_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
            # 写完必须主动踢掉消息缓存——mtime 精度只有 ~15ms，
            # 同一窗口内连续 append 时 mtime 可能没变，缓存会误判"文件
            # 没变"而漏掉刚写的消息（回归用例 test_fork_session_does_not_mutate_source 盯着）
            self._msgs_cache.pop(session_id, None)
            # 新轮次号回写内存表（下一条消息纯内存查表，不再读盘）
            self._turn_state[session_id] = turn_index
            # 同步刷新目录卡片（内存）——磁盘写盘去抖：每条消息都原子重写
            # 整个 index.json 是 IO 热点（大小 O(历史会话总数)），改为
            # 累积 ≥_INDEX_FLUSH_COUNT 条或距上次 ≥_INDEX_FLUSH_SECONDS 秒
            # 才真正落盘。卡片只是目录册，落后几条可接受；消息正文在
            # .jsonl 里永不丢；进程退出由 atexit 注册的 flush_index 兜底。
            index = self._load_index()
            for entry in index:
                if entry["id"] == session_id:
                    entry["updated_at"] = now
                    entry["message_count"] = entry.get("message_count", 0) + 1
                    break
            self._index_dirty_count += 1
            _now_mono = time.monotonic()
            if (self._index_dirty_count >= _INDEX_FLUSH_COUNT
                    or _now_mono - self._index_last_flush >= _INDEX_FLUSH_SECONDS):
                self._save_index()
                self._index_dirty_count = 0
                self._index_last_flush = _now_mono
        return msg_id

    def flush_index(self) -> None:
        """把内存里还没落盘的目录卡片更新写进 index.json（去抖的显式收尾口）。

        批量写完、测试断言前、进程退出前调用；不调也只是磁盘卡片落后
        几条（消息正文不受影响）。幂等：没有脏数据时是空操作。

        参数：无
        返回：无。
        """
        with self._lock:
            if self._index_dirty_count > 0:
                self._save_index()
                self._index_dirty_count = 0
                self._index_last_flush = time.monotonic()

    def get_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """读某个会话的消息历史（给恢复会话/上层展示用）。

        参数：
            session_id：会话 ID
            limit：只要最后 N 条（按写入顺序），不填给全部

        返回：消息 dict 列表，只保留 role/content/tool_calls 等
        对外字段（id/timestamp 等内部记账字段已剥掉）。
        """
        msgs = self._read_session_msgs(session_id)
        if limit:
            msgs = msgs[-limit:]  # 只保留最后 N 条
        # 转成兼容格式（去掉 id/timestamp/turn_index 等内部字段）
        result = []
        for m in msgs:
            msg = {
                "role": m.get("role"),
                "content": m.get("content", ""),
            }
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                msg["name"] = m["name"]
            result.append(msg)
        return result

    def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """分页列出会话，最近动过的排最前（按 updated_at 倒序）。

        参数：
            limit：一页最多几条，默认 50
            offset：跳过前几条（翻页用），默认 0

        返回：会话元数据 dict 列表。
        """
        with self._lock:
            index = list(self._load_index())
        sorted_index = sorted(
            index, key=lambda s: s.get("updated_at", ""), reverse=True
        )
        return sorted_index[offset:offset + limit]

    def get_session(self, session_id: str) -> Optional[dict]:
        """查一个会话的元数据（标题/时间/计数这些，不含消息）。

        参数：
            session_id：会话 ID

        返回：元数据 dict 的副本；找不到返回 None。
        """
        with self._lock:
            index = self._load_index()
        for s in index:
            if s["id"] == session_id:
                return dict(s)
        return None

    def set_title(self, session_id: str, title: str) -> None:
        """改会话标题（顺手刷新 updated_at）。

        参数：
            session_id：会话 ID
            title：新标题

        返回：无。会话不存在时静默不动。
        """
        with self._lock:
            index = self._load_index()
            for s in index:
                if s["id"] == session_id:
                    s["title"] = title
                    s["updated_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._save_index()

    def delete_session(self, session_id: str) -> None:
        """删除会话——但其实是"假删"：消息文件改名成 .bak 备份，随时可恢复。

        这是项目"完全可逆"铁律的体现，符合数据永不真删的约定。

        从目录卡片移除和文件改名这两步必须在
        同一把锁里做完，否则中途被打断会出现两步只做一半的脏状态。

        参数：
            session_id：要删的会话 ID

        返回：无。
        """
        with self._lock:
            path = self._session_file(session_id)
            if path.exists():
                bak = path.with_suffix(".jsonl.bak")
                try:
                    # 已有同名 .bak 就先清掉，否则改名会撞名失败
                    if bak.exists():
                        bak.unlink()
                    path.rename(bak)
                except OSError:
                    # Windows 上 rename 偶发失败 → 兜底直接删（牺牲可逆保可用）
                    path.unlink(missing_ok=True)
            index = self._load_index()
            self._index_cache = [s for s in index if s["id"] != session_id]
            self._save_index()

    # ------------------------------------------------------------------
    # 会话 fork（克隆）
    # ------------------------------------------------------------------

    def fork_session(
        self,
        source_session_id: str,
        *,
        title: Optional[str] = None,
    ) -> str:
        """把一个现有会话整个克隆成新会话——消息文件原样复制一份，
        之后两边各改各的互不影响（像复印一份档案）。

        参数：
            source_session_id：被克隆的源会话 ID
            title：新会话标题；不填就自动叫 "Fork of <源标题>"

        返回：新会话的 session_id。源会话不存在时抛 ValueError。
        """
        source = self.get_session(source_session_id)
        if source is None:
            raise ValueError(f"source session 不存在: {source_session_id}")

        src_title = source.get("title") or source_session_id[:8]
        new_id = self.create_session(
            title=title or f"Fork of {src_title}",
            model=source.get("model"),
            provider=source.get("provider"),
        )

        src_path = self._session_file(source_session_id)
        dst_path = self._session_file(new_id)
        if src_path.exists():
            with self._lock:
                dst_path.write_text(
                    src_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                # 把新会话的消息计数改准（复制来的不是 0）
                line_count = sum(
                    1 for line in dst_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                index = self._load_index()
                for s in index:
                    if s["id"] == new_id:
                        s["message_count"] = line_count
                        break
                self._save_index()
        return new_id

    # ------------------------------------------------------------------
    # 全文搜索（用 Python 正则扫描）
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        tool_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[dict]:
        """在所有会话的消息里做全文搜索（Python 正则实现：re.escape + 忽略大小写扫所有 .jsonl）。

        - 性能：一万条消息约 100ms（单用户场景够用）
        - 中文子串：直接子串匹配（re.search 天然支持）
        - snippet：截取关键词周围若干字符的上下文片段

        参数：
            query：搜索关键词（按字面匹配，忽略大小写）
            limit：最多返回几条，默认 10
            session_id：只在指定会话里搜（不填搜全部）
            role：只搜指定角色（如 "user"）的消息
            tool_name：只搜调过指定工具的消息
            since：只搜这个时间点之后的消息（ISO 时间字符串）
            until：只搜这个时间点之前的消息

        返回：命中结果 dict 列表（含 content/session_id/snippet 等）。
        """
        if not query.strip():
            return []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            return []

        results = []
        # 指定了 session_id 就只扫那一个档案袋，否则扫最近 1000 个会话
        scan_sessions = (
            [self.get_session(session_id)] if session_id else self.list_sessions(limit=1000)
        )
        scan_sessions = [s for s in scan_sessions if s]  # 滤掉查不存在的会话返回的 None

        for s in scan_sessions:
            sid = s["id"]
            title = s.get("title")
            for m in self._read_session_msgs(sid):
                content = m.get("content", "") or ""
                if not pattern.search(content):
                    continue
                # 命中关键词后，再过一遍各筛选条件
                if role and m.get("role") != role:
                    continue
                if since and m.get("timestamp", "") < since:
                    continue
                if until and m.get("timestamp", "") > until:
                    continue
                if tool_name:
                    tcs = m.get("tool_calls") or []
                    if not any(
                        isinstance(tc, dict)
                        and isinstance(tc.get("function"), dict)
                        and tc["function"].get("name") == tool_name
                        for tc in tcs
                    ):
                        continue
                results.append({
                    "content": content,
                    "role": m.get("role"),
                    "session_id": sid,
                    "timestamp": m.get("timestamp"),
                    "title": title,
                    "snippet": self._make_snippet(content, query),
                    "rank": 0,
                })
                if len(results) >= limit * 3:
                    break
            if len(results) >= limit * 3:
                break
        return results[:limit]

    @staticmethod
    def _make_snippet(content: str, query: str, context_chars: int = 50) -> str:
        """（内部）生成摘要片段：截取关键词第一次出现位置前后各 context_chars 个字符，
        不够截的地方省略号表示。

        参数：
            content：消息全文
            query：搜索关键词
            context_chars：关键词前后各带多少字符，默认 50

        返回：截好的片段字符串；没找到关键词就返回开头 100 字符。
        """
        idx = content.lower().find(query.lower())
        if idx == -1:
            return content[:100]
        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(query) + context_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        return prefix + content[start:end] + suffix

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """汇总统计整个会话库：会话数/消息数/各工具被调了多少次/角色分布等（给 /stats 命令做数据源）。

        参数：无。

        返回：统计 dict，含 sessions/messages/earliest/latest/
        top_sessions（消息最多的前 5 个会话）/tool_calls（前 10 个高频
        工具）/role_distribution（各角色消息条数）。
        """
        index = self.list_sessions(limit=10000)
        sessions_count = len(index)
        messages_count = sum(s.get("message_count", 0) for s in index)
        timestamps_created = [s.get("created_at") for s in index if s.get("created_at")]
        timestamps_updated = [s.get("updated_at") for s in index if s.get("updated_at")]
        earliest = min(timestamps_created) if timestamps_created else None
        latest = max(timestamps_updated) if timestamps_updated else None
        top_sessions = sorted(
            index, key=lambda s: s.get("message_count", 0), reverse=True
        )[:5]
        # 角色分布 + 工具调用统计（要逐条翻所有 .jsonl 档案袋）
        role_dist = {}
        tool_counter = {}
        for s in index:
            for m in self._read_session_msgs(s["id"]):
                role = m.get("role", "unknown")
                role_dist[role] = role_dist.get(role, 0) + 1
                tcs = m.get("tool_calls")
                if not isinstance(tcs, list):
                    continue  # 防御：tool_calls 字段可能被改坏成字符串等，跳过不崩
                for tc in tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            name = fn.get("name")
                            if name:
                                tool_counter[name] = tool_counter.get(name, 0) + 1
        sorted_tools = sorted(
            tool_counter.items(), key=lambda x: x[1], reverse=True
        )[:10]
        return {
            "sessions": sessions_count,
            "messages": messages_count,
            "earliest": earliest,
            "latest": latest,
            "top_sessions": top_sessions,
            "tool_calls": [{"name": n, "count": c} for n, c in sorted_tools],
            "role_distribution": role_dist,
        }
