"""会话存储：SQLite + FTS5。

两张表：
1. sessions  - 会话元信息（id, 标题, 时间戳, 消息数）
2. messages  - 消息内容（role, content, session_id, 时间戳）

加一个 FTS5 虚拟表：
3. messages_fts - messages 表的全文索引

通过触发器自动维护 FTS 索引（消息插入/删除时）。
"""

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
-- 会话元信息
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    model TEXT,
    provider TEXT
);

-- 消息内容
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- system/user/assistant/tool
    content TEXT NOT NULL,
    tool_calls TEXT,              -- JSON（assistant 的工具调用）
    tool_call_id TEXT,            -- tool 消息的配对 id
    timestamp TEXT NOT NULL,
    turn_index INTEGER NOT NULL,  -- 第几轮对话
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, turn_index);

-- FTS5 全文索引（虚拟表，unicode61 tokenizer，多语言但 CJK 子串弱）
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    role,
    session_id UNINDEXED,
    message_id UNINDEXED,
    tokenize = 'unicode61'   -- 支持多语言
);

-- 触发器：消息插入时自动更新 FTS 索引
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages
BEGIN
    INSERT INTO messages_fts(content, role, session_id, message_id)
    VALUES (new.content, new.role, new.session_id, new.id);
END;

-- 触发器：消息删除时清理 FTS 索引
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages
BEGIN
    DELETE FROM messages_fts WHERE message_id = old.id;
END;

-- ============================================================
-- memories 表（与 memory_store.py 双写）
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,           -- 业务 id（memory_store 生成）
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    type TEXT NOT NULL,            -- user/feedback/project/reference/other
    body TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
    -- 注意：未声明 WITHOUT ROWID，SQLite 自动给隐藏 INTEGER rowid，
    -- FTS5 用它做关联。
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);

-- memories FTS5 trigram 索引（CJK 子串友好）
-- 不用 content_rowid 关联，靠 new.rowid ↔ memories_fts.rowid 自增对齐
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    name, description, body,
    tokenize = 'trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, name, description, body)
    VALUES (new.rowid, new.name, new.description, COALESCE(new.body, ''));
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
BEGIN
    DELETE FROM memories_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories
BEGIN
    DELETE FROM memories_fts WHERE rowid = old.rowid;
    INSERT INTO memories_fts(rowid, name, description, body)
    VALUES (new.rowid, new.name, new.description, COALESCE(new.body, ''));
END;

-- ============================================================
-- tasks 表（与 task_store.py 双写）
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    owner TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    blocked_by TEXT,                -- JSON list
    body_json TEXT                  -- 完整 JSON（其他字段都塞这）
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

-- tasks FTS5 trigram 索引
CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
    subject, description,
    tokenize = 'trigram'
);

CREATE TRIGGER IF NOT EXISTS tasks_fts_ai AFTER INSERT ON tasks
BEGIN
    INSERT INTO tasks_fts(rowid, subject, description)
    VALUES (new.rowid, new.subject, COALESCE(new.description, ''));
END;

CREATE TRIGGER IF NOT EXISTS tasks_fts_ad AFTER DELETE ON tasks
BEGIN
    DELETE FROM tasks_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS tasks_fts_au AFTER UPDATE ON tasks
BEGIN
    DELETE FROM tasks_fts WHERE rowid = old.rowid;
    INSERT INTO tasks_fts(rowid, subject, description)
    VALUES (new.rowid, new.subject, COALESCE(new.description, ''));
END;
"""


def is_fts5_available() -> bool:
    """检查 SQLite 是否支持 FTS5。"""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test_fts5 USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def is_trigram_available() -> bool:
    """检查 FTS5 trigram tokenizer 是否可用（SQLite >= 3.34 with FTS5 enabled）。"""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test_tri USING fts5(c, tokenize='trigram')")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def _contains_cjk(s: str) -> bool:
    """判断字符串是否含 CJK 字符（用于触发 trigram fallback）。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


class SessionStore:
    """会话存储管理器。"""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # 持久连接：所有操作共用一条连接，避免每次开关的 2-5ms 开销
        # WAL 模式下读写互不阻塞；多线程访问通过 _conn_lock 串行化
        self._conn_lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # 自动提交（每条 SQL 独立事务）
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass  # 某些平台不支持 WAL
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """获取持久连接（加锁，上下文退出时释放锁但不关闭连接）。

        用法：with self._get_conn() as conn: ...
        所有 SQL 都在这条连接上串行执行（_conn_lock 保证线程安全）。
        """
        self._conn_lock.acquire()
        try:
            yield self._conn
        finally:
            self._conn_lock.release()

    def _init_schema(self):
        """初始化数据库 schema。"""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        """关闭持久连接（测试或 shutdown 时调用，Windows 上不关会锁文件）。"""
        with self._conn_lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __del__(self):
        # GC 兜底：测试结束时不让连接泄漏锁住文件
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 会话生命周期
    # ------------------------------------------------------------------

    def create_session(
        self,
        *,
        title: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """创建新会话，返回 session_id。"""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO sessions (id, title, created_at, updated_at, model, provider)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, title, now, now, model, provider),
            )

        return session_id

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """追加一条消息到会话。"""
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            # 单次事务：SELECT + INSERT + UPDATE + FTS trigger 一次性 commit
            # 原 isolation_level=None 下每条 SQL 自动 commit，4 次 fsync；
            # 显式 BEGIN/COMMIT 只 1 次 fsync，省 5-15ms
            conn.execute("BEGIN")
            try:
                # 获取当前 turn_index
                row = conn.execute(
                    "SELECT MAX(turn_index) as max_turn FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                # user 消息开始新的 turn
                current_turn = (row["max_turn"] or 0) if row else 0
                if role == "user":
                    current_turn += 1

                conn.execute(
                    """INSERT INTO messages
                       (id, session_id, role, content, tool_calls, tool_call_id,
                        timestamp, turn_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        msg_id, session_id, role, content,
                        json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                        tool_call_id,
                        now, current_turn,
                    ),
                )

                # 更新会话的 updated_at 和消息计数
                conn.execute(
                    """UPDATE sessions
                       SET updated_at = ?, message_count = message_count + 1
                       WHERE id = ?""",
                    (now, session_id),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return msg_id

    def get_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """获取会话的消息历史。"""
        with self._get_conn() as conn:
            query = """
                SELECT role, content, tool_calls, tool_call_id, timestamp, turn_index
                FROM messages
                WHERE session_id = ?
                ORDER BY rowid
            """
            params: list = [session_id]
            if limit:
                # 取最后 N 条
                query = """
                    SELECT * FROM (
                        SELECT role, content, tool_calls, tool_call_id,
                               timestamp, turn_index
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY rowid
                    ) ORDER BY timestamp DESC LIMIT ?
                """
                params.append(limit)

            rows = conn.execute(query, params).fetchall()

        messages = []
        for row in rows:
            msg = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except json.JSONDecodeError:
                    pass
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            messages.append(msg)

        return messages

    def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """列出会话（按更新时间倒序）。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, title, created_at, updated_at, message_count, model
                   FROM sessions
                   ORDER BY updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取单个会话信息。"""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT id, title, created_at, updated_at, message_count, model, provider
                   FROM sessions WHERE id = ?""",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_title(self, session_id: str, title: str) -> None:
        """设置会话标题。"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """删除会话（级联删除消息和 FTS 索引）。"""
        with self._get_conn() as conn:
            # 先删消息（触发 FTS 清理 trigger）
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )

    # ------------------------------------------------------------------
    # P2-12 NEW: resume / fork
    # ------------------------------------------------------------------

    def fork_session(
        self,
        source_session_id: str,
        *,
        title: Optional[str] = None,
    ) -> str:
        """克隆现有会话为新会话（消息全复制）。

        用途：在现有对话基础上做实验分支，不破坏原对话。
        resume 已由 RuntimeContext.resume_session 提供（加载消息到 agent 内存），
        本方法只做"克隆到新 session_id"。

        参数：
            source_session_id: 被克隆的源会话 ID
            title: 新会话标题；None 时默认 "Fork of <源标题>"

        返回新 session_id。源会话不变。
        """
        source = self.get_session(source_session_id)
        if source is None:
            raise ValueError(
                f"source session 不存在: {source_session_id}"
            )

        src_title = source.get("title") or source_session_id[:8]
        new_id = self.create_session(
            title=title or f"Fork of {src_title}",
            model=source.get("model"),
            provider=source.get("provider"),
        )

        # 批量复制所有消息（单连接 + executemany，避免 N 次 append_message 的开关连接 + fsync）
        # 100 条消息从 5-15s 降到 100-300ms
        msgs = self.get_messages(source_session_id)
        if msgs:
            now = datetime.now(timezone.utc).isoformat()
            # 先算 turn_index（user 消息开始新 turn，与 append_message 语义一致）
            rows = []
            current_turn = 0
            for m in msgs:
                role = m["role"]
                if role == "user":
                    current_turn += 1
                tool_calls = m.get("tool_calls")
                rows.append((
                    str(uuid.uuid4()),
                    new_id,
                    role,
                    m.get("content") or "",
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    m.get("tool_call_id"),
                    now,
                    current_turn,
                ))
            with self._get_conn() as conn:
                conn.execute("BEGIN")
                try:
                    conn.executemany(
                        """INSERT INTO messages
                           (id, session_id, role, content, tool_calls, tool_call_id,
                            timestamp, turn_index)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    conn.execute(
                        """UPDATE sessions
                           SET updated_at = ?, message_count = message_count + ?
                           WHERE id = ?""",
                        (now, len(rows), new_id),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

        return new_id

    # ------------------------------------------------------------------
    # 全文搜索
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
        """全文搜索消息，支持过滤。

        使用 FTS5，支持：
        - 关键词匹配
        - 短语匹配（用引号）
        - 相关性排序

        过滤参数（B2 增强）：
        - session_id: 限定会话
        - role: 限定角色（user/assistant/tool）
        - tool_name: 限定消息含特定工具调用（Python 层过滤）
        - since/until: ISO8601 时间范围（字符串比较）

        返回匹配的消息 + 会话信息。
        """
        # FTS5 查询构造
        fts_query = self._build_fts_query(query)
        if fts_query == '""':
            return []

        # 动态构造 WHERE 子句
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [fts_query]

        if session_id:
            where_clauses.append("m.session_id = ?")
            params.append(session_id)
        if role:
            where_clauses.append("m.role = ?")
            params.append(role)
        if since:
            where_clauses.append("m.timestamp >= ?")
            params.append(since)
        if until:
            where_clauses.append("m.timestamp <= ?")
            params.append(until)

        where_sql = " AND ".join(where_clauses)

        # 取候选（不加 limit，Python 层 tool_name 过滤后再截断）
        # 但为防卡死，给一个 10x limit 上限
        fetch_limit = limit * 10 if tool_name else limit
        sql = f"""
            SELECT m.content, m.role, m.session_id, m.timestamp,
                   m.tool_calls,
                   s.title,
                   snippet(messages_fts, 0, '<<', '>>', '...', 20) as snippet,
                   rank
            FROM messages_fts
            JOIN messages m ON messages_fts.message_id = m.id
            LEFT JOIN sessions s ON m.session_id = s.id
            WHERE {where_sql}
            ORDER BY rank
            LIMIT ?
        """
        params.append(fetch_limit)

        try:
            with self._get_conn() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS 搜索失败: %s", e)
            return []

        results = [dict(row) for row in rows]

        # 中文子串 fallback：unicode61 默认按词分，搜"配置"找不到含"系统配置"的消息
        # 此时切到 trigram 索引（如果可用且 query 含 CJK）
        if not results and _contains_cjk(query) and is_trigram_available():
            results = self._search_messages_trigram(
                query=query,
                limit=fetch_limit,
                session_id=session_id,
                role=role,
                since=since,
                until=until,
            )

        # tool_name 过滤（Python 层，解析 tool_calls JSON）
        if tool_name:
            filtered = []
            for r in results:
                tc = r.get("tool_calls")
                if not tc:
                    continue
                try:
                    calls = json.loads(tc) if isinstance(tc, str) else tc
                    if isinstance(calls, list):
                        for c in calls:
                            if isinstance(c, dict):
                                fn = c.get("function", {})
                                if isinstance(fn, dict) and fn.get("name") == tool_name:
                                    filtered.append(r)
                                    break
                except (json.JSONDecodeError, TypeError):
                    continue
            results = filtered[:limit]
        else:
            results = results[:limit]

        # 移除内部字段 tool_calls（保持向后兼容）
        for r in results:
            r.pop("tool_calls", None)

        return results

    def get_stats(self) -> dict:
        """聚合统计：会话/消息/工具调用频次/角色分布。

        返回：
            {
                "sessions": int,
                "messages": int,
                "earliest": Optional[str],     # ISO 时间
                "latest": Optional[str],
                "top_sessions": List[dict],    # Top 5 按 message_count
                "tool_calls": List[dict],      # Top 10 工具 [{"name": ..., "count": ...}]
                "role_distribution": dict,     # {"user": int, "assistant": int, ...}
            }
        """
        with self._get_conn() as conn:
            # 总览
            overview = conn.execute(
                """SELECT
                       COUNT(*) AS sessions,
                       COALESCE(SUM(message_count), 0) AS messages,
                       MIN(created_at) AS earliest,
                       MAX(updated_at) AS latest
                   FROM sessions"""
            ).fetchone()

            # Top 5 最长会话
            top_sessions = conn.execute(
                """SELECT id, title, message_count, model, updated_at
                   FROM sessions
                   ORDER BY message_count DESC LIMIT 5"""
            ).fetchall()

            # 角色分布
            roles = conn.execute(
                """SELECT role, COUNT(*) AS cnt
                   FROM messages GROUP BY role"""
            ).fetchall()
            role_dist = {r["role"]: r["cnt"] for r in roles}

            # 工具调用统计：扫所有 messages.tool_calls（JSON 数组）
            # SQLite 没有 JSON 解析（除非装了 JSON1 扩展），这里读所有非空 tool_calls 在 Python 里解析
            tool_rows = conn.execute(
                "SELECT tool_calls FROM messages WHERE tool_calls IS NOT NULL"
            ).fetchall()

        tool_counter: dict = {}
        for r in tool_rows:
            try:
                calls = json.loads(r["tool_calls"])
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            fn = call.get("function", {})
                            name = fn.get("name") if isinstance(fn, dict) else None
                            if name:
                                tool_counter[name] = tool_counter.get(name, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        sorted_tools = sorted(
            tool_counter.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "sessions": overview["sessions"] if overview else 0,
            "messages": overview["messages"] if overview else 0,
            "earliest": overview["earliest"] if overview else None,
            "latest": overview["latest"] if overview else None,
            "top_sessions": [dict(s) for s in top_sessions],
            "tool_calls": [{"name": n, "count": c} for n, c in sorted_tools],
            "role_distribution": role_dist,
        }

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """构造 FTS5 查询。

        简单策略：
        - 按空格分词
        - 每个词加前缀通配符（*）
        - 用 AND 连接
        """
        clean = query.strip()
        if not clean:
            return '""'

        words = clean.split()
        # 每个词加前缀匹配（双引号包裹防止特殊字符）
        fts_terms = [f'"{w}"*' for w in words if w]
        return " AND ".join(fts_terms) if fts_terms else '""'

    # ------------------------------------------------------------------
    # trigram fallback：CJK 子串搜索
    # ------------------------------------------------------------------

    def _search_messages_trigram(
        self,
        *,
        query: str,
        limit: int,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[dict]:
        """用 messages_fts_trigram 跑 CJK 子串搜索。

        trigram tokenizer 把文本切成 3-gram，对中文子串匹配强。
        SQLite >= 3.34 + FTS5 才支持，初始化时已检测。
        """
        # 这里复用 messages_fts 的 column 结构（content/role/session_id/message_id）
        # 但 trigram 索引是另一个表 messages_fts_trigram（schema 里没创建，
        # 因为现在主表 messages_fts 已经覆盖；这里保留接口供未来扩展）。
        # 当前实现：直接 LIKE 全表扫（数据小时性能可接受，未来再建独立 trigram 表
        # 或迁移主 tokenizer）。
        where_clauses = ["m.content LIKE ?"]
        params: list = [f"%{query}%"]
        if session_id:
            where_clauses.append("m.session_id = ?")
            params.append(session_id)
        if role:
            where_clauses.append("m.role = ?")
            params.append(role)
        if since:
            where_clauses.append("m.timestamp >= ?")
            params.append(since)
        if until:
            where_clauses.append("m.timestamp <= ?")
            params.append(until)
        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT m.content, m.role, m.session_id, m.timestamp,
                   NULL as tool_calls,
                   s.title,
                   m.content as snippet,
                   0 as rank
            FROM messages m
            LEFT JOIN sessions s ON m.session_id = s.id
            WHERE {where_sql}
            ORDER BY m.timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        try:
            with self._get_conn() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("trigram fallback 搜索失败: %s", e)
            return []
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # memories 表（与 memory_store.py 双写）
    # ------------------------------------------------------------------

    def save_memory(
        self,
        *,
        id: str,
        name: str,
        description: str,
        type: str,
        body: str,
        created_at: str,
        updated_at: str,
        archived: int = 0,
    ) -> None:
        """插入或更新一条记忆。"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, name, description, type, body, created_at, updated_at, archived)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (id, name, description, type, body, created_at, updated_at, archived),
            )

    def get_memory(self, memory_id: str) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_memories(self, *, include_archived: bool = False) -> List[dict]:
        with self._get_conn() as conn:
            if include_archived:
                sql = "SELECT * FROM memories ORDER BY updated_at DESC"
                rows = conn.execute(sql).fetchall()
            else:
                sql = (
                    "SELECT * FROM memories WHERE archived = 0 "
                    "ORDER BY updated_at DESC"
                )
                rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]

    def archive_memory(self, memory_id: str) -> bool:
        """软删除：标 archived=1。返回是否命中。"""
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE memories SET archived = 1, updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), memory_id),
            )
            return cur.rowcount > 0

    def update_memory(
        self,
        memory_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> bool:
        """部分更新记忆。返回是否命中。"""
        sets: list[str] = []
        params: list = []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if type is not None:
            sets.append("type = ?")
            params.append(type)
        if body is not None:
            sets.append("body = ?")
            params.append(body)
        if updated_at is not None:
            sets.append("updated_at = ?")
            params.append(updated_at)
        if not sets:
            return False
        params.append(memory_id)
        with self._get_conn() as conn:
            cur = conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            return cur.rowcount > 0

    def search_memories(self, query: str, *, limit: int = 10) -> List[dict]:
        """trigram 全文搜索记忆（中文子串友好）。

        短查询（< 3 字符）走 LIKE：trigram tokenizer 切不出 3-gram，
        对 2 字符中文（如"配置"）不命中。
        """
        if not is_trigram_available() or len(query) < 3:
            return self._search_memories_like(query, limit=limit)

        # trigram 用双引号包字符串做 phrase 查询
        fts_query = f'"{query}"'
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT m.id, m.name, m.description, m.type,
                              snippet(memories_fts, 1, '<<', '>>', '...', 10) as snippet,
                              m.updated_at
                       FROM memories_fts
                       JOIN memories m ON m.rowid = memories_fts.rowid
                       WHERE memories_fts MATCH ? AND m.archived = 0
                       ORDER BY rank LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
                results = [dict(r) for r in rows]
            # trigram 没命中再 fallback 到 LIKE（覆盖短词或边界情况）
            if not results:
                return self._search_memories_like(query, limit=limit)
            return results
        except sqlite3.OperationalError as e:
            logger.warning("memories_fts 搜索失败: %s", e)
            return self._search_memories_like(query, limit=limit)

    def _search_memories_like(self, query: str, *, limit: int = 10) -> List[dict]:
        """LIKE 兜底搜索。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, name, description, type,
                          COALESCE(
                              substr(description, 1, 100),
                              substr(body, 1, 100)
                          ) as snippet,
                          updated_at
                   FROM memories
                   WHERE archived = 0 AND (
                       name LIKE ? OR description LIKE ? OR body LIKE ?
                   )
                   ORDER BY updated_at DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # tasks 表（与 task_store.py 双写）
    # ------------------------------------------------------------------

    def save_task(
        self,
        *,
        id: str,
        subject: str,
        description: str,
        status: str,
        owner: Optional[str] = None,
        created_at: str,
        updated_at: str,
        blocked_by: Optional[list] = None,
        body_json: Optional[str] = None,
    ) -> None:
        """插入或更新一个任务（body_json 是完整 task dict 的 JSON 字符串）。"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tasks
                   (id, subject, description, status, owner,
                    created_at, updated_at, blocked_by, body_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    id, subject, description, status, owner,
                    created_at, updated_at,
                    json.dumps(blocked_by or []),
                    body_json,
                ),
            )

    def get_task(self, task_id: str) -> Optional[dict]:
        """取任务的 body_json 字段（完整 task dict）。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT body_json FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row or not row["body_json"]:
                return None
            try:
                return json.loads(row["body_json"])
            except (json.JSONDecodeError, TypeError):
                return None

    def list_tasks(
        self, *, status: Optional[str] = None, include_deleted: bool = False
    ) -> List[dict]:
        """列出任务（返回完整 body_json 解出来的 list）。"""
        with self._get_conn() as conn:
            if status:
                sql = "SELECT body_json FROM tasks WHERE status = ? ORDER BY created_at"
                rows = conn.execute(sql, (status,)).fetchall()
            elif not include_deleted:
                sql = "SELECT body_json FROM tasks WHERE status != 'deleted' ORDER BY created_at"
                rows = conn.execute(sql).fetchall()
            else:
                sql = "SELECT body_json FROM tasks ORDER BY created_at"
                rows = conn.execute(sql).fetchall()
        result: List[dict] = []
        for r in rows:
            if r["body_json"]:
                try:
                    result.append(json.loads(r["body_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue
        return result

    def delete_task(self, task_id: str) -> bool:
        """硬删除（task_store 用软删除 status=deleted，这里硬删 SQLite 行）。"""
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cur.rowcount > 0

    def search_tasks(self, query: str, *, limit: int = 10) -> List[dict]:
        """trigram 搜索任务的 subject/description。返回完整 task dict 列表。"""
        if not is_trigram_available() or len(query) < 3:
            rows = self._search_tasks_like_rows(query, limit=limit)
        else:
            fts_query = f'"{query}"'
            try:
                with self._get_conn() as conn:
                    rows = conn.execute(
                        """SELECT t.body_json
                           FROM tasks_fts
                           JOIN tasks t ON t.rowid = tasks_fts.rowid
                           WHERE tasks_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (fts_query, limit),
                    ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("tasks_fts 搜索失败: %s", e)
                rows = self._search_tasks_like_rows(query, limit=limit)
            else:
                if not rows:
                    rows = self._search_tasks_like_rows(query, limit=limit)
        result: List[dict] = []
        for r in rows:
            if r["body_json"]:
                try:
                    result.append(json.loads(r["body_json"]))
                except (json.JSONDecodeError, TypeError):
                    continue
        return result

    def _search_tasks_like_rows(self, query: str, *, limit: int = 10):
        """LIKE 兜底搜任务（返回 sqlite3.Row 列表）。"""
        with self._get_conn() as conn:
            return conn.execute(
                """SELECT body_json FROM tasks
                   WHERE subject LIKE ? OR description LIKE ?
                   ORDER BY created_at LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
