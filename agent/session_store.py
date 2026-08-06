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
    name TEXT,                    -- tool 消息的工具名（对齐 Claude Code 会话恢复）
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
"""

# 注：memories 和 tasks 表已移除（2026-07-20）
# 原设计：与 memory_store.py / task_store.py 双写到 SQLite
# 移除原因：search_memories / search_tasks 在业务代码里无调用方，FTS5 索引从未被使用
# 现设计：memories/tasks 只用文件存储（.memory/*.md 和 .tasks/*.json）
# sessions.db 只保留 sessions + messages 两张表（session_search 工具依赖）


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
        """初始化数据库 schema（含旧库迁移：补 name 列）。"""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
            # 旧库迁移：messages 表补 name 列（SQLite 无 ADD COLUMN IF NOT EXISTS，try/except 幂等）
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN name TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在（新库由 SCHEMA_SQL 直接建）

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
        name: Optional[str] = None,
    ) -> str:
        """追加一条消息到会话。

        name：tool 消息的工具名（对齐 Claude Code 会话恢复，恢复时还原完整工具轮次）。
        """
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
                        name, timestamp, turn_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        msg_id, session_id, role, content,
                        json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                        tool_call_id,
                        name,
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
                SELECT role, content, tool_calls, tool_call_id, name,
                       timestamp, turn_index
                FROM messages
                WHERE session_id = ?
                ORDER BY rowid
            """
            params: list = [session_id]
            if limit:
                # 取最后 N 条
                query = """
                    SELECT * FROM (
                        SELECT role, content, tool_calls, tool_call_id, name,
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
            if row["name"]:
                msg["name"] = row["name"]
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
        """删除会话（级联删除消息和 FTS 索引）。

        S7 fix: 用显式 BEGIN/COMMIT 包两条 DELETE，
        避免自动提交模式下中途崩溃留孤儿 messages。
        """
        with self._get_conn() as conn:
            # 显式开事务（isolation_level=None 模式下不会自动 BEGIN）
            conn.execute("BEGIN")
            try:
                # 先删消息（触发 FTS 清理 trigger）
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,),
                )
                conn.execute("COMMIT")
            except Exception:
                # 任意异常 → ROLLBACK，session 和 messages 都不变
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

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
                    m.get("name"),
                    now,
                    current_turn,
                ))
            with self._get_conn() as conn:
                conn.execute("BEGIN")
                try:
                    conn.executemany(
                        """INSERT INTO messages
                           (id, session_id, role, content, tool_calls, tool_call_id,
                            name, timestamp, turn_index)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    # memories / tasks 表已移除（2026-07-20）
    # 原设计：与 memory_store.py / task_store.py 双写到 SQLite，FTS5 搜索
    # 移除原因：search_memories / search_tasks 在业务代码里无调用方（死代码）
    # 现设计：memories/tasks 纯文件存储（.memory/*.md 和 .tasks/*.json）
    # 本类只保留 sessions + messages 两张表（session_search 工具依赖）
    # ------------------------------------------------------------------
