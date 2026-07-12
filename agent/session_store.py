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

-- FTS5 全文索引（虚拟表）
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


def is_fts5_available() -> bool:
    """检查 SQLite 是否支持 FTS5。"""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test_fts5 USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


class SessionStore:
    """会话存储管理器。"""

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器，确保关闭）。

        注意：SQLite 默认不启用外键，要显式开启。
        WAL 模式让并发读不阻塞写。

        用法：with self._get_conn() as conn: ...
        """
        conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # 自动提交
        )
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass  # 某些平台不支持 WAL
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _init_schema(self):
        """初始化数据库 schema。"""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)

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
    # 全文搜索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[dict]:
        """全文搜索消息。

        使用 FTS5，支持：
        - 关键词匹配
        - 短语匹配（用引号）
        - 相关性排序

        返回匹配的消息 + 会话信息。
        """
        # FTS5 查询构造
        fts_query = self._build_fts_query(query)
        if fts_query == '""':
            return []

        try:
            with self._get_conn() as conn:
                if session_id:
                    sql = """
                        SELECT m.content, m.role, m.session_id, m.timestamp,
                               s.title,
                               snippet(messages_fts, 0, '<<', '>>', '...', 20) as snippet,
                               rank
                        FROM messages_fts
                        JOIN messages m ON messages_fts.message_id = m.id
                        LEFT JOIN sessions s ON m.session_id = s.id
                        WHERE messages_fts MATCH ?
                          AND m.session_id = ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (fts_query, session_id, limit)).fetchall()
                else:
                    sql = """
                        SELECT m.content, m.role, m.session_id, m.timestamp,
                               s.title,
                               snippet(messages_fts, 0, '<<', '>>', '...', 20) as snippet,
                               rank
                        FROM messages_fts
                        JOIN messages m ON messages_fts.message_id = m.id
                        LEFT JOIN sessions s ON m.session_id = s.id
                        WHERE messages_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    rows = conn.execute(sql, (fts_query, limit)).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS 搜索失败: %s", e)
            return []

        return [dict(row) for row in rows]

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
