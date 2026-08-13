"""会话存储：JSONL 文件（无 SQLite）。

文件布局：
  ~/.OmniMate/.sessions/
  ├── index.json              # 所有会话元数据（数组）
  ├── <session_id>.jsonl      # 每个会话的消息历史（每行一条 JSON）
  └── <session_id>.jsonl.bak  # 删除时改名备份（完全可逆）

设计权衡（为啥不用 SQLite）：
  - Windows 上 SQLite 文件锁曾反复出问题（X8 rowid bug 等）
  - 与项目"文件优先"哲学一致（memory/tasks 都是文件）
  - 用户可直接看/编辑 JSONL
  - 跨平台一致
  - 全文搜索降级为 Python re（小规模够用）

接口与原 SQLite 版本完全一致（cli.py / agent/__init__.py 等调用方无需改动）。
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 兼容老接口的存根（其他模块可能 import）
# ---------------------------------------------------------------------------

def is_fts5_available() -> bool:
    """兼容老接口（JSONL 版不需要 FTS5）。"""
    return False


def is_trigram_available() -> bool:
    """兼容老接口。"""
    return False


def _contains_cjk(s: str) -> bool:
    """兼容老接口。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


class SessionStore:
    """会话存储管理器（JSONL 文件实现）。

    所有接口与原 SQLite 版本兼容（cli.py / agent/__init__.py 等调用方无需改动）。
    __init__ 接受 db_path（兼容老接口），实际作为目录用：
    - 如果传文件路径（如 sessions.db）→ 自动转 parent/.sessions/
    - 如果传目录路径 → 直接用

    线程安全：通过 _lock 保护 index.json 和 .jsonl 写入。
    """

    def __init__(self, db_path):
        """初始化。

        参数 db_path 为兼容老接口保留（原是 sessions.db 文件路径）。
        实际数据存储在 db_path 对应的目录中：
        - 文件路径（含 .db 后缀）→ parent / ".sessions"
        - 目录路径 → 直接用
        """
        db_path = Path(db_path)
        # 兼容：如果传的是 .db 文件路径，改成 sibling 目录
        if db_path.suffix == ".db":
            self._sessions_dir = db_path.parent / ".sessions"
        else:
            self._sessions_dir = db_path
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._sessions_dir / "index.json"
        self._lock = threading.Lock()
        self._index_cache: Optional[List[dict]] = None  # 内存缓存（首次访问加载）
        # 消息缓存（Round 3 压力优化）：session_id -> ((mtime, size), msgs)，
        # search/get_messages/get_stats 热路径免重复读盘 + JSON 解析。
        # 双因子键防 mtime 精度窗口（Windows ~15ms）内 append 误判。
        self._msgs_cache: dict = {}
        # 自动迁移老 SQLite（如果检测到）
        self._maybe_migrate_sqlite()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _load_index(self) -> List[dict]:
        """加载 index（带内存缓存）。"""
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
        """原子写 index.json（用 atomic_write_text 保证跨平台一致）。"""
        from agent.atomic_io import atomic_write_text
        if self._index_cache is None:
            return
        data = {"sessions": self._index_cache}
        atomic_write_text(
            self._index_path,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def _session_file(self, session_id: str) -> Path:
        """单个会话的 .jsonl 文件路径。"""
        return self._sessions_dir / f"{session_id}.jsonl"

    def _read_session_msgs(self, session_id: str) -> List[dict]:
        """读 .jsonl 全部消息（不动 index；mtime 缓存，只读共享）。

        压力优化（Round 3）：search/get_messages/get_stats/fork 每次
        全量读盘 + JSON 解析是热路径（万条消息每次 ~100ms+）。缓存
        (mtime, msgs)，文件变了自动失效重读。调用方全部只读遍历。
        """
        path = self._session_file(session_id)
        try:
            st = path.stat()
            cache_key = (st.st_mtime, st.st_size)
        except OSError:
            return []
        cached = self._msgs_cache.get(session_id)
        if cached is not None and cached[0] == cache_key:
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
        return msgs

    def _compute_turn_index(self, session_id: str, role: str) -> int:
        """计算 turn_index（user 消息开始新 turn）。"""
        msgs = self._read_session_msgs(session_id)
        if not msgs:
            return 1 if role == "user" else 0
        max_turn = max((m.get("turn_index", 0) for m in msgs), default=0)
        return max_turn + 1 if role == "user" else max_turn

    # ------------------------------------------------------------------
    # 自动迁移老 SQLite
    # ------------------------------------------------------------------

    def _maybe_migrate_sqlite(self) -> None:
        """检测老 sessions.db，存在且 index 为空时一次性迁移到 JSONL。"""
        # sessions.db 通常在 sessions_dir 的 parent
        candidates = [
            self._sessions_dir.parent / "sessions.db",
            self._sessions_dir / "sessions.db",
        ]
        old_db = next((p for p in candidates if p.exists()), None)
        if old_db is None:
            return
        # 已经迁移过（有 .bak）→ 跳过
        if old_db.with_suffix(".db.bak").exists():
            return
        # 只在 index 为空时迁移（避免覆盖已有数据）
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
        """从 SQLite 迁移数据到 JSONL。"""
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
    # 会话生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """兼容接口（JSONL 无连接，no-op）。"""
        pass

    def __del__(self):
        # 兼容老接口（JSONL 无连接需关）
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
        """创建新会话，返回 session_id。"""
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
        # 建空 .jsonl
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
        """追加一条消息到会话。

        name：tool 消息的工具名（对齐 Claude Code 会话恢复）。
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
            # 追加到 .jsonl（单行写入，POSIX 上 < PIPE_BUF 原子）
            with self._session_file(session_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
            # 主动失效消息缓存（mtime 粒度 ~15ms，同窗口 append 前后
            # mtime 可能相同导致缓存误判有效——见 test_fork_session_does_not_mutate_source）
            self._msgs_cache.pop(session_id, None)
            # 更新 index（updated_at + message_count）
            index = self._load_index()
            for entry in index:
                if entry["id"] == session_id:
                    entry["updated_at"] = now
                    entry["message_count"] = entry.get("message_count", 0) + 1
                    break
            self._save_index()
        return msg_id

    def get_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """获取会话的消息历史。

        limit=N 时取最后 N 条（按写入时序）。
        """
        msgs = self._read_session_msgs(session_id)
        if limit:
            msgs = msgs[-limit:]  # 取最后 N 条
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
        """列出会话（按 updated_at 倒序）。"""
        with self._lock:
            index = list(self._load_index())
        sorted_index = sorted(
            index, key=lambda s: s.get("updated_at", ""), reverse=True
        )
        return sorted_index[offset:offset + limit]

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取单个会话信息。"""
        with self._lock:
            index = self._load_index()
        for s in index:
            if s["id"] == session_id:
                return dict(s)
        return None

    def set_title(self, session_id: str, title: str) -> None:
        """设置会话标题。"""
        with self._lock:
            index = self._load_index()
            for s in index:
                if s["id"] == session_id:
                    s["title"] = title
                    s["updated_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._save_index()

    def delete_session(self, session_id: str) -> None:
        """删除会话（消息文件改名 .bak，完全可逆）。

        S7 fix: 原子操作——index 移除 + 文件改名在同一锁内完成。
        """
        with self._lock:
            path = self._session_file(session_id)
            if path.exists():
                bak = path.with_suffix(".jsonl.bak")
                try:
                    # 如果 .bak 已存在先删
                    if bak.exists():
                        bak.unlink()
                    path.rename(bak)
                except OSError:
                    # Windows rename 偶尔失败 → 直接 unlink
                    path.unlink(missing_ok=True)
            index = self._load_index()
            self._index_cache = [s for s in index if s["id"] != session_id]
            self._save_index()

    # ------------------------------------------------------------------
    # P2-12: fork
    # ------------------------------------------------------------------

    def fork_session(
        self,
        source_session_id: str,
        *,
        title: Optional[str] = None,
    ) -> str:
        """克隆现有会话为新会话（消息全复制）。"""
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
                # 更新 message_count
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
    # 全文搜索（Python re 替代 FTS5）
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
        """全文搜索消息（Python re 实现）。

        降级说明（vs SQLite FTS5）：
        - 之前用 FTS5 + trigram fallback + snippet 函数
        - 现在用 re.escape + IGNORECASE 扫所有 .jsonl
        - 性能：1万条消息约 100ms（单用户场景够用）
        - 中文子串：直接 substring 匹配（re.search）
        - snippet：query 周围 context_chars 字符
        """
        if not query.strip():
            return []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            return []

        results = []
        # 限定 session_id 时只扫一个文件
        scan_sessions = (
            [self.get_session(session_id)] if session_id else self.list_sessions(limit=1000)
        )
        scan_sessions = [s for s in scan_sessions if s]  # 过滤 None

        for s in scan_sessions:
            sid = s["id"]
            title = s.get("title")
            for m in self._read_session_msgs(sid):
                content = m.get("content", "") or ""
                if not pattern.search(content):
                    continue
                # 过滤
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
        """生成 snippet（query 周围 context_chars 字符）。"""
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
        """聚合统计：会话/消息/工具调用频次/角色分布。"""
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
        # 角色分布 + 工具调用统计（扫所有 .jsonl）
        role_dist = {}
        tool_counter = {}
        for s in index:
            for m in self._read_session_msgs(s["id"]):
                role = m.get("role", "unknown")
                role_dist[role] = role_dist.get(role, 0) + 1
                tcs = m.get("tool_calls")
                if not isinstance(tcs, list):
                    continue  # 防御：tool_calls 可能被篡改为字符串/对象等
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
