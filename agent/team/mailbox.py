"""异步队列邮箱，补 MessageBus 的同步 request-response。

语义对比：
- MessageBus.send_request → 等响应（同步）
- Mailbox.send → 异步投递（不等响应，对方何时处理都行）

存储：每个 teammate 一个 mailbox 目录，inbox.jsonl 追加。
read 状态走 sidecar（read_ids.jsonl 追加）——mark_read O(k) 写而非
全量重写 inbox（压力测试教训：万封邮件时全量重写是 O(n²)，500 轮 30s）。
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

# 复用 MessageBus 的文件锁
from agent.team.bus import _with_lock


class Mailbox:
    """异步队列邮箱。

    与 MessageBus 的区别：
    - MessageBus 是同步 request-response（read_inbox 消费式清空）
    - Mailbox 是异步队列（保留邮件直到 mark_read / clear）

    存储：
    - 邮件：{base}/.mailboxes/{name}/inbox.jsonl（append-only，永不重写）
    - 已读：{base}/.mailboxes/{name}/read_ids.jsonl（append-only sidecar）
      读取时合并（inbox 行内 read:true ∪ sidecar），向后兼容旧格式。
    锁：复用 _with_lock（Windows msvcrt / POSIX fcntl），每个 name 一把锁。
    """

    def __init__(self, base_dir: Path):
        self._base = Path(base_dir) / ".mailboxes"
        self._locks_dir = Path(base_dir) / "locks"
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, name: str) -> Path:
        return self._base / name / "inbox.jsonl"

    def _read_ids_path(self, name: str) -> Path:
        return self._base / name / "read_ids.jsonl"

    def _lock_path(self, name: str) -> Path:
        return self._locks_dir / f"mailbox-{name}.lock"

    def send(
        self,
        *,
        to: str,
        from_: str,
        content: str,
        kind: str = "message",
    ) -> str:
        """投递邮件（异步，不等响应）。

        fail-open：写盘失败只 log warning，返回 msg_id（不抛异常）。
        """
        msg_id = uuid.uuid4().hex[:12]
        msg = {
            "id": msg_id,
            "from": from_,
            "to": to,
            "kind": kind,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "read": False,
        }

        # path 和 lock_path 在闭包里算，fail-open 时整段 try 捕获（含路径构造异常）
        def _append():
            path = self._inbox_path(to)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        try:
            _with_lock(self._lock_path(to), _append)
        except Exception as e:
            logger.warning("mailbox send 失败（fail-open）: %s", e)
        return msg_id

    def _read_inbox_raw(self, name: str) -> List[dict]:
        """读原始 inbox（不合并 read 状态）。"""
        path = self._inbox_path(name)
        if not path.exists():
            return []
        msgs = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning("mailbox read 失败: %s", e)
        return msgs

    def _load_read_ids(self, name: str) -> Set[str]:
        """读 sidecar 已读 id 集合（文件不存在返回空集）。"""
        path = self._read_ids_path(name)
        if not path.exists():
            return set()
        ids = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    ids.add(line)
        except Exception as e:
            logger.warning("mailbox read_ids 读失败: %s", e)
        return ids

    def _read_all(self, name: str) -> List[dict]:
        """读全部邮件并合并 read 状态（inbox 行内 read ∪ sidecar）。"""
        msgs = self._read_inbox_raw(name)
        sidecar_read = self._load_read_ids(name)
        if sidecar_read:
            for m in msgs:
                if not m.get("read", False) and m.get("id") in sidecar_read:
                    m["read"] = True
        return msgs

    def check_unread(self, name: str) -> List[dict]:
        """返回未读邮件（按 jsonl 原顺序，即写入顺序）。"""
        return [m for m in self._read_all(name) if not m.get("read", False)]

    def check_all(self, name: str) -> List[dict]:
        """返回所有邮件（已读 + 未读）。"""
        return self._read_all(name)

    def mark_read(self, name: str, msg_ids: List[str]) -> int:
        """批量标记已读。返回实际新标记的数量。

        O(k) 写：新 id 追加到 read_ids.jsonl sidecar（不重写 inbox）。
        压力测试教训：全量重写在万封邮件时 O(n²)，500 轮注入 30s。
        fail-open：写盘失败只 log warning，返回 0。
        """
        def _update():
            msgs = self._read_inbox_raw(name)
            valid_ids = {m.get("id") for m in msgs}
            known_read = {m["id"] for m in msgs if m.get("read", False)}
            known_read |= self._load_read_ids(name)
            to_mark = (set(msg_ids) & valid_ids) - known_read
            if to_mark:
                path = self._read_ids_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    for mid in to_mark:
                        f.write(mid + "\n")
            return len(to_mark)

        try:
            return _with_lock(self._lock_path(name), _update)
        except Exception as e:
            logger.warning("mailbox mark_read 失败: %s", e)
            return 0

    def clear(self, name: str) -> int:
        """清空 name 的 mailbox（inbox + read sidecar）。返回删除数量。

        fail-open：写盘失败只 log warning，返回 0。
        """
        def _do_clear():
            msgs = self._read_inbox_raw(name)
            count = len(msgs)
            inbox = self._inbox_path(name)
            inbox.parent.mkdir(parents=True, exist_ok=True)
            inbox.write_text("", encoding="utf-8")
            sidecar = self._read_ids_path(name)
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except Exception as e:
                    logger.warning("mailbox read_ids 删除失败: %s", e)
            return count

        try:
            return _with_lock(self._lock_path(name), _do_clear)
        except Exception as e:
            logger.warning("mailbox clear 失败: %s", e)
            return 0
