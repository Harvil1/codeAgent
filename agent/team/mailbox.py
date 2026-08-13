"""异步队列邮箱，补 MessageBus 的同步 request-response。

语义对比：
- MessageBus.send_request → 等响应（同步）
- Mailbox.send → 异步投递（不等响应，对方何时处理都行）

存储：每个 teammate 一个 mailbox 目录，inbox.jsonl 追加。
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# 复用 MessageBus 的文件锁
from agent.team.bus import _with_lock


class Mailbox:
    """异步队列邮箱。

    与 MessageBus 的区别：
    - MessageBus 是同步 request-response（read_inbox 消费式清空）
    - Mailbox 是异步队列（保留邮件直到 mark_read / clear）

    存储：每个 teammate 一个 mailbox 目录（{base}/.mailboxes/{name}/inbox.jsonl）。
    锁：复用 _with_lock（Windows msvcrt / POSIX fcntl），每个 name 一把锁。
    """

    def __init__(self, base_dir: Path):
        self._base = Path(base_dir) / ".mailboxes"
        self._locks_dir = Path(base_dir) / "locks"
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, name: str) -> Path:
        return self._base / name / "inbox.jsonl"

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

    def _read_all(self, name: str) -> List[dict]:
        """读取 name 的全部邮件（jsonl 逐行解析）。"""
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

    def _write_all(self, name: str, msgs: List[dict]) -> None:
        """全量覆写 name 的 inbox。"""
        path = self._inbox_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)
        text += "\n" if msgs else ""
        path.write_text(text, encoding="utf-8")

    def check_unread(self, name: str) -> List[dict]:
        """返回未读邮件（按 jsonl 原顺序，即写入顺序）。"""
        return [m for m in self._read_all(name) if not m.get("read", False)]

    def check_all(self, name: str) -> List[dict]:
        """返回所有邮件（已读 + 未读）。"""
        return self._read_all(name)

    def mark_read(self, name: str, msg_ids: List[str]) -> int:
        """批量标记已读。返回实际新标记的数量。

        fail-open：写盘失败只 log warning，返回 0。
        """
        def _update():
            msgs = self._read_all(name)
            id_set = set(msg_ids)
            count = 0
            for m in msgs:
                if m.get("id") in id_set and not m.get("read", False):
                    m["read"] = True
                    count += 1
            self._write_all(name, msgs)
            return count

        try:
            return _with_lock(self._lock_path(name), _update)
        except Exception as e:
            logger.warning("mailbox mark_read 失败: %s", e)
            return 0

    def clear(self, name: str) -> int:
        """清空 name 的 mailbox。返回删除数量。

        fail-open：写盘失败只 log warning，返回 0。
        """
        def _do_clear():
            msgs = self._read_all(name)
            count = len(msgs)
            self._write_all(name, [])
            return count

        try:
            return _with_lock(self._lock_path(name), _do_clear)
        except Exception as e:
            logger.warning("mailbox clear 失败: %s", e)
            return 0
