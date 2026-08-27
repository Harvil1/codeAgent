"""异步信箱：发出去就算完事，对方什么时候看都行。

和 bus（消息总线——成员间发信的邮局，本目录 bus.py）的分工作个对比：
- bus 的 send_request：一问一答，发出去要等回音（同步）
- 本模块 Mailbox：像塞传单进信箱，投递完就走，不等人回（异步）

存储方式：每个成员一个信箱目录，inbox.jsonl 只追加不覆盖。
「已读」状态单独记在一个小账本（sidecar，read_ids.jsonl，也是只追加）——
标已读若重写整个 inbox 就是每次写全量，一万封邮件时反复全量重写
是 O(n²)；往小账本追加几行（O(k)）就快了。
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

# 文件锁直接复用 bus.py 里那套（不另造轮子）
from agent.team.bus import _with_lock


class Mailbox:
    """异步信箱：邮件一直躺着，直到被标已读或整箱清空。

    和 bus 的关键区别：bus 读一次就把信全拿走（信箱清空）；本类是
    队列语义，邮件留在箱子里，已读/未读都能随时再查。

    存储布局：
    - 邮件本体：{base}/.mailboxes/{name}/inbox.jsonl（只追加，永不重写）
    - 已读小账本：{base}/.mailboxes/{name}/read_ids.jsonl（只追加）
      读取时合并两处（邮件行内自带的 read:true ∪ 小账本），
      这样旧格式（没有小账本时）也照样能用。
    锁：复用 bus 的 _with_lock（Windows 用 msvcrt / 类 Unix 用 fcntl），
    每个信箱名独立一把锁，互不排队。
    """

    def __init__(self, base_dir: Path):
        """建好目录。

        参数：
            base_dir：根目录，信箱建在它下面的 .mailboxes/ 里，
            锁文件放在 base_dir/locks/ 里
        """
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
        """投一封信（异步，不等人回）。

        参数：
            to：收件人名字
            from_：发件人名字（避开 Python 关键字 from 所以带下划线）
            content：信的正文
            kind：邮件类型标签，默认 "message"

        返回：新邮件的 id 字符串。

        容错（fail-open）：写盘失败只记 warning、照常返回 msg_id，
        不抛异常——发信是辅助动作，不值得炸掉调用方。
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

        # 路径在闭包内部算：这样 try 能兜住路径构造本身的异常，fail-open 覆盖整段
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
        """读原始邮件列表（不管已读未读，也不合并小账本）。

        参数：name：信箱主人名字。
        返回：邮件 dict 列表；文件不存在或读失败返回空列表（出错只记日志）。
        """
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
        """读已读小账本。

        参数：name：信箱主人名字。
        返回：已读邮件 id 的集合；账本文件不存在（从没标过已读）返回空集。
        """
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
        """读全部邮件并合并已读状态（行内 read 标记 ∪ 小账本）。

        参数：name：信箱主人名字。返回：带最终 read 状态的邮件 dict 列表。
        """
        msgs = self._read_inbox_raw(name)
        sidecar_read = self._load_read_ids(name)
        if sidecar_read:
            for m in msgs:
                if not m.get("read", False) and m.get("id") in sidecar_read:
                    m["read"] = True
        return msgs

    def check_unread(self, name: str) -> List[dict]:
        """查未读邮件。

        参数：name：信箱主人名字。
        返回：未读邮件列表，按写入顺序排（先来的在前）。
        """
        return [m for m in self._read_all(name) if not m.get("read", False)]

    def check_all(self, name: str) -> List[dict]:
        """查整箱邮件（已读 + 未读都在）。

        参数：name：信箱主人名字。返回：全部邮件 dict 列表。
        """
        return self._read_all(name)

    def mark_read(self, name: str, msg_ids: List[str]) -> int:
        """批量标已读（像在信封上盖章）。

        参数：
            name：信箱主人名字
            msg_ids：要标的邮件 id 列表（不存在的 id 自动忽略）

        返回：本次真正新标上的数量（本来就已读的不重复计）。

        性能取舍：只往小账本追加几行（O(k)），绝不重写整个 inbox
        （一万封邮件时全量重写是 O(n²)，太慢）。
        容错（fail-open）：写盘失败只记 warning、返回 0，不抛异常。
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
        """清空整个信箱（邮件 + 已读小账本一起删）。

        参数：name：信箱主人名字。
        返回：删掉的邮件数量；失败只记 warning 返回 0（fail-open，不抛异常）。
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
