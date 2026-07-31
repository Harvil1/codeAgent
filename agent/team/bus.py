"""JSONL 文件消息总线。

每个 agent 一个 inbox 文件 (~/.OmniMate/.team/inbox/{name}.jsonl)。
所有读写用文件锁序列化（Windows msvcrt / POSIX fcntl，spike 验证过）。
"""
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

VALID_TYPES = {"message", "request", "response", "shutdown"}


@dataclass
class TeamMessage:
    """单条消息。"""
    id: str
    from_: str        # 发送者（不用 built-in `from`）
    to: str
    type: str
    content: str
    ts: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 跨平台文件锁
# ---------------------------------------------------------------------------

def _acquire_lock(fileobj, timeout: float = 30.0):
    """获取独占锁。超时抛 TimeoutError。"""
    deadline = time.time() + timeout
    if sys.platform == "win32":
        import msvcrt
        while True:
            try:
                msvcrt.locking(fileobj.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"file lock timeout after {timeout}s")
                time.sleep(0.01)
    else:
        import fcntl
        while True:
            try:
                fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"file lock timeout after {timeout}s")
                time.sleep(0.01)


def _release_lock(fileobj):
    """释放锁。"""
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)


def _with_lock(lock_path: Path, fn):
    """获取 lock_path 的独占锁后执行 fn。

    锁超时时 fail-open（log warning + 执行 fn 不持锁），
    避免单进程死锁拖垮整个 agent。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        locked = False
        try:
            _acquire_lock(lf)
            locked = True
        except TimeoutError:
            logger.warning("file lock 超时，fail-open: %s", lock_path)
        try:
            return fn()
        finally:
            if locked:
                _release_lock(lf)


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class MessageBus:
    """JSONL 消息总线。"""

    def __init__(self, *, team_dir: Path):
        self._team_dir = Path(team_dir)
        self._inbox_dir = self._team_dir / "inbox"
        self._locks_dir = self._team_dir / "locks"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, name: str) -> Path:
        return self._inbox_dir / f"{name}.jsonl"

    def _lock_path(self, name: str) -> Path:
        return self._locks_dir / f"inbox-{name}.lock"

    def send(
        self,
        *,
        from_: str,
        to: str,
        type_: str,
        content: str,
        request_id: Optional[str] = None,
    ) -> str:
        """追加一条消息到 `to` 的 inbox。返回 message_id。

        P1-8 类型校验：type_='response' 时 request_id 必填（防孤儿 response）。
        """
        if type_ not in VALID_TYPES:
            raise ValueError(
                f"type_ 必须是 {VALID_TYPES} 之一，实际: {type_}"
            )
        # P1-8: response 必须配对 request_id
        if type_ == "response" and not request_id:
            raise ValueError(
                "type='response' 的消息必须传 request_id（防孤儿 response）"
            )
        msg_id = uuid.uuid4().hex[:12]
        msg = {
            "id": msg_id,
            "from": from_,
            "to": to,
            "type": type_,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
        }

        def _append():
            inbox = self._inbox_path(to)
            with open(inbox, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        _with_lock(self._lock_path(to), _append)
        return msg_id

    # ---- P1-8 NEW: request-response 协议化便捷方法 ----
    def send_request(
        self, *, from_: str, to: str, content: str,
    ) -> str:
        """发 type='request' 消息，自动生成 request_id（req_ 前缀）。

        返回 request_id，调用方拿它等响应。
        """
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        self.send(
            from_=from_, to=to, type_="request",
            content=content, request_id=request_id,
        )
        return request_id

    def send_response(
        self, *, from_: str, to: str, request_id: str, content: str,
    ) -> str:
        """用 request_id 回复（type='response'）。

        request_id 必填（None 时 raise ValueError）。
        """
        if not request_id:
            raise ValueError(
                "send_response 必须传 request_id（关联原 request）"
            )
        return self.send(
            from_=from_, to=to, type_="response",
            content=content, request_id=request_id,
        )

    @staticmethod
    def find_response(
        messages: List[TeamMessage], request_id: str,
    ) -> Optional[TeamMessage]:
        """从消息列表里找匹配 request_id 的 response。找不到返回 None。

        典型用法：调用方 send_request 后周期性 read_inbox，
        用 find_response 找出对应的 response，没找到就继续 poll。
        """
        for m in messages:
            if (
                m.type == "response"
                and m.request_id == request_id
            ):
                return m
        return None

    def read_inbox(self, name: str) -> List[TeamMessage]:
        """消费式读取：返回所有消息，清空文件。"""
        def _consume():
            inbox = self._inbox_path(name)
            if not inbox.exists():
                return []
            text = inbox.read_text(encoding="utf-8")
            msgs = []
            for line in text.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    msgs.append(TeamMessage(
                        id=data["id"],
                        from_=data["from"],
                        to=data["to"],
                        type=data["type"],
                        content=data["content"],
                        ts=data["ts"],
                        request_id=data.get("request_id"),
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("inbox 消息解析失败 (%s): %s", line[:100], e)
            # 清空（保留文件）
            inbox.write_text("", encoding="utf-8")
            return msgs

        return _with_lock(self._lock_path(name), _consume)

    def list_inboxes(self) -> List[str]:
        """列出所有 inbox 文件名（去 .jsonl 后缀）。"""
        return [p.stem for p in self._inbox_dir.glob("*.jsonl")]
