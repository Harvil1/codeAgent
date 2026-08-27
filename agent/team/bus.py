"""消息总线（成员间发信的邮局）：用 JSONL 文件落地。

每个成员一个收件箱文件（~/.OmniMate/.team/inbox/{name}.jsonl，一行一封信）。
发信 = 往对方的文件里追加一行；收信 = 把自己文件整个读走并清空。

为什么用文件 + 锁而不是消息队列中间件：团队成员是各自独立的子进程，
文件是最简单可靠的共享方式。所有读写都套跨平台文件锁串行化
（Windows 用 msvcrt，类 Unix 用 fcntl——提前做过技术验证 spike）。
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

# 合法的消息类型：普通消息 / 请求（要回音）/ 响应（回音）/ 关停指令
VALID_TYPES = {"message", "request", "response", "shutdown"}


@dataclass
class TeamMessage:
    """总线上流转的一封信。"""
    id: str
    from_: str        # 发件人。字段名带下划线是因为 `from` 是 Python 关键字，不能用
    to: str
    type: str
    content: str
    ts: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 跨平台文件锁（同一时刻只让一个进程碰文件，防止互相踩）
# ---------------------------------------------------------------------------

def _acquire_lock(fileobj, timeout: float = 30.0):
    """抢锁：拿到才准动文件，抢不到就反复试。

    参数：
        fileobj：已打开的锁文件
        timeout：最多等多久（秒），默认 30

    拿到锁后返回；超时还抢不到就抛 TimeoutError。
    """
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
    """还锁：用完就放，别占着茅坑。参数 fileobj 是锁文件。无返回值。"""
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)


class MessageBusLockTimeout(RuntimeError):
    """抢锁超时的专用异常（fail-closed：宁可拒绝操作，也不无锁硬干）。"""


def _with_lock(lock_path: Path, fn):
    """锁住某件事再干：抢到 lock_path 的独占锁后执行 fn，最后还锁。

    参数：
        lock_path：锁文件路径
        fn：要在锁内执行的操作（无参函数），返回值原样透传

    历史踩坑：锁超时必须 fail-closed——fail-open（超时照干不误）会让
    read_inbox 的「读全量 + 清空」在无锁并发下丢信：
    A 和 B 同时读到同一批信，A 先清空，B 再清空时会把 C 刚写进来的
    新信一起清掉。对邮箱来说，丢信比「这次操作失败」伤害大得多，
    所以超时抛 MessageBusLockTimeout。
    死锁兜底交给调用方：捕获这个异常重试或降级
    （各个调用点外面都已包了 try/except）。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lf:
        try:
            _acquire_lock(lf)
        except TimeoutError as e:
            logger.error("file lock 超时，fail-closed 拒绝操作: %s", lock_path)
            raise MessageBusLockTimeout(str(e)) from e
        try:
            return fn()
        finally:
            _release_lock(lf)


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class MessageBus:
    """基于 JSONL 文件的消息总线（本目录各成员的公共邮局）。"""

    def __init__(self, *, team_dir: Path):
        """建好收件箱目录和锁目录。

        参数：team_dir：团队工作目录，收件箱在 team_dir/inbox/，
        锁文件在 team_dir/locks/。
        """
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
        """发一封信：追加到收件人的收件箱文件末尾。

        参数：
            from_：发件人名字（from 是关键字所以带下划线）
            to：收件人名字
            type_：消息类型，必须是 VALID_TYPES 之一
            content：正文
            request_id：关联的请求 ID；type_='response' 时必填

        返回：新生成的 message_id。

        历史踩坑：response 不带 request_id 会变成「孤儿回音」
        ——没人知道它在回复谁，所以强制校验。
        """
        if type_ not in VALID_TYPES:
            raise ValueError(
                f"type_ 必须是 {VALID_TYPES} 之一，实际: {type_}"
            )
        # response 必须带能对上号的 request_id，否则就是孤儿回音
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

    # ---- 一问一答的便捷方法 ----
    def send_request(
        self, *, from_: str, to: str, content: str,
    ) -> str:
        """发一个「请求」：自动生成 req_ 前缀的回执编号。

        参数：
            from_：发件人名字
            to：收件人名字
            content：请求正文

        返回：request_id。调用方存着它，之后凭编号对回音。
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
        """回信：凭 request_id 对上原请求（type='response'）。

        参数：
            from_：回信人名字
            to：收回信的人（通常是原请求发起者）
            request_id：要回复的那个请求的编号；不传直接 ValueError
            content：回信正文

        返回：消息 id。
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
        """从一堆消息里挑出对得上编号的回音。

        参数：
            messages：消息列表（通常是刚 read_inbox 拿到的）
            request_id：要找的那个请求的编号

        返回：匹配的 response 消息；还没到就返回 None。

        典型用法：发完 send_request 后隔几秒读一次收件箱，用这个方法
        找回音，没找到就继续等下一轮。
        """
        for m in messages:
            if (
                m.type == "response"
                and m.request_id == request_id
            ):
                return m
        return None

    def read_inbox(self, name: str) -> List[TeamMessage]:
        """取信（一次全取走）：返回箱里所有消息，然后把箱子清空。

        参数：name：收件人名字。
        返回：TeamMessage 列表；箱子不存在就是空的。
        单行解析失败只记 warning 跳过那一行，不影响其他信。
        """
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
            # 取完清空内容但保留文件本身（省得下次还要判断建文件）
            inbox.write_text("", encoding="utf-8")
            return msgs

        return _with_lock(self._lock_path(name), _consume)

    def list_inboxes(self) -> List[str]:
        """列出现在有哪些收件箱。

        返回：名字列表（文件名去掉 .jsonl 后缀，即成员名）。无参数。
        """
        return [p.stem for p in self._inbox_dir.glob("*.jsonl")]
