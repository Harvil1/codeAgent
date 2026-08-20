"""ChannelInbox（频道收件箱）——接收外部 MCP 服务器主动推来的消息，存成文件等主循环来取。

背景：MCP（外部工具接入协议）的服务器除了等我们调用，也可能主动推
notification（通知）——像快递员直接往你家门口放包裹。本模块就是那个
"门口"：消息落成一个个小 JSON 文件，攒在 ~/.OmniMate 下。

在项目里的位置：agent/mcp_client.py 的 reader（读消息）线程往这里 push；
主循环（agent/__init__.py 的 _assemble_turn_messages）每轮来取走没消费的，
包成一条临时（ephemeral——只在本轮发给 LLM、不进历史）user 消息。

使用约定：
- push(server, payload)：MCP reader 线程调用（线程安全，内部加锁）
- unconsumed(limit)：取出还没消费的消息，按时间从早到晚排
- mark_consumed(ids)：批量删掉已消费的消息文件
- format_digest(msgs)：把消息拼成 LLM 能读的一段文字
- fail-open：所有写盘失败只记 log 不抛错——收件箱坏了不能拖垮对话
"""
import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class ChannelInbox:
    """收件箱本体：收消息（push）、看未读（unconsumed）、删已读（mark_consumed）。"""

    def __init__(self, base_dir: Path):
        """建好收件箱目录（<base_dir>/.inbox/）并准备锁。

        参数：
            base_dir：收件箱的父目录（一般是 omnimate home）。
        """
        self._inbox_dir = Path(base_dir) / ".inbox"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # 只增不减的序号：时间戳只精确到秒，同一秒来多条消息时靠它排出先后
        self._seq = 0

    def push(self, server: str, payload: dict) -> str:
        """MCP 通知到达时调用：把消息写成一个 JSON 文件存进收件箱。

        线程安全说明：这个方法在 MCP 的 reader 线程里跑，同一时刻主线程
        可能在读收件箱——所以要加锁，避免两边打架。

        参数：
            server：消息来自哪个 MCP 服务器（名字）。
            payload：消息内容 dict（服务器推什么就是什么）。
        返回：这条消息的 id（uuid 随机串，消费/去重时用）。
        异常：永不抛——写盘失败只记 warning（fail-open）。
        """
        msg_id = uuid.uuid4().hex[:12]
        ts_compact = datetime.now().strftime("%Y%m%dT%H%M%S")
        ts_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            seq = self._seq
            self._seq += 1
        msg = {
            "id": msg_id,
            "server": server,
            "payload": payload,
            "ts": ts_iso,
            "_seq": seq,  # 内部字段：同一秒内靠它保序
        }
        # 每次都重新包一遍 Path，是为了让测试能把 _inbox_dir 换成字符串路径
        path = Path(self._inbox_dir) / f"{ts_compact}_{seq:06d}_{msg_id}.json"
        try:
            with self._lock:
                path.write_text(
                    json.dumps(msg, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as e:
            logger.warning("channel push fail-open: %s", e)
        return msg_id

    def unconsumed(self, limit: int = 50) -> List[dict]:
        """取出所有还没消费的消息（时间早的排前面，最多 limit 条）。

        参数：
            limit：最多取几条（默认 50）。
        返回：消息 dict 的列表；读不了的文件直接跳过，不抛错。
        """
        try:
            files = sorted(Path(self._inbox_dir).glob("*.json"))
        except Exception:
            return []
        msgs = []
        for path in files:
            try:
                msg = json.loads(path.read_text(encoding="utf-8"))
                msgs.append(msg)
                if len(msgs) >= limit:
                    break
            except Exception:
                continue
        # 先按时间、再按序号排：早来的在前；同一秒内按 push 的先后
        msgs.sort(key=lambda m: (m.get("ts", ""), m.get("_seq", 0)))
        return msgs

    def mark_consumed(self, msg_ids: List[str]) -> int:
        """批量删掉已经消费过的消息文件（已通过 user 消息呈现给 LLM 的）。

        参数：
            msg_ids：要删的消息 id 列表。
        返回：实际删掉的条数。

        设计取舍（R30c-C7 裁决，有意为之）：流程是"unconsumed 读文件列表 →
        注入 user 消息 → mark_consumed 按 id 删文件"，前两步和第三步之间
        没有原子性保证。如果注入完、还没来得及删，进程崩了——下一轮会把
        同一批消息再注入一遍。这是 at-least-once（至少送达一次）的投递语义：
        重复好过丢失。消息带 uuid，重复了模型侧也能识别；没做"消费到第几条"
        的位点持久化，因为那点复杂度不值当。
        """
        id_set = set(msg_ids)
        count = 0
        try:
            for path in self._inbox_dir.glob("*.json"):
                try:
                    msg = json.loads(path.read_text(encoding="utf-8"))
                    if msg.get("id") in id_set:
                        path.unlink()
                        count += 1
                except Exception:
                    continue
        except Exception as e:
            logger.warning("channel mark_consumed fail-open: %s", e)
        return count

    def format_digest(self, msgs: List[dict]) -> str:
        """把消息列表拼成 LLM 能直接读的一段文字（每条一行：时间 + 来源 + 内容）。

        参数：
            msgs：消息 dict 的列表。
        返回：拼接好的字符串；超长的单条内容截到 500 字符（加 ... 结尾）；
            空列表返回空串。
        """
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            server = m.get("server", "unknown")
            payload = m.get("payload", {})
            ts = m.get("ts", "")
            # 内容转成紧凑 JSON；太长就截断，免得撑爆上下文
            payload_str = json.dumps(payload, ensure_ascii=False)
            if len(payload_str) > 500:
                payload_str = payload_str[:500] + "..."
            lines.append(f"[{ts}] [{server}] {payload_str}")
        return "\n".join(lines)
