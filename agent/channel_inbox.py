"""ChannelInbox：接收 MCP server notifications 并落地为文件。

主循环 _assemble_turn_messages 检查 unconsumed，注入 ephemeral user 消息。

设计：
- push(server, payload) 在 MCP reader 线程被调（thread-safe，加锁）
- unconsumed(limit) 返回按 ts 升序的消息列表
- mark_consumed(ids) 批量删除已消费消息
- format_digest(msgs) 格式化为 LLM 可读字符串
- fail-open：所有写盘失败只 log，不抛
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
    """接收 MCP server notifications 并落地为文件。"""

    def __init__(self, base_dir: Path):
        self._inbox_dir = Path(base_dir) / ".inbox"
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # 单调递增计数器：同一秒内多条消息时保序（ts 只精确到秒）
        self._seq = 0

    def push(self, server: str, payload: dict) -> str:
        """MCP notification 到达时调用。返回 message_id。fail-open。

        线程安全：reader 线程调这个方法时，主线程可能正在 unconsumed。
        加锁避免写冲突。
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
            "_seq": seq,  # 内部字段：同秒内保序
        }
        # 每次 push 重新解析 Path（允许测试 monkey-patch _inbox_dir 为 str）
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
        """返回所有未消费消息（按 ts 升序，最多 limit 条）。"""
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
        # 按 (ts, _seq) 升序（早的在前；同秒内按 push 顺序）
        msgs.sort(key=lambda m: (m.get("ts", ""), m.get("_seq", 0)))
        return msgs

    def mark_consumed(self, msg_ids: List[str]) -> int:
        """批量删除已消费消息（已通过 user 消息呈现）。返回删除数。

        R30c-C7 消费语义（有意为之）：unconsumed 读文件列表 → 注入 user 消息
        → mark_consumed 按 id 删除，两步之间**无原子性**。注入后、删除前进程
        崩溃 → 下轮重复注入同批消息。这是 at-least-once 投递（重复好过丢失），
        消息 id 带 uuid 去重由模型侧容忍；不做消费位点持久化（复杂度不值）。
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
        """把消息列表格式化为 LLM 可读的字符串。"""
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            server = m.get("server", "unknown")
            payload = m.get("payload", {})
            ts = m.get("ts", "")
            # payload 简短序列化
            payload_str = json.dumps(payload, ensure_ascii=False)
            if len(payload_str) > 500:
                payload_str = payload_str[:500] + "..."
            lines.append(f"[{ts}] [{server}] {payload_str}")
        return "\n".join(lines)
