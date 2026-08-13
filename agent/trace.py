"""本地 trace sink：每次 LLM / 工具调用都记一条 jsonl。

通过 HookRegistry 接入（零侵入主循环）。fail-open：写盘失败只 log。

设计要点（与 CLAUDE.md 约定对齐）：
- 文件 I/O 一律 ``encoding="utf-8"``（Windows cp1252 会乱码）
- fail-open：写盘失败只 log warning，不抛异常（避免影响主流程）
- 线程安全：threading.Lock 保护并发写
- 每天一个 jsonl 文件，按日期命名 ``YYYY-MM-DD.jsonl``
- agent_id 默认 "main"，子代理可通过 set_agent_id 切换，或 emit 时覆盖
"""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class TraceSink:
    """本地 trace 落地器。每天一个 jsonl 文件。

    使用方式（Task 5 才接入 AIAgent）::

        sink = TraceSink(base_dir=Path("~/.OmniMate").expanduser())
        sink.emit("pre_llm_call", input_tokens=100, model="deepseek-chat")

    写盘路径：``<base_dir>/.trace/<YYYY-MM-DD>.jsonl``
    """

    def __init__(self, base_dir: Path):
        # 构造时即建目录；如果失败让调用方知道（构造失败比运行时 fail-open 更早暴露问题）
        self._trace_dir = Path(base_dir) / ".trace"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_agent_id = "main"

    def set_agent_id(self, agent_id: str) -> None:
        """子代理启动时切换 agent_id（用于区分 main / subagent_xxx）。

        Task 5 hook 接入时在 ``register_subagent_start`` 里调。
        """
        self._current_agent_id = agent_id

    def emit(self, event: str, **fields: Any) -> None:
        """追加一条 trace 记录到今天的 jsonl 文件。

        fail-open：写盘失败只 log warning，不抛异常。

        Args:
            event: 事件类型（pre_llm_call / post_llm_call / post_tool_use / ...）
            **fields: 任意附加字段。特殊键 ``agent_id`` 会覆盖当前 agent_id。
        """
        try:
            record: dict[str, Any] = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "event": event,
                "agent_id": fields.pop("agent_id", self._current_agent_id),
            }
            record.update(fields)
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = self._trace_dir / f"{date_str}.jsonl"
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logger.warning("trace emit fail-open: %s", e)

    def _read_day(self, date_str: str) -> List[dict]:
        """读取某天的全部记录（跳过空行/坏行）。"""
        path = self._trace_dir / f"{date_str}.jsonl"
        if not path.exists():
            return []
        records: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 坏行跳过（fail-open：不因单行损坏丢全部）
                continue
        return records

    def query(
        self,
        date_str: Optional[str] = None,
        agent_id: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """查询 trace 记录。

        Args:
            date_str: 日期（``YYYY-MM-DD``），None 表示今天
            agent_id: 过滤 agent_id（None = 不过滤）
            event: 过滤事件类型（None = 不过滤）
            limit: 返回最近 N 条（默认 100）

        Returns:
            匹配的记录列表（按时间升序，取最后 ``limit`` 条）
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        records = self._read_day(date_str)
        if agent_id:
            records = [r for r in records if r.get("agent_id") == agent_id]
        if event:
            records = [r for r in records if r.get("event") == event]
        return records[-limit:]

    def summary(self, date_str: Optional[str] = None) -> dict:
        """返回当日聚合统计。

        Returns:
            dict 包含：

            - ``total_events``：事件总数
            - ``by_event``：{event_type: count}
            - ``total_input_tokens``：input_tokens 累加
            - ``total_output_tokens``：output_tokens 累加
            - ``error_count``：含 "fail" 字样或 ``error`` 字段的事件数
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        records = self._read_day(date_str)
        by_event: dict[str, int] = {}
        total_in = 0
        total_out = 0
        error_count = 0
        for r in records:
            evt = r.get("event", "unknown")
            by_event[evt] = by_event.get(evt, 0) + 1
            total_in += int(r.get("input_tokens", 0) or 0)
            total_out += int(r.get("output_tokens", 0) or 0)
            if "fail" in evt or r.get("error"):
                error_count += 1
        return {
            "total_events": len(records),
            "by_event": by_event,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "error_count": error_count,
        }
