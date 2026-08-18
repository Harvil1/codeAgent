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
            # R19 #24：fields 值级秘密 redact（fail-open——redact 失败原样记录）
            try:
                from agent.secret_scanner import redact_fields
                fields = redact_fields(fields)
            except Exception:
                pass
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


# =============================================================================
# CCAR8 Task 5: trace sink hook 接入
# =============================================================================


def _register_trace_hooks(hooks_registry, sink: "TraceSink") -> None:
    """把 sink 接到 6 个 hook 点。fail-open（hook 内部已 try/except）。

    6 个 hook：
      - PRE_LLM_CALL / POST_LLM_CALL：token 用量追踪
      - POST_TOOL_USE / POST_TOOL_USE_FAILURE：工具调用追踪
      - SUBAGENT_START / SUBAGENT_STOP：子代理生命周期

    HookRegistry 各事件的 fn 签名不同，严格按调用约定写：
      - pre_llm_call:    fn(messages, tools)
      - post_llm_call:   fn(response)
      - post_tool_use:   fn(tool_name, args, result)
      - post_tool_use_failure: fn(payload: dict)   ← 不是 3 参数！
      - subagent_start/stop: fn(payload: dict)
    """
    hooks_registry.register_pre_llm_call(
        lambda messages, tools: (
            sink.emit(
                "pre_llm_call",
                input_tokens=_estimate_messages_tokens(messages),
                model=None,
            )
            or None  # 返回 None 表示不修改
        ),
        name="trace_pre_llm_call",
    )
    hooks_registry.register_post_llm_call(
        lambda response: (
            sink.emit(
                "post_llm_call",
                output_tokens=_extract_response_tokens(response),
                duration_ms=None,
                model=_extract_response_model(response),
            )
            or None
        ),
        name="trace_post_llm_call",
    )
    hooks_registry.register_post_tool_use(
        lambda tool_name, args, result: (
            sink.emit("post_tool_use", tool=tool_name) or None
        ),
        name="trace_post_tool_use",
    )
    # POST_TOOL_USE_FAILURE 的 fn 签名是 fn(payload: dict)（见 hooks.py:622）
    # 不是 (tool_name, args, result)——brief 原文是错的，这里按实际签名实现
    hooks_registry.register_post_tool_use_failure(
        lambda payload: (
            sink.emit(
                "tool_failed",
                tool=payload.get("tool"),
                error=str(payload.get("error", ""))[:200],
            )
            or None
        ),
        name="trace_post_tool_use_failure",
    )
    hooks_registry.register_subagent_start(
        lambda payload: sink.emit(
            "subagent_start",
            subagent_type=payload.get("subagent_type") or payload.get("subagent"),
            session_id=payload.get("session_id", ""),
        ),
        name="trace_subagent_start",
    )
    hooks_registry.register_subagent_stop(
        lambda payload: sink.emit(
            "subagent_stop",
            subagent_type=payload.get("subagent_type") or payload.get("subagent"),
            session_id=payload.get("session_id", ""),
        ),
        name="trace_subagent_stop",
    )


def _estimate_messages_tokens(messages: Optional[list]) -> int:
    """粗估 messages 总 token 数（混合中英文系数，R30d-D6）。

    OpenAI/Anthropic 都没有 client-side token 计数，这里用经验比例估：
    ASCII ≈ 4 字符/token，CJK ≈ 1.5 字符/token（中文实际 1-2 字符/token，
    此前统一 4 字符/token 对中文会低估 2.5 倍+，/trace 成本观测失真）。
    fail-open：任何异常返回 0。
    """
    try:
        ascii_chars = 0
        cjk_chars = 0
        for m in messages or []:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            for ch in str(content):
                if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" \
                        or "\uff00" <= ch <= "\uffef":
                    cjk_chars += 1
                else:
                    ascii_chars += 1
        # CJK: 1.5 字符/token → *2/3；ASCII: 4 字符/token → *1/4（整数运算）
        return ascii_chars // 4 + (cjk_chars * 2) // 3
    except Exception:
        return 0


def _extract_response_tokens(response) -> int:
    """从 LLM response 提取 completion_tokens（兼容对象/dict）。fail-open 返回 0。"""
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0
        # 对象路径
        ct = getattr(usage, "completion_tokens", None)
        if ct is None and isinstance(usage, dict):
            ct = usage.get("completion_tokens")
        return int(ct or 0)
    except Exception:
        return 0


def _extract_response_model(response) -> Optional[str]:
    """从 LLM response 提取 model 名（兼容对象/dict）。fail-open 返回 None。"""
    try:
        if hasattr(response, "model"):
            return response.model
        if isinstance(response, dict):
            return response.get("model")
    except Exception:
        pass
    return None
