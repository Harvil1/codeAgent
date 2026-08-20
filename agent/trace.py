"""本地运行日志（trace）：每次调 LLM、用工具，都顺手记一行到本地 jsonl 文件。

打个比方：这就像给 agent 装一个"行车记录仪"——不干预开车，只默默录像，
事后可以用 /trace 命令回放"今天花了多少 token、哪些工具出错、子代理干了啥"。

在项目里的位置：通过 HookRegistry（钩子注册表——在各关键节点插回调的机制）
挂进主循环，零侵入（主循环代码一行不用改）。记录失败不报错不影响主流程
（fail-open，宁可不记也不能拖垮对话）。

设计要点（与 CLAUDE.md 约定对齐）：
- 文件读写一律指定 ``encoding="utf-8"``（Windows 默认编码 cp1252 会乱码）
- fail-open：写盘失败只打 log warning 不抛异常（记录器不能反过来害了主流程）
- 线程安全：用 threading.Lock（一把锁）保证多线程同时写不串行错乱
- 每天一个文件，文件名就是日期 ``YYYY-MM-DD.jsonl``
- 记录归属的 agent_id 默认 "main"（主代理）；子代理用 set_agent_id 切换自己的
  名字，或者在单次 emit 时临时指定
"""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class TraceSink:
    """记录器的本体：负责把一条条 trace 记录写进当天的 jsonl 文件。

    用法示例（Task 5 通过 hook 接入 AIAgent，平时不直接调）::

        sink = TraceSink(base_dir=Path("~/.OmniMate").expanduser())
        sink.emit("pre_llm_call", input_tokens=100, model="deepseek-chat")

    写盘位置：``<base_dir>/.trace/<YYYY-MM-DD>.jsonl``
    """

    def __init__(self, base_dir: Path):
        # 构造时就建目录，建失败当场报错——问题越早暴露越好，
        # 别等运行中 fail-open 静默吞掉半天的记录才发现目录根本不存在
        self._trace_dir = Path(base_dir) / ".trace"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_agent_id = "main"

    def set_agent_id(self, agent_id: str) -> None:
        """换当前记录归属的名字。给子代理用：启动时调一下，之后的记录都算它的。

        参数：
            agent_id：新名字，比如 "subagent_explore"（默认是 "main" 主代理）。
        Task 5 接入 hook 时，在 ``register_subagent_start``（子代理启动钩子）里调。
        """
        self._current_agent_id = agent_id

    def emit(self, event: str, **fields: Any) -> None:
        """往今天的 jsonl 文件末尾追加一条记录（一行一个 JSON 对象）。

        记录失败绝不抛异常（fail-open）——只打 warning，不连累主流程。

        参数：
            event：事件类型名（如 pre_llm_call / post_llm_call / post_tool_use）。
            **fields：随事件附带的其他信息，想带什么带什么。其中特殊键
                ``agent_id`` 可以临时顶掉当前默认归属（只对这一条生效）。
        """
        try:
            record: dict[str, Any] = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "event": event,
                "agent_id": fields.pop("agent_id", self._current_agent_id),
            }
            # R19 #24 加的保险：落盘前先给字段值"打码"——扫出疑似密钥就替换掉。
            # 打码本身失败也不拦（fail-open），原样记录总比丢记录强
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
        """读回某一天的全部记录。空行和解析不了的坏行直接跳过，不因一粒老鼠屎坏一锅粥。

        参数：
            date_str：日期字符串（``YYYY-MM-DD``）。

        返回：
            当天所有能解析的记录组成的列表。
        """
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
                # 单行坏了只丢这一行（fail-open：不能因为一行损坏把整天记录都不要了）
                continue
        return records

    def query(
        self,
        date_str: Optional[str] = None,
        agent_id: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """按条件翻查 trace 记录（/trace 命令背后调的就是它）。

        参数：
            date_str：查哪一天（``YYYY-MM-DD`` 格式）；None 表示今天。
            agent_id：只看某个 agent 的记录；None 表示不挑。
            event：只看某类事件；None 表示不挑。
            limit：最多返回最近多少条（默认 100）。

        返回：
            符合条件的记录列表，按时间从早到晚排，只取最后 limit 条。
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
        """把一天的记录汇总成一张"账单"：总共多少事件、花了多少 token、出过错没有。

        参数：
            date_str：统计哪一天（``YYYY-MM-DD``）；None 表示今天。

        返回：
            dict，包含：

            - ``total_events``：事件总条数
            - ``by_event``：各事件类型分别多少条 {事件类型: 条数}
            - ``total_input_tokens``：所有记录里 input_tokens 加起来
            - ``total_output_tokens``：所有记录里 output_tokens 加起来
            - ``error_count``：失败事件数（事件名带 "fail" 或带 error 字段的）
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
# CCAR8 Task 5: 把 trace 记录器挂到 hook 上（这一步之后才算真正接到主循环）
# =============================================================================


def _register_trace_hooks(hooks_registry, sink: "TraceSink") -> None:
    """把记录器接到 6 个钩子点上，agent 一有动静就自动记一笔。

    参数：
        hooks_registry：钩子注册表（agent/hooks.py 的 HookRegistry）。
        sink：要接上的 TraceSink 实例。

    整体 fail-open：钩子回调内部已经 try/except，出错也不会打断主流程。

    6 个钩子分别管：
      - PRE_LLM_CALL / POST_LLM_CALL：调 LLM 前后（追踪 token 用量）
      - POST_TOOL_USE / POST_TOOL_USE_FAILURE：工具调用成功/失败
      - SUBAGENT_START / SUBAGENT_STOP：子代理开始/结束

    注意坑：HookRegistry 每种事件要求的回调参数个数不一样，必须严格照着写：
      - pre_llm_call:    fn(messages, tools)
      - post_llm_call:   fn(response)
      - post_tool_use:   fn(tool_name, args, result)
      - post_tool_use_failure: fn(payload: dict)   ← 是 1 个 dict 参数，不是 3 参数！
      - subagent_start/stop: fn(payload: dict)
    """
    hooks_registry.register_pre_llm_call(
        lambda messages, tools: (
            sink.emit(
                "pre_llm_call",
                input_tokens=_estimate_messages_tokens(messages),
                model=None,
            )
            or None  # 钩子约定：回调返回 None 表示"不改动任何东西，只是旁观"
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
    # 历史踩坑：POST_TOOL_USE_FAILURE 的回调签名是 fn(payload: dict)（见
    # hooks.py:622），不是 (tool_name, args, result)——当初设计简报上写错了，
    # 这里按真实签名实现，写错参数签名就是静默失效
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
    """粗略估算一批消息大约占多少 token（混合中英文比例，R30d-D6 调整）。

    为什么只能估：OpenAI/Anthropic 都没提供在本地数 token 的官方办法，
    只能按经验比例算：英文约 4 个字符算 1 个 token，中文约 1.5 个字符算
    1 个 token。

    参数：
        messages：消息列表（可以为 None）。

    返回：
        估算的 token 总数。任何异常都返回 0（fail-open，估算器不能崩）。

    历史踩坑：以前不分中英文统一按 4 字符/token 算，但中文实际 1-2 字符就是
    一个 token，等于把中文成本低估了 2.5 倍以上，/trace 看成本完全失真。
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
        # 换算：中文 1.5 字符/token 即除以 3/2，英文 4 字符/token 即除以 4（用整数运算避免浮点）
        return ascii_chars // 4 + (cjk_chars * 2) // 3
    except Exception:
        return 0


def _extract_response_tokens(response) -> int:
    """从 LLM 的返回结果里抠出输出 token 数（completion_tokens）。

    参数：
        response：LLM 返回结果，可能是对象也可能是 dict（两条路都兼容）。

    返回：
        输出 token 数；找不到或出错返回 0（fail-open）。
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0
        # 先当对象试（属性访问），失败再当 dict 试
        ct = getattr(usage, "completion_tokens", None)
        if ct is None and isinstance(usage, dict):
            ct = usage.get("completion_tokens")
        return int(ct or 0)
    except Exception:
        return 0


def _extract_response_model(response) -> Optional[str]:
    """从 LLM 的返回结果里抠出用的是哪个模型（model 名）。

    参数：
        response：LLM 返回结果，对象和 dict 两种形态都兼容。

    返回：
        模型名字符串；取不到返回 None（fail-open）。
    """
    try:
        if hasattr(response, "model"):
            return response.model
        if isinstance(response, dict):
            return response.get("model")
    except Exception:
        pass
    return None
