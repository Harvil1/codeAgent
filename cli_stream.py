"""流式回答渲染器（claude code 风格：无框直排）。

大白话：模型吐字像水龙头滴水，这个渲染器把水滴一行行接好直接排上屏——
不画框（claude code 同款），思考流用暗色斜体跟正文区分。

规则：
- 行缓冲：攒到换行才打整行（半行打出去遇到重排会撕烂）；
- 首个可见字符前先空一行（跟前面的工具块视觉分隔）；
- markdown 表格行进侧缓冲，块结束整块重排再打（cli_mdtables 负责
  CJK 对齐）；
- 超长半行按终端宽强制冲行（首字延迟感知优化——不能让用户干等
  第一个换行符）；
- 思考流（reasoning）暗色斜体、无框，且永远排在正文前面——回答
  内容先押后，等思考流结束再放；
- 每轮 LLM 调用（done 事件）冲掉缓冲；下一轮重新来。
"""

import logging
import shutil

from cli_mdtables import (
    is_table_divider,
    looks_like_table_row,
    realign_markdown_tables,
)
from cli_ui import emit_ansi

logger = logging.getLogger(__name__)

_RST = "\033[0m"       # ANSI 复位
_DIM = "\033[2;3m"     # 暗色 + 斜体（思考流用）


def _terminal_width_for_streaming() -> int:
    """流式正文可用的显示格数（无框直排，留 2 格防抖）。"""
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    return max(20, cols - 2)


def cprint(text: str) -> None:
    """统一打印漏斗：全程序唯一出口 cli_ui.emit_ansi。

    为什么统一出口：工作线程的打印必须搬进 UI 事件循环（否则把
    spinner/状态栏冻进滚动历史）；换肤/日志镜像将来也只改一处。
    """
    emit_ansi(text)


class StreamBoxRenderer:
    """流式回答渲染状态机（一轮 LLM 一个实例周期，可复用后 reset）。

    参数：
        print_fn: 打印函数（默认 cprint；verify 注入收集器单测）
    """

    def __init__(self, print_fn=None):
        self._print = print_fn or cprint
        self.reset()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """每轮 LLM 调用开始前清板。"""
        self._buf = ""                  # 半行缓冲
        self._opened = False            # 正文块开张了吗（开张前先空一行）
        self._reasoning_opened = False  # 思考流开始了吗（开始前先空一行）
        self._reasoning_buf = ""        # 思考半行缓冲
        self._deferred = ""             # 思考流没结束前押后的正文
        self._table_buf = []            # 表格侧缓冲
        self._in_table = False
        self._last_progress = ("", 0.0)  # 心跳去重：(消息, 上次打印时刻)

    # ------------------------------------------------------------------
    # 事件入口（cli 的流式回调把事件原样递进来）
    # ------------------------------------------------------------------

    def on_event(self, event: dict) -> None:
        """总分发：content/reasoning/tool_call_start/progress/done。"""
        try:
            etype = event.get("type")
            if etype == "content":
                self.on_delta(event.get("delta") or "")
            elif etype == "reasoning":
                self.on_reasoning_delta(event.get("delta") or "")
            elif etype == "tool_call_start":
                self.on_tool_boundary(event.get("name") or "?")
            elif etype == "progress":
                self.on_progress(
                    event.get("message") or "",
                    int(event.get("elapsed_seconds") or 0),
                )
            elif etype == "done":
                self.flush()
        except Exception:
            logger.exception("流式渲染事件处理失败（不中断流）")

    # ------------------------------------------------------------------
    # 思考框
    # ------------------------------------------------------------------

    def on_reasoning_delta(self, text: str) -> None:
        """思考流：暗色斜体逐行直打（无框）。正文押后，思考先行。"""
        if not text:
            return
        if not self._reasoning_opened:
            self._reasoning_opened = True
            self._print("\n")   # 跟前面的工具块空一行隔开
        self._reasoning_buf += text
        while "\n" in self._reasoning_buf:
            line, self._reasoning_buf = self._reasoning_buf.split("\n", 1)
            if line.strip():
                self._print(f"{_DIM}{line}{_RST}")
        # 思考半行太长也冲（80 格）
        if len(self._reasoning_buf) >= 80:
            self._print(f"{_DIM}{self._reasoning_buf}{_RST}")
            self._reasoning_buf = ""

    def _close_reasoning_stream(self) -> None:
        """思考流收尾：冲掉半行（押后的正文交给调用方补发）。"""
        if not self._reasoning_opened:
            return
        if self._reasoning_buf.strip():
            self._print(f"{_DIM}{self._reasoning_buf}{_RST}")
        self._reasoning_buf = ""
        self._reasoning_opened = False

    # ------------------------------------------------------------------
    # 回答框
    # ------------------------------------------------------------------

    def on_delta(self, text: str) -> None:
        """回答流：直排 → 行缓冲 → 整行打；表格行进侧缓冲。"""
        if not text:
            return

        # 思考流还没结束：正文押后（思考永远排在正文前面）
        if self._reasoning_opened:
            self._deferred += text
            return

        # 首个可见字符才开张（lstrip 掉开头的空行，先补一个空行分隔）
        if not self._opened:
            text = text.lstrip("\n")
            if not text:
                return
            self._opened = True
            self._print("\n")

        self._buf += text
        self._emit_complete_lines()

        # TTFT 感知：超长半行强制冲行（表格形态的半行除外——要整块重排）
        if self._buf and not self._in_table \
                and not self._buf.lstrip().startswith("|"):
            wrap_w = max(40, _terminal_width_for_streaming())
            while len(self._buf) >= wrap_w:
                cut = self._buf.rfind(" ", 0, wrap_w)
                if cut <= 0:
                    cut = wrap_w   # 一整条拆不开的串——硬折
                chunk, self._buf = (
                    self._buf[:cut],
                    self._buf[cut:].lstrip(" "),
                )
                self._emit_one(chunk)

    def _emit_one(self, line: str) -> None:
        """打一行正文（无缩进直排）。"""
        self._print(line)

    def _emit_complete_lines(self) -> None:
        """把缓冲里的整行打掉；表格行攒侧缓冲。"""

        def _flush_table_buf():
            buf = self._table_buf
            self._table_buf = []
            self._in_table = False
            if not buf:
                return
            block = realign_markdown_tables(
                "\n".join(buf), _terminal_width_for_streaming()
            )
            for ln in block.split("\n"):
                self._emit_one(ln)

        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._in_table:
                if looks_like_table_row(line) or is_table_divider(line):
                    self._table_buf.append(line)
                    continue
                _flush_table_buf()   # 表格块结束，先冲表再打当前行
            elif looks_like_table_row(line):
                self._table_buf.append(line)
                self._in_table = True
                continue
            self._emit_one(line)

    # ------------------------------------------------------------------
    # 边界事件
    # ------------------------------------------------------------------

    def on_tool_boundary(self, name: str) -> None:
        """模型切去调工具：冲掉半行缓冲就完事（工具事件行另有渲染器负责）。"""
        self._flush_partial_line()

    def on_progress(self, message: str, elapsed: int) -> None:
        """子代理运行中的进度心跳（框保持开着，冲半行防撕行）。

        去重：多个并行子代理的心跳常常是同一句「任务仍在执行...」，
        同一消息 45 秒内只打一次——不然一屏全是复读机。
        """
        if not message:
            return
        import time as _time
        now = _time.monotonic()
        last_msg, last_ts = self._last_progress
        if message == last_msg and (now - last_ts) < 45:
            return
        self._last_progress = (message, now)
        self._flush_partial_line()
        self._print(f"{_DIM}⟳ 子代理[{elapsed}s] {message}{_RST}")

    def _flush_partial_line(self) -> None:
        """把半行缓冲冲掉（不打关框线）——给插入行让路。"""
        # 表格侧缓冲也一起冲（表格中途被打断就按已到的行重排）
        if self._table_buf:
            block = realign_markdown_tables(
                "\n".join(self._table_buf), _terminal_width_for_streaming()
            )
            for ln in block.split("\n"):
                self._emit_one(ln)
            self._table_buf = []
            self._in_table = False
        if self._buf:
            self._emit_one(self._buf)
            self._buf = ""

    def flush(self) -> None:
        """一轮结束（done 事件）：收思考流 → 冲表格/半行 → 清板。"""
        # 押后的正文在思考流结束后补发（思考永远在前）
        deferred, self._deferred = self._deferred, ""
        self._close_reasoning_stream()
        if deferred:
            self.on_delta(deferred)
        self._flush_partial_line()
        self.reset()
