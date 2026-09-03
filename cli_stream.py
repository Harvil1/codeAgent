"""流式回答框渲染器（从 hermes 的 _emit_stream_text 一族移植）。

大白话：模型吐字像水龙头滴水，这个渲染器负责把水滴接进一只
「圆角框」里一行行摆好——

    ╭─ ⚕ CodeAgent ─────────────╮
        第一行回答……
        第二行回答……
    ╰───────────────────────────╯

规则（跟 hermes 学的）：
- 行缓冲：攒到换行才打整行（半行打出去遇到重排会撕烂）；
- 首个可见字符才开框头（纯工具轮不画空框）；
- markdown 表格行进侧缓冲，块结束整块重排再打（cli_mdtables 负责
  CJK 对齐）；
- 超长半行按终端宽强制冲行（首字延迟感知优化——不能让用户干等
  第一个换行符）；
- 思考流（reasoning）画在暗色框里，且永远排在回答框前面——回答
  内容先押后，等思考框关了再放；
- 每轮 LLM 调用（done 事件）冲掉缓冲并关框；下一轮重新开框。
"""

import logging
import shutil

from cli_mdtables import (
    _disp_width,
    is_table_divider,
    looks_like_table_row,
    realign_markdown_tables,
)

logger = logging.getLogger(__name__)

_STREAM_PAD = "    "   # 流式正文 4 格缩进（跟 Panel 内边距对齐）
_RST = "\033[0m"       # ANSI 复位
_DIM = "\033[2m"       # 暗色（思考框用）


def _terminal_width_for_streaming() -> int:
    """流式框内可用的显示格数（去掉缩进 + 边距，留 2 格防抖）。"""
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    return max(20, cols - len(_STREAM_PAD) - 2)


def _box_width() -> int:
    """整框宽度（= 终端宽，地板 32——窄终端上框头数学不能出负数）。"""
    try:
        w = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        w = 80
    return max(32, int(w or 80))


def cprint(text: str) -> None:
    """统一打印漏斗：ANSI 文本走 prompt_toolkit 的 print_formatted_text。

    为什么统一出口：patch_stdout 会把打印排队到操作台重绘之后——
    任何线程直接 print 也安全，但走这一个口子，将来要加节流/换肤/
    日志镜像只改一处。失败退回裸 print（不许因为打印挂了断流）。
    """
    try:
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import ANSI
        print_formatted_text(ANSI(text))
    except Exception:
        try:
            print(text)
        except Exception:
            pass


def _skin():
    """拿当前皮肤（拿不到给 None，调用方都有兜底）。"""
    try:
        import cli_skin
        return cli_skin.get_active_skin()
    except Exception:
        return None


def _hex_ansi(key: str, fallback: str = "", *, bold: bool = False) -> str:
    """皮肤色键 → 24bit ANSI 转义（mono 皮肤/拿不到给空串=不上色）。"""
    skin = _skin()
    hex_color = skin.get_color(key, "") if skin else ""
    if not hex_color:
        return ""
    try:
        import cli_skin
        return cli_skin.hex_to_truecolor_ansi(hex_color, bold=bold)
    except Exception:
        return ""


class StreamBoxRenderer:
    """一只流式回答框的状态机（一轮 LLM 一个实例周期，可复用后 reset）。

    参数：
        print_fn: 打印函数（默认 cprint；verify 注入收集器单测）
        width_fn: 框宽函数（默认 _box_width；测试可钉死）
    """

    def __init__(self, print_fn=None, width_fn=None):
        self._print = print_fn or cprint
        self._width_fn = width_fn or _box_width
        self.reset()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """每轮 LLM 调用开始前清板。"""
        self._buf = ""                  # 半行缓冲
        self._box_opened = False        # 回答框头打了吗
        self._text_ansi = ""            # 正文色（开框时按皮肤算好）
        self._reasoning_opened = False  # 思考框头打了吗
        self._reasoning_buf = ""        # 思考半行缓冲
        self._deferred = ""             # 思考框没关前押后的正文
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
        """思考流：暗色框逐行打。正文押后，等思考框关了再放。"""
        if not text:
            return
        if not self._reasoning_opened:
            self._reasoning_opened = True
            label = "思考"
            w = self._width_fn()
            fill = w - 2 - _disp_width(label)
            self._print(f"{_DIM}┌─{label}{'─' * max(fill, 0)}┐{_RST}")
        self._reasoning_buf += text
        while "\n" in self._reasoning_buf:
            line, self._reasoning_buf = self._reasoning_buf.split("\n", 1)
            if line.strip():
                self._print(f"{_DIM}{line}{_RST}")
        # 思考半行太长也冲（80 格，hermes 同款阈值）
        if len(self._reasoning_buf) >= 80:
            self._print(f"{_DIM}{self._reasoning_buf}{_RST}")
            self._reasoning_buf = ""

    def _close_reasoning_box(self) -> None:
        """关思考框（押后的正文交给调用方补发）。"""
        if not self._reasoning_opened:
            return
        if self._reasoning_buf.strip():
            self._print(f"{_DIM}{self._reasoning_buf}{_RST}")
        self._reasoning_buf = ""
        self._reasoning_opened = False
        w = self._width_fn()
        self._print(f"{_DIM}└{'─' * max(w - 2, 0)}┘{_RST}")

    # ------------------------------------------------------------------
    # 回答框
    # ------------------------------------------------------------------

    def on_delta(self, text: str) -> None:
        """回答流：开框 → 行缓冲 → 整行打；表格行进侧缓冲。"""
        if not text:
            return

        # 思考框还开着：正文押后（思考永远排在回答前面）
        if self._reasoning_opened:
            self._deferred += text
            return

        # 首个可见字符才开框头（lstrip 掉开头的空行）
        if not self._box_opened:
            text = text.lstrip("\n")
            if not text:
                return
            self._box_opened = True
            skin = _skin()
            label = (skin.get_branding("response_label", "⚕ CodeAgent")
                     if skin else "⚕ CodeAgent")
            accent = _hex_ansi("response_border", bold=True)
            self._text_ansi = _hex_ansi("banner_text")
            w = self._width_fn()
            fill = w - 2 - _disp_width(label)
            self._print(
                f"\n{accent}╭─{label}{'─' * max(fill, 0)}╮{_RST}"
            )

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
        """打一行框内正文（缩进 + 正文色）。"""
        tc = self._text_ansi
        self._print(f"{_STREAM_PAD}{tc}{line}{_RST}" if tc
                    else f"{_STREAM_PAD}{line}")

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
        """模型切去调工具：冲半行 + 关框 + 打一行准备提示。"""
        self._flush_partial_line()
        self._close_box()
        prefix = "┊"
        skin = _skin()
        if skin:
            prefix = skin.tool_prefix or "┊"
        self._print(f"{_DIM}{prefix} ⟳ 准备调用 {name}…{_RST}")

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

    def _close_box(self) -> None:
        """关回答框（下一轮 on_delta 重新开框）。"""
        if not self._box_opened:
            return
        self._flush_partial_line()
        self._box_opened = False
        self._text_ansi = ""
        accent = _hex_ansi("response_border", bold=True)
        w = self._width_fn()
        self._print(f"{accent}╰{'─' * max(w - 2, 0)}╯{_RST}")

    def flush(self) -> None:
        """一轮结束（done 事件）：关思考框 → 冲表格/半行 → 关回答框 → 清板。"""
        # 押后的正文在思考框关闭后补发（思考永远在前）
        deferred, self._deferred = self._deferred, ""
        self._close_reasoning_box()
        if deferred:
            self.on_delta(deferred)
        self._close_box()
        self.reset()
