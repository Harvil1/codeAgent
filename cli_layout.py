"""常驻操作台布局——hermes 默认 CLI 骨架的复刻（块①）。

设计一句话（跟 hermes 学的）：**屏幕上下分工**——
- 底部几行是 prompt_toolkit Application 管的「操作台」：
  spinner 行 + 多行输入区 + 分隔线 + 状态栏；
- 上面滚走的对话区不归布局管，还是 console.print + patch_stdout。
- 老主循环搬进工作线程，靠 _input_q 队列跟 UI 线程传话：
  Enter 键位回调是队列的生产端，老主循环还是消费端（零改动）。

本文件分三层：
1. 纯函数层（状态栏段/spinner 文案/提交小函数）——可单测，不碰终端；
2. 组装层 build_application（布局+键位+样式）；
3. 运行层（spinner 线程、节流重绘、跨线程退出请求）。
"""

import logging
import math
import os
import time

logger = logging.getLogger(__name__)

# spinner 动画帧（盲文点阵转圈，一圈 10 帧）
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# 状态栏三档宽度阈值（跟 hermes 学的：窄屏只留最要紧的）
_TIER_NARROW = 52    # < 52 列：模型 + 计时
_TIER_MEDIUM = 76    # < 76 列：加 目录/后台/正在跑的工具


def _fmt_elapsed(seconds) -> str:
    """秒数 → 状态栏计时文案（None/未开始返回空串）。"""
    if not seconds or seconds < 0:
        return ""
    return f"{int(seconds)}s"


def status_bar_segments(rt, width: int, elapsed_s=None) -> list:
    """状态栏内容段（纯函数）：按宽度三档过滤，返回段文本列表。

    大白话：状态栏像行李箱——宽屏全装上，中屏扔掉按键说明书，
    窄屏只留证件（模型）和手表（计时）。

    参数：
        rt: RuntimeContext（读 agent.model/workspace_cwd/bg_count/
            event_pending——全是现有黑板，零新状态）
        width: 终端列数
        elapsed_s: 回合已进行秒数（None=空闲）

    返回：str 列表，调用方用 " │ " 拼接。
    """
    segs = []
    try:
        model = getattr(getattr(rt, "agent", None), "model", "") or ""
        if model:
            segs.append(f"⚡{model}")
    except Exception:
        pass
    if width >= _TIER_NARROW:
        try:
            cwd = getattr(rt, "workspace_cwd", "") or ""
            if cwd:
                tail = str(cwd).replace("\\", "/").rstrip("/").split("/")[-1]
                segs.append(f"📂{tail}")
        except Exception:
            pass
        bg = getattr(rt, "bg_count", None)
        if bg:
            segs.append(f"☂{bg}个后台")
        pend = getattr(rt, "event_pending", None)
        if pend:
            segs.append(f"◐{pend[-1]}")   # 正在跑的工具（最晚出发的）
    if elapsed_s is not None:
        t = _fmt_elapsed(elapsed_s)
        if t:
            segs.append(t)
    if width >= _TIER_MEDIUM:
        segs.append("Enter发送 Alt+↵换行")
    return segs


def spinner_text(frame: int, rt, turn_started_at, now: float) -> str:
    """spinner 行文案（纯函数）：空闲返回空串（行隐藏）。

    参数：
        frame: 动画帧下标（spinner 线程递增，这里只取模）
        rt: 读 turn_active / event_pending
        turn_started_at: 回合开始时刻（time.monotonic 值，None=没在跑）
        now: 当前时刻（传进来而不是函数内取，方便测试）

    返回：如 "⠋ terminal 3s" / "⠋ 思考中… 3s" / ""（空闲）
    """
    try:
        if not getattr(rt, "turn_active", False):
            return ""
        elapsed = (now - turn_started_at) if turn_started_at else 0.0
        t = _fmt_elapsed(elapsed)
        pend = getattr(rt, "event_pending", None)
        mark = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        if pend:
            return f"{mark} {pend[-1]} {t}".strip()
        return f"{mark} 思考中… {t}".strip()
    except Exception:
        return ""   # 纯视觉，任何异常都当「不显示」


def submit_input(buffer, input_queue) -> None:
    """提交小函数（键位回调只是薄壳，逻辑全在这——方便单测）。

    大白话：把输入框里的字塞给老主循环的队列，然后擦黑板。
    空白输入不入队（老语义：空行直接跳过，不烧一轮 LLM）。

    参数：
        buffer: prompt_toolkit 的 Buffer（TextArea 的肚子）
        input_queue: 老主循环消费的 _input_q
    """
    try:
        text = buffer.text
        if text.strip():
            input_queue.put(text)
        # validate_and_handle 会把非空文本记进历史再把框清空
        #（等价于老 PromptSession 按回车的动作）
        buffer.validate_and_handle()
    except Exception:
        logger.exception("提交输入失败（这行字丢了，但不许炸输入线程）")


# ---------------------------------------------------------------------------
# 组装层：布局 + 键位 + 样式
# ---------------------------------------------------------------------------

def _term_width() -> int:
    """当前终端列数（拿不到按 80 兜底——纯视觉，不许炸）。"""
    try:
        from prompt_toolkit.application import get_app
        return get_app().output.get_size().columns or 80
    except Exception:
        return 80


def _estimate_input_height(text: str, width: int) -> int:
    """估算输入框该占几行（hermes 同款思路的简化版）。

    每行按「字符数 + 2（❯ 提示符）」折算占几屏行，硬换行如实算，
    长句软折行按宽度折算（中英混排按 1 字符 1 格近似，够用）。
    上限 8 行（超过就在框里滚动，不把状态栏顶出屏幕）。
    """
    if not text:
        return 1
    rows = 0
    for line in text.split("\n"):
        rows += max(1, math.ceil((len(line) + 2) / max(10, width)))
    return min(8, max(1, rows))


class _GrayHint:
    """空输入时在光标后面渲染一行灰字提示（hermes 的 placeholder 同款）。

    实现成 InputProcessor：不改 Buffer 里的真文本（提交的还是空串），
    只在「渲染那一刻」往首行尾部贴灰字——像玻璃上贴的便利贴，揭掉不留痕。
    """

    def __init__(self, text: str):
        self._text = text

    def apply_transformation(self, transform_input):
        from prompt_toolkit.layout.processors import Transformation
        try:
            buffer = transform_input.buffer_control.buffer
            if transform_input.lineno == 0 and not buffer.text:
                return Transformation(
                    transform_input.fragments
                    + [("class:placeholder", self._text)]
                )
        except Exception:
            pass
        return Transformation(transform_input.fragments)


def _build_key_bindings(input_queue, eof_sentinel, interrupt_fn):
    """键位路由表（大白话：操作台怎么响应特殊键）。

    - Enter：补全菜单开着就先收菜单（pt 惯例：第一下选中、第二下提交）；
      没开菜单就提交——submit_input 入队 + 清框
    - Alt+↵ / (Windows) Ctrl+↵：输入框里插换行（多行编辑）
    - Ctrl+C：三档同老语义——有字清行 / 回合中调 interrupt_fn /
      空闲空框塞 EOF 哨兵（统一退出通道）
    - Ctrl+D：空框塞 EOF 哨兵；有字删光标前一个字符（readline 惯例）
    """
    import sys
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        b = event.app.current_buffer
        if b.complete_state:
            b.complete_state = None   # 先收补全菜单，下一记 Enter 才提交
        else:
            submit_input(b, input_queue)

    @kb.add("escape", "enter")        # Alt+Enter（pt 把它解析成 esc+enter）
    def _newline(event):
        event.app.current_buffer.insert_text("\n")

    if sys.platform == "win32":
        # Windows Terminal 把 Ctrl+Enter 发成裸 LF（c-j）——绑成换行，
        # 用户按 Ctrl+↵ 就是想要换行而不是提交（POSIX 的坑块③再统一处理）
        @kb.add("c-j")
        def _newline_win(event):
            event.app.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def _ctrl_c(event):
        b = event.app.current_buffer
        if b.text:
            b.reset()
        elif interrupt_fn is not None and interrupt_fn():
            pass   # 回合进行中：中断回合；操作台保持，继续听输入
        elif eof_sentinel is not None and input_queue is not None:
            input_queue.put(eof_sentinel)   # 空闲空框：走统一退出通道

    @kb.add("c-d")
    def _ctrl_d(event):
        b = event.app.current_buffer
        if not b.text:
            if eof_sentinel is not None and input_queue is not None:
                input_queue.put(eof_sentinel)
        else:
            b.delete_before_cursor()

    return kb


def build_application(rt, *, completer=None, interrupt_fn=None,
                      input_queue=None, eof_sentinel=None,
                      history_path=None):
    """组装常驻操作台；环境不支持时返回 None（调用方降级 console.input）。

    参数：
        rt: RuntimeContext（状态栏/spinner 读它的现有黑板）
        completer: SlashCompleter（cli_input.build_completer 产物）
        interrupt_fn: Ctrl+C 回合中中断判断器（cli_input.build_interrupt_fn 产物）
        input_queue: 老主循环的 _input_q（Enter 提交塞这里）
        eof_sentinel: 退出哨兵对象（Ctrl+C/Ctrl+D 空闲空框时塞队列）
        history_path: 输入历史文件 Path（None 用内存历史，测试方便）

    返回：prompt_toolkit Application；None=环境不可用（降级）。
    """
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.layout import (
            ConditionalContainer, HSplit, Layout, Window,
        )
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import TextArea

        # ---- 终端输出：先试真的，拿不到（管道/mintty 等）就降级 ----
        # 大白话：先看这个终端能不能画界面；画不了就返回 None 让调用方
        # 走 console.input 老路。只有 verify 显式声明「无头构建」时才
        # 用假输出把布局拼出来检查（真终端拿不到输出还给假输出的话，
        # 界面会静默变成隐形的——比降级糟糕得多）。
        try:
            from prompt_toolkit.output import create_output
            output = create_output()
        except Exception as _oe:
            if os.environ.get("CODEAGENT_LAYOUT_HEADLESS") != "1":
                logger.error(
                    "终端输出不可用，界面降级为 console.input: %s", _oe)
                return None
            from prompt_toolkit.output import DummyOutput
            output = DummyOutput()

        # ---- 状态黑板（spinner 线程写，渲染闭包读）----
        state = {"turn_started": None, "was_active": False, "frame": 0}

        def _status_bar_text():
            try:
                elapsed = None
                if getattr(rt, "turn_active", False) and state["turn_started"]:
                    elapsed = time.monotonic() - state["turn_started"]
                return " │ ".join(
                    status_bar_segments(rt, _term_width(), elapsed_s=elapsed)
                ) or "CodeAgent"
            except Exception:
                return "CodeAgent"

        def _spinner_line():
            return spinner_text(
                state["frame"], rt, state["turn_started"], time.monotonic()
            )

        # ---- 输入区：多行 TextArea（历史/补全/灰字提示全挂上）----
        history = (FileHistory(str(history_path))
                   if history_path else InMemoryHistory())
        input_area = TextArea(
            height=lambda: Dimension(
                min=1, max=8,
                preferred=_estimate_input_height(input_area.text, _term_width()),
            ),
            prompt=[("class:prompt", "❯ ")],
            multiline=True,
            wrap_lines=True,
            history=history,
            completer=completer,
            complete_while_typing=True,
        )
        # 灰字提示贴在渲染层（不动真文本）
        try:
            input_area.control.input_processors.append(
                _GrayHint("发送消息，/help 查命令；Alt+↵ 换行")
            )
        except Exception:
            pass   # 提示贴不上就裸奔，不挡输入

        # ---- 各层容器（自上而下）----
        spinner_row = ConditionalContainer(
            Window(FormattedTextControl(_spinner_line), height=1),
            filter=Condition(lambda: bool(_spinner_line())),
        )
        separator = Window(
            height=1, char="─", style="class:separator",
        )
        status_bar = Window(
            FormattedTextControl(_status_bar_text),
            height=1, wrap_lines=False, style="class:status-bar",
        )

        app = Application(
            layout=Layout(HSplit([spinner_row, input_area, separator, status_bar])),
            key_bindings=_build_key_bindings(input_queue, eof_sentinel, interrupt_fn),
            output=output,
            style=Style.from_dict({
                "prompt": "bold fg:#00aa88",
                "status-bar": "reverse",
                "separator": "fg:#555555",
                "placeholder": "fg:#777777",
            }),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,   # 退出时擦掉操作台，不冻进滚动历史
        )
        # 把状态黑板挂到 app 上：spinner 线程要读写它（翻帧/记回合起点）
        app._codeagent_ui_state = state
        return app
    except Exception as e:
        logger.error("Application 创建失败，界面降级为 console.input: %s", e)
        return None


# ---------------------------------------------------------------------------
# 运行层：spinner 线程 + 节流重绘 + 跨线程退出
# ---------------------------------------------------------------------------

def invalidate_throttled(app, min_interval: float = 0.1) -> None:
    """节流版 invalidate：转圈动画 0.1s 一帧，狂刷只会闪瞎眼。"""
    last = getattr(app, "_codeagent_last_invalidate", 0.0)
    now = time.monotonic()
    if now - last < min_interval:
        return
    app._codeagent_last_invalidate = now
    try:
        app.invalidate()
    except Exception:
        pass   # 纯视觉，刷失败就等下一拍


def start_spinner_thread(rt, app, stop_event):
    """spinner 线程：0.1s 一拍——翻帧 + 盯 turn_active 翻沿记回合起点 + 重绘。

    大白话：像值班室的老大爷，每 0.1 秒睁一次眼——
    看见回合刚开始（turn_active 从 False 翻 True）就按下秒表，
    看见回合结束就归零；顺手把转圈动画翻一帧。
    """
    from threading import Thread

    state = getattr(app, "_codeagent_ui_state", None)

    def _loop():
        while not stop_event.is_set():
            time.sleep(0.1)
            try:
                if state is not None:
                    active = bool(getattr(rt, "turn_active", False))
                    if active and not state["was_active"]:
                        state["turn_started"] = time.monotonic()  # 按秒表
                    elif not active:
                        state["turn_started"] = None              # 归零
                    state["was_active"] = active
                    state["frame"] = (state["frame"] + 1) % len(SPINNER_FRAMES)
                invalidate_throttled(app)
            except Exception:
                pass   # 动画线程挂了不许连累任何人

    t = Thread(target=_loop, daemon=True, name="cli-spinner")
    t.start()
    return t


def request_app_exit(app) -> None:
    """工作线程请 UI 退出（跨线程必须走 call_soon_threadsafe 转发）。

    app 没在跑/已退出/压根是 None（降级模式）都静默返回。
    """
    if app is None:
        return
    try:
        app.loop.call_soon_threadsafe(app.exit)
    except Exception:
        pass
