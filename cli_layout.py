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

# 键盘协议增强（Shift/Ctrl+Enter 别名、焦点噪声序列忽略）——import 即装，
# 失败静默（pt 缺失环境到不了这里，装不上也不挡界面）
try:
    import cli_pt_extras
    cli_pt_extras.install_all()
except Exception:
    pass

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


def _spinner_frames() -> list:
    """动画帧：皮肤声明了就用皮肤的，否则盲文帧（皮肤可换帧不换代码）。"""
    try:
        import cli_skin
        frames = cli_skin.get_active_skin().spinner_frames
        if frames:
            return list(frames)
    except Exception:
        pass
    return SPINNER_FRAMES


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
        frames = _spinner_frames()
        mark = frames[frame % len(frames)]
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
# 补全器 + 中断判断器（原 cli_input.py 迁入——老降级界面已删，
# 这两个零件是新旧界面共用的中立件，归到界面总仓 cli_layout）
# ---------------------------------------------------------------------------

def build_interrupt_fn(is_active_fn, do_interrupt_fn):
    """造 Ctrl+C 的「回合中中断」判断器。

    参数：
        is_active_fn：() -> bool，当前是否有 AI 回合在跑
        do_interrupt_fn：() -> None，真正执行中断（agent.interrupt + 取消子代理）

    返回：() -> bool——True 表示「回合在跑，已触发中断」（键位保持提示符）；
        False 表示「空闲」（键位走退出通道）。任何异常都按 False 处理
        （fail-open：中断通道出问题不能挡住退出语义）。
    """
    def _interrupt():
        try:
            if not is_active_fn():
                return False
            do_interrupt_fn()
            return True
        except Exception:
            return False
    return _interrupt


# 补全器基类：prompt_toolkit 可用就继承它的 Completer（真终端的异步补全
# 通道调 get_completions_async——那是基类方法，裸鸭子类没有，会在打字
# 的时候崩掉"Unhandled exception in event loop"）。
try:
    from prompt_toolkit.completion import Completer as _PtCompleter
except Exception:  # pragma: no cover - 理论上到不了这（pt 是硬依赖）
    _PtCompleter = object


class SlashCompleter(_PtCompleter):
    """三级补全：命令名（注册表+技能+技能束同一池）→ 命令参数。

    实现成 prompt_toolkit 的 Completer 协议（继承基类拿默认的异步包装，
    铁律：任何异常都吞掉返回空——补全挂了不能挡住打字。
    """

    def __init__(self, registry_tokens, arg_completers, dynamic_tokens_fn):
        self._registry_tokens = list(registry_tokens)   # 含别名
        self._arg_completers = dict(arg_completers)
        self._dynamic_tokens_fn = dynamic_tokens_fn     # () -> 技能/技能束命令名

    def get_completions(self, document, complete_event):
        try:
            text = document.text
            if not text.startswith("/"):
                return
            parts = text.split()
            if len(parts) <= 1 and not text.endswith(" "):
                # 一级：命令名补全（静态注册表 + 动态技能池合并）
                tokens = set(self._registry_tokens)
                try:
                    tokens.update(self._dynamic_tokens_fn() or [])
                except Exception:
                    pass
                frag = parts[0] if parts else ""
                for t in sorted(tokens):
                    if t.startswith(frag):
                        from prompt_toolkit.completion import Completion
                        yield Completion(t, start_position=-len(frag))
            else:
                # 二级：该命令的参数补全（替换正在输入的最后一个词，
                # 而不是光标处硬塞——/sessions re 选 resume 要变成
                # /sessions resume，不是 /sessions reresume）
                fn = self._arg_completers.get(parts[0])
                if fn:
                    from prompt_toolkit.completion import Completion
                    frag = text[len(parts[0]):].lstrip().split()[-1] if \
                        text[len(parts[0]):].lstrip().split() else ""
                    for cand in fn(text) or []:
                        if frag and not str(cand).startswith(frag):
                            continue  # 前缀不匹配的候选不出（对齐一级行为）
                        yield Completion(
                            str(cand),
                            start_position=-len(frag) if frag else 0,
                        )
        except Exception:
            return


def build_completer(rt):
    """从注册表 + rt 的技能/技能束命令组装补全器。

    动态部分用闭包按需现取（技能中途增删也能补全到最新）。
    """
    import cli_commands as cc

    def dynamic_tokens():
        try:
            return list(getattr(rt, "skill_commands", {}) or {}) + \
                   list(getattr(rt, "bundle_commands", {}) or {})
        except Exception:
            return []

    return SlashCompleter(
        registry_tokens=cc.all_tokens(),
        arg_completers=cc.arg_completer_map(),
        dynamic_tokens_fn=dynamic_tokens,
    )


# ---------------------------------------------------------------------------
# 终端自举：确保进程有真控制台（git-bash 用 winpty 中转）
# ---------------------------------------------------------------------------

def _stdio_is_console() -> bool:
    """stdout 和 stdin 是不是都是控制台句柄（pt 画界面/读按键的硬条件）。

    大白话：光「进程挂着控制台」不算数——ConPTY/重定向环境下进程可能
    有控制台对象但 stdio 是管道，pt 照样画不了。必须 stdout/stdin
    两个都是真控制台才放行。
    """
    try:
        import ctypes
        from ctypes import byref
        k32 = ctypes.windll.kernel32
        for std_handle in (-11, -10):   # STD_OUTPUT_HANDLE / STD_INPUT_HANDLE
            h = k32.GetStdHandle(std_handle)
            mode = ctypes.c_uint32()
            if not h or not k32.GetConsoleMode(h, byref(mode)):
                return False
        return True
    except Exception:
        return True   # 探测失败按「可用」处理（别误杀正常环境）


def ensure_interactive_terminal() -> None:
    """Windows 终端自举：stdio 是控制台直接过；mintty 用 winpty 自救；
    其余（管道/CI）明说退出。

    大白话：新界面（Application 操作台）在 Windows 上靠控制台 API 画图
    和读按键。四种终端里 cmd / Windows Terminal / PowerShell 的 stdio
    天然是控制台；git-bash 的 mintty 给的是管道——进程连控制台对象都没
    有，得靠 winpty（Git for Windows 自带）造一个隐藏控制台当中转站，
    把进程重新拉起来。有控制台对象但 stdio 被重定向（CI/管道喂脚本），
    救不了也不该救——明说并退出，老降级界面已经删了，不装哑巴。

    防循环护栏：winpty 重启前设 CODEAGENT_WINPTY_REEXEC 环境变量，
    重启后的子进程带上它——万一 winpty 没给出控制台，也不再重试，
    直接走报错退出（不然就无限重启套娃了）。
    """
    import shutil
    import subprocess
    import sys

    if os.name != "nt":
        return                      # POSIX 家族本就有 pty，不折腾
    if _stdio_is_console():
        return

    # stdio 不是控制台。区分两种情况：
    # a) 连控制台对象都没有（mintty）→ winpty 能救
    # b) 有控制台对象但 stdio 被重定向 → 救不了，直接报错
    try:
        import ctypes
        has_console_window = ctypes.windll.kernel32.GetConsoleWindow() != 0
    except Exception:
        has_console_window = True

    if not has_console_window and not os.environ.get("CODEAGENT_WINPTY_REEXEC"):
        winpty = shutil.which("winpty")
        if winpty:
            # Popen 用列表传参（自动处理带空格的路径，如
            # C:\\Program Files\\Git\\usr\\bin\\winpty.exe）；
            # os.execv 在 Windows 上不带队引号，路径带空格就炸
            child_env = dict(os.environ)
            child_env["CODEAGENT_WINPTY_REEXEC"] = "1"
            try:
                proc = subprocess.Popen(
                    [winpty, sys.executable,
                     os.path.abspath(sys.argv[0])] + list(sys.argv[1:]),
                    env=child_env,
                )
                sys.exit(proc.wait())
            except Exception as e:
                print(f"winpty 重新拉起失败: {e}", file=sys.stderr)
        else:
            print(
                "当前在 git-bash（mintty）里但没有找到 winpty——"
                "Git for Windows 自带它，请检查 PATH。",
                file=sys.stderr,
            )
    print(
        "CodeAgent 需要交互式终端：请用 cmd / Windows Terminal / "
        "PowerShell，或在 git-bash 里确保 winpty 可用后重试"
        "（stdio 不能被重定向）。",
        file=sys.stderr,
    )
    sys.exit(1)


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
    """组装常驻操作台；环境不支持（如 stdout 被重定向）返回 None。

    老降级界面已删：返回 None 对调用方是致命错误（打印原因后退出）。
    verify 的无头构建（CODEAGENT_LAYOUT_HEADLESS=1）例外——它只检查
    布局拼装，不真跑。

    参数：
        rt: RuntimeContext（状态栏/spinner 读它的现有黑板）
        completer: SlashCompleter（cli_layout.build_completer 产物）
        interrupt_fn: Ctrl+C 回合中中断判断器（cli_layout.build_interrupt_fn 产物）
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

        # ---- 终端输出：先试真的，拿不到（stdout 被重定向等）就失败 ----
        # 大白话：先看这个终端能不能画界面；画不了就返回 None，由调用方
        # 打印原因后退出（老降级界面已删）。只有 verify 显式声明「无头
        # 构建」时才用假输出把布局拼出来检查（真终端拿不到输出还给假
        # 输出的话，界面会静默变成隐形的——比报错退出糟糕得多）。
        try:
            from prompt_toolkit.output import create_output
            output = create_output()
        except Exception as _oe:
            if os.environ.get("CODEAGENT_LAYOUT_HEADLESS") != "1":
                logger.error("终端输出不可用，界面起不来: %s", _oe)
                return None
            from prompt_toolkit.output import DummyOutput
            output = DummyOutput()

        # ---- Windows legacy 控制台开 VT 解释（rich 彩色的救命开关）----
        # 大白话：rich 的彩色输出经 patch_stdout 是「原始 ANSI 字节直写」，
        # legacy 控制台不认 VT 就满屏 ?[1;2m 乱码（我们自己的框线颜色走
        # pt 解析路径，不受此影响）。幂等、失败忽略——现代终端本来就开着。
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import byref
                k32 = ctypes.windll.kernel32
                h = k32.GetStdHandle(-11)   # STD_OUTPUT_HANDLE
                mode = ctypes.c_uint32()
                if h and k32.GetConsoleMode(h, byref(mode)):
                    k32.SetConsoleMode(
                        h, mode.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            except Exception:
                pass

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
        # 提示符（❯ / > …）和配色跟当前皮肤走
        import cli_skin
        prompt_symbol = cli_skin.get_active_prompt_symbol("❯")
        input_area = TextArea(
            height=lambda: Dimension(
                min=1, max=8,
                preferred=_estimate_input_height(input_area.text, _term_width()),
            ),
            prompt=[("class:prompt", f"{prompt_symbol} ")],
            multiline=True,
            wrap_lines=True,
            history=history,
            completer=completer,
            complete_while_typing=True,
        )
        # 灰字提示贴在渲染层（不动真文本）
        try:
            input_area.control.input_processors.append(
                _GrayHint("发送消息，/help 查命令；Alt/Shift/Ctrl+↵ 换行")
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

        _style_base = {
            "prompt": "bold fg:#00aa88",
            "status-bar": "reverse",
            "separator": "fg:#555555",
            "placeholder": "fg:#777777",
        }
        _style_base.update(cli_skin.get_pt_style_overrides())
        app = Application(
            layout=Layout(HSplit([spinner_row, input_area, separator, status_bar])),
            key_bindings=_build_key_bindings(input_queue, eof_sentinel, interrupt_fn),
            output=output,
            style=Style.from_dict(_style_base),
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
                    state["frame"] = (state["frame"] + 1) % len(_spinner_frames())
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


def install_input_bridge(app) -> None:
    """注册跨线程输入桥：工作线程的提问 → pt 的 run_in_terminal 通道。

    大白话：stdin 被 pt 独占后，工作线程（审批/确认）里的 input() 读
    不到字。桥把提问函数调度到 UI 线程执行——pt 会先收起界面、把终端
    还给经典输入，用户答完再恢复界面。app 为 None 时注销桥（直读）。
    """
    import asyncio
    import cli_ui

    if app is None:
        cli_ui.set_input_bridge(None)
        return

    def _bridge(func):
        future = asyncio.run_coroutine_threadsafe(
            app.run_in_terminal_async(func), app.loop)
        return future.result()

    cli_ui.set_input_bridge(_bridge)
