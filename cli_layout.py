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

# 页脚两档宽度阈值（claude code 风格：窄屏只留模型段）
_TIER_NARROW = 52    # < 52 列：只有 ⏵⏵ 模型
_TIER_MEDIUM = 76    # >= 76 列：给按键说明


def status_bar_segments(rt, width: int, elapsed_s=None) -> list:
    """页脚内容段（纯函数）：claude code 风格的底部条。

    长相：``⏵⏵ {模型} · Ctrl+C 中断 · ctrl+t 任务面板``——
    按宽度两档过滤：窄屏只留模型，中屏以上给全量按键说明。

    参数：
        rt: RuntimeContext（读 agent.model/bg_count——现有黑板）
        width: 终端列数
        elapsed_s: 兼容旧签名的保留参数（不再展示计时——计时在 spinner 行）

    返回：str 列表，调用方用 " · " 拼接。
    """
    segs = []
    try:
        model = getattr(getattr(rt, "agent", None), "model", "") or ""
        if model:
            segs.append(f"⏵⏵ {model}")
        else:
            segs.append("⏵⏵ CodeAgent")
    except Exception:
        segs.append("⏵⏵ CodeAgent")
    if width >= _TIER_NARROW:
        bg = getattr(rt, "bg_count", None)
        if bg:
            segs.append(f"☂{bg}个后台")
    if width >= _TIER_MEDIUM:
        segs.append("Ctrl+C 中断 · ctrl+t 任务面板")
    return segs


def spinner_text(frame: int, rt, turn_started_at, now: float) -> str:
    """spinner 行文案（纯函数）：空闲返回空串（行隐藏）。

    长相委托给 cli_live（claude code 同款）：``✶ Cooking… (3m 11s · ↓ 17.6k tokens)``。
    函数保留在这层是为了兼容既有调用方/测试签名。
    """
    try:
        import cli_live
        return cli_live.spinner_text(frame, rt, turn_started_at, now)
    except Exception:
        return ""   # 纯视觉，任何异常都当「不显示」


def _tighten_next_render(app=None) -> None:
    """掀掉 renderer 的 last_height 地板，下一帧按内容高度渲染。

    大白话：prompt_toolkit 为了防闪烁，渲染高度只涨不跌（取
    max(min高度, 上一帧高度, 内容高度)）——回合中 live 面板撑高过、
    或输入框敲过多行，结束/清空后高度也不回落，多余行全塞给弹性
    窗口，输入框就赖在 8 行高的大空箱子上下不来。把 _last_screen
    掏空，下一帧就缩回实际高度。私有属性 + try/except 兜底——
    pt 版本变了顶多不缩高，绝不炸。
    """
    try:
        if app is None:
            from prompt_toolkit.application import get_app
            app = get_app()
        if app is not None:
            app.renderer._last_screen = None
    except Exception:
        pass


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
        # 框清空了 → 渲染高度也跟着回落（多行输入别赖着 8 行高）
        _tighten_next_render()
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

    def __init__(self, registry_tokens, arg_completers, dynamic_tokens_fn,
                 meta_fn=None):
        self._registry_tokens = list(registry_tokens)   # 含别名
        self._arg_completers = dict(arg_completers)
        self._dynamic_tokens_fn = dynamic_tokens_fn     # () -> 技能/技能束命令名
        self._meta_fn = meta_fn                          # token -> 一句描述（菜单右列）

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
                        # display_meta 带一句描述——补全菜单右列显示它
                        #（claude code 同款：命令 + 说明两列）
                        meta = ""
                        if self._meta_fn is not None:
                            try:
                                meta = self._meta_fn(t) or ""
                            except Exception:
                                meta = ""
                        yield Completion(t, start_position=-len(frag),
                                         display_meta=meta)
            else:
                # 二级：该命令的参数补全（替换正在输入的最后一个词，
                # 而不是光标处硬塞——/sessions re 选 resume 要变成
                # /sessions resume，不是 /sessions reresume）
                fn = self._arg_completers.get(parts[0])
                if fn:
                    from prompt_toolkit.completion import Completion
                    # 正在补的词 = 命令后最后一个词；行尾是空格时补的是
                    # 「下一个新词」（frag 为空，不过滤——全量候选让
                    # 用户翻着看）
                    rest = text[len(parts[0]):].lstrip()
                    rest_words = rest.split()
                    if not rest_words or text.endswith(" "):
                        frag = ""
                    else:
                        frag = rest_words[-1]
                    # 兼容两种签名：老的单参 fn(frag)——拿最后那个词；
                    # 新的双参 fn(frag, 全文)——能看上下文（比如
                    # /plugin install 补市场名、/plugin disable 补已装名）。
                    # 老实说：以前这里传的是全文，单参补全器拿 "/plugin x"
                    # 当前缀过滤永远滤空——等于从没生效过，这次捎带修活。
                    try:
                        cands = fn(frag, text)
                    except TypeError:
                        cands = fn(frag)
                    for cand in cands or []:
                        meta = ""
                        if isinstance(cand, (tuple, list)) and len(cand) == 2:
                            cand, meta = cand[0], cand[1]
                        if frag and not str(cand).startswith(frag):
                            continue  # 前缀不匹配的候选不出（对齐一级行为）
                        yield Completion(
                            str(cand),
                            start_position=-len(frag) if frag else 0,
                            display_meta=str(meta),
                        )
        except Exception:
            return


def build_completer(rt):
    """从注册表 + rt 的技能/技能束命令组装补全器。

    动态部分用闭包按需现取（技能中途增删也能补全到最新）。
    meta_fn 给每个 token 配一句描述——补全菜单右列显示（/help 同款信息）。
    """
    import cli_commands as cc

    def dynamic_tokens():
        try:
            return list(getattr(rt, "skill_commands", {}) or {}) + \
                   list(getattr(rt, "bundle_commands", {}) or {})
        except Exception:
            return []

    def token_meta(token):
        """token → 一句话描述：注册表命令给 summary，技能/技能束给类型标注。"""
        try:
            cmd = cc.lookup(token)
            if cmd is not None:
                return getattr(cmd, "summary", "") or ""
            if token in (getattr(rt, "skill_commands", None) or {}):
                return "技能"
            if token in (getattr(rt, "bundle_commands", None) or {}):
                return "技能束"
        except Exception:
            pass
        return ""

    return SlashCompleter(
        registry_tokens=cc.all_tokens(),
        arg_completers=cc.arg_completer_map(),
        dynamic_tokens_fn=dynamic_tokens,
        meta_fn=token_meta,
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


def _clip_plain(s: str, n: int) -> str:
    """纯文本截断：超长加省略号（补全候选一行一条，不许折行）。"""
    s = str(s or "")
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


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


def is_double_press(state: dict, now: float, window: float = 2.0) -> bool:
    """判断这次按键是不是「窗口期内的第二击」（纯函数，可单测）。

    大白话：第一次按下时只记时间不打架；2 秒内又按下一次才算双击
    （返回 True 并清零计时）；超时的第二击当作新一轮的第一击。

    参数：
        state: {"t": 上次按下时刻} 字典（调用方持有，函数原地更新）
        now: 当前时刻（time.monotonic 值）
        window: 双击窗口秒数，默认 2.0
    返回：True=这是窗口内的第二击。
    """
    last = state.get("t", 0.0)
    hit = (now - last) < window
    state["t"] = 0.0 if hit else now
    return hit


def _build_key_bindings(input_queue, eof_sentinel, interrupt_fn, force_exit_fn=None):
    """键位路由表（大白话：操作台怎么响应特殊键）。

    - Enter：补全菜单开着就先收菜单（pt 惯例：第一下选中、第二下提交）；
      没开菜单就提交——submit_input 入队 + 清框
    - Alt+↵ / (Windows) Ctrl+↵：输入框里插换行（多行编辑）
    - Ctrl+C：有字清行 / 回合中第一击中断本轮（提示再按强退）、
      2 秒内第二击强制退出一切 / 空闲空框塞 EOF 哨兵（统一退出通道）
    - Ctrl+D：空框塞 EOF 哨兵；有字删光标前一个字符（readline 惯例）
    """
    import sys
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()
    _cc_state = {"t": 0.0}   # Ctrl+C 双击检测器的心跳本

    @kb.add("enter")
    def _submit(event):
        b = event.app.current_buffer
        if b.complete_state:
            # 菜单里有高亮项（↑/↓/Tab 选过）→ Enter 先把它填进输入框
            #（claude code 同款：回车=采纳建议），下一记 Enter 再提交；
            # 没高亮就收菜单照常提交
            idx = b.complete_state.complete_index
            if idx is not None:
                try:
                    b.apply_completion(
                        b.complete_state.completions[idx])
                except Exception:
                    pass
                b.complete_state = None
                return
            b.complete_state = None   # 先收补全菜单，下一记 Enter 才提交
        else:
            submit_input(b, input_queue)

    @kb.add("escape", "enter")        # Alt+Enter（pt 把它解析成 esc+enter）
    def _newline(event):
        event.app.current_buffer.insert_text("\n")

    # ---- 补全候选环选（内联候选区没有 pt 菜单的默认 ↑↓/Tab 键位，自补）----
    # 候选区开着：↑/↓ 环选、Tab 选下一个；没开：↑/↓ 照旧移光标、
    # Tab 开出候选列表（不高亮，光看；再按 Tab 才开始选）

    @kb.add("up")
    def _up(event):
        b = event.app.current_buffer
        if b.complete_state and b.complete_state.completions:
            b.complete_previous()
        else:
            b.cursor_up()

    @kb.add("down")
    def _down(event):
        b = event.app.current_buffer
        if b.complete_state and b.complete_state.completions:
            b.complete_next()
        else:
            b.cursor_down()

    @kb.add("tab")
    def _tab(event):
        b = event.app.current_buffer
        if b.complete_state and b.complete_state.completions:
            b.complete_next()
        elif b.completer is not None:
            b.start_completion(select_first=False)

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
            # 清行后渲染高度回落（多行敲剩的字被清掉，框别赖高）
            _tighten_next_render(event.app)
            return
        if interrupt_fn is not None and interrupt_fn():
            # 回合进行中：第一击中断本轮；2 秒内第二击强制退出一切
            #（子代理/后台任务/所有线程——交给 cli.py 的 force_exit_fn）
            if is_double_press(_cc_state, time.monotonic()) \
                    and force_exit_fn is not None:
                force_exit_fn()
            else:
                from cli_ui import console
                console.print(
                    "[yellow]⚡ 已请求中断本轮；"
                    "再按一次 Ctrl+C 强制退出所有任务[/yellow]"
                )
            return
        if eof_sentinel is not None and input_queue is not None:
            input_queue.put(eof_sentinel)   # 空闲空框：走统一退出通道

    @kb.add("c-d")
    def _ctrl_d(event):
        b = event.app.current_buffer
        if not b.text:
            if eof_sentinel is not None and input_queue is not None:
                input_queue.put(eof_sentinel)
        else:
            b.delete_before_cursor()

    @kb.add("c-t")
    def _ctrl_t(event):
        # 任务面板开关（claude code 同款 ctrl+t：嫌子代理树/任务清单
        # 碍眼就藏起来，再按一次放出来——纯视觉，失败就当没按过）
        try:
            import cli_live
            hidden = cli_live.toggle_panel()
            from cli_ui import console
            from rich.text import Text
            console.print(Text(
                "[任务面板已隐藏]" if hidden else "[任务面板已显示]",
                style="dim"))
        except Exception:
            pass

    return kb


def build_application(rt, *, completer=None, interrupt_fn=None,
                      input_queue=None, eof_sentinel=None,
                      force_exit_fn=None, history_path=None):
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
                return " · ".join(
                    status_bar_segments(rt, _term_width())
                ) or "CodeAgent"
            except Exception:
                return "CodeAgent"

        def _spinner_line():
            return spinner_text(
                state["frame"], rt, state["turn_started"], time.monotonic()
            )

        def _live_lines():
            """live 区的行列表 [(style, text), ...]（一行一个元素）。"""
            try:
                head = _spinner_line()
                if not head:
                    return []
                import cli_live
                return [("", head)] + cli_live.panel_lines(_term_width())
            except Exception:
                return []

        def _live_text():
            """live 区内容（spinner 行 + 子代理树/任务清单面板）。

            返回 pt 的 fragment 列表；**行与行之间必须显式塞 ("", "\\n")**
            ——pt 的 fragment 是直接拼接的，不塞换行符整块会挤成一行、
            被终端软换行搅成一锅粥。空列表 = 整块隐藏（idle 时 live 区
            整个收起——任务清单只在回合进行中挂着，回合收尾由
            cli_events 落一段静态快照进滚动历史）。
            """
            frags = []
            for i, (style, text) in enumerate(_live_lines()):
                if i > 0:
                    frags.append(("", "\n"))
                frags.append((style, text))
            return frags

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
            # 永不拉伸超过内容行数：非全屏 app 的渲染高度有「只涨不跌」
            # 的地板（见 _tighten_next_render），弹性窗口会把多余行全吃
            # 掉——空输入框也会被顶到 max=8 变大空箱子。锁死伸展后，
            # 高度永远 = 内容行数（初始 1 行，Shift/Alt/Ctrl+↵ 换行随行数长高）
            dont_extend_height=True,
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

        # ---- 补全候选区（内联区块，不悬浮）----
        # 为什么不用 pt 的 CompletionsMenu 浮层：浮层叠在内容上、
        # 消失不留痕，观感就是「悬浮窗一闪而过」。改成布局里的普通
        # 区块——立在输入框正上方，像任务面板一样落地，出现/收起
        # 都推着版面走，不遮任何字。
        def _completion_lines():
            """当前补全状态里的候选列表 [(style, text), ...]（纯函数）。

            可滚动：窗口固定露 8 行，↑↓/Tab 环选到哪行窗口就跟到哪行
            （窗口位置由选中下标直接推导，不存状态——选中项永远可见，
            环选到尾部会自动往下滚）。窗口外还有候选时上下给指示行，
            全部候选都能翻到，不是只看前几条。
            """
            try:
                cs = input_area.buffer.complete_state
                if not cs or not cs.completions:
                    return []
                comps = cs.completions
                total = len(comps)
                visible = 8
                idx = cs.complete_index   # Tab/↑↓ 环选到的下标（None=未选）
                # 窗口起点：选中项落在窗口最后一行（往下翻）或顶部
                #（往上翻回首页），未选中时从 0 开始
                if idx is None:
                    scroll = 0
                else:
                    scroll = max(0, idx - visible + 1) if idx >= visible else 0
                lo, hi = scroll, min(scroll + visible, total)

                lines = []
                if lo > 0:
                    lines.append((
                        "class:hint-dim",
                        f"  ▲ … 上方还有 {lo} 条（↑ 继续翻）",
                    ))
                for i in range(lo, hi):
                    c = comps[i]
                    cur = (i == idx)
                    mark = "▶ " if cur else "  "
                    label = getattr(c, "display_text", None) or c.text
                    meta = getattr(c, "display_meta_text", "") or ""
                    one = f"{mark}{label}" + (f"  {meta}" if meta else "")
                    lines.append((
                        "class:hint-current" if cur else "class:hint-dim",
                        _clip_plain(one, _term_width()),
                    ))
                if hi < total:
                    lines.append((
                        "class:hint-dim",
                        f"  ▼ … 下方还有 {total - hi} 条（↓ 继续翻）",
                    ))
                return lines
            except Exception:
                return []

        completions_area = ConditionalContainer(
            Window(
                FormattedTextControl(
                    lambda: [frag for line in _completion_lines()
                             for frag in (line, ("", "\n"))][:-1] or [],
                    show_cursor=False,
                ),
                height=lambda: Dimension(
                    min=0, max=10,   # 8 条候选 + 上下最多两条滚动指示行
                    preferred=len(_completion_lines()),
                ),
                dont_extend_height=True,
                wrap_lines=False,
            ),
            filter=Condition(lambda: bool(_completion_lines())),
        )
        # 挂到 app 上给 verify 检查用（内联候选区必须在场——浮层版已废）
        app_probe_completions_area = completions_area

        # ---- 各层容器（自上而下：live 区 / ─── / 输入区 / ─── / 页脚）----
        # live 区 = spinner 行 + 子代理树 + 任务清单（claude code 同款：
        # 回合进行中挂在输入框上方，回合结束整块收起）
        live_area = ConditionalContainer(
            Window(
                FormattedTextControl(_live_text, show_cursor=False),
                height=lambda: Dimension(
                    min=0, max=20,
                    preferred=max(1, len(_live_lines())),
                ),
                dont_extend_height=True,
                # 禁软换行：行超宽直接裁掉——软换行会让实际行数超过
                # 高度回调报的数，布局错位、面板撕碎
                wrap_lines=False,
            ),
            filter=Condition(lambda: bool(_live_lines())),
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
            "status-bar": "fg:#888888",
            "separator": "fg:#555555",
            "placeholder": "fg:#777777",
            "live-dim": "fg:#8a8a8a",
            # 补全候选区（内联区块版：暗字列表 + 选中项高亮）
            "hint-dim": "fg:#8a8a8a",
            "hint-current": "fg:#00aa88 bold",
        }
        _style_base.update(cli_skin.get_pt_style_overrides())
        # 布局自上而下：live 区 / ─── / 补全候选区 / 输入区 / ─── / 页脚。
        # 补全候选区不是浮层——是普通区块，出现时把输入框往下推，
        # 不叠在任何内容上面（用户反馈：浮层观感是「悬浮窗」，不要）
        body = HSplit(
            [live_area, separator, completions_area, input_area,
             separator, status_bar])
        app = Application(
            layout=Layout(body),
            key_bindings=_build_key_bindings(
                input_queue, eof_sentinel, interrupt_fn, force_exit_fn),
            output=output,
            style=Style.from_dict(_style_base),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,   # 退出时擦掉操作台，不冻进滚动历史
        )
        # 把状态黑板挂到 app 上：spinner 线程要读写它（翻帧/记回合起点）
        app._codeagent_ui_state = state
        app._codeagent_completions_area = app_probe_completions_area
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
                        try:
                            import cli_live
                            cli_live.turn_started(getattr(rt, "agent", None))
                        except Exception:
                            pass
                    elif not active:
                        state["turn_started"] = None              # 归零
                        try:
                            import cli_live
                            cli_live.turn_ended()
                        except Exception:
                            pass
                        # 回合结束 live 面板收起 → 渲染高度回落
                        #（不掀地板的话，面板撑高过的行数会赖着，
                        # 全塞给输入框把它顶成大空箱子）
                        _tighten_next_render(app)
                    state["was_active"] = active
                    state["frame"] = (state["frame"] + 1) % 1000
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
    """注册跨线程输入桥：工作线程的提问 → 终端让渡通道。

    大白话：stdin 被 pt 独占后，工作线程（审批/确认/提问选择器）里的
    input() 读不到字。桥把提问函数搬进 UI 线程、经 cli_ui.terminal_
    handover 真正挂起界面（擦屏+退出 raw 模式），终端还给经典输入，
    用户答完再恢复界面。app 为 None 时注销桥（直读）。
    注意不走 pt 的 run_in_terminal——那个靠 ContextVar 找 app，跨线程
    调度过来永远拿 None，界面根本不挂起（详见 terminal_handover 注释）。
    """
    import asyncio
    import cli_ui

    if app is None:
        cli_ui.set_input_bridge(None)
        return

    def _bridge(func):
        # 界面在退出（强退/EOF 收摊）→ 终端已回经典模式，原地问就行；
        # 别再排协程——app.loop 那会儿可能已经停了，排进去的协程永远
        # 不跑（还会冒「coroutine never awaited」警告），提问线程干等
        try:
            if getattr(app, "is_done", True):
                return func()
            future = asyncio.run_coroutine_threadsafe(
                cli_ui.terminal_handover(app, func), app.loop)
        except Exception as e:
            # 排队失败 = 协程从没跑过 = func 一次都没执行——原地补跑
            # （此时终端多半已不在 raw 模式，直读可行）
            cli_ui.logger.warning("输入桥排队失败，原地执行: %s", e)
            return func()
        # 不设超时：审批/提问本来就要等用户慢慢想，只阻塞提问的
        # 工作线程，UI 事件循环照常转（fn 在执行器线程里跑）。
        # 注意：能走到这说明调度成功——func 的异常经 future 原样抛回，
        # 绝不能在这接住重跑（提问读一半失败不能再来一遍）
        return future.result()

    cli_ui.set_input_bridge(_bridge)
