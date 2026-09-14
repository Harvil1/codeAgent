"""ask_user 的提问面板——claude code 的 AskUserQuestion 同款界面。

大白话：AI 想问用户选择题，就画一块这样的面板（1:1 复刻 claude code）：

  ───────────────────────────（终端全宽分隔线）
   [ ] FAQ 位置              ← 短标题；多选画勾选框，有勾选变 [x]
                              （空行）
  FAQ 常见问题板块加在哪里？   ← 问题正文
                              （空行）
    1. 价格页（推荐）
       转化决策点，问题围绕免费额度……   ← 描述按终端宽度换行
    5. Type something.         ← 自定义输入
  ───────────────────────────（第二条分隔线）
  > 6. Chat about this         ← 自由对话逃生项，永远最后；> 是光标
                              （空行）
  Enter to select · ↑/↓ to navigate · ctrl+g to edit in Notepad · Esc to cancel

多选：选项前画 [ ]/[x]，空格切换，提示栏多一句 space to toggle。
选中 Type something./Chat about this 后面板不退，底部变输入行直接打字
（Enter 提交、Esc 返回）；ctrl+g 拉记事本写长答案。

运行环境：这个面板跑在 cli_ui 的 input 桥里（pt 的 run_in_terminal
通道）——主界面先收起来、终端还给经典模式，这里再用一个临时的小
prompt_toolkit Application 画面板；答完退出，主界面恢复。

fail-open 铁律：任何异常都往「降级为老式编号输入」走——提问通道
断了不能把 AI 的提问吞了；记事本起不来只返回 None，不阻断。
"""

import logging
import shutil
import unicodedata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 纯函数：宽度/换行/布局文本拼装（可单测，不碰终端）
# ---------------------------------------------------------------------------

def _visual_width(s: str) -> int:
    """CJK 宽度感知的显示宽度：全宽字符（中日韩）算 2 列，其他算 1 列。

    为什么要它：终端里一个汉字占两格，len() 数不出真实宽度，
    换行/对齐按 len() 算必乱套。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(s or ""))


def wrap_cjk(text: str, width: int) -> list:
    """按显示宽度把文本拆成多行（宽字符不劈成两半）。

    大白话：像排版工一样，一行塞满就另起一行，汉字这种"宽家伙"
    整个搬下去，不会劈成乱码。

    参数：
        text: 原文（strip 后处理）
        width: 目标行宽（显示列数）
    返回：行列表；空文本返回 []。
    """
    text = str(text or "").strip()
    if not text:
        return []
    if width <= 0 or _visual_width(text) <= width:
        return [text]
    lines = []
    cur = []
    cur_w = 0
    for ch in text:
        ch_w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cur and cur_w + ch_w > width:
            lines.append("".join(cur))
            cur = []
            cur_w = 0
        cur.append(ch)
        cur_w += ch_w
    if cur:
        lines.append("".join(cur))
    return lines


def hint_text(multi: bool, mode: str = "list") -> str:
    """底部提示栏文案：选项态（单/多选）和输入态三种。"""
    if mode == "input":
        return "Enter 提交 · Esc 返回 · ctrl+g 记事本编辑"
    if multi:
        return ("Enter to select · ↑/↓ to navigate · space to toggle"
                " · ctrl+g to edit in Notepad · Esc to cancel")
    return ("Enter to select · ↑/↓ to navigate"
            " · ctrl+g to edit in Notepad · Esc to cancel")


def render_fragments(question, header, options, cursor, checked, multi,
                     width=80, with_hint=True):
    """整幅提问面板的 pt 片段列表（纯函数，可单测）。

    布局 1:1 复刻 claude code 的 AskUserQuestion（见模块头图示）。
    options 最后两项约定为 special=="type" / "chat"（ask_via_selector
    自动追加），special=="chat" 前画第二条分隔线；special 项不画
    勾选框、不挂描述。

    参数：
        question: 问题正文（多行原样）
        header: 短标题（空串则不画标题行）
        options: [{label, description?, special?}, ...]
        cursor: 光标所在下标（> 标记 + 高亮）
        checked: 多选已勾选项下标集合
        multi: 是否多选
        width: 面板宽度（分隔线长度/换行宽度）
        with_hint: True 带底部提示栏（单测整幅用）；
            False 不带（交互态由外层 hint Window 动态画）
    返回：[(style, text), ...] pt 片段。
    """
    frags = []
    divider = "─" * max(10, width)

    def _row(style, text):
        frags.append((style, text))
        frags.append(("", "\n"))

    _row("class:q-divider", divider)
    if header:
        if multi:
            box = "x" if checked else " "
            _row("class:q-header", f" [{box}] {header}")
        else:
            _row("class:q-header", f" {header}")
        _row("", "")
    for ln in (question or "").splitlines() or [""]:
        _row("class:q-title", ln)
    _row("", "")

    for i, opt in enumerate(options or []):
        special = opt.get("special")
        if special == "chat":
            _row("class:q-divider", divider)
        selected = (i == cursor)
        style = "class:q-selected" if selected else "class:q-opt"
        mark = "> " if selected else "  "
        prefix = f"{mark}{i + 1}. "
        if multi and not special:
            prefix += "[x] " if i in checked else "[ ] "
        label_lines = wrap_cjk(opt.get("label", ""),
                               width - _visual_width(prefix))
        _row(style, prefix + (label_lines[0] if label_lines else ""))
        pad = " " * _visual_width(prefix)
        for cont in label_lines[1:]:
            _row(style, f"{pad}{cont}")
        desc = opt.get("description") or ""
        if not special and desc:
            for dl in wrap_cjk(desc, width - 5):
                _row("class:q-desc", f"     {dl}")

    _row("", "")
    if with_hint:
        _row("class:q-hint", hint_text(multi))
    return frags


# ---------------------------------------------------------------------------
# ctrl+g 记事本编辑（Windows 优先）
# ---------------------------------------------------------------------------

def edit_in_notepad(initial: str = "", editor: str = None):
    """弹系统记事本让用户编辑长答案（ctrl+g），返回保存的文本。

    大白话流程：临时写个 txt → 拉起记事本、堵着等用户改完关窗口 →
    读回内容当答案。好比让用户去隔壁房间写板书，写完拍回来。

    参数：
        initial: 预填文本（输入态把草稿带进去）
        editor: 编辑器命令（测试注入用；默认 notepad.exe）
    返回：用户保存的文本（strip 过，空文本算没写返回 None）；
        任何一步出问题（记事本没起来/文件读不回）也返回 None——
        fail-open，绝不抛异常阻断提问。
    """
    import os
    import subprocess
    import tempfile
    fd = None
    path = ""
    try:
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="codeagent_ask_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(initial or "")
        fd = None
        subprocess.run([editor or "notepad.exe", path], check=False)
        # utf-8-sig：容忍记事本存出的 BOM 头
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read().strip()
        return text or None
    except Exception as e:
        logger.debug("ctrl+g 记事本编辑不可用: %s", e)
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 交互主体：临时 pt Application
# ---------------------------------------------------------------------------

def run_selector(question, header, options, multi=False):
    """画 claude code 同款提问面板，返回 (result, cancelled)。

    键位：↑/↓ 移光标、1-9 直达、Enter 选定（多选=提交勾选）、空格勾选、
    Esc/Ctrl+C 取消、ctrl+g 记事本；选中 Type something./Chat about this
    后进输入态（面板底部变输入行，Enter 提交、Esc 返回列表）。

    参数：
        question: 问题正文（多行原样展示）
        header: 短标题（空串不画标题行）
        options: 调用方已追加 special=="type"/"chat" 两个固定项
        multi: True 多选；False 单选

    返回：
        (result: dict, cancelled: bool)
        - 正常: ({"answers": [标签或用户文本], "chat": bool}, False)
        - 取消: ({}, True)
        - pt 不可用/渲染炸了: 抛异常给上层降级（老式编号输入）
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (ConditionalContainer, HSplit, Layout,
                                       VSplit, Window)
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    state = {"cursor": 0, "checked": set(), "mode": "list",
             "input_target": "type", "done": False,
             "result": None, "cancelled": False}
    input_buf = Buffer(multiline=False)
    in_list = Condition(lambda: state["mode"] == "list")

    kb = KeyBindings()

    def _finish(result, cancelled=False):
        state["result"] = result
        state["cancelled"] = cancelled
        state["done"] = True
        try:
            from prompt_toolkit.application import get_app
            get_app().exit()
        except Exception:
            pass

    def _submit_custom(text, chat):
        """把一段自由文本当答案提交（Type something/Chat/记事本共用）。"""
        text = (text or "").strip()
        if not text:
            return
        _finish({"answers": [text], "chat": bool(chat)})

    def _focus(win):
        try:
            from prompt_toolkit.application import get_app
            get_app().layout.focus(win)
        except Exception:
            pass

    def _enter_input_mode(target):
        state["mode"] = "input"
        state["input_target"] = target
        _focus(input_field_window)

    def _enter_list_action():
        if not options:
            _finish({}, cancelled=True)
            return
        i = state["cursor"]
        opt = options[i]
        special = opt.get("special")
        if special in ("type", "chat"):
            _enter_input_mode(special)
            return
        if multi:
            # 多选 Enter = 提交已勾选项（一个没勾就先勾上当前项提交）
            if not state["checked"]:
                state["checked"].add(i)
            _finish({"answers": [options[j]["label"]
                                 for j in sorted(state["checked"])],
                     "chat": False})
        else:
            _finish({"answers": [opt["label"]], "chat": False})

    # enter 用 eager：抢在输入框控件自己的 enter 绑定之前拿到键，
    # 否则单行 Buffer 的 enter 会被控件吞掉（插换行/触发 accept），
    # 面板就关不掉也提交不了
    @kb.add("enter", eager=True)
    def _enter(event):
        if state["mode"] == "input":
            _submit_custom(input_buf.text, state["input_target"] == "chat")
        else:
            _enter_list_action()

    @kb.add("up", filter=in_list)
    def _up(event):
        state["cursor"] = (state["cursor"] - 1) % max(1, len(options))

    @kb.add("down", filter=in_list)
    def _down(event):
        state["cursor"] = (state["cursor"] + 1) % max(1, len(options))

    @kb.add("space", filter=in_list)
    def _space(event):
        if multi and options and not options[state["cursor"]].get("special"):
            i = state["cursor"]
            if i in state["checked"]:
                state["checked"].discard(i)
            else:
                state["checked"].add(i)

    @kb.add("escape", eager=True)
    def _esc(event):
        if state["mode"] == "input":
            state["mode"] = "list"
            _focus(list_window)
        else:
            _finish({}, cancelled=True)

    @kb.add("c-c")
    def _cc(event):
        _finish({}, cancelled=True)

    @kb.add("c-g", eager=True)
    def _cg(event):
        # 记事本编辑：输入态带草稿进去，回来续写；列表态直接当自定义答案
        initial = input_buf.text if state["mode"] == "input" else ""
        text = edit_in_notepad(initial)
        if text is None:
            return
        if state["mode"] == "input":
            from prompt_toolkit.document import Document
            input_buf.document = Document(text, len(text))
        else:
            _submit_custom(text, chat=False)

    # 数字直达：1-9（多选=打勾，单选=立即选定；special 项=进输入态）
    for digit in range(1, 10):
        @kb.add(str(digit), filter=in_list)
        def _pick(event, _d=digit):
            if _d > len(options):
                return
            i = _d - 1
            special = options[i].get("special")
            if special in ("type", "chat"):
                state["cursor"] = i
                _enter_input_mode(special)
                return
            if multi:
                state["cursor"] = i
                if i in state["checked"]:
                    state["checked"].discard(i)
                else:
                    state["checked"].add(i)
            else:
                _finish({"answers": [options[i]["label"]], "chat": False})

    # ---- 布局：面板主体 + 输入行（输入态才出现）+ 提示栏 ----
    list_window = Window(
        FormattedTextControl(
            lambda: render_fragments(
                question, header, options, state["cursor"],
                state["checked"], multi,
                width=shutil.get_terminal_size((80, 24)).columns,
                with_hint=False),
            show_cursor=False),
        dont_extend_height=True)

    input_field_window = Window(BufferControl(buffer=input_buf),
                                wrap_lines=True)
    input_row = ConditionalContainer(
        VSplit([
            Window(FormattedTextControl(lambda: "> "), width=2,
                   dont_extend_width=True),
            input_field_window,
        ]),
        filter=Condition(lambda: state["mode"] == "input"))

    hint_window = Window(
        FormattedTextControl(
            lambda: hint_text(multi, state["mode"]),
            show_cursor=False),
        dont_extend_height=True)

    app = Application(
        layout=Layout(HSplit([list_window, input_row, hint_window])),
        key_bindings=kb,
        style=Style.from_dict({
            "q-title": "bold",
            "q-header": "bold",
            "q-selected": "fg:#00aa88",
            "q-opt": "",
            "q-desc": "fg:#777777",
            "q-divider": "fg:#555555",
            "q-hint": "fg:#777777",
        }),
        full_screen=False,
    )
    # 关键：面板必须开**新线程**跑。本函数跑在 input 桥的
    # run_in_terminal 通道里（主界面事件循环的线程），线程里已有事件
    # 循环在转——app.run() 内部的 asyncio.run 会当场炸「cannot be
    # called from a running event loop」。新线程自带新循环，互不打架；
    # 主界面此刻已挂起（不读 stdin），stdin 让给面板，答完还回来。
    # 线程里的异常带回主线程重抛（ask_via_selector 接住走降级）——
    # 不然线程静默死掉、用户看着提问没反应。
    import threading
    _err = {}

    def _run_app():
        try:
            app.run()
        except Exception as e:  # noqa: BLE001
            _err["e"] = e

    _th = threading.Thread(target=_run_app, daemon=True, name="cli-question")
    _th.start()
    _th.join()
    if "e" in _err:
        raise _err["e"]

    if state["cancelled"]:
        return {}, True
    return state["result"] or {}, False


# ---------------------------------------------------------------------------
# 组合入口：面板 + 降级（cli.py 的桥接函数调这个）
# ---------------------------------------------------------------------------

def ask_via_selector(question, options, multi, header="",
                     fallback_input=None):
    """完整提问流程：先试方向键面板，炸了退回 fallback_input 编号输入。

    参数：
        question/options/multi: 同 run_selector（options 不用带 special 项，
            本函数自动追加 Type something./Chat about this）
        header: 短标题（空串自动截问题前 12 字兜底）
        fallback_input: fn(prompt) -> str，降级用的老式输入
            （cli.py 传 console.input 的包装）

    返回：{"answers": [str], "chat": bool}（answers 空 = 用户取消）。
    """
    opts = [dict(o) for o in (options or [])]
    opts.append({"label": "Type something.", "special": "type"})
    opts.append({"label": "Chat about this", "special": "chat"})
    short_header = ((header or "").strip() or (question or "").strip())[:12]
    try:
        result, cancelled = run_selector(question, short_header, opts, multi)
        if cancelled:
            return {"answers": [], "chat": False}
        return result or {"answers": [], "chat": False}
    except Exception as e:
        # 面板起不来（无头/终端不支持）——降级老式编号输入
        logger.debug("方向键提问面板不可用，降级编号输入: %s", e)
        return _fallback_number_input(question, opts, multi, fallback_input)


def _fallback_number_input(question, options, multi, fallback_input) -> dict:
    """老式降级：打印选项 + 敲序号（面板画不出来时的保底通道）。"""
    lines = [question, ""]
    for i, opt in enumerate(options):
        label = opt.get("label", "")
        desc = opt.get("description") or ""
        lines.append(f"{i + 1}. {label}" + (f" — {desc}" if desc else ""))
    lines.append("（输序号可选多项逗号分隔；直接输入文字=自定义答案）")
    try:
        from cli_ui import emit_ansi
        emit_ansi("\n".join(lines) + "\n")
    except Exception:
        pass
    raw = ((fallback_input("选择/输入 > ") if fallback_input else "")
           or "").strip()
    answers = []
    chat = False
    n = len(options)
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= n:
            i = int(part) - 1
            special = options[i].get("special")
            if special:
                # special 项也走自由输入；选 Chat about this 记 chat 标记
                custom = ((fallback_input("请输入你的答案 > ")
                           if fallback_input else "") or "").strip()
                if custom:
                    answers.append(custom)
                    chat = (special == "chat")
            else:
                answers.append(options[i]["label"])
        elif not part.isdigit():
            answers.append(part)
    if not multi:
        answers = answers[:1]
    return {"answers": answers, "chat": chat}
