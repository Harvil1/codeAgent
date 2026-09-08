"""ask_user 的方向键选择器——claude code 的 AskUserQuestion 同款界面。

大白话：AI 想问用户选择题，老界面是「打印一个面板 + 敲序号」；新界面
跟 claude code 一样——问题贴在墙上，选项排成一列，↑/↓ 移动 > 光标、
Enter 选中、数字直达、Esc 取消、多选空格打勾，最后一项永远是
「Type something.」（自己输入）。

运行环境：这个选择器跑在 cli_ui 的 input 桥里（pt 的 run_in_terminal
通道）——主界面先收起来、终端还给经典模式，这里再用一个临时的小
prompt_toolkit Application 画选项列表；答完退出，主界面恢复。

fail-open 铁律：任何异常都往「降级为老式编号输入」走——提问通道
断了不能把 AI 的提问吞了。
"""

import logging

logger = logging.getLogger(__name__)

# 选项 label 最长显示宽度（超了截断——列表不换行才立得住）
_LABEL_LIMIT = 60


def _clip(s: str, n: int) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# 纯函数：界面文本拼装（可单测，不碰终端）
# ---------------------------------------------------------------------------

def build_option_rows(options: list, multi: bool) -> list:
    """把选项列表拼成显示行：[(text, kind), ...]。

    kind ∈ {"opt", "desc"}；text 是不含光标/序号的正文。
    描述行缩进对齐选项正文（claude code 的描述挂在选项下一行）。
    """
    rows = []
    for opt in options or []:
        label = opt.get("label", "")
        desc = opt.get("description", "")
        mark = "○ " if multi else ""
        rows.append(("opt", f"{mark}{_clip(label, _LABEL_LIMIT)}"))
        if desc:
            rows.append(("desc", f"   {_clip(desc, _LABEL_LIMIT)}"))
    return rows


def render_fragments(options: list, cursor: int, checked: set,
                     multi: bool) -> list:
    """当前界面完整片段列表（pt fragment 元组）。

    长相（单选）：

        1. 只做防丢失核心包（推荐）      ← 光标行前有 >
           3 个 P0 bug…
        2. 防丢失 + 性能热点

    多选：选项前画 ○/●（空格切换），光标仍是 >。
    """
    frags = []
    for i, opt in enumerate(options):
        selected = (i == cursor)
        cursor_mark = "> " if selected else "  "
        label = _clip(opt.get("label", ""), _LABEL_LIMIT)
        if multi:
            box = "● " if i in checked else "○ "
            body = f"{box}{label}"
        else:
            body = label
        style = "class:q-selected" if selected else "class:q-opt"
        frags.append((style, f"{cursor_mark}{i + 1}. "))
        frags.append((style, body))
        frags.append(("", "\n"))
        desc = opt.get("description", "")
        if desc:
            pad = "   " if selected else "   "
            frags.append(("class:q-desc", f"{pad}{_clip(desc, _LABEL_LIMIT)}"))
            frags.append(("", "\n"))
    return frags


# ---------------------------------------------------------------------------
# 交互主体：临时 pt Application
# ---------------------------------------------------------------------------

def run_selector(question: str, options: list, multi: bool = False):
    """画一个内联选择器，返回 (answers, cancelled)。

    参数：
        question: 问题正文（多行支持，原样展示）
        options: [{label, description}, ...]（调用方已追加「Type something.」
                 自由输入项）
        multi: True 多选（空格打勾、Enter 提交）；False 单选（Enter 即选）

    返回：
        (answers: list[str], cancelled: bool)
        - 正常选择：answers 是 label 列表，cancelled=False
        - 选了自由输入项：answers=["__custom__"]，调用方再追问文本
        - Esc 取消：answers=[]，cancelled=True
        - pt 不可用/渲染炸了：抛异常给上层降级（老式编号输入）
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    state = {"cursor": 0, "checked": set(), "done": False,
             "answers": None, "cancelled": False}

    kb = KeyBindings()

    def _finish(answers, cancelled=False):
        state["answers"] = answers
        state["cancelled"] = cancelled
        state["done"] = True
        try:
            from prompt_toolkit.application import get_app
            get_app().exit()
        except Exception:
            pass

    @kb.add("up")
    def _up(event):
        state["cursor"] = (state["cursor"] - 1) % max(1, len(options))

    @kb.add("down")
    def _down(event):
        state["cursor"] = (state["cursor"] + 1) % max(1, len(options))

    @kb.add("enter")
    def _enter(event):
        if not options:
            _finish(None, cancelled=True)
            return
        i = state["cursor"]
        if multi:
            # 多选 Enter = 提交已勾选项（一个没勾就勾上当前项提交）
            if not state["checked"]:
                state["checked"].add(i)
            _finish([options[j]["label"] for j in sorted(state["checked"])])
        else:
            _finish([options[i]["label"]])

    @kb.add("space")
    def _space(event):
        if multi and options:
            i = state["cursor"]
            if i in state["checked"]:
                state["checked"].discard(i)
            else:
                state["checked"].add(i)

    @kb.add("escape")
    def _esc(event):
        _finish(None, cancelled=True)

    @kb.add("c-c")
    def _cc(event):
        _finish(None, cancelled=True)

    # 数字直达：1-9 直接选中该序号（多选=打勾，单选=立即返回）
    for digit in range(1, 10):
        @kb.add(str(digit))
        def _pick(event, _d=digit):
            if _d <= len(options):
                if multi:
                    state["cursor"] = _d - 1
                    if (_d - 1) in state["checked"]:
                        state["checked"].discard(_d - 1)
                    else:
                        state["checked"].add(_d - 1)
                else:
                    _finish([options[_d - 1]["label"]])

    def _body_fragments():
        frags = []
        # 问题头：加粗提问正文（多行原样）
        for ln in (question or "").splitlines() or [""]:
            frags.append(("class:q-title", ln))
            frags.append(("", "\n"))
        frags.append(("", "\n"))
        frags.extend(render_fragments(options, state["cursor"],
                                      state["checked"], multi))
        frags.append(("", "\n"))
        hint = ("Enter 选择 · ↑/↓ 移动 · 空格勾选 · 数字直达 · Esc 取消"
                if multi else
                "Enter to select · ↑/↓ to navigate · 数字直达 · Esc to cancel")
        frags.append(("class:q-hint", hint))
        return frags

    app = Application(
        layout=Layout(HSplit([Window(
            FormattedTextControl(_body_fragments, show_cursor=False),
            dont_extend_height=True,
        )])),
        key_bindings=kb,
        style=Style.from_dict({
            "q-title": "bold",
            "q-selected": "fg:#00aa88",
            "q-opt": "",
            "q-desc": "fg:#777777",
            "q-hint": "fg:#777777",
        }),
        full_screen=False,
    )
    # 关键：选择器必须开**新线程**跑。本函数跑在 input 桥的
    # run_in_terminal 通道里（主界面事件循环的线程），线程里已有事件
    # 循环在转——app.run() 内部的 asyncio.run 会当场炸「cannot be
    # called from a running event loop」。新线程自带新循环，互不打架；
    # 主界面此刻已挂起（不读 stdin），stdin 让给选择器，答完还回来。
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
        return [], True
    answers = state["answers"] or []
    return answers, False


# ---------------------------------------------------------------------------
# 组合入口：选择器 + 自由输入追问（cli.py 的桥接函数调这个）
# ---------------------------------------------------------------------------

def ask_via_selector(question: str, options: list, multi: bool,
                      fallback_input):
    """完整提问流程：先试方向键选择器，炸了退回 fallback_input。

    参数：
        question/options/multi: 同 run_selector
        fallback_input: fn(prompt) -> str，降级用的老式输入
            （cli.py 传 console.input 的包装）

    返回：answers 字符串列表（空列表 = 用户取消）。
    """
    # 自由输入口永远保留——预设选项不可能穷尽用户的想法
    opts = list(options or [])
    opts.append({"label": "Type something.", "description": "自己输入答案"})
    try:
        answers, cancelled = run_selector(question, opts, multi)
        if cancelled:
            return []
        if answers == ["Type something."]:
            raw = fallback_input("请输入你的答案 > ").strip()
            return [raw] if raw else []
        return answers
    except Exception as e:
        # 选择器起不来（无头/终端不支持）——降级老式编号输入
        logger.debug("方向键选择器不可用，降级编号输入: %s", e)
        return _fallback_number_input(question, opts, multi, fallback_input)


def _fallback_number_input(question: str, options: list, multi: bool,
                           fallback_input) -> list:
    """老式降级：打印选项 + 敲序号（选择器画不出来时的保底通道）。"""
    lines = [question, ""]
    for i, opt in enumerate(options):
        label = opt.get("label", "")
        desc = opt.get("description", "")
        lines.append(f"{i + 1}. {label}" + (f" — {desc}" if desc else ""))
    try:
        from cli_ui import emit_ansi
        emit_ansi("\n".join(lines) + "\n")
    except Exception:
        pass
    raw = (fallback_input(
        "选择序号（逗号分隔/直接输入文字）> ") or "").strip()
    answers = []
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(options):
            answers.append(options[int(part) - 1]["label"])
        elif not part.isdigit():
            answers.append(part)
    # 单选只取第一个
    return answers if multi else answers[:1]
