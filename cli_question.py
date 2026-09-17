"""ask_user 的提问面板——claude code 的 AskUserQuestion 同款界面（v2）。

大白话：AI 问用户选择题，画一块和 claude code 一样的面板：

  ─────────────────────────────（终端全宽分隔线）
   [x] 任务方向  [ ] 兴趣领域     ← 问题标签行：这批问题答到哪了（>1 问才显示）
   [ ] FAQ 位置                ← 当前题短标题；多选画勾选框（勾过变 [x]）
                              （空行）
  FAQ 常见问题板块加在哪里？       ← 问题正文
                              （空行）
    1. 价格页（推荐）
       转化决策点，问题围绕免费额度……   ← 描述按终端宽度换行
  > 2. Type something▌          ← 行内输入行！光标落上直接打字（打字替换占位符）
    3. ✓ Submit                  ← 多选才有：统一提交行
  ─────────────────────────────（分隔线）
    4. Chat about this           ← 不想选？退出问卷用自己的话聊（claude code 的
                                    「中止问卷转对话」）
  Enter/Space to select · ↑/↓ to navigate · ctrl+g to edit in Notepad · Esc to cancel

交互（对齐 claude code 实测行为）：
  - 自填行（Type something.）就是个行内输入框：光标移上去直接敲键盘输入，
    占位符让位给正文；行聚焦时数字键当文本敲、Esc 先清稿再取消。
  - 单选：选项上回车/空格即选定；自填行有字回车即提交自填。
  - 多选：选项上回车/空格=勾选；自填行打字自动算作已选；
    Submit 行统一提交，一个没勾又没自填时 Submit 无效。
  - Chat about this：回车退出面板，直接在命令行输入想说的话
    （已答过的题保留在汇总里）。
  - ctrl+g：拉记事本写长答案，保存关闭后内容填进自填行。

运行环境：面板跑在 cli_ui 的 input 桥里（pt 的 run_in_terminal 通道）。
fail-open 铁律：面板炸了降级为老式编号输入；记事本起不来只返回 None。
"""

import logging
import shutil
import unicodedata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 纯函数：宽度/换行/提示栏/行号（可单测，不碰终端）
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


def hint_text(multi: bool, custom_focused: bool = False) -> str:
    """底部提示栏文案：单选/多选/自填行聚焦三种状态。"""
    if custom_focused:
        return "Enter to submit · Esc to clear · ctrl+g to edit in Notepad"
    if multi:
        return ("Enter/Space to toggle · ↑/↓ to navigate · Submit 提交"
                " · ctrl+g to edit in Notepad · Esc to cancel")
    return ("Enter to select · ↑/↓ to navigate"
            " · ctrl+g to edit in Notepad · Esc to cancel")


def row_indices(n_options: int, multi: bool,
                allow_custom: bool = True, allow_chat: bool = True) -> dict:
    """行号表：普通选项 0..N-1、自填行 N、(多选)Submit 行、Chat 永远最后。

    大白话：整块面板的可选行从上到下编了号，数字直达键按这个对号入座。
    allow_custom/allow_chat 为 False 时不画对应行（-1 表示不存在）；
    多选没有自填行时 Submit 直接排在选项后面。
    """
    custom_i = n_options if allow_custom else -1
    if multi:
        submit_i = n_options + 1 if allow_custom else n_options
        last = submit_i
    else:
        submit_i = None
        last = custom_i if allow_custom else n_options - 1
    chat_i = last + 1 if allow_chat else -1
    total = (chat_i + 1) if allow_chat else (last + 1)
    return {"custom": custom_i, "submit": submit_i, "chat": chat_i,
            "total": max(total, n_options)}


def build_above(question, header, options, cursor, checked, multi,
                width=80, chips=None):
    """面板上半部片段：分隔线 + (多问时)问题标签行 + 标题 + 问题 + 普通选项。

    自填行不在这里——交互层用真输入框画它（见 run_selector）；
    静态整幅渲染走 render_fragments（自填行画占位文本版）。
    """
    frags = []
    divider = "─" * max(10, width)

    def _row(style, text):
        frags.append((style, text))
        frags.append(("", "\n"))

    _row("class:q-divider", divider)
    if chips:
        # 问题标签行：已答 [x]（亮），未答 [ ]（暗）——像进度打卡。
        # 每个 chip 自带前后各一格空隙，相邻两个正好隔 2 格（CC 同款）
        for label, answered in chips:
            style = "class:q-chip-done" if answered else "class:q-chip"
            frags.append((style, f" [{'x' if answered else ' '}] {label} "))
        frags.append(("", "\n"))
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
        selected = (i == cursor)
        style = "class:q-selected" if selected else "class:q-opt"
        mark = "> " if selected else "  "
        prefix = f"{mark}{i + 1}. "
        if multi:
            prefix += "[x] " if i in checked else "[ ] "
        label_lines = wrap_cjk(opt.get("label", ""),
                               width - _visual_width(prefix))
        _row(style, prefix + (label_lines[0] if label_lines else ""))
        pad = " " * _visual_width(prefix)
        for cont in label_lines[1:]:
            _row(style, f"{pad}{cont}")
        desc = opt.get("description") or ""
        if desc:
            for dl in wrap_cjk(desc, width - 5):
                _row("class:q-desc", f"     {dl}")
    return frags


def custom_row_fragments(idx: int, cursor_on: bool, multi: bool,
                         custom_text: str = "") -> list:
    """自填行（Type something.）的静态片段——占位/有字两态。

    交互层用「前缀窗 + 真输入窗」画同一行（前缀窗占位符在空稿时显示，
    见 run_selector._prefix_frags）；本函数给 render_fragments（纯函数
    整幅渲染）和单测用，两处长相保持一致。
    """
    mark = "> " if cursor_on else "  "
    prefix = f"{mark}{idx + 1}. "
    if multi:
        prefix += "[x] " if custom_text.strip() else "[ ] "
    if custom_text:
        body = custom_text + ("▌" if cursor_on else "")
        style = "class:q-selected" if cursor_on else "class:q-opt"
    else:
        body = "Type something."
        style = "class:q-desc"
    return [(style, prefix + body), ("", "\n")]


def build_below(n_options, cursor, checked, multi, custom_text,
                width=80, with_hint=True, custom_focused=False,
                allow_chat=True):
    """面板下半部片段：自填行之后的一切——(多选)Submit 行、分隔线、
    Chat about this 行、空行、提示栏。allow_chat=False 不画 Chat 行。"""
    frags = []
    divider = "─" * max(10, width)
    idx = row_indices(n_options, multi, allow_chat=allow_chat)

    def _row(style, text):
        frags.append((style, text))
        frags.append(("", "\n"))

    if multi:
        i = idx["submit"]
        selected = (cursor == i)
        mark = "> " if selected else "  "
        style = "class:q-selected" if selected else "class:q-desc"
        _row(style, f"{mark}{i + 1}. ✓ Submit")
    _row("class:q-divider", divider)
    if allow_chat:
        i = idx["chat"]
        selected = (cursor == i)
        mark = "> " if selected else "  "
        style = "class:q-selected" if selected else "class:q-opt"
        _row(style, f"{mark}{i + 1}. Chat about this")
    _row("", "")
    if with_hint:
        _row("class:q-hint", hint_text(multi, custom_focused))
    return frags


def render_fragments(question, header, options, cursor, checked, multi,
                     width=80, chips=None, custom_text="",
                     custom_focused=False):
    """整幅面板静态片段（自填行画占位文本版）——verify/降级展示用。

    参数：
        chips: [(标签, 是否已答), ...] 多问时的进度标签行；单问传 None 不画
        custom_text: 自填行已有文字（多选时有字算勾上）
        custom_focused: 光标是否在自填行（影响提示栏文案）
    """
    frags = list(build_above(question, header, options, cursor, checked,
                             multi, width=width, chips=chips))
    idx = row_indices(len(options or []), multi)
    frags.extend(custom_row_fragments(
        idx["custom"], cursor == idx["custom"], multi, custom_text))
    frags.extend(build_below(len(options or []), cursor, checked, multi,
                             custom_text, width=width,
                             custom_focused=custom_focused))
    return frags


# ---------------------------------------------------------------------------
# ctrl+g 记事本编辑（Windows 优先）
# ---------------------------------------------------------------------------

def edit_in_notepad(initial: str = "", editor: str = None):
    """弹系统记事本让用户编辑长答案（ctrl+g），返回保存的文本。

    大白话流程：临时写个 txt → 拉起记事本、堵着等用户改完关窗口 →
    读回内容当答案。好比让用户去隔壁房间写板书，写完拍回来。

    返回：用户保存的文本（strip 过，空文本算没写返回 None）；
        任何一步出问题也返回 None——fail-open，绝不抛异常阻断提问。
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
                logger.warning("异常被吞(fail-open)", exc_info=True)
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)


# ---------------------------------------------------------------------------
# 交互主体：临时 pt Application（一问一面板）
# ---------------------------------------------------------------------------

def run_selector(question, header, options, multi=False, chips=None,
                 allow_custom=True, allow_chat=True):
    """画一问的 CC 同款面板（自填行是行内真输入框）。

    键位：↑/↓ 移光标（光标落到自填行时焦点给输入框、直接打字）、
    1-9 直达、Enter 选项=选/勾、空格同 Enter（选项上）、
    多选 Submit 行统一提交、Esc 先清自填稿再取消、ctrl+g 记事本
    （内容填进自填行）、Chat about this 转对话。

    参数：
        question/header/options/multi: 这一问的内容（options 只有普通选项，
            自填/Submit/Chat 行由本函数按行号表自动画）
        chips: 多问时的进度标签行（单问传 None）
        allow_custom: False 时不画"Type something."自填行（审批面板用）
        allow_chat: False 时不画"Chat about this"行（审批面板用）

    返回：{"answers": [选项label或自填文本], "cancelled": bool, "chat": bool}
        chat=True 表示用户选了 Chat about this（answers 为空）；
        cancelled=True 表示 Esc/Ctrl+C 取消。
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    idx = row_indices(len(options or []), multi,
                      allow_custom=allow_custom, allow_chat=allow_chat)
    state = {"cursor": 0, "checked": set(), "done": False, "result": None}
    custom_buf = Buffer(multiline=False)
    custom_focused = Condition(lambda: state["cursor"] == idx["custom"])

    kb = KeyBindings()

    def _finish(answers=None, cancelled=False, chat=False):
        state["result"] = {"answers": answers or [], "cancelled": cancelled,
                           "chat": chat}
        state["done"] = True
        try:
            from prompt_toolkit.application import get_app
            get_app().exit()
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)

    def _custom_text():
        return custom_buf.text.strip()

    def _submit_multi():
        labels = [options[j]["label"] for j in sorted(state["checked"])]
        text = _custom_text()
        if text:
            labels.append(text)   # 自填打字自动算作已选（CC 同款）
        _finish(labels)

    def _sync_focus():
        # 光标落在自填行 → 焦点给真输入框（打字进 Buffer）；
        # 移开 → 焦点还回上半窗（文本窗无按键绑定，全靠 app 级绑定）
        try:
            from prompt_toolkit.application import get_app
            win = (custom_field_window
                   if state["cursor"] == idx["custom"] else above_window)
            get_app().layout.focus(win)
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)

    def _move(delta):
        state["cursor"] = (state["cursor"] + delta) % max(1, idx["total"])
        _sync_focus()

    def _act_row(i):
        """对第 i 行执行「回车动作」：选项=选/勾，自填=提交（单选）或跳
        Submit（多选），Submit=统一提交，Chat=转对话。"""
        if i < len(options):
            if multi:
                if i in state["checked"]:
                    state["checked"].discard(i)
                else:
                    state["checked"].add(i)
            else:
                _finish([options[i]["label"]])
        elif i == idx["custom"]:
            text = _custom_text()
            if not text:
                return  # 空自填不提交（CC 同款）
            if multi:
                state["cursor"] = idx["submit"]  # 打过字自动算已选，跳去 Submit
                _sync_focus()
            else:
                _finish([text])
        elif multi and i == idx["submit"]:
            if state["checked"] or _custom_text():
                _submit_multi()   # 空选空填时 Submit 无效（CC 同款）
        elif i == idx["chat"]:
            _finish(chat=True)

    # enter 用 eager：抢在输入框控件自己的 enter 绑定之前拿到键
    @kb.add("enter", eager=True)
    def _enter(event):
        _act_row(state["cursor"])

    @kb.add("up")
    def _up(event):
        _move(-1)

    @kb.add("down")
    def _down(event):
        _move(1)

    # 空格：自填行聚焦时不绑（落给输入框打空格），其余行=回车动作
    @kb.add("space", filter=~custom_focused)
    def _space(event):
        if state["cursor"] < len(options):
            _act_row(state["cursor"])

    @kb.add("escape", eager=True)
    def _esc(event):
        if state["cursor"] == idx["custom"] and custom_buf.text:
            custom_buf.reset()   # 先清自填稿，再按一次 Esc 才取消（CC 同款）
            return
        _finish(cancelled=True)

    @kb.add("c-c")
    def _cc(event):
        _finish(cancelled=True)

    @kb.add("c-g", eager=True)
    def _cg(event):
        # 记事本编辑：内容填进自填行，用户过目后自己回车提交
        # （草稿总是带上——非自填行按 c-g 也不该把已打的稿清掉）
        initial = _custom_text()
        text = edit_in_notepad(initial)
        if text is None:
            return
        custom_buf.document = Document(text, len(text))
        state["cursor"] = idx["custom"]
        _sync_focus()

    # 数字直达：自填行聚焦时不绑（数字当文本敲进输入框，CC 同款）
    for digit in range(1, 10):
        @kb.add(str(digit), filter=~custom_focused)
        def _pick(event, _d=digit):
            if _d <= idx["total"]:
                _act_row(_d - 1)

    # ---- 布局：上半窗 + 自填行（前缀窗 + 真输入窗）+ 下半窗 ----
    above_window = Window(
        FormattedTextControl(
            lambda: build_above(
                question, header, options, state["cursor"],
                state["checked"], multi,
                width=shutil.get_terminal_size((80, 24)).columns,
                chips=chips),
            show_cursor=False),
        dont_extend_height=True)

    def _prefix_frags():
        # 自填行前缀：序号 + (多选勾选框) + 空稿时的占位提示语。
        # 占位符画在前缀窗里——输入框一有字它就让位（窗宽自动收缩），
        # 这就是「提示语变输入」的实现窍门。
        on = state["cursor"] == idx["custom"]
        style = "class:q-selected" if on else "class:q-opt"
        frags = [(style, ("> " if on else "  ") + f"{idx['custom'] + 1}. ")]
        if multi:
            frags.append((style, "[x] " if custom_buf.text.strip()
                          else "[ ] "))
        if not custom_buf.text:
            frags.append(("class:q-desc", "Type something."))
        return frags

    custom_prefix_window = Window(
        FormattedTextControl(_prefix_frags),
        dont_extend_width=True, dont_extend_height=True)
    custom_field_window = Window(BufferControl(buffer=custom_buf),
                                 wrap_lines=True, dont_extend_height=True)
    custom_row = VSplit([custom_prefix_window, custom_field_window])

    below_window = Window(
        FormattedTextControl(
            lambda: build_below(
                len(options), state["cursor"], state["checked"], multi,
                custom_buf.text,
                width=shutil.get_terminal_size((80, 24)).columns,
                custom_focused=(state["cursor"] == idx["custom"]),
                allow_chat=allow_chat),
            show_cursor=False),
        dont_extend_height=True)

    # allow_custom=False 时不画自填行（审批面板只要干净三选）
    _children = [above_window]
    if allow_custom:
        _children.append(custom_row)
    _children.append(below_window)

    app = Application(
        layout=Layout(HSplit(_children)),
        key_bindings=kb,
        style=Style.from_dict({
            "q-title": "bold",
            "q-header": "bold",
            "q-selected": "fg:#00aa88",
            "q-opt": "",
            "q-desc": "fg:#777777",
            "q-divider": "fg:#555555",
            "q-chip": "fg:#777777",
            "q-chip-done": "fg:#00aa88",
            "q-hint": "fg:#777777",
        }),
        full_screen=False,
    )
    # 关键：面板必须开**新线程**跑。本函数跑在 input 桥的
    # run_in_terminal 通道里（主界面事件循环的线程），线程里已有事件
    # 循环在转——app.run() 内部的 asyncio.run 会当场炸「cannot be
    # called from a running event loop」。新线程自带新循环，互不打架；
    # 主界面此刻已挂起（不读 stdin），stdin 让给面板，答完还回来。
    # 线程里的异常带回主线程重抛（ask_via_selector 接住走降级）。
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
    return state["result"] or {"answers": [], "cancelled": True,
                               "chat": False}


# ---------------------------------------------------------------------------
# 组合入口：一批问题逐个放面板 + 降级（cli.py 的桥接函数调这个）
# ---------------------------------------------------------------------------

def ask_via_selector(questions, fallback_input=None):
    """整批问题逐个放面板（顶部标签行显示进度），答完汇总返回。

    大白话：一份问卷有几道题，就一题一题放面板让用户作答；顶部标签
    行显示答到哪了；中途选 Chat about this 就地转对话（已答的保留）。

    参数：
        questions: [{question, header?, options, multi?}, ...]（1-4 问）
        fallback_input: fn(prompt) -> str，降级/Chat 追问用的老式输入

    返回：{"answers": [{"question", "answers", "multi"}...],
          "chat": str|None, "cancelled": bool}
    """
    qs = []
    for q in questions or []:
        question = (q.get("question") or "").strip()
        if not question:
            continue
        qs.append({
            "question": question,
            "header": ((q.get("header") or "").strip() or question[:12])[:12],
            "options": [dict(o) for o in (q.get("options") or [])],
            "multi": bool(q.get("multi", False)),
        })
    results = []

    def _chat_flow():
        """用户选了 Chat about this：退出问卷，命令行追问想说的话。"""
        text = ""
        try:
            text = ((fallback_input("用自己的话聊聊（Enter 发送，空=取消）> ")
                     if fallback_input else "") or "").strip()
        except (EOFError, KeyboardInterrupt):
            text = ""
        return {"answers": results, "chat": text or None,
                "cancelled": not text}

    # === 可回退导航：i 可以前进也可以后退（重答上一题） ===
    i = 0
    while i < len(qs):
        q = qs[i]
        chips = ([(qq["header"], j < i) for j, qq in enumerate(qs)]
                 if len(qs) > 1 else None)
        # 第一题之后的题加一个「← 上一题」导航选项（放在 Chat about this 前）
        opts = list(q["options"])
        if i > 0:
            opts.append({"label": "← 上一题", "description": "回去改上一题的答案"})
        try:
            res = run_selector(q["question"], q["header"], opts,
                               q["multi"], chips=chips)
        except Exception as e:
            logger.debug("方向键提问面板不可用，降级编号输入: %s", e)
            res = _fallback_number_input(q["question"], q["options"],
                                         q["multi"], fallback_input)
        if res.get("cancelled"):
            return {"answers": results, "chat": None, "cancelled": True}
        if res.get("chat"):
            return _chat_flow()
        # 检查是否选了「← 上一题」
        picked = res.get("answers") or []
        if any(a == "← 上一题" for a in picked):
            i = max(0, i - 1)
            # 回退时截掉上一题的结果（重答会覆盖）
            if i < len(results):
                results = results[:i]
            continue
        # 正常作答：如果是回退后重答，确保 results 长度对齐
        while len(results) > i:
            results.pop()
        while len(results) < i:
            # 理论不该到这（中间题不会跳过），兜底用空答案填
            results.append({"question": qs[len(results)]["question"],
                            "answers": [], "multi": False})
        results.append({"question": q["question"],
                        "answers": [a for a in picked if a != "← 上一题"],
                        "multi": q["multi"]})
        i += 1
    return {"answers": results, "chat": None, "cancelled": False}


def _fallback_number_input(question, options, multi, fallback_input) -> dict:
    """老式降级：打印选项 + 敲序号（面板画不出来时的保底通道）。

    返回与 run_selector 同款 dict（chat=True 时由 ask_via_selector 追问文本）。
    """
    n = len(options)
    idx = row_indices(n, multi)
    lines = [question, ""]
    for i, opt in enumerate(options):
        label = opt.get("label", "")
        desc = opt.get("description") or ""
        lines.append(f"{i + 1}. {label}" + (f" — {desc}" if desc else ""))
    lines.append(f"{idx['custom'] + 1}. Type something.（直接输文字=自填）")
    if multi:
        lines.append(f"{idx['submit'] + 1}. ✓ Submit")
    lines.append(f"{idx['chat'] + 1}. Chat about this")
    lines.append("（多选逗号分隔）")
    try:
        from cli_ui import emit_ansi
        emit_ansi("\n".join(lines) + "\n")
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)
    try:
        raw = ((fallback_input("选择/输入 > ") if fallback_input else "")
               or "").strip()
    except (EOFError, KeyboardInterrupt):
        return {"answers": [], "cancelled": True, "chat": False}
    answers = []
    chat = False
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= idx["total"]:
            i = int(part) - 1
            if i < n:
                answers.append(options[i]["label"])
            elif i == idx["chat"]:
                chat = True
            # 自填/Submit 行的序号没有独立意义（自填走输文字），跳过
        elif not part.isdigit():
            answers.append(part)
    if not multi:
        answers = answers[:1]
    return {"answers": answers, "cancelled": False, "chat": chat}
