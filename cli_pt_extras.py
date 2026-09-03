"""prompt_toolkit 输入解析表的增强补丁（从 hermes 移植）。

cli_layout import 本模块时立即安装一次。每个安装器往 prompt_toolkit 的
``ANSI_SEQUENCES`` 表里塞几条映射，让现代键盘协议（Kitty / xterm
modifyOtherKeys）发出的字节序列解码成我们已经绑好的键元组——
不用再写新的键位绑定，终端发什么都能落到正确的处理函数上。

单独放一个文件（不塞进 cli_layout）是为了能脱离整个界面运行时、
单独单测这些注册。
"""

from __future__ import annotations


def install_shift_enter_alias() -> int:
    """把 Shift+Enter 的字节序列映射到 Alt+Enter 的键元组。

    这样终端就算对 Shift+Enter 发出独立序列，也会命中现有的
    Alt+Enter 换行处理器。

    覆盖的序列：
      - "\\x1b[13;2u"     — Kitty 键盘协议 / CSI-u，修饰键=2（Shift）
      - "\\x1b[27;2;13~"  — xterm modifyOtherKeys=2，修饰键=2（Shift）
      - "\\x1b[27;2;13u"  — 某些发送端用的另一种排序

    CSI-u 序列 stock prompt_toolkit 里没有；modifyOtherKeys 变体
    ``\\x1b[27;2;13~`` 有——但被映射成裸 ``Keys.ControlM``（Shift+Enter
    和 Enter 行为一模一样），这正是本安装器要修的 bug。所以这几条
    无条件覆写；其余 ``\\x1b[27;...;13~``（Ctrl/Alt 的变体）不动。

    macOS Terminal 和 stock Windows Terminal 对 Enter 和 Shift+Enter
    发的是同一个字节——应用层无解，这些序列根本到不了我们手里。

    返回：实际改了映射的序列条数。
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    alt_enter = (Keys.Escape, Keys.ControlM)
    changed = 0
    for seq in ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b[27;2;13u"):
        if ANSI_SEQUENCES.get(seq) != alt_enter:
            ANSI_SEQUENCES[seq] = alt_enter
            changed += 1
    return changed


def install_ctrl_enter_alias() -> int:
    """把 Ctrl+Enter 的字节序列映射到 Alt+Enter 的键元组（换行）。

    覆盖的序列：
      - "\\x1b[13;5u"     — Kitty 键盘协议 / CSI-u，修饰键=5（Ctrl）
      - "\\x1b[27;5;13~"  — xterm modifyOtherKeys=2，修饰键=5（Ctrl）
      - "\\x1b[27;5;13u"  — 另一种排序

    stock prompt_toolkit 一条都没映射。不装这个别名的话，
    Kitty/mintty/xterm-modifyOtherKeys 用户永远得不到 Ctrl+Enter 换行
    ——按键变成一串裸 CSI 序列漏进默认的插字处理器。

    返回：实际改了映射的序列条数。
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    alt_enter = (Keys.Escape, Keys.ControlM)
    changed = 0
    for seq in ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[27;5;13u"):
        if ANSI_SEQUENCES.get(seq) != alt_enter:
            ANSI_SEQUENCES[seq] = alt_enter
            changed += 1
    return changed


def install_ignored_terminal_sequences() -> int:
    """把终端发的噪声序列映射成 ``Keys.Ignore``，VT100 解析器直接吃掉。

    目前覆盖焦点上报：
      - ``\\x1b[I`` — 终端重新获得焦点（focus in）
      - "\\x1b[O" — 终端失去焦点（focus out）

    Ghostty / iTerm2 / 部分 xterm 在用户切标签页/窗口时会吐这些序列。
    prompt_toolkit 默认不认识，解析器把它们当普通按键（ESC、``[``、
    ``I``/``O``）漏进输入缓冲，输入框里凭空多出 "[I" 这种鬼字。

    在解析器层面注册成 Ignore，比事后正则清洗干净——字节根本到不了
    缓冲区。用 setdefault：用户或下游已注册的优先。

    返回：实际改了映射的序列条数。
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
    except Exception:
        return 0

    changed = 0
    for seq in ("\x1b[I", "\x1b[O"):
        if seq not in ANSI_SEQUENCES:
            ANSI_SEQUENCES[seq] = Keys.Ignore
            changed += 1
    return changed


def install_all() -> int:
    """一次装齐三个安装器（cli_layout import 本模块时调用）。"""
    return (
        install_shift_enter_alias()
        + install_ctrl_enter_alias()
        + install_ignored_terminal_sequences()
    )
