"""输入层——prompt_toolkit 会话工厂 + 快捷键 + 降级通道。

分工：这个文件只管「怎么读一行（或多行）输入」；读到的内容怎么处理
（队列、哨兵、slash 分发）全在 cli.run_interactive，一根线不动。

降级通道：prompt_toolkit 不可用或会话创建失败时返回 None，调用方
（cli 的输入线程）自动退回 console.input——功能不丢，只是没了
历史/补全这些糖。
"""

import logging

from cli_ui import console

logger = logging.getLogger(__name__)

_PROMPT_TEXT = "你: "


def _build_key_bindings():
    """两组快捷键（大白话：输入框的行为规则）。

    - Ctrl+C：框里有字 → 清行（防误触丢半行字）；空框 → 退出程序
      （抛 KeyboardInterrupt，与老语义「等输入时 Ctrl+C=退出」一致）
    - Esc 回车：提交多行内容（单行直接回车提交，互不干扰）
    """
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("c-c")
    def _clear_or_exit(event):
        buffer = event.app.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt)

    @kb.add("escape", "enter")
    def _submit(event):
        event.app.current_buffer.validate_and_handle()

    return kb


def build_prompt_session(completer=None, toolbar_fn=None):
    """造全局 PromptSession；环境不支持时返回 None（降级通道）。

    参数：
        completer: prompt_toolkit Completer（Task 5 接入，先留参数位）
        toolbar_fn: 底部工具栏刷新函数（Task 6 接入）

    返回：PromptSession 实例；None 表示环境不可用，调用方退回 console.input。
    """
    try:
        from constants import get_codeagent_home
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_path = get_codeagent_home() / ".input_history"
        return PromptSession(
            history=FileHistory(str(history_path)),
            key_bindings=_build_key_bindings(),
            completer=completer,
            bottom_toolbar=toolbar_fn,
            complete_while_typing=True,
            mouse_support=False,
        )
    except Exception as e:
        logger.error("PromptSession 创建失败，输入层降级为 console.input: %s", e)
        return None


def read_line(rt):
    """输入线程专用：阻塞读一行（或多行）输入。

    session 可用走 prompt_toolkit（历史/补全/粘贴全套）；不可用退回
    原来的 console.input（提示文案保持「你: 」一致）。
    """
    session = getattr(rt, "prompt_session", None)
    if session is None:
        return console.input("[bold cyan]你:[/bold cyan] ")
    return session.prompt(_PROMPT_TEXT)


def invalidate(session) -> None:
    """后台线程刷新底部工具栏用的安全阀门——任何异常都吞掉
    （刷新失败顶多工具栏不更新，绝不能连累输入线程）。"""
    try:
        if session is not None:
            session.app.invalidate()
    except Exception:
        pass
