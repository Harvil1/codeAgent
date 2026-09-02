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


class SlashCompleter:
    """三级补全：命令名（注册表+技能+技能束同一池）→ 命令参数。

    实现成 prompt_toolkit 的 Completer 协议（实现 get_completions 生成器）。
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
                # 二级：该命令的参数补全
                fn = self._arg_completers.get(parts[0])
                if fn:
                    from prompt_toolkit.completion import Completion
                    for cand in fn(text) or []:
                        yield Completion(str(cand))
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
