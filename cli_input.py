"""输入层——降级通道 + 补全器 + 中断判断器。

分工：真终端界面在 cli_layout.py（常驻 Application 操作台）；这个文件
只留三样东西——
1. 降级通道 read_line（pt 不可用/终端画不了界面时，console.input 读行）；
2. SlashCompleter 三级补全器（cli_layout 的 TextArea 和降级通道共用）；
3. build_interrupt_fn（Ctrl+C 回合中中断判断器，两边共用）。

读到的内容怎么处理（队列、哨兵、slash 分发）全在 cli.run_interactive，
一根线不动。
"""

import logging

from cli_ui import console

logger = logging.getLogger(__name__)

_PROMPT_TEXT = "你: "


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


def read_line(rt):
    """降级通道专用：阻塞读一行（或多行）输入。

    真终端界面在 cli_layout（rt.prompt_session 装的是 Application）；
    走到这里说明操作台没建起来（返回 None），退回 console.input
    （提示文案保持「你: 」一致，功能不丢，只是没了历史/补全这些糖）。
    """
    session = getattr(rt, "prompt_session", None)
    if session is None:
        return console.input("[bold cyan]你:[/bold cyan] ")
    return session.prompt(_PROMPT_TEXT)


# 补全器基类：prompt_toolkit 可用就继承它的 Completer（真终端的异步补全
# 通道调 get_completions_async——那是基类方法，裸鸭子类没有，会在打字
# 的时候崩掉"Unhandled exception in event loop"）；pt 缺失（降级通道）就
# 退化成裸 object，同步 get_completions 照样能用。
try:
    from prompt_toolkit.completion import Completer as _PtCompleter
except Exception:  # pragma: no cover - 降级环境
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
