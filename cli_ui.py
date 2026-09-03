"""CLI 的共享输出工具。

为什么单独一个文件：Rich 的 Console（终端美化输出的对象）必须全局只有
一份。cli.py 和各个 cli_*_cmds 命令模块都从这儿 import 同一个 console，
这样测试里替换掉 `cli.console.print` 时，所有命令处理函数的输出都会被
"截获"，测试才好断言。

BridgeConsole（hermes 的 ChatConsole 同款思路）——两座桥：

1. print 桥：rich 的彩色输出是「原始 ANSI 字节」（\\x1b[2m 这种），
   经 patch_stdout 直写到 legacy 控制台，控制台不认 VT 就满屏
   ?[1;2m 乱码。桥先让 rich 渲染进内存串，再走 prompt_toolkit 的
   print_formatted_text(ANSI(...)) 打印——pt 自己解析 ANSI、按输出
  驱动上色，跟终端开没开 VT 无关（流式框的颜色从来不乱，就是因为
   走这条路；cprint 在工作线程直调这条路也实测无恙）。
2. input 桥：stdin 被 pt 独占（raw 模式读按键），工作线程（审批/
   确认）里 console.input 永远读不到字。桥把提问搬进 pt 的
   run_in_terminal 通道：暂停界面渲染 → 终端还给经典输入 →
   用户答完 → 恢复界面。
"""

import logging
import re
import threading
from io import StringIO

from rich.console import Console

logger = logging.getLogger("cli")

# OSC-8 超链接序列：Win32 控制台画不了，漏过去就是一坨乱码——剥掉
_OSC8_RE = re.compile(r"\x1b\]8;[^\x1b]*\x1b\\")

# input 桥（cli_layout 装配时注册；None=直读）
_input_bridge = None


def set_input_bridge(fn) -> None:
    """注册/注销 input 桥（fn: Callable[[Callable], Any]——把函数搬进
    pt 管理的终端里执行并回传返回值）。"""
    global _input_bridge
    _input_bridge = fn


def _emit_ansi(text: str) -> None:
    """ANSI 文本 → pt 解析通道打印（print_formatted_text）。

    任何线程都能调（内部自动落到正确上下文）；失败退回裸 print
    （打印挂了不能断业务）。
    """
    try:
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import ANSI
        print_formatted_text(ANSI(text), end="")
    except Exception:
        try:
            print(text, end="")
        except Exception:
            pass


class BridgeConsole(Console):
    """共享 Console：print 走 pt 解析通道，input 走 run_in_terminal。

    print 细节：先用「内存里的真彩 Console」把参数渲染成 ANSI 文本
    （markup/表格/样式与原 print 完全兼容），剥掉 OSC-8 后经 pt 打印。
    调用方显式传了 file= 的照旧直写（人家就是要写到别处去）。
    """

    def print(self, *args, **kwargs):
        if kwargs.get("file") is not None:
            return Console.print(self, *args, **kwargs)
        try:
            import shutil
            width = shutil.get_terminal_size((80, 24)).columns
            sio = StringIO()
            inner = Console(
                file=sio, force_terminal=True, color_system="truecolor",
                width=width, legacy_windows=False,
            )
            inner.print(*args, **kwargs)
            text = _OSC8_RE.sub("", sio.getvalue())
        except Exception:
            # 渲染桥出问题：退回原生直写（老行为——乱码也比丢输出强）
            return Console.print(self, *args, **kwargs)
        _emit_ansi(text)
        return None

    def input(self, prompt="", **kwargs):
        if _input_bridge is not None and \
                threading.current_thread() is not threading.main_thread():
            try:
                return _input_bridge(
                    lambda: Console.input(self, prompt, **kwargs)
                )
            except Exception as e:
                logger.warning("输入桥改道失败，退回直读: %s", e)
        return Console.input(self, prompt, **kwargs)


console = BridgeConsole()
