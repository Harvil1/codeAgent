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

# 当前在跑的 pt Application（cli.py 装配时注册；None=没有界面在跑）。
# 为什么要有它：get_app_or_none() 靠 ContextVar 取 app，只对 UI 线程生效；
# 工作线程（老主循环/流式渲染）里拿到的永远是 None——不补这个全局引用，
# 工作线程的打印就只能裸写终端，会把底部操作台行冻进滚动历史（残影）。
_active_app = None


def set_input_bridge(fn) -> None:
    """注册/注销 input 桥（fn: Callable[[Callable], Any]——把函数搬进
    pt 管理的终端里执行并回传返回值）。"""
    global _input_bridge
    _input_bridge = fn


def set_active_app(app) -> None:
    """登记/注销当前在跑的 pt Application（cli.py 界面装配和收摊时各调一次）。"""
    global _active_app
    _active_app = app


def emit_ansi(text: str) -> None:
    """ANSI 文本 → 终端打印（全程序唯一打印出口，任何线程都能调）。

    大白话：有 pt 界面在跑时，工作线程的打印必须「搬进 UI 事件循环」执行
    （run_in_terminal_async 会先收起底部操作台行、打完再重绘）——直接裸写
    会把 spinner/状态栏冻进滚动历史。UI 线程自己（主线程）不用搬，pt 的
    print_formatted_text 在 app 上下文里本来就协调好。失败退回直写
    （打印挂了不能断业务）。
    """
    def _render():
        try:
            from prompt_toolkit import print_formatted_text
            from prompt_toolkit.formatted_text import ANSI
            print_formatted_text(ANSI(text), end="")
        except Exception:
            try:
                print(text, end="")
            except Exception:
                pass

    app = _active_app
    if app is None:
        _render()
        return
    try:
        import threading
        # 主线程就是 UI 线程（app.run 在主线程跑）——不用搬，直接打
        if threading.current_thread() is threading.main_thread():
            _render()
            return
        if getattr(app, "is_done", True):
            _render()   # 界面正在退出——直写，别再排队
            return
        import asyncio
        fut = asyncio.run_coroutine_threadsafe(
            app.run_in_terminal_async(_render), app.loop)
        # 等渲染完成再返回：同一个出口排队走，行序天然有保证
        #（不等的话两次打印任务并发，收起/重绘交错会撕行）
        fut.result(timeout=5)
    except Exception:
        _render()


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
        emit_ansi(text)
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
