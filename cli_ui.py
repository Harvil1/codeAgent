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


async def terminal_handover(app, fn):
    """把 fn 搬进 UI 线程执行，执行期间真正挂起界面（终端让渡给 fn）。

    大白话：工作线程想直接跟用户终端打交道（读按键/画选择器），必须先
    让 pt 界面收摊（擦屏、退出 raw 模式、停止重绘），fn 干完再重开——
    不挂起的话 stdin 还被 pt 独占，fn 里的 input() 会永远读不到字。

    为什么不复用 pt 的 in_terminal()：它靠 ContextVar 找 app，而我们是
    从工作线程跨循环调度过来的，新上下文里 get_app_or_none() 永远是
    None、界面根本不会挂起；旧代码引用的那个 pt Application 终端让渡
    方法在 3.0.53 里压根不存在（输入桥从第一天起就是坏的，审批提问一
    触发就卡死——本函数就是补这个洞）。这里照上游逻辑复刻，但显式收
    app 实例，不依赖 ContextVar。

    fn 本身的异常原样传播（绝不重跑——提问读一半失败不能再来一遍）；
    挂起/恢复失败只打 warning，fail-open 原地执行 fn。

    参数：app——pt Application（不在跑就原地执行）；fn——() -> 结果。
    返回：fn 的返回值。
    """
    import asyncio

    if app is None or not getattr(app, "_is_running", False):
        return fn()

    # ---- 挂起：擦屏 + 停重绘 + 脱离 raw 模式 ----
    detach_cm = cooked_cm = None
    try:
        detach_cm = app.input.detach()
        detach_cm.__enter__()
        cooked_cm = app.input.cooked_mode()
        cooked_cm.__enter__()
        app.renderer.erase()
        app._running_in_terminal = True   # pt 靠它停掉周期重绘
    except Exception as e:
        for cm in (cooked_cm, detach_cm):
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    logger.warning("异常被吞(fail-open)", exc_info=True)
        logger.warning("终端让渡挂起失败，原地执行（界面可能残影）: %s", e)
        return fn()

    # ---- fn 跑在执行器线程：阻塞读 stdin 不许卡 UI 事件循环 ----
    try:
        return await asyncio.to_thread(fn)
    finally:
        app._running_in_terminal = False
        for cm in (cooked_cm, detach_cm):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.warning("异常被吞(fail-open)", exc_info=True)
        try:
            app.renderer.reset()
            app._request_absolute_cursor_position()
            app._redraw()
        except Exception:
            logger.warning("异常被吞(fail-open)", exc_info=True)


def run_with_input_bridge(fn):
    """把 fn 搬进 pt 的 run_in_terminal 通道执行（跨线程借用终端）。

    大白话：工作线程想直接跟用户终端打交道（比如画方向键选择器），
    必须先让主界面挂起、把 stdin 让出来——这就是 input 桥的通道。
    没桥（无界面/直跑）就在原地执行，行为不变。

    参数：fn: () -> 结果
    返回：fn 的返回值（桥执行失败按原地执行兜底）。
    """
    if _input_bridge is not None and \
            threading.current_thread() is not threading.main_thread():
        try:
            return _input_bridge(fn)
        except Exception as e:
            logger.warning("input 桥借用失败，原地执行: %s", e)
    return fn()


def emit_ansi(text: str) -> None:
    """ANSI 文本 → 终端打印（全程序唯一打印出口，任何线程都能调）。

    大白话：工作线程直接写就行——main.py 用 patch_stdout 把 sys.stdout
    包成了代理，它会把跨线程的写「搬进 UI 事件循环 + 正确挂起/恢复界面」
    （代理在 loop 线程上下文里跑，ContextVar 认得出 app，这套收放是
    对的）。我们 print_formatted_text 写的就是这个被代理的 sys.stdout。

    教训（别走回头路）：曾经在这里用 terminal_handover 手动「挂起→打→
    重绘」，结果在 mintty/winpty 下每打一行就往滚动历史泄漏一份界面
    快照（重绘定位靠 CPR，winpty 层不可靠）——界面满屏重影。打印走
    patch_stdout 代理才是正路；terminal_handover 只留给输入桥（审批
    提问要真正接管 stdin，没有替代品，泄漏一份快照可接受）。
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
                logger.warning("异常被吞(fail-open)", exc_info=True)

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
        """工作线程的 input 走 pt 终端让渡桥（挂起界面→读→恢复）。

        桥失败时**绝不退回直读**——直读会跟 pt 抢 stdin，把 prompt_toolkit
        的事件循环打崩（Executor shutdown 满屏崩栈，实测吃过亏）。
        改为抛 EOFError：审批回调接住后自动拒绝（fail-closed），模型
        收到 permission denied 走别的路——比整个 UI 崩掉好一万倍。
        """
        if _input_bridge is not None and \
                threading.current_thread() is not threading.main_thread():
            try:
                return _input_bridge(
                    lambda: Console.input(self, prompt, **kwargs)
                )
            except Exception as e:
                logger.warning(
                    "输入桥改道失败（自动拒绝而非直读防崩UI）: %s", e,
                )
                raise EOFError(
                    f"终端让渡失败，无法读取用户输入: {e}",
                )
        # 主线程（无 pt 争抢）照常直读
        return Console.input(self, prompt, **kwargs)


console = BridgeConsole()
