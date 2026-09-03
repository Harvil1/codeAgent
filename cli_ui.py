"""CLI 的共享输出工具。

为什么单独一个文件：Rich 的 Console（终端美化输出的对象）必须全局只有
一份。cli.py 和各个 cli_*_cmds 命令模块都从这儿 import 同一个 console，
这样测试里替换掉 `cli.console.print` 时，所有命令处理函数的输出都会被
"截获"，测试才好断言。

BridgeConsole：带「跨线程输入桥」的 Console。新界面里 stdin 被
prompt_toolkit 独占（raw 模式读按键），工作线程（agent 回合、子代理）
里再调 console.input 会永远读不到——审批提问挂着没人能答。桥把这种
提问搬进 pt 的 run_in_terminal 通道：暂停界面渲染 → 把终端还给经典
输入 → 用户答完 → 收回界面（hermes 的 sudo/secret 同款机制）。
"""

import logging
import threading

from rich.console import Console

logger = logging.getLogger("cli")

# 输入桥：cli_layout 装配时注册（工作线程 → pt 终端通道）；None=直读
_input_bridge = None


def set_input_bridge(fn) -> None:
    """注册/注销输入桥（fn: Callable[[Callable], Any]——把函数搬进
    pt 管理的终端里执行并回传返回值）。"""
    global _input_bridge
    _input_bridge = fn


class BridgeConsole(Console):
    """共享 Console：input() 遇上「工作线程 + 桥已注册」时改道 pt 通道。

    print 不做任何改道——patch_stdout 兜底打印安全，彩色 ANSI 直写的
    乱码问题由 cli_layout 启动时开控制台 VT 解释解决（那才是根因）。
    桥本身出异常就退回直读：桥挂了说明界面出了大事，宁可这个提问
    失败，不能让整个输入层崩掉。
    """

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
