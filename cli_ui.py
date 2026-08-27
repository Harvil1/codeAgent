"""CLI 的共享输出工具。

为什么单独一个文件：Rich 的 Console（终端美化输出的对象）必须全局只有
一份。cli.py 和各个 cli_*_cmds 命令模块都从这儿 import 同一个 console，
这样测试里替换掉 `cli.console.print` 时，所有命令处理函数的输出都会被
"截获"，测试才好断言。
"""

import logging

from rich.console import Console

console = Console()
logger = logging.getLogger("cli")
