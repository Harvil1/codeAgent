"""CLI 共享 UI 基元（R30 瘦身抽离）。

console 必须全 CLI 唯一实例：cli.py 与各 cli_*_cmds 模块都从这里 import，
保证测试 monkeypatch `cli.console.print` 对所有命令处理函数全局生效。
"""

import logging

from rich.console import Console

console = Console()
logger = logging.getLogger("cli")
