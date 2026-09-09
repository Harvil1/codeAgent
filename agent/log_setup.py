"""运行日志装配——logs/codeagent.log 滚动文件 + 控制台 WARNING 保底。

为什么需要这个模块：项目里到处都是 logger.info/warning，但从没有装配
过任何 handler——所有输出走 root logger 的 lastResort 兜底（只打
WARNING+ 到 stderr、INFO 全丢、重启即焚）。结果就是 logs/ 目录建了
一直空着，出了事（比如「反思 LLM 调用失败: Connection error」）只能
从屏幕回显瞄一眼，事后无从查证。

装配两件套（幂等，重复调用原样返回）：
1. RotatingFileHandler → logs/codeagent.log：INFO 级全量、5MB×3 轮换
   （够回溯几轮长会话，又不至于把磁盘吃穿）；
2. StreamHandler(stderr)：WARNING+ 照旧上屏——装了文件 handler 后
   lastResort 自动退位，不加这个控制台就再也看不见报错了。
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

# 幂等标记：装配过就记住日志文件路径，重复调用直接返回
_configured_path: Optional[Path] = None


def setup_logging(logs_dir, *, level: int = logging.INFO) -> Path:
    """装配 root logger 的文件 + 控制台双 handler（幂等）。

    参数：
        logs_dir：日志目录（~/.codeAgent/logs；不存在会建）
        level：文件日志级别（默认 INFO）

    返回：日志文件路径（已装配过时返回上次那份，不重复挂 handler）。
    """
    global _configured_path
    if _configured_path is not None:
        return _configured_path

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "codeagent.log"

    fmt = logging.Formatter(_FORMAT)
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    _configured_path = path
    return path


def reset_logging() -> None:
    """拆掉装配（测试隔离专用：把 root 还原成裸奔状态）。

    运行时没人调；verify 的日志检查跑完用它收摊，免得后续检查的
    日志全写进马上要删的临时目录。
    """
    global _configured_path
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
    _configured_path = None
