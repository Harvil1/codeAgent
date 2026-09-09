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
import os
import sys
import threading
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

# 幂等标记：装配过就记住日志文件路径，重复调用直接返回
_configured_path: Optional[Path] = None


def _resolve_level(default: int = logging.INFO) -> int:
    """日志级别：环境变量 CODEAGENT_LOG_LEVEL 可覆盖（DEBUG=全量诊断）。

    排查疑难杂症时 `set CODEAGENT_LOG_LEVEL=DEBUG` 再启动，DEBUG 级
    全部落文件；平时 INFO（够用不吵）。
    """
    name = (os.environ.get("CODEAGENT_LOG_LEVEL") or "").strip().upper()
    if name:
        resolved = logging.getLevelName(name)
        if isinstance(resolved, int):
            return resolved
    return default


def _install_excepthooks() -> None:
    """把「没人接的异常」也收进日志文件。

    为什么必须：线程里未捕获的异常默认只打到 stderr——关掉终端就没了，
    排查时最想看的恰恰是这种炸栈（反思线程/spinner/工作线程崩了，
    主流程 fail-open 吞掉后你甚至不知道它崩过）。
    - threading.excepthook：管所有线程的未捕获异常
    - sys.excepthook：主管主线程兜底（正常路径都有 try/except，双保险）
    """

    def _thread_hook(args):
        logger = logging.getLogger("crash.thread")
        logger.critical(
            "线程未捕获异常 [%s] %s",
            args.thread.name if args.thread else "?",
            args.exc_value,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _main_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return  # Ctrl+C 走正常退出语义，不算崩
        logging.getLogger("crash.main").critical(
            "主线程未捕获异常 %s", exc_value,
            exc_info=(exc_type, exc_value, exc_tb),
        )

    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass
    try:
        sys.excepthook = _main_hook
    except Exception:
        pass


def setup_logging(logs_dir, *, level: Optional[int] = None) -> Path:
    """装配 root logger 的文件 + 控制台双 handler + 异常兜底钩子（幂等）。

    参数：
        logs_dir：日志目录（~/.codeAgent/logs；不存在会建）
        level：文件日志级别；None 时用 INFO，可被环境变量
            CODEAGENT_LOG_LEVEL 覆盖（排查时设 DEBUG）

    返回：日志文件路径（已装配过时返回上次那份，不重复挂 handler）。
    """
    global _configured_path
    if _configured_path is not None:
        return _configured_path

    if level is None:
        level = _resolve_level()
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
    _install_excepthooks()
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
