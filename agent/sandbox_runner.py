"""OS 沙箱 wrapper 构造器（跨平台）。

对齐 Claude Code `/sandbox` 行为：
  - Linux:  Bubblewrap（bwrap）内核命名空间隔离
  - macOS:  Seatbelt（sandbox-exec）profile 隔离
  - Windows / 其他: 不支持，调用方 fail-open 降级

公开 API：
  - SandboxUnavailableError：wrapper 构造失败的异常基类
  - is_available() -> bool：当前平台是否有可用沙箱
  - availability_reason() -> str：不可用原因（用于警告日志）
  - wrap_command(command, *, cwd, writable_roots) -> list[str]：包装 argv
"""
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class SandboxUnavailableError(RuntimeError):
    """沙箱不可用（平台未实现 / 二进制未装 / 配置错 / wrapper 构造失败）。"""


# ---------------------------------------------------------------------------
# 可用性检测（带缓存，60s TTL）
# ---------------------------------------------------------------------------

_AVAILABILITY_CACHE_TTL_S = 60.0
_availability_cache: Optional[Tuple[bool, str, float]] = None  # (ok, reason, ts)


def _detect_availability() -> Tuple[bool, str]:
    """真实检测一次（不读缓存）。返回 (ok, reason)。"""
    platform = sys.platform

    if platform == "linux":
        if shutil.which("bwrap"):
            return True, ""
        return False, "未安装 bwrap（Bubblewrap）。Debian/Ubuntu: sudo apt install bubblewrap"

    if platform == "darwin":
        if shutil.which("sandbox-exec"):
            return True, ""
        return False, "未找到 sandbox-exec（macOS 系统自带，正常不会缺）"

    if platform == "win32":
        return False, "Windows 不支持原生沙箱，请用 WSL2 或 Docker"

    return False, f"不支持的平台: {platform}"


def is_available() -> bool:
    """当前平台是否有可用沙箱（60s 缓存）。"""
    global _availability_cache
    now = time.monotonic()
    if _availability_cache is not None:
        ok, _, ts = _availability_cache
        if now - ts < _AVAILABILITY_CACHE_TTL_S:
            return ok
    ok, reason = _detect_availability()
    _availability_cache = (ok, reason, now)
    return ok


def availability_reason() -> str:
    """不可用原因（用于 fail-open 警告日志）。"""
    global _availability_cache
    now = time.monotonic()
    if _availability_cache is None or now - _availability_cache[2] >= _AVAILABILITY_CACHE_TTL_S:
        ok, reason = _detect_availability()
        _availability_cache = (ok, reason, now)
        if ok:
            return ""
        return reason
    if _availability_cache[0]:
        return ""
    return _availability_cache[1]


def reset_availability_cache() -> None:
    """清缓存（测试用）。"""
    global _availability_cache
    _availability_cache = None
