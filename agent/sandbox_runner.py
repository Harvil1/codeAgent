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


# ---------------------------------------------------------------------------
# Linux: Bubblewrap (bwrap)
# ---------------------------------------------------------------------------

# 系统目录只读 bind（保证命令能跑：bash / glibc / 配置文件等）
_BWRAP_RO_DIRS = [
    "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32",
    "/etc", "/dev", "/proc", "/sys",
    # /tmp 走 bwrap 内部 tmpfs（不 bind 宿主 /tmp，避免泄漏）
]


def _bwrap_wrap(
    command: str,
    *,
    cwd: str,
    writable_roots: List[str],
) -> List[str]:
    """构造 bwrap argv（Linux）。

    返回 argv 列表，调用方用 subprocess.run(argv, shell=False)。
    """
    argv: List[str] = ["bwrap", "--die-with-parent", "--new-session"]

    # 系统目录只读 bind（只 bind 存在的）
    for d in _BWRAP_RO_DIRS:
        p = Path(d)
        if p.exists():
            argv += ["--ro-bind", d, d]

    # /tmp 走沙箱内 tmpfs（不泄漏宿主 /tmp）
    argv += ["--tmpfs", "/tmp"]

    # 可写目录：cwd 必须在首位
    writable = [cwd] + [r for r in writable_roots if r and r != cwd]
    for r in writable:
        argv += ["--bind", r, r]

    # 末尾：bash -c command
    argv += ["--", "bash", "-c", command]
    return argv


# ---------------------------------------------------------------------------
# 公开入口：wrap_command
# ---------------------------------------------------------------------------

def wrap_command(
    command: str,
    *,
    cwd: str,
    writable_roots: List[str],
) -> List[str]:
    """把 shell 命令包装成沙箱 argv。

    调用约定：调用方（terminal_tool）负责收集 writable_roots，
    默认应含 cwd + ~/.OmniMate + config["security"]["sandbox_writable_roots"]。
    本函数不重复添加 cwd（只在 bwrap/seatbelt 内部把 cwd bind 进可写区）。

    返回 argv 列表，传给 subprocess.run(argv, shell=False)。
    平台不支持 / 依赖缺失时抛 SandboxUnavailableError。

    内部分支：
      - Linux:  _bwrap_wrap()
      - macOS:  _seatbelt_wrap()（Task 3 实现）
      - 其他:   抛 SandboxUnavailableError
    """
    if sys.platform == "linux":
        if not shutil.which("bwrap"):
            raise SandboxUnavailableError(
                "未安装 bwrap（Bubblewrap）。Debian/Ubuntu: sudo apt install bubblewrap"
            )
        return _bwrap_wrap(command, cwd=cwd, writable_roots=writable_roots)

    if sys.platform == "darwin":
        # macOS 分支在 Task 3 实现，先抛占位
        raise SandboxUnavailableError("macOS 沙箱尚未实现（Task 3 引入）")

    raise SandboxUnavailableError(
        f"不支持的平台: {sys.platform}（仅支持 Linux + macOS）"
    )
