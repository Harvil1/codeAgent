# agent/win_job_object.py
"""Windows Job Object 进程管控沙箱（CCAR12，ctypes 零依赖）。

诚实定位：**进程管控**（子进程树不逃逸 + 句柄关闭即全树清理 + 可选
内存/进程数上限），**不是文件系统隔离**——文件防线仍靠项目既有的
safe_path / 白名单 / 黑名单层（设计决策，用户已 ack）。
"""
import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_IS_WIN = os.name == "nt"

if _IS_WIN:
    import ctypes.wintypes as wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BASIC_LIMIT(ctypes.Structure):
        """JOBOBJECT_BASIC_LIMIT_INFORMATION——字段顺序/类型是 Win32 ABI，
        不能动。Affinity 用指针尺寸类型（= ULONG_PTR，x64 上 8 字节）。"""

        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(wintypes.ULONG)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMIT(ctypes.Structure):
        """JOBOBJECT_EXTENDED_LIMIT_INFORMATION。"""

        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    # LimitFlags 常量（winnt.h）
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    _JobObjectExtendedLimitInformation = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # 关键：默认 restype=c_int 会截断 64 位句柄，必须显式声明
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _create_raw_job() -> Optional[int]:
    """CreateJobObjectW → 句柄 int。失败 None（fail-open）。"""
    if not _IS_WIN:
        return None
    try:
        h = _kernel32.CreateJobObjectW(None, None)
        return int(h) if h else None
    except Exception as e:
        logger.warning("CreateJobObjectW 失败: %s", e)
        return None


class WinJobObject:
    """Job Object 封装。句柄必须保活到子进程结束（早关触发全树清理）。"""

    def __init__(self, *, memory_limit_mb: Optional[int] = None):
        self._handle = _create_raw_job()
        self._closed = False
        if self._handle is None:
            raise RuntimeError("Job Object 创建失败（fail-open 由调用方降级）")
        self._configure(memory_limit_mb)

    def _configure(self, memory_limit_mb: Optional[int]) -> None:
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = 100  # 子进程上限
        if memory_limit_mb:
            info.BasicLimitInformation.LimitFlags |= (
                _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            )
            info.ProcessMemoryLimit = memory_limit_mb * 1024 * 1024
        ok = _kernel32.SetInformationJobObject(
            self._handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            logger.warning(
                "SetInformationJobObject 失败: %s", ctypes.get_last_error()
            )

    def assign_process(self, pid: int) -> bool:
        if self._closed or self._handle is None:
            return False
        try:
            h_proc = _kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
            )
            if not h_proc:
                return False
            ok = _kernel32.AssignProcessToJobObject(self._handle, h_proc)
            _kernel32.CloseHandle(h_proc)
            return bool(ok)
        except Exception as e:
            logger.warning("assign_process(%s) 失败: %s", pid, e)
            return False

    def kill(self) -> None:
        """TerminateJobObject：终止整个 job 内的进程树。"""
        if self._handle and not self._closed:
            _kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """关句柄 → KILL_ON_JOB_CLOSE 自动清理全部子进程。幂等。"""
        if self._handle and not self._closed:
            _kernel32.CloseHandle(self._handle)
            self._closed = True


def create_job_for_subprocess(popen) -> Optional[WinJobObject]:
    """给已启动的 Popen 挂 Job Object。任何失败返回 None（fail-open，
    调用方走原路径）。注意：返回的 job 实例必须保活到 popen.wait() 后。"""
    if not _IS_WIN or popen is None or popen.pid is None:
        return None
    try:
        job = WinJobObject()
        if job.assign_process(popen.pid):
            return job
        job.close()
        return None
    except Exception as e:
        logger.warning("create_job_for_subprocess fail-open: %s", e)
        return None
