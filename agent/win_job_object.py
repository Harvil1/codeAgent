# agent/win_job_object.py
"""Windows 的 Job Object 进程笼子：用系统自带机制管住子进程树。

Job Object 是 Windows 内核提供的"进程编组"机制——把一堆进程关进同一个
"笼子"（job）里，就能对整笼统一管理：关笼门（关句柄）时整笼进程全部结束，
还能限制笼内的进程数量和内存用量。这里用 ctypes（Python 自带的调系统
DLL 工具）直接调 kernel32 里的相关函数，不装任何第三方包。

诚实的定位（设计决策，用户知情认可）：它管的是**进程**，不是**文件**——
能保证子进程及其后代一个都跑不掉、结束时一锅端、不撑爆内存；但它不隔离
文件系统（笼里的进程照样能碰文件）。文件这一层的防线仍然是项目原有的
safe_path 路径白名单和黑名单，两层叠加各管各的。

在项目里的位置：被 agent/sandbox_runner.py 调用；后者再供 terminal 工具
和 hook 执行使用。
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
        """Job Object 的基础限制信息结构（对应系统结构体
        JOBOBJECT_BASIC_LIMIT_INFORMATION）。字段的名字、顺序、类型必须和
        Windows 系统规定的内存布局一字不差（ABI 契约），改一个就全乱套。
        Affinity 字段用指针尺寸类型是为了在 64 位系统上占满 8 字节
        （等价于系统里的 ULONG_PTR）。"""

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
        """Job Object 的完整限制信息结构（基础信息 + IO 统计 + 内存上限），
        对应系统结构体 JOBOBJECT_EXTENDED_LIMIT_INFORMATION。"""

        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    # LimitFlags 用到的开关位（系统头文件 winnt.h 里定义的魔法数字，
    # 按位或组合起来决定 job 开哪些限制）
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    _JobObjectExtendedLimitInformation = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # 历史踩坑：不显式声明返回值类型时，ctypes 默认按 32 位
    # 整数处理，会把 64 位的句柄截断——截断的句柄是无效句柄，后续调用全失败。
    # 所以每个函数的返回类型/参数类型都必须一一写明。
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
    """调系统函数 CreateJobObjectW 建一个空 job，返回它的句柄（整数编号）。

    参数：无。
    返回：句柄整数；不在 Windows 上或创建失败返回 None（不抛异常，
    让调用方自行降级——fail-open）。
    """
    if not _IS_WIN:
        return None
    try:
        h = _kernel32.CreateJobObjectW(None, None)
        return int(h) if h else None
    except Exception as e:
        logger.warning("CreateJobObjectW 失败: %s", e)
        return None


class WinJobObject:
    """把"创建 job → 配限制 → 关进程进笼 → 收尾"这套流程包成对象。

    使用铁律：job 对象必须活到子进程全部跑完之后再 close()。因为本类
    开了"关句柄 = 杀整笼"的开关，提前 close 会把还在干活的子进程连
    同后代全部误杀。

    参数（构造时）：
        memory_limit_mb: 每个进程的内存上限（兆字节）；None = 不限内存。
    """

    def __init__(self, *, memory_limit_mb: Optional[int] = None):
        self._handle = _create_raw_job()
        self._closed = False
        if self._handle is None:
            raise RuntimeError("Job Object 创建失败（fail-open 由调用方降级）")
        self._configure(memory_limit_mb)

    def _configure(self, memory_limit_mb: Optional[int]) -> None:
        """给 job 设置限制规则：关笼杀全笼、崩了别拖累、进程数上限、内存上限。

        参数：
            memory_limit_mb: 单进程内存上限（兆字节）；None = 不设内存限制。
        """
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = 100  # 笼内最多 100 个进程，防 fork 炸弹式膨胀
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
        """把一个进程（按进程号）关进这个 job 笼子。

        参数：
            pid: 要关进笼子的进程号
        返回：True = 关进去了；False = 已经 close 过 / 系统拒绝 / 出错
            （不抛异常，让调用方降级）。
        """
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
        """主动杀掉笼子里的整棵进程树（调系统函数 TerminateJobObject）。"""
        if self._handle and not self._closed:
            _kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """关掉 job 句柄。因为建笼时开了"关句柄即杀全笼"的开关，这一下
        会让笼内所有还活着的进程被系统自动清掉。重复调也安全（幂等）。"""
        if self._handle and not self._closed:
            _kernel32.CloseHandle(self._handle)
            self._closed = True


def create_job_for_subprocess(popen) -> Optional[WinJobObject]:
    """给一个已经启动的子进程（subprocess.Popen 对象）套上 job 笼子。

    这是外部最常用的入口。任何一步失败都返回 None（不抛异常），调用方
    拿到 None 就走原本的执行路径，沙箱缺失不阻断任务——fail-open。

    参数：
        popen: 已启动的 subprocess.Popen 对象
    返回：套好笼子的 WinJobObject 实例；失败/非 Windows 返回 None。
    注意：返回的 job 对象必须一直持有，等 popen.wait()（进程真正结束）
    之后才能 close()——提前丢掉/close 会触发"杀全笼"误杀还在跑的子进程。
    """
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
