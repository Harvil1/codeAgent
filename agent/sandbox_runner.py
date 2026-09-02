"""OS 沙箱的"包装器构造器"：把要在沙箱里跑的命令包装成各平台需要的形态。

沙箱（sandbox）就是给命令造一个受限的笼子——命令在里面跑，搞不坏系统
其他部分。这个文件负责跨平台统一入口：

  - Linux:  Bubblewrap（bwrap，系统级隔离工具）——用内核的命名空间机制
            把命令关进独立的"小房间"，只有指定目录可写
  - macOS:  Seatbelt（sandbox-exec，系统自带）——生成一份规则文件
            （profile），规则写明哪些文件操作放行、哪些拒绝
  - Windows: Job Object 进程笼子（实现见 agent/win_job_object.py）。
            注意这条路子完全不同：命令不做任何包装，照常启动（Popen）
            之后再把它套进 job——管的是"进程跑不跑得掉"，文件防线
            仍由 safe_path 白名单层负责
  - 其他平台: 不支持；调用方自行降级（fail-open：沙箱没有就照常跑，
            只警告不阻断）

给谁用：tools/terminal_tool.py（terminal 命令执行）和 agent/hook_exec.py
（hook 命令执行）在沙箱开启时调这里。

公开 API（一句话导览）：
  - SandboxUnavailableError：沙箱构造失败的异常类型
  - is_available() -> bool：当前平台有没有可用的沙箱
  - availability_reason() -> str：不可用的原因（写警告日志用）
  - wrap_command(command, *, cwd, writable_roots) -> list[str]：把命令包装成
    沙箱 argv（Linux/macOS 用；Windows 不走这条路）
  - uses_job_object() -> bool：当前平台是否走 Windows 的 Job Object 模式
  - attach_job(popen)：给已启动的进程套 job 笼子（失败返回 None 不阻断）
  - run_with_job_object(cmd, ...)：Job Object 模式的统一执行流程（启动进程
    →套笼子→等输出→关笼子；超时会先清干净再重新抛出 TimeoutExpired，
    具体怎么处理错误由调用方自己决定）
  - sandbox_description() -> str：当前平台沙箱机制的描述文案（/sandbox
    status 命令显示用）
"""
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class SandboxUnavailableError(RuntimeError):
    """沙箱用不了的统一异常：平台没实现 / 依赖工具没装 / 配置有错 / 包装失败。"""


# ---------------------------------------------------------------------------
# 可用性检测（结果缓存 60 秒，避免每条命令都探测一遍）
# ---------------------------------------------------------------------------

_AVAILABILITY_CACHE_TTL_S = 60.0
_availability_cache: Optional[Tuple[bool, str, float]] = None  # 缓存内容：(是否可用, 原因, 探测时间)


def _detect_availability() -> Tuple[bool, str]:
    """老老实实探测一次当前平台沙箱是否可用（不读缓存）。

    返回：(是否可用, 不可用原因)。可用时原因为空字符串。
    Linux 查 bwrap 是否装了，macOS 查 sandbox-exec 是否在，Windows 查
    win_job_object 模块能否导入，其他平台直接判不可用。
    """
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
        # Windows 走 Job Object 进程笼子：win_job_object 模块能导入就算可用
        # （它只依赖系统自带的 kernel32，不需要装额外工具）
        try:
            from agent.win_job_object import create_job_for_subprocess  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"win_job_object 模块不可用（fail-open）: {e}"

    return False, f"不支持的平台: {platform}"


def is_available() -> bool:
    """当前平台有没有可用沙箱。结果缓存 60 秒（探测要查 PATH，没必要每条命令查一遍）。"""
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
    """沙箱不可用时给出人话原因（比如缺什么工具、怎么装），写警告日志用。可用则返回空字符串。"""
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
    """清掉可用性缓存。测试用（测试里模拟装/卸工具后需要重新探测）。"""
    global _availability_cache
    _availability_cache = None


# ---------------------------------------------------------------------------
# Windows: Job Object 进程笼子
# ---------------------------------------------------------------------------

def uses_job_object() -> bool:
    """当前平台沙箱是不是走 Windows 的 Job Object 路线。

    bwrap/Seatbelt 是"先包装命令再启动"，Job Object 是"先正常启动再套
    笼子"——调用方按这个判断分流。
    """
    return sys.platform == "win32"


def attach_job(popen):
    """给一个已经启动的子进程（Popen 对象）套上 Job Object 笼子。

    内部转发给 win_job_object 模块；任何失败只记警告并返回 None，不阻断
    命令执行（fail-open：没有笼子也比任务跑不了强）。

    参数：
        popen: 已启动的 subprocess.Popen 对象

    返回：job 笼子实例。调用方必须把它拿在手里，等 popen.wait()（进程
    真正结束）之后才能 close()——因为笼子配了"关笼门即杀全笼"的开关，
    提前 close 会把还在跑的子进程树误杀。失败返回 None。
    """
    try:
        from agent.win_job_object import create_job_for_subprocess
        return create_job_for_subprocess(popen)
    except Exception as e:
        logger.warning("attach_job fail-open: %s", e)
        return None


def run_with_job_object(
    cmd,
    *,
    shell: bool = False,
    timeout=None,
    env=None,
    cwd=None,
    input=None,
    errors=None,
) -> subprocess.CompletedProcess:
    """Windows Job Object 模式下跑一条命令的统一入口（terminal 工具和 hook 执行共用）。流程：启动进程 → 套笼子 → 收输出 → 关笼子。

    参数：
        cmd: 要跑的命令。terminal 传的是字符串并配 shell=True；
            hook 传的是参数列表（argv）
        shell: 是否经 shell 解释（原样传给 Popen）
        timeout: 等命令跑完的最长秒数；None = 一直等
        env: 子进程的环境变量字典；None = 继承父进程的
        cwd: 子进程的工作目录；None = 不指定（用当前目录）
        input: 要喂给命令 stdin 的内容；None = 不开 stdin 管道
        errors: 输出解码出错时的处理策略（如 "replace"）；None = 用
            Popen 默认的严格模式

    返回：CompletedProcess（含返回码、stdout、stderr）。

    几个关键细节（别改）：
      - 命令不做任何包装，照常 Popen 启动（输出走管道、按文本模式、
        强制 utf-8 解码）
      - 套笼子失败（返回 None）只警告不阻断——没笼子也继续跑
      - 笼子必须等进程结束后再关：早关会触发"杀全笼"，把还在跑的
        子进程树误杀
      - 超时的处理分两种情况：有笼子时，finally 里的 job.close() 自带
        "关笼杀全笼"，超时进程会被清掉，不用重复杀；没笼子时（attach
        失败降级），超时进程会变孤儿继续赖着跑——必须手动补杀一刀，
        再 communicate 收尸，防止进程和管道僵死（与标准库
        subprocess.run 的内部做法一致）。两种情况最后都会把
        TimeoutExpired 异常重新抛出去——至于超时算错误还是算别的，
        由各调用方自己决定（terminal 转成 error JSON，hook 返回 None）。
    """
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "shell": shell,
        "env": env,
    }
    if input is not None:
        popen_kwargs["stdin"] = subprocess.PIPE
    if cwd is not None:
        popen_kwargs["cwd"] = cwd
    if errors is not None:
        popen_kwargs["errors"] = errors
    proc = subprocess.Popen(cmd, **popen_kwargs)

    job = attach_job(proc)
    if job is None:
        logger.warning(
            "Windows Job Object attach 失败，fail-open 继续执行: %s",
            str(cmd)[:80],
        )
    try:
        if input is not None:
            stdout, stderr = proc.communicate(input=input, timeout=timeout)
        else:
            stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if job is None:
            try:
                proc.kill()
            except OSError:
                pass  # 可能进程恰好自己退了（竞态）：杀失败没关系，别让它盖住"超时"这个正主
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass  # 收尸失败也不影响把超时错误如实上报
        raise
    finally:
        if job is not None:
            job.close()
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr,
    )


def sandbox_description() -> str:
    """用一句话说清当前平台的沙箱是怎么运作的（/sandbox status 显示用）。"""
    if sys.platform == "win32":
        return "Job Object 模式（进程管控；文件防线=safe_path 白名单层）"
    if sys.platform == "linux":
        return "bwrap（Bubblewrap）内核命名空间隔离"
    if sys.platform == "darwin":
        return "sandbox-exec（Seatbelt）profile 隔离"
    return f"不支持的平台: {sys.platform}"


# ---------------------------------------------------------------------------
# Linux: Bubblewrap (bwrap)
# ---------------------------------------------------------------------------

# 系统目录清单：这些目录以"只读挂载"（bind）方式放进沙箱——只给看不给改。
# 不挂这些命令根本跑不起来（bash、系统库、配置文件都在里面）。
_BWRAP_RO_DIRS = [
    "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32",
    "/etc", "/dev", "/proc", "/sys",
    # /tmp 特殊处理：不挂宿主机的 /tmp（那里面可能有敏感东西），改用沙箱
    # 内部自带的临时内存盘（tmpfs）
]


def _bwrap_wrap(
    command: str,
    *,
    cwd: str,
    writable_roots: List[str],
) -> List[str]:
    """把 shell 命令包装成 bwrap 启动参数（Linux 用）。

    返回：参数列表（argv），调用方拿它 subprocess.run(argv, shell=False) 启动，
    命令就跑在 bwrap 笼子里了。

    参数：
        command: 原始 shell 命令字符串
        cwd: 工作目录（会被设为沙箱内第一个可写目录）
        writable_roots: 允许写入的其他目录列表
    """
    # --die-with-parent：父进程（我们）死了，沙箱里的命令陪葬；
    # --new-session：新会话，脱离当前终端的控制（防终端信号干扰）
    argv: List[str] = ["bwrap", "--die-with-parent", "--new-session"]

    # 系统目录只读挂载（只挂真实存在的，不存在会报错）
    for d in _BWRAP_RO_DIRS:
        p = Path(d)
        if p.exists():
            argv += ["--ro-bind", d, d]

    # /tmp 用沙箱内部的临时内存盘，和宿主机的 /tmp 隔开
    # 注意：当 cwd 本身就在 /tmp 下时跳过这一步——
    # 下面又要往沙箱里挂真正的 /tmp/cwd，和这里的临时盘会打架
    _cwd_str = str(cwd)
    if not (_cwd_str == "/tmp" or _cwd_str.startswith("/tmp/")):
        argv += ["--tmpfs", "/tmp"]

    # 可写目录挂载：cwd 必须排第一个（命令在里面干活）
    writable = [cwd] + [r for r in writable_roots if r and r != cwd]
    for r in writable:
        argv += ["--bind", r, r]

    # 收尾：真正的命令用 bash -c 执行
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
    """把一条 shell 命令包装成"在沙箱里跑"的启动参数（Linux/macOS 用）。

    调用约定：writable_roots 由调用方（terminal 工具）负责收集，默认应包含
    ~/.codeAgent 和配置里的 security.sandbox_writable_roots。cwd 不用传进
    writable_roots——本函数内部会自动把 cwd 放到可写区第一位。

    参数：
        command: 原始 shell 命令字符串
        cwd: 命令的工作目录（自动成为第一个可写目录）
        writable_roots: 额外允许写入的目录列表

    返回：参数列表（argv），传给 subprocess.run(argv, shell=False) 即在沙箱中执行。
    平台不支持或依赖工具缺失时抛 SandboxUnavailableError（调用方降级处理）。

    内部按平台分流：Linux 走 _bwrap_wrap()，macOS 走 _seatbelt_wrap()，
    其他平台抛异常。
    """
    if sys.platform == "linux":
        if not shutil.which("bwrap"):
            raise SandboxUnavailableError(
                "未安装 bwrap（Bubblewrap）。Debian/Ubuntu: sudo apt install bubblewrap"
            )
        return _bwrap_wrap(command, cwd=cwd, writable_roots=writable_roots)

    if sys.platform == "darwin":
        if not shutil.which("sandbox-exec"):
            raise SandboxUnavailableError(
                "未找到 sandbox-exec（macOS 系统自带，正常不会缺）"
            )
        return _seatbelt_wrap(command, cwd=cwd, writable_roots=writable_roots)

    raise SandboxUnavailableError(
        f"不支持的平台: {sys.platform}（仅支持 Linux + macOS）"
    )


# ---------------------------------------------------------------------------
# macOS: Seatbelt (sandbox-exec)
# ---------------------------------------------------------------------------

# 下面是 Seatbelt 规则文件的辅助函数
def _seatbelt_escape_path(p: str) -> str:
    """把路径里的特殊字符转义掉再插进 Seatbelt 规则文件（Scheme 语法）——路径里带反斜杠或双引号会破坏规则文件语法、甚至注入额外的放行规则（防注入）。

    参数：
        p: 要插进规则文件的路径
    返回：转义后的安全路径。
    """
    return p.replace("\\", "\\\\").replace('"', '\\"')


_SEATBELT_PROFILE_TEMPLATE = """\
(version 1)
(deny default)

;; 基础系统调用放行
(allow process-fork)
(allow process-exec)
(allow signal*)
(allow sysctl*)
(allow process-info* (target self))
(allow mach-lookup*)
(allow ipc-posix*)

;; 文件读全放开（第一版只防写）
(allow file-read*)

;; 文件写：默认拒，仅放行 cwd + writable_roots
(deny file-write*)
{write_rules}

;; 网络全放开（用户决策：不做网络隔离）
(allow network*)
(allow network-outbound*)
(allow network-inbound*)
"""


def _write_seatbelt_profile(
    *,
    cwd: str,
    writable_roots: List[str],
) -> Path:
    """把规则内容写成一份 .sb 规则文件，存到 ~/.codeAgent/.sandbox/ 下（sandbox-exec 按文件执行，每次跑命令现写一份；文件名带随机编号，并发互不覆盖）。

    参数：
        cwd: 工作目录（会成为第一个允许写入的目录）
        writable_roots: 额外允许写入的目录列表

    返回：刚写好的规则文件路径。
    """
    import uuid
    try:
        from constants import get_codeagent_home
    except ImportError:
        # 测试环境兜底：常规模块导入不了时，直接用家目录下的 .codeAgent
        get_codeagent_home = lambda: Path.home() / ".codeAgent"  # noqa: E731

    sandbox_dir = get_codeagent_home() / ".sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # 写新规则文件前顺手清掉 7 天前的老文件——
    # 每条命令都生成一份 .sb，不清就会无限堆积
    try:
        _cleanup_old_seatbelt_profiles(sandbox_dir, max_age_days=7)
    except Exception as e:
        logger.warning("清理旧 seatbelt profile 失败（忽略）: %s", e)

    profile_path = sandbox_dir / f"seatbelt-{uuid.uuid4().hex[:8]}.sb"

    # 拼装"允许写入"规则：cwd 排第一，去重后逐个生成放行条目
    all_writable = [cwd] + [r for r in writable_roots if r and r != cwd]
    rules = []
    for root in all_writable:
        # subpath 语义：这个目录本身和它下面的所有内容都允许写
        rules.append(f'(allow file-write* (subpath "{_seatbelt_escape_path(root)}"))')
    write_rules_block = "\n".join(rules) if rules else ";; (无额外可写路径)"

    content = _SEATBELT_PROFILE_TEMPLATE.format(write_rules=write_rules_block)
    profile_path.write_text(content, encoding="utf-8")
    return profile_path


def _cleanup_old_seatbelt_profiles(sandbox_dir: Path, *, max_age_days: int = 7) -> int:
    """清掉超过 max_age_days 天没动过的老 .sb 规则文件（防无限堆积）。

    参数：
        sandbox_dir: 规则文件所在目录
        max_age_days: 文件保留天数上限，超过就删
    返回：实际删掉的文件数。某个文件删除失败只记日志，不影响其他文件继续清。
    """
    import time
    if not sandbox_dir.exists():
        return 0
    threshold = time.time() - max_age_days * 86400
    deleted = 0
    for p in sandbox_dir.glob("seatbelt-*.sb"):
        try:
            if p.stat().st_mtime < threshold:
                p.unlink(missing_ok=True)
                deleted += 1
        except Exception as e:
            logger.debug("清理 profile %s 失败（忽略）: %s", p, e)
    if deleted:
        logger.info("清理了 %d 个过期 seatbelt profile（>=%d 天）", deleted, max_age_days)
    return deleted


def _seatbelt_wrap(
    command: str,
    *,
    cwd: str,
    writable_roots: List[str],
) -> List[str]:
    """把命令包装成 sandbox-exec 启动参数（macOS 用）。

    参数：
        command: 原始 shell 命令字符串
        cwd: 工作目录（写入规则文件时排第一的可写目录）
        writable_roots: 额外允许写入的目录列表

    返回：参数列表，形如
    ["sandbox-exec", "-p", 规则文件路径, "bash", "-c", 命令]——
    即"用这份规则文件来跑这条命令"。
    """
    profile_path = _write_seatbelt_profile(cwd=cwd, writable_roots=writable_roots)
    return [
        "sandbox-exec",
        "-p", str(profile_path),
        "bash", "-c", command,
    ]
