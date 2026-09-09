"""终端工具：替 AI 执行 shell 命令（比如跑脚本、装依赖包、看目录）。

这是项目里副作用最大的工具——它能真删文件、真改系统，所以执行前必须过
三道安全闸门（详见 agent/permission.py：黑名单硬拒 → 只读快速通道 →
拿不准就问用户）。输出超过 5 万字符会被截断，防止一次性把对话上下文撑爆。

在项目里的位置：属于 core 工具集（LLM 直接可见的基础工具之一），注册进
tools/registry.py 的中央工具表，由 model_tools.py 的分发逻辑调到这里。
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from agent.output_offload import finalize_tool_output as _finalize_output
from agent.permission import get_default_checker
from tools._common import get_mode_override_from_kwargs
from tools.registry import registry

logger = logging.getLogger(__name__)


# 输出超过这个字符数就截断——不然一条命令的海量输出会撑爆对话上下文（context）
MAX_OUTPUT_CHARS = 50000

# 超时上限。模型有时会传 1e9 这种天文数字当超时，
# 会把整个会话挂死，所以钳到 600 秒封顶；想调可改 config 的
# security.max_terminal_timeout（单条命令超时上限，毫秒）
MAX_TIMEOUT_SECONDS = 600
# 拦截"光睡大觉"的命令。裸 sleep 就是干等，
# 白白占满超时窗口还什么都没干，遇到该引导模型改用后台任务工具 bg_task
_BARE_SLEEP_RE = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

# 这些是带窗口的程序（浏览器、记事本等）。它们启动后不退出，
# 如果给它们接管道（读输出的管子），程序会攥着管子不放，整条命令永远等下去——
# 所以命中后干脆不建管道
_GUI_PROCESS_KEYWORDS = frozenset((
    "chrome.exe", "firefox.exe", "msedge.exe",
    "notepad.exe", "explorer.exe",
))
_GUI_LAUNCH_RE = re.compile(r"^\s*start\s", re.IGNORECASE)


_GUI_SEG_SPLIT_RE = re.compile(r"&&|\|\||;|\|")


def _is_gui_launch(command: str) -> bool:
    """判断一条命令是不是在启动带窗口的 GUI 程序（如 chrome.exe）。

    不能按"关键字在命令里出现就算"来判断——`echo "chrome.exe"` 这种只是
    想把名字打印出来的命令会被误判成启动浏览器，输出被全部丢弃，模型拿到
    假结果。所以只看每个分段（用 && | ; 切开）的第一个词（剥掉引号和标点、
    只取文件名部分）是不是 GUI 程序名——程序名出现在参数位置的一律不算。

    参数：
        command：完整的 shell 命令字符串。

    返回：True 表示这是在启动 GUI 程序（需要特殊处理），False 表示不是。
    """
    if _GUI_LAUNCH_RE.match(command):
        return True
    for seg in _GUI_SEG_SPLIT_RE.split(command):
        toks = seg.strip().split()
        if not toks:
            continue
        first = toks[0].strip("\"'").rstrip(",;.").lower()
        base = first.replace("\\", "/").split("/")[-1]
        if base in _GUI_PROCESS_KEYWORDS:
            return True
    return False


def check_terminal_requirements() -> bool:
    """检查终端工具能不能用。简化版：永远返回可用（True）。

    其他工具（如 LSP）依赖外部程序，存在"装没装"的问题，需要这种开关函数；
    终端工具在任何系统上都能跑，直接返回 True。

    返回：True（总是可用）。
    """
    return True


def _truncate_output(text: str) -> str:
    """把超长输出"掐头去尾留中间空"——保留开头一半和结尾一半，中间换成提示。

    一条命令可能输出几十万字符，全塞给模型会撑爆上下文。开头和结尾通常
    最有用（开头是命令回显，结尾是错误信息），所以两头都留、中间砍掉，
    并提示模型可以用 head/tail 自己分段看。

    参数：
        text：命令的原始输出。

    返回：截断后的文本（没超长就原样返回）。
    """
    if not text or len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return (
        text[:half]
        + f"\n\n... [输出已截断：总 {len(text)} 字符，"
        f"已显示前后各 {half} 字符。如需完整输出请用 head/tail/split] ...\n\n"
        + text[-half:]
    )


TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": (
        "执行 shell 命令并返回输出。用于运行脚本、安装包、"
        "操作文件系统、管理进程等。命令在指定工作目录执行。"
        "注意 shell 环境：Windows 上走 cmd.exe（PATH 里有 git-bash 的 "
        "Unix 工具，ls/cat/head/grep/find/wc 可用）；PowerShell 动词"
        "（Get-Content/Get-ChildItem 等）不可用——读文件用 cat，"
        "数行数用 cat x | wc -l，别试 PowerShell 写法。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数（默认 120）",
                "default": 120,
            },
            "cwd": {
                "type": "string",
                "description": "工作目录（默认当前目录）",
            },
        },
        "required": ["command"],
    },
}


def _load_session_env_overrides() -> dict:
    """从环境变量 $CODEAGENT_ENV_FILE 指向的文件里读出"export 键=值"形式的行，打包成字典返回。

    会话启动钩子（SessionStart hook）设置的环境变量写进这个文件"存档"。
    终端工具每次执行命令前把这份存档合并进子进程的环境变量，让钩子里配好的
    东西（比如 nvm/pyenv/conda 这类工具的路径设置）在后面每条命令里都生效。

    参数：无（文件路径从进程环境变量里自己找）。

    返回：{变量名: 值} 字典；文件不存在或读不了就返回空字典（不报错）。
    """
    env_path = os.environ.get("CODEAGENT_ENV_FILE")
    if not env_path:
        return {}
    try:
        p = Path(env_path)
        if not p.is_file():
            return {}
        overrides = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 两种写法都认：带 export 前缀的和不带的
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            # 去掉包围引号
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            if k:
                overrides[k] = v
        return overrides
    except Exception:
        return {}


def _handle_terminal(args: dict, **kwargs) -> str:
    """真正执行 shell 命令的主函数（工具的核心逻辑）。

    干什么：拿到模型要跑的命令，一路做完安检、套上沙箱（如果开了）、
    真正跑起来、把输出安全地交回去。

    整体流程（每一步都是为了不出事）：
      1. 先过安检：工作目录要在白名单里、命令要过三道权限闸门
      2. 按需套 OS 沙箱（操作系统级的"隔离笼子"）
      3. 用 subprocess 真正执行
      4. 输出太长就先无损落盘（存到文件里给个指针），实在不行再截断

    参数：
        args：工具参数字典，来自模型——command（命令本身）、
            timeout（超时秒数）、cwd（在哪个目录跑）。
        **kwargs：运行环境上下文，由分发器注入——permission_checker
            （权限检查器）、config（配置）、tool_call_id（本次调用编号，
            落盘时用来命名）、codeagent_home（数据目录）、sandbox_mode
            （沙箱开关）等。

    返回：JSON 字符串（项目铁律：所有工具都返回 JSON；出错时是
        {"error": ..., "error_type": ...} 格式）。
    """
    command = (args.get("command") or "").strip()
    if not command:
        return json.dumps({"error": "command 不能为空"}, ensure_ascii=False)

    # 模型经常把超时传成字符串 '10'，subprocess 只认数字，不转就会被忽略
    try:
        timeout = float(args.get("timeout", 120) or 120)
    except (TypeError, ValueError):
        timeout = 120.0

    # 裸 sleep 2 秒以上就是干等，占满超时窗口还没产出，拦下来引导改用后台任务
    _sleep_m = _BARE_SLEEP_RE.match(command)
    if _sleep_m and float(_sleep_m.group(1)) >= 2:
        return json.dumps({
            "error": (
                f"裸 sleep {_sleep_m.group(1)}s 会占满超时窗口且无产出。"
                f"等待/轮询类需求请用 bg_task 后台运行；短暂停顿用 sleep <2s。"
            ),
            "error_type": "bare_sleep",
            "command": command,
        }, ensure_ascii=False)

    # 超时封顶——防止模型传 1e9 这种数字把会话挂死
    _cfg = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
    _max_timeout = float(
        (_cfg.get("security") or {}).get("max_terminal_timeout")
        or MAX_TIMEOUT_SECONDS
    )
    timeout_clamped = False
    if timeout > _max_timeout:
        timeout = _max_timeout
        timeout_clamped = True
    cwd = args.get("cwd")
    if not cwd:
        # 并发跑的子代理（主对话派出去的分身）各有各的工作目录：优先读线程隔离的
        # ContextVar（每个线程各存各的，不会串），拿不到再退回 os.getcwd()
        from agent.workspace_context import get_workspace_cwd
        cwd = get_workspace_cwd()

    # 工作目录本身也必须过 safe_path 安检——
    # 漏检的话模型可以把工作目录设到 ~/.ssh 再跑 cat * 偷读私钥
    from agent.permission import safe_path
    cwd_perm = safe_path(cwd, write=False)
    if not cwd_perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: cwd {cwd_perm.reason}",
            "error_type": "permission_denied",
            "gate": cwd_perm.gate,
            "cwd": cwd,
        }, ensure_ascii=False)

    # 权限三道闸门检查。优先用外面注入进来的检查器（cli.py 注入的那个带
    # "问用户"的回调），没有就用全局默认的（没回调，只能自动裁决）。
    # 子代理的权限模式必须透传——从 agent_ref 里取出
    # 子代理自己的权限模式覆盖值，让它按自己的模式做决策，而不是去改
    # 全局检查器（那样会污染别的线程，不安全）。
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = get_mode_override_from_kwargs(kwargs)
    perm = checker.check(command, cwd=cwd, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: {perm.reason}",
            "error_type": "permission_denied",
            "gate": perm.gate,
            "command": command,
        }, ensure_ascii=False)

    # === OS 沙箱（操作系统级隔离笼子）套壳 ===
    # 沙箱开着时：Linux/macOS 把命令包进 bwrap/seatbelt 的启动参数里；
    # Windows 走 Job Object 模式——命令原样跑，启动后把整个
    # 进程树挂进一个"作业对象"里管起来；沙箱工具不可用时放行降级（fail-open，
    # 警告一声但照跑，不能因为沙箱缺失把功能整个废掉）。
    # 沙箱开关优先从 kwargs 读（方便测试和子代理透传）；
    # 没传时从默认权限检查器上读（生产路径：用户敲 /sandbox on 就是设在它身上）
    sandbox_mode = kwargs.get("sandbox_mode")
    if sandbox_mode is None:
        try:
            sandbox_mode = checker.sandbox_mode
        except AttributeError:
            sandbox_mode = "off"
    wrapped_argv = None
    sandbox_active = False
    win_job_mode = False  # Windows Job Object 模式：命令不包装，启动后再挂进作业对象
    if sandbox_mode == "on":
        # GUI 程序在沙箱里根本起不来，只能跳过沙箱
        is_gui_launch = _is_gui_launch(command)  # 判定方式见 _is_gui_launch 的说明（首个词级）
        if is_gui_launch:
            logger.warning("GUI 命令跳过 OS 沙箱: %s", command[:80])
        else:
            try:
                from agent.sandbox_runner import (
                    wrap_command, is_available, availability_reason,
                    SandboxUnavailableError, uses_job_object,
                )
                if is_available():
                    if uses_job_object():
                        # Windows Job Object：命令不包装，
                        # 正常启动后把进程挂进作业对象里管控
                        sandbox_active = True
                        win_job_mode = True
                    else:
                        # 收集沙箱里允许写的目录：cwd + ~/.codeAgent + 配置里额外加的
                        writable_roots = []
                        try:
                            from constants import get_codeagent_home
                            writable_roots.append(str(get_codeagent_home()))
                        except Exception:
                            writable_roots.append(str(Path.home() / ".codeAgent"))
                        cfg = kwargs.get("config") or {}
                        extra = ((cfg.get("security") or {}).get("sandbox_writable_roots") or [])
                        writable_roots.extend(extra)

                        wrapped_argv = wrap_command(
                            command, cwd=cwd, writable_roots=writable_roots,
                        )
                        sandbox_active = True
                else:
                    logger.warning(
                        "OS 沙箱不可用（%s），fail-open 降级到原路径",
                        availability_reason(),
                    )
            except SandboxUnavailableError as e:
                logger.warning("OS 沙箱 wrapper 构造失败，fail-open: %s", e)
            except Exception as e:
                logger.warning("OS 沙箱 wrapper 异常，fail-open: %s", e)

    try:
        # 沙箱环境变量：把密钥类（API key、数据库密码等）洗掉，防止泄给子进程
        from agent.sandbox_env import build_safe_env
        safe_env = build_safe_env()
        # 合并 CODEAGENT_ENV_FILE 里钩子（hook，在特定时机自动执行的小脚本）写入的环境变量
        safe_env.update(_load_session_env_overrides())

        # 检测是不是在启动 GUI 程序（start / Chrome / 浏览器等）：
        # 这类程序不退出，subprocess 的输出管道会一直等它 → 整条命令卡死。
        # 解法：不给它建管道（输出直接丢进 DEVNULL 黑洞），start 命令立刻返回
        is_gui_launch = _is_gui_launch(command)  # 判定方式见 _is_gui_launch 的说明（首个词级）

        if is_gui_launch:
            # GUI 命令：不建管道，避免子进程攥着管道不放导致卡死
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
            )
            # 如实说明"没抓到输出"——假装有结果返回
            # "(GUI 程序已启动)"会误导模型以为那是命令的真实输出
            return json.dumps({
                "stdout": "(GUI 程序已启动；其 stdout/stderr 未捕获——不建管道防卡死，输出不可用)",
                "stderr": "",
                "exit_code": result.returncode,
                "command": command,
                "cwd": cwd,
            }, ensure_ascii=False)
        elif sandbox_active and win_job_mode:
            # Windows Job Object 模式：命令原样启动，启动后把整棵
            # 进程树挂进作业对象管控（挂失败也放行继续跑，fail-open）。
            # 公共流程抽到了 sandbox_runner.run_with_job_object（超时后先收拾
            # 残局再抛 TimeoutExpired，由本函数外层的 except 转成 error JSON）
            from agent.sandbox_runner import run_with_job_object
            result = run_with_job_object(
                command,
                shell=True,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
                errors="replace",
            )
        elif sandbox_active and wrapped_argv:
            # OS 沙箱路径：用包好的启动参数直接跑，不再经过 shell
            result = subprocess.run(
                wrapped_argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
                encoding="utf-8",
                errors="replace",
            )
        else:
            # 普通路径：沙箱没开，或者沙箱不可用降级了
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
                encoding="utf-8",
                errors="replace",
            )
        # 大输出先把完整原文存到磁盘（offload，回传预览
        # + 文件位置指针），实在存不了才截断。顺序不能反——先截断再落盘的话，
        # 磁盘上存的也是残缺版，中间那段永久丢了；stderr 也走同一条落盘路。
        tool_call_id = kwargs.get("tool_call_id")
        config = kwargs.get("config")
        codeagent_home = kwargs.get("codeagent_home")

        def _emit(raw: str, suffix: str = ""):
            """处理一份超长输出：优先把原文无损存盘（offload），存不了才截断兜底。

            参数：
                raw：原始输出文本（stdout 或 stderr）。
                suffix：拼在本次调用编号后面的后缀，用来区分 stdout/stderr
                    两个落盘文件名（如 "_stderr"）。

            返回：(处理后的文本, 是否走了落盘) 二元组。落盘成功时文本是
                "预览 + 文件在哪"的说明；失败时是掐头去尾的截断版。
            """
            if raw and len(raw) > MAX_OUTPUT_CHARS:
                tc_id = f"{tool_call_id}{suffix}" if tool_call_id else None
                final = _finalize_output(raw, tc_id, codeagent_home, config)
                if final != raw:
                    # 落盘成功（返回预览 + full_at 文件指针）或 IO 出错时的降级说明
                    return final, True
            return _truncate_output(raw), False

        final_stdout, stdout_offloaded = _emit(result.stdout)
        final_stderr, stderr_offloaded = _emit(result.stderr, "_stderr")

        return json.dumps({
            "stdout": final_stdout,
            "stderr": final_stderr,
            "exit_code": result.returncode,
            "command": command,
            "cwd": cwd,
            "stdout_truncated": len(result.stdout or "") > MAX_OUTPUT_CHARS,
            "stderr_truncated": len(result.stderr or "") > MAX_OUTPUT_CHARS,
            "stdout_offloaded": stdout_offloaded,
            "stderr_offloaded": stderr_offloaded,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        # 超时了就引导改用后台任务——terminal 超时会
        # 杀掉进程，长任务应该用 bg_task 后台跑（不堵对话，跑完还通知）
        return json.dumps({
            "error": (
                f"命令超时（{timeout}秒，进程已终止）。长时间任务请改用 bg_task "
                f"后台运行（不阻塞对话、完成时通知），不要在 terminal 里等"
            ),
            "command": command,
            "timeout": timeout,
            "timeout_clamped": timeout_clamped,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "command": command,
        }, ensure_ascii=False)


# 模块级注册：这个文件一被 import 就自动把工具登记进中央注册表
registry.register(
    name="terminal",
    toolset="core",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    isConcurrencySafe=False,  # 会执行 shell 命令（副作用最强的操作），绝不能并发，必须排队串行
)
