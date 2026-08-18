"""终端工具：执行 shell 命令。

集成权限检查（三道闸门）和输出截断（50000 字符）。
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


# 输出截断阈值（防止爆 context）
MAX_OUTPUT_CHARS = 50000

# GUI 程序关键字（命中后 subprocess 不建管道，避免子进程继承管道卡死）
_GUI_PROCESS_KEYWORDS = frozenset((
    "chrome.exe", "firefox.exe", "msedge.exe",
    "notepad.exe", "explorer.exe",
))
_GUI_LAUNCH_RE = re.compile(r"^\s*start\s", re.IGNORECASE)


_GUI_SEG_SPLIT_RE = re.compile(r"&&|\|\||;|\|")


def _is_gui_launch(command: str) -> bool:
    """GUI 启动判定（R30c-B5：每段首 token 级，不再子串匹配）。

    此前 `kw in command` 子串匹配，`echo "chrome.exe"` / `grep chrome.exe log`
    会被误判 GUI → DEVNULL 吞掉全部输出，模型拿到假结果。现在只认每个
    复合段的**首 token**（剥包裹引号/尾标点，取路径 basename）是 GUI
    可执行名——GUI 程序只能作为段的动词启动，出现在参数位的一律不算。
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
    """检查终端工具是否可用。简化版：总是可用。"""
    return True


def _truncate_output(text: str) -> str:
    """截断超长输出，保留前后各一半，加续写提示。"""
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
    """从 $OMNIMATE_ENV_FILE 读 export K=V 行返回 dict（对齐 Claude Code CLAUDE_ENV_FILE）。

    SessionStart hook 写入的 env 持久化到这个文件，terminal_tool 每次执行
    命令前 merge 到 subprocess env，让 hook 设的环境变量对后续命令可见
    （nvm/pyenv/conda 用户刚需）。
    """
    env_path = os.environ.get("OMNIMATE_ENV_FILE")
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
            # 支持 `export K=V` 和 `K=V`
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
    """实际执行终端命令。

    流程：
      1. 权限检查（三道闸门）
      2. subprocess 执行
      3. 输出截断
    """
    command = (args.get("command") or "").strip()
    if not command:
        return json.dumps({"error": "command 不能为空"}, ensure_ascii=False)

    # timeout 强制转 float(LLM 常传字符串 '10' 导致 subprocess 忽略)
    try:
        timeout = float(args.get("timeout", 120) or 120)
    except (TypeError, ValueError):
        timeout = 120.0
    cwd = args.get("cwd")
    if not cwd:
        # 并发子代理 workspace：优先读 ContextVar（线程隔离），fallback 到 os.getcwd()
        from agent.workspace_context import get_workspace_cwd
        cwd = get_workspace_cwd()

    # S4 fix: cwd 必须过 safe_path（之前可在 ~/.ssh 跑 cat * 读私钥）
    from agent.permission import safe_path
    cwd_perm = safe_path(cwd, write=False)
    if not cwd_perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: cwd {cwd_perm.reason}",
            "error_type": "permission_denied",
            "gate": cwd_perm.gate,
            "cwd": cwd,
        }, ensure_ascii=False)

    # 权限检查（闸门 1/2/3）
    # 优先用注入的 permission_checker（cli.py 注入带 callback 的），
    # 没有则用全局默认（无 callback）。
    # 必修 1：子代理 permission_mode 透传——从 agent_ref.permission_mode 提取 mode override，
    # 让子代理按自己的 mode 做权限决策（不污染全局 checker，线程安全）。
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

    # === OS 沙箱 wrapper 注入 ===
    # sandbox_mode="on" 时：Linux/macOS 把 command 包进 bwrap/seatbelt argv；
    # Windows（CCAR12）走 Job Object 模式——命令不包装，启动后挂 job；
    # 不可用 → fail-open 警告并降级到原 shell=True 路径
    # C1 fix: sandbox_mode 优先从 kwargs 读（测试/子代理透传）；
    # 缺省时从默认 PermissionChecker 读（生产路径：/sandbox on 设到 checker 上）
    sandbox_mode = kwargs.get("sandbox_mode")
    if sandbox_mode is None:
        try:
            sandbox_mode = checker.sandbox_mode
        except AttributeError:
            sandbox_mode = "off"
    wrapped_argv = None
    sandbox_active = False
    win_job_mode = False  # CCAR12: Windows Job Object 模式（命令不包装，启动后挂 job）
    if sandbox_mode == "on":
        # GUI 命令强制跳过 sandbox（GUI 程序在沙箱里启不来）
        is_gui_launch = _is_gui_launch(command)  # R30c-B5：token 级判定
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
                        # CCAR12: Windows Job Object——命令不包装，
                        # 正常 Popen 启动后 attach job（进程管控）
                        sandbox_active = True
                        win_job_mode = True
                    else:
                        # 收集 writable_roots：cwd + ~/.OmniMate + config 扩展
                        # I5 fix: 移除冗余 inline import（Path 已在模块顶部导入）
                        writable_roots = []
                        try:
                            from constants import get_omnimate_home
                            writable_roots.append(str(get_omnimate_home()))
                        except Exception:
                            writable_roots.append(str(Path.home() / ".OmniMate"))
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
        # 沙箱环境变量:洗掉密钥类(API key/数据库密码等),防泄漏给子进程
        from agent.sandbox_env import build_safe_env
        safe_env = build_safe_env()
        # 阶段 5 NEW: merge OMNIMATE_ENV_FILE 里的 hook 写入的 env 覆盖
        safe_env.update(_load_session_env_overrides())

        # 检测 GUI 程序启动命令(start / Chrome / 浏览器等)
        # GUI 程序不退出 → subprocess 管道永远等 → 卡死
        # 修复:不创建管道(DEVNULL),start 命令立即返回
        is_gui_launch = _is_gui_launch(command)  # R30c-B5：token 级判定

        if is_gui_launch:
            # GUI 命令:不创建管道,避免子进程继承管道导致卡死
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                cwd=cwd,
                env=safe_env,
            )
            # R30c-B5：如实告知输出未捕获（此前假装有结果 "(GUI 程序已启动)"）
            return json.dumps({
                "stdout": "(GUI 程序已启动；其 stdout/stderr 未捕获——不建管道防卡死，输出不可用)",
                "stderr": "",
                "exit_code": result.returncode,
                "command": command,
                "cwd": cwd,
            }, ensure_ascii=False)
        elif sandbox_active and win_job_mode:
            # CCAR12: Windows Job Object 模式——命令不包装正常启动，
            # 启动后 attach job（进程树管控；attach 失败 fail-open 继续执行）。
            # 公共路径提取到 sandbox_runner.run_with_job_object（超时收尸后
            # 重抛 TimeoutExpired，由本函数外层 except 转 error JSON）
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
            # OS 沙箱路径：用包装后的 argv，shell=False
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
            # 原路径：sandbox off 或 fail-open
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
        # 截断（防止爆 context）
        stdout_truncated_raw = _truncate_output(result.stdout)
        stderr_truncated_raw = _truncate_output(result.stderr)

        # 大输出 offload（Phase 1 后始终启用）
        tool_call_id = kwargs.get("tool_call_id")
        config = kwargs.get("config")
        omnimate_home = kwargs.get("omnimate_home")
        final_stdout = _finalize_output(stdout_truncated_raw, tool_call_id, omnimate_home, config)
        stdout_offloaded = final_stdout != stdout_truncated_raw

        return json.dumps({
            "stdout": final_stdout,
            "stderr": stderr_truncated_raw,
            "exit_code": result.returncode,
            "command": command,
            "cwd": cwd,
            "stdout_truncated": len(result.stdout or "") > MAX_OUTPUT_CHARS,
            "stderr_truncated": len(result.stderr or "") > MAX_OUTPUT_CHARS,
            "stdout_offloaded": stdout_offloaded,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"命令超时（{timeout}秒）",
            "command": command,
            "timeout": timeout,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "command": command,
        }, ensure_ascii=False)


# 模块级注册（import 时自动执行）
registry.register(
    name="terminal",
    toolset="core",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    isConcurrencySafe=False,  # 副作用：执行 shell 命令（最强副作用），必须串行
)
