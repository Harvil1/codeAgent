"""终端工具：执行 shell 命令。

集成权限检查（三道闸门）和输出截断（50000 字符）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from agent.output_offload import finalize_tool_output as _finalize_output
from agent.permission import get_default_checker
from tools.registry import registry


# 输出截断阈值（防止爆 context）
MAX_OUTPUT_CHARS = 50000


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

    timeout = args.get("timeout", 120)
    cwd = args.get("cwd") or os.getcwd()

    # 权限检查（闸门 1/2/3）
    checker = kwargs.get("permission_checker") or get_default_checker()
    perm = checker.check(command, cwd=cwd)
    if not perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: {perm.reason}",
            "error_type": "permission_denied",
            "gate": perm.gate,
            "command": command,
        }, ensure_ascii=False)

    try:
        # 沙箱环境变量:洗掉密钥类(API key/数据库密码等),防泄漏给子进程
        from agent.sandbox_env import build_safe_env
        safe_env = build_safe_env()
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
        harvil_home = kwargs.get("harvil_home")
        final_stdout = _finalize_output(stdout_truncated_raw, tool_call_id, harvil_home, config)
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
)
