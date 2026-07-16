"""execute_code 工具：在独立子进程执行 Python 代码块。

让 LLM 能直接写 Python 代码执行，而不只是调工具。

设计：
  - 独立子进程（subprocess.run + sys.executable -c）
  - 权限审批：首次调用走 PermissionChecker 审批 callback，会话内缓存
  - 超时：默认 30 秒（config.execute_code.default_timeout 可配）
  - 输出 cap：stdout/stderr 各 5000 字符（超出截断）
  - 工作目录：默认 cwd（和 terminal 一致）

安全默认 > 事后补救：默认 require_approval=True。
"""
import json
import logging
import subprocess
import sys
from typing import Optional

from agent.permission import get_default_checker
from tools.registry import registry

logger = logging.getLogger(__name__)


# 输出截断阈值（单通道）。5000 字符够看绝大多数 print 输出，
# 更大的数据用 write_file 落盘后 read_file 分段看。
_MAX_OUTPUT_CHARS = 5000


def _truncate_output(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """截断超长输出，保留前 limit 字符，加续写提示。"""
    if not text or len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n... [输出已截断：总 {len(text)} 字符，"
        f"已显示前 {limit} 字符。如需完整输出请写文件后分段读取] ..."
    )


EXECUTE_CODE_SCHEMA = {
    "name": "execute_code",
    "description": (
        "执行 Python 代码块。在独立子进程中运行，不影响主进程。\n"
        "适合：数据处理、循环计算、原型验证、批量文件操作等复杂逻辑。\n"
        "返回 stdout + stderr + exit_code。输出超过 5000 字符截断。\n"
        "注意：首次执行会请求用户审批（会话内缓存）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python 代码（完整可执行）",
            },
            "timeout": {
                "type": "number",
                "description": "超时秒数（默认 30）",
            },
        },
        "required": ["code"],
    },
}


# 会话级审批缓存：本次会话批准过的代码不再重复问。
# 以代码前 200 字符做指纹（平衡唯一性和缓存命中率）。
_approved_code_fingerprints: set = set()


def _check_execute_code_enabled(*, config: Optional[dict] = None, **_) -> bool:
    """check_fn：config.execute_code.enabled 控制可见性。"""
    return (config or {}).get("execute_code", {}).get("enabled", True)


def _request_approval(code: str, checker, **kwargs) -> Optional[str]:
    """请求用户审批 execute_code。

    返回 None 表示通过，返回 JSON 字符串表示拒绝（直接短路返回给 LLM）。

    审批链：
      1. 会话缓存命中 → 通过
      2. 持久化白名单命中（checker._persistent_whitelist）→ 通过
      3. checker.approval_callback 问用户 → 通过则加入会话缓存
      4. 无 callback → 拒绝（安全默认）
    """
    fingerprint = code[:200]

    # 1. 会话缓存
    if fingerprint in _approved_code_fingerprints:
        return None

    # 2. 持久化白名单（如果 checker 支持）
    persistent = getattr(checker, "_persistent_whitelist", set())
    approval_key = f"execute_code: {fingerprint}"
    if approval_key in persistent:
        _approved_code_fingerprints.add(fingerprint)
        return None

    # 3. 审批 callback
    callback = getattr(checker, "approval_callback", None)
    if callback is None:
        # 无 callback 时默认拒绝（安全默认 > 事后补救）
        return json.dumps(
            {
                "error": "execute_code 需要用户审批，但无审批 callback 可用",
                "error_type": "permission_denied",
                "gate": "approval",
            },
            ensure_ascii=False,
        )

    try:
        approved = bool(callback(approval_key))
    except Exception:
        approved = False

    if not approved:
        return json.dumps(
            {
                "error": "用户拒绝执行代码",
                "error_type": "permission_denied",
                "gate": "approval",
            },
            ensure_ascii=False,
        )

    # 批准：加入会话缓存
    _approved_code_fingerprints.add(fingerprint)
    return None


def _handle_execute_code(args: dict, **kwargs) -> str:
    """在独立子进程执行 Python 代码。

    流程：
      1. 参数校验（code 非空）
      2. 权限审批（config.execute_code.require_approval，默认 True）
      3. subprocess.run + sys.executable -c
      4. 输出截断（5000 字符）
    """
    code = (args.get("code") or "").strip()
    if not code:
        return json.dumps(
            {"error": "code 不能为空", "error_type": "invalid_args"},
            ensure_ascii=False,
        )

    config = kwargs.get("config") or {}
    ec_config = config.get("execute_code", {})

    # 超时：args.timeout 覆盖 config 默认值
    timeout_arg = args.get("timeout")
    if timeout_arg is None:
        actual_timeout = ec_config.get("default_timeout", 30)
    else:
        try:
            actual_timeout = float(timeout_arg)
        except (TypeError, ValueError):
            actual_timeout = ec_config.get("default_timeout", 30)

    # 权限审批：复用 PermissionChecker 的审批 callback（和 terminal 共享）
    if ec_config.get("require_approval", True):
        checker = kwargs.get("permission_checker") or get_default_checker()
        deny_result = _request_approval(code, checker)
        if deny_result is not None:
            return deny_result

    # 工作目录：config.cwd → 继承 terminal 的 cwd
    cwd = ec_config.get("cwd")

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=actual_timeout,
            cwd=cwd,
        )
        stdout_truncated = _truncate_output(result.stdout or "")
        stderr_truncated = _truncate_output(result.stderr or "")

        return json.dumps(
            {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": stdout_truncated,
                "stderr": stderr_truncated,
                "stdout_truncated": len(result.stdout or "") > _MAX_OUTPUT_CHARS,
                "stderr_truncated": len(result.stderr or "") > _MAX_OUTPUT_CHARS,
            },
            ensure_ascii=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "error": f"代码执行超时（{actual_timeout}s）",
                "error_type": "execute_timeout",
                "timeout": actual_timeout,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("execute_code 执行失败")
        return json.dumps(
            {"error": f"执行失败: {e}", "error_type": "execute_error"},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动执行）
registry.register(
    name="execute_code",
    toolset="core",
    schema=EXECUTE_CODE_SCHEMA,
    handler=_handle_execute_code,
    check_fn=_check_execute_code_enabled,
    emoji="🐍",
)
