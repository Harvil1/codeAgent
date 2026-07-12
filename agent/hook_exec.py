"""声明式 hook 的子进程执行 + JSON IPC 协议。

子进程：
- stdin 收到 payload 的 JSON
- stdout 期望是合法 JSON（空 stdout = 空 dict）
- 超时 kill
- exit code != 0 / 启动失败 / JSON 解析失败 → None（fail-open）
"""
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def run_script_hook(hook, payload: dict) -> Optional[dict]:
    """在子进程中执行声明式 hook。

    参数：
        hook: Hook 实例（kind="declarative"，script 非 None）
        payload: 要传给子进程的 dict（含 event/session_id/timestamp/事件字段）

    返回：
        解析后的 dict（可能为空 dict 表示 allow），或 None（任何故障）
    """
    if hook.script is None:
        logger.warning("hook %s 缺 script 配置", hook.name)
        return None

    payload_json = json.dumps(payload, ensure_ascii=False)
    env = {**os.environ, **(hook.script.env or {})}

    try:
        proc = subprocess.run(
            hook.script.command,
            input=payload_json,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=hook.script.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("hook %s 超时 (%.1fs)", hook.name, hook.script.timeout)
        return None
    except OSError as e:
        logger.warning("hook %s 启动失败: %s", hook.name, e)
        return None

    if proc.returncode != 0:
        logger.warning("hook %s exit %d: %s",
                       hook.name, proc.returncode, (proc.stderr or "")[:200])
        return None

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {}  # 空 stdout = allow

    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            logger.warning("hook %s stdout 非合法 JSON dict: %r", hook.name, parsed)
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("hook %s stdout 非合法 JSON: %s", hook.name, e)
        return None
