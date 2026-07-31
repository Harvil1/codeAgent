#!/usr/bin/env python
"""audit_log.py — USER_PROMPT_SUBMIT + POST_TOOL_USE hook.

把用户 prompt 和工具调用结果写到日志文件，便于审计/调试。

IPC 协议：
  USER_PROMPT_SUBMIT stdin: {"event": "...", "prompt": "...", ...}
  POST_TOOL_USE      stdin: {"event": "...", "tool_name": "...", "result": "...", ...}
  stdout: 空（不修改原数据，只写日志）

启用方法（settings.json）：
{
  "hooks": {
    "user_prompt_submit": [{
      "name": "audit-log",
      "command": ["python", "hooks/audit_log.py"],
      "timeout": 2.0,
      "env": {"AUDIT_LOG_PATH": "~/.OmniMate/.audit.log"}
    }],
    "post_tool_use": [{
      "name": "audit-log",
      "command": ["python", "hooks/audit_log.py"],
      "timeout": 2.0,
      "env": {"AUDIT_LOG_PATH": "~/.OmniMate/.audit.log"}
    }]
  }
}
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def get_log_path() -> Path:
    raw = os.environ.get("AUDIT_LOG_PATH", "~/.OmniMate/.audit.log")
    return Path(raw).expanduser()


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    event = payload.get("event", "unknown")
    ts = datetime.now().isoformat(timespec="seconds")
    session = payload.get("session_id", "?")[:8]

    if event == "user_prompt_submit":
        prompt = payload.get("prompt", "")
        # 截断长 prompt
        if len(prompt) > 500:
            prompt = prompt[:497] + "..."
        line = f"[{ts}] [{session}] USER: {prompt}\n"
    elif event == "post_tool_use":
        tool = payload.get("tool_name", "?")
        result = payload.get("result", "")
        if len(result) > 200:
            result = result[:197] + "..."
        line = f"[{ts}] [{session}] TOOL {tool}: {result}\n"
    else:
        line = f"[{ts}] [{session}] {event}\n"

    try:
        log_path = get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        # fail-open：日志写不进去不应阻塞 agent
        pass


if __name__ == "__main__":
    main()
