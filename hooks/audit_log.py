#!/usr/bin/env python
"""audit_log.py — 内置 hook 示例：审计日志。

干什么：用户每次提交 prompt、每次工具调用结束，都往日志文件里记一笔，
方便事后查账（这个 agent 到底干了什么、工具返回了什么）。

跟主程序怎么通信（hook 都是独立小进程，走 stdin/stdout 传 JSON）：
  USER_PROMPT_SUBMIT（用户提交输入时）stdin 收：{"event": "...", "prompt": "...", ...}
  POST_TOOL_USE（工具跑完后）stdin 收：{"event": "...", "tool_name": "...", "result": "...", ...}
  stdout 不输出任何东西（只旁观记账，不改原数据）

怎么启用（在 settings.json 里加）：
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
    """日志文件写到哪：优先读环境变量 AUDIT_LOG_PATH，没设就用默认的 ~/.OmniMate/.audit.log。

    返回：展开 ~ 后的 Path。
    """
    raw = os.environ.get("AUDIT_LOG_PATH", "~/.OmniMate/.audit.log")
    return Path(raw).expanduser()


def main():
    """入口：从 stdin 读事件 JSON，拼一行日志追加写进日志文件。

    背景：stdin 读不进有效 JSON 就安静退出（hook 不能因为自己挂了连累主程序）。
    """
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    event = payload.get("event", "unknown")
    ts = datetime.now().isoformat(timespec="seconds")
    session = payload.get("session_id", "?")[:8]

    if event == "user_prompt_submit":
        prompt = payload.get("prompt", "")
        # 太长的 prompt 只留前 500 字符（日志是查线索的，不是全文备份）
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
        # fail-open：日志写不进去就写不进去，绝不能因此卡住 agent 主流程
        pass


if __name__ == "__main__":
    main()
