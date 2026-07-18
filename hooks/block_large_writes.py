#!/usr/bin/env python
"""block_large_writes.py — PRE_TOOL_USE hook.

write_file 写超大文件（默认 5MB）时拒绝，防意外写爆磁盘。

IPC 协议（PRE_TOOL_USE）：
  stdin:  {"event": "pre_tool_use", "tool_name": "write_file",
           "args": {"path": "...", "content": "..."}, ...}
  stdout: {"action": "deny", "reason": "..."} 或空

启用方法（settings.json）：
{
  "hooks": {
    "pre_tool_use": [{
      "name": "block-large-writes",
      "command": ["python", "hooks/block_large_writes.py"],
      "timeout": 2.0,
      "env": {"MAX_WRITE_BYTES": "5242880"}
    }]
  }
}
"""
import json
import os
import sys


DEFAULT_MAX = 5 * 1024 * 1024  # 5 MB


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    tool = payload.get("tool_name")
    if tool != "write_file":
        return

    args = payload.get("args", {})
    content = args.get("content", "")
    if not isinstance(content, str):
        return

    max_bytes = int(os.environ.get("MAX_WRITE_BYTES", DEFAULT_MAX))
    actual_bytes = len(content.encode("utf-8"))

    if actual_bytes > max_bytes:
        print(json.dumps({
            "action": "deny",
            "reason": (
                f"文件过大：{actual_bytes} bytes > 上限 {max_bytes} bytes "
                f"({max_bytes // (1024*1024)}MB)。如需写入大文件，"
                f"调高 MAX_WRITE_BYTES 环境变量或拆分写入。"
            ),
        }))


if __name__ == "__main__":
    main()
