#!/usr/bin/env python
"""block_large_writes.py — 内置 hook：拦截超大写入。

干什么：write_file 要写的内容超过上限（默认 5MB）就直接拒绝——
agent 失控狂写大文件时，这道闸能把磁盘保住。

跟主程序怎么通信（PRE_TOOL_USE，工具执行前触发）：
  stdin 收：{"event": "pre_tool_use", "tool_name": "write_file",
           "args": {"path": "...", "content": "..."}, ...}
  stdout 回：{"action": "deny", "reason": "..."} 拒绝；不输出就放行

启用方法（settings.json 里加，MAX_WRITE_BYTES 可调上限）：
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


DEFAULT_MAX = 5 * 1024 * 1024  # 默认上限 5MB


def main():
    """入口：从 stdin 读工具调用信息，超限的 write_file 输出 deny。"""
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
