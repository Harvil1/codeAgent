#!/usr/bin/env python
"""python_syntax_check.py — PRE_TOOL_USE hook.

write_file 写 .py 文件时先做语法检查，失败则拒绝（防 agent 写坏代码到磁盘）。

IPC 协议（PRE_TOOL_USE）：
  stdin:  {"event": "pre_tool_use", "tool_name": "write_file",
           "args": {"path": "...", "content": "..."}, ...}
  stdout: {"action": "deny", "reason": "..."} 或 {"action": "allow"} 或空

启用方法（settings.json）：
{
  "hooks": {
    "pre_tool_use": [{
      "name": "python-syntax-check",
      "command": ["python", "hooks/python_syntax_check.py"],
      "timeout": 3.0
    }]
  }
}
"""
import ast
import json
import sys


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    tool = payload.get("tool_name")
    if tool != "write_file":
        return  # 只管 write_file

    args = payload.get("args", {})
    path = args.get("path", "")
    content = args.get("content", "")

    # 只检查 .py 文件
    if not path.endswith(".py"):
        return

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(json.dumps({
            "action": "deny",
            "reason": f"Python 语法错误（line {e.lineno}): {e.msg}",
        }))


if __name__ == "__main__":
    main()
