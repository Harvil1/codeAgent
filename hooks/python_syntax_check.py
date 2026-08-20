#!/usr/bin/env python
"""python_syntax_check.py — 内置 hook：Python 语法门禁。

干什么：write_file 要写 .py 文件时，先用 ast 把内容解析一遍，
语法不过关就拒绝写入——别让 agent 把跑不起来的坏代码落到磁盘上。

跟主程序怎么通信（PRE_TOOL_USE，工具执行前触发）：
  stdin 收：{"event": "pre_tool_use", "tool_name": "write_file",
           "args": {"path": "...", "content": "..."}, ...}
  stdout 回：语法错时 {"action": "deny", "reason": "..."}；不输出就放行

启用方法（settings.json 里加）：
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
    """入口：从 stdin 读工具调用信息，是写 .py 文件就先验语法。"""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    tool = payload.get("tool_name")
    if tool != "write_file":
        return  # 别的工具不归我管

    args = payload.get("args", {})
    path = args.get("path", "")
    content = args.get("content", "")

    # 只管 .py 文件，其他文件不拦
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
