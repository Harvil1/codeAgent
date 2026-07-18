#!/usr/bin/env python
"""secrets_redactor.py — POST_TOOL_USE hook.

扫描工具输出，把疑似密钥的模式替换为 [REDACTED:类型]。
防止 API key 等 accidentally 暴露给 LLM（被存进 conversation history）。

IPC 协议（POST_TOOL_USE）：
  stdin:  {"event": "post_tool_use", "tool_name": "...", "result": "...", ...}
  stdout: {"result": "脱敏后的字符串"} 或空（不修改）

启用方法（settings.json）：
{
  "hooks": {
    "post_tool_use": [{
      "name": "secrets-redactor",
      "command": ["python", "hooks/secrets_redactor.py"],
      "timeout": 5.0
    }]
  }
}
"""
import json
import re
import sys


PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:api_key]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"), "[REDACTED:bearer]"),
    (re.compile(r"api_key[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"), "[REDACTED:api_key]"),
    (re.compile(r"token[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"), "[REDACTED:token]"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
     "[REDACTED:pem_key]"),
]


def redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for pat, repl in PATTERNS:
        out = pat.sub(repl, out)
    return out


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # fail-open

    result = payload.get("result")
    if not isinstance(result, str):
        return  # 只处理字符串结果

    redacted = redact(result)
    if redacted != result:
        print(json.dumps({"result": redacted}))


if __name__ == "__main__":
    main()
