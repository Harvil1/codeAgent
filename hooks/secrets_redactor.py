#!/usr/bin/env python
"""secrets_redactor.py — 内置 hook：秘密遮蔽。

干什么：工具输出里经常一不小心带出 API key、token、私钥这类敏感信息
（比如 cat 了一个配置文件）。这个 hook 在结果交给 LLM 之前扫一遍，
把长得像密钥的片段替换成 [REDACTED:类型]。不遮的话，密钥会跟着
对话历史一路存下去、还发给别人家的模型服务。

跟主程序怎么通信（POST_TOOL_USE，工具跑完后触发）：
  stdin 收：{"event": "post_tool_use", "tool_name": "...", "result": "...", ...}
  stdout 回：{"result": "脱敏后的字符串"}；不需要改就什么都不输出

启用方法（settings.json 里加）：
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


# 要抓的秘密模式清单：（正则, 替换成什么）——sk- 开头的 key、Bearer 令牌、
# api_key/token 赋值、PEM 私钥块
PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:api_key]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"), "[REDACTED:bearer]"),
    (re.compile(r"api_key[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"), "[REDACTED:api_key]"),
    (re.compile(r"token[\"\s:=]+[\"']?[A-Za-z0-9]{16,}"), "[REDACTED:token]"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
     "[REDACTED:pem_key]"),
]


def redact(text: str) -> str:
    """把文本里所有疑似密钥的片段替换成 [REDACTED:类型] 标记。

    参数：
        text：待脱敏的文本（不是字符串就原样退回）

    返回：
        替换后的文本。逐条模式过一遍清单，谁命中换谁。
    """
    if not isinstance(text, str):
        return text
    out = text
    for pat, repl in PATTERNS:
        out = pat.sub(repl, out)
    return out


def main():
    """入口：从 stdin 读工具结果，脱敏后有改动才输出新结果。

    背景：没改动就不输出——主程序把"无输出"理解为"保持原样"。
    """
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # 读不进来就安静退场（fail-open）

    result = payload.get("result")
    if not isinstance(result, str):
        return  # 只处理字符串型的结果

    redacted = redact(result)
    if redacted != result:
        print(json.dumps({"result": redacted}))


if __name__ == "__main__":
    main()
