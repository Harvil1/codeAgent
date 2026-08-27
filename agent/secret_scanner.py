"""公共秘密扫描器。

防止 API 密钥、token 这类敏感信息混进持久化数据。规则族覆盖
gitleaks 的核心规则，接入四条链路（各链路策略不同，
按"秘密该不该出现在这里"分寸处理）：

- **记忆写入**（memory_store.save/update）：命中 → 直接 ValueError 拒绝写入
  （fail-closed，宁拒不收——秘密不该进记忆库；报错信息只含规则 ID +
  截断片段，不在报错里二次泄露完整密钥）
- **curator 改写产物**（safe_rewrite_body 的 new_body）：命中 → 拒绝这次
  改写、保留原文（LLM 重写时混进来的密钥挡在门外）
- **trace 日志**（trace.emit 的 fields）：值命中 → 替换成 [REDACTED:规则名]
  （日志是 fail-open 通道：redact 后照常记，不因秘密丢日志）
- **handoff**：_scan_for_secrets 迁移到公共规则（保持 transcript 原语义）

规则族（全部合成一条正则 + 命名组，一次扫描跑完；命名组名即规则 ID）：
openai(sk-...) / anthropic(sk-ant-) / github-pat(ghp_...) / aws(AKIA...)
/ google(AIza...) / slack(xox...) / bearer / api_key= / token= / PEM /
jwt(eyJ..)
"""
import re
from typing import Any, Dict, List

# 命名组名即规则 ID（对齐 gitleaks 的规则命名风格）
SECRET_RULES_RE = re.compile(
    r"(?P<openai>sk-(?!ant-)[A-Za-z0-9_\-]{20,})"
    r"|(?P<anthropic>sk-ant-[A-Za-z0-9_\-]{20,})"
    r"|(?P<github_pat>(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,})"
    r"|(?P<aws_access_token>AKIA[0-9A-Z]{16})"
    r"|(?P<google_api_key>AIza[0-9A-Za-z_\-]{35})"
    r"|(?P<slack_token>xox[baprs]-[A-Za-z0-9\-]{10,})"
    r"|(?P<bearer>Bearer\s+[A-Za-z0-9_\-\.]{20,})"
    r"|(?P<api_key>api_key[\"\s:=]+[\"']?[A-Za-z0-9]{16,})"
    r"|(?P<token>token[\"\s:=]+[\"']?[A-Za-z0-9]{16,})"
    r"|(?P<pem>-----BEGIN [A-Z ]+PRIVATE KEY-----)"
    r"|(?P<jwt>eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,})"
)


def scan_text(text: str) -> List[Dict[str, Any]]:
    """扫描一段文本，返回命中列表。

    背景：所有链路的扫描都走这一个函数。

    参数：
    - text：待扫描的文本

    返回：命中列表，每项 {"rule": 规则 ID, "snippet": 截断到 50 字符的
    片段}——故意不给完整值，避免报告本身变成泄露渠道。
    """
    if not text:
        return []
    hits: List[Dict[str, Any]] = []
    for m in SECRET_RULES_RE.finditer(text):
        rule = next((k for k, v in m.groupdict().items() if v), "unknown")
        hits.append({
            "rule": rule,
            "snippet": m.group(0)[:50],  # 截断，防报告里再次暴露完整密钥
        })
    return hits


def find_secrets_in(*texts: str) -> List[Dict[str, Any]]:
    """一次扫描多段文本（记忆条目的 name/description/summary/body 一锅端）。

    参数：
    - *texts：任意多段文本（None / 非字符串的自动跳过）

    返回：所有段命中的汇总列表。
    """
    all_hits: List[Dict[str, Any]] = []
    for t in texts:
        if isinstance(t, str):
            all_hits.extend(scan_text(t))
    return all_hits


def redact_value(value: Any) -> Any:
    """把值里命中的秘密替换成 [REDACTED:规则名]，其余原样保留。

    背景：给 trace 日志用——日志不能因为含秘密就丢，但秘密也不能进日志。

    参数：
    - value：任意值（非字符串的不动，直接返回）

    返回：替换后的值（原值无命中时就是它自己）。
    """
    if not isinstance(value, str):
        return value
    hits = scan_text(value)
    if not hits:
        return value
    redacted = value
    for m in SECRET_RULES_RE.finditer(value):
        rule = next((k for k, v in m.groupdict().items() if v), "unknown")
        redacted = redacted.replace(m.group(0), f"[REDACTED:{rule}]")
    return redacted


def redact_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """对 trace.emit 的整个 fields 字典做值级 redact（fail-open）。

    参数：
    - fields：字段名 → 值 的字典

    返回：每个字符串值都过一遍 redact_value 的新字典；处理过程出错就
    原样返回（redact 是加固不是关卡，不能因此丢日志）。
    """
    try:
        return {k: redact_value(v) for k, v in fields.items()}
    except Exception:
        return fields
