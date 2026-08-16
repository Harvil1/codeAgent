"""公共秘密扫描器（R19 #24）。

对齐 CC teamMemorySync/secretScanner 的 gitleaks 核心规则族，把 handoff
已有的 5 类模式抽成公共模块并扩展，接入全链路：

- **记忆写入**（memory_store.save/update）：命中 → ValueError 拒绝写入
  （fail-closed——秘密不该进记忆库，错误信息只含规则 ID + 截断 snippet）
- **curator 改写产物**（safe_rewrite_body 的 new_body）：命中 → 拒绝改写
  保留原文（LLM 重写引入的密钥挡在门外）
- **trace**（trace.emit 的 fields）：值命中 → 替换为 [REDACTED:rule]
  （fail-open 日志通道，redact 而非拒绝）
- **handoff**：_scan_for_secrets 迁移到公共规则（保持 transcript 语义）

规则族（合并单正则 + 命名组，一次扫描；命名组即规则 ID）：
openai(sk-...) / anthropic(sk-ant-) / github-pat(ghp_...) / aws(AKIA...)
/ google(AIza...) / slack(xox...) / bearer / api_key= / token= / PEM /
jwt(eyJ..)
"""
import re
from typing import Any, Dict, List

# 命名组名即规则 ID（对齐 gitleaks 规则名风格）
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
    """扫描单段文本。返回命中列表（只含规则 ID + 截断 snippet，不泄露完整值）。"""
    if not text:
        return []
    hits: List[Dict[str, Any]] = []
    for m in SECRET_RULES_RE.finditer(text):
        rule = next((k for k, v in m.groupdict().items() if v), "unknown")
        hits.append({
            "rule": rule,
            "snippet": m.group(0)[:50],  # 截断防再次暴露
        })
    return hits


def find_secrets_in(*texts: str) -> List[Dict[str, Any]]:
    """扫描多段文本（save 的 name/description/summary/body 一次过）。"""
    all_hits: List[Dict[str, Any]] = []
    for t in texts:
        if isinstance(t, str):
            all_hits.extend(scan_text(t))
    return all_hits


def redact_value(value: Any) -> Any:
    """trace 用：值含命中 → 替换为 [REDACTED:rule]（保类型其余原样）。"""
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
    """trace.emit 的 fields 整体 redact（值级，fail-open）。"""
    try:
        return {k: redact_value(v) for k, v in fields.items()}
    except Exception:
        return fields
