"""工具可见性规则（T6，核心机制对齐第 6 项）。

settings.json 的 permissions 段：
    {"permissions": {"allow": [], "deny": [...]}}

语法（第一版收窄，不搬 Bash(cmd:*) 子命令级——那是权限闸门的事）：
    - 精确名：read_file
    - 前缀通配：mcp__server__*（fnmatch）
    - 整服务器：mcp__server（匹配 mcp__server 自身 + mcp__server__<tool> 全部）

语义：
    - deny 匹配 → 工具从 LLM 可见性移除 + registry.dispatch 防御性拒绝
    - allow 匹配 → 豁免 deny（deny mcp__foo + allow mcp__foo__bar → bar 可见）
    - 两个列表都空（默认）→ 无任何影响

应用点：
    - model_tools.get_tool_definitions（LLM 看不到）
    - tools/registry.dispatch（可见性过滤之外的防御纵深）

settings.json 用 mtime+size 双因子缓存（Windows mtime 精度 ~15ms，
同窗口写文件单因子缓存会误判有效——历史踩过）。
"""
import fnmatch
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# mtime+size 双因子缓存（None = 未加载）
_rules_cache: Optional[Dict] = {
    "key": None,       # (mtime_ns, size)
    "rules": {"allow": [], "deny": []},
}


def reset_rules_cache() -> None:
    """测试用：清空规则缓存，下次强制重读 settings.json。"""
    global _rules_cache
    _rules_cache = {"key": None, "rules": {"allow": [], "deny": []}}


def load_tool_permission_rules() -> Dict[str, List[str]]:
    """读 settings.json 的 permissions 段（mtime+size 双因子缓存）。fail-open。"""
    try:
        from agent.settings import settings_path
        path = settings_path()
        if not path.exists():
            return {"allow": [], "deny": []}
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if _rules_cache["key"] == key:
            return _rules_cache["rules"]

        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        sec = data.get("permissions") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            rules = {"allow": [], "deny": []}
        else:
            rules = {
                "allow": [str(r) for r in (sec.get("allow") or []) if isinstance(r, (str,))],
                "deny": [str(r) for r in (sec.get("deny") or []) if isinstance(r, (str,))],
            }
        _rules_cache["key"] = key
        _rules_cache["rules"] = rules
        return rules
    except Exception as e:
        logger.debug("load_tool_permission_rules fail-open: %s", e)
        return {"allow": [], "deny": []}


def tool_matches(rule: str, tool_name: str) -> bool:
    """单条规则是否匹配工具名。

    - 精确名相等
    - fnmatch 通配（mcp__server__*）
    - 整服务器：规则无通配符且工具名以 <rule>__ 开头（mcp__foo 匹配 mcp__foo__bar）
    """
    if not rule:
        return False
    if rule == tool_name:
        return True
    if any(ch in rule for ch in "*?["):
        return fnmatch.fnmatchcase(tool_name, rule)
    return tool_name.startswith(rule + "__")


def is_tool_denied(tool_name: str, rules: Optional[Dict[str, List[str]]] = None) -> bool:
    """工具是否被 permissions 规则拒绝（allow 豁免优先于 deny）。"""
    if rules is None:
        rules = load_tool_permission_rules()
    allow = rules.get("allow") or []
    for r in allow:
        if tool_matches(r, tool_name):
            return False  # allow 豁免
    for r in (rules.get("deny") or []):
        if tool_matches(r, tool_name):
            return True
    return False
