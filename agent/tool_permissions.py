"""工具可见性规则（T6，核心机制对齐第 6 项）+ 内容级权限规则（R16 #3）。

settings.json 的 permissions 段：
    {"permissions": {"allow": [], "deny": [...], "ask": [...]}}

## 工具可见性规则（T6）

语法（第一版收窄）：
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

## 内容级权限规则（R16 #3，对齐 CCB Bash(npm publish:*) 语法）

形如 ``Bash(...)`` / ``Terminal(...)`` 的条目是**命令内容级**规则，由
PermissionChecker.check 消费（与工具可见性互不干扰——含括号的条目在
tool_matches 下永不匹配工具名）。内容三形态：

    - 精确：Bash(npm test)           —— 整条命令逐字相等
    - 旧前缀：Bash(npm:*)             —— npm 或 npm <args>（词边界）
    - 通配：Bash(git *)               —— git 任意子命令（\\* 转义字面量；
                                          尾部单独 " *" 时裸命令也匹配，对齐
                                          前缀语义）

语义（check_command_rules，deny > ask > allow）：
    - deny 命中 → 任何模式都拒（含 bypassPermissions——用户显式 deny 是最高意图）
    - ask 命中 → 强制审批（bypass 也不豁免，对齐 CC 内容级 ask 语义）
    - allow 命中 → 跳过注入面/破坏性审批（硬底线 fatal/黑名单/危险删除不受影响）

遮蔽检测（对齐 CC shadowedRuleDetection）：
    allow 的内容级规则被同工具整级（裸 "Bash"/"Terminal"）deny/ask 遮蔽时
    永不可达 → 加载时 logger.warning 告警（detect_shadowed_command_rules）。

settings.json 用 mtime+size 双因子缓存（Windows mtime 精度 ~15ms，
同窗口写文件单因子缓存会误判有效——历史踩过）。
"""
import fnmatch
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# mtime+size 双因子缓存（None = 未加载）
_rules_cache: Optional[Dict] = {
    "key": None,       # (mtime_ns, size)
    "rules": {"allow": [], "deny": [], "ask": []},
}


def reset_rules_cache() -> None:
    """测试用：清空规则缓存，下次强制重读 settings.json。"""
    global _rules_cache
    _rules_cache = {"key": None, "rules": {"allow": [], "deny": [], "ask": []}}


def load_tool_permission_rules() -> Dict[str, List[str]]:
    """读 settings.json 的 permissions 段（mtime+size 双因子缓存）。fail-open。

    R16 #3：新增 ask 列表（内容级强制审批规则）；缓存未命中时跑遮蔽检测告警。
    """
    try:
        from agent.settings import settings_path
        path = settings_path()
        if not path.exists():
            return {"allow": [], "deny": [], "ask": []}
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if _rules_cache["key"] == key:
            return _rules_cache["rules"]

        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        sec = data.get("permissions") if isinstance(data, dict) else None
        if not isinstance(sec, dict):
            rules = {"allow": [], "deny": [], "ask": []}
        else:
            rules = {
                "allow": [str(r) for r in (sec.get("allow") or []) if isinstance(r, (str,))],
                "deny": [str(r) for r in (sec.get("deny") or []) if isinstance(r, (str,))],
                # R16 #3: ask 列表（内容级强制审批规则）
                "ask": [str(r) for r in (sec.get("ask") or []) if isinstance(r, (str,))],
            }
        _rules_cache["key"] = key
        _rules_cache["rules"] = rules
        # R16 #3: 遮蔽检测（仅缓存未命中时跑——首次加载/settings 变更后）
        for warning in detect_shadowed_command_rules(rules):
            logger.warning("权限规则遮蔽: %s", warning)
        return rules
    except Exception as e:
        logger.debug("load_tool_permission_rules fail-open: %s", e)
        return {"allow": [], "deny": [], "ask": []}


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


# ---------------------------------------------------------------------------
# R16 #3：内容级权限规则（Bash(...) / Terminal(...) 语法）
# ---------------------------------------------------------------------------

# 内容级规则外壳（Bash 沿用 CC 语法习惯；Terminal 是 OmniMate 原生工具名）
_CMD_RULE_RE = re.compile(r"^(?:Bash|Terminal)\((.*)\)$", re.IGNORECASE | re.DOTALL)
# 旧前缀语法 x:* 结尾
_LEGACY_PREFIX_RE = re.compile(r"^(.+):\*$", re.DOTALL)

# 解析结果缓存（规则字符串 → ("exact", c) / ("prefix", p) / ("wildcard", w) / None）
# 规则条目少量且稳定，进程内字典缓存即可。
_parsed_rule_cache: Dict[str, Optional[Tuple[str, str]]] = {}
# 通配正则缓存
_wildcard_regex_cache: Dict[str, "re.Pattern"] = {}


def _has_wildcards(content: str) -> bool:
    """内容是否含未转义的 *（排除结尾 :* 的旧前缀语法）。"""
    if content.endswith(":*"):
        return False
    for i, ch in enumerate(content):
        if ch != "*":
            continue
        backslashes = 0
        j = i - 1
        while j >= 0 and content[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 0:
            return True
    return False


def parse_command_rule(rule: str) -> Optional[Tuple[str, str]]:
    """解析一条规则字符串 → (kind, value)；非内容级命令规则返回 None。

    kind: "exact"（逐字）/ "prefix"（x:* 旧前缀）/ "wildcard"（含未转义 *）
    """
    cached = _parsed_rule_cache.get(rule)
    if cached is not None or rule in _parsed_rule_cache:
        return _parsed_rule_cache.get(rule)
    result: Optional[Tuple[str, str]] = None
    m = _CMD_RULE_RE.match(rule.strip())
    if m:
        content = m.group(1)
        pm = _LEGACY_PREFIX_RE.match(content)
        if pm and not _has_wildcards(content):
            result = ("prefix", pm.group(1))
        elif _has_wildcards(content):
            result = ("wildcard", content)
        else:
            result = ("exact", content)
    if len(_parsed_rule_cache) > 500:
        _parsed_rule_cache.clear()
    _parsed_rule_cache[rule] = result
    return result


def _count_unescaped_stars(p: str) -> int:
    """源模式中未转义 * 的个数。"""
    count = 0
    i = 0
    while i < len(p):
        if p[i] == "\\" and i + 1 < len(p) and p[i + 1] in ("*", "\\"):
            i += 2  # 转义对整体跳过
            continue
        if p[i] == "*":
            count += 1
        i += 1
    return count


def _wildcard_regex(pattern: str) -> "re.Pattern":
    """通配模式 → 全串匹配正则（\\* 字面量；尾部唯一 " *" 时裸命令也匹配）。"""
    cached = _wildcard_regex_cache.get(pattern)
    if cached is not None:
        return cached
    p = pattern.strip()

    def build(src: str) -> List[str]:
        parts: List[str] = []
        i = 0
        while i < len(src):
            c = src[i]
            if c == "\\" and i + 1 < len(src) and src[i + 1] in ("*", "\\"):
                parts.append(re.escape(src[i + 1]))
                i += 2
                continue
            if c == "*":
                parts.append(".*")
            else:
                parts.append(re.escape(c))
            i += 1
        return parts

    # 尾部单独 " *"（唯一通配）→ 可选参数组，对齐前缀语义：git * 匹配裸 git。
    # 在源模式层判断（不能用拼接结果切片——转义空格等形态会让切片错位）。
    if p.endswith(" *") and _count_unescaped_stars(p) == 1:
        parts = build(p[:-2]) + ["(?: .*)?"]
    else:
        parts = build(p)
    compiled = re.compile("^" + "".join(parts) + "$", re.DOTALL)
    if len(_wildcard_regex_cache) > 500:
        _wildcard_regex_cache.clear()
    _wildcard_regex_cache[pattern] = compiled
    return compiled


# ---------------------------------------------------------------------------
# R25 #1：命令形态归一化（剥 env 前缀 + 安全包装词，防规则绕过）
# 对齐 CCB bashPermissions.stripAllLeadingEnvVars + SAFE_WRAPPER 剥离。
# ---------------------------------------------------------------------------

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_WRAPPER_DURATION_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")


def _normalize_command_for_rules(command: str) -> str:
    """剥掉命令头部的环境变量赋值前缀和安全包装词，用于内容级规则匹配。

    防的是：用户 deny 了 ``Bash(rm:*)``，agent 发 ``FOO=1 rm xxx`` 绕过匹配。
    只剥**头部**（对齐 CCB 语义）；复合命令中段的 env 赋值（``a && B=1 rm``）
    是既有匹配语义未覆盖的形态，记录为已知限制不在此处理。

    剥离形态（定点迭代直到剥不动）：
      - ``NAME=value`` 赋值 token
      - ``env`` 后跟任意个赋值 token（env 本身也剥）
      - ``nohup``
      - ``timeout <时长>``（两 token 一起）
      - ``nice`` / ``nice -n <数字>``
      - ``stdbuf`` 后跟任意个 ``-`` 开头的选项 token
    """
    tokens = command.split()
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        if tok == "env":
            # env 后面跟的赋值也剥；env 后面直接是命令则只剥 env
            i += 1
            continue
        if tok == "nohup":
            i += 1
            continue
        if tok == "timeout" and i + 1 < n and _WRAPPER_DURATION_RE.match(tokens[i + 1]):
            i += 2
            continue
        if tok == "nice":
            if i + 2 < n and tokens[i + 1] == "-n" and tokens[i + 2].lstrip("-").isdigit():
                i += 3
            else:
                i += 1
            continue
        if tok == "stdbuf":
            i += 1
            while i < n and tokens[i].startswith("-"):
                i += 1
            continue
        break
    return " ".join(tokens[i:]) if i else command


def command_rule_matches(rule: str, command: str) -> bool:
    """内容级规则是否匹配命令（exact / prefix / wildcard 三形态）。"""
    parsed = parse_command_rule(rule)
    if parsed is None or not command:
        return False
    kind, value = parsed
    if kind == "exact":
        return command == value
    if kind == "prefix":
        # 词边界前缀：npm 匹配 "npm" / "npm test"，不匹配 "npmx"
        return command == value or command.startswith(value + " ")
    return _wildcard_regex(value).match(command) is not None


def check_command_rules(command: str, rules: Optional[Dict[str, List[str]]] = None) -> str:
    """内容级规则判定（R16 #3 主入口，PermissionChecker.check 调用）。

    返回 "deny" / "ask" / "allow" / "none"，优先级 deny > ask > allow。
    工具可见性条目（read_file 等无括号形态）不参与——parse 返回 None。
    R25 #1：匹配前先剥 env 前缀/安全包装词（FOO=bar rm xxx 绕不过 deny(rm)）。
    """
    if rules is None:
        rules = load_tool_permission_rules()
    command = _normalize_command_for_rules(command)
    for r in (rules.get("deny") or []):
        if command_rule_matches(r, command):
            return "deny"
    for r in (rules.get("ask") or []):
        if command_rule_matches(r, command):
            return "ask"
    for r in (rules.get("allow") or []):
        if command_rule_matches(r, command):
            return "allow"
    return "none"


def detect_shadowed_command_rules(rules: Dict[str, List[str]]) -> List[str]:
    """遮蔽检测（对齐 CC shadowedRuleDetection 的核心子集）。

    allow 的内容级规则被同工具**整级** deny/ask 规则（裸 "Bash"/"Terminal"，
    无括号）遮蔽时永不可达 → 返回告警消息列表。

    CC 的 source 体系（userSettings/projectSettings/…）OmniMate 没有对应物
    （单一 settings.json），只做同源遮蔽判定；sandbox 豁免特例也不适用。
    """
    warnings: List[str] = []
    tool_wide = {"Bash", "Terminal"}
    deny_wide = any(r.strip() in tool_wide for r in (rules.get("deny") or []))
    ask_wide = any(r.strip() in tool_wide for r in (rules.get("ask") or []))
    if not deny_wide and not ask_wide:
        return warnings
    for r in (rules.get("allow") or []):
        if parse_command_rule(r) is None:
            continue  # 只检查内容级 allow（整级 allow 不在遮蔽语义内）
        if deny_wide:
            warnings.append(
                f"allow 规则 {r!r} 被整级 deny 规则（Bash/Terminal）遮蔽，永不可达"
                f"——请删除其中之一"
            )
        elif ask_wide:
            warnings.append(
                f"allow 规则 {r!r} 被整级 ask 规则（Bash/Terminal）遮蔽，"
                f"ask 优先级更高导致该 allow 永不生效——请删除其中之一"
            )
    return warnings
