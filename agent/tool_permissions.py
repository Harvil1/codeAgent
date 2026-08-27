"""工具可见性规则 + 命令内容级权限规则。

大白话：这个文件管两件事——"哪些工具允许给 AI 看到"和"哪些命令内容要
拦/要问"。规则全部来自 settings.json 的 permissions 段，长这样：
    {"permissions": {"allow": [], "deny": [...], "ask": [...]}}

## 一、工具可见性规则（按工具名匹配，管"AI 能不能用某个工具"）

写法（第一版故意收得很窄）：
    - 精确名：read_file
    - 前缀通配：mcp__server__*（fnmatch 通配符匹配）
    - 整服务器：mcp__server（既匹配它自己，也匹配该 server 下全部工具
      mcp__server__<tool>）

效果：
    - deny 命中 → 工具从 AI 的工具列表里藏起来，就算代码硬调 dispatch
      也会被防御性拒绝（两道保险）
    - allow 命中 → 抵消 deny（deny 了 mcp__foo 整个 server，又 allow 了
      mcp__foo__bar 这一个工具 → bar 仍然可用）。注意 allow 的唯一作用就是
      豁免 deny，不是"只允许这些"的白名单模式
    - 两个列表都空（默认）→ 什么都不影响

生效位置：
    - model_tools.get_tool_definitions（发给 AI 的工具清单里直接不出现）
    - tools/registry.dispatch（可见性过滤之外的第二道防线）

## 二、内容级权限规则（按命令内容匹配，管"某条具体命令要不要拦"）

写成 ``Bash(...)`` / ``Terminal(...)`` 形态的条目是命令内容级规则，由
PermissionChecker.check 消费（跟工具可见性互不干扰——带括号的条目在
tool_matches 里永远匹配不到工具名）。内容有三种写法：

    - 精确：Bash(npm test)   —— 命令逐字相等才算命中
    - 旧前缀：Bash(npm:*)    —— npm 或 npm <任意参数>（按词边界，npmx 不算）
    - 通配：Bash(git *)      —— git 任意子命令（\\* 是转义后的字面量星号；
                                尾部单独 " *" 时裸命令 git 本身也算命中，
                                对齐前缀语义）

效果（check_command_rules，优先级 deny > ask > allow）：
    - deny 命中 → 任何模式都拒，连 bypassPermissions 也不豁免——用户显式
      deny 是最高意图
    - ask 命中 → 强制走审批（bypass 也不豁免）
    - allow 命中 → 跳过注入面/破坏性审批（但 fatal 硬底线/黑名单/危险删除
      这些更靠前的检查不受影响）

遮蔽检测：allow 的内容级规则如果被
同工具的整级规则（裸的 "Bash"/"Terminal"）deny/ask 盖住，就永远不会生效
——加载时用 logger.warning 提醒用户（detect_shadowed_command_rules）。

历史踩坑（缓存）：settings.json 用 mtime+size 双因子判断变没变。Windows
的 mtime 精度只有 ~15ms，同一段时间窗口内写的文件，只看 mtime 的单因子
缓存会误判"没变"导致读到旧内容——真踩过。
"""
import fnmatch
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# mtime+size 双因子缓存（key 为 None 表示还没加载过）
_rules_cache: Optional[Dict] = {
    "key": None,       # (mtime_ns, size)
    "rules": {"allow": [], "deny": [], "ask": []},
}


def reset_rules_cache() -> None:
    """测试用：清空规则缓存，让下一次调用强制重读 settings.json。

    为什么需要：正常路径有缓存，测试改了 settings.json 后不重读会拿到旧规则。
    """
    global _rules_cache
    _rules_cache = {"key": None, "rules": {"allow": [], "deny": [], "ask": []}}


def load_tool_permission_rules() -> Dict[str, List[str]]:
    """读 settings.json 的 permissions 段，返回 allow/deny/ask 三个规则列表。

    干什么：把用户配置的权限规则从磁盘读进内存（带缓存）。

    为什么需要：工具可见性和命令内容级判定（见本文件其他函数）都以这份
    规则为准。缓存用 mtime+size 双因子判断文件变没变（原因见模块头的历史
    踩坑说明），没变就直接用上次的解析结果。

    返回：{"allow": [...], "deny": [...], "ask": [...]}（ask 是
    内容级强制审批规则）。文件不存在或读取出错时返回三个空列表
    （fail-open，出错就当没有任何规则，不拦正常使用）。缓存未命中（首次
    加载或文件刚改过）时会顺带跑一次遮蔽检测并告警。
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
                # ask 列表（内容级强制审批规则）
                "ask": [str(r) for r in (sec.get("ask") or []) if isinstance(r, (str,))],
            }
        _rules_cache["key"] = key
        _rules_cache["rules"] = rules
        # 遮蔽检测：只在缓存未命中时跑（首次加载/文件刚改过）
        for warning in detect_shadowed_command_rules(rules):
            logger.warning("权限规则遮蔽: %s", warning)
        return rules
    except Exception as e:
        logger.debug("load_tool_permission_rules fail-open: %s", e)
        return {"allow": [], "deny": [], "ask": []}


def tool_matches(rule: str, tool_name: str) -> bool:
    """判断单条规则是否匹配某个工具名。

    干什么：给一条规则和一个工具名，看它们对不对得上。

    为什么需要：可见性判定（is_tool_denied 等）要逐条规则做匹配。

    参数：
        rule: 规则字符串（如 "read_file" / "mcp__foo__*" / "mcp__foo"）。
        tool_name: 工具名。

    返回：匹配返回 True。三种匹配方式——名字完全相等；规则含通配符时
    按 fnmatch 通配（mcp__server__*）；规则不含通配符但工具名以
    "<规则>__" 开头时算整服务器命中（mcp__foo 匹配 mcp__foo__bar）。
    """
    if not rule:
        return False
    if rule == tool_name:
        return True
    if any(ch in rule for ch in "*?["):
        return fnmatch.fnmatchcase(tool_name, rule)
    return tool_name.startswith(rule + "__")


def is_tool_denied(tool_name: str, rules: Optional[Dict[str, List[str]]] = None) -> bool:
    """判断某个工具是否被 permissions 规则拒绝。

    干什么：先查 allow 列表再看 deny 列表——allow 命中就直接放行
    （豁免优先），否则 deny 命中才算被拒。

    为什么需要：这是工具可见性的主判断，供工具清单生成和 dispatch 防线调用。

    参数：
        tool_name: 工具名。
        rules: 可选，外部已加载的规则字典；不传就现场读 settings.json。

    返回：True 表示被 deny（且没有 allow 豁免）。
    """
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
# 内容级权限规则（Bash(...) / Terminal(...) 语法）
# ---------------------------------------------------------------------------

# 内容级规则的外壳格式（Bash / Terminal 是两种命令执行工具名，都认）
_CMD_RULE_RE = re.compile(r"^(?:Bash|Terminal)\((.*)\)$", re.IGNORECASE | re.DOTALL)
# 旧前缀语法（以 x:* 结尾）
_LEGACY_PREFIX_RE = re.compile(r"^(.+):\*$", re.DOTALL)

# 解析结果缓存：规则字符串 → ("exact", 内容) / ("prefix", 前缀) /
# ("wildcard", 模式) / None（不是内容级规则）。规则条目少且稳定，
# 进程内一个字典缓存就够。
_parsed_rule_cache: Dict[str, Optional[Tuple[str, str]]] = {}
# 通配模式转成的正则的缓存
_wildcard_regex_cache: Dict[str, "re.Pattern"] = {}


def _has_wildcards(content: str) -> bool:
    """判断内容里有没有未转义的 *（结尾的 :* 算旧前缀语法，不算通配）。

    参数：
        content: 规则括号里的内容字符串。

    返回：有未转义星号返回 True（按通配规则处理）。
    """
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
    """解析一条规则字符串，识别它是哪种内容级命令规则。

    干什么：看规则是不是 Bash(...)/Terminal(...) 形态，是的话再分成三种：
    逐字（exact）、旧前缀 x:*（prefix）、含未转义 * 的通配（wildcard）。

    为什么需要：内容级判定（command_rule_matches）需要先知道规则形态
    才知道怎么匹配。结果带缓存，同一条规则只解析一次。

    参数：
        rule: 规则字符串，如 "Bash(npm test)" 或 "read_file"。

    返回：(形态, 值) 二元组，形态为 "exact"/"prefix"/"wildcard" 之一；
    不是内容级命令规则（比如普通工具名）返回 None。
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
    """数一数模式串里未转义 * 的个数（转义对 \\* 整体跳过）。

    参数：
        p: 通配模式字符串。

    返回：未转义星号的数量。
    """
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
    """把通配模式编译成"整串匹配"的正则（带缓存）。

    干什么：把 Bash(git *) 这种通配写法翻译成正则，* 变 .*，\\* 保持
    字面星号。

    为什么需要：通配匹配每次都现场编译太浪费；模式少且稳定，缓存复用。

    参数：
        pattern: 规则括号里的通配模式。

    返回：编译好的正则对象（全串匹配）。特殊处理：尾部单独一个 " *" 且是
    唯一通配符时，让裸命令也匹配（git * 匹配光杆的 git）——对齐前缀语义。
    """
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

    # 尾部单独 " *" 且是唯一通配符 → 编译成可选参数组：git * 连光杆 git 也匹配
    # （对齐前缀语义）。必须在源模式层判断，不能对拼接结果切片——转义空格
    # 之类的形态会让切片错位。
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
# 命令形态归一化——剥掉命令头部的"伪装前缀"，防规则被绕过。
# ---------------------------------------------------------------------------

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
# env 赋值 token 里带引号/$/反引号时不剥——剥一半会造出更怪的形态，
# 保守起见整条命令都不归一化（见 _normalize_command_for_rules 的 aggressive 参数）
_SUSPICIOUS_ENV_TOKEN_RE = re.compile(r"[\"'`$]")
_WRAPPER_DURATION_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")


def _normalize_command_for_rules(command: str, *, aggressive: bool = False) -> str:
    """剥掉命令头部的环境变量赋值前缀和安全包装词，让内容级规则能对上号。

    干什么：把 ``FOO=1 nohup rm xxx`` 这种加了"前缀装饰"的命令还原成
    ``rm xxx``，再做规则匹配。

    为什么需要：防绕过。用户 deny 了 ``Bash(rm:*)``，如果直接拿原文匹配，
    ``FOO=1 rm xxx`` 就对不上规则、溜过去了。只剥**头部**；
    复合命令中段的 env 赋值（``a && B=1 rm``）是既有匹配语义没覆盖的形态，
    记为已知限制，这里不处理。

    参数：
        command: 原始命令字符串。
        aggressive: 是否激进剥离（见下）。

    剥离形态（循环剥直到剥不动为止）：
      - ``NAME=value`` 赋值 token。含引号/命令替换等可疑字符时按 aggressive
        分流：默认（allow 匹配用）整条停手不剥——剥一半会造出更怪的形态，
        保守返回原命令；aggressive=True（deny/ask 匹配用）照样剥——历史踩坑：
        不剥的话 ``FOO="x" rm -rf data`` 就绕过了
        ``deny Bash(rm:*)``。这是不对称语义：收紧方向（deny/ask）
        永不因剥不动而放行
      - ``env`` 本身 + 它的 ``-`` 开头选项及选项参数（``-i`` / ``-u NAME``）
        + 后续赋值 token
      - ``nohup``
      - ``timeout <时长>``（两个 token 一起剥）
      - ``nice`` / ``nice -n <数字>`` / ``nice -5``（POSIX 的隐式优先级写法）
      - ``stdbuf`` 及其后任意个 ``-`` 开头的选项 token

    返回：剥完前缀后的命令字符串（没得剥就原样返回）。
    """
    tokens = command.split()
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if _ENV_ASSIGN_RE.match(tok):
            if _SUSPICIOUS_ENV_TOKEN_RE.search(tok) and not aggressive:
                break  # 可疑 env token（含引号/$/反引号）：整条停止归一化，返回原命令
            i += 1
            continue
        if tok == "env":
            i += 1
            # 连 env 的选项和选项参数一起吃掉（-i / -u NAME / --ignore-environment 等）
            while i < n and tokens[i].startswith("-") and tokens[i] != "-":
                if tokens[i] in ("-u", "--unset") and i + 1 < n:
                    i += 2
                else:
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
            elif i + 1 < n and re.match(r"^-+\d+$", tokens[i + 1]):
                i += 2  # nice -5（POSIX 的隐式优先级写法）
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
    """判断一条内容级规则是否匹配某条命令（exact/prefix/wildcard 三种形态各用各的比法）。

    参数：
        rule: 规则字符串（如 "Bash(npm test)"）。
        command: 待匹配的命令。

    返回：匹配返回 True；规则不是内容级命令规则或命令为空返回 False。
    """
    parsed = parse_command_rule(rule)
    if parsed is None or not command:
        return False
    kind, value = parsed
    if kind == "exact":
        return command == value
    if kind == "prefix":
        # 词边界前缀：npm 匹配 "npm" / "npm test"，但不匹配 "npmx"（防误伤）
        return command == value or command.startswith(value + " ")
    return _wildcard_regex(value).match(command) is not None


def check_command_rules(command: str, rules: Optional[Dict[str, List[str]]] = None) -> str:
    """内容级规则判定的主入口，PermissionChecker.check 调它。

    干什么：拿一条命令去对照 allow/deny/ask 三张规则表，给出最终裁决。

    为什么需要：这是"用户显式配置压过一切算法"的实现点——deny 任何模式
    都拒、ask 强制审批，见模块头说明。

    参数：
        command: 待判定的命令字符串。
        rules: 可选，外部已加载的规则；不传就现场读 settings.json。

    返回："deny" / "ask" / "allow" / "none" 之一，优先级 deny > ask > allow。
    工具可见性条目（read_file 这类不带括号的）不参与——解析时返回 None。

    几条重要的匹配规则（历史演进攒下的取舍）：
    - 匹配前先剥掉 env 前缀/安全包装词——FOO=bar rm xxx 绕不过
      deny(rm)。
    - 剥离是不对称的——deny/ask 用激进剥离（可疑 env token 也剥），
      allow 只认保守剥离（防 ``FOO=$(evil) cmd`` 被激进剥完后误命中 allow）。
    - AST 解析成功时 deny/ask **逐段**匹配（复合命令后半段命中
      即命中——只收紧不放宽）；allow 保持整串匹配且在复合命令上不生效
      （"allow 必须覆盖全部段"语义）；AST 解析失败时 deny/ask
      整串仍生效，但 allow 整串命中也不放行（fail-safe：解析
      不了就不放宽，宁可多问一次）。
    """
    if rules is None:
        rules = load_tool_permission_rules()
    # 归一化出两个版本——deny/ask 用激进剥离（可疑 env token 也剥），
    # allow 用保守剥离（碰到可疑 env token 就停手）。不对称语义。
    cmd_c = _normalize_command_for_rules(command)
    cmd_a = _normalize_command_for_rules(command, aggressive=True)

    def _match_pair(cmd_conservative: str, cmd_aggressive: str) -> str:
        """对一条命令串做三级判定（deny/ask/allow，都不是则 none）。

        参数：
            cmd_conservative: 保守剥离后的命令（allow 用）。
            cmd_aggressive: 激进剥离后的命令（deny/ask 用）。

        返回："deny" / "ask" / "allow" / "none"。
        规则：deny/ask 拿激进形态去匹配（收紧方向不因剥不动而放行）；
        allow 只拿保守形态匹配（放宽方向不因激进剥离而误放行——
        ``FOO=$(evil) git status`` 不能命中 allow(git status)）。
        """
        for r in (rules.get("deny") or []):
            if command_rule_matches(r, cmd_aggressive):
                return "deny"
        for r in (rules.get("ask") or []):
            if command_rule_matches(r, cmd_aggressive):
                return "ask"
        for r in (rules.get("allow") or []):
            if command_rule_matches(r, cmd_conservative):
                return "allow"
        return "none"

    whole = _match_pair(cmd_c, cmd_a)
    # AST 逐段信息；AST 解析失败时 allow 整串命中也不再放行（fail-safe）
    from agent.bash_ast import parse_info
    info = parse_info(cmd_c)
    if whole == "allow" and (
        info is None
        or (len(info["segments"]) > 1 and not info["has_substitution"])
    ):
        # 顶层复合命令（&&/;/| 且无命令替换）上 allow 整串命中也不放宽——
        # 否则前缀规则会盖到 && 后面的段（"allow 须覆盖全部段"）。
        # AST 解析失败（info is None）：没法证明"allow 覆盖了全部段"，同样
        # 不放宽（fail-safe，宁可升审批也不误放行；历史踩坑：
        # 此前 fail-open 直接放行是一个绕过面）。
        # 含命令替换时（rm -rf x $(gen)）维持整串 allow 的现状（fail-open，
        # 不为了收紧 allow 引入新的拒绝面）。
        whole = "none"
    if whole in ("deny", "ask", "allow"):
        return whole

    if info is None:
        return "none"
    seg_results = []
    for tokens in info["segments"]:
        seg_c = _normalize_command_for_rules(" ".join(tokens))
        seg_a = _normalize_command_for_rules(" ".join(tokens), aggressive=True)
        seg_results.append(_match_pair(seg_c, seg_a))
    if "deny" in seg_results:
        return "deny"
    if "ask" in seg_results:
        return "ask"
    return "none"


def detect_shadowed_command_rules(rules: Dict[str, List[str]]) -> List[str]:
    """遮蔽检测。

    干什么：找出永远不会生效的 allow 规则并给出告警文案。

    为什么需要：如果用户写了裸的 "Bash"/"Terminal" 整级 deny 或 ask，
    它的优先级会压过所有内容级 allow——后者写得再多也永远轮不到生效
    （被"遮蔽"）。配置错了却不吭声，用户会误以为 allow 生效了。

    参数：
        rules: 已加载的规则字典（allow/deny/ask 三列表）。

    返回：告警消息列表（每条对应一个被遮蔽的 allow 规则）；没有遮蔽
    返回空列表。本项目只有单一 settings.json（没有多配置来源要跨源
    比对），所以只做同源判定。
    """
    warnings: List[str] = []
    tool_wide = {"Bash", "Terminal"}
    deny_wide = any(r.strip() in tool_wide for r in (rules.get("deny") or []))
    ask_wide = any(r.strip() in tool_wide for r in (rules.get("ask") or []))
    if not deny_wide and not ask_wide:
        return warnings
    for r in (rules.get("allow") or []):
        if parse_command_rule(r) is None:
            continue  # 只检查内容级 allow（整级 allow 本身不含括号，不在遮蔽语义内）
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
