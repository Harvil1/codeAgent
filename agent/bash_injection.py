"""Bash 命令注入面检测（R16 #1）。

对齐 CCB BashTool/bashSecurity.ts 的「注入面模式」分桶（BASH_SECURITY_CHECK_IDS
1-24 的核心子集），适配 OmniMate：**命中 → 升审批**（不进黑名单硬拒）。
CC 同款语义：这些形态本身不一定是攻击，但会让"所见命令"与"实际执行"不一致，
必须让用户看到原文后手动批准。

与 CC 的差异（如实记录）：
- 无 shell-quote / tree-sitter 解析器 → 跳过 validateMalformedTokenInjection
  （依赖 token 流的未闭合定界符检测）和 hasShellQuoteSingleQuoteBug
  （shell-quote 特有的单引号反斜杠 bug）
- 不拦普通重定向 < / >（CC validateRedirections 对任何重定向 ask；OmniMate
  只读通道已把含重定向命令排除出快速通道，写目标另走 safe_path/check_path
  白名单层——全套搬过来对 OmniMate 场景过噪）
- quoted heredoc 剥离是简化版（CC 的 isSafeHeredoc/stripSafeHeredocSubstitutions
  连 $(cat <<'EOF'...) 整块剥除；本实现按行剥体，起始行保留 << 前缀，
  $( 形态仍会命中 $() 检查 → 多问一次审批，方向保守）
- obfuscated flags 只移植正则子集 + 「引号内容以 dash 开头」扫描器，
  CC 的引号链式拼接状态机（"-""exec 等形态）未全量移植

检查实现分层（对齐 CC extractQuotedContent 的三种视图）：
- raw            原始命令
- with_dq        剥单引号内容（双引号内容保留——双引号内 $() ` 仍展开）
- fully          剥全部引号内容（单双引号内都不展开的形态用这个视图）
- keepq          剥引号内容但保留引号字符（检测引号邻接，词中 # 用）
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# 视图抽取（对齐 extractQuotedContent）
# ---------------------------------------------------------------------------

def _extract_quoted_content(command: str, is_jq: bool = False):
    """返回 (with_dq, fully, keepq) 三种引号视图。

    is_jq=True 时双引号字符保留在 with_dq（对齐 CC 的 jq 特例——jq 过滤器
    里的引号元字符形态需要被引号元字符检查看到）。
    """
    with_dq: list = []
    fully: list = []
    keepq: list = []
    in_sq = in_dq = False
    escaped = False
    for ch in command:
        if escaped:
            escaped = False
            if not in_sq:
                with_dq.append(ch)
                if not in_dq:
                    fully.append(ch)
                    keepq.append(ch)
            continue
        if ch == "\\" and not in_sq:
            escaped = True
            if not in_sq:
                with_dq.append(ch)
                if not in_dq:
                    fully.append(ch)
                    keepq.append(ch)
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            keepq.append(ch)
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            keepq.append(ch)
            if not is_jq:
                continue
        if not in_sq:
            with_dq.append(ch)
            if not in_dq:
                fully.append(ch)
                keepq.append(ch)
    return "".join(with_dq), "".join(fully), "".join(keepq)


# ---------------------------------------------------------------------------
# quoted heredoc 剥离（简化版）
# ---------------------------------------------------------------------------

_HEREDOC_OPEN_RE = re.compile(
    r"<<-?[ \t]*(?:'([A-Za-z_]\w*)'|\\([A-Za-z_]\w*))[ \t]*$"
)


def _strip_quoted_heredocs(command: str) -> str:
    """剥除 quoted/escaped 定界符 heredoc 的正文（正文是字面量，不展开）。

    未加引号的 heredoc（<<EOF）正文会展开 $()/反引号，必须保留给检查器看。
    找不到闭合定界符时整段保留（保守方向）。
    """
    lines = command.split("\n")
    out: list = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _HEREDOC_OPEN_RE.search(line)
        if m:
            delim = m.group(1) or m.group(2)
            dash = "<<-" in line[: m.start() + 3]
            j = i + 1
            closed = False
            while j < n:
                cand = re.sub(r"^\t*", "", lines[j]) if dash else lines[j]
                if re.fullmatch(re.escape(delim) + r"[ \t]*", cand):
                    closed = True
                    break
                j += 1
            if closed:
                # 保留 << 之前的命令部分（如 "cat " / "echo $("）
                prefix = line[: m.start()].rstrip()
                out.append(prefix if prefix else "true")
                i = j + 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 各类检查（预编译正则）
# ---------------------------------------------------------------------------

# 控制字符（0x00-0x08/0x0B/0x0C/0x0E-0x1F/0x7F；bash 静默丢弃、混淆检查器）
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode 空白（解析器与 bash 分词不一致）
_UNICODE_WS_RE = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
)
# 命令替换/进程替换/zsh 展开（跑在 with_dq 视图：单引号内不展开）
_SUBSTITUTION_PATTERNS = [
    (re.compile(r"<\("), "进程替换 <()"),
    (re.compile(r">\("), "进程替换 >()"),
    (re.compile(r"=\("), "zsh 进程替换 =()"),
    # zsh =cmd 词首展开（=curl → /usr/bin/curl，可绕前缀规则；不匹配 VAR=val）
    (re.compile(r"(?:^|[\s;&|])=[A-Za-z_]"), "zsh =cmd 展开"),
    (re.compile(r"\$\("), "$() 命令替换"),
    (re.compile(r"\$\{"), "${} 参数替换"),
    (re.compile(r"\$\["), "$[] 算术展开"),
    (re.compile(r"~\["), "zsh ~[] 参数展开"),
    (re.compile(r"\(e:"), "zsh glob 限定符 (e:"),
    (re.compile(r"\(\+"), "zsh glob 限定符 (+"),
    (re.compile(r"\}\s*always\s*\{"), "zsh always 块"),
    (re.compile(r"<#"), "PowerShell 注释语法 <#"),
]
# IFS 注入
_IFS_RE = re.compile(r"\$IFS|\$\{[^}]*IFS")
# /proc/*/environ（环境变量泄露）
_PROC_ENVIRON_RE = re.compile(r"/proc/.*/environ")
# /dev/tcp|udp 网络伪设备
_NETWORK_DEVICE_RE = re.compile(r"""/dev/(tcp|udp)/[^/\s"'`$]+/\d+""", re.IGNORECASE)
# 引号参数里藏 shell 元字符（"a;b" 作为参数）
_QUOTED_METACHAR_RE = re.compile(r"""(?:^|\s)["'][^"']*[;&][^"']*["'](?:\s|$)""")
# 重定向/管道上下文里的变量
_DANGEROUS_VAR_RE = re.compile(r"[<>|]\s*\$[A-Za-z_]|\$[A-Za-z_][A-Za-z0-9_]*\s*[|<>]")
# 换行分隔多命令（\<换行> 续行豁免：前导是 空白+反斜杠）
_NEWLINE_CMD_RE = re.compile(r"(?<![\s]\\)[\n\r]\s*\S")
# 词中 #（bash 字面量 vs 注释剥离解析器的差异；排除 bash 求长语法 ${#var}）
_MID_WORD_HASH_RE = re.compile(r"\S(?<!\$\{)#")
# ANSI-C / locale 引号（可编码任意字符）
_ANSI_C_QUOTE_RE = re.compile(r"\$'[^']*'")
_LOCALE_QUOTE_RE = re.compile(r'\$"[^"]*"')
_EMPTY_SPECIAL_QUOTE_DASH_RE = re.compile(r"\$['\"]{2}\s*-")
_EMPTY_QUOTE_DASH_RE = re.compile(r"""(?:^|\s)(?:''|"")+\s*-""")
_EMPTY_PAIR_QUOTED_DASH_RE = re.compile(r"""(?:""|'')+['"]-""")
_TRIPLE_QUOTE_START_RE = re.compile(r"""(?:^|\s)['"]{3,}""")
_QUOTE_DASH_FULLY_RE = re.compile(r"""\s['"`]-""")
_DOUBLE_QUOTE_DASH_FULLY_RE = re.compile(r"""['"`]{2}-""")
# 引号内容以 dash+flag 字符开头（quoted flag 混淆：" -f" / '--flag'）
_QUOTED_FLAG_CONTENT_RE = re.compile(r"^-+[A-Za-z0-9$`]")
# zsh 危险 builtin（zmodload 系）
_ZSH_DANGEROUS_COMMANDS = frozenset({
    "zmodload", "emulate",
    "sysopen", "sysread", "syswrite", "sysseek",
    "zpty", "ztcp", "zsocket", "mapfile",
    "zf_rm", "zf_mv", "zf_ln", "zf_chmod", "zf_chown",
    "zf_mkdir", "zf_rmdir", "zf_chgrp",
})
_ZSH_PRECOMMAND_MODIFIERS = frozenset({"command", "builtin", "noglob", "nocorrect"})
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_FC_E_RE = re.compile(r"\s-\S*e")
# jq 危险面
_JQ_SYSTEM_RE = re.compile(r"\bsystem\s*\(")
_JQ_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:-f\b|--from-file|--rawfile|--slurpfile|-L\b|--library-path)"
)
# 操作符集合（反斜杠转义操作符检测用）
_SHELL_OPERATORS = frozenset(";|&<>")


def _has_unescaped_char(content: str, char: str) -> bool:
    """内容里是否有未转义的指定单字符（跳过 \\x 转义对）。"""
    i = 0
    while i < len(content):
        if content[i] == "\\" and i + 1 < len(content):
            i += 2
            continue
        if content[i] == char:
            return True
        i += 1
    return False


def _scan_quotes(command: str):
    """通用引号状态扫描，yield (idx, char, in_sq, in_dq, escaped)。

    语义对齐 CC：反斜杠在单引号内是字面量；先处理反斜杠再处理引号翻转。
    """
    in_sq = in_dq = False
    escaped = False
    for i, ch in enumerate(command):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and not in_sq:
            escaped = True
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            continue
        yield i, ch, in_sq, in_dq


def _has_backslash_escaped_whitespace(command: str) -> bool:
    """引号外的 反斜杠+空格/tab（bash 单 token vs 解析器双 token）。"""
    in_sq = in_dq = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_sq:
            if not in_dq and i + 1 < n and command[i + 1] in (" ", "\t"):
                return True
            i += 2
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "'" and not in_dq:
            in_sq = not in_sq
        i += 1
    return False


def _has_backslash_escaped_operator(command: str) -> bool:
    r"""引号外的 \<operator>（\; \| \& \< \>，隐藏命令结构）。

    已知误报：find . -exec cmd {} \; —— 多问一次审批，可接受。
    """
    in_sq = in_dq = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_sq:
            if not in_dq and i + 1 < n and command[i + 1] in _SHELL_OPERATORS:
                return True
            i += 2
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        i += 1
    return False


def _check_carriage_return_outside_dq(command: str) -> bool:
    """CR 出现在双引号外（bash IFS 不含 CR，分词差异面）。"""
    if "\r" not in command:
        return False
    for _i, ch, _sq, in_dq in _scan_quotes(command):
        if ch == "\r" and not in_dq:
            return True
    return False


def _check_comment_quote_desync(command: str) -> bool:
    """未引用 # 注释里含引号字符（让引号跟踪器失步）。"""
    in_sq = in_dq = False
    escaped = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if in_sq:
            if ch == "'":
                in_sq = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if in_dq:
            if ch == '"':
                in_dq = False
            i += 1
            continue
        if ch == "'":
            in_sq = True
            i += 1
            continue
        if ch == '"':
            in_dq = True
            i += 1
            continue
        if ch == "#":
            line_end = command.find("\n", i)
            comment = command[i + 1: n if line_end == -1 else line_end]
            if re.search(r"['\"]", comment):
                return True
            if line_end == -1:
                break
            i = line_end
        i += 1
    return False


def _check_quoted_newline_hash(command: str) -> bool:
    """引号内换行 + 下一行 # 开头（对基于行的检查隐藏参数）。"""
    if "\n" not in command or "#" not in command:
        return False
    for i, ch, in_sq, in_dq in _scan_quotes(command):
        if ch == "\n" and (in_sq or in_dq):
            next_nl = command.find("\n", i + 1)
            line_end = len(command) if next_nl == -1 else next_nl
            if command[i + 1: line_end].lstrip().startswith("#"):
                return True
    return False


def _is_escaped_at(content: str, pos: int) -> bool:
    backslashes = 0
    i = pos - 1
    while i >= 0 and content[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def _check_brace_expansion(fully: str) -> bool:
    """未引用的花括号展开（{a,b} / {1..5}——bash 展开成多词）。

    fully 视图已剥全部引号内容（引号内不展开），只查裸花括号。
    含 CC 的闭括号盈余防御（引号剥离造成计数失配 → 直接判可疑）。
    """
    opens = closes = 0
    for i, ch in enumerate(fully):
        if ch == "{" and not _is_escaped_at(fully, i):
            opens += 1
        elif ch == "}" and not _is_escaped_at(fully, i):
            closes += 1
    if opens > 0 and closes > opens:
        return True

    for i, ch in enumerate(fully):
        if ch != "{" or _is_escaped_at(fully, i):
            continue
        # 找配对 }（嵌套深度跟踪）
        depth = 1
        matching = -1
        for j in range(i + 1, len(fully)):
            cj = fully[j]
            if cj == "{" and not _is_escaped_at(fully, j):
                depth += 1
            elif cj == "}" and not _is_escaped_at(fully, j):
                depth -= 1
                if depth == 0:
                    matching = j
                    break
        if matching == -1:
            continue
        # 顶层 , 或 .. 触发展开
        inner_depth = 0
        for k in range(i + 1, matching):
            ck = fully[k]
            if ck == "{" and not _is_escaped_at(fully, k):
                inner_depth += 1
            elif ck == "}" and not _is_escaped_at(fully, k):
                inner_depth -= 1
            elif inner_depth == 0:
                if ck == "," or (ck == "." and k + 1 < matching and fully[k + 1] == "."):
                    return True
    return False


def _check_quoted_flag_obfuscation(command: str, base: str) -> Optional[str]:
    """引号混写的 flag（"-f" / '--flag' / $'..' / 空引号对 + dash）。

    echo 简单命令豁免（对齐 CC echo + 无操作符例外——echo 的 ANSI-C 等
    形态只影响输出内容，无执行面）。
    """
    if base == "echo" and not re.search(r"[|&;]", command):
        return None
    if _ANSI_C_QUOTE_RE.search(command):
        return "ANSI-C 引号 $'..'（可编码任意字符）"
    if _LOCALE_QUOTE_RE.search(command):
        return 'locale 引号 $".."'
    if _EMPTY_SPECIAL_QUOTE_DASH_RE.search(command):
        return "空特殊引号 + dash（$''-）"
    if _EMPTY_QUOTE_DASH_RE.search(command):
        return "空引号对 + dash（''- / \"\"-）"
    if _EMPTY_PAIR_QUOTED_DASH_RE.search(command):
        return "空引号对邻接引号 dash"
    if _TRIPLE_QUOTE_START_RE.search(command):
        return "词首连续 3+ 引号字符"

    # 引号内容以 dash+flag 字符开头（空白后跟引号，内容 ^-+[a-zA-Z0-9$`]）
    n = len(command)
    in_sq = in_dq = False
    escaped = False
    i = 0
    while i < n - 1:
        c = command[i]
        nxt = command[i + 1]
        if escaped:
            escaped = False
            i += 1
            continue
        if c == "\\" and not in_sq:
            escaped = True
            i += 2
            continue
        if c == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if c == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if in_sq or in_dq:
            i += 1
            continue
        if c.isspace() and nxt in ("'", '"', "`"):
            j = i + 2
            content = []
            while j < n and command[j] != nxt:
                content.append(command[j])
                j += 1
            if j < n and _QUOTED_FLAG_CONTENT_RE.match("".join(content)):
                return "引号内 flag 形态（\"-x\" / '--flag'）"
        i += 1
    return None


def _zsh_base_command(command: str) -> str:
    """剥掉 env 赋值和 zsh precommand 修饰符后的首命令。"""
    for token in command.strip().split():
        if _ENV_ASSIGN_RE.match(token):
            continue
        if token in _ZSH_PRECOMMAND_MODIFIERS:
            continue
        return token
    return ""


def check_injection_surface(command: str) -> Optional[str]:
    """检查命令是否含注入面形态。命中返回中文原因（用于审批提示），否则 None。

    这是 R16 #1 的主入口，由 PermissionChecker.check 在闸门 1 之后、
    只读快速通道之前调用——命中走审批而非硬拒（对齐 CC ask 语义）。
    """
    if not command or not command.strip():
        return None

    # 控制字符最先（防后续所有正则被绕过）
    if _CONTROL_RE.search(command):
        return "含非打印控制字符"
    if _UNICODE_WS_RE.search(command):
        return "含 Unicode 空白字符"

    # 未完成片段（tab 开头 / flag 开头 / 操作符续行开头）
    stripped = command.strip()
    if re.match(r"^\s*\t", command):
        return "以 tab 开头的未完成片段"
    if stripped.startswith("-"):
        return "以 flag 开头的未完成片段"
    if re.match(r"^\s*(?:&&|\|\||;|>>?|<)", command):
        return "以操作符开头的续行片段"

    # quoted heredoc 正文按字面量剥除后再做模式检查
    heredoc_stripped = _strip_quoted_heredocs(command)
    base = command.split(" ")[0] or ""
    with_dq, fully, _keepq = _extract_quoted_content(
        heredoc_stripped, is_jq=(base == "jq")
    )
    keepq = _extract_quoted_content(command)[2]  # 词中 # 用原始命令的视图

    # jq 危险面（system() 执行任意命令；-f 族读文件进变量）
    if base == "jq":
        if _JQ_SYSTEM_RE.search(command):
            return "jq system() 函数（执行任意命令）"
        if _JQ_FLAGS_RE.search(command[3:].strip()):
            return "jq 危险 flag（-f/--from-file/--rawfile/--slurpfile/-L）"

    # 引号混写 flag
    obf = _check_quoted_flag_obfuscation(command, base)
    if obf:
        return obf

    # 引号参数里藏元字符
    if _QUOTED_METACHAR_RE.search(with_dq):
        return "引号参数内含 shell 元字符（; 或 &）"

    # 重定向/管道上下文里的变量
    if _DANGEROUS_VAR_RE.search(fully):
        return "重定向/管道上下文中的变量"

    # 注释引号失步 / 引号内换行 + # 行
    if _check_comment_quote_desync(command):
        return "# 注释内含引号字符（引号跟踪失步面）"
    if _check_quoted_newline_hash(command):
        return "引号内换行且下一行 # 开头"

    # CR 在双引号外（分词差异）
    if _check_carriage_return_outside_dq(command):
        return "双引号外的回车符（\\r 分词差异）"

    # 换行分隔多命令（\<换行> 续行豁免）
    if _NEWLINE_CMD_RE.search(fully):
        return "换行分隔多条命令"

    # IFS 注入
    if _IFS_RE.search(command):
        return "IFS 变量用法"

    # /proc/*/environ
    if _PROC_ENVIRON_RE.search(command):
        return "访问 /proc/*/environ（环境变量泄露面）"

    # 命令替换 / 进程替换 / zsh 展开（with_dq 视图：单引号内不展开）
    for pattern, desc in _SUBSTITUTION_PATTERNS:
        if pattern.search(with_dq):
            return desc

    # 未转义反引号（with_dq 视图）
    if _has_unescaped_char(with_dq, "`"):
        return "反引号命令替换"

    # 反斜杠转义空白 / 操作符
    if _has_backslash_escaped_whitespace(command):
        return "反斜杠转义空白字符（分词差异）"
    if _has_backslash_escaped_operator(command):
        return "反斜杠转义 shell 操作符（隐藏命令结构）"

    # 词中 #（keepq 视图：保留引号字符以捕捉 'x'# 邻接形态）
    if _MID_WORD_HASH_RE.search(keepq):
        return "词中 # （注释剥离差异面）"

    # 花括号展开（fully 视图）
    if _check_brace_expansion(fully):
        return "花括号展开（{a,b} 改变参数形态）"

    # zsh 危险 builtin
    zbase = _zsh_base_command(command)
    if zbase in _ZSH_DANGEROUS_COMMANDS:
        return f"zsh 危险 builtin（{zbase}，可绕过二进制检查）"
    if zbase == "fc" and _FC_E_RE.search(command.strip()):
        return "fc -e（经编辑器执行任意命令）"

    # /dev/tcp|udp 网络伪设备
    if _NETWORK_DEVICE_RE.search(fully):
        return "/dev/tcp|udp 网络伪设备（无网络工具外泄面）"

    return None
