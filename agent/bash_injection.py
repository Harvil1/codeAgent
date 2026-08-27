"""Bash 命令「注入面」检查。

这个文件是干嘛的：检查一条要执行的 bash 命令里，有没有「看起来无害、
实际执行时会长出别的东西」的写法。好比收快递时发现箱子被重新封过——
不一定有毒，但必须当面拆开看一眼才敢签收。

关键语义：命中不是直接拒绝，而是**升级成让用户手动审批**。为什么不打死？
因为这些写法本身不一定是攻击，只是会让「你看到的命令」和「实际跑的命令」
不一致——所以至少要让人看到原文、亲手点批准。

实现上的取舍（都是有原因的）：
- 我们没有 shell-quote / tree-sitter 这类解析器，所以跳过两个检查：
  要靠 token 流才能发现的未闭合引号、shell-quote 库特有的单引号反斜杠 bug
- 普通重定向 < / > 不拦：只读快速通道已经把带重定向的命令排除在免审
  之外了，写入目标另有 safe_path/check_path 白名单把关——对任何重定向
  都问一次对本场景太吵（动不动就弹审批）
- 「带引号的 heredoc」剥除是简化版：只按行剥正文，起始行还留着 <<
  前缀，所以 $(cat <<'EOF'...) 形态仍会被 $() 检查命中——结果是多问
  一次审批，方向偏保守（宁可多问不漏问）
- 混淆 flag 检查只做了正则子集 + 「引号内容以横杠开头」的扫描器；
  引号链式拼接状态机（识别 "-""exec 这类拼法）没有实现

检查用的「视图」分层（三种看法）：
- raw            命令原文
- with_dq        去掉单引号内容后的视图（双引号内容保留——因为双引号里
                  的 $() 和反引号依然会被 shell 展开，得让检查器看见）
- fully          单双引号内容都去掉（引号里不展开的形态用这个视图查）
- keepq          去掉引号内容但保留引号字符本身（查引号挨着 # 的形态用）
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# 视图抽取
# ---------------------------------------------------------------------------

def _extract_quoted_content(command: str, is_jq: bool = False):
    """生成三种「引号视图」，返回 (with_dq, fully, keepq) 三个字符串，把命令按引号状态拆出三种看法供不同检查器选用。

    bash 里单引号内容完全不展开、双引号内容部分展开，检查器必须
    区分「引号里的字符」和「裸字符」，否则会被引号骗过。

    参数：
        command —— 要分析的命令原文
        is_jq   —— 命令是不是 jq。True 时双引号字符会保留在 with_dq 视图里
                   （jq 特例：jq 过滤器里的引号元字符形态需要
                   被引号元字符检查看到）

    返回：三个字符串的元组 (with_dq, fully, keepq)。
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
    r"""把「带引号/转义定界符」的 heredoc 正文从命令里拿掉，返回剩余部分。

    定界符加了引号（<<'EOF'）或反斜杠（<<\EOF）时，正文是纯字面量、
    不会被 shell 展开——里面的 $() 之类的可疑形态是纸老虎，
    拿掉可以少误报。没加引号的 heredoc（<<EOF）正文会展开 $() 和反引号，
    必须留给检查器看。找不到闭合定界符时整段保留（宁可多查不漏查）。
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
                # << 前面如果有真命令（如 "cat "）要留着，不能整行丢
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

# 控制字符（0x00-0x08/0x0B/0x0C/0x0E-0x1F/0x7F）——bash 会静默丢弃它们，
# 可能用来迷惑基于文本的检查器
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode 空白——不同解析器对它们的分词结果可能不一致
_UNICODE_WS_RE = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
)
# 命令替换/进程替换/zsh 展开形态（在 with_dq 视图上查：单引号内不展开）
_SUBSTITUTION_PATTERNS = [
    (re.compile(r"<\("), "进程替换 <()"),
    (re.compile(r">\("), "进程替换 >()"),
    (re.compile(r"=\("), "zsh 进程替换 =()"),
    # zsh 的 =cmd 写法在词首会把命令名换成完整路径（=curl → /usr/bin/curl），
    # 可以绕过按命令名的前缀规则；正则设计成不匹配 VAR=val 赋值
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
# IFS 注入（改分隔符变量让字符串被重新切分执行）
_IFS_RE = re.compile(r"\$IFS|\$\{[^}]*IFS")
# 读 /proc/*/environ（偷看别的进程环境变量）
_PROC_ENVIRON_RE = re.compile(r"/proc/.*/environ")
# /dev/tcp|udp 网络伪设备（不用 curl 也能发起网络连接）
_NETWORK_DEVICE_RE = re.compile(r"""/dev/(tcp|udp)/[^/\s"'`$]+/\d+""", re.IGNORECASE)
# 引号参数里藏 shell 元字符（例如把 "a;b" 当参数传，展开后变两条命令）
_QUOTED_METACHAR_RE = re.compile(r"""(?:^|\s)["'][^"']*[;&][^"']*["'](?:\s|$)""")
# 重定向/管道旁边出现变量（重定向目标可能是变量，运行前看不出来）
_DANGEROUS_VAR_RE = re.compile(r"[<>|]\s*\$[A-Za-z_]|\$[A-Za-z_][A-Za-z0-9_]*\s*[|<>]")
# 换行分隔的多条命令（豁免「空白+反斜杠+换行」的续行写法）
_NEWLINE_CMD_RE = re.compile(r"(?<![\s]\\)[\n\r]\s*\S")
# 词中间出现 #（bash 当字面量、但会剥注释的解析器当注释起点，两边理解不一致；
# 排除 bash 求字符串长度的 ${#var} 语法）
_MID_WORD_HASH_RE = re.compile(r"\S(?<!\$\{)#")
# ANSI-C / locale 引号（$'..' 里可以编码任意字符，混淆检查器）
_ANSI_C_QUOTE_RE = re.compile(r"\$'[^']*'")
_LOCALE_QUOTE_RE = re.compile(r'\$"[^"]*"')
_EMPTY_SPECIAL_QUOTE_DASH_RE = re.compile(r"\$['\"]{2}\s*-")
_EMPTY_QUOTE_DASH_RE = re.compile(r"""(?:^|\s)(?:''|"")+\s*-""")
_EMPTY_PAIR_QUOTED_DASH_RE = re.compile(r"""(?:""|'')+['"]-""")
_TRIPLE_QUOTE_START_RE = re.compile(r"""(?:^|\s)['"]{3,}""")
_QUOTE_DASH_FULLY_RE = re.compile(r"""\s['"`]-""")
_DOUBLE_QUOTE_DASH_FULLY_RE = re.compile(r"""['"`]{2}-""")
# 引号内容以 dash+flag 字符开头（把 flag 藏进引号里混淆：" -f" / '--flag'）
_QUOTED_FLAG_CONTENT_RE = re.compile(r"^-+[A-Za-z0-9$`]")
# zsh 危险 builtin（zmodload 一族，能加载模块绕过二进制名检查）
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
# jq 的危险面
_JQ_SYSTEM_RE = re.compile(r"\bsystem\s*\(")
_JQ_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:-f\b|--from-file|--rawfile|--slurpfile|-L\b|--library-path)"
)
# 操作符集合（查「反斜杠转义操作符」时用）
_SHELL_OPERATORS = frozenset(";|&<>")


def _has_unescaped_char(content: str, char: str) -> bool:
    r"""看内容里有没有出现「未被反斜杠转义」的指定单字符（转义对 \\x 要整对跳过，否则会把 `\`` 误当成真反引号）。

    参数：
        content —— 要查的字符串
        char    —— 要找的单字符（如反引号）

    返回：True 表示存在未转义的目标字符。
    """
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
    """逐字符扫描命令，边扫边维护引号状态，逐个产出 (idx, char, in_sq, in_dq)——公共生成器，多个检查器共用（各写一套引号状态机容易出错）。

    参数：
        command —— 要扫描的命令

    产出：每个字符一个元组（下标、字符、是否在单引号内、是否在双引号内）。
    语义要点：反斜杠在单引号内是字面量；先处理反斜杠再处理引号开合。
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
    r"""检查引号外有没有「反斜杠+空格/tab」的写法——`\ ` 在 bash 里把空格粘进一个词，基于文本的解析器可能把它切成两个词，两边分词不一致就可能被钻空子。

    参数：
        command —— 命令原文

    返回：True 表示存在这种形态。
    """
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
    r"""检查引号外有没有「反斜杠+操作符」（\; \| \& \< \>）的写法——转义的操作符能把命令的真实结构藏起来，让检查器看不出这里其实有分隔/管道。

    参数：
        command —— 命令原文

    返回：True 表示存在这种形态。
    已知误报：find . -exec cmd {} \; 这类合法写法也会命中——代价是
    多问一次审批，可接受。
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
    """检查回车符（\r）是否出现在双引号外——bash 的默认分隔符列表里没有 \r（带 \r 的内容并进词里），很多解析器却按换行符家族切分，两边分词结果不同。

    参数：
        command —— 命令原文

    返回：True 表示双引号外存在 \r。
    """
    if "\r" not in command:
        return False
    for _i, ch, _sq, in_dq in _scan_quotes(command):
        if ch == "\r" and not in_dq:
            return True
    return False


def _check_comment_quote_desync(command: str) -> bool:
    """检查未加引号的 # 注释内容里是否藏了引号字符——注释里的引号会被简单的「跟踪引号开合」解析器当真，之后所有引号状态全部错位（失步），后续检查全被带偏。

    参数：
        command —— 命令原文

    返回：True 表示存在这种形态。
    """
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
    """检查「引号内的换行 + 下一行以 # 开头」的组合——按行逐行检查的工具会把 # 行当注释跳过，但这段其实在引号里、是真参数。

    参数：
        command —— 命令原文

    返回：True 表示存在这种形态。
    """
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
    """判断 content[pos] 处的字符是否被反斜杠转义（往前数连续反斜杠的奇偶）。"""
    backslashes = 0
    i = pos - 1
    while i >= 0 and content[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def _check_brace_expansion(fully: str) -> bool:
    """检查是否存在未加引号的花括号展开（{a,b} / {1..5}）——bash 会把 {a,b} 展开成多个词，检查器看到的参数形态和实际执行的不一样。fully 视图已经把引号内容剥掉（引号内不展开），这里查到的都是裸花括号。

    参数：
        fully —— 剥掉全部引号内容后的命令视图

    返回：True 表示存在花括号展开形态。
    另有闭括号盈余防御：右括号比左括号多说明引号剥离造成了计数
    错位，直接判可疑。
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
        # 找配对的 }，用嵌套深度跟踪
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
        # 括号对最外层出现 , 或 .. 就是会触发展开的形态
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
    """检查「用引号混写把 flag 藏起来」的形态（"-f" / '--flag' / $'..' / 空引号对+横杠）——命令行参数解析器通常把引号剥掉再看 flag，安全检查却可能因为引号的存在没认出这是个 flag，两边认知不一致就有绕过的空间。

    参数：
        command —— 命令原文
        base    —— 命令的首个词（命令名，如 echo）

    返回：命中时返回中文原因（用于审批提示文案），没命中返回 None。
    豁免规则：纯 echo 且没有管道/分号等操作符时放行——echo 的
    ANSI-C 之类形态只影响输出的文字内容，没有执行面。
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

    # 引号内容以 dash+flag 字符开头的情况（空白后跟引号，内容形如 ^-+[a-zA-Z0-9$`]）
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
    """剥掉环境变量赋值和 zsh 前置修饰符，返回真正的首命令名——FOO=1 command builtin cmd 这种写法里真正的命令藏在后面，不剥掉就拿首词对危险命令表会查错对象。

    参数：
        command —— 命令原文

    返回：剥完后的首命令词；全是赋值/修饰符时返回空串。
    """
    for token in command.strip().split():
        if _ENV_ASSIGN_RE.match(token):
            continue
        if token in _ZSH_PRECOMMAND_MODIFIERS:
            continue
        return token
    return ""


def check_injection_surface(command: str) -> Optional[str]:
    """主入口：检查命令里有没有注入面形态，有则返回中文原因，没有返回 None。

    调用位置在 PermissionChecker.check 里、硬拒绝黑名单（闸门 1）之后、
    只读快速通道之前——命中走用户审批而不是硬拒（ask 语义：所见非所执行
    ≠ 一定是攻击）。

    参数：
        command —— 待检查的命令原文

    返回：命中时返回中文原因（直接用于审批提示给用户看）；干净返回 None。
    """
    if not command or not command.strip():
        return None

    # 控制字符最先查——它可能让后面所有基于文本的正则全部失灵
    if _CONTROL_RE.search(command):
        return "含非打印控制字符"
    if _UNICODE_WS_RE.search(command):
        return "含 Unicode 空白字符"

    # 残缺片段（tab 开头 / flag 开头 / 操作符续行开头——多半是拼接命令的半截）
    stripped = command.strip()
    if re.match(r"^\s*\t", command):
        return "以 tab 开头的未完成片段"
    if stripped.startswith("-"):
        return "以 flag 开头的未完成片段"
    if re.match(r"^\s*(?:&&|\|\||;|>>?|<)", command):
        return "以操作符开头的续行片段"

    # quoted heredoc 的正文是字面量，剥掉之后再跑模式检查
    heredoc_stripped = _strip_quoted_heredocs(command)
    base = command.split(" ")[0] or ""
    with_dq, fully, _keepq = _extract_quoted_content(
        heredoc_stripped, is_jq=(base == "jq")
    )
    keepq = _extract_quoted_content(command)[2]  # 词中 # 检查要用原始命令的视图

    # jq 危险面（system() 能执行任意命令；-f 一族会把文件内容读进变量）
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

    # 双引号外的回车（分词差异）
    if _check_carriage_return_outside_dq(command):
        return "双引号外的回车符（\\r 分词差异）"

    # 换行分隔多条命令（「反斜杠+换行」的续行写法豁免）
    if _NEWLINE_CMD_RE.search(fully):
        return "换行分隔多条命令"

    # IFS 注入
    if _IFS_RE.search(command):
        return "IFS 变量用法"

    # /proc/*/environ
    if _PROC_ENVIRON_RE.search(command):
        return "访问 /proc/*/environ（环境变量泄露面）"

    # 命令替换 / 进程替换 / zsh 展开（用 with_dq 视图：单引号内不展开）
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

    # 词中 #（用 keepq 视图：保留引号字符才能捕捉 'x'# 这种引号挨着 # 的形态）
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

    # /dev/tcp|udp 网络伪设备（机器上没装网络工具也能往外发数据）
    if _NETWORK_DEVICE_RE.search(fully):
        return "/dev/tcp|udp 网络伪设备（无网络工具外泄面）"

    return None
