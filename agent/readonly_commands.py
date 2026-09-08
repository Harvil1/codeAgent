"""只读命令判定表（从 permission.py 簇 D 纯搬迁而来）。

大白话：这张表回答一个问题——"这条命令是不是只看不动手？"（ls、
git status、cat 这类看一眼就完事的命令）。判成"只读"的命令享受两个
特权：一是走免审批快速通道（零打扰直接放行），二是算并发安全
（可以多条同时跑，terminal 工具的并发分组就用 is_readonly_command）。

判定分层：先正则（复合命令切段 + 前缀表 + 写形态参数拦截），拿不准
再 bashlex AST 兜底（引号里的 && 只是参数不是分隔符，AST 分得清）；
保守优先——识别不了的形态一律不算只读。

为什么单独一个文件：permission.py 是全项目安全地基，太重了；这簇
判定只依赖 re/typing/bash_ast，不依赖 permission 任何东西——本模块
不反向 import permission（叶子模块，谁都可靠它，它不靠别人）。
外部契约：permission 主文件 re-export is_readonly_command，
agent/__init__、streaming_executor 的延迟 import 零感知。
"""
import re
from typing import List


# ---------------------------------------------------------------------------
# 识别"只读命令"——它们可走免审批快速通道,
# 也算并发安全(concurrency-safe),可以同时多个一起跑
# ---------------------------------------------------------------------------

# 只读前缀表(按"前缀 + 空格"匹配,后面参数随便)
# 注意:只收"绝对无副作用"的形态——git branch/tag/remote 这类命令只把
# 它们的只读子命令形态列进来(带名字参数的 "git branch x" 是建分支,是写)
_READONLY_PREFIXES = frozenset({
    # 文件/目录查看
    "ls", "dir", "tree", "pwd", "cat", "head", "tail", "wc", "file", "stat",
    "du", "df", "which", "where", "whereis", "type",
    # 搜索(find 的 -delete/-exec 等写形态由 _READONLY_FORBIDDEN_TOKENS 拦)
    "find", "grep", "rg", "ag", "findstr",
    # git 只读子命令(写形态 push/commit/checkout 等不进表;
    # branch/tag/remote 只收只读子形态——带名字参数的 "git branch x" 是写)
    "git status", "git log", "git diff", "git show", "git blame",
    "git shortlog", "git describe", "git rev-parse", "git ls-files",
    "git ls-remote", "git remote -v",
    "git branch -a", "git branch -v", "git branch -r",
    "git branch --list", "git branch --all", "git branch --show-current",
    "git tag -l", "git tag --list",
    "git stash list", "git config --get", "git worktree list",
    # 包管理只读
    "pip list", "pip show", "pip freeze", "uv pip list",
    "npm list", "npm ls",
    # 版本/环境信息
    "git --version", "python --version", "python3 --version",
    "node --version", "java -version", "go version",
    "rustc --version", "cargo --version", "uv --version", "pytest --version",
    # 系统信息(env 不进表:env VAR=x cmd 这种写法能执行任意命令)
    "whoami", "hostname", "uname", "date", "printenv", "echo",
    "id", "systeminfo", "tasklist",
})

# 只读段里禁止出现的参数(出现就说明它其实是写操作,比如 find -delete)
# 含 "--output":git diff/show 的 --output=<文件> 会写文件,
# 不能享受只读快速通道和并发放宽(重定向 > 已由 _REDIRECT_RE 拦,
# 这里拦的是参数形态的写)
_READONLY_FORBIDDEN_TOKENS = (
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprintf", "-fls", "-fprint0",
    "--output",
)

# 复合命令切分(按 && || ; | & 后台 换行 这些分隔符切开)+ 子命令替换
# ($() 反引号)+ 重定向(> >>)的正则。
# & 后台分隔和换行分隔也要切——切完逐段判只读,
# 交给 AST/正则精确判定,不能让 "ls & rm xxx" 混成一段蒙混过关
_COMPOUND_SPLIT_RE = re.compile(r"&&|\|\||;|\||&|\r|\n")
_SUBSHELL_RE = re.compile(r"\$\(|`")
# 进程替换形态 <( ) >( )——实测 bashlex 把它和 $() 分开识别,
# 正则层也得单独拦一道
_PROCSUB_RE = re.compile(r"[<>]\(")
_REDIRECT_RE = re.compile(r"(?:^|\s|\d)>{1,2}(?:&\d+)?")


def _is_readonly_segment(seg: str) -> bool:
    """判断单个命令段是否只读(前缀表匹配 + 写形态参数拦截)。

    参数:
        seg: 一段命令(已经按分隔符切开的)。

    返回:True 表示只读。空段(比如结尾多个分号)直接算通过;
    含命令替换或重定向的一律不算。
    """
    seg = seg.strip()
    if not seg:
        return True  # 空段(比如尾随的分号)忽略
    if _SUBSHELL_RE.search(seg) or _REDIRECT_RE.search(seg):
        return False
    lowered = seg.lower()
    for token in _READONLY_FORBIDDEN_TOKENS:
        if token in lowered:
            return False
    for prefix in _READONLY_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + " "):
            return True
    return False


def _tokens_readonly(tokens: List[str]) -> bool:
    """判断一个词(token)序列是否只读(AST 段专用的词序列匹配版)。

    和 _is_readonly_segment 干的活一样,但输入是拆好的词列表而不是字符串。
    为什么单独要一版:AST 解析出来的段天然是词列表,按**词序列**匹配前缀
    比按字符串前缀匹配更精确——`gitx status` 不会因为 "git" 字符串前缀
    而被误判成只读。

    规则:
    - 开头的环境变量赋值词跳过(FOO=1 ls 的动词是 ls);
    - 出现任何禁用参数(find -delete 之类)→ False;
    - 动词序列必须命中 _READONLY_PREFIXES 里的词序列前缀才算只读。

    参数:
        tokens: 词列表。

    返回:True 表示只读。
    """
    toks = [t for t in tokens if t]
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
        i += 1
    rest = [t.lower() for t in toks[i:]]
    if not rest:
        return False
    joined = " ".join(rest)
    for token in _READONLY_FORBIDDEN_TOKENS:
        if token in joined:
            return False
    for prefix in _READONLY_PREFIXES:
        p = prefix.split()
        if rest[: len(p)] == p:
            return True
    return False


def _is_readonly_command(command: str) -> bool:
    """判断整条命令是否只读(正则判定 + AST 兜底)。

    干什么:复合命令(含 && / || / ; / | / $() / 反引号)必须**每一段**
    都是只读才算只读;出现重定向(> >>)直接判非只读。保守优先:识别不了
    的形态一律不算只读。

    为什么重要:这个判定直接决定命令能不能走免审批快速通道、能不能并发跑,
    判松了危险命令就会蒙混过关。

    参数:
        command: 完整命令字符串。

    返回:True 表示只读。

    AST 兜底:正则判"非只读"时,再用 bashlex 精确解析一次——
    引号里的 && 只是参数不是分隔符(正则会误切),AST 分得清;但每段动词
    仍然必须在 _READONLY_PREFIXES 表内(正判不放宽白名单面)。
    解析失败就维持正则的结论(fail-open,不引入新拒绝)。
    """
    if not command or not command.strip():
        return False
    # 含替换形态($() 反引号 <() >())的命令不许走正则快速通道直通——必须
    # 交 AST 裁决(引号里的字面 $() AST 能分清;`cat <(ls)` 这类形态一旦
    # 免审直通就是 Critical 级放行)
    has_sub_form = bool(_SUBSHELL_RE.search(command) or _PROCSUB_RE.search(command))
    regex_ok = True
    for seg in _COMPOUND_SPLIT_RE.split(command):
        if not _is_readonly_segment(seg):
            regex_ok = False
            break
    if regex_ok and not has_sub_form:
        return True

    # AST 兜底(解析一次,逐段判)
    from agent.bash_ast import parse_info
    info = parse_info(command)
    if info is None:
        return False  # 解析不了 → 维持正则的结论(False)
    if info["has_redirect"] or info["has_substitution"]:
        return False
    for tokens in info["segments"]:
        seg_text = " ".join(tokens)
        # 纵深防御:万一 bashlex 漏检了替换形态,正则再兜一道(双保险)
        if _SUBSHELL_RE.search(seg_text):
            return False
        if not _tokens_readonly(tokens):
            return False
    return True


def is_readonly_command(command: str) -> bool:
    """公开入口:给 terminal 工具的并发分组用(和审批快速通道共用同一张判定表)。

    参数:
        command: 完整命令字符串。

    返回:True 表示只读(可并发安全执行)。
    """
    return _is_readonly_command(command)
