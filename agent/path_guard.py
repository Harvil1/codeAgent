"""路径安全簇（从 permission.py 簇 C 纯搬迁而来）。

大白话：这里管的是"哪些路径碰不得、哪些路径写得"——AI 读写任何文件
之前都要先过这一关。四样家当：
- 保护表 _PROTECTED_PATHS（~/.ssh、/etc、C:\\Windows 这类读写都拒的地方，
  模块加载时预先 resolve 成绝对路径存 _PROTECTED_PATHS_RESOLVED）；
- 写保护表 _WRITE_PROTECTED_PATHS（CodeAgent 自己的项目代码目录——
  读可以、改不行，防 AI 给自己"动手术"；try-import constants 失败时
  静默空表，时机是模块加载时）；
- 可疑路径形态检查（8.3 短名、NTFS ADS 冒号、长路径前缀、尾点、
  DOS 设备名、三连点段、波浪变体、glob 元字符这些"改名绕过安检"的
  花招，全平台都查）；
- 写白名单 _EXTRA_ALLOWED_ROOTS（/add-dir 追加的额外目录；读写全走
  簇内 add/list/remove/clear 四件函数，自包含不外泄）。

safe_path 是函数版安检（不带审批，工具层直接用）；带审批的版本是
PermissionChecker.check_path（留在主文件，热区不拆），两者白名单同源
（default_allowed_roots）、硬底线同序（保护先于白名单——顺序铁律）。

为什么单独一个文件：permission.py 是全项目安全地基，太重了；这一簇
除返回类型 PermissionResult（safe_path 函数体内延迟 import，防与主文件
re-export 循环咬死）外不依赖 permission 任何东西——准叶子模块，
谁都可靠它。两处模块加载时副作用（保护表 resolve 循环、写保护表
try-import）随迁整块保持时机：本模块在主文件 import 块第一时间被
加载，相对所有使用方与拆分前一致。

外部契约：permission 主文件 re-export 九个符号（safe_path、
is_protected_path、is_write_protected_path、check_suspicious_path、
add/list/remove/clear_extra_allowed_roots、default_allowed_roots），
tools/file_operations 与 tools/glob_tool 的模块级 import、
memory_store 等延迟 import 与 check_path 类方法调用点零感知。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
# 模块加载时预先算好绝对路径(免得每次 safe_path 都重复对 17 条路径做 resolve:
# 3 密钥目录 + 3 CLI 凭证目录 + 8 Unix 系统目录 + 3 Windows 系统目录)
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
    logger.warning("异常被吞(fail-open)", exc_info=True)


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
        logger.warning("异常被吞(fail-open)", exc_info=True)
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
    # 延迟 import:返回类型 PermissionResult 定义在主文件 agent.permission——
    # 模块级互相 import 会咬死(循环 import)，按仓库惯例放函数体内延迟拿。
    # 接手者知悉:get_type_hints(safe_path) 会因 PermissionResult 不在本模块
    # 命名空间而 NameError——当前无消费者,故不做处理
    from agent.permission import PermissionResult

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
