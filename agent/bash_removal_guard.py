"""危险删除与 cd+git 防护（从 permission.py 簇 F 纯搬迁而来）。

大白话：这里管两类"删不得/进不得"的专项拦截——
- 危险删除路径判定：rm/rmdir/del/erase/rd 这些删除命令的目标如果是
  裸通配符 *、根目录、家目录、根/盘根的直接子目录，直接拒——审批也
  解不了锁（用户手滑点 y 正是这类检查要防的事故）；
- cd+git 组合：cd 进一个攻击者可控的目录再跑 git，恶意构造的裸仓库
  （bare repo）能借 core.fsmonitor 之类的配置注入执行任意命令；
  顺序敏感——git 在 cd **之前**跑操作的是原目录，不构成这个攻击面。

为什么单独一个文件：permission.py 是全项目安全地基，太重了；这一簇
只依赖 re/os/pathlib/typing 标准库，不依赖 permission 任何东西——本模块
不反向 import permission（叶子模块，谁都可靠它，它不靠别人）。

注意：_CMD_SEGMENT_SPLIT_RE 与 readonly_commands 的 _COMPOUND_SPLIT_RE
长得一模一样是有意的历史重复——两边注释关注点各有侧重（这边强调补上
& 后台分隔和 \r\n 换行，防 "echo hi\nrm -rf /" 被当成一段漏过本闸门），
不合并、各自演化。

外部契约：permission 主文件 re-export 五个符号（check_dangerous_removal/
is_dangerous_removal_path/_has_cd_git_combo/_CMD_SEGMENT_SPLIT_RE/
_MAX_COMPOUND_SEGMENTS）——check() 的三处调用点（危险删除、复合命令
段数上限闸门、cd+git 组合）与公开名 from-import 契约零感知。
"""
import os
import re
from pathlib import Path
from typing import Optional


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


# git 只读子命令白名单：cd 后跑这些不触发审批（不加载 hooks、不写状态、
# 不执行任意代码——`git log/status/diff/show` 只读仓库元数据）
_GIT_READONLY_SUBCMDS = frozenset({
    "log", "status", "diff", "show", "blame", "shortlog", "describe",
    "branch", "tag", "remote", "rev-parse", "ls-files", "ls-remote",
    "cat-file", "name-rev", "reflog", "stash--list", "config", "-l",
    "--version", "help", "whatchanged", "grep",
})


def _is_readonly_git(toks: list) -> bool:
    """判断 git 命令是否只读（子命令在白名单里且不带写 flag。）

    白名单覆盖 log/status/diff/show 等纯查询——它们不加载 hooks、
    不执行 clean/fsmonitor，cd 后跑它们不构成注入面。config -l 例外
    （config 不带 -l 是写操作）。
    """
    if len(toks) < 2:
        return False
    sub = toks[1].lower()
    if sub in ("config",) and len(toks) >= 3 and toks[2] in ("-l", "--list"):
        return True
    return sub in _GIT_READONLY_SUBCMDS


def _has_cd_git_combo(command: str) -> bool:
    """复合命令里 cd 之后再出现 git(含 xargs git)就返回 True。

    为什么要抓这个组合:cd 进一个攻击者可控的目录再跑 git——恶意构造的
    bare repo(裸仓库)可以通过 core.fsmonitor 之类的配置注入执行任意命令。
    顺序敏感:git 在 cd **之前**跑,
    操作的是原目录,不构成这个攻击面,返回 False。

    只读 git 命令（log/status/diff/show 等）豁免：它们不加载 hooks
    不写状态，cd 后跑不构成注入面——不豁免的话 `cd 项目 && git log`
    这种最常见的排查命令每次都弹审批，用户烦死了。

    参数:
        command: 完整命令字符串。

    返回:True 表示存在"cd 之后的 git"且不是只读命令。
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
            if saw_cd and not _is_readonly_git(toks):
                return True
    return False
# Windows del/rd 的斜杠式选项(/s /q);注意别把 /usr 这种真路径也当成选项。
# 第二分支要求恰好一个字母——裸 "/"（0 字母）不是选项，那是 rm -rf / 的
# 根目标，跳过它等于本层漏拦（R14 终审发现；修复前 ^/[A-Za-z]?$ 的 ? 放走裸 /）
_CMD_FLAG_RE = re.compile(r"^-[A-Za-z]*$|^/[A-Za-z]$")


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
