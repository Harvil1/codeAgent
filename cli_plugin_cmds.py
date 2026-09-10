"""插件管理命令集——/plugin：安装/卸载/启停/脚手架。

对标 claude code 的 /plugin：插件是一个「包裹」，往 ~/.codeAgent/plugins/
下放一个带 plugin.json 的目录就算一个插件（对标官方的约定优于配置——
没有清单时退化为按目录结构自动发现）。本项目第一版插件内容 = 技能
（plugin.json + skills/<技能名>/SKILL.md，由 constants.all_skills_dirs
在扫描技能时一并吃进来），以后可以往包裹里加更多类型的货。

大白话流程（本文件全部围绕这个）：
  /plugin                    看货架上有什么（表格：名字/版本/描述/启停/技能数）
  /plugin install <来源>     买一个回来——来源可以是本地目录，也可以是
                             git 仓库地址（gitee/github 都行），owner/repo
                             简写按 GitHub 处理
  /plugin uninstall <名字>   退货（删目录，删前问一句）
  /plugin enable|disable     启用/停用（改 plugin.json 的 enabled 字段）
  /plugin create <名字>      按模板搓一个空插件骨架（本地开发用）

装完立即生效：现场重扫技能命令挂回 RuntimeContext，不用重启
（对标官方 "Plugin is now active" 的即时生效语义）。
"""
# 注解延迟求值：rt: RuntimeContext 注解延迟到调用时（RuntimeContext 在
# cli.py，直接 import 会循环依赖），与 cli_diag_cmds 同款处理
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.table import Table

from cli_commands import slash_command
from cli_ui import console
from constants import plugins_dir

import logging

logger = logging.getLogger(__name__)

# owner/repo 简写的识别正则：两段、只含字母数字点横杠下划线
_OWNER_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _manifest_path(plugin_path: Path) -> Path:
    """插件清单文件的路径（<插件目录>/plugin.json）。"""
    return Path(plugin_path) / "plugin.json"


def _load_manifest(plugin_path: Path) -> dict:
    """读插件清单；不存在或读坏返回空 dict（fail-open，列表时不炸）。"""
    mp = _manifest_path(plugin_path)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("插件清单读坏 %s: %s", mp, e)
        return {}


def _save_manifest(plugin_path: Path, data: dict) -> None:
    """把清单写回去（原子写：先写临时文件再改名，断电不会写坏）。"""
    from agent.atomic_io import atomic_write_text
    atomic_write_text(
        _manifest_path(plugin_path),
        json.dumps(data, ensure_ascii=False, indent=2),
    )


def _iter_installed() -> list:
    """列出所有已安装插件：[(名字, 清单dict, 目录Path), ...] 按名字排序。

    判据和 constants.all_skills_dirs 一致——plugins/ 下的子目录带
    plugin.json 就算一个插件。
    """
    root = plugins_dir()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _manifest_path(d).exists():
            out.append((d.name, _load_manifest(d), d))
    return out


def _count_skills(plugin_path: Path) -> int:
    """数一个插件带了几份技能（skills/ 下带 SKILL.md 的子目录数）。"""
    skills = Path(plugin_path) / "skills"
    if not skills.is_dir():
        return 0
    return sum(
        1 for sub in skills.iterdir()
        if sub.is_dir() and (sub / "SKILL.md").exists()
    )


def _sanitize_name(name: str) -> str:
    """插件名消毒：只留安全字符（防来源里塞路径分隔符越界写目录）。"""
    cleaned = re.sub(r"[^\w.-]", "-", str(name or "").strip())
    return cleaned or "unnamed-plugin"


def _refresh_skills(rt) -> None:
    """现场重扫技能命令并挂回 RuntimeContext——装/卸/启停完立即生效。

    对照 cli.py 启动时那两句（scan_skill_commands / scan_bundle_commands），
    这里原样再跑一遍，保证装完的插件技能当场就能 /技能名 调用。
    """
    try:
        from agent.skill_commands import (
            scan_skill_commands, scan_bundle_commands,
        )
        from constants import all_skills_dirs, skills_dir as _user_skills
        rt.skill_commands = scan_skill_commands(all_skills_dirs())
        rt.bundle_commands = scan_bundle_commands(_user_skills())
    except Exception as e:
        logger.warning("插件变更后重扫技能失败（重启后仍会生效）: %s", e)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def _cmd_list() -> None:
    """/plugin 或 /plugin list：表格展示所有已安装插件。"""
    items = _iter_installed()
    if not items:
        console.print(
            f"[yellow]还没有安装任何插件[/yellow]（目录 {plugins_dir()} 为空）\n"
            "[dim]试试 /plugin install <本地目录或 git 地址>，"
            "或 /plugin create <名字> 搓一个[/dim]"
        )
        return

    table = Table(title="已安装插件")
    table.add_column("名字", style="cyan")
    table.add_column("版本")
    table.add_column("描述")
    table.add_column("状态")
    table.add_column("技能数", justify="right")
    for name, mf, path in items:
        enabled = bool(mf.get("enabled", True))
        table.add_row(
            name,
            str(mf.get("version", "?")),
            str(mf.get("description", ""))[:60] or "（无描述）",
            "[green]启用[/green]" if enabled else "[dim]停用[/dim]",
            str(_count_skills(path)),
        )
    console.print(table)


def _find_plugin_root(root: Path) -> Path | None:
    """在克隆下来的仓库里找插件根：优先仓库根的 plugin.json；
    没有就看一级子目录里是否恰好一个带 plugin.json 的（常见「仓库里
    套插件目录」布局）；都找不到返回 None。"""
    if _manifest_path(root).exists():
        return root
    candidates = [
        d for d in root.iterdir()
        if d.is_dir() and _manifest_path(d).exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _cmd_install(source: str) -> None:
    """/plugin install <来源>：从本地目录或 git 仓库安装插件。

    来源三种写法：
    - 本地路径：./my-plugin 或 D:\\plugins\\my-plugin（目录里要有 plugin.json）
    - git 地址：https://gitee.com/xxx/plugin.git（gitee/github/任意 git 都行）
    - owner/repo 简写：anthropics/claude-code → 按 GitHub 处理
    """
    source = source.strip().strip('"').strip("'")
    if not source:
        console.print("[red]用法：/plugin install <本地目录 | git地址 | owner/repo>[/red]")
        return

    tmp_dir = None
    try:
        # ---- 1. 判定来源类型，最终都归结为「一个本地插件目录」 ----
        if "://" in source or source.startswith("git@"):
            src_root, tmp_dir = _clone_to_temp(source)
        elif _OWNER_REPO_RE.match(source):
            src_root, tmp_dir = _clone_to_temp(
                f"https://github.com/{source}.git"
            )
        else:
            src_root = Path(source).expanduser().resolve()
            if not src_root.is_dir():
                console.print(f"[red]目录不存在：{src_root}[/red]")
                return

        if _find_plugin_root(src_root) is None:
            console.print(
                f"[red]来源里找不到 plugin.json[/red]——插件得是个带 "
                "plugin.json 的目录（仓库根或唯一一级子目录）。"
                f"找过：{src_root}"
            )
            return
        src_root = _find_plugin_root(src_root)

        # ---- 2. 定名字、落位置 ----
        manifest = _load_manifest(src_root)
        name = _sanitize_name(manifest.get("name") or src_root.name)
        target = plugins_dir() / name
        target.parent.mkdir(parents=True, exist_ok=True)
        is_update = target.exists()

        # ---- 3. 复制（排除 .git；已存在 = 覆盖更新） ----
        if is_update:
            shutil.rmtree(target)
        shutil.copytree(
            src_root, target,
            ignore=shutil.ignore_patterns(".git"),
        )

        n = _count_skills(target)
        verb = "更新" if is_update else "安装"
        console.print(
            f"[green]插件 {name} {verb}完成[/green]"
            f"（{target}，带 {n} 个技能）"
        )
        if n == 0:
            console.print(
                "[yellow]注意：这个插件没有 skills/<技能名>/SKILL.md，"
                "装了也不会新增技能命令[/yellow]"
            )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _clone_to_temp(url: str):
    """把 git 仓库浅克隆到临时目录，返回 (克隆目录, 临时目录)。

    返回临时目录是为了让调用方用完删掉（clone 深度 1，省流量）。
    克隆失败直接打印错误并抛 RuntimeError（由上层兜底成友好提示）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="codeagent-plugin-"))
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(tmp / "repo")],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("本机没有 git 命令，没法从远程安装")
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("git clone 超时（120 秒）")
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(
            f"git clone 失败：{(result.stderr or '').strip()[:200]}"
        )
    return tmp / "repo", tmp


def _resolve_installed(name: str) -> Path | None:
    """按名字找已装插件目录；找不到打印提示并返回 None。"""
    for pname, _mf, path in _iter_installed():
        if pname == name:
            return path
    console.print(f"[red]没找到插件：{name}（/plugin 看看名字）[/red]")
    return None


def _cmd_uninstall(name: str) -> None:
    """/plugin uninstall <名字>：删掉插件目录（删前问一句）。"""
    path = _resolve_installed(name)
    if path is None:
        return
    n = _count_skills(path)
    answer = console.input(
        f"[bold]卸载插件 {name}（{n} 个技能一并移除）？y/N >[/bold] "
    ).strip().lower()
    if answer not in ("y", "yes"):
        console.print("已取消")
        return
    shutil.rmtree(path)
    console.print(f"[green]插件 {name} 已卸载[/green]（{path}）")


def _cmd_set_enabled(name: str, enabled: bool) -> None:
    """/plugin enable|disable <名字>：改清单里的 enabled 字段。"""
    path = _resolve_installed(name)
    if path is None:
        return
    manifest = _load_manifest(path)
    if not manifest:
        # 清单读坏了（或手删了）：造个最小清单把名字保住
        manifest = {"name": name}
    manifest["enabled"] = enabled
    _save_manifest(path, manifest)
    console.print(
        f"[green]插件 {name} 已{'启用' if enabled else '停用'}[/green]"
        + ("" if enabled else "[dim]（技能命令下轮生效或重启）[/dim]")
    )


def _cmd_create(name: str) -> None:
    """/plugin create <名字>：搓一个插件骨架（本地开发起点）。

    结构（对齐 constants.plugins_dir 认的格式）：
        <name>/plugin.json            清单
        <name>/skills/<name>/SKILL.md 技能模板
    """
    name = _sanitize_name(name)
    if not name or name == "unnamed-plugin":
        console.print("[red]用法：/plugin create <名字>[/red]")
        return
    target = plugins_dir() / name
    if target.exists():
        console.print(f"[red]已存在同名插件：{target}[/red]")
        return

    skill_dir = target / "skills" / name
    skill_dir.mkdir(parents=True)
    _save_manifest(target, {
        "name": name,
        "version": "0.1.0",
        "description": f"{name} 插件（/plugin create 生成）",
        "enabled": True,
    })
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {name} 技能——干什么用的、什么时机用，写一句话\n"
        "---\n\n"
        f"# {name}\n\n"
        "在这里写「怎么做」的操作步骤。\n",
        encoding="utf-8",
    )
    console.print(
        f"[green]插件骨架已建好[/green]：{target}\n"
        f"[dim]往 {skill_dir / 'SKILL.md'} 里写内容，"
        "然后 /plugin enable " + name + "（新装的默认已启用，"
        "重扫后 /" + name + " 即可调用）[/dim]"
    )


# ---------------------------------------------------------------------------
# 总分发 + 注册
# ---------------------------------------------------------------------------

def _handle_plugin_command(args: str, rt) -> bool:
    """/plugin 总分发：看第一个参数转给对应子命令。

    参数：
        args：空 = 列表；否则第一个词是子命令
              （list/install/uninstall/enable/disable/create）
        rt：RuntimeContext（装/卸/启停后用它重扫技能命令）
    返回：
        bool —— True 表示命令已处理
    """
    parts = args.strip().split()
    if not parts:
        _cmd_list()
        return True

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    try:
        if sub in ("list", "ls"):
            _cmd_list()
        elif sub == "install":
            _cmd_install(" ".join(parts[1:]))
        elif sub == "uninstall":
            if not rest:
                console.print("[red]用法：/plugin uninstall <名字>[/red]")
            else:
                _cmd_uninstall(rest)
        elif sub in ("enable", "disable"):
            if not rest:
                console.print(f"[red]用法：/plugin {sub} <名字>[/red]")
            else:
                _cmd_set_enabled(rest, sub == "enable")
        elif sub == "create":
            _cmd_create(" ".join(parts[1:]))
        else:
            console.print(
                f"[red]未知子命令：{sub}[/red]\n"
                "可用：list, install, uninstall, enable, disable, create"
            )
            return True
    except Exception as e:
        # 子命令失败别炸 REPL：打印错误 + 记日志（fail-open 但大声）
        console.print(f"[red]插件命令执行失败：{e}[/red]")
        logger.warning("/plugin %s 失败: %s", sub, e, exc_info=True)
        return True

    # 装了/卸了/启停了才需要重扫（list 出错不折腾）
    if sub in ("install", "uninstall", "enable", "disable", "create"):
        _refresh_skills(rt)
    return True


@slash_command(
    name="/plugin", category="插件",
    usage="/plugin [list|install|uninstall|enable|disable|create]",
    summary="插件管理：安装/卸载/启停/脚手架",
    aliases=["/plugins"],
    arg_completer=lambda text: [
        s for s in (
            "list", "install ", "uninstall ", "enable ", "disable ", "create ",
        ) if s.startswith(text)
    ],
)
def cmd_plugin(args: str, rt) -> bool:
    return _handle_plugin_command(args, rt)
