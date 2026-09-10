"""插件管理命令集——/plugin：安装/卸载/启停/市场挑选/脚手架。

对标 claude code 的 /plugin。插件是一个「包裹」，往 ~/.codeAgent/plugins/
下放一个带 plugin.json 的目录就算一个插件。本项目第一版插件内容 = 技能
（plugin.json + skills/<技能名>/SKILL.md，由 constants.all_skills_dirs
在扫描技能时一并吃进来）。

大白话流程（本文件全部围绕这个）：
  /plugin                    看自己货架上有什么（名字/版本/描述/启停/技能数）
  /plugin install <来源>     买一个回来——本地目录 / git 地址 / owner/repo
                             简写 / 名字@市场（从已添加的市场里挑）
  /plugin market             【市场】打开挑选器：方向键浏览所有已添加市场
                             里的插件，空格勾选、Enter 直接安装
  /plugin market add <来源>  添加一个市场（git 仓库 / owner/repo / 本地目录）
  /plugin market list        看添加了哪些市场
  /plugin market update [名] 刷新市场目录册（不写名字 = 全刷）
  /plugin market remove <名> 移除市场（只删目录册，已装插件不动）
  /plugin uninstall <名字>   退货（删目录，删前问一句）
  /plugin enable|disable     启用/停用（改 plugin.json 的 enabled 字段）
  /plugin create <名字>      按模板搓一个空插件骨架（本地开发用）

市场是什么：一个 git 仓库（或本地目录），根下放一份目录册
（marketplace.json，也认官方的 .claude-plugin/marketplace.json——
所以 claude code 官方市场的仓库也能挂进来挑，技能格式两边通用）：

    { "name": "my-plugins", "owner": {"name": "xx"},
      "plugins": [
        {"name": "demo", "source": "./plugins/demo", "description": "..."},
        {"name": "remote", "source": {"repo": "owner/repo"}, "description": "..."}
      ] }

source 的写法（官方四型全支持）：相对路径（市场仓库里的目录）、
git 地址（字符串 URL / {"url": ...}）、GitHub 简写（{"repo": "o/r"}）、
git-subdir（{"url", "path", "ref"?}——克隆整个仓库后取 path 子目录）。

**插件 MCP 接线**：插件根下带 .mcp.json 的（官方 MCP 类插件长这样），
装上/启用即自动连接其 MCP server 并登记工具（装 = 授权连接，对标
官方「插件启用即启动」）；卸载/停用即断开（工具靠 check_fn 自动下架）。
启动时 initialize_mcp 也会扫一遍已启用插件补连线。与用户级
.mcp.json 同名的跳过——用户手写的优先级最高。

**内置市场**：config.plugins.builtin_marketplaces 默认带着官方
claude-plugins-official——首次 /plugin market 浏览时自动拉取（开箱
即用）；用户 remove 过的不会复活（记号文件防僵尸）；地址可以在
settings.json 覆盖（GitHub 拉不动换镜像）。

市场本体克隆在 ~/.codeAgent/plugins/marketplaces/<名>/ 下——注意它和
插件同住 plugins/，但根下没有 plugin.json，不会被误认成插件。

装完立即生效：现场重扫技能命令挂回 RuntimeContext，不用重启。
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
# 小工具：插件清单 / 已装列表
# ---------------------------------------------------------------------------

def _find_manifest(plugin_path: Path) -> Path | None:
    """找插件清单文件，返回真实路径；没有返回 None。

    认两个位置（兼容官方 claude code 插件布局）：
    - <插件目录>/plugin.json                     （本项目约定）
    - <插件目录>/.claude-plugin/plugin.json      （官方布局）
    """
    for cand in (
        Path(plugin_path) / "plugin.json",
        Path(plugin_path) / ".claude-plugin" / "plugin.json",
    ):
        if cand.exists():
            return cand
    return None


def _load_manifest(plugin_path: Path) -> dict:
    """读插件清单；不存在或读坏返回空 dict（fail-open，列表时不炸）。"""
    mp = _find_manifest(plugin_path)
    if mp is None:
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("插件清单读坏 %s: %s", mp, e)
        return {}


def _save_manifest(plugin_path: Path, data: dict) -> None:
    """把清单写回去（原子写：先写临时文件再改名，断电不会写坏）。

    写回「找到的那份」清单——官方布局的插件改 enabled 就落在
    .claude-plugin/plugin.json 里，不另造文件；一份清单不管在哪都只有一份。
    """
    from agent.atomic_io import atomic_write_text
    mp = _find_manifest(plugin_path) or (Path(plugin_path) / "plugin.json")
    atomic_write_text(
        mp, json.dumps(data, ensure_ascii=False, indent=2),
    )


def _iter_installed() -> list:
    """列出所有已安装插件：[(名字, 清单dict, 目录Path), ...] 按名字排序。

    判据和 constants.all_skills_dirs 一致——plugins/ 下的子目录带
    plugin.json 就算一个插件；marketplaces/ 是市场目录册，显式跳过。
    """
    root = plugins_dir()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "marketplaces":
            continue
        if _find_manifest(d) is not None:
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
    """插件名/市场名消毒：只留安全字符（防来源里塞路径分隔符越界写目录）。"""
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
# 插件 MCP 接线：带 .mcp.json 的插件装上即连（对标官方「插件启用即启动」）
# ---------------------------------------------------------------------------

def _wire_plugin_mcp(plugin_name: str) -> None:
    """把指定插件自带的 MCP server 连上并登记工具（装/启用后调）。

    优先级规则跟启动时一致：与用户级 .mcp.json 同名的跳过（用户手写
    的最大）；已经在连的同名不抢（connect_one 本身幂等，这里显式跳过
    是为了不做重复登记）。连不上只记日志——插件装好了不能因为一个
    server 起不来就报失败。
    """
    try:
        from agent.mcp_client import load_mcp_config, get_mcp_manager
        from tools.mcp_tool import (
            collect_plugin_mcp_servers, connect_servers_and_register,
        )
        user_cfg = load_mcp_config()
        clients = getattr(get_mcp_manager(), "_clients", {}) or {}
        wanted = {}
        for conn, info in collect_plugin_mcp_servers().items():
            if info["plugin"] != plugin_name:
                continue
            if conn in user_cfg or conn in clients:
                continue
            wanted[conn] = info["cfg"]
        if not wanted:
            return
        n = connect_servers_and_register(wanted)
        console.print(
            f"[green]插件 {plugin_name} 的 MCP server 已接线[/green]"
            f"（{len(wanted)} 个 server，登记 {n} 个工具）"
        )
    except Exception as e:
        logger.warning("插件 %s 的 MCP 接线失败（重启后会再试）: %s", plugin_name, e)


def _unwire_plugin_mcp(plugin_name: str) -> None:
    """断开指定插件名下的 MCP server（卸载/停用前调）。

    必须在插件目录/清单还在盘上时调——连接名是扫 .mcp.json 算出来的，
    先删后断就找不到归属了。断开后工具不用手动反注册：注册时带的
    check_fn（server 连着才出场）会自动把它们藏掉。
    """
    try:
        from agent.mcp_client import get_mcp_manager
        from tools.mcp_tool import collect_plugin_mcp_servers
        manager = get_mcp_manager()
        for conn, info in collect_plugin_mcp_servers().items():
            if info["plugin"] != plugin_name:
                continue
            try:
                if manager.disconnect_one(conn):
                    console.print(f"[dim]MCP server {conn} 已断开（工具随之下架）[/dim]")
            except Exception as e:
                logger.debug("断开插件 MCP %s 失败（重启后自然干净）: %s", conn, e)
    except Exception as e:
        logger.debug("插件 %s 的 MCP 拆线检查失败: %s", plugin_name, e)


# ---------------------------------------------------------------------------
# 安装核心（本地目录 / git / 市场 各条路最终都汇到「复制一个目录」）
# ---------------------------------------------------------------------------

def _find_plugin_root(root: Path) -> Path | None:
    """在来源目录里找插件根：优先根上的清单（plugin.json 或
    .claude-plugin/plugin.json）；没有就看一级子目录里是否恰好一个带
    清单的（常见「仓库里套插件目录」布局）；都找不到返回 None。"""
    if _find_manifest(root) is not None:
        return root
    candidates = [
        d for d in root.iterdir()
        if d.is_dir() and _find_manifest(d) is not None
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _install_dir(src_root: Path) -> str | None:
    """把一个「已经是插件目录」的来源装进 plugins/，返回插件名。

    已存在同名 = 覆盖更新（升级语义）；复制时剥掉 .git。
    来源找不到 plugin.json 时打印原因并返回 None。
    """
    plugin_root = _find_plugin_root(src_root)
    if plugin_root is None:
        console.print(
            f"[red]来源里找不到 plugin.json[/red]——插件得是个带 "
            f"plugin.json 的目录（仓库根或唯一一级子目录）。找过：{src_root}"
        )
        return None
    manifest = _load_manifest(plugin_root)
    name = _sanitize_name(manifest.get("name") or plugin_root.name)
    target = plugins_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    is_update = target.exists()
    if is_update:
        _rmtree_force(target)
    shutil.copytree(plugin_root, target, ignore=shutil.ignore_patterns(".git"))

    n = _count_skills(target)
    console.print(
        f"[green]插件 {name} {'更新' if is_update else '安装'}完成[/green]"
        f"（{target}，带 {n} 个技能）"
    )
    if n == 0:
        console.print(
            "[yellow]注意：这个插件没有 skills/<技能名>/SKILL.md，"
            "装了也不会新增技能命令[/yellow]"
        )
    # 插件自带 MCP server 的当场接线（装 = 授权连接）
    _wire_plugin_mcp(name)
    return name


def _clone_to_temp(url: str, ref: str | None = None):
    """把 git 仓库浅克隆到临时目录，返回 (克隆目录, 临时目录)。

    ref 是官方目录册里的版本钉子（分支/标签名）——给了就按那个版
    本克隆（--branch），没给用默认分支。sha 钉子需要整仓 fetch，第
    一版不支持（忽略）。
    返回临时目录是为了让调用方用完删掉（clone 深度 1，省流量）。
    克隆失败抛 RuntimeError（由上层兜底成友好提示）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="codeagent-plugin-"))
    try:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        result = subprocess.run(
            cmd + [url, str(tmp / "repo")],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        _rmtree_force(tmp, ignore_errors=True)
        raise RuntimeError("本机没有 git 命令，没法从远程安装")
    except subprocess.TimeoutExpired:
        _rmtree_force(tmp, ignore_errors=True)
        raise RuntimeError("git clone 超时（120 秒）")
    if result.returncode != 0:
        _rmtree_force(tmp, ignore_errors=True)
        raise RuntimeError(
            f"git clone 失败：{(result.stderr or '').strip()[:200]}"
        )
    return tmp / "repo", tmp


def _git_origin(dir_path: Path) -> str | None:
    """读一个 git 克隆的远程地址（market update 重新拉取用）。"""
    try:
        r = subprocess.run(
            ["git", "-C", str(dir_path), "config", "--get",
             "remote.origin.url"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def _cmd_install(source: str) -> None:
    """/plugin install <来源>：装一个插件。

    来源四种写法：
    - 本地路径：./my-plugin（目录里要有 plugin.json）
    - git 地址：https://gitee.com/xxx/plugin.git（任意 git 托管都行）
    - owner/repo 简写：anthropics/claude-code → 按 GitHub 处理
    - 名字@市场：demo@my-plugins → 从已添加的市场目录册里找着装
    """
    source = source.strip().strip('"').strip("'")
    if not source:
        console.print(
            "[red]用法：/plugin install <本地目录 | git地址 | owner/repo | 名字@市场>[/red]"
        )
        return

    # ---- 市场引用：名字@市场 → 从目录册解析出真实来源再走统一安装 ----
    if "@" in source and "://" not in source and not source.startswith("git@"):
        name, _, market = source.partition("@")
        resolved = _resolve_market_source(name.strip(), market.strip())
        if resolved is None:
            return
        _install_resolved(resolved)
        return

    tmp_dir = None
    try:
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
        _install_dir(src_root)
    except RuntimeError as e:
        console.print(f"[red]安装失败：{e}[/red]")
    finally:
        if tmp_dir is not None:
            _rmtree_force(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 市场：目录册的增删查 + 挑选安装
# ---------------------------------------------------------------------------

def _marketplaces_root() -> Path:
    """市场目录册们住的目录：~/.codeAgent/plugins/marketplaces/。"""
    return plugins_dir() / "marketplaces"


def _rmtree_force(path, ignore_errors: bool = False) -> None:
    """删目录（Windows 加固版）。

    git 克隆里的 pack 文件（.git/objects/pack/*）带只读位，普通
    shutil.rmtree 会 PermissionError「拒绝访问」——先扫一遍去掉只读
    位再删，就稳了。不存在的路径直接跳过。
    ignore_errors 语义同 shutil.rmtree：删失败也不抛（临时目录收尾用）。
    """
    p = Path(path)
    if not p.exists():
        return
    import os
    import stat
    if os.name == "nt":
        for f in p.rglob("*"):
            try:
                os.chmod(f, stat.S_IWRITE)
            except OSError:
                pass  # 个别文件改不了位也没关系，rmtree 再试
    shutil.rmtree(p, ignore_errors=ignore_errors)


# ---------------------------------------------------------------------------
# 内置市场：开箱即用（官方 claude code 插件市场）
# ---------------------------------------------------------------------------

def _builtin_marketplace_defs() -> list:
    """读内置市场清单（config.plugins.builtin_marketplaces）。

    默认值在 config.py::DEFAULT_CONFIG（官方市场）；用户可以在
    settings.json 里覆盖（比如把 GitHub 地址换成镜像）。
    配置读不到时用代码里这份兜底——保证极端情况下官方市场仍开箱即用。
    """
    try:
        from config import load_config
        defs = (load_config().get("plugins", {}) or {}).get(
            "builtin_marketplaces"
        )
        if isinstance(defs, list) and defs:
            out = [
                {"name": str(d.get("name") or ""), "url": str(d.get("url") or "")}
                for d in defs
                if isinstance(d, dict) and d.get("url")
            ]
            if out:
                return out
    except Exception as e:
        logger.debug("读内置市场配置失败，用代码兜底: %s", e)
    return [{
        "name": "claude-plugins-official",
        "url": "https://github.com/anthropics/claude-plugins-official.git",
    }]


def _removed_marker(name: str) -> Path:
    """「用户删过这个市场」的记号文件路径（防内置市场删了又复活）。"""
    return _marketplaces_root() / f".removed-{name}"


def _ensure_builtin_marketplaces() -> None:
    """把内置市场补齐：还没添加过的自动拉取（失败不吵，fail-open）。

    开箱即用语义：新用户第一次 /plugin market 浏览就能看到官方市场
    的插件，不用先手动 add。网络拉不动只提示一句，用户仍可以
    /plugin market add 换镜像或加自己的市场。
    用户主动 remove 过的（有记号）不再复活——尊重用户的删除。
    """
    try:
        existing = {n for n, _c, _d in _iter_marketplaces()}
        for d in _builtin_marketplace_defs():
            name = _sanitize_name(d["name"])
            if not name or name in existing:
                continue
            if _removed_marker(name).exists():
                continue
            console.print(f"[dim]首次使用：拉取内置市场 {name}...[/dim]")
            _cmd_market_add(d["url"])
    except Exception as e:
        logger.warning("内置市场拉取失败（fail-open）: %s", e)


def _read_catalog(mkt_dir: Path) -> dict | None:
    """读一个市场目录的目录册。

    认两个位置：根下 marketplace.json（本项目约定），或
    .claude-plugin/marketplace.json（官方 claude code 市场格式——
    技能结构两边通用，官方市场仓库挂进来也能挑）。
    读不到/读坏返回 None。
    """
    for cand in (
        mkt_dir / "marketplace.json",
        mkt_dir / ".claude-plugin" / "marketplace.json",
    ):
        if not cand.exists():
            continue
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("plugins"), list):
                return data
            logger.warning("市场目录册格式不对 %s（缺 plugins 数组）", cand)
        except Exception as e:
            logger.warning("市场目录册读坏 %s: %s", cand, e)
        return None
    return None


def _iter_marketplaces() -> list:
    """列出所有已添加市场：[(市场名, 目录册dict, 目录Path), ...]。

    市场名优先用目录册里的 name 字段，没有就用目录名。
    """
    root = _marketplaces_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        catalog = _read_catalog(d)
        if catalog is None:
            continue
        out.append((_sanitize_name(catalog.get("name") or d.name), catalog, d))
    return out


def _iter_market_entries() -> list:
    """把所有市场的插件条目摊平：[(市场名, 条目dict), ...]。

    条目里没名字的跳过（没名字没法装也没法挑）。
    """
    entries = []
    for market_name, catalog, _dir in _iter_marketplaces():
        for entry in catalog.get("plugins", []):
            if isinstance(entry, dict) and entry.get("name"):
                entries.append((market_name, entry))
    return entries


def _resolve_entry_source(mkt_dir: Path, entry: dict):
    """把目录册条目的 source 翻译成统一来源（官方四型全支持）。

    返回四元组 (kind, url或路径, 子目录, ref)：
    - ("local", 市场里的绝对路径, None, None)   字符串相对路径（或绝对路径）
    - ("git", url, None, None)                  字符串 git 地址 / {"url": ...}
    - ("git", url, None, None)                  {"repo": "owner/repo"} → GitHub
    - ("git-subdir", url, path, ref?)           {"source": "git-subdir",
                                                 "url", "path", "ref"?}——
                                                 克隆整个仓库后取 path 子目录
    认不出的打印原因返回 None。
    """
    src = entry.get("source")
    if isinstance(src, str):
        if "://" in src or src.startswith("git@"):
            return ("git", src, None, None)
        # 相对路径：相对市场目录解析（也兼容写绝对路径的）
        p = Path(src)
        if not p.is_absolute():
            p = Path(mkt_dir) / p
        return ("local", str(p), None, None)
    if isinstance(src, dict):
        if src.get("repo"):
            return ("git", f"https://github.com/{src['repo']}.git", None, None)
        url = src.get("url")
        if url and src.get("path"):
            return (
                "git-subdir", str(url), str(src["path"]),
                str(src.get("ref") or "") or None,
            )
        if url:
            return ("git", str(url), None, None)
    console.print(
        f"[red]条目 {entry.get('name')} 的 source 认不出：{src!r}[/red]"
        "（支持：相对路径 / git 地址 / {{repo|url}} / git-subdir）"
    )
    return None


def _install_resolved(resolved) -> None:
    """按 _resolve_entry_source 的结果统一安装（三条路汇到一处）。

    local：直接装目录；git：克隆整个仓库装；git-subdir：克隆整个
    仓库后取指定子目录装（子目录才是插件本体）。
    """
    kind, value = resolved[0], resolved[1]
    subdir = resolved[2] if len(resolved) > 2 else None
    ref = resolved[3] if len(resolved) > 3 else None
    if kind == "local":
        _install_dir(Path(value))
        return
    tmp_dir = None
    try:
        src_root, tmp_dir = _clone_to_temp(value, ref=ref)
        if subdir:
            src_root = src_root / subdir
        _install_dir(src_root)
    except RuntimeError as e:
        console.print(f"[red]安装失败：{e}[/red]")
    finally:
        if tmp_dir is not None:
            _rmtree_force(tmp_dir, ignore_errors=True)


def _resolve_market_source(name: str, market: str):
    """按「插件名@市场名」找到条目并解析出真实来源。

    找不到打印提示（顺带列出该市场/全部市场里可选的名字）并返回 None。
    """
    all_entries = _iter_market_entries()
    if not all_entries:
        # 一个市场都没有：先把内置市场补齐再试一次（官方市场开箱即用）
        _ensure_builtin_marketplaces()
        all_entries = _iter_market_entries()
        if not all_entries:
            console.print(
                "[red]还没有可用的市场[/red]——/plugin market add <git地址>"
                "添加（镜像或私有仓库都行），再 /plugin install 名字@市场"
            )
            return None

    candidates = [
        (m, e) for m, e in all_entries
        if e["name"] == name and (not market or m == market)
    ]
    if not candidates:
        console.print(f"[red]市场里没有插件：{name}"
                      f"{'@' + market if market else ''}[/red]")
        shown = ", ".join(
            f"{e['name']}@{m}" for m, e in all_entries[:20]
        ) or "（市场目录册是空的）"
        console.print(f"[dim]可选：{shown}[/dim]")
        return None

    market_name, entry = candidates[0]
    mkt_dir = next(
        (d for mn, _c, d in _iter_marketplaces() if mn == market_name), None
    )
    if mkt_dir is None:
        return None
    return _resolve_entry_source(mkt_dir, entry)


def _cmd_market_add(source: str) -> None:
    """/plugin market add <来源>：添加一个市场（克隆/复制目录册到本地）。

    来源：git 地址 / owner/repo 简写 / 本地目录（本地开发用，会复制一份）。
    """
    source = source.strip().strip('"').strip("'")
    if not source:
        console.print("[red]用法：/plugin market add <git地址 | owner/repo | 本地目录>[/red]")
        return

    tmp_dir = None
    try:
        if "://" in source or source.startswith("git@"):
            mkt_root, tmp_dir = _clone_to_temp(source)
        elif _OWNER_REPO_RE.match(source):
            mkt_root, tmp_dir = _clone_to_temp(
                f"https://github.com/{source}.git"
            )
        else:
            mkt_root = Path(source).expanduser().resolve()
            if not mkt_root.is_dir():
                console.print(f"[red]目录不存在：{mkt_root}[/red]")
                return

        catalog = _read_catalog(mkt_root)
        if catalog is None:
            console.print(
                f"[red]来源里找不到目录册[/red]——市场得是个带 "
                "marketplace.json（或 .claude-plugin/marketplace.json）的"
                f"目录。找过：{mkt_root}"
            )
            return

        name = _sanitize_name(catalog.get("name") or mkt_root.name)
        target = _marketplaces_root() / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _rmtree_force(target)
        if tmp_dir is not None:
            # git 来源：整个克隆搬过来（.git 留着）——market update
            # 靠它记住远程地址，剥了就没法刷新了
            shutil.move(str(mkt_root), str(target))
        else:
            shutil.copytree(
                mkt_root, target, ignore=shutil.ignore_patterns(".git"),
            )
        # 重新添加 = 用户改主意了：清掉「删过」记号，内置市场恢复自动补齐
        _removed_marker(name).unlink(missing_ok=True)
        console.print(
            f"[green]市场 {name} 已添加[/green]（{target}，"
            f"收录 {len(catalog.get('plugins', []))} 个插件）\n"
            f"[dim]/plugin market 打开挑选器，或 /plugin install <名字>@{name}[/dim]"
        )
    except RuntimeError as e:
        console.print(f"[red]添加市场失败：{e}[/red]")
    finally:
        if tmp_dir is not None:
            _rmtree_force(tmp_dir, ignore_errors=True)


def _cmd_market_list() -> None:
    """/plugin market list：列出已添加的市场和各自收录的插件。"""
    mkts = _iter_marketplaces()
    if not mkts:
        console.print(
            f"[yellow]还没有添加任何市场[/yellow]（{_marketplaces_root()} 为空）\n"
            "[dim]/plugin market add <git地址> 添加，"
            "然后 /plugin market 挑选安装[/dim]"
        )
        return
    for name, catalog, _dir in mkts:
        plugins = [
            e.get("name", "?") for e in catalog.get("plugins", [])
            if isinstance(e, dict)
        ]
        console.print(
            f"[cyan]{name}[/cyan]（{len(plugins)} 个插件）："
            + ("  ".join(plugins) if plugins else "（空目录册）")
        )


def _cmd_market_update(name: str) -> None:
    """/plugin market update [名字]：刷新市场目录册（不写名字 = 全刷）。

    只能刷新 git 来源的市场（本地复制来的没有远程可拉）——重新浅克隆
    一份替换本地目录。
    """
    mkts = _iter_marketplaces()
    if not mkts:
        console.print("[yellow]还没有添加任何市场[/yellow]")
        return

    targets = [(n, d) for n, _c, d in mkts if not name or n == name]
    if name and not targets:
        console.print(f"[red]没找到市场：{name}（/plugin market list 看看名字）[/red]")
        return

    for mkt_name, mkt_dir in targets:
        url = _git_origin(mkt_dir)
        if url is None:
            console.print(
                f"[yellow]{mkt_name}：本地复制来的市场，没有远程可拉，跳过[/yellow]"
            )
            continue
        tmp_dir = None
        try:
            fresh, tmp_dir = _clone_to_temp(url)
            _rmtree_force(mkt_dir)
            # 整克隆搬过来（.git 留着，下次 update 还要靠它找远程）
            shutil.move(str(fresh), str(mkt_dir))
            catalog = _read_catalog(mkt_dir)
            console.print(
                f"[green]市场 {mkt_name} 已刷新[/green]"
                f"（收录 {len((catalog or {}).get('plugins', []))} 个插件）"
            )
        except RuntimeError as e:
            console.print(f"[red]刷新 {mkt_name} 失败：{e}[/red]")
        finally:
            if tmp_dir is not None:
                _rmtree_force(tmp_dir, ignore_errors=True)


def _cmd_market_remove(name: str) -> None:
    """/plugin market remove <名字>：移除市场目录册（已装插件不动）。"""
    mkts = _iter_marketplaces()
    target = next((d for n, _c, d in mkts if n == name), None)
    if target is None:
        console.print(f"[red]没找到市场：{name}（/plugin market list 看看名字）[/red]")
        return
    answer = console.input(
        f"[bold]移除市场 {name}（只删目录册，已装插件不动）？y/N >[/bold] "
    ).strip().lower()
    if answer not in ("y", "yes"):
        console.print("已取消")
        return
    _rmtree_force(target)
    # 落一个「删过」记号：内置市场不会再自动复活（想恢复 = 重新 add）
    _marketplaces_root().mkdir(parents=True, exist_ok=True)
    _removed_marker(name).write_text("removed", encoding="utf-8")
    console.print(f"[green]市场 {name} 已移除[/green]（{target}）")


def _cmd_market_pick(rt) -> None:
    """市场浏览 + 选号安装（列表式，不走浮窗选择器）。

    交互三步，全部留在对话流里（表格进滚动历史，用完不消失）：
      1. 输入关键字过滤（名字/描述模糊匹配，回车 = 全部）——官方市场
         近 300 个条目，不过滤表格太长
      2. 打印带序号的表格（插件名/市场/描述/已装标记）
      3. 输入序号或名字安装，多个用空格隔开（也能写 名字@市场）；q 取消

    为什么不用方向键选择器：那个要独占终端（挂起主界面），主界面被
    藏起来、选完整块消失不留痕——观感就是「浮窗一闪而过」。
    """
    entries = _iter_market_entries()
    if not entries:
        console.print(
            "[yellow]市场里还没有可挑的插件[/yellow]——"
            "/plugin market add <git地址> 先添加一个市场"
        )
        return

    # ---- 1. 关键字过滤 ----
    kw = console.input(
        "[bold]输入关键字过滤（名字/描述，回车=全部，q 退出） >[/bold] "
    ).strip()
    if kw.lower() in ("q", "quit", "exit"):
        return
    kw_l = kw.lower()
    shown = [
        (m, e) for m, e in entries
        if not kw_l or kw_l in e["name"].lower()
        or kw_l in str(e.get("description", "") or "").lower()
    ]
    if not shown:
        console.print(f"[yellow]没有匹配「{kw}」的插件[/yellow]")
        return

    # ---- 2. 序号表格 ----
    installed = {n for n, _mf, _d in _iter_installed()}
    title = f"市场插件（{len(shown)} 条"
    if kw:
        title += f"，过滤「{kw}」"
    title += "）"
    table = Table(title=title)
    table.add_column("序号", justify="right", style="cyan")
    table.add_column("插件")
    table.add_column("市场")
    table.add_column("描述")
    table.add_column("状态")
    for i, (m, e) in enumerate(shown, 1):
        desc = str(e.get("description", "") or "")[:52]
        table.add_row(
            str(i), e["name"], m, desc,
            "[green]已装[/green]" if e["name"] in installed else "",
        )
    console.print(table)

    # ---- 3. 选号/选名安装 ----
    pick = console.input(
        "[bold]输入序号或名字安装（多个空格隔开，q 取消） >[/bold] "
    ).strip()
    if not pick or pick.lower() in ("q", "quit", "exit"):
        return

    mkts = {mn: d for mn, _c, d in _iter_marketplaces()}
    for tok in pick.split():
        target = None
        if tok.isdigit() and 1 <= int(tok) <= len(shown):
            target = shown[int(tok) - 1]
        else:
            # 名字 或 名字@市场
            name, _, market = tok.partition("@")
            hits = [
                (m, e) for m, e in shown
                if e["name"] == name and (not market or m == market)
            ] or [
                (m, e) for m, e in entries
                if e["name"] == name and (not market or m == market)
            ]
            if not hits:
                console.print(f"[red]没找到：{tok}[/red]")
                continue
            target = hits[0]
        market_name, entry = target
        mkt_dir = mkts.get(market_name)
        if mkt_dir is None:
            continue
        resolved = _resolve_entry_source(mkt_dir, entry)
        if resolved is None:
            continue
        _install_resolved(resolved)


# ---------------------------------------------------------------------------
# 其余子命令
# ---------------------------------------------------------------------------

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
    _unwire_plugin_mcp(name)   # 先拆 MCP 线（目录还在才能算出归属）
    _rmtree_force(path)
    console.print(f"[green]插件 {name} 已卸载[/green]（{path}）")


def _cmd_set_enabled(name: str, enabled: bool) -> None:
    """/plugin enable|disable <名字>：改清单里的 enabled 字段。"""
    path = _resolve_installed(name)
    if path is None:
        return
    if not enabled:
        # 停用前先拆 MCP 线——清单还是 enabled 时才能算出连接归属
        _unwire_plugin_mcp(name)
    manifest = _load_manifest(path)
    if not manifest:
        # 清单读坏了（或手删了）：造个最小清单把名字保住
        manifest = {"name": name}
    manifest["enabled"] = enabled
    _save_manifest(path, manifest)
    if enabled:
        # 启用后接线（清单已置 enabled，扫描能认到它）
        _wire_plugin_mcp(name)
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


def _cmd_list() -> None:
    """/plugin 或 /plugin list：表格展示所有已安装插件。"""
    items = _iter_installed()
    if not items:
        console.print(
            f"[yellow]还没有安装任何插件[/yellow]（目录 {plugins_dir()} 为空）\n"
            "[dim]试试 /plugin market add <git地址> 添加市场后 "
            "/plugin market 挑选安装；或 /plugin create <名字> 搓一个[/dim]"
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


# ---------------------------------------------------------------------------
# 总分发 + 注册
# ---------------------------------------------------------------------------

def _handle_plugin_command(args: str, rt) -> bool:
    """/plugin 总分发：看第一个参数转给对应子命令。

    参数：
        args：空 = 列表；否则第一个词是子命令
              （list/install/uninstall/enable/disable/create/market...）
        rt：RuntimeContext（装/卸/启停后用它重扫技能命令）
    返回：
        bool —— True 表示命令已处理
    """
    parts = args.strip().split()
    if not parts:
        _cmd_list()
        return True

    sub = parts[0].lower()
    rest = " ".join(parts[1:])

    try:
        if sub in ("list", "ls"):
            _cmd_list()
        elif sub == "install":
            _cmd_install(rest)
        elif sub == "market" or sub == "marketplace":
            # market 二级分发：空 = 挑选器；add/list/update/remove
            mparts = rest.strip().split()
            msub = mparts[0].lower() if mparts else ""
            mrest = " ".join(mparts[1:])
            if msub in ("", "browse", "pick", "list"):
                # 浏览/列表前先把内置市场补齐（官方市场开箱即用；
                # 用户删过的不复活，见 _ensure_builtin_marketplaces）
                if msub != "list" or not _iter_marketplaces():
                    _ensure_builtin_marketplaces()
            if msub in ("", "browse", "pick"):
                _cmd_market_pick(rt)
            elif msub == "add":
                if not mrest:
                    console.print("[red]用法：/plugin market add <git地址 | owner/repo | 本地目录>[/red]")
                else:
                    _cmd_market_add(mrest)
            elif msub == "list":
                _cmd_market_list()
            elif msub == "update":
                _cmd_market_update(mrest)
            elif msub == "remove":
                if not mrest:
                    console.print("[red]用法：/plugin market remove <名字>[/red]")
                else:
                    _cmd_market_remove(mrest)
            else:
                console.print(
                    f"[red]未知市场子命令：{msub}[/red]\n"
                    "可用：add, list, update, remove（不带参数 = 打开挑选器）"
                )
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
            _cmd_create(rest)
        else:
            console.print(
                f"[red]未知子命令：{sub}[/red]\n"
                "可用：list, install, uninstall, enable, disable, create, market"
            )
            return True
    except Exception as e:
        # 子命令失败别炸 REPL：打印错误 + 记日志（fail-open 但大声）
        console.print(f"[red]插件命令执行失败：{e}[/red]")
        logger.warning("/plugin %s 失败: %s", sub, e, exc_info=True)
        return True

    # 装了/卸了/启停了才需要重扫（list/纯浏览不折腾；market 挑选器可能装
    # 东西，除了纯 list 都重扫）
    market_first = rest.strip().split()[0].lower() if rest.strip() else ""
    market_touched = (
        sub in ("market", "marketplace") and market_first not in ("", "list")
    )
    if sub in ("install", "uninstall", "enable", "disable", "create") \
            or market_touched:
        _refresh_skills(rt)
    return True


@slash_command(
    name="/plugin", category="插件",
    usage="/plugin [list|install|uninstall|enable|disable|create|market]",
    summary="插件管理（看已装/启停/卸载）；/plugins 逛市场装新的",
    arg_completer=lambda text: [
        s for s in (
            "list", "install ", "uninstall ", "enable ", "disable ",
            "create ", "market ", "market add ", "market list ",
            "market update ", "market remove ",
        ) if s.startswith(text)
    ],
)
def cmd_plugin(args: str, rt) -> bool:
    return _handle_plugin_command(args, rt)


@slash_command(
    name="/plugins", category="插件", usage="/plugins",
    summary="逛插件市场：关键字过滤 + 序号选装（内置官方市场开箱即用）",
)
def cmd_plugins_browse(args: str, rt) -> bool:
    # 直达市场挑选（和 /plugin market 同一条路）；确保内置市场在场
    try:
        _ensure_builtin_marketplaces()
        _cmd_market_pick(rt)
        _refresh_skills(rt)
    except Exception as e:
        console.print(f"[red]逛市场失败：{e}[/red]")
        logger.warning("/plugins 失败: %s", e, exc_info=True)
    return True
