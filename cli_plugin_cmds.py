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

# 上一次 /plugins 打印的序号表 [(市场名, 条目dict), ...]，给
# 「/plugin install <序号>」当索引用——进程内会话状态，重启即清。
# 为什么要有它：序号比长插件名好敲（42crunch-api-security-testing
# 谁也不想手打）。
_last_pick_table: list = []


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
    """插件名/市场名消毒：只留安全字符，且必须还是"一个文件名"。

    点号在正则白名单里（版本号要用），所以纯点号（"."、".."）能活着
    穿过字符替换——拼目录时它们会走出插件根（".." 就是 agent home
    本身），覆盖安装时等于整锅端走，这里必须拦下回退默认名。
    """
    cleaned = re.sub(r"[^\w.-]", "-", str(name or "").strip())
    if (
        not cleaned
        or cleaned.strip(".") == ""          # "." / ".." / "..."
        or Path(cleaned).name != cleaned      # 带分隔符或盘符形态
    ):
        return "unnamed-plugin"
    return cleaned


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
                logger.warning("断开插件 MCP %s 失败（重启后自然干净）: %s", conn, e)
    except Exception as e:
        logger.warning("插件 %s 的 MCP 拆线检查失败: %s", plugin_name, e)


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


# CC 工具名适配的两张表（安装外来插件时改写 .md 用）——跟
# tools/registry.py 的 _CC_TOOL_ALIASES 同一套映射
_CC_UNIQUE_TOKENS = {   # 全词替换安全（本语料里只会是工具名）
    "TodoWrite": "task_create",
    "WebFetch": "web_fetch",
    "AskUserQuestion": "ask_user",
    "NotebookEdit": "notebook_edit",
    "EnterPlanMode": "plan_mode_v2_dispatch",
    "ExitPlanMode": "exit_plan_mode",
}
_CC_CONTEXT_TOOLS = {   # 有歧义（也是普通英文词）：只换明确工具语境
    "Task": "delegate_task", "Agent": "delegate_task",
    "Read": "read_file", "Write": "write_file",
    "Edit": "str_replace", "Update": "str_replace",
    "Grep": "search_files", "Glob": "glob",
    "Bash": "terminal", "Skill": "load_skill",
}


def _adapt_cc_plugin_md(plugin_dir) -> int:
    """把外来插件 .md 里的 claude code 专属写法改成本项目的。

    三类替换：
    1. 产品名：Claude Code→CodeAgent、CLAUDE.md→CODEAGENT.md（全词安全）
    2. 独特工具名（TodoWrite 等 CamelCase 独此一家）：全词替换
    3. 歧义工具名（Task/Read/Write 也是普通英文词）：只换 `反引号包裹`
       和 "X tool" 两种明确指工具的语境——"Write design doc" 这种
       普通动词句不动
    返回替换总处数（0 = 本来就适配/没有此类写法）。
    """
    total = 0
    for md in Path(plugin_dir).rglob("*.md"):
        try:
            t = md.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = t
        t = t.replace("Claude Code", "CodeAgent")
        t = t.replace("CLAUDE.md", "CODEAGENT.md")
        for cc, ours in _CC_UNIQUE_TOKENS.items():
            total += len(re.findall(rf"\b{cc}\b", t))
            t = re.sub(rf"\b{cc}\b", ours, t)
        for cc, ours in _CC_CONTEXT_TOOLS.items():
            n1 = t.count(f"`{cc}`")
            t = t.replace(f"`{cc}`", f"`{ours}`")
            pat = rf"\b{cc} tool\b"
            n2 = len(re.findall(pat, t))
            t = re.sub(pat, f"{ours} tool", t)
            total += n1 + n2
        if t != orig:
            md.write_text(t, encoding="utf-8")
    return total


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

    # 外来插件（claude code 生态）的 .md 自动适配：工具名/产品名改写
    # 成本项目的（已适配的插件 0 命中，无副作用）
    try:
        n_adapt = _adapt_cc_plugin_md(target)
        if n_adapt:
            console.print(
                f"[dim]已适配 {n_adapt} 处 claude code 专属写法"
                "（工具名/文档名 → 本项目对应）[/dim]"
            )
    except Exception as e:
        logger.warning("CC 适配改写失败（不影响安装）: %s", e)

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
        logger.warning("异常被吞(fail-open)", exc_info=True)
    return None


def _install_market_entry(market_name: str, entry: dict) -> None:
    """按（市场名, 目录册条目）解析来源并安装（序号/裸名/带市场名共用）。"""
    mkt_dir = next(
        (d for mn, _c, d in _iter_marketplaces() if mn == market_name),
        None,
    )
    if mkt_dir is None:
        return
    resolved = _resolve_entry_source(mkt_dir, entry)
    if resolved is None:
        return
    _install_resolved(resolved)


def _cmd_install(source: str) -> None:
    """/plugin install <来源>：装插件，一次可以装多个（空格隔开）。

    每个来源的写法：
    - 序号：3 —— 查上次 /plugins 打印的序号表（比敲长名字省事）
    - 名字@市场：demo@my-plugins —— 指明从哪个市场装
    - 名字：demo —— 在所有市场里找，唯一命中才装；重名会列出候选
    - 本地路径：./my-plugin（目录里要有 plugin.json）
    - git 地址：https://gitee.com/xxx/plugin.git（任意 git 托管都行）
    - owner/repo 简写：anthropics/claude-code → 按 GitHub 处理
    """
    toks = source.strip().strip('"').strip("'").split()
    if not toks:
        console.print(
            "[red]用法：/plugin install <序号 | 名字 | 名字@市场 | "
            "本地目录 | git地址 | owner/repo>（多个空格隔开）[/red]"
        )
        return
    if len(toks) > 1:
        # 多来源语法优先：逐个装（所以本地路径带空格的场景不支持）
        for t in toks:
            _cmd_install(t)
        return
    src = toks[0]

    # ---- 序号：上次 /plugins 表格的行号 ----
    if src.isdigit():
        i = int(src)
        if 1 <= i <= len(_last_pick_table):
            market_name, entry = _last_pick_table[i - 1]
            _install_market_entry(market_name, entry)
        else:
            console.print(
                f"[red]序号 {src} 不在上次列表里[/red]"
                "（先 /plugins 看表；重启后序号表会清空）"
            )
        return

    # ---- 名字@市场：目录册解析后统一安装 ----
    if "@" in src and "://" not in src and not src.startswith("git@"):
        name, _, market = src.partition("@")
        resolved = _resolve_market_source(name.strip(), market.strip())
        if resolved is None:
            return
        _install_resolved(resolved)
        return

    # ---- 本地路径 / git 地址 / owner-repo ----
    if "://" in src or src.startswith("git@") or _OWNER_REPO_RE.match(src) \
            or Path(src).expanduser().is_dir():
        tmp_dir = None
        try:
            if "://" in src or src.startswith("git@"):
                src_root, tmp_dir = _clone_to_temp(src)
            elif _OWNER_REPO_RE.match(src):
                src_root, tmp_dir = _clone_to_temp(
                    f"https://github.com/{src}.git"
                )
            else:
                src_root = Path(src).expanduser().resolve()
                if not src_root.is_dir():
                    console.print(f"[red]目录不存在：{src_root}[/red]")
                    return
            _install_dir(src_root)
        except RuntimeError as e:
            console.print(f"[red]安装失败：{e}[/red]")
        finally:
            if tmp_dir is not None:
                _rmtree_force(tmp_dir, ignore_errors=True)
        return

    # ---- 裸名字：在所有市场里找 ----
    hits = [(m, e) for m, e in _iter_market_entries() if e["name"] == src]
    if not hits:
        console.print(
            f"[red]不认得这个来源：{src}[/red]"
            "（本地目录不存在、也不是市场里的插件名；/plugins 看看目录）"
        )
        return
    if len(hits) > 1:
        console.print(f"[yellow]「{src}」在多个市场都有，写明哪个：[/yellow]")
        for m, _e in hits:
            console.print(f"  {src}@{m}")
        return
    _install_market_entry(hits[0][0], hits[0][1])


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

    默认值在 config.py::DEFAULT_CONFIG：第一名是仓库内置的本地市场
    （path 条目，离线可用），第二名是 claude code 官方市场（url 条目，
    GitHub）。用户可以在 settings.json 里覆盖（比如换镜像地址）。
    配置读不到时用代码里这份兜底——保证极端情况下开箱即用。

    返回：[{"name", "url"} 或 {"name", "path"}]，顺序即优先级。
    """
    try:
        from config import load_config
        defs = (load_config().get("plugins", {}) or {}).get(
            "builtin_marketplaces"
        )
        if isinstance(defs, list) and defs:
            out = [
                {
                    "name": str(d.get("name") or ""),
                    **({"url": str(d.get("url"))} if d.get("url") else {}),
                    **({"path": str(d.get("path"))} if d.get("path") else {}),
                }
                for d in defs
                if isinstance(d, dict) and (d.get("url") or d.get("path"))
            ]
            if out:
                return out
    except Exception as e:
        logger.warning("读内置市场配置失败，用代码兜底: %s", e)
    return [
        {
            "name": "claude-plugins-official",
            "url": "https://gitee.com/hong-wei-h/code-agent-plugin-official.git",
        },
    ]


def _removed_marker(name: str) -> Path:
    """「用户删过这个市场」的记号文件路径（防内置市场删了又复活）。"""
    return _marketplaces_root() / f".removed-{name}"


def _ensure_builtin_marketplaces() -> None:
    """把内置市场补齐：还没添加过的自动添加（失败不吵，fail-open）。

    开箱即用语义：新用户第一次 /plugins 浏览就能看到市场——本地
    内置市场（离线必成）打底，官方 GitHub 市场做补充（拉不动只
    跳过一行提示，不挡本地市场）。path 条目按项目根解析（随仓库
    走的目录）；url 条目走 git 克隆。
    用户主动 remove 过的（有记号）不再复活——尊重用户的删除。
    """
    try:
        existing = {n for n, _c, _d in _iter_marketplaces()}
        from constants import project_root
        for d in _builtin_marketplace_defs():
            name = _sanitize_name(d.get("name") or "")
            if not name or name in existing:
                continue
            if _removed_marker(name).exists():
                continue
            if d.get("path"):
                # 本地市场：仓库里的目录，复制即添加（无网络需求）
                src = Path(d["path"])
                if not src.is_absolute():
                    src = project_root() / src
                if not src.is_dir():
                    logger.warning("内置本地市场目录不存在: %s", src)
                    continue
                console.print(f"[dim]首次使用：装载内置市场 {name}...[/dim]")
                _cmd_market_add(str(src))
            elif d.get("url"):
                console.print(f"[dim]首次使用：拉取内置市场 {name}...[/dim]")
                _cmd_market_add(d["url"])
    except Exception as e:
        logger.warning("内置市场添加失败（fail-open）: %s", e)


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


def _cmd_market_pick(rt, keyword: str = "") -> None:
    """市场浏览（纯打印，零交互输入——彻底不碰终端让渡/浮窗）。

    只做一件事：按关键字过滤条目 → 打印带序号的表格 → 提示怎么装。
    安装是**下一条斜杠命令**（/plugin install <序号或名字>），在主
    输入框里敲——整个流程没有任何一步要挂起主界面，表格也留在滚动
    历史里不消失。序号表存进 _last_pick_table 供 install 查。

    参数：
        rt：RuntimeContext（本流程不改技能，留着签名统一）
        keyword：过滤关键字（空 = 全部）
    """
    global _last_pick_table
    entries = _iter_market_entries()
    if not entries:
        console.print(
            "[yellow]市场里还没有可挑的插件[/yellow]——"
            "/plugin market add <git地址> 先添加一个市场"
        )
        return

    kw_l = (keyword or "").strip().lower()
    shown = [
        (m, e) for m, e in entries
        if not kw_l or kw_l in e["name"].lower()
        or kw_l in str(e.get("description", "") or "").lower()
    ]
    if not shown:
        console.print(f"[yellow]没有匹配「{keyword}」的插件[/yellow]")
        return

    installed = {n for n, _mf, _d in _iter_installed()}
    title = f"市场插件（{len(shown)} 条"
    if kw_l:
        title += f"，过滤「{keyword.strip()}」"
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
    # 序号表留给 /plugin install <序号> 查
    _last_pick_table = shown
    console.print(
        "[dim]安装：/plugin install <序号或名字>（多个空格隔开，"
        "如 /plugin install 1 3）；过滤重看：/plugins <关键字>[/dim]"
    )


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


def _do_uninstall(name: str) -> None:
    """卸载的核心动作（不带 y/N——命令行走 y/N，浏览器走二次 Enter）。"""
    path = next(
        (p for n, _m, p in _iter_installed() if n == name), None,
    )
    if path is None:
        console.print(f"[red]没找到插件：{name}（/plugin 看看名字）[/red]")
        return
    _unwire_plugin_mcp(name)   # 先拆 MCP 线（目录还在才能算出归属）
    _rmtree_force(path)
    console.print(f"[green]插件 {name} 已卸载[/green]（{path}）")


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
    _do_uninstall(name)


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


# ---------------------------------------------------------------------------
# 内联浏览器入口：/plugin 滚动管理已装；/plugins 先选市场再滚选安装
# ---------------------------------------------------------------------------

def _open_installed_browser() -> None:
    """/plugin：内联浏览器滚动查看已装插件（Enter 两次 = 卸载）。

    列表是 cli_layout 的内联浏览器（普通区块不悬浮）：↑↓ 滚动、
    Enter 第一次亮确认、第二次把卸载命令塞给工作线程执行。
    """
    import cli_layout
    items = _iter_installed()
    if not items:
        console.print(
            f"[yellow]还没有安装任何插件[/yellow]（目录 {plugins_dir()} 为空）\n"
            "[dim]/plugins 逛市场挑着装，或 /plugin create <名字> 搓一个[/dim]"
        )
        return
    entries = []
    for name, mf, path in items:
        state = "启用" if mf.get("enabled", True) else "停用"
        desc = str(mf.get("description", "") or "")[:36]
        meta = f"v{mf.get('version', '?')} · {state} · {_count_skills(path)}技能"
        if desc:
            meta += f" · {desc}"
        entries.append({
            "label": name, "meta": meta,
            "danger": True,
            "confirm_msg": f"再按 Enter 卸载 {name}（Esc 取消）",
            "cmd": f"/plugin __uninstall_now {name}",
        })
    cli_layout.browser_open("已安装插件（Enter 两次 = 卸载）", entries,
                            replace=True)


def _open_market_browser() -> None:
    """/plugins：第一级选市场（Enter 进入），第二级滚选插件安装。

    两级列表都走内联浏览器；Esc 从插件级返回市场级、再按退出。
    """
    import cli_layout
    _ensure_builtin_marketplaces()
    mkts = _iter_marketplaces()
    if not mkts:
        console.print(
            "[yellow]还没有可用的市场[/yellow]——"
            "/plugin market add <git地址> 添加一个"
        )
        return
    items = []
    for name, catalog, _dir in mkts:
        n = len(catalog.get("plugins", []))
        items.append({
            "label": name, "meta": f"{n} 个插件",
            # open 回调当场切到第二级（闭包锁住市场名）
            "open": (lambda nm=name: _open_market_plugins_browser(nm)),
        })
    cli_layout.browser_open("选择市场（Enter 进入，Esc 退出）", items,
                            replace=True)


def _open_market_plugins_browser(market_name: str) -> None:
    """市场浏览器的第二级：某个市场里的插件列表（Enter 安装）。"""
    import cli_layout
    entries = [(m, e) for m, e in _iter_market_entries() if m == market_name]
    if not entries:
        console.print(f"[yellow]市场 {market_name} 里没有条目[/yellow]")
        return
    installed = {n for n, _mf, _d in _iter_installed()}
    items = []
    for m, e in entries:
        tag = "已装 · " if e["name"] in installed else ""
        desc = tag + str(e.get("description", "") or "")[:40]
        items.append({
            "label": e["name"], "meta": desc,
            "cmd": f"/plugin install {e['name']}@{m}",
        })
    # 不带 replace：browser_open 会把市场级列表拍成返回栈（Esc 回得去）
    cli_layout.browser_open(
        f"市场 {market_name}（Enter 安装，Esc 返回市场列表）", items,
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
        # /plugin 裸命令：滚动浏览器管理已装插件（Enter 两次 = 卸载）
        _open_installed_browser()
        return True

    sub = parts[0].lower()
    rest = " ".join(parts[1:])

    try:
        if sub in ("list", "ls"):
            _cmd_list()
        elif sub == "__uninstall_now":
            # 内部命令：浏览器里二次 Enter 已确认的卸载（不再问 y/N）
            if rest:
                _do_uninstall(rest)
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
                # 不认识的词不当错误：当过滤关键字直接浏览
                # （/plugin market design = 过滤含 design 的插件）
                _cmd_market_pick(rt, rest.strip())
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
    if sub in ("install", "uninstall", "__uninstall_now",
               "enable", "disable", "create") \
            or market_touched:
        _refresh_skills(rt)
    return True


def _plugin_arg_completer(frag: str, full_text: str = "") -> list:
    """/plugin 的参数补全：看已敲的子命令决定补什么（配合内联候选区上下滚选）。

    - 还没敲子命令 → 补子命令（带一句描述）
    - install → 补市场里的插件（名字@市场 + 描述）——边打字边过滤，
      ↑↓/Tab 环选、Enter 采纳，等于「市场也用滚动选装」
    - uninstall/enable/disable → 补已装插件名
    """
    toks = (full_text or "").split()
    frag = frag or ""
    try:
        # 算「正在补的是第几个词」：行尾空格 = 补下一个新词，
        # 否则 = 正在敲的最后一个词。位置 1 是子命令，位置 2 起
        # 才轮到各子命令自己的参数
        if (full_text or "").endswith(" "):
            pos = len(toks)
        else:
            pos = len(toks) - 1
        level = toks[1] if (pos >= 2 and len(toks) > 1) else "sub"

        if level == "sub":
            subs = [
                ("list", "看已装插件"), ("install", "装插件（可补市场里的名字）"),
                ("uninstall", "卸载"), ("enable", "启用"), ("disable", "停用"),
                ("create", "搓插件骨架"), ("market", "市场管理"),
            ]
            return [(s, d) for s, d in subs if s.startswith(frag)]
        if level == "install":
            entries = sorted(
                _iter_market_entries(), key=lambda x: x[1]["name"],
            )
            return [
                (f"{e['name']}@{m}",
                 str(e.get("description", "") or "")[:60])
                for m, e in entries
            ]
        if level in ("uninstall", "enable", "disable"):
            return [
                (n, str(mf.get("description", "") or "")[:60])
                for n, mf, _d in _iter_installed()
            ]
    except Exception:
        pass  # 补全挂了不能挡打字
    return []


@slash_command(
    name="/plugin", category="插件",
    usage="/plugin [list|install|uninstall|enable|disable|create|market]",
    summary="插件管理（看已装/启停/卸载）；/plugins 逛市场装新的",
    arg_completer=_plugin_arg_completer,
)
def cmd_plugin(args: str, rt) -> bool:
    return _handle_plugin_command(args, rt)


@slash_command(
    name="/plugins", category="插件", usage="/plugins",
    summary="逛插件市场：先选市场（Enter 进入），再滚动选插件安装",
)
def cmd_plugins_browse(args: str, rt) -> bool:
    # 两级内联浏览器：第一级选市场、第二级滚选插件；Enter 安装的动作
    # 塞回命令队列由工作线程执行（克隆是网络活，不卡 UI）
    try:
        _open_market_browser()
    except Exception as e:
        console.print(f"[red]逛市场失败：{e}[/red]")
        logger.warning("/plugins 失败: %s", e, exc_info=True)
    return True
