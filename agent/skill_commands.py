"""技能扫描与 slash 命令映射。

技能（沉淀成 Markdown 文件的「怎么做」知识，AI 可以自己创建和改进——这是自学习的关键）
每个技能是一个目录，里面放一个 SKILL.md。本文件在启动时扫描所有技能目录，
把每个 SKILL.md 映射成一条 /<技能名> 斜杠命令，供用户在 CLI 里敲。

给谁用：cli.py（启动建命令表）和主循环（触发技能时拼消息）用。

关键设计（历史共识，别破坏）：技能正文走 user 消息注入，绝不进 system prompt！
system prompt 在会话开始时构建一次，之后改一个字都会让前缀缓存失效、token 成本翻倍；
而技能内容经常变，所以放在 user 消息里，改技能不碰缓存。
"""

import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import yaml

logger = logging.getLogger(__name__)

# 技能名里不合法的字符（只留小写字母、数字、连字符，其余全删）
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")


def scan_skill_commands(skills_dirs) -> Dict[str, dict]:
    """把技能目录扫一遍，生成 {"/命令名": 技能信息} 字典。

    背景：CLI 的斜杠命令不是手写死的，而是从技能目录里现场扫出来的——
    每个带 SKILL.md 的子目录就是一条 /<技能名> 命令。

    参数：
        skills_dirs：技能目录。可以传一个目录（Path 或字符串），
            也可以传一个目录列表。传列表时按顺序扫，后面的会覆盖前面的
            同名技能——典型用法 [内置目录, 用户目录]，这样用户可以
            用自己的版本顶掉内置版本。

    返回：{"/命令名": {...}}，每项包含：
      - name：技能名
      - description：描述（来自 frontmatter，没有就空串）
      - skill_md_path：SKILL.md 文件路径
      - skill_dir：技能所在目录
      - context / files：frontmatter 里的扩展声明（见下方行内注释）
    """
    # 传单个目录时也包成列表，统一走循环（老调用方兼容）
    if isinstance(skills_dirs, (str, Path)):
        skills_dirs = [skills_dirs]

    commands = {}
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            continue

        for skill_md in skills_dir.glob("*/SKILL.md"):
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, body = parse_frontmatter(content)

                name = frontmatter.get("name", skill_md.parent.name)

                # 命令名统一成「小写+连字符」，避免空格/下划线/大写混出多条命令
                cmd_name = name.lower().replace(" ", "-").replace("_", "-")
                cmd_name = _SKILL_INVALID_CHARS.sub("", cmd_name)

                if not cmd_name:
                    continue

                # frontmatter 写 user-invocable: false → 不注册成用户命令
                # （用户敲不出来，但模型仍可通过 load_skill 自动用它）
                if frontmatter.get("user-invocable", True) is False:
                    continue

                commands[f"/{cmd_name}"] = {
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "skill_md_path": str(skill_md),
                    "skill_dir": str(skill_md.parent),
                    "context": frontmatter.get("context"),  # 历史轮次（round3）加的字段：None=主对话内跑，"fork"=开子代理隔离跑
                    "files": frontmatter.get("files") or [],  # 历史轮次（T3）加的字段：触发时要一并注入的参考文件清单
                }
            except Exception as e:
                logger.warning("解析技能失败 %s: %s", skill_md, e)

    return commands


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """把 SKILL.md 开头的 YAML 头拆出来。

    背景：技能文件分两段——开头的「---」围起来的元信息（YAML，叫 frontmatter，
    存技能名/描述/触发条件等）和后面的正文（给 AI 看的操作指引）。

    参数：
        content：SKILL.md 的完整文本。

    返回：(frontmatter 字典, 正文文本)。文件开头不是「---」或 YAML 坏了，
    就返回 ({}, 原文)——宁可当成没有元信息，不让整个技能加载失败。
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
        if not isinstance(frontmatter, dict):
            frontmatter = {}
    except yaml.YAMLError:
        # YAML 语法坏了就退回土办法：逐行按「key: value」硬拆，保证还能读到基本字段
        frontmatter = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    return frontmatter, parts[2]


# ---------------------------------------------------------------------------
# 条件技能动态激活（R19 第 25 项）：frontmatter 写了 paths: 的技能，
# 在被触碰的文件匹配 paths 模式时自动注入提示——不用用户敲命令。
# ---------------------------------------------------------------------------

# frontmatter 摘要缓存：{SKILL.md 路径: ((修改时间纳秒, 文件大小), 摘要字典)}
# 为什么缓存：每次文件读写都会触发一轮匹配扫描，不缓存就得反复解析全部 SKILL.md。
# 为什么用 mtime+size 两个因子：历史踩坑——Windows 的 mtime 精度只有 ~15 毫秒，
# 同一窗口内改文件单看时间会误判「没变过」导致丢更新，加 size 才稳。
_fm_summary_cache: dict = {}


def _iter_skill_fm_summaries(skills_dirs=None):
    """逐个吐出「声明了 paths 触发条件」的技能摘要。

    参数：
        skills_dirs：技能目录（单个或列表）。不传时用项目的全部默认目录
            （constants.all_skills_dirs()），拿不到就直接返回。

    产出（yield）：每个摘要是一个 {name, description, paths} 字典；
    没有 paths 的技能不产出——它们走常规索引，不归这里管。
    """
    if skills_dirs is None:
        try:
            from constants import all_skills_dirs
            skills_dirs = all_skills_dirs()
        except Exception:
            return
    if isinstance(skills_dirs, (str, Path)):
        skills_dirs = [skills_dirs]
    for sd in skills_dirs:
        sd = Path(sd)
        if not sd.exists():
            continue
        for skill_md in sd.glob("*/SKILL.md"):
            try:
                stat = skill_md.stat()
                cache_key = (stat.st_mtime_ns, stat.st_size)
                cached = _fm_summary_cache.get(str(skill_md))
                if cached is not None and cached[0] == cache_key:
                    fm = cached[1]
                else:
                    fm_raw, _body = parse_frontmatter(
                        skill_md.read_text(encoding="utf-8")
                    )
                    fm = {
                        "name": fm_raw.get("name", skill_md.parent.name),
                        "description": fm_raw.get("description", ""),
                        "paths": fm_raw.get("paths") or [],
                    }
                    if len(_fm_summary_cache) > 1000:
                        # 防无界膨胀：缓存太大就整体清空重来（重新解析一遍，代价可接受）
                        _fm_summary_cache.clear()
                    _fm_summary_cache[str(skill_md)] = (cache_key, fm)
                if fm["paths"]:
                    yield fm
            except Exception:
                continue


def path_matches_skill_paths(paths, file_path: str, cwd: str = None) -> bool:
    """判断某个文件路径是否命中技能声明的 paths 模式。

    背景：这是条件技能的核心判定——文件读写一发生，就拿被碰的文件路径
    对照各技能的 paths 列表，命中就注入该技能的提示。语义对齐 Claude Code
    的 parseSkillPaths（只按文件路径匹配，不看别的东西）。

    参数：
        paths：技能 frontmatter 里的模式列表（如 ["*.py", "src/**"]）。
        file_path：被触碰的文件路径（绝对或相对都行）。
        cwd：当前工作目录。给了的话，绝对路径会先换算成相对路径再匹配。

    返回：True=命中（该技能应激活）。

    支持三种写法：
    - 纯文件名通配：``*.py`` 匹配 ``main.py``
    - 目录通配：``src/**`` 匹配 src/ 下面任意深度的文件
    - 完整路径或相对路径通配（``**/test_*.py`` 这种）
    """
    import fnmatch
    if not paths or not file_path:
        return False
    norm = str(file_path).replace("\\", "/")
    fname = norm.rsplit("/", 1)[-1]
    cwd_n = str(cwd).replace("\\", "/") if cwd else None
    for pat in paths or []:
        if not isinstance(pat, str) or not pat:
            continue
        p = pat.replace("\\", "/")
        if fnmatch.fnmatch(fname, p):
            return True
        if p.endswith("/**"):
            base = p[:-3].lstrip("/")
            if f"/{base}/" in norm or norm.startswith(base + "/"):
                return True
        if fnmatch.fnmatch(norm, p) or fnmatch.fnmatch(norm, f"*/{p}"):
            return True
        if cwd_n and norm.startswith(cwd_n + "/"):
            rel = norm[len(cwd_n) + 1:]
            if fnmatch.fnmatch(rel, p):
                return True
    return False


def find_conditional_skill_matches(file_path: str, skills_dirs=None) -> list:
    """找出 paths 能匹配到指定文件的条件技能。

    参数：
        file_path：被触碰的文件路径。
        skills_dirs：技能目录（单个或列表）。不传时自动用常规目录
            加上动态发现的嵌套目录（见下）。

    返回：[{name, description, paths}]，只含有 paths 的技能——
    没有 paths 的技能已经在常规索引里，这里只管动态激活专用池。

    历史轮次（R24 第 26 项）：skills_dirs 不传时，除常规目录外还会自动
    并入**动态发现的嵌套技能目录**（从文件所在位置一路向上到 cwd，沿途的
    .omnimate/skills 和 .claude/skills 都算）。

    fail-open：出任何异常都返回空列表，绝不让匹配流程炸掉主对话。
    """
    try:
        cwd = None
        try:
            from agent.workspace_context import get_workspace_cwd
            cwd = get_workspace_cwd()
        except Exception:
            pass
        if skills_dirs is None:
            # 历史轮次（R24 #26）：常规目录 + 嵌套发现目录，后者放后面让同名覆盖生效
            try:
                from constants import all_skills_dirs
                base_dirs = list(all_skills_dirs())
            except Exception:
                base_dirs = []
            skills_dirs = base_dirs + discover_skill_dirs_for_path(file_path, cwd)
        return [
            {"name": fm["name"], "description": fm["description"], "paths": fm["paths"]}
            for fm in _iter_skill_fm_summaries(skills_dirs)
            if path_matches_skill_paths(fm["paths"], file_path, cwd)
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 动态技能目录发现（R24 第 26 项，对齐 Claude Code 的 discoverSkillDirsForPaths）
# ---------------------------------------------------------------------------

# 认可的嵌套技能目录名（自家 .omnimate/skills 为主；.claude/skills 是为了兼容 CC 的项目结构）
_NESTED_SKILL_DIR_NAMES = (".omnimate/skills", ".claude/skills")
# 向上找时跳过这些目录——node_modules 里被人塞一个技能也不该被信任（防投毒）
_SKIP_DIR_NAMES = {"node_modules", ".git", "__pycache__", ".venv", "venv"}


def discover_skill_dirs_for_path(file_path, cwd=None) -> list:
    """从文件所在目录一路向上走到 cwd，把沿途的嵌套技能目录找出来。

    背景：技能不一定只放在全局目录——项目里任何子目录下的
    .omnimate/skills 或 .claude/skills 都算「本项目这块区域专属技能」。

    参数：
        file_path：出发的文件路径（相对路径会按 cwd 补成绝对路径）。
        cwd：向上走的终点。不传时自动取当前工作区目录。

    返回：目录 Path 列表，可能为空。顺序是深路径优先——离文件越近的排
    越前，这样同名技能「近的覆盖远的」（对齐 CC 的覆盖语义）。
    路径里穿过 node_modules 等跳过目录的不收（防投放）。
    文件不在 cwd 内时返回空（如 ~/.OmniMate 的落盘文件不适用本机制）。
    fail-open：任何异常返回空列表。
    """
    try:
        if not file_path:
            return []
        if cwd is None:
            try:
                from agent.workspace_context import get_workspace_cwd
                cwd = get_workspace_cwd()
            except Exception:
                return []
        if not cwd:
            return []
        f = Path(str(file_path))
        if not f.is_absolute():
            f = Path(cwd) / f
        cwd_p = Path(cwd).resolve()
        f_resolved = f.resolve()
        if not f_resolved.is_relative_to(cwd_p):
            return []  # 文件在 cwd 外（如 ~/.OmniMate 的大输出落盘文件）没有「向上到 cwd」的概念，不适用
        found = []
        for parent in f_resolved.parents:
            for name in _NESTED_SKILL_DIR_NAMES:
                cand = parent / name
                if not cand.is_dir():
                    continue
                try:
                    parts = set(cand.relative_to(cwd_p).parts)
                except ValueError:
                    parts = set()
                if parts & _SKIP_DIR_NAMES:
                    continue  # 目录夹在 node_modules 等跳过目录下面，不信任它（防投放）
                if cand not in found:
                    found.append(cand)
            if parent == cwd_p:
                break
        return found
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 技能附件（历史轮次 T3，核心机制对齐第 3 项）：
# frontmatter 写 files: 即可声明参考文件，技能触发时一并注入。
# ---------------------------------------------------------------------------

MAX_ATTACHMENT_FILES = 5                 # 一个技能最多带 5 个附件（防塞爆上下文）
DEFAULT_ATTACHMENT_MAX_CHARS = 8000      # 单个附件最多读 8000 字符（config 可调）


def read_skill_attachment_files(
    skill_dir, files, *, max_chars: int = DEFAULT_ATTACHMENT_MAX_CHARS,
) -> list:
    """读取技能 frontmatter 里 files: 声明的参考文件（触发时随技能一并注入）。

    参数：
        skill_dir：技能所在目录，files 里的相对路径以它为基准解析。
        files：声明的文件清单（字符串或列表）。可以是 None（无附件）。
        max_chars：单文件读取的字符上限（keyword-only，默认 8000，
            可用 config skills.file_attachment_max_chars 调）。

    返回：[{"path": 相对路径, "content": 文件文本}]；无附件返回 []。

    安全与兜底：
    - 用 ../ 逃出技能目录的路径直接跳过（防路径穿越）
    - 超长文件截到 max_chars
    - 最多带 MAX_ATTACHMENT_FILES（5）个
    - 文件不存在时生成「(文件缺失，已跳过)」占位条目，不让一个坏声明
      拖垮整个技能加载（fail-open）
    """
    if not files:
        return []
    if isinstance(files, str):
        files = [files]
    base = Path(skill_dir).resolve()
    items = []
    for rel in list(files)[:MAX_ATTACHMENT_FILES]:
        try:
            rel = str(rel).strip()
            if not rel:
                continue
            p = (base / rel).resolve()
            if base not in p.parents and p != base:
                continue  # 解析完跑到技能目录外面去了（../ 穿越），不读
            if not p.exists() or not p.is_file():
                items.append({"path": rel, "content": "(文件缺失，已跳过)"})
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[附件截断]"
            items.append({"path": rel, "content": content})
        except Exception as e:
            logger.debug("技能附件读取失败 %s: %s", rel, e)
    return items


def format_skill_attachments(items: list) -> str:
    """把附件列表拼成一段可注入对话的文本。

    参数：
        items：read_skill_attachment_files 返回的 [{"path", "content"}] 列表。

    返回：拼好的 Markdown 文本块；空列表返回空串。
    """
    if not items:
        return ""
    blocks = []
    for it in items:
        blocks.append(f"### {it['path']}\n```\n{it['content']}\n```")
    return (
        "[技能附件：frontmatter files: 声明的参考文件，触发时一并注入]\n\n"
        + "\n\n".join(blocks)
    )


def execute_skill(
    skill_md_path: str,
    user_message: str,
) -> str:
    """技能触发时拼装最终注入的 user 消息。

    参数：
        skill_md_path：SKILL.md 文件路径（文件不存在就直接返回原消息）。
        user_message：用户的原始消息文本。

    返回：拼好的新 user 消息 = 技能正文 + 附件（frontmatter files: 声明的
    参考文件）+ 用户原话。

    为什么拼在 user 消息里：system prompt 会话开始后动一下都会让前缀缓存
    失效、成本翻倍，所以技能内容一律走 user 消息（prompt cache 保护铁律）。
    """
    path = Path(skill_md_path)
    if not path.exists():
        return user_message

    content = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    attach = format_skill_attachments(
        read_skill_attachment_files(path.parent, frontmatter.get("files"))
    )

    # 三段拼装：技能正文 → 附件 → 用户原话，中间空行分隔
    parts = [f"[技能已加载]\n\n{body.strip()}"]
    if attach:
        parts.append(attach)
    parts.append(f"---\n用户消息: {user_message}")
    return "\n\n".join(parts)


def scan_bundle_commands(skills_dir: Path) -> Dict[str, dict]:
    """把配置文件里的技能束也注册成斜杠命令。

    背景：技能束（一次打包加载多个技能的配置，见 skill_bundle.py）也是
    一种可触发的东西，所以也要有命令。命令格式是 /bundle:<名字>，
    与普通技能命令 /<名字> 区分开。

    参数：
        skills_dir：技能根目录（本函数保留此参数是为了接口一致，实际用不到）。

    返回：{"/bundle:名字": {name, description, skills, is_bundle: True}}。
    """
    from agent.skill_bundle import load_bundles_config
    bundles = load_bundles_config()
    commands = {}
    for name, cfg in bundles.items():
        cmd_name = f"/bundle:{name}"
        commands[cmd_name] = {
            "name": name,
            "description": cfg.get("description", ""),
            "skills": cfg.get("skills", []),
            "is_bundle": True,
        }
    return commands


def execute_bundle(bundle_name: str, user_message: str, skills_dir: Path) -> str:
    """触发技能束：把束里所有技能正文合并成一条 user 消息。

    参数：
        bundle_name：技能束名。
        user_message：用户的原始消息。
        skills_dir：技能根目录（找 SKILL.md 用）。

    返回：拼好的 user 消息。束不存在或缺技能时也会返回带提示的文本，
    不会抛异常。
    """
    from agent.skill_bundle import load_bundle
    result = load_bundle(bundle_name, skills_dir)
    if "error" in result:
        return f"[技能束加载失败: {result['error']}]\n\n用户消息: {user_message}"

    body = result.get("body", "")
    loaded = result.get("skills_loaded", [])
    missing = result.get("skills_missing", [])

    parts = [f"[技能束已加载: {bundle_name}（{len(loaded)} 个技能）]\n"]
    if missing:
        parts.append(f"[注意: {len(missing)} 个技能未找到: {', '.join(missing)}]\n\n")
    parts.append(f"{body}\n\n---\n用户消息: {user_message}")
    return "".join(parts)
