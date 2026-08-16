"""技能扫描与 slash 命令映射。

启动时扫描技能目录，把每个 SKILL.md 映射成 /<skill-name> 命令。

关键：技能正文作为 user 消息注入，不是 system prompt！
这样修改技能不会破坏 system prompt 缓存。
"""

import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import yaml

logger = logging.getLogger(__name__)

# 技能名合法字符（小写字母、数字、连字符）
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")


def scan_skill_commands(skills_dirs) -> Dict[str, dict]:
    """扫描技能目录(支持单目录或多目录列表),返回 {command_name: skill_info}。

    多目录场景(内置 + 用户):按列表顺序扫描,**后者覆盖前者**。
    典型用法:scan_skill_commands([builtin_dir, user_dir])
    → 同名时 user_dir 的技能覆盖 builtin_dir(用户能改内置)。

    skill_info 包含:
      - name: 技能名
      - description: 描述
      - skill_md_path: SKILL.md 路径
      - skill_dir: 技能目录
    """
    # 兼容单目录输入(Path 或 str)
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

                # 标准化命令名：小写、连字符
                cmd_name = name.lower().replace(" ", "-").replace("_", "-")
                cmd_name = _SKILL_INVALID_CHARS.sub("", cmd_name)

                if not cmd_name:
                    continue

                # user-invocable: false → 不进 slash 命令（对用户隐藏，但模型仍可自动触发）
                if frontmatter.get("user-invocable", True) is False:
                    continue

                commands[f"/{cmd_name}"] = {
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "skill_md_path": str(skill_md),
                    "skill_dir": str(skill_md.parent),
                    "context": frontmatter.get("context"),  # round3: None | "fork"
                    "files": frontmatter.get("files") or [],  # T3: 附件声明
                }
            except Exception as e:
                logger.warning("解析技能失败 %s: %s", skill_md, e)

    return commands


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """解析 YAML frontmatter，返回 (frontmatter, body)。

    使用 pyyaml 精确解析。无 frontmatter 时返回 ({}, content)。
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
        # yaml 解析失败时回退到简单解析
        frontmatter = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    return frontmatter, parts[2]


# ---------------------------------------------------------------------------
# R19 #25：条件技能动态激活（paths 匹配被触碰的文件）
# ---------------------------------------------------------------------------

# frontmatter 摘要缓存 {skill_md_path: ((mtime_ns, size), fm)}
# 文件操作高频触发扫描，mtime+size 缓存避免每次重解析全部 SKILL.md
# （双因子——Windows mtime 精度踩过坑）。
_fm_summary_cache: dict = {}


def _iter_skill_fm_summaries(skills_dirs=None):
    """扫技能目录，yield 有 paths 的 frontmatter 摘要 dict。"""
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
                        _fm_summary_cache.clear()
                    _fm_summary_cache[str(skill_md)] = (cache_key, fm)
                if fm["paths"]:
                    yield fm
            except Exception:
                continue


def path_matches_skill_paths(paths, file_path: str, cwd: str = None) -> bool:
    """文件路径是否匹配技能的 paths 模式（对齐 CC parseSkillPaths 文件语义）。

    匹配形态：
    - 文件名 glob：``*.py`` 匹配 ``main.py``
    - 相对路径 glob：``src/**`` 匹配 src/ 下任意文件（路径段前缀）
    - 全路径/相对 cwd 路径 fnmatch（``**/test_*.py`` 等）
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
    """返回 paths 匹配 file_path 的条件技能 [{name, description, paths}]。

    只返回有 paths 的技能（无 paths 的已在常规索引——动态激活专用）。
    fail-open：任何异常返回空列表。
    """
    try:
        cwd = None
        try:
            from agent.workspace_context import get_workspace_cwd
            cwd = get_workspace_cwd()
        except Exception:
            pass
        return [
            {"name": fm["name"], "description": fm["description"], "paths": fm["paths"]}
            for fm in _iter_skill_fm_summaries(skills_dirs)
            if path_matches_skill_paths(fm["paths"], file_path, cwd)
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# T3（核心机制对齐第 3 项）：frontmatter files: 附件
# ---------------------------------------------------------------------------

MAX_ATTACHMENT_FILES = 5                 # 附件总数上限
DEFAULT_ATTACHMENT_MAX_CHARS = 8000      # 单附件字符上限（config 可调）


def read_skill_attachment_files(
    skill_dir, files, *, max_chars: int = DEFAULT_ATTACHMENT_MAX_CHARS,
) -> list:
    """读 frontmatter files: 声明的参考文件（T3，触发时一并注入）。

    - 路径相对技能目录解析；逃出技能目录（../ traversal）跳过
    - 单文件截到 max_chars（config skills.file_attachment_max_chars）
    - 总数上限 MAX_ATTACHMENT_FILES（5）
    - 缺失文件生成占位条目（提示跳过，fail-open）

    返回 [{"path": 相对路径, "content": 文本}]，无附件返回 []。
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
                continue  # 逃出技能目录，跳过
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
    """附件列表格式化为文本段（slash 注入用）。"""
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
    """加载技能内容，作为 user 消息返回。

    技能正文 + 附件（T3：frontmatter files:）+ 用户的原始消息组合成新的
    user 消息。这保证了不修改 system prompt（保护 prompt cache）。
    """
    path = Path(skill_md_path)
    if not path.exists():
        return user_message

    content = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    attach = format_skill_attachments(
        read_skill_attachment_files(path.parent, frontmatter.get("files"))
    )

    # 把技能指令 + 附件 + 用户消息组合
    parts = [f"[技能已加载]\n\n{body.strip()}"]
    if attach:
        parts.append(attach)
    parts.append(f"---\n用户消息: {user_message}")
    return "\n\n".join(parts)


def scan_bundle_commands(skills_dir: Path) -> Dict[str, dict]:
    """扫描技能束配置，返回 {command_name: bundle_info}。

    技能束命令格式 /bundle:<name>，与普通技能命令 /<name> 区分。
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
    """加载技能束里所有技能的正文，合并成 user 消息返回。"""
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
