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
