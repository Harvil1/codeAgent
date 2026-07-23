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

                commands[f"/{cmd_name}"] = {
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "skill_md_path": str(skill_md),
                    "skill_dir": str(skill_md.parent),
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


def execute_skill(
    skill_md_path: str,
    user_message: str,
) -> str:
    """加载技能内容，作为 user 消息返回。

    技能正文 + 用户的原始消息组合成新的 user 消息。
    这保证了不修改 system prompt（保护 prompt cache）。
    """
    path = Path(skill_md_path)
    if not path.exists():
        return user_message

    content = path.read_text(encoding="utf-8")
    _, body = parse_frontmatter(content)

    # 把技能指令 + 用户消息组合
    return (
        f"[技能已加载]\n\n"
        f"{body.strip()}\n\n"
        f"---\n"
        f"用户消息: {user_message}"
    )


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
