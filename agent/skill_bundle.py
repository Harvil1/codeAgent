"""技能束（Skill Bundles）：一次加载多个技能。

配置文件 ~/.OmniMate/.skill-bundles.json：
    {
      "bundles": {
        "python-dev": {
          "skills": ["pytest", "venv", "mypy"],
          "description": "Python 开发全套"
        }
      }
    }

通过 load_skill(name="bundle:python-dev") 一次性加载一组技能，
合并所有技能正文返回。

设计动机：减少 LLM 多次 load_skill 调用（节省往返）。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


def bundles_config_path() -> Path:
    """技能束配置文件路径。"""
    from constants import get_omnimate_home
    return get_omnimate_home() / ".skill-bundles.json"


def load_bundles_config(config_path: Optional[Path] = None) -> Dict[str, dict]:
    """加载技能束配置。

    返回 {bundle_name: {skills: [...], description: "..."}}。
    文件不存在时返回空字典。
    """
    if config_path is None:
        config_path = bundles_config_path()

    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        bundles = data.get("bundles", {}) or {}
        if not isinstance(bundles, dict):
            return {}
        return bundles
    except Exception as e:
        logger.warning("加载技能束配置失败 %s: %s", config_path, e)
        return {}


def load_bundle(
    bundle_name: str,
    skills_dir: Path,
    config_path: Optional[Path] = None,
) -> dict:
    """加载一个技能束里的所有技能，合并正文返回。

    参数：
        bundle_name: 技能束名
        skills_dir: 技能根目录
        config_path: 技能束配置文件（默认 ~/.OmniMate/.skill-bundles.json）

    返回：
        {
            "bundle": "python-dev",
            "description": "Python 开发全套",
            "skills_loaded": ["pytest", "venv"],
            "skills_missing": ["nope"],
            "body": "合并的技能正文",
        }

    如果技能束不存在，返回 {"error": "..."}。
    """
    bundles = load_bundles_config(config_path)

    if bundle_name not in bundles:
        return {
            "error": f"技能束不存在: {bundle_name}",
            "available": list(bundles.keys()),
        }

    bundle_cfg = bundles[bundle_name] or {}
    skill_names = bundle_cfg.get("skills", []) or []
    description = bundle_cfg.get("description", "")

    skills_loaded = []
    skills_missing = []
    body_parts = []

    skills_dir = Path(skills_dir)

    for sname in skill_names:
        skill_md = skills_dir / sname / "SKILL.md"
        if not skill_md.exists():
            skills_missing.append(sname)
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            _, body = parse_frontmatter(content)
            body_parts.append(f"=== 技能: {sname} ===\n{body.strip()}")
            skills_loaded.append(sname)
        except Exception as e:
            logger.warning("加载技能 %s 失败: %s", sname, e)
            skills_missing.append(sname)

    merged_body = "\n\n".join(body_parts)

    return {
        "bundle": bundle_name,
        "description": description,
        "skills_loaded": skills_loaded,
        "skills_missing": skills_missing,
        "body": merged_body,
    }


def list_bundles(config_path: Optional[Path] = None) -> Dict[str, dict]:
    """列出所有配置的技能束（仅元信息，不含正文）。"""
    bundles = load_bundles_config(config_path)
    result = {}
    for name, cfg in bundles.items():
        result[name] = {
            "skills": cfg.get("skills", []),
            "description": cfg.get("description", ""),
            "skill_count": len(cfg.get("skills", []) or []),
        }
    return result
