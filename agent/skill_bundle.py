"""技能束（Skill Bundle）：把一组技能打包，一次全加载。

打个比方：技能是单曲，技能束是歌单——点一次歌单，里面所有歌一起放。
配置文件在 ~/.OmniMate/.skill-bundles.json：
    {
      "bundles": {
        "python-dev": {
          "skills": ["pytest", "venv", "mypy"],
          "description": "Python 开发全套"
        }
      }
    }

用法：load_skill(name="bundle:python-dev") 一次性加载一组技能，
把所有技能正文合并成一段返回。

给谁用：agent 侧的 load_skill 工具和 cli 侧的 /bundle:<名字> 命令都走这里。

为什么要有它：不用束的话 AI 得挨个 load_skill 三次、多跑三轮往返；
打包后一次搞定，省 token 也省时间。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


def bundles_config_path() -> Path:
    """返回技能束配置文件的路径（~/.OmniMate/.skill-bundles.json）。

    为什么做成函数而不是常量：agent home 可以被 OMNIMATE_HOME 环境变量
    覆盖，得每次现场算，写死会在切换 profile 时指错地方。
    """
    from constants import get_omnimate_home
    return get_omnimate_home() / ".skill-bundles.json"


def load_bundles_config(config_path: Optional[Path] = None) -> Dict[str, dict]:
    """读取技能束配置文件。

    参数：
        config_path：配置文件路径。不传时用默认的 ~/.OmniMate/.skill-bundles.json。

    返回：{束名: {"skills": [...], "description": "..."}}；
    文件不存在或 JSON 坏了都返回空字典（fail-open，不打断调用方）。
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
    """加载一个技能束：把束里所有技能的正文读出来合并成一段。

    参数：
        bundle_name：技能束名（配置文件里 bundles 下面的键）。
        skills_dir：技能根目录，束里的技能名按 <skills_dir>/<技能名>/SKILL.md 找。
        config_path：配置文件路径。不传时用默认的 ~/.OmniMate/.skill-bundles.json。

    返回：
        {
            "bundle": "python-dev",              # 束名
            "description": "Python 开发全套",    # 束描述
            "skills_loaded": ["pytest", "venv"], # 成功加载的技能
            "skills_missing": ["nope"],          # 没找到/读失败的技能
            "body": "合并的技能正文",             # 各技能正文拼成的大段文本
        }

    束名不存在时返回 {"error": "...", "available": [...可用的束名]}。
    个别技能缺了不影响其他技能——缺的记进 skills_missing，不抛异常。
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
    """列出所有配置的技能束。

    参数：
        config_path：配置文件路径。不传时用默认路径。

    返回：{束名: {"skills": [...], "description": "...", "skill_count": 数量}}。
    只给元信息（名字/包含哪些技能/描述），不含技能正文——列表场景不需要
    拖着大段文本跑。
    """
    bundles = load_bundles_config(config_path)
    result = {}
    for name, cfg in bundles.items():
        result[name] = {
            "skills": cfg.get("skills", []),
            "description": cfg.get("description", ""),
            "skill_count": len(cfg.get("skills", []) or []),
        }
    return result
