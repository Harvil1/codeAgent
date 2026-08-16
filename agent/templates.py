"""cron 任务模板（R26 #18）。

常用作业写成"菜谱卡片"放固定目录，cron_create(template=名字) 照单点菜，
不用每次口述一遍配置。发现两个目录（项目级覆盖用户级同名）：
  - ~/.OmniMate/templates/*.md
  - <cwd>/.omnimate/templates/*.md
frontmatter 字段：cron（必需）/ message（缺省用 body）/ catch_up / recurring。
对齐 CCB jobs/templates.ts；frontmatter 解析复用 skill_commands.parse_frontmatter。
"""
import logging
from pathlib import Path
from typing import Dict

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)

_cache: dict = {"key": None, "templates": {}}


def _invalidate_cache() -> None:
    """测试/调试用：清缓存。"""
    _cache["key"] = None
    _cache["templates"] = {}


def _template_dirs() -> list:
    dirs = []
    try:
        from constants import get_omnimate_home
        dirs.append(get_omnimate_home() / "templates")
    except Exception:
        pass
    try:
        from agent.workspace_context import get_workspace_cwd
        dirs.append(Path(get_workspace_cwd()) / ".omnimate" / "templates")
    except Exception:
        pass
    return dirs


def load_task_templates() -> Dict[str, dict]:
    """扫描模板目录（mtime+size 双因子缓存）。fail-open。"""
    try:
        stat_key = []
        files = []
        for d in _template_dirs():
            if not d.exists():
                continue
            for md in sorted(d.glob("*.md")):
                files.append(md)
                st = md.stat()
                stat_key.append((str(md), st.st_mtime_ns, st.st_size))
        key = tuple(stat_key)
        if _cache["key"] == key:
            return _cache["templates"]
        result: Dict[str, dict] = {}
        for md in files:  # 后扫的项目级覆盖用户级同名
            try:
                content = md.read_text(encoding="utf-8")
                fm, body = parse_frontmatter(content)
                cron = str(fm.get("cron") or "").strip()
                if not cron:
                    continue
                result[md.stem] = {
                    "cron": cron,
                    "message": str(fm.get("message") or body.strip()),
                    "catch_up": bool(fm.get("catch_up", False)),
                    "recurring": bool(fm.get("recurring", True)),
                }
            except Exception as e:
                logger.debug("模板解析失败 %s: %s", md, e)
        _cache["key"] = key
        _cache["templates"] = result
        return result
    except Exception as e:
        logger.debug("load_task_templates fail-open: %s", e)
        return {}
