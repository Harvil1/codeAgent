"""cron 任务模板（R26 #18）——把常用的定时作业写成"菜谱卡片"，照单点菜即可。

cron（定时调度）的基础知识：让程序按时间表自动干活，比如"每天早上 9 点
跑一次日报"。以前每次创建定时任务都得口述一遍配置；有了模板，写一张
卡片放固定目录，cron_create(template=名字) 直接按卡片下单。

在项目里的位置：给 tools/cron_tool.py 提供模板目录；解析 frontmatter
（Markdown 顶部 --- 包起来的元数据块）复用 agent/skill_commands.py 的
parse_frontmatter，不重复造轮子。对齐 CCB 的 jobs/templates.ts。

扫两个目录（项目级同名覆盖用户级）：
  - ~/.OmniMate/templates/*.md        用户级（跨项目通用）
  - <当前项目>/.omnimate/templates/*.md  项目级

卡片 frontmatter 字段：
  - cron：时间表（必需，缺了这张卡片直接不收）
  - message：到点要发给 agent 的话（不写就用正文 body 代替）
  - catch_up：错过了要不要补跑（默认不补）
  - recurring：是不是重复任务（默认是）
"""
import logging
from pathlib import Path
from typing import Dict

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)

_cache: dict = {"key": None, "templates": {}}


def _invalidate_cache() -> None:
    """清空缓存（测试/调试用，下次扫描强制重读磁盘）。"""
    _cache["key"] = None
    _cache["templates"] = {}


def _template_dirs() -> list:
    """列出要扫的模板目录：用户级 templates + 当前项目的 .omnimate/templates。

    返回：目录 Path 列表（某个来源取不到就跳过，不报错）。
    """
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
    """扫描模板目录，返回"模板名 → 配置 dict"（带缓存，整体 fail-open）。

    缓存用 mtime+size 双因子（修改时间+文件大小）判断"变没变"——Windows
    上修改时间精度只有约 15 毫秒，光看时间会误判"没变"，要加上大小一起
    比对才靠谱。fail-open：目录不存在、卡片解析失败、任何意外都跳过或
    返回空，绝不抛错。

    参数：无。
    返回：{文件名（去 .md）: {cron, message, catch_up, recurring}}；
        没有 cron 字段的卡片不收。
    """
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
        for md in files:  # 项目级目录排在后面后扫，同名卡片自然顶掉用户级的
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
