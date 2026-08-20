"""发现磁盘上的工作流脚本（R28 W4，蓝图 §2）——找到它们、读出内容、缓存住。

在项目里的位置：给 tools/workflow_tool.py 提供"有哪些现成工作流可跑"的目录。

编排脚本是数据不是代码：放在两个固定目录里（像菜谱放菜谱架）：
  - ~/.OmniMate/workflows/*.py        用户级（跨项目通用）
  - <当前项目>/.omnimate/workflows/*.py  项目级（同名时覆盖用户级那份）

缓存策略：mtime（修改时间）+ size（文件大小）双因子判断"变没变"。
为什么两个一起看：Windows 上修改时间精度只有约 15 毫秒，同一窗口内改
文件光看时间会误判"没变"；加上文件大小一起比对才靠谱（历史踩坑，模式
抄自 agent/templates.py）。
"""
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_cache: dict = {"key": None, "scripts": {}}


def _invalidate_cache() -> None:
    """清空缓存（下次扫描强制重读磁盘）。"""
    _cache["key"] = None
    _cache["scripts"] = {}


def _script_dirs() -> list:
    """列出要扫的目录：用户级 workflows + 当前项目的 .omnimate/workflows。

    返回：目录 Path 列表（某个来源取不到就跳过，不报错）。
    """
    dirs = []
    try:
        from constants import get_omnimate_home
        dirs.append(get_omnimate_home() / "workflows")
    except Exception:
        pass
    try:
        from agent.workspace_context import get_workspace_cwd
        dirs.append(Path(get_workspace_cwd()) / ".omnimate" / "workflows")
    except Exception:
        pass
    return dirs


def load_workflow_scripts() -> Dict[str, str]:
    """扫描工作流目录，返回"脚本名 → 脚本内容"的字典（带缓存）。

    整体 fail-open（坏不了主流程）：目录不存在、文件读不了、任何意外
    都只是跳过或返回空，绝不抛错。

    参数：无。
    返回：{文件名（去 .py）: 脚本文本}。项目级目录后扫，同名会顶掉
        用户级的——这就是"项目覆盖用户"的实现方式。
    """
    try:
        stat_key = []
        files = []
        for d in _script_dirs():
            if not d.exists():
                continue
            for py in sorted(d.glob("*.py")):
                files.append(py)
                st = py.stat()
                stat_key.append((str(py), st.st_mtime_ns, st.st_size))
        key = tuple(stat_key)
        if _cache["key"] == key:
            return _cache["scripts"]
        result: Dict[str, str] = {}
        for py in files:
            try:
                result[py.stem] = py.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug("workflow 脚本读取失败 %s: %s", py, e)
        _cache["key"] = key
        _cache["scripts"] = result
        return result
    except Exception as e:
        logger.debug("load_workflow_scripts fail-open: %s", e)
        return {}
