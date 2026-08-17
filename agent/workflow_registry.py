"""workflows 目录发现（R28 W4，蓝图 §2）。

编排脚本是数据：~/.OmniMate/workflows/*.py（用户级）+
<cwd>/.omnimate/workflows/*.py（项目级，覆盖同名）。
mtime+size 双因子缓存（复刻 agent/templates.py 模式）。
"""
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_cache: dict = {"key": None, "scripts": {}}


def _invalidate_cache() -> None:
    _cache["key"] = None
    _cache["scripts"] = {}


def _script_dirs() -> list:
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
    """扫描目录（fail-open）。项目级后扫覆盖用户级同名。"""
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
