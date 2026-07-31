"""常量和路径函数。

无依赖的底层模块，提供 agent home 等路径解析。
支持 OMNIMATE_HOME 环境变量覆盖默认路径（测试/开发用）。

所有数据（API key、记忆、工具、技能、会话、任务）统一在 ~/.OmniMate/ 下：
  - Linux/macOS: ~/.OmniMate/
  - Windows:     C:\\Users\\<user>\\.OmniMate\\

设计原则：跨平台目录一致，避免自动迁移带来的路径漂移。
未来若做成 Windows 安装包，可重新启用 AppData 定位 + 一次性迁移。
"""

import os
from pathlib import Path


def _default_omnimate_home() -> Path:
    """返回默认数据目录。

    优先级：
      1. OMNIMATE_HOME 环境变量（覆盖默认，测试/开发用）
      2. ~/.OmniMate/（跨平台一致）
    """
    env_override = os.environ.get("OMNIMATE_HOME")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".OmniMate"


def _default_logs_dir() -> Path:
    """日志目录（统一在 agent home 下，便于排查）。"""
    return _default_omnimate_home() / "logs"


def get_omnimate_home() -> Path:
    """获取 agent home 目录。

    每次调用都重新检查 OMNIMATE_HOME 环境变量（测试时可临时覆盖）。
    """
    return _default_omnimate_home()


def display_omnimate_home() -> str:
    """用于显示给用户的路径字符串。"""
    return str(get_omnimate_home())


def skills_dir() -> Path:
    """用户技能库根目录(每个子目录是一个技能)。

    用户自己创建/agent 自动创建的技能放这里。跨机器需 rsync 跟随用户数据。
    """
    return get_omnimate_home() / "skills"


def builtin_skills_dir() -> Path:
    """内置技能目录(项目代码自带,跟 git 走)。

    内置技能跟用户数据分离:
    - 内置:本项目 skills/ 目录(开发者维护,装哪台机器都一样)
    - 用户:~/.OmniMate/skills/(用户/agent 维护,跨机器要 rsync)

    同名时用户目录优先(用户可覆盖内置)。
    """
    # 项目根 = constants.py 的父目录(constants.py 在项目根)
    return Path(__file__).resolve().parent / "skills"


def project_root() -> Path:
    """项目根目录(constants.py 所在目录)。

    用于写保护:agent 不能修改项目自身的代码。
    """
    return Path(__file__).resolve().parent


def all_skills_dirs() -> list:
    """所有技能扫描目录(内置 + 用户 + 已启用 plugin,顺序决定优先级)。

    返回 [builtin, user, plugin1/skills, ...],后者覆盖前者
    (用户/plugin 优先于内置)。plugin 通过 plugin.json 的 enabled 字段控制。
    """
    dirs = [builtin_skills_dir(), skills_dir()]
    # 已启用 plugin 的 skills 目录
    plugins_root = plugins_dir()
    if plugins_root.exists():
        import json as _json
        for plugin_dir in sorted(plugins_root.iterdir()):
            if not plugin_dir.is_dir():
                continue
            manifest = plugin_dir / "plugin.json"
            if not manifest.exists():
                continue
            try:
                data = _json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("enabled", True):  # 默认启用
                    skills = plugin_dir / "skills"
                    if skills.is_dir():
                        dirs.append(skills)
            except Exception:
                continue
    return dirs


def plugins_dir() -> Path:
    """Plugin 目录(每个子目录是一个 plugin:plugin.json + skills/)。

    plugin 结构:
        ~/.OmniMate/plugins/<name>/plugin.json   (manifest: name/version/description/enabled)
        ~/.OmniMate/plugins/<name>/skills/<skill>/SKILL.md
    启动时 all_skills_dirs() 扫已启用 plugin 的 skills/。
    """
    return get_omnimate_home() / "plugins"


def logs_dir() -> Path:
    """日志目录（统一在 ~/.OmniMate/logs 下）。"""
    return _default_logs_dir()


def archive_dir() -> Path:
    """归档目录（curator 把不用的技能移到这里，永不删除）。"""
    return get_omnimate_home() / "skills" / ".archive"


def memory_file() -> Path:
    """MEMORY.md 路径（agent 的笔记：环境事实、项目约定）。"""
    return get_omnimate_home() / "MEMORY.md"


def user_file() -> Path:
    """USER.md 路径（用户画像：偏好、沟通风格）。"""
    return get_omnimate_home() / "USER.md"


def config_path() -> Path:
    """config.yaml 路径。"""
    return get_omnimate_home() / "config.yaml"


def env_file() -> Path:
    """.env 路径（密钥）。"""
    return get_omnimate_home() / ".env"


def sessions_db_path() -> Path:
    """会话数据库路径。"""
    return get_omnimate_home() / "sessions.db"
