"""常量和路径函数。

无依赖的底层模块，提供 agent home 等路径解析。
支持 AGENT_HOME 环境变量覆盖默认路径（~/.agent）。
"""

import os
from pathlib import Path


def get_agent_home() -> Path:
    """获取 agent home 目录（支持 AGENT_HOME 环境变量）。"""
    home_env = os.environ.get("AGENT_HOME")
    if home_env:
        return Path(home_env).expanduser()
    return Path.home() / ".agent"


def display_agent_home() -> str:
    """用于显示给用户的路径字符串。"""
    return str(get_agent_home())


def skills_dir() -> Path:
    """技能库根目录（每个子目录是一个技能）。"""
    return get_agent_home() / "skills"


def logs_dir() -> Path:
    """日志目录。"""
    return get_agent_home() / "logs"


def archive_dir() -> Path:
    """归档目录（curator 把不用的技能移到这里，永不删除）。"""
    return get_agent_home() / "skills" / ".archive"


def memory_file() -> Path:
    """MEMORY.md 路径（agent 的笔记：环境事实、项目约定）。"""
    return get_agent_home() / "MEMORY.md"


def user_file() -> Path:
    """USER.md 路径（用户画像：偏好、沟通风格）。"""
    return get_agent_home() / "USER.md"


def config_path() -> Path:
    """config.yaml 路径。"""
    return get_agent_home() / "config.yaml"


def env_file() -> Path:
    """.env 路径（密钥）。"""
    return get_agent_home() / ".env"


def sessions_db_path() -> Path:
    """会话数据库路径。"""
    return get_agent_home() / "sessions.db"
