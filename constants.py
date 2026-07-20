"""常量和路径函数。

无依赖的底层模块，提供 agent home 等路径解析。
支持 AGENT_HOME 环境变量覆盖默认路径（测试/开发用）。

所有数据（API key、记忆、工具、技能、会话、任务）统一在 ~/.agent/ 下：
  - Linux/macOS: ~/.agent/
  - Windows:     C:\\Users\\<user>\\.agent\\

设计原则：跨平台目录一致，避免自动迁移带来的路径漂移。
未来若做成 Windows 安装包，可重新启用 AppData 定位 + 一次性迁移。
"""

import os
from pathlib import Path


def _default_agent_home() -> Path:
    """返回默认数据目录。

    优先级：
      1. AGENT_HOME 环境变量（覆盖默认，测试/开发用）
      2. ~/.agent/（跨平台一致）
    """
    env_override = os.environ.get("AGENT_HOME")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".agent"


def _default_logs_dir() -> Path:
    """日志目录（统一在 agent home 下，便于排查）。"""
    return _default_agent_home() / "logs"


def get_agent_home() -> Path:
    """获取 agent home 目录。

    每次调用都重新检查 AGENT_HOME 环境变量（测试时可临时覆盖）。
    """
    return _default_agent_home()


def display_agent_home() -> str:
    """用于显示给用户的路径字符串。"""
    return str(get_agent_home())


def skills_dir() -> Path:
    """技能库根目录（每个子目录是一个技能）。"""
    return get_agent_home() / "skills"


def logs_dir() -> Path:
    """日志目录（Windows 上独立到 LOCALAPPDATA）。"""
    return _default_logs_dir()


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
