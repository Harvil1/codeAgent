"""常量和路径函数。

无依赖的底层模块，提供 agent home 等路径解析。
支持 AGENT_HOME 环境变量覆盖默认路径（测试/开发用）。

Windows 桌面应用定位：
  - 数据目录：%APPDATA%\\HermesAgent\\（用户配置、数据库、记忆）
  - 日志目录：%LOCALAPPDATA%\\HermesAgent\\logs\\（可重建、可清理）
  - 老目录（~/.agent）首次启动时自动迁移（Windows 上）

Linux/macOS：
  - 数据目录：~/.agent/
  - 日志目录：~/.agent/logs/
"""

import os
import sys
from pathlib import Path


def _default_agent_home() -> Path:
    """根据平台返回默认数据目录。

    优先级：
      1. AGENT_HOME 环境变量（覆盖默认，测试/开发用）
      2. Windows: %APPDATA%\\HermesAgent\\
      3. Linux/macOS: ~/.agent/
    """
    env_override = os.environ.get("AGENT_HOME")
    if env_override:
        return Path(env_override).expanduser()

    if sys.platform == "win32":
        # %APPDATA% = C:\\Users\\<user>\\AppData\\Roaming
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "HermesAgent"
        # 兜底（APPDATA 罕见缺失，但 Python 启动时若被刻意清空 env 会用到）
        return Path.home() / "AppData" / "Roaming" / "HermesAgent"

    # Linux/macOS 保持旧行为
    return Path.home() / ".agent"


def _default_logs_dir() -> Path:
    """日志目录（Windows 独立到 LOCALAPPDATA，便于卸载清理）。"""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "HermesAgent" / "logs"
        return Path.home() / "AppData" / "Local" / "HermesAgent" / "logs"
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
