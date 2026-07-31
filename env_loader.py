"""加载 .env 文件中的环境变量（API key 等）。

使用 python-dotenv，不覆盖已存在的环境变量。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_path: Optional[Path] = None) -> None:
    """加载 .env 文件。

    优先级：已有的环境变量 > .env 文件。
    即 .env 不会覆盖 shell 里已 export 的值。
    """
    if env_path is None:
        # 延迟导入避免循环
        from constants import env_file
        env_path = env_file()

    if env_path and Path(env_path).exists():
        load_dotenv(env_path, override=False)


def get_env(name: str, default: str = "") -> str:
    """读取环境变量。"""
    return os.environ.get(name, default)


def require_env(name: str) -> str:
    """读取必需的环境变量，缺失则抛错。"""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"必需的环境变量 {name} 未设置。请在 {env_file_help()} 中配置。"
        )
    return value


def env_file_help() -> str:
    """返回 .env 路径的提示字符串。"""
    try:
        from constants import env_file
        return str(env_file())
    except Exception:
        return "~/.OmniMate/.env"
