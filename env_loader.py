"""加载 .env 文件里的环境变量（API key 之类）。

背景：API key 这类敏感值不想写进代码，就放在 .env 文本文件里
（一行一个 KEY=VALUE），这个模块负责在程序启动时把它们读进
环境变量。底层用 python-dotenv 库实现。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_path: Optional[Path] = None) -> None:
    """把 .env 文件里的 KEY=VALUE 读进环境变量。

    优先级：已经在环境里存在的变量 > .env 文件——也就是说
    .env 不会覆盖 shell 里已经 export 过的值（shell 里手动的
    设置优先级更高，方便临时调试换 key）。

    参数：
        env_path：可选，.env 文件路径；不传就用默认位置
            <agent home>/.env。文件不存在就什么都不做。

    返回：
        无（直接改进程的环境变量）。
    """
    if env_path is None:
        # 函数内才 import，避免和 constants 循环依赖
        from constants import env_file
        env_path = env_file()

    if env_path and Path(env_path).exists():
        load_dotenv(env_path, override=False)


def get_env(name: str, default: str = "") -> str:
    """读一个环境变量，没有就返回默认值。

    参数：
        name：环境变量名。
        default：可选，变量不存在时返回的值（默认空字符串）。

    返回：
        环境变量的值，或 default。
    """
    return os.environ.get(name, default)


def require_env(name: str) -> str:
    """读一个"必须有"的环境变量，缺失直接抛错。

    背景：有些变量（如 API key）缺了程序没法跑，与其让后面
    某处冒出难懂的报错，不如在这里就地报清楚怎么配。

    参数：
        name：环境变量名。

    返回：
        环境变量的值（非空才返回）。

    抛错：
        RuntimeError——变量没设置或为空时，提示去 .env 里配置。
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"必需的环境变量 {name} 未设置。请在 {env_file_help()} 中配置。"
        )
    return value


def env_file_help() -> str:
    """生成"你的 .env 在哪里"的提示文字（给报错信息用）。

    返回：
        .env 文件路径的字符串；连路径都拿不到时退回
        写死的 ~/.OmniMate/.env。
    """
    try:
        from constants import env_file
        return str(env_file())
    except Exception:
        return "~/.OmniMate/.env"
