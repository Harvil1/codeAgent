"""Profile（画像）系统：让一个 agent 能开多个互相隔离的"分身账号"。

每个 profile（分身账号）都有自己独立的一套家当：
  - config.yaml（配置）
  - .env（密钥等环境变量）
  - MEMORY.md / USER.md（记忆）
  - sessions.db（会话数据库）
  - skills/（技能库）
  - logs/（日志）

隔离靠 CODEAGENT_HOME 环境变量实现——把"家目录"指到不同文件夹，
数据自然互不可见。本模块是最底层的地基，CLI 启动最先调它。

⚠️ apply_profile 必须在任何 import 之前调用！
   因为 agent 家目录的取值函数（get_codeagent_home）在别的模块
   加载时可能就已经被读了，晚了就换不回来了。
"""

import os
import shutil
from pathlib import Path
from typing import List


def get_profiles_root() -> Path:
    """所有分身账号的存放根目录。

    返回：~/.codeAgent/profiles（每个子文件夹就是一个账号）。
    """
    return Path.home() / ".codeAgent" / "profiles"


def list_profiles() -> List[str]:
    """列出现有的所有账号名。

    返回：账号名列表（按字母排序）；目录还不存在时返回空列表。
    """
    root = get_profiles_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def apply_profile(profile_name: str) -> None:
    """切换到指定账号：把家目录环境变量指过去（CODEAGENT_HOME 指到该账号
    的文件夹，之后所有读写都落在它自己的地盘里）。

    ⚠️ 必须在任何 import 之前调用（晚了家目录就被别人读走了）！

    参数：
        profile_name: 账号名。"default" 表示用默认家目录 ~/.codeAgent。

    返回：无。账号不存在时抛 ValueError。
    """
    if profile_name == "default":
        # 默认账号不需要设环境变量——不设就天然用 ~/.codeAgent；
        # 反而要清掉可能残留的旧值，防止串号
        os.environ.pop("CODEAGENT_HOME", None)
        return

    profile_path = get_profiles_root() / profile_name
    if not profile_path.exists():
        raise ValueError(f"profile 不存在: {profile_name}")

    os.environ["CODEAGENT_HOME"] = str(profile_path)


def create_profile(name: str, *, clone_from: str = None) -> Path:
    """新建一个分身账号（建好技能库、日志、.env 等骨架目录才能正常跑）。

    参数：
        name: 新账号名。
        clone_from: 可选。从哪个老账号抄一份配置过来（只抄 config.yaml，
            不影响老账号）；可以是 "default" 或其他账号名。

    返回：新账号的文件夹路径。账号已存在时抛 ValueError。
    """
    root = get_profiles_root()
    root.mkdir(parents=True, exist_ok=True)

    new_path = root / name
    if new_path.exists():
        raise ValueError(f"profile 已存在: {name}")

    new_path.mkdir(parents=True)

    # 搭好必需的骨架：技能目录、日志目录、空的 .env 文件
    (new_path / "skills").mkdir()
    (new_path / "logs").mkdir()
    (new_path / ".env").touch()

    if clone_from:
        # 从老账号把配置文件抄过来（只抄 config.yaml，有才抄）
        if clone_from == "default":
            source = Path.home() / ".codeAgent"
        else:
            source = root / clone_from

        if (source / "config.yaml").exists():
            shutil.copy2(source / "config.yaml", new_path / "config.yaml")

    return new_path


def delete_profile(name: str) -> None:
    """删掉一个分身账号（连文件夹带全部数据，救不回来）。

    参数：
        name: 要删的账号名。默认账号 "default" 不许删。

    返回：无。账号不存在时抛 ValueError。
    """
    if name == "default":
        raise ValueError("不能删除 default profile")

    root = get_profiles_root()
    profile_path = root / name
    if not profile_path.exists():
        raise ValueError(f"profile 不存在: {name}")

    shutil.rmtree(profile_path)


def get_current_profile() -> str:
    """看现在用的是哪个账号（从家目录环境变量反推）。

    返回：账号名。没设环境变量就是 "default"；设了但不在 profiles
    目录下面（用户自己指的别处）就返回 "custom"。
    """
    agent_home = os.environ.get("CODEAGENT_HOME")
    if not agent_home:
        return "default"

    path = Path(agent_home)
    root = get_profiles_root()
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        # 家目录不在 profiles 目录下 → 是用户自定义的路径
        return "custom"
