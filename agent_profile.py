"""Profile 系统：支持多个隔离的 agent 实例。

每个 profile 有自己的：
  - config.yaml
  - .env
  - MEMORY.md / USER.md
  - sessions.db
  - skills/
  - logs/

通过 AGENT_HOME 环境变量实现隔离。

⚠️ apply_profile 必须在任何 import 之前调用！
   因为 get_agent_home() 在模块加载时可能被读取。
"""

import os
import shutil
from pathlib import Path
from typing import List


def get_profiles_root() -> Path:
    """所有 profile 的根目录。"""
    return Path.home() / ".agent" / "profiles"


def list_profiles() -> List[str]:
    """列出所有 profile 名。"""
    root = get_profiles_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def apply_profile(profile_name: str) -> None:
    """应用 profile：设置 AGENT_HOME 环境变量。

    ⚠️ 必须在任何 import 之前调用！
    """
    if profile_name == "default":
        # 默认不设置 AGENT_HOME（用 ~/.agent）
        os.environ.pop("AGENT_HOME", None)
        return

    profile_path = get_profiles_root() / profile_name
    if not profile_path.exists():
        raise ValueError(f"profile 不存在: {profile_name}")

    os.environ["AGENT_HOME"] = str(profile_path)


def create_profile(name: str, *, clone_from: str = None) -> Path:
    """创建新 profile。

    clone_from：从已有 profile 克隆配置（不影响源）。
                可以是 "default" 或 profile 名。
    """
    root = get_profiles_root()
    root.mkdir(parents=True, exist_ok=True)

    new_path = root / name
    if new_path.exists():
        raise ValueError(f"profile 已存在: {name}")

    new_path.mkdir(parents=True)

    # 创建必需的子目录
    (new_path / "skills").mkdir()
    (new_path / "logs").mkdir()
    (new_path / ".env").touch()

    if clone_from:
        # 从已有 profile 复制配置
        if clone_from == "default":
            source = Path.home() / ".agent"
        else:
            source = root / clone_from

        if (source / "config.yaml").exists():
            shutil.copy2(source / "config.yaml", new_path / "config.yaml")

    return new_path


def delete_profile(name: str) -> None:
    """删除 profile（不可恢复）。"""
    if name == "default":
        raise ValueError("不能删除 default profile")

    root = get_profiles_root()
    profile_path = root / name
    if not profile_path.exists():
        raise ValueError(f"profile 不存在: {name}")

    shutil.rmtree(profile_path)


def get_current_profile() -> str:
    """当前 profile 名（根据 AGENT_HOME 推断）。"""
    agent_home = os.environ.get("AGENT_HOME")
    if not agent_home:
        return "default"

    path = Path(agent_home)
    root = get_profiles_root()
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "custom"
