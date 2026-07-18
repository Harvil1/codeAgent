"""Windows 数据目录迁移：把 ~/.agent/ 的老数据复制到 %APPDATA%\\HermesAgent\\。

仅 Windows 首次启动时检测；非 Windows 直接 no-op。
迁移策略：复制不删除（老目录留作备份），写迁移标记避免重复迁移。
"""
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LEGACY_MARKER = ".migrated_to_appdata"
LEGACY_DIR_NAME = ".agent"


def _legacy_dir() -> Path:
    """老数据目录（~/.agent）。"""
    return Path.home() / LEGACY_DIR_NAME


def needs_migration(new_home: Path | None = None) -> bool:
    """判断是否需要迁移。

    条件：
      1. 必须是 Windows 平台
      2. 老目录 ~/.agent 存在
      3. 新目录里没有迁移标记
      4. 新目录与老目录不是同一个（避免 AGENT_HOME 指向 ~/.agent 时自复制）
    """
    if sys.platform != "win32":
        return False
    legacy = _legacy_dir()
    if not legacy.exists():
        return False
    # AGENT_HOME 可能指向老目录（开发/测试场景），此时不该迁移
    from constants import get_agent_home
    target = new_home if new_home is not None else get_agent_home()
    try:
        if target.resolve() == legacy.resolve():
            return False
    except OSError:
        return False
    if (target / LEGACY_MARKER).exists():
        return False
    return True


def migrate(new_home: Path | None = None) -> bool:
    """执行迁移。返回 True 表示实际迁移了，False 表示跳过。

    失败不抛异常（记日志），调用方可继续启动。
    """
    if not needs_migration(new_home):
        return False

    legacy = _legacy_dir()
    from constants import get_agent_home
    target = new_home if new_home is not None else get_agent_home()

    try:
        target.mkdir(parents=True, exist_ok=True)

        # 复制（不删原目录）；同名跳过
        for item in legacy.iterdir():
            target_item = target / item.name
            if target_item.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target_item)
            else:
                shutil.copy2(item, target_item)

        # 写迁移标记
        (target / LEGACY_MARKER).write_text(
            f"已从 {legacy} 迁移到 {target}\n",
            encoding="utf-8",
        )
        logger.info("数据已从 %s 迁移到 %s", legacy, target)
        print(f"✅ 数据已迁移到 {target}")
        print(f"   老目录 {legacy} 保留作为备份，确认无误后可手动删除。")
        return True
    except Exception as e:
        logger.warning("数据迁移失败（不阻塞启动）: %s", e)
        return False
