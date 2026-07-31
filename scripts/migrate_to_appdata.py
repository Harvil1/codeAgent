"""[DEPRECATED] Windows 数据目录迁移脚本。

历史背景：曾经把 ~/.OmniMate/ 自动搬到 %APPDATA%\\HermesAgent\\（Windows 桌面应用规范）。
当前状态：已禁用。所有数据统一在 ~/.OmniMate/ 下（跨平台一致，避免路径漂移）。

保留此文件仅为向后兼容（main.py / 测试可能 import），所有函数返回 no-op。
未来若做成 Windows 安装包，可重新启用迁移逻辑（从 git 历史恢复）。
"""


def needs_migration(new_home=None) -> bool:
    """[DEPRECATED] 永远返回 False。迁移功能已禁用。"""
    return False


def migrate(new_home=None) -> bool:
    """[DEPRECATED] 永远返回 False（未执行迁移）。"""
    return False
