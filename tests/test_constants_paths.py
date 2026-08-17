"""constants.py 和 migrate_to_appdata.py 的测试。

验证：
- AGENT_HOME 环境变量优先级最高
- 跨平台默认 ~/.OmniMate（Windows/Linux/macOS 一致，不再有 AppData 分流）
- logs_dir 统一在 ~/.OmniMate/logs 下
- 迁移脚本已永久禁用（needs_migration / migrate 均 no-op）
"""
import sys
from pathlib import Path

import constants
from constants import (
    get_omnimate_home,
    logs_dir,
    skills_dir,
    memory_file,
    sessions_db_path,
)


# ----------------------------------------------------------------------------
# get_omnimate_home：env var 覆盖 + 跨平台默认
# ----------------------------------------------------------------------------

def test_agent_home_env_override(monkeypatch, tmp_path):
    """AGENT_HOME 覆盖一切默认值。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    assert get_omnimate_home() == tmp_path


def test_agent_home_env_expanduser(monkeypatch):
    """AGENT_HOME 支持 ~ 展开。"""
    monkeypatch.setenv("OMNIMATE_HOME", "~/foo")
    result = get_omnimate_home()
    assert "~" not in str(result)
    assert result.name == "foo"


def test_agent_home_win32_defaults_to_dot_agent(monkeypatch):
    """Windows 也默认 ~/.OmniMate（跨平台一致，不走 AppData）。"""
    monkeypatch.delenv("OMNIMATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    # 即使 APPDATA 设置了，也不应影响（旧逻辑会读 APPDATA，新逻辑忽略）
    monkeypatch.setenv("APPDATA", "C:/Users/test/AppData/Roaming")
    fake_home = Path("C:/Users/test")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert get_omnimate_home() == fake_home / ".OmniMate"


def test_agent_home_linux_default(monkeypatch):
    """Linux 默认 ~/.OmniMate。"""
    monkeypatch.delenv("OMNIMATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    fake_home = Path("/home/testuser")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert get_omnimate_home() == fake_home / ".OmniMate"


def test_agent_home_darwin_default(monkeypatch):
    """macOS 默认 ~/.OmniMate。"""
    monkeypatch.delenv("OMNIMATE_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_home = Path("/Users/testuser")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert get_omnimate_home() == fake_home / ".OmniMate"


# ----------------------------------------------------------------------------
# logs_dir：统一在 ~/.OmniMate/logs（不再有 LOCALAPPDATA 分流）
# ----------------------------------------------------------------------------

def test_logs_dir_win32_under_agent_home(monkeypatch, tmp_path):
    """Windows 日志也在 ~/.OmniMate/logs（不再走 LOCALAPPDATA）。"""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # 设置了也不该被用
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    assert logs_dir() == tmp_path / "logs"


def test_logs_dir_linux_under_agent_home(monkeypatch, tmp_path):
    """Linux 日志在 agent home 下。"""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    assert logs_dir() == tmp_path / "logs"


# ----------------------------------------------------------------------------
# 子路径函数：基于 get_omnimate_home
# ----------------------------------------------------------------------------

def test_subpath_funcs(monkeypatch, tmp_path):
    """skills/memory/sessions 都基于 agent_home。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
    assert skills_dir() == tmp_path / "skills"
    assert memory_file() == tmp_path / "MEMORY.md"
    assert sessions_db_path() == tmp_path / "sessions.db"


# ----------------------------------------------------------------------------
# 迁移脚本：已永久禁用（no-op）
# ----------------------------------------------------------------------------

def test_migration_never_needed(monkeypatch, tmp_path):
    """[DEPRECATED] needs_migration 永远返回 False，无论平台/目录状态。"""
    from scripts.migrate_to_appdata import needs_migration
    # Windows + 老目录存在 + 无标记 → 旧逻辑会迁移，新逻辑永远 False
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "x.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert needs_migration(new_home=tmp_path / "newhome") is False


def test_migration_is_noop(monkeypatch, tmp_path):
    """[DEPRECATED] migrate 永远返回 False（不执行任何复制）。"""
    from scripts.migrate_to_appdata import migrate
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "MEMORY.md").write_text("index", encoding="utf-8")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    new_home = tmp_path / "newhome"
    new_home.mkdir()

    result = migrate(new_home=new_home)
    assert result is False
    # 确认没复制任何东西过去
    assert not (new_home / "MEMORY.md").exists()
    assert not (new_home / ".migrated_to_appdata").exists()
