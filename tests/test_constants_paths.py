"""constants.py 和 migrate_to_appdata.py 的测试。

验证：
- AGENT_HOME 环境变量优先级最高
- Windows 上默认 %APPDATA%\\HermesAgent
- Linux/macOS 默认 ~/.agent
- 迁移脚本多重检查，安全跳过
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import constants
from constants import (
    get_agent_home,
    logs_dir,
    skills_dir,
    memory_file,
    sessions_db_path,
)


# ----------------------------------------------------------------------------
# get_agent_home：env var / 平台判断
# ----------------------------------------------------------------------------

def test_agent_home_env_override(monkeypatch, tmp_path):
    """AGENT_HOME 覆盖一切默认值。"""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    assert get_agent_home() == tmp_path


def test_agent_home_env_expanduser(monkeypatch):
    """AGENT_HOME 支持 ~ 展开。"""
    monkeypatch.setenv("AGENT_HOME", "~/foo")
    result = get_agent_home()
    assert "~" not in str(result)
    assert result.name == "foo"


def test_agent_home_win32_appdata(monkeypatch):
    """Windows 默认 %APPDATA%\\HermesAgent。"""
    monkeypatch.delenv("AGENT_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    fake_appdata = "C:/Users/test/AppData/Roaming"
    monkeypatch.setenv("APPDATA", fake_appdata)
    assert get_agent_home() == Path(fake_appdata) / "HermesAgent"


def test_agent_home_win32_appdata_missing_fallback(monkeypatch):
    """Windows 但 APPDATA 缺失时用 Path.home 兜底。"""
    monkeypatch.delenv("AGENT_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = "C:/Users/testuser"
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: Path(fake_home)))
    result = get_agent_home()
    assert "HermesAgent" in result.parts
    assert result.parent.name == "Roaming"


def test_agent_home_linux_default(monkeypatch):
    """Linux 默认 ~/.agent。"""
    monkeypatch.delenv("AGENT_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    fake_home = Path("/home/testuser")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert get_agent_home() == fake_home / ".agent"


def test_agent_home_darwin_default(monkeypatch):
    """macOS 默认 ~/.agent。"""
    monkeypatch.delenv("AGENT_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_home = Path("/Users/testuser")
    monkeypatch.setattr(constants.Path, "home", classmethod(lambda cls: fake_home))
    assert get_agent_home() == fake_home / ".agent"


# ----------------------------------------------------------------------------
# logs_dir：Windows 独立到 LOCALAPPDATA
# ----------------------------------------------------------------------------

def test_logs_dir_win32_localappdata(monkeypatch, tmp_path):
    """Windows 日志到 %LOCALAPPDATA%\\HermesAgent\\logs。"""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert logs_dir() == tmp_path / "HermesAgent" / "logs"


def test_logs_dir_linux_under_agent_home(monkeypatch, tmp_path):
    """Linux 日志在 agent home 下。"""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    assert logs_dir() == tmp_path / "logs"


# ----------------------------------------------------------------------------
# 子路径函数：基于 get_agent_home
# ----------------------------------------------------------------------------

def test_subpath_funcs(monkeypatch, tmp_path):
    """skills/memory/sessions 都基于 agent_home。"""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    assert skills_dir() == tmp_path / "skills"
    assert memory_file() == tmp_path / "MEMORY.md"
    assert sessions_db_path() == tmp_path / "sessions.db"


# ----------------------------------------------------------------------------
# 迁移脚本
# ----------------------------------------------------------------------------

def test_migration_skipped_on_non_windows(monkeypatch):
    """非 Windows 平台不迁移。"""
    monkeypatch.setattr(sys, "platform", "linux")
    from scripts.migrate_to_appdata import needs_migration
    assert needs_migration() is False


def test_migration_skipped_when_no_legacy(monkeypatch, tmp_path):
    """Windows 但无 ~/.agent 时不迁移。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    # 模拟 Path.home() 返回空目录（没有 .agent）
    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))
    assert mig.needs_migration(new_home=tmp_path / "newhome") is False


def test_migration_skipped_when_marker_exists(monkeypatch, tmp_path):
    """新目录有迁移标记时跳过。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "some_file.txt").write_text("data", encoding="utf-8")

    new_home = tmp_path / "newhome"
    new_home.mkdir()
    (new_home / ".migrated_to_appdata").write_text("done", encoding="utf-8")

    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))
    assert mig.needs_migration(new_home=new_home) is False


def test_migration_skipped_when_target_is_legacy(monkeypatch, tmp_path):
    """AGENT_HOME 指向 ~/.agent 时不自复制（开发场景）。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)

    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))
    # AGENT_HOME 指向 legacy 自己
    monkeypatch.setenv("AGENT_HOME", str(legacy))
    assert mig.needs_migration() is False


def test_migration_copies_files(monkeypatch, tmp_path):
    """正常迁移：复制老目录内容到新目录。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "MEMORY.md").write_text("index", encoding="utf-8")
    (legacy / "config.yaml").write_text("llm:", encoding="utf-8")
    sub = legacy / "skills"
    sub.mkdir()
    (sub / "skill1.md").write_text("# skill", encoding="utf-8")

    new_home = tmp_path / "newhome"

    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))
    assert mig.needs_migration(new_home=new_home) is True

    ok = mig.migrate(new_home=new_home)
    assert ok is True

    # 文件被复制
    assert (new_home / "MEMORY.md").read_text(encoding="utf-8") == "index"
    assert (new_home / "config.yaml").exists()
    assert (new_home / "skills" / "skill1.md").exists()
    # 老目录还在
    assert legacy.exists()
    # 迁移标记
    assert (new_home / ".migrated_to_appdata").exists()


def test_migration_does_not_overwrite(monkeypatch, tmp_path):
    """新目录已有同名文件时不覆盖。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "MEMORY.md").write_text("old", encoding="utf-8")

    new_home = tmp_path / "newhome"
    new_home.mkdir()
    (new_home / "MEMORY.md").write_text("existing", encoding="utf-8")

    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))
    mig.migrate(new_home=new_home)

    # 新目录里的原文件没被覆盖
    assert (new_home / "MEMORY.md").read_text(encoding="utf-8") == "existing"


def test_migration_idempotent(monkeypatch, tmp_path):
    """迁移后再次调用 needs_migration 返回 False。"""
    monkeypatch.setattr(sys, "platform", "win32")
    fake_home = tmp_path / "fakehome"
    legacy = fake_home / ".agent"
    legacy.mkdir(parents=True)
    (legacy / "x.txt").write_text("x", encoding="utf-8")

    new_home = tmp_path / "newhome"
    import scripts.migrate_to_appdata as mig
    monkeypatch.setattr(mig.Path, "home", classmethod(lambda cls: fake_home))

    assert mig.needs_migration(new_home=new_home) is True
    mig.migrate(new_home=new_home)
    assert mig.needs_migration(new_home=new_home) is False
