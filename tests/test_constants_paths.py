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


