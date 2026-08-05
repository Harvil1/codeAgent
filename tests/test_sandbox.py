"""sandbox_runner 单元测试。"""
import shutil
import sys
from unittest.mock import patch

import pytest


def test_sandbox_unavailable_error_is_runtime_error():
    """SandboxUnavailableError 必须是 RuntimeError 子类（调用方好捕获）。"""
    from agent.sandbox_runner import SandboxUnavailableError
    assert issubclass(SandboxUnavailableError, RuntimeError)


def test_is_available_linux_with_bwrap():
    """Linux 上 bwrap 存在 → True。"""
    from agent.sandbox_runner import is_available
    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value="/usr/bin/bwrap"):
        # 清缓存
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is True


def test_is_available_linux_without_bwrap():
    """Linux 上 bwrap 不存在 → False，reason 含'未安装'。"""
    from agent.sandbox_runner import is_available, availability_reason
    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value=None):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is False
        reason = availability_reason()
        assert "未安装" in reason or "bwrap" in reason


def test_is_available_macos_with_sandbox_exec():
    """macOS 上 sandbox-exec 存在 → True。"""
    from agent.sandbox_runner import is_available
    with patch("sys.platform", "darwin"), \
         patch("shutil.which", return_value="/usr/bin/sandbox-exec"):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is True


def test_is_available_windows_unsupported():
    """Windows 平台 → False，reason 明确说不支持。"""
    from agent.sandbox_runner import is_available, availability_reason
    with patch("sys.platform", "win32"):
        import agent.sandbox_runner as mod
        mod._availability_cache = None
        assert is_available() is False
        reason = availability_reason()
        assert "Windows" in reason or "不支持" in reason


def test_is_available_caches_result(monkeypatch):
    """is_available 第二次调用不重新 which（缓存生效）。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None
    call_count = {"n": 0}

    def fake_which(name):
        call_count["n"] += 1
        return "/usr/bin/bwrap"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", fake_which)
    mod.is_available()
    mod.is_available()
    mod.is_available()
    assert call_count["n"] == 1  # 只调用 1 次


def test_bwrap_wrap_basic_structure():
    """bwrap argv 必须含 --die-with-parent + bash -c + 命令。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("ls -la", cwd="/tmp/work", writable_roots=[])
    assert argv[0] == "bwrap"
    assert "--die-with-parent" in argv
    assert "--new-session" in argv
    # 末尾必须是 ["--", "bash", "-c", command]
    tail = argv[-4:]
    assert tail[0] == "--"
    assert tail[1] == "bash"
    assert tail[2] == "-c"
    assert tail[3] == "ls -la"


def test_bwrap_wrap_cwd_bind_exists():
    """cwd 必须以 --bind 形式加入（可写）。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("pwd", cwd="/home/user/proj", writable_roots=[])
    # 找到 --bind /home/user/proj /home/user/proj
    found = False
    for i, a in enumerate(argv):
        if a == "--bind" and i + 2 < len(argv):
            if argv[i + 1] == "/home/user/proj" and argv[i + 2] == "/home/user/proj":
                found = True
                break
    assert found, f"cwd 未以 --bind 加入: {argv}"


def test_bwrap_wrap_writable_roots_added():
    """每个 writable_root 都进 --bind。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("x", cwd="/tmp/a",
                       writable_roots=["/home/user/.OmniMate", "/tmp/logs"])
    bind_pairs = []
    for i, a in enumerate(argv):
        if a == "--bind" and i + 2 < len(argv):
            bind_pairs.append((argv[i + 1], argv[i + 2]))
    assert ("/home/user/.OmniMate", "/home/user/.OmniMate") in bind_pairs
    assert ("/tmp/logs", "/tmp/logs") in bind_pairs


def test_bwrap_wrap_includes_ro_system_dirs():
    """系统目录必须以 --ro-bind 形式加入（只读）。"""
    import agent.sandbox_runner as mod
    # 跨平台：强制 _BWRAP_RO_DIRS 里的目录"存在"，以验证过滤逻辑
    ro_dirs = set(mod._BWRAP_RO_DIRS)

    class FakePath:
        def __init__(self, p):
            self._p = str(p)

        def exists(self):
            return self._p in ro_dirs

    with patch.object(mod, "Path", FakePath):
        argv = mod._bwrap_wrap("x", cwd="/tmp/a", writable_roots=[])
    ro_targets = []
    for i, a in enumerate(argv):
        if a == "--ro-bind" and i + 2 < len(argv):
            ro_targets.append(argv[i + 1])
    # 至少 /usr 和 /etc（Linux 必需）
    assert "/usr" in ro_targets
    assert "/etc" in ro_targets


def test_bwrap_wrap_no_unshare_net():
    """网络不做隔离：argv 不含 --unshare-net。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("x", cwd="/tmp/a", writable_roots=[])
    assert "--unshare-net" not in argv


# ---------------------------------------------------------------------------
# Task 3: macOS Seatbelt
# ---------------------------------------------------------------------------

def test_seatbelt_profile_has_deny_default(tmp_path, monkeypatch):
    """生成的 .sb 文件必须含 (deny default)。"""
    from agent.sandbox_runner import _write_seatbelt_profile
    # 把 omnimate home 指到临时目录，避免污染
    monkeypatch.setattr(
        "constants.get_omnimate_home",
        lambda: tmp_path,
    )
    profile = _write_seatbelt_profile(
        cwd="/Users/test/proj",
        writable_roots=["/Users/test/.OmniMate"],
    )
    content = profile.read_text(encoding="utf-8")
    assert "(deny default)" in content or "(deny default)" in content.replace("\n", " ")
    assert "(version 1)" in content


def test_seatbelt_profile_allows_cwd_write(tmp_path, monkeypatch):
    """profile 必须允许 cwd 子路径写入。"""
    from agent.sandbox_runner import _write_seatbelt_profile
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)
    profile = _write_seatbelt_profile(
        cwd="/Users/test/proj",
        writable_roots=[],
    )
    content = profile.read_text(encoding="utf-8")
    assert "/Users/test/proj" in content
    assert "file-write" in content


def test_seatbelt_profile_allows_writable_roots(tmp_path, monkeypatch):
    """每个 writable_root 都进 allow file-write。"""
    from agent.sandbox_runner import _write_seatbelt_profile
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)
    profile = _write_seatbelt_profile(
        cwd="/Users/test/proj",
        writable_roots=["/Users/test/.OmniMate", "/tmp/logs"],
    )
    content = profile.read_text(encoding="utf-8")
    assert "/Users/test/.OmniMate" in content
    assert "/tmp/logs" in content


def test_seatbelt_wrap_returns_sandbox_exec_argv(tmp_path, monkeypatch):
    """wrap_command 在 macOS 返回 sandbox-exec argv。"""
    import agent.sandbox_runner as mod
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None)
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)
    mod._availability_cache = None
    argv = mod.wrap_command("ls", cwd="/Users/test/proj", writable_roots=[])
    assert argv[0] == "sandbox-exec"
    assert "-p" in argv
    # 末尾含 bash -c
    assert "bash" in argv
    assert "-c" in argv
    assert "ls" in argv


def test_seatbelt_profile_filename_unique(tmp_path, monkeypatch):
    """两次调用生成不同 .sb 文件名（uuid）。"""
    from agent.sandbox_runner import _write_seatbelt_profile
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)
    p1 = _write_seatbelt_profile(cwd="/x", writable_roots=[])
    p2 = _write_seatbelt_profile(cwd="/x", writable_roots=[])
    assert p1 != p2
