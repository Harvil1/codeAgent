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


# ---------------------------------------------------------------------------
# Task 5: terminal_tool 注入 sandbox wrapper（集成测试）
# ---------------------------------------------------------------------------

def test_terminal_with_sandbox_off_uses_shell_true(monkeypatch):
    """sandbox off → subprocess.run 用 shell=True（原路径）。"""
    import tools.terminal_tool as tt
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        captured["cmd"] = cmd
        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)
    result = tt._handle_terminal({"command": "echo hi"}, sandbox_mode="off")
    assert captured["shell"] is True
    import json
    parsed = json.loads(result)
    assert parsed["stdout"] == "ok"


def test_terminal_with_sandbox_on_uses_argv(monkeypatch):
    """sandbox on + 可用 → subprocess.run 用 argv + shell=False。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        captured["cmd"] = cmd
        captured["is_list"] = isinstance(cmd, list)
        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(sr, "is_available", lambda: True)
    monkeypatch.setattr(sr, "wrap_command",
                        lambda cmd, **kw: ["bwrap", "--", "bash", "-c", cmd])
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    result = tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    assert captured["shell"] is False
    assert captured["is_list"] is True
    assert captured["cmd"][0] == "bwrap"


def test_terminal_with_sandbox_on_unavailable_falls_back(monkeypatch):
    """sandbox on 但不可用 → fail-open 回退到 shell=True。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(sr, "is_available", lambda: False)
    monkeypatch.setattr(sr, "availability_reason", lambda: "未安装 bwrap")
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    assert captured["shell"] is True  # 回退到原路径


def test_terminal_gui_command_skips_sandbox(monkeypatch):
    """GUI 命令（start / Chrome 等）跳过 sandbox，走原路径。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        class R:
            stdout = ""
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(sr, "is_available", lambda: True)
    monkeypatch.setattr(sr, "wrap_command",
                        lambda cmd, **kw: ["bwrap", "bash", "-c", cmd])
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    tt._handle_terminal(
        {"command": "start notepad.exe"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    # GUI 路径用 shell=True
    assert captured["shell"] is True


# ---------------------------------------------------------------------------
# Final review fixes: C1 / I1 / I2
# ---------------------------------------------------------------------------

def test_terminal_reads_sandbox_mode_from_default_checker(monkeypatch):
    """C1 fix: 不传 sandbox_mode kwarg 时，从默认 PermissionChecker 读。

    生产链路 agent/__init__.py → handle_function_call → dispatch 不传 sandbox_mode,
    必须从 PermissionChecker.sandbox_mode 读才不致 feature 失效。
    """
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    from agent.permission import PermissionChecker, get_default_checker, set_default_checker

    # 保存原 checker
    original = get_default_checker()
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        captured["is_list"] = isinstance(cmd, list)

        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    try:
        # 临时换默认 checker，开 sandbox
        temp_checker = PermissionChecker()
        temp_checker.set_sandbox_mode("on")
        set_default_checker(temp_checker)

        monkeypatch.setattr(sr, "is_available", lambda: True)
        monkeypatch.setattr(sr, "wrap_command",
                            lambda cmd, **kw: ["bwrap", "--", "bash", "-c", cmd])
        monkeypatch.setattr(tt.subprocess, "run", fake_run)
        monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

        # 不传 sandbox_mode kwarg —— 模拟生产路径
        tt._handle_terminal(
            {"command": "echo hi"},
            omnimate_home="/tmp/fake_home",
        )
        assert captured["shell"] is False
        assert captured["is_list"] is True
    finally:
        set_default_checker(original)


def test_seatbelt_profile_escapes_paths(tmp_path, monkeypatch):
    """I1 fix: 含特殊字符的路径要被转义（防 profile 注入）。"""
    from agent.sandbox_runner import _write_seatbelt_profile, _seatbelt_escape_path
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)

    # 模拟恶意路径：尝试闭合 subpath 并注入额外 allow
    evil_path = '/tmp/evil")) (allow file-write* (subpath "/etc'
    profile = _write_seatbelt_profile(cwd=evil_path, writable_roots=[])
    content = profile.read_text(encoding="utf-8")

    # 转义验证：evil_path 里的 " 必须以 \" 形式出现在 profile 里（Scheme 转义）
    # 确保不会因原始 " 构成 Scheme 语法边界 → 注入额外规则
    # 关键标志：未转义的裸 " 后跟 ) 应不存在
    # （转义后会变成 \")，即不应出现 `(")` 这样闭合 subpath 后紧跟 `)` 的模式
    import re
    # 找所有 allow file-write* (subpath "...") 形式（正常应只有 1 条规则行）
    # 转义后 evil_path 整体作为一个字符串字面量，仍是一条 allow 规则
    rule_lines = [ln for ln in content.splitlines()
                  if "(allow file-write*" in ln and "subpath" in ln]
    assert len(rule_lines) == 1, (
        f"profile 注入泄漏（应只有 1 条 allow 规则行）: {content}"
    )

    # 验证转义函数本身
    assert _seatbelt_escape_path('a"b') == 'a\\"b'
    assert _seatbelt_escape_path('a\\b') == 'a\\\\b'


def test_bwrap_wrap_cwd_under_tmp_skips_tmpfs():
    """I2 fix: cwd 在 /tmp 下时跳过 --tmpfs /tmp（避免 bind/tmpfs 冲突）。"""
    from agent.sandbox_runner import _bwrap_wrap

    def _has_tmpfs(argv):
        return any(argv[i] == "--tmpfs" and argv[i + 1] == "/tmp"
                   for i in range(len(argv) - 1))

    # cwd 是 /tmp
    argv = _bwrap_wrap("x", cwd="/tmp", writable_roots=[])
    assert not _has_tmpfs(argv), f"cwd=/tmp 时不应加 --tmpfs /tmp: {argv}"

    # cwd 在 /tmp 下
    argv2 = _bwrap_wrap("x", cwd="/tmp/work", writable_roots=[])
    assert not _has_tmpfs(argv2), f"cwd=/tmp/work 时不应加 --tmpfs /tmp: {argv2}"

    # 正常 cwd（不在 /tmp 下）仍应加 --tmpfs /tmp
    argv3 = _bwrap_wrap("x", cwd="/home/user/proj", writable_roots=[])
    assert _has_tmpfs(argv3), f"正常 cwd 应加 --tmpfs /tmp: {argv3}"
