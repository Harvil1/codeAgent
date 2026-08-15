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


def test_is_available_windows_with_job_object(monkeypatch):
    """CCAR12: Windows + win_job_object 可导入 → True（Job Object 模式）。"""
    import agent.sandbox_runner as mod
    monkeypatch.setattr(sys, "platform", "win32")
    mod._availability_cache = None
    assert mod.is_available() is True
    assert mod.availability_reason() == ""


def test_is_available_windows_import_fails_failopen(monkeypatch):
    """CCAR12: Windows + win_job_object 导入失败 → False（fail-open）。"""
    import agent.sandbox_runner as mod
    monkeypatch.setattr(sys, "platform", "win32")
    # sys.modules 塞 None 让 `from agent.win_job_object import ...` 抛 ImportError
    monkeypatch.setitem(sys.modules, "agent.win_job_object", None)
    mod._availability_cache = None
    try:
        assert mod.is_available() is False
        reason = mod.availability_reason()
        assert "win_job_object" in reason or "不可用" in reason
    finally:
        mod._availability_cache = None


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
    # CCAR12: 这些测试验证 bwrap/seatbelt argv 路径，强制关 Job Object 模式
    monkeypatch.setattr(sr, "uses_job_object", lambda: False)
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
        # CCAR12: 验证 argv 路径，强制关 Job Object 模式
        monkeypatch.setattr(sr, "uses_job_object", lambda: False)
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

    # 强化断言（针对 fix-review 指出的薄弱点）：
    # 直接检测注入的指纹 —— 未转义的 " 紧跟 1+ 个 ) 再跟 (allow file-write
    # （即 evil 路径成功闭合 subpath 并注入了新规则）。
    # 转义后 " 变成 \"，negative lookbehind (?<!\\) 排除转义版本。
    # 逐行检测（避免误匹配合法多行规则间的换行）。
    import re
    injection_pattern = re.compile(r'(?<!\\)"\)+[ \t]*\([ \t]*allow[ \t]+file-write')
    for line in content.splitlines():
        assert not injection_pattern.search(line), (
            f"profile 单行检测到注入指纹: {line}\n--- 完整内容 ---\n{content}"
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


# ---------------------------------------------------------------------------
# CCAR12 Task 2: Windows Job Object 模式
# ---------------------------------------------------------------------------

def test_uses_job_object_only_on_windows(monkeypatch):
    """uses_job_object() 仅 win32 返回 True。"""
    import agent.sandbox_runner as mod
    monkeypatch.setattr(sys, "platform", "win32")
    assert mod.uses_job_object() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert mod.uses_job_object() is False


def test_sandbox_description_windows_text(monkeypatch):
    """Windows 描述文案含 Job Object + 进程管控定位。"""
    import agent.sandbox_runner as mod
    monkeypatch.setattr(sys, "platform", "win32")
    desc = mod.sandbox_description()
    assert "Job Object" in desc
    assert "进程管控" in desc
    assert "safe_path" in desc


def test_attach_job_forwards_to_win_job_object(monkeypatch):
    """attach_job 转发 win_job_object.create_job_for_subprocess（fail-open）。"""
    import agent.sandbox_runner as mod
    sentinel = object()
    monkeypatch.setattr(
        "agent.win_job_object.create_job_for_subprocess", lambda p: sentinel
    )
    assert mod.attach_job("fake_popen") is sentinel

    # 转发抛异常 → fail-open 返回 None（不冒泡）
    def boom(p):
        raise RuntimeError("mock boom")
    monkeypatch.setattr("agent.win_job_object.create_job_for_subprocess", boom)
    assert mod.attach_job("fake_popen") is None


class _FakeJob:
    """哨兵 job：记录事件顺序。"""

    def __init__(self, events):
        self._events = events

    def close(self):
        self._events.append("close")


class _FakePopen:
    """假 Popen：配合 terminal_tool 的 Windows job 分支。"""

    def __init__(self, cmd, events, **kwargs):
        self.cmd = cmd
        self.pid = 12345
        self.returncode = 0
        self._events = events
        events.append("popen")

    def communicate(self, timeout=None):
        self._events.append("communicate")
        return ("hi", "")

    def kill(self):
        self._events.append("kill")


def _win_job_env(monkeypatch, tt, sr, job, events):
    """公共脚手架：mock Windows Job Object 沙箱环境。"""
    monkeypatch.setattr(sr, "is_available", lambda: True)
    monkeypatch.setattr(sr, "uses_job_object", lambda: True)
    monkeypatch.setattr(sr, "attach_job", lambda popen: job)
    # wrap_command 必须不被调用（Job Object 模式命令不包装）
    monkeypatch.setattr(
        sr, "wrap_command",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不应调用 wrap_command")),
    )
    monkeypatch.setattr(
        tt.subprocess, "Popen",
        lambda cmd, **kw: _FakePopen(cmd, events, **kw),
    )
    monkeypatch.setattr(
        tt.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Job 模式不走 subprocess.run")),
    )
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)


def test_terminal_sandbox_windows_uses_job(monkeypatch):
    """sandbox on + Windows：命令正常跑 + job attach 被调 + close 在 communicate 后。"""
    import json
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []
    job = _FakeJob(events)
    _win_job_env(monkeypatch, tt, sr, job, events)
    captured = {}
    monkeypatch.setattr(
        sr, "attach_job",
        lambda popen: (captured.setdefault("attached", []).append(popen), job)[1],
    )

    result = tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    parsed = json.loads(result)
    # 命令正常执行，输出正常
    assert parsed["stdout"] == "hi"
    assert parsed["exit_code"] == 0
    # attach 被调且拿到的是 Popen 实例
    assert len(captured["attached"]) == 1
    assert isinstance(captured["attached"][0], _FakePopen)
    # 顺序：popen → attach(在 Popen 之后，events 里无标记但 attached 非空) →
    # communicate → close（close 必须在 communicate 之后，句柄保活到进程结束）
    assert events == ["popen", "communicate", "close"]


def test_terminal_sandbox_windows_attach_fail_failopen(monkeypatch):
    """attach 失败返回 None → fail-open：命令照常执行，不阻断。"""
    import json
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []
    _win_job_env(monkeypatch, tt, sr, None, events)  # job=None 模拟 attach 失败

    result = tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    parsed = json.loads(result)
    assert parsed["stdout"] == "hi"  # 命令仍执行
    assert "error" not in parsed
    # 无 job → 无 close 事件
    assert events == ["popen", "communicate"]


def test_terminal_sandbox_windows_timeout_closes_job(monkeypatch):
    """超时 → communicate 抛 TimeoutExpired → finally 里 job.close 仍被调
    （KILL_ON_JOB_CLOSE 顺带清理整棵子进程树）。"""
    import json
    import subprocess as sp
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []

    class TimeoutPopen(_FakePopen):
        def communicate(self, timeout=None):
            self._events.append("communicate")
            raise sp.TimeoutExpired(cmd=self.cmd, timeout=timeout)

    job = _FakeJob(events)
    _win_job_env(monkeypatch, tt, sr, job, events)
    monkeypatch.setattr(
        tt.subprocess, "Popen", lambda cmd, **kw: TimeoutPopen(cmd, events, **kw)
    )

    result = tt._handle_terminal(
        {"command": "ping -n 100 localhost"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    parsed = json.loads(result)
    assert "超时" in parsed.get("error", "")
    # close 在 communicate 之后仍被调用（finally 保活语义）
    assert events == ["popen", "communicate", "close"]


def test_terminal_sandbox_windows_timeout_kills_process_without_job(monkeypatch):
    """超时 + attach 失败（job=None）→ proc.kill() 必须被调（CCAR13 A2）。

    job=None 时没有 KILL_ON_JOB_CLOSE 兜底，不杀就变孤儿进程继续跑；
    杀完还要 communicate 收尸（回收管道/句柄，对齐 subprocess.run 内部语义）。
    """
    import json
    import subprocess as sp
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []

    class TimeoutNoJobPopen(_FakePopen):
        def communicate(self, timeout=None):
            self._events.append("communicate")
            raise sp.TimeoutExpired(cmd=self.cmd, timeout=timeout)

    _win_job_env(monkeypatch, tt, sr, None, events)  # job=None 模拟 attach 失败
    monkeypatch.setattr(
        tt.subprocess, "Popen", lambda cmd, **kw: TimeoutNoJobPopen(cmd, events, **kw)
    )

    result = tt._handle_terminal(
        {"command": "ping -n 100 localhost", "timeout": 1},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    parsed = json.loads(result)
    assert "超时" in parsed.get("error", "")
    # kill 被调 + 收尸 communicate；无 job → 无 close 事件
    assert events == ["popen", "communicate", "kill", "communicate"]


def test_terminal_sandbox_windows_timeout_with_job_does_not_kill(monkeypatch):
    """超时 + job 非 None → 不调 proc.kill()（finally 的 job.close 带
    KILL_ON_JOB_CLOSE 清整棵树，重复杀是多余动作——CCAR13 A2 约束）。"""
    import json
    import subprocess as sp
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []

    class TimeoutJobPopen(_FakePopen):
        def communicate(self, timeout=None):
            self._events.append("communicate")
            raise sp.TimeoutExpired(cmd=self.cmd, timeout=timeout)

    job = _FakeJob(events)
    _win_job_env(monkeypatch, tt, sr, job, events)
    monkeypatch.setattr(
        tt.subprocess, "Popen", lambda cmd, **kw: TimeoutJobPopen(cmd, events, **kw)
    )

    result = tt._handle_terminal(
        {"command": "ping -n 100 localhost", "timeout": 1},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    parsed = json.loads(result)
    assert "超时" in parsed.get("error", "")
    # 有 job：close 清树，kill 不出现
    assert events == ["popen", "communicate", "close"]


def test_terminal_sandbox_off_does_not_attach_job(monkeypatch):
    """sandbox off → 不走 Job Object（attach 不被调，走 subprocess.run）。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    events = []
    job = _FakeJob(events)
    monkeypatch.setattr(sr, "attach_job", lambda p: (_ for _ in ()).throw(
        AssertionError("sandbox off 不应 attach job")))
    monkeypatch.setattr(
        sr, "uses_job_object", lambda: (_ for _ in ()).throw(
            AssertionError("sandbox off 不应查询 job 模式")))

    def fake_run(cmd, *args, **kwargs):
        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    result = tt._handle_terminal({"command": "echo hi"}, sandbox_mode="off")
    import json
    parsed = json.loads(result)
    assert parsed["stdout"] == "ok"
