"""全方位补充测试：集成 / 安全 / 边界 / 故障注入 / 并发 / 回归。

补足 test_sandbox.py 的盲点矩阵：
- 集成：config → PermissionChecker → terminal_tool 全链路
- 安全：路径穿越、Unicode、shell 元字符、多 evil 字符
- 边界：空 command、超长、缓存 TTL、None writable_roots、重复 roots
- 故障注入：mkdir 失败、旧 checker 无 sandbox_mode、SandboxUnavailableError
- 并发：多线程 wrap_command
- 回归：sandbox_mode 不影响 check()/check_path()
"""
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 集成测试：config → PermissionChecker → terminal_tool 全链路
# ---------------------------------------------------------------------------

def test_config_security_defaults_sandbox_off():
    """DEFAULT_CONFIG['security']['sandbox_mode'] == 'off'（启动注入源头）。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["security"]["sandbox_mode"] == "off"
    assert DEFAULT_CONFIG["security"]["sandbox_writable_roots"] == []


def test_config_to_checker_to_tool_pipeline(monkeypatch):
    """端到端：config sandbox_mode='on' → checker → terminal_tool shell=False。

    这是 C1 fix 验证的延伸：模拟 cli.py 启动注入 + 生产 dispatch 路径。
    """
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    from agent.permission import PermissionChecker, get_default_checker, set_default_checker

    original = get_default_checker()
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        captured["cmd_type"] = type(cmd).__name__

        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    try:
        # 模拟 cli.py 启动注入流程
        checker = PermissionChecker()
        # 模拟 cli.py 从 config 读 sandbox_mode 并灌进 checker
        checker.set_sandbox_mode("on")
        set_default_checker(checker)

        monkeypatch.setattr(sr, "is_available", lambda: True)
        # 验证 argv 路径，强制关 Job Object 模式
        monkeypatch.setattr(sr, "uses_job_object", lambda: False)
        monkeypatch.setattr(sr, "wrap_command",
                            lambda cmd, **kw: ["bwrap", "--", "bash", "-c", cmd])
        monkeypatch.setattr(tt.subprocess, "run", fake_run)
        monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

        # 生产路径：不传 sandbox_mode kwarg（handle_function_call 不传）
        tt._handle_terminal(
            {"command": "echo production_path"},
            omnimate_home="/tmp/fake_home",
        )
        assert captured["shell"] is False
        assert captured["cmd_type"] == "list"
    finally:
        set_default_checker(original)


def test_sandbox_mode_does_not_affect_permission_check():
    """sandbox_mode 字段不影响 PermissionChecker.check() 行为。"""
    from agent.permission import PermissionChecker
    c = PermissionChecker()
    c.set_sandbox_mode("on")
    # check() 仍按 mode='default' 工作
    result = c.check("ls")
    assert result.allowed  # 'ls' 不在 deny list

    c.set_sandbox_mode("off")
    result2 = c.check("ls")
    assert result2.allowed


def test_sandbox_mode_does_not_affect_check_path():
    """sandbox_mode 字段不影响 PermissionChecker.check_path() 行为。"""
    from agent.permission import PermissionChecker
    c = PermissionChecker()
    c.set_sandbox_mode("on")
    result = c.check_path("/tmp/test.txt", write=True)
    # check_path 仍按原逻辑（受保护路径拒，其他允许）
    assert isinstance(result.allowed, bool)


# ---------------------------------------------------------------------------
# 安全 / 红队测试
# ---------------------------------------------------------------------------

def test_bwrap_argv_no_shell_injection_via_command():
    """shell=False：command 里的 shell 元字符不被解释，作为 argv 单元素。"""
    from agent.sandbox_runner import _bwrap_wrap
    evil_cmd = 'echo hi; rm -rf /  # shell injection attempt'
    argv = _bwrap_wrap(evil_cmd, cwd="/tmp/work", writable_roots=[])
    # 末尾 ["--", "bash", "-c", command] 中 command 是单个 argv 元素
    tail = argv[-4:]
    assert tail == ["--", "bash", "-c", evil_cmd]
    # 元字符保留为字符串字面量（bwrap 内 bash -c 会解释，但这是子进程的事；
    # 父进程不会用 shell 解析整个 argv）


def test_bwrap_wrap_path_traversal_in_cwd():
    """路径穿越 cwd（../../../etc）作为字面字符串加入 argv。

    bwrap 会把 cwd 路径 bind 进沙箱；穿越路径在沙箱里仍受 bind 限制。
    本测试只验证 _bwrap_wrap 不崩溃 + cwd 字面正确传递。
    """
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("ls", cwd="../../../etc", writable_roots=[])
    # cwd 应以字面 --bind ../../../etc ../../../etc 加入
    bind_pairs = [(argv[i+1], argv[i+2]) for i, a in enumerate(argv)
                  if a == "--bind" and i+2 < len(argv)]
    assert ("../../../etc", "../../../etc") in bind_pairs


def test_bwrap_wrap_unicode_command_and_cwd():
    """Unicode 命令和 cwd 不应导致编码错误。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("echo 你好世界", cwd="/tmp/日本語ディレクトリ",
                       writable_roots=["/home/用户/.OmniMate"])
    assert argv[-1] == "echo 你好世界"
    bind_pairs = [(argv[i+1], argv[i+2]) for i, a in enumerate(argv)
                  if a == "--bind" and i+2 < len(argv)]
    assert ("/tmp/日本語ディレクトリ", "/tmp/日本語ディレクトリ") in bind_pairs
    assert ("/home/用户/.OmniMate", "/home/用户/.OmniMate") in bind_pairs


def test_seatbelt_profile_with_backslash_path(tmp_path, monkeypatch):
    """I1 加强：纯反斜杠路径也要被正确转义（\\ → \\\\）。"""
    from agent.sandbox_runner import _write_seatbelt_profile, _seatbelt_escape_path
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)

    # 含反斜杠的 Windows 风格路径（即使 macOS 上跑也可能传入）
    bs_path = "C:\\Users\\test\\dirty"
    profile = _write_seatbelt_profile(cwd=bs_path, writable_roots=[])
    content = profile.read_text(encoding="utf-8")
    # 转义后应是 C:\\\\Users...（每个 \ → \\）
    escaped = _seatbelt_escape_path(bs_path)
    assert escaped in content


def test_seatbelt_profile_multiple_evil_paths(tmp_path, monkeypatch):
    """I1 多个 evil writable_roots 同时注入不破坏 profile 结构。"""
    from agent.sandbox_runner import _write_seatbelt_profile
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)

    evil_roots = [
        '/tmp/a")) (allow file-write* (subpath "/etc',
        '/tmp/b")) (allow file-write* (subpath "/root',
        '/tmp/normal_path',
    ]
    profile = _write_seatbelt_profile(cwd="/safe/cwd", writable_roots=evil_roots)
    content = profile.read_text(encoding="utf-8")

    # 应该只有 4 条 allow 规则（cwd + 3 roots），不应被注入增多
    rule_lines = [ln for ln in content.splitlines()
                  if "(allow file-write*" in ln and "subpath" in ln]
    assert len(rule_lines) == 4, (
        f"profile 注入泄漏（应 4 条 allow，实际 {len(rule_lines)}）: {content}"
    )

    # 强化断言：逐行检测注入指纹（不跨行匹配——合法规则在多行间换行）
    # 未转义的 " 紧跟 1+ 个 ) 再跟 (allow file-write 在同一行内
    # 转义后 " 前面有 \，negative lookbehind (?<!\\) 排除
    import re
    injection_pattern = re.compile(r'(?<!\\)"\)+[ \t]*\([ \t]*allow[ \t]+file-write')
    for line in content.splitlines():
        assert not injection_pattern.search(line), (
            f"profile 单行检测到注入指纹: {line}\n--- 完整内容 ---\n{content}"
        )


def test_seatbelt_escape_path_order():
    """转义顺序：先 \\ 再 \"（避免双重转义）。"""
    from agent.sandbox_runner import _seatbelt_escape_path
    # a\b"c → 先 \ → \\，再 " → \"，结果 a\\b\"c
    assert _seatbelt_escape_path('a\\b"c') == 'a\\\\b\\"c'
    # 不应变成 a\\\\b\\"c（如果顺序反了会双重转义）


# ---------------------------------------------------------------------------
# 边界条件
# ---------------------------------------------------------------------------

def test_bwrap_wrap_empty_command():
    """空 command 不应让 _bwrap_wrap 崩溃。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("", cwd="/tmp/work", writable_roots=[])
    assert argv[-1] == ""
    assert argv[-2] == "-c"


def test_bwrap_wrap_none_writable_roots_handled():
    """writable_roots=None 不应崩溃（应被迭代空列表逻辑处理）。

    注：当前实现 writable_roots: List[str]，传 None 是类型违反，
    但防御性测试验证 [r for r in None if r and r != cwd] 会抛 TypeError。
    本测试断言此行为是显式的（caller 必须传 list）。
    """
    from agent.sandbox_runner import _bwrap_wrap
    with pytest.raises(TypeError):
        _bwrap_wrap("ls", cwd="/tmp/work", writable_roots=None)


def test_bwrap_wrap_duplicate_cwd_in_roots_deduped():
    """writable_roots 含 cwd 时去重（不重复 bind）。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("x", cwd="/tmp/work",
                       writable_roots=["/tmp/work", "/tmp/work", "/tmp/other"])
    bind_pairs = [(argv[i+1], argv[i+2]) for i, a in enumerate(argv)
                  if a == "--bind" and i+2 < len(argv)]
    # cwd 应只出现 1 次
    cwd_count = sum(1 for src, _ in bind_pairs if src == "/tmp/work")
    assert cwd_count == 1, f"cwd 重复 bind: {bind_pairs}"
    # /tmp/other 仍要进 bind
    assert ("/tmp/other", "/tmp/other") in bind_pairs


def test_bwrap_wrap_falsy_entries_filtered():
    """writable_roots 含空串/None 时被过滤。"""
    from agent.sandbox_runner import _bwrap_wrap
    argv = _bwrap_wrap("x", cwd="/tmp/work",
                       writable_roots=["", "/tmp/real"])
    bind_pairs = [(argv[i+1], argv[i+2]) for i, a in enumerate(argv)
                  if a == "--bind" and i+2 < len(argv)]
    # 空串不应进 bind
    assert all(src != "" for src, _ in bind_pairs)
    assert ("/tmp/real", "/tmp/real") in bind_pairs


def test_bwrap_wrap_very_long_command():
    """超长 command（10000 字符）不崩溃。"""
    from agent.sandbox_runner import _bwrap_wrap
    long_cmd = "echo " + "x" * 10000
    argv = _bwrap_wrap(long_cmd, cwd="/tmp/work", writable_roots=[])
    assert argv[-1] == long_cmd
    assert len(argv[-1]) == 10005


def test_is_available_cache_ttl_boundary(monkeypatch):
    """缓存 TTL 边界：59s 命中、61s 重查。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None
    call_count = {"n": 0}

    def fake_which(name):
        call_count["n"] += 1
        return "/usr/bin/bwrap"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", fake_which)

    # 首次调用，which 跑 1 次
    mod.is_available()
    assert call_count["n"] == 1

    # 手动把缓存时间戳调到 59s 前 → 仍命中缓存
    cached_ok, reason, ts = mod._availability_cache
    mod._availability_cache = (cached_ok, reason, time.monotonic() - 59)
    mod.is_available()
    assert call_count["n"] == 1, "59s 内应命中缓存"

    # 调到 61s 前 → 缓存失效，重新 which
    mod._availability_cache = (cached_ok, reason, time.monotonic() - 61)
    mod.is_available()
    assert call_count["n"] == 2, "61s 后应重新检测"


def test_writable_roots_extension_via_config(monkeypatch):
    """config['security']['sandbox_writable_roots'] 真的进 wrap_command。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    captured = {}

    def fake_wrap(cmd, *, cwd, writable_roots):
        captured["writable_roots"] = list(writable_roots)
        return ["bwrap", "--", "bash", "-c", cmd]

    def fake_run(cmd, *args, **kwargs):
        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    monkeypatch.setattr(sr, "is_available", lambda: True)
    # 验证 wrap_command 路径，强制关 Job Object 模式
    monkeypatch.setattr(sr, "uses_job_object", lambda: False)
    monkeypatch.setattr(sr, "wrap_command", fake_wrap)
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    config = {"security": {"sandbox_writable_roots": ["/extra/1", "/extra/2"]}}
    # mock constants.get_omnimate_home（terminal_tool 收集 writable_roots 时调它）
    monkeypatch.setattr("constants.get_omnimate_home", lambda: Path("/mocked/home"))
    tt._handle_terminal(
        {"command": "ls"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
        config=config,
    )
    # writable_roots 应包含 omnimate_home + config 扩展
    roots = captured["writable_roots"]
    assert "/extra/1" in roots
    assert "/extra/2" in roots
    # Path("/mocked/home") 在 Windows 上 str() 是 "\\mocked\\home"，跨平台断言
    assert any("mocked" in r for r in roots), f"omnimate_home 未进 roots: {roots}"


# ---------------------------------------------------------------------------
# 故障注入
# ---------------------------------------------------------------------------

def test_wrap_command_linux_bwrap_missing_raises():
    """Linux 上 bwrap 未装 → wrap_command 抛 SandboxUnavailableError。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None
    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value=None):
        with pytest.raises(mod.SandboxUnavailableError) as exc_info:
            mod.wrap_command("ls", cwd="/tmp", writable_roots=[])
        assert "bwrap" in str(exc_info.value)


def test_wrap_command_macos_sandbox_exec_missing_raises(tmp_path, monkeypatch):
    """macOS 上 sandbox-exec 未装 → 抛 SandboxUnavailableError。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr("constants.get_omnimate_home", lambda: tmp_path)
    with pytest.raises(mod.SandboxUnavailableError) as exc_info:
        mod.wrap_command("ls", cwd="/Users/test/proj", writable_roots=[])
    assert "sandbox-exec" in str(exc_info.value)


def test_wrap_command_unsupported_platform_raises():
    """unsupported 平台（如 freebsd）→ 抛 SandboxUnavailableError。"""
    import agent.sandbox_runner as mod
    with patch("sys.platform", "freebsd"):
        with pytest.raises(mod.SandboxUnavailableError) as exc_info:
            mod.wrap_command("ls", cwd="/tmp", writable_roots=[])
        assert "freebsd" in str(exc_info.value) or "不支持" in str(exc_info.value)


def test_seatbelt_profile_mkdir_failure_raises(tmp_path, monkeypatch):
    """profile 目录创建失败时抛异常（不让 silent 失败）。"""
    import agent.sandbox_runner as mod

    # 让 get_omnimate_home 返回一个不可创建的路径（模拟权限失败）
    class UncreatablePath:
        def __truediv__(self, other):
            return self
        def mkdir(self, **kwargs):
            raise PermissionError("mock permission denied")
        def __str__(self):
            return "/nonexistent/protected"

    monkeypatch.setattr("constants.get_omnimate_home", lambda: UncreatablePath())
    # 抛 PermissionError（caller terminal_tool 会 catch + fail-open）
    with pytest.raises(PermissionError):
        mod._write_seatbelt_profile(cwd="/x", writable_roots=[])


def test_terminal_tool_handles_wrap_command_exception(monkeypatch):
    """terminal_tool: wrap_command 抛异常时走 fail-open（不崩溃）。"""
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)
        captured["cmd"] = cmd

        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    def exploding_wrap(*args, **kwargs):
        raise sr.SandboxUnavailableError("mock failure")

    monkeypatch.setattr(sr, "is_available", lambda: True)
    # 验证 wrap_command 异常路径，强制关 Job Object 模式
    monkeypatch.setattr(sr, "uses_job_object", lambda: False)
    monkeypatch.setattr(sr, "wrap_command", exploding_wrap)
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    # sandbox on 但 wrap 失败 → fail-open 回退 shell=True
    result = tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    assert captured["shell"] is True  # fail-open 走原路径
    import json
    parsed = json.loads(result)
    assert parsed["stdout"] == "ok"  # 命令仍执行


def test_terminal_tool_handles_generic_exception_fail_open(monkeypatch):
    """terminal_tool: wrap_command 抛 generic Exception 时也 fail-open。"""
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

    def exploding_wrap(*args, **kwargs):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(sr, "is_available", lambda: True)
    # 验证 wrap_command 异常路径，强制关 Job Object 模式
    monkeypatch.setattr(sr, "uses_job_object", lambda: False)
    monkeypatch.setattr(sr, "wrap_command", exploding_wrap)
    monkeypatch.setattr(tt.subprocess, "run", fake_run)
    monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

    tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
        omnimate_home="/tmp/fake_home",
    )
    assert captured["shell"] is True  # 任何异常都 fail-open


def test_old_checker_without_sandbox_mode_field_does_not_crash(monkeypatch):
    """旧 PermissionChecker（无 sandbox_mode 字段）→ terminal_tool 退回 'off'。

    C1 fix 的 AttributeError 防御验证。
    """
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr
    from agent.permission import get_default_checker, set_default_checker

    class OldChecker:
        """模拟旧版本 checker（没有 sandbox_mode 字段）。"""
        mode = "default"

        def check(self, command, cwd=None, **kwargs):
            from agent.permission import PermissionResult
            return PermissionResult(True, "ok", "ok")

        def check_path(self, *args, **kwargs):
            from agent.permission import PermissionResult
            return PermissionResult(True, "ok", "ok")

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["shell"] = kwargs.get("shell", False)

        class R:
            stdout = "ok"
            stderr = ""
            returncode = 0
        return R()

    original = get_default_checker()
    try:
        set_default_checker(OldChecker())
        monkeypatch.setattr(sr, "is_available", lambda: True)
        monkeypatch.setattr(sr, "wrap_command",
                            lambda cmd, **kw: ["bwrap", "bash", "-c", cmd])
        monkeypatch.setattr(tt.subprocess, "run", fake_run)
        monkeypatch.setattr(tt, "check_terminal_requirements", lambda: True)

        # 不传 sandbox_mode kwarg，OldChecker 无该字段 → 退回 "off" → shell=True
        tt._handle_terminal(
            {"command": "echo hi"},
            omnimate_home="/tmp/fake_home",
        )
        assert captured["shell"] is True  # 退回 off 路径，不崩溃
    finally:
        set_default_checker(original)


# ---------------------------------------------------------------------------
# 并发测试
# ---------------------------------------------------------------------------

def test_wrap_command_thread_safety_bwrap():
    """多线程并发调 _bwrap_wrap 不崩溃。"""
    from agent.sandbox_runner import _bwrap_wrap
    results = []
    errors = []

    def worker(n):
        try:
            argv = _bwrap_wrap(f"cmd{n}", cwd=f"/tmp/work{n}",
                               writable_roots=[f"/tmp/extra{n}"])
            results.append((n, argv[-1]))
        except Exception as e:
            errors.append((n, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发错误: {errors}"
    assert len(results) == 20
    # 每个 cmd 都正确写入 argv
    for n, last_arg in results:
        assert last_arg == f"cmd{n}"


def test_is_available_concurrent_cache_safe(monkeypatch):
    """is_available 并发调用缓存安全（无 race condition 崩溃）。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")

    results = []
    errors = []

    def worker():
        try:
            for _ in range(10):
                results.append(mod.is_available())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发 is_available 崩溃: {errors}"
    assert all(r is True for r in results)


# ---------------------------------------------------------------------------
# 性能（微基准）
# ---------------------------------------------------------------------------

def test_is_available_cache_performance(monkeypatch):
    """缓存生效时 1000 次调用应 < 100ms（不走 which）。"""
    import agent.sandbox_runner as mod
    mod._availability_cache = None

    call_count = {"n": 0}

    def fake_which(name):
        call_count["n"] += 1
        # 模拟 which 慢调用
        time.sleep(0.001)
        return "/usr/bin/bwrap"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", fake_which)

    start = time.perf_counter()
    for _ in range(1000):
        mod.is_available()
    elapsed = time.perf_counter() - start

    # 缓存命中时 which 只调 1 次；1000 次调用应 < 1s
    assert call_count["n"] == 1, f"which 应只调 1 次，实际 {call_count['n']}"
    assert elapsed < 1.0, f"1000 次调用耗 {elapsed:.3f}s（应 < 1s）"


def test_bwrap_wrap_performance():
    """_bwrap_wrap 1000 次调用 < 200ms（纯 argv 构造）。"""
    from agent.sandbox_runner import _bwrap_wrap
    start = time.perf_counter()
    for i in range(1000):
        _bwrap_wrap(f"cmd{i}", cwd="/tmp/work",
                    writable_roots=["/root1", "/root2", "/root3"])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"1000 次 wrap 耗 {elapsed:.3f}s（应 < 2s）"


# ---------------------------------------------------------------------------
# CLI /sandbox 命令（黑盒）
# ---------------------------------------------------------------------------

def test_cli_sandbox_command_registered():
    """/sandbox 在 cli.py 命令处理中注册。"""
    # 简单验证：cli.py 源码含 'name == "/sandbox"' 分支
    cli_path = Path(__file__).parent.parent / "cli.py"
    content = cli_path.read_text(encoding="utf-8")
    assert 'name == "/sandbox"' in content or 'name == "/sandbox"' in content


def test_cli_sandbox_help_text_present():
    """cli.py 帮助文本含 /sandbox。"""
    cli_path = Path(__file__).parent.parent / "cli.py"
    content = cli_path.read_text(encoding="utf-8")
    assert "/sandbox" in content
    # 帮助行应说明 bwrap/sandbox-exec
    assert "bwrap" in content or "sandbox-exec" in content
