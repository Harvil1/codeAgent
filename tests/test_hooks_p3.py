"""P3.2-P3.8 hooks 扩展测试。

覆盖：
- P3.2 dispatch_hook 接入 feature flag 门控
- P3.3 STOP_FAILURE 事件
- P3.4 WORKTREE_CREATE / WORKTREE_REMOVE 事件
- P3.5 if 条件过滤（permission rule 语法）
- P3.6 once 一次性 hook
- P3.7 exit code 2 = blocking 协议
- P3.8 hook 命令套 sandbox
"""
import sys
from unittest import mock

import pytest

from agent import hook_exec
from agent.hooks import (
    Hook, HookEvent, HookScriptConfig, HookRegistry,
)


# ============================================================================
# P3.2: dispatch_hook feature flag 门控
# ============================================================================

def _reset_hook_exec_providers():
    """每个测试前后重置 provider，避免互相污染。"""
    hook_exec._CONFIG_PROVIDER = None
    hook_exec._AUX_ROUTER_PROVIDER = None


@pytest.fixture(autouse=True)
def _clean_providers():
    _reset_hook_exec_providers()
    yield
    _reset_hook_exec_providers()


def test_p32_dispatch_command_always_allowed_without_config():
    """command 类型无 config 注入时永远允许（向后兼容）。"""
    called = []
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('{}')"],
        ),
    )
    with mock.patch.object(hook_exec, "run_script_hook",
                           side_effect=lambda h, p: called.append(h) or {}):
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result == {}
    assert len(called) == 1


def test_p32_dispatch_http_blocked_when_flag_off():
    """http handler 在 flag OFF 时被门控跳过。"""
    hook_exec.set_config_provider(lambda: {
        "features": {"hook_http_handler": {"enabled": False}},
    })
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(handler_type="http", url="http://example.com"),
    )
    called = []
    with mock.patch.object(hook_exec, "run_http_hook",
                           side_effect=lambda h, p: called.append(h) or {"ok": 1}):
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result is None
    assert len(called) == 0  # 被门控，未调


def test_p32_dispatch_http_allowed_when_flag_on():
    """http handler 在 flag ON 时正常跑。"""
    hook_exec.set_config_provider(lambda: {
        "features": {"hook_http_handler": {"enabled": True}},
    })
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(handler_type="http", url="http://example.com"),
    )
    called = []
    with mock.patch.object(hook_exec, "run_http_hook",
                           side_effect=lambda h, p: called.append(h) or {"ok": 1}):
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result == {"ok": 1}
    assert len(called) == 1


def test_p32_dispatch_mcp_tool_blocked_when_flag_off():
    """mcp_tool handler 在 flag OFF 时被门控。"""
    hook_exec.set_config_provider(lambda: {
        "features": {"hook_mcp_tool_handler": {"enabled": False}},
    })
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(
            handler_type="mcp_tool", mcp_server="s", mcp_tool="t",
        ),
    )
    with mock.patch.object(hook_exec, "run_mcp_tool_hook",
                           return_value={"ok": 1}) as m:
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result is None
    m.assert_not_called()


def test_p32_dispatch_agent_blocked_when_flag_off():
    """agent handler 在 flag OFF 时被门控。"""
    hook_exec.set_config_provider(lambda: {
        "features": {"hook_agent_handler": {"enabled": False}},
    })
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(handler_type="agent", prompt="判断"),
    )
    with mock.patch.object(hook_exec, "run_agent_hook",
                           return_value={"ok": 1}) as m:
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result is None
    m.assert_not_called()


def test_p32_dispatch_prompt_never_gated():
    """prompt handler 无 flag 门控（基线，永远允许）。"""
    hook_exec.set_config_provider(lambda: {
        "features": {},  # 没有任何 hook flag
    })
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(handler_type="prompt", prompt="判断"),
    )
    with mock.patch.object(hook_exec, "run_prompt_hook",
                           return_value={"ok": 1}) as m:
        result = hook_exec.dispatch_hook(hook, {"event": "stop"})
    assert result == {"ok": 1}
    m.assert_called_once()


# ============================================================================
# P3.3: STOP_FAILURE 事件
# ============================================================================

def test_p33_stop_failure_event_exists():
    """HookEvent 含 STOP_FAILURE。"""
    assert hasattr(HookEvent, "STOP_FAILURE")
    assert HookEvent.STOP_FAILURE.value == "stop_failure"


def test_p33_register_and_run_stop_failure():
    """register_stop_failure + run_stop_failure，通知型。"""
    reg = HookRegistry()
    seen = []

    def fn(payload):
        seen.append(payload)

    reg.register_stop_failure(fn, name="logger")
    reg.run_stop_failure({
        "session_id": "s1",
        "error": "LLM timeout",
        "error_type": "TimeoutError",
    })
    assert len(seen) == 1
    assert seen[0]["error"] == "LLM timeout"


def test_p33_stop_failure_fail_open():
    """hook 异常不抛（fail-open）。"""
    reg = HookRegistry()

    def bad_fn(payload):
        raise ValueError("boom")

    reg.register_stop_failure(bad_fn, name="bad")
    # 不抛
    reg.run_stop_failure({"session_id": "s1", "error": "x"})


# ============================================================================
# P3.4: WORKTREE_CREATE / WORKTREE_REMOVE 事件
# ============================================================================

def test_p34_worktree_events_exist():
    """HookEvent 含 WORKTREE_CREATE / WORKTREE_REMOVE。"""
    assert hasattr(HookEvent, "WORKTREE_CREATE")
    assert HookEvent.WORKTREE_CREATE.value == "worktree_create"
    assert hasattr(HookEvent, "WORKTREE_REMOVE")
    assert HookEvent.WORKTREE_REMOVE.value == "worktree_remove"


def test_p34_register_and_run_worktree_create():
    reg = HookRegistry()
    seen = []
    reg.register_worktree_create(lambda p: seen.append(p), name="audit")
    reg.run_worktree_create({
        "session_id": "s1",
        "path": "/tmp/wt-abc",
        "branch": "omnimate/x/abc",
    })
    assert len(seen) == 1
    assert seen[0]["path"] == "/tmp/wt-abc"


def test_p34_register_and_run_worktree_remove():
    reg = HookRegistry()
    seen = []
    reg.register_worktree_remove(lambda p: seen.append(p), name="audit")
    reg.run_worktree_remove({
        "session_id": "s1",
        "path": "/tmp/wt-abc",
    })
    assert len(seen) == 1


# ============================================================================
# P3.5: if 条件过滤（permission rule 语法）
# ============================================================================

def test_p55_if_match_tool_name_exact():
    from agent.hook_filter import match_if_condition
    assert match_if_condition("terminal", {"command": "ls"}, "terminal") is True
    assert match_if_condition("read_file", {"path": "a.txt"}, "terminal") is False


def test_p55_if_match_wildcard_pattern():
    from agent.hook_filter import match_if_condition
    assert match_if_condition(
        "terminal", {"command": "git status"}, "terminal(git *)"
    ) is True
    assert match_if_condition(
        "terminal", {"command": "ls"}, "terminal(git *)"
    ) is False


def test_p55_if_match_glob_arg():
    from agent.hook_filter import match_if_condition
    assert match_if_condition(
        "read_file", {"path": "src/main.py"}, "read_file(*.py)"
    ) is True
    assert match_if_condition(
        "read_file", {"path": "README.md"}, "read_file(*.py)"
    ) is False


def test_p55_if_empty_condition_always_matches():
    """无 if 条件 = 永远匹配（默认行为）。"""
    from agent.hook_filter import match_if_condition
    assert match_if_condition("terminal", {}, None) is True
    assert match_if_condition("terminal", {}, "") is True


def test_p55_if_invalid_condition_fails_open():
    """条件语法错（无括号/无 tool 名）→ fail-open 返回 True（不阻塞 hook）。"""
    from agent.hook_filter import match_if_condition
    # 无括号的纯字符串 = tool 名匹配
    assert match_if_condition("terminal", {}, "terminal") is True
    # 非法格式 → fail-open
    assert match_if_condition("terminal", {}, "(((") is True


def test_p55_registry_skips_non_matching_hook():
    """run_pre_tool_use 时，if 不匹配的声明式 hook 不被调用。"""
    reg = HookRegistry()
    called = []

    def make_hook(if_cond):
        h = Hook(
            name=f"h-{if_cond or 'none'}",
            event=HookEvent.PRE_TOOL_USE,
            kind="declarative",
            script=HookScriptConfig(
                handler_type="command",
                command=[sys.executable, "-c", "print('{}')"],
                if_condition=if_cond,
            ),
        )
        return h

    reg.register_declarative(make_hook("terminal(git *)"))
    reg.register_declarative(make_hook("read_file(*.py)"))
    # 程序式 hook 不受 if 条件过滤（只有声明式 hook 支持）
    reg.register_pre_tool_use(
        lambda name, args: called.append(("prog", name)) or None,
        name="prog",
    )

    with mock.patch.object(hook_exec, "run_script_hook",
                           side_effect=lambda h, p: called.append((h.name, p)) or None):
        # 触发 terminal 非 git 命令
        reg.run_pre_tool_use("terminal", {"command": "ls"}, session_id="s1")
    # 程序式 hook 永远调
    assert ("prog", "terminal") in called
    # 声明式 hook 的 if 不匹配 → 不调
    declared_names = [c[0] for c in called if c[0] != "prog"]
    assert "h-terminal(git *)" not in declared_names
    assert "h-read_file(*.py)" not in declared_names


def test_p55_registry_calls_matching_hook():
    """if 匹配时声明式 hook 被正常调用。"""
    reg = HookRegistry()
    called = []
    h = Hook(
        name="git-audit",
        event=HookEvent.PRE_TOOL_USE,
        kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('{}')"],
            if_condition="terminal(git *)",
        ),
    )
    reg.register_declarative(h)
    with mock.patch.object(hook_exec, "run_script_hook",
                           side_effect=lambda hh, p: called.append(hh.name) or None):
        reg.run_pre_tool_use("terminal", {"command": "git status"}, session_id="s1")
    assert "git-audit" in called


# ============================================================================
# P3.6: once 一次性 hook
# ============================================================================

def test_p36_once_declarative_hook_runs_only_once():
    """once=True 的声明式 hook 第一次跑后被消费，第二次跳过。"""
    reg = HookRegistry()
    called = []
    h = Hook(
        name="once-h",
        event=HookEvent.STOP,
        kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('{\"continue\": \"again\"}')"],
        ),
        once=True,
    )
    reg.register_declarative(h)
    with mock.patch.object(hook_exec, "run_script_hook",
                           side_effect=lambda hh, p: called.append(p) or {"continue": "again"}):
        # 第一次：跑
        r1 = reg.run_stop(session_id="s1", max_fires=10)
        assert r1 == "again"
        assert len(called) == 1
        # 第二次：被消费，跳过
        r2 = reg.run_stop(session_id="s1", max_fires=10)
        assert r2 is None
        assert len(called) == 1  # 还是 1


def test_p36_non_once_hook_runs_every_time():
    """once=False（默认）每次都跑。"""
    reg = HookRegistry()
    called = []
    h = Hook(
        name="every-h",
        event=HookEvent.STOP,
        kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('')"],
        ),
    )
    reg.register_declarative(h)
    with mock.patch.object(hook_exec, "run_script_hook",
                           side_effect=lambda hh, p: called.append(p) or {}):
        reg.run_stop(session_id="s1")
        reg.run_stop(session_id="s1")
        reg.run_stop(session_id="s1")
    assert len(called) == 3


def test_p36_once_consumed_per_hook():
    """不同 hook 的 once 独立消费。"""
    reg = HookRegistry()
    calls = {"a": 0, "b": 0}
    for n in ["a", "b"]:
        h = Hook(
            name=n, event=HookEvent.STOP, kind="declarative",
            script=HookScriptConfig(
                handler_type="command",
                command=[sys.executable, "-c", "print('')"],
            ),
            once=True,
        )
        reg.register_declarative(h)
    orig = hook_exec.run_script_hook

    def mock_run(hh, p):
        calls[hh.name] += 1
        return {"continue": "x"}

    with mock.patch.object(hook_exec, "run_script_hook", side_effect=mock_run):
        reg.run_stop(session_id="s1", max_fires=10)
        reg.run_stop(session_id="s1", max_fires=10)
    assert calls == {"a": 1, "b": 1}


# ============================================================================
# P3.7: exit code 2 = blocking 协议
# ============================================================================

def test_p37_exit_code_2_treated_as_block():
    """command hook exit 2 + stderr → 返回 {"action": "block", "reason": stderr}。"""
    from agent.hook_exec import run_script_hook
    hook = Hook(
        name="h", event=HookEvent.PRE_TOOL_USE, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c",
                     "import sys; sys.stderr.write('forbidden'); sys.exit(2)"],
        ),
    )
    # exit 2 → 返回特殊 block dict（不是 None，与 fail-open 区分）
    result = run_script_hook(hook, {"event": "pre_tool_use"})
    assert result is not None
    assert result.get("action") == "block"
    assert "forbidden" in result.get("reason", "")


def test_p37_exit_code_0_still_allowed():
    """exit 0 不受影响（向后兼容）。"""
    from agent.hook_exec import run_script_hook
    hook = Hook(
        name="h", event=HookEvent.PRE_TOOL_USE, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('{}')"],
        ),
    )
    result = run_script_hook(hook, {"event": "pre_tool_use"})
    assert result == {}


def test_p37_exit_code_1_still_fail_open():
    """exit 1（其他非 0/2）仍 fail-open 返回 None。"""
    from agent.hook_exec import run_script_hook
    hook = Hook(
        name="h", event=HookEvent.PRE_TOOL_USE, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "import sys; sys.exit(1)"],
        ),
    )
    result = run_script_hook(hook, {"event": "pre_tool_use"})
    assert result is None


def test_p37_pre_tool_use_block_propagates_to_deny():
    """声明式 hook 返回 block 时，run_pre_tool_use 把它转成 deny。"""
    reg = HookRegistry()
    h = Hook(
        name="blocker",
        event=HookEvent.PRE_TOOL_USE,
        kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c",
                     "import sys; sys.stderr.write('blocked-reason'); sys.exit(2)"],
        ),
    )
    reg.register_declarative(h)
    deny, mod = reg.run_pre_tool_use("terminal", {"command": "rm -rf /"},
                                     session_id="s1")
    assert deny is not None
    assert "blocked-reason" in deny


# ============================================================================
# P3.8: hook 命令套 sandbox（fail-open on Windows/不支持平台）
# ============================================================================

def test_p38_sandbox_unavailable_falls_back_graceously():
    """sandbox 不可用时（Windows）hook 正常跑（fail-open 降级）。"""
    from agent.hook_exec import run_script_hook
    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c", "print('{}')"],
        ),
        use_sandbox=True,  # 即使请求 sandbox
    )
    # 在 Windows 上 sandbox_runner.is_available() 返回 False，
    # hook 应正常跑（fail-open 降级到无沙箱）
    result = run_script_hook(hook, {"event": "stop"})
    assert result == {}


def test_p38_sandbox_not_enabled_by_default():
    """默认 use_sandbox=False（不启用）。"""
    h = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(command=["echo"]),
    )
    assert h.use_sandbox is False


# ============================================================================
# CCAR14 Task 2: Windows + use_sandbox=True 接 Job Object
# （对齐 terminal_tool 的 CCAR12 模式：命令不包装 + Popen + attach_job +
#   try communicate / finally close）
# ============================================================================

class _FakeJob:
    """哨兵 job：记录事件顺序（close 时机 = 句柄保活语义）。"""

    def __init__(self, events):
        self._events = events

    def close(self):
        self._events.append("close")


class _FakeHookPopen:
    """假 Popen：配合 hook_exec 的 Windows job 分支（照 CCAR12 Task 2 模式）。"""

    def __init__(self, argv, events, **kwargs):
        self.argv = argv
        self.pid = 12345
        self.returncode = 0
        self._events = events
        events.append("popen")

    def communicate(self, input=None, timeout=None):
        self._events.append("communicate")
        return ('{"decision": "allow"}', "")

    def kill(self):
        self._events.append("kill")


def _win_job_hook_env(monkeypatch, sr, job, events):
    """公共脚手架：mock Windows Job Object 沙箱环境（hook 版）。"""
    monkeypatch.setattr(sr, "is_available", lambda: True)
    monkeypatch.setattr(sr, "uses_job_object", lambda: True)
    monkeypatch.setattr(sr, "attach_job", lambda popen: job)
    # wrap_command 必须不被调用（Job Object 模式命令不包装）
    monkeypatch.setattr(
        sr, "wrap_command",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不应调用 wrap_command")),
    )
    monkeypatch.setattr(
        hook_exec.subprocess, "Popen",
        lambda argv, **kw: _FakeHookPopen(argv, events, **kw),
    )
    monkeypatch.setattr(
        hook_exec.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Job 模式不走 subprocess.run")),
    )


def _sandbox_hook():
    """use_sandbox=True 的 command hook。"""
    return Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=["python", "-c", "print('{}')"],
        ),
        use_sandbox=True,
    )


def test_ccar14_win_job_event_order(monkeypatch):
    """Windows + use_sandbox=True：Popen 正常启动（不包装）→ attach →
    communicate → close（close 必须在 communicate 之后，句柄保活到进程结束）。"""
    import agent.sandbox_runner as sr

    events = []
    job = _FakeJob(events)
    _win_job_hook_env(monkeypatch, sr, job, events)
    captured = {}
    monkeypatch.setattr(
        sr, "attach_job",
        lambda popen: (captured.setdefault("attached", []).append(popen), job)[1],
    )

    result = hook_exec.run_script_hook(_sandbox_hook(), {"event": "stop"})
    # 命令输出正常解析
    assert result == {"decision": "allow"}
    # attach 被调且拿到的是 Popen 实例
    assert len(captured["attached"]) == 1
    assert isinstance(captured["attached"][0], _FakeHookPopen)
    # 事件序：popen → communicate → close
    assert events == ["popen", "communicate", "close"]


def test_ccar14_win_job_command_not_wrapped(monkeypatch):
    """Job Object 模式下 Popen 收到的是原始 command（不包装、无前缀）。"""
    import agent.sandbox_runner as sr

    events = []
    _win_job_hook_env(monkeypatch, sr, _FakeJob(events), events)
    captured = {}
    monkeypatch.setattr(
        hook_exec.subprocess, "Popen",
        lambda argv, **kw: (
            captured.setdefault("argv", argv),
            _FakeHookPopen(argv, events, **kw),
        )[1],
    )

    hook = _sandbox_hook()
    result = hook_exec.run_script_hook(hook, {"event": "stop"})
    assert result == {"decision": "allow"}
    # Popen 收到的就是 hook.script.command 原始 list（未被沙箱包装）
    assert list(captured["argv"]) == list(hook.script.command)


def test_ccar14_win_job_attach_fail_failopen(monkeypatch):
    """attach 失败返回 None → fail-open：hook 照常执行，不阻断。"""
    import agent.sandbox_runner as sr

    events = []
    _win_job_hook_env(monkeypatch, sr, None, events)  # job=None 模拟 attach 失败

    result = hook_exec.run_script_hook(_sandbox_hook(), {"event": "stop"})
    assert result == {"decision": "allow"}  # 命令仍执行
    # 无 job → 无 close 事件
    assert events == ["popen", "communicate"]


def test_ccar14_win_job_timeout_closes_job(monkeypatch):
    """超时 → communicate 抛 TimeoutExpired → finally 里 job.close 仍被调
    （KILL_ON_JOB_CLOSE 顺带清理子进程树）；返回 None（fail-open）。"""
    import subprocess as sp
    import agent.sandbox_runner as sr

    events = []

    class TimeoutPopen(_FakeHookPopen):
        def communicate(self, input=None, timeout=None):
            self._events.append("communicate")
            raise sp.TimeoutExpired(cmd=self.argv, timeout=timeout)

    _win_job_hook_env(monkeypatch, sr, _FakeJob(events), events)
    monkeypatch.setattr(
        hook_exec.subprocess, "Popen",
        lambda argv, **kw: TimeoutPopen(argv, events, **kw),
    )

    result = hook_exec.run_script_hook(_sandbox_hook(), {"event": "stop"})
    assert result is None
    # close 在 communicate 之后仍被调用（finally 保活语义）
    assert events == ["popen", "communicate", "close"]


def test_ccar14_unix_wrapper_path_regression(monkeypatch):
    """Unix 回归：uses_job_object=False → 仍走 _wrap_with_sandbox 的
    wrap_command 包装 + subprocess.run（不挂 job、不走 Popen）。"""
    import agent.sandbox_runner as sr

    monkeypatch.setattr(sr, "is_available", lambda: True)
    monkeypatch.setattr(sr, "uses_job_object", lambda: False)
    wrapped_argv = ["bwrap", "--ro-bind", "/usr", "/usr", "--", "bash", "-c", "echo hi"]
    wrap_mock = mock.MagicMock(return_value=wrapped_argv)
    monkeypatch.setattr(sr, "wrap_command", wrap_mock)
    monkeypatch.setattr(
        sr, "attach_job",
        lambda p: (_ for _ in ()).throw(AssertionError("Unix 路径不应 attach job")),
    )
    monkeypatch.setattr(
        hook_exec.subprocess, "Popen",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Unix 路径不走 Popen")),
    )

    captured = {}

    class _FakeRunResult:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeRunResult()

    monkeypatch.setattr(hook_exec.subprocess, "run", fake_run)

    result = hook_exec.run_script_hook(_sandbox_hook(), {"event": "stop"})
    assert result == {}
    # wrap_command 被调用 + subprocess.run 收到包装后的 argv
    wrap_mock.assert_called_once()
    assert captured["argv"] == wrapped_argv


def test_ccar14_no_sandbox_does_not_attach_job(monkeypatch):
    """use_sandbox=False → 不挂 job（attach/uses_job_object 都不被查，
    走原 subprocess.run 路径）。"""
    import agent.sandbox_runner as sr

    monkeypatch.setattr(
        sr, "attach_job",
        lambda p: (_ for _ in ()).throw(AssertionError("use_sandbox=False 不应 attach job")))
    monkeypatch.setattr(
        sr, "uses_job_object",
        lambda: (_ for _ in ()).throw(
            AssertionError("use_sandbox=False 不应查询 job 模式")))

    class _FakeRunResult:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(
        hook_exec.subprocess, "run", lambda argv, **kw: _FakeRunResult())
    monkeypatch.setattr(
        hook_exec.subprocess, "Popen",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("无沙箱路径不走 Popen")))

    hook = Hook(
        name="h", event=HookEvent.STOP, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=["python", "-c", "print('{}')"],
        ),
        # use_sandbox 默认 False
    )
    result = hook_exec.run_script_hook(hook, {"event": "stop"})
    assert result == {}
