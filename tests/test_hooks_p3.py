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
    from agent.hook_exec import run_script_hook, HookBlockedError
    hook = Hook(
        name="h", event=HookEvent.PRE_TOOL_USE, kind="declarative",
        script=HookScriptConfig(
            handler_type="command",
            command=[sys.executable, "-c",
                     "import sys; sys.stderr.write('forbidden'); sys.exit(2)"],
        ),
    )
    # 默认（raise_on_block=False）返回特殊 block dict
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
