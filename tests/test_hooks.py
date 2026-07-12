"""Hooks 系统测试。"""
import pytest
from agent.hooks import (
    HookEvent, Hook, HookScriptConfig, HookRegistry,
)


# ---------------------------------------------------------------------------
# 类型 + 枚举
# ---------------------------------------------------------------------------

def test_hook_event_has_four_values():
    assert {e.value for e in HookEvent} == {
        "user_prompt_submit", "pre_tool_use", "post_tool_use", "stop",
    }


def test_hook_dataclass_programmatic():
    h = Hook(name="x", event=HookEvent.STOP, kind="programmatic", fn=lambda: None)
    assert h.kind == "programmatic"
    assert h.fail_closed is False


def test_hook_script_config_defaults():
    c = HookScriptConfig(command=["echo"])
    assert c.timeout == 10.0
    assert c.env is None


# ---------------------------------------------------------------------------
# USER_PROMPT_SUBMIT
# ---------------------------------------------------------------------------

def test_user_prompt_submit_no_hooks_passthrough():
    reg = HookRegistry()
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello"


def test_user_prompt_submit_single_hook_modifies():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: p.upper(), name="upper")
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "HELLO"


def test_user_prompt_submit_chain_composition():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: p + "1", name="a")
    reg.register_user_prompt_submit(lambda p: p + "2", name="b")
    assert reg.run_user_prompt_submit("x", session_id="s1") == "x12"


def test_user_prompt_submit_none_passthrough():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="noop")
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello"


def test_user_prompt_submit_exception_isolated():
    """hook 抛异常时视为 None，不影响链。"""
    reg = HookRegistry()
    def bad(p): raise ValueError("boom")
    reg.register_user_prompt_submit(bad, name="bad")
    reg.register_user_prompt_submit(lambda p: p + "_ok", name="ok")
    # bad 抛异常被吞，ok 仍然执行
    assert reg.run_user_prompt_submit("hello", session_id="s1") == "hello_ok"


# ---------------------------------------------------------------------------
# PRE_TOOL_USE
# ---------------------------------------------------------------------------

def test_pre_tool_use_no_hooks_returns_none_none():
    reg = HookRegistry()
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "ls"}, session_id="s")
    assert deny is None
    assert modified is None


def test_pre_tool_use_allow_when_none():
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: None, name="ok")
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "ls"}, session_id="s")
    assert deny is None
    assert modified is None


def test_pre_tool_use_deny():
    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "rm"}, session_id="s")
    assert deny == "blocked"
    assert modified is None


def test_pre_tool_use_deny_short_circuits():
    """首个 deny 胜出，后续不跑。"""
    calls = []
    def h1(n, a): calls.append("h1"); return {"deny": "first"}
    def h2(n, a): calls.append("h2"); return None
    reg = HookRegistry()
    reg.register_pre_tool_use(h1, name="h1")
    reg.register_pre_tool_use(h2, name="h2")
    deny, _ = reg.run_pre_tool_use("t", {}, session_id="s")
    assert deny == "first"
    assert calls == ["h1"]


def test_pre_tool_use_modify_args():
    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda n, a: {"modify_args": {"cmd": "safe"}}, name="m"
    )
    deny, modified = reg.run_pre_tool_use("terminal", {"cmd": "rm"}, session_id="s")
    assert deny is None
    assert modified == {"cmd": "safe"}


def test_pre_tool_use_modify_chain():
    """modify_args 链式累积，hook2 看到 hook1 的修改。"""
    def h1(n, a): return {"modify_args": {**a, "x": 1}}
    def h2(n, a): return {"modify_args": {**a, "y": 2}}
    reg = HookRegistry()
    reg.register_pre_tool_use(h1, name="h1")
    reg.register_pre_tool_use(h2, name="h2")
    deny, modified = reg.run_pre_tool_use("t", {"orig": 0}, session_id="s")
    assert deny is None
    assert modified == {"orig": 0, "x": 1, "y": 2}


def test_pre_tool_use_exception_isolated():
    def bad(n, a): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_pre_tool_use(bad, name="bad")
    deny, modified = reg.run_pre_tool_use("t", {"a": 1}, session_id="s")
    assert deny is None
    assert modified is None


# ---------------------------------------------------------------------------
# POST_TOOL_USE
# ---------------------------------------------------------------------------

def test_post_tool_use_no_hooks_passthrough():
    reg = HookRegistry()
    assert reg.run_post_tool_use("t", {}, "result", session_id="s") == "result"


def test_post_tool_use_single_modifies():
    reg = HookRegistry()
    reg.register_post_tool_use(lambda n, a, r: r.upper(), name="upper")
    assert reg.run_post_tool_use("t", {}, "hello", session_id="s") == "HELLO"


def test_post_tool_use_chain():
    reg = HookRegistry()
    reg.register_post_tool_use(lambda n, a, r: r + "1", name="a")
    reg.register_post_tool_use(lambda n, a, r: r + "2", name="b")
    assert reg.run_post_tool_use("t", {}, "x", session_id="s") == "x12"


def test_post_tool_use_exception_isolated():
    def bad(n, a, r): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_post_tool_use(bad, name="bad")
    assert reg.run_post_tool_use("t", {}, "hello", session_id="s") == "hello"


# ---------------------------------------------------------------------------
# STOP
# ---------------------------------------------------------------------------

def test_stop_no_hooks_returns_none():
    reg = HookRegistry()
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_all_none_returns_none():
    reg = HookRegistry()
    reg.register_stop(lambda: None, name="a")
    reg.register_stop(lambda: None, name="b")
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_first_non_none_wins():
    calls = []
    def h1(): calls.append("h1"); return None
    def h2(): calls.append("h2"); return "continue msg"
    def h3(): calls.append("h3"); return "should not run"
    reg = HookRegistry()
    reg.register_stop(h1, name="h1")
    reg.register_stop(h2, name="h2")
    reg.register_stop(h3, name="h3")
    msg = reg.run_stop(session_id="s", max_fires=3)
    assert msg == "continue msg"
    assert calls == ["h1", "h2"]


def test_stop_max_fires_returns_none_when_exceeded():
    """超过 max_fires 后强制返回 None。"""
    reg = HookRegistry()
    reg.register_stop(lambda: "loop", name="l")
    # 模拟已触发 max_fires 次
    reg._stop_fire_count = 3
    assert reg.run_stop(session_id="s", max_fires=3) is None


def test_stop_exception_isolated():
    def bad(): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_stop(bad, name="bad")
    reg.register_stop(lambda: "after", name="ok")
    msg = reg.run_stop(session_id="s", max_fires=3)
    assert msg == "after"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_all():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="x")
    reg.register_pre_tool_use(lambda n, a: None, name="y")
    reg.clear()
    assert all(len(reg._hooks[e]) == 0 for e in HookEvent)


def test_clear_single_event():
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: None, name="x")
    reg.register_pre_tool_use(lambda n, a: None, name="y")
    reg.clear(HookEvent.USER_PROMPT_SUBMIT)
    assert len(reg._hooks[HookEvent.USER_PROMPT_SUBMIT]) == 0
    assert len(reg._hooks[HookEvent.PRE_TOOL_USE]) == 1
