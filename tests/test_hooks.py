"""Hooks 系统测试。"""
from agent.hooks import (
    HookEvent, Hook, HookScriptConfig, HookRegistry,
)


# ---------------------------------------------------------------------------
# 类型 + 枚举
# ---------------------------------------------------------------------------

def test_hook_event_has_four_values():
    """枚举值集合（P2-13 后扩到 11 个）。"""
    assert {e.value for e in HookEvent} == {
        "user_prompt_submit", "pre_tool_use", "post_tool_use", "stop",
        "pre_llm_call", "post_llm_call",
        # P2-13 新增
        "session_start", "session_end",
        "pre_compact", "post_compact",
        "config_change",
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


def test_pre_tool_use_fail_closed_exception_denies():
    """fail_closed=True 时 hook 异常 → 视为拒绝。"""
    from agent.hooks import HookRegistry
    def bad(n, a): raise ValueError("boom")
    reg = HookRegistry()
    reg.register_pre_tool_use(bad, name="bad", fail_closed=True)
    deny, modified = reg.run_pre_tool_use("t", {}, session_id="s")
    # fail_closed → 异常被吞但 deny 返回错误消息
    assert deny is not None
    assert "boom" in deny or "ValueError" in deny
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


# ---------------------------------------------------------------------------
# Declarative 集成（T3）
# ---------------------------------------------------------------------------

import sys
from agent.hook_exec import run_script_hook  # 验证 import 可达


def _make_declarative_hook(event, stdout_json_str, name="decl"):
    """构造一个声明式 hook：子进程 echo 一段 JSON。"""
    # 用 python -c "print('...')" 模拟
    cmd = [sys.executable, "-c", f"print('{stdout_json_str}')"]
    return Hook(
        name=name, event=event, kind="declarative",
        script=HookScriptConfig(command=cmd, timeout=5.0),
    )


def test_user_prompt_submit_declarative_modifies():
    """声明式 hook 通过 stdout {prompt: ...} 修改。"""
    hook = _make_declarative_hook(
        HookEvent.USER_PROMPT_SUBMIT, '{"prompt": "DECL"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_user_prompt_submit("hello", session_id="s") == "DECL"


def test_pre_tool_use_declarative_deny():
    hook = _make_declarative_hook(
        HookEvent.PRE_TOOL_USE, '{"action": "deny", "reason": "blocked"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    deny, _ = reg.run_pre_tool_use("t", {}, session_id="s")
    assert deny == "blocked"


def test_pre_tool_use_declarative_modify():
    hook = _make_declarative_hook(
        HookEvent.PRE_TOOL_USE, '{"action": "modify", "args": {"x": 1}}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    _, modified = reg.run_pre_tool_use("t", {}, session_id="s")
    assert modified == {"x": 1}


def test_post_tool_use_declarative_modifies():
    hook = _make_declarative_hook(
        HookEvent.POST_TOOL_USE, '{"result": "NEW"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_post_tool_use("t", {}, "old", session_id="s") == "NEW"


def test_stop_declarative_continue():
    hook = _make_declarative_hook(
        HookEvent.STOP, '{"continue": "go on"}',
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    assert reg.run_stop(session_id="s", max_fires=3) == "go on"


def test_programmatic_runs_before_declarative():
    """同 event 内程序式先于声明式执行。"""
    order = []
    # 程序式 hook 记录顺序
    reg = HookRegistry()
    reg.register_user_prompt_submit(
        lambda p: order.append("prog") or p + "_p", name="prog"
    )
    # 声明式 hook 也记录（通过修改 prompt 标识）
    hook = _make_declarative_hook(
        HookEvent.USER_PROMPT_SUBMIT, '{"prompt": "FROM_DECL"}',
    )
    reg.register_declarative(hook)
    result = reg.run_user_prompt_submit("start", session_id="s")
    # 程序式先跑（把 start → start_p），声明式后跑（覆盖为 FROM_DECL）
    assert order == ["prog"]
    assert result == "FROM_DECL"


def test_declarative_failure_isolated():
    """声明式 hook 子进程失败时视为 None。"""
    # 用不存在的可执行文件
    bad_hook = Hook(
        name="bad", event=HookEvent.USER_PROMPT_SUBMIT, kind="declarative",
        script=HookScriptConfig(command=["./nonexistent-xyz"], timeout=1.0),
    )
    reg = HookRegistry()
    reg.register_declarative(bad_hook)
    assert reg.run_user_prompt_submit("hello", session_id="s") == "hello"


# ============ P2-13: 新事件类型测试 ============

def test_hook_event_has_new_types():
    """HookEvent 枚举包含 P2-13 新增的 5 个事件类型。"""
    new_events = {
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
        HookEvent.PRE_COMPACT,
        HookEvent.POST_COMPACT,
        HookEvent.CONFIG_CHANGE,
    }
    assert len(new_events) == 5


def test_register_session_start_invokes_callback():
    """register_session_start 注册的回调会被 run_session_start 调用。"""
    reg = HookRegistry()
    calls = []
    reg.register_session_start(
        lambda payload: calls.append(payload), name="log_session"
    )
    reg.run_session_start({"session_id": "s1", "started_at": "2026-07-18"})
    assert len(calls) == 1
    assert calls[0]["session_id"] == "s1"


def test_session_start_fail_open():
    """SESSION_START hook 抛异常时不影响主流程（fail-open）。"""
    reg = HookRegistry()

    def boom(payload):
        raise ValueError("hook broke")
    reg.register_session_start(boom)

    # 不应抛
    reg.run_session_start({"session_id": "s1"})


def test_session_end_called_in_order():
    """多个 SESSION_END hook 按 register 顺序调。"""
    reg = HookRegistry()
    order = []
    reg.register_session_end(lambda p: order.append("first"), name="a")
    reg.register_session_end(lambda p: order.append("second"), name="b")
    reg.run_session_end({"session_id": "s1", "reason": "quit"})
    assert order == ["first", "second"]


def test_pre_compact_can_abort():
    """PRE_COMPACT hook 返回 {abort: True} 时，run_pre_compact 返回 abort=True。

    场景：用户配置"不允许 L4 LLM 压缩"的 hook。
    """
    reg = HookRegistry()
    reg.register_pre_compact(
        lambda payload: {"abort": True} if payload.get("layer") == "llm" else None,
        name="block_llm_compact",
    )
    result = reg.run_pre_compact({"layer": "llm", "messages_count": 200})
    assert result["abort"] is True


def test_pre_compact_no_abort_returns_none():
    """PRE_COMPACT 所有 hook 返回 None 时不 abort。"""
    reg = HookRegistry()
    reg.register_pre_compact(lambda p: None, name="observer")
    result = reg.run_pre_compact({"layer": "snip"})
    assert result["abort"] is False


def test_post_compact_receives_before_after():
    """POST_COMPACT hook 收到 before/after 消息数。"""
    reg = HookRegistry()
    captured = []
    reg.register_post_compact(
        lambda p: captured.append(p), name="metrics"
    )
    reg.run_post_compact({
        "messages_before": 200,
        "messages_after": 60,
        "layer": "llm",
    })
    assert captured[0]["messages_before"] == 200
    assert captured[0]["messages_after"] == 60


def test_config_change_receives_diff():
    """CONFIG_CHANGE hook 收到 old/new/changed_keys。"""
    reg = HookRegistry()
    captured = []
    reg.register_config_change(
        lambda p: captured.append(p), name="audit"
    )
    reg.run_config_change({
        "changed_keys": ["model.name", "model.fallback_model"],
        "old": {"model": {"name": "deepseek-chat"}},
        "new": {"model": {"name": "deepseek-reasoner"}},
    })
    assert "model.name" in captured[0]["changed_keys"]


def test_new_hooks_isolated_per_registry():
    """不同 HookRegistry 实例的 hook 不互相影响。"""
    reg1 = HookRegistry()
    reg2 = HookRegistry()
    reg1.register_session_start(lambda p: None, name="a")
    # reg2 不应有这个 hook
    assert len(reg2._hooks[HookEvent.SESSION_START]) == 0
