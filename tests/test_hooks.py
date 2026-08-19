"""Hooks 系统测试。"""
from agent.hooks import (
    HookEvent, Hook, HookScriptConfig, HookRegistry,
)


# ---------------------------------------------------------------------------
# 类型 + 枚举
# ---------------------------------------------------------------------------

def test_hook_event_has_four_values():
    """枚举值集合（P2-13 后扩到 11 个，round3 再加 7 个，P3.3-P3.4 再加 3 个，Task N 再加 6 个，共 27 个）。"""
    assert {e.value for e in HookEvent} == {
        "user_prompt_submit", "pre_tool_use", "post_tool_use", "stop",
        "pre_llm_call", "post_llm_call",
        # P2-13 新增
        "session_start", "session_end",
        "pre_compact", "post_compact",
        "config_change",
        # round3 新增
        "post_tool_use_failure", "subagent_start", "subagent_stop",
        "task_created", "task_completed",
        "permission_request", "permission_denied",
        # P3.3-P3.4 新增
        "stop_failure", "worktree_create", "worktree_remove",
        # Task N 新增
        "file_changed", "cwd_changed", "instructions_loaded",
        "setup", "teammate_idle", "elicitation_started",
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
    """首个 deny 胜出。

    R30g-H5 语义更新：所有匹配 hook 都会执行（并行聚合，对齐 CCB——
    一个 hook 的判决不应掩盖其他 hook 的判决/改参），deny 聚合后返回
    首个（注册序）deny 的原因。旧"短路不跑后续"已被聚合取代。
    """
    calls = []
    def h1(n, a): calls.append("h1"); return {"deny": "first"}
    def h2(n, a): calls.append("h2"); return None
    reg = HookRegistry()
    reg.register_pre_tool_use(h1, name="h1")
    reg.register_pre_tool_use(h2, name="h2")
    deny, _ = reg.run_pre_tool_use("t", {}, session_id="s")
    assert deny == "first"
    assert calls == ["h1", "h2"], "聚合语义：所有 hook 都执行"


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


# ============ F1: handler_type 多类型（http/mcp_tool/prompt/agent）============

def test_http_hook_posts_and_parses(monkeypatch):
    """http 类型 hook：POST JSON，解析响应。

    R16 #4 起 run_http_hook 有 SSRF 预检（真实 getaddrinfo）——测试用
    环回地址保持 hermetic（假域名会被本机 DNS 劫持成私网地址而误拦）。
    """
    import agent.hook_exec as he
    captured = {}
    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"permissionDecision": "allow"}
    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()
    monkeypatch.setattr(he.requests, "post", fake_post)
    from agent.hooks import HookScriptConfig, Hook, HookEvent
    cfg = HookScriptConfig(handler_type="http", url="http://127.0.0.1:9911/hook", timeout=5)
    hook = Hook(name="h", event=HookEvent.PRE_TOOL_USE, kind="declarative", script=cfg)
    out = he.dispatch_hook(hook, {"event": "pre_tool_use", "tool": "terminal"})
    assert captured["url"] == "http://127.0.0.1:9911/hook"
    assert out == {"permissionDecision": "allow"}


def test_command_hook_still_works():
    """command 类型向后兼容。"""
    import agent.hook_exec as he
    from agent.hooks import HookScriptConfig, Hook, HookEvent
    cfg = HookScriptConfig(handler_type="command", command=["python", "-c", "print('{\"x\":1}')"], timeout=5)
    hook = Hook(name="c", event=HookEvent.PRE_TOOL_USE, kind="declarative", script=cfg)
    out = he.dispatch_hook(hook, {"event": "pre_tool_use"})
    assert out == {"x": 1}


def test_unknown_handler_type_returns_none():
    """未知 handler_type 不崩，返回 None（fail-open）。"""
    import agent.hook_exec as he
    from agent.hooks import HookScriptConfig, Hook, HookEvent
    cfg = HookScriptConfig(handler_type="bogus", command=[])
    hook = Hook(name="u", event=HookEvent.PRE_TOOL_USE, kind="declarative", script=cfg)
    assert he.dispatch_hook(hook, {}) is None


# ============ F2: hook_loader._parse_hook 按 type 解析 5 种 handler ============

def test_parse_http_hook():
    from agent.hook_loader import _parse_hook
    from agent.hooks import HookEvent
    h = _parse_hook({"name": "h", "type": "http", "url": "https://x/y", "timeout": 3},
                    HookEvent.PRE_TOOL_USE)
    assert h is not None
    assert h.script.handler_type == "http"
    assert h.script.url == "https://x/y"
    assert h.script.timeout == 3


def test_parse_command_backward_compat():
    """老格式（无 type，只有 command）仍能解析。"""
    from agent.hook_loader import _parse_hook
    from agent.hooks import HookEvent
    h = _parse_hook({"name": "c", "command": ["echo", "hi"]}, HookEvent.STOP)
    assert h.script.handler_type == "command"
    assert h.script.command == ["echo", "hi"]


def test_parse_mcp_tool_requires_server_and_tool():
    from agent.hook_loader import _parse_hook
    from agent.hooks import HookEvent
    # 缺 tool → 跳过（返回 None）
    h = _parse_hook({"name": "m", "type": "mcp_tool", "server": "s"}, HookEvent.PRE_TOOL_USE)
    assert h is None
    # 完整
    h2 = _parse_hook({"name": "m", "type": "mcp_tool", "server": "s", "tool": "t"},
                     HookEvent.PRE_TOOL_USE)
    assert h2.script.mcp_server == "s"
    assert h2.script.mcp_tool == "t"


# ============ round3: 7 个关键事件注册与触发 ============

import pytest


@pytest.mark.parametrize("event_name,register_fn,run_fn,payload", [
    ("post_tool_use_failure", "register_post_tool_use_failure", "run_post_tool_use_failure", {"tool": "terminal", "error": "x", "error_type": "tool_exception"}),
    ("subagent_start", "register_subagent_start", "run_subagent_start", {"subagent": "x", "goal": "g"}),
    ("subagent_stop", "register_subagent_stop", "run_subagent_stop", {"subagent": "x", "success": True}),
    ("task_created", "register_task_created", "run_task_created", {"task_id": "t1", "subject": "s"}),
    ("task_completed", "register_task_completed", "run_task_completed", {"task_id": "t1", "unblocked": []}),
    ("permission_request", "register_permission_request", "run_permission_request", {"command": "rm x", "reason": "destructive"}),
    ("permission_denied", "register_permission_denied", "run_permission_denied", {"command": "rm -rf /", "reason": "fatal", "deny_type": "fatal"}),
])
def test_round3_hook_events_register_and_run(event_name, register_fn, run_fn, payload):
    """7 个新事件都能 register 程序式 hook 并被 run 触发。"""
    from agent.hooks import HookRegistry, HookEvent
    reg = HookRegistry()
    assert HookEvent(event_name)  # 枚举存在
    captured = []
    getattr(reg, register_fn)(lambda p: captured.append(p), name="t")
    getattr(reg, run_fn)(payload)
    assert len(captured) == 1
    assert captured[0] is payload


def test_round3_hook_events_fail_open():
    """hook 抛异常不影响主流程（fail-open）。"""
    from agent.hooks import HookRegistry
    reg = HookRegistry()

    def bad(payload):
        raise RuntimeError("boom")
    reg.register_post_tool_use_failure(bad, name="bad")
    # 不应抛
    reg.run_post_tool_use_failure({"tool": "x"})


# ---------------------------------------------------------------------------
# R30 审计 Medium-6：慢 hook 不得冻结事件循环
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_tool_use_slow_hook_does_not_block_event_loop():
    """async 调用点的 hook 链必须移出事件循环线程执行。

    旧实现：model_tools.handle_function_call（async）直调
    hooks_registry.run_pre_tool_use（sync，内部等待声明式子进程 hook/
    线程池结果）→ 事件循环线程被阻塞，流式输出与并发 safe 工具全部冻结，
    与 hooks.py docstring "不拖累主循环" 的宣称不符（只对 hook 之间成立）。
    修复：热点调用点（PRE/POST_TOOL_USE 等）经 asyncio.to_thread 执行。
    """
    import asyncio as _aio
    import time as _time
    import model_tools as mt
    from agent.hooks import HookRegistry

    reg = HookRegistry()

    def slow_hook(name, args):
        _time.sleep(0.4)  # 模拟慢 hook（子进程/慢回调同构）
        return None

    reg.register_pre_tool_use(slow_hook, name="slow")

    ticks = []

    async def ticker():
        while True:
            ticks.append(1)
            await _aio.sleep(0.02)

    tk = _aio.create_task(ticker())
    await mt.handle_function_call(
        "no_such_tool", "{}",
        hooks_registry=reg, session_id="s-m6",
    )
    tk.cancel()
    try:
        await tk
    except _aio.CancelledError:
        pass
    # 0.4s 慢 hook 期间 20ms 间隔的 ticker 应跳动 ~20 次；
    # 旧实现循环被阻塞 → ticks 停在 1-2
    assert len(ticks) >= 5, f"事件循环被慢 hook 阻塞（ticks={len(ticks)}）"


# ---------------------------------------------------------------------------
# R30 审计 L14：并行 hook 的 modify_args 有序合并
# ---------------------------------------------------------------------------

def test_pre_tool_use_parallel_modify_args_merged(monkeypatch):
    """多个 declarative hook 同时改参：按注册顺序叠加合并，不再后到整体替换。

    并行路径每个 hook 基于**原始参数**计算（programmatic 串行链式天然带前序
    修改，掩盖不了这个 bug）；旧聚合 modified_args = 后到者整体替换，先到
    hook 的修改静默丢失。新语义：首个 hook 的返回为基底，后续按键覆盖叠加
    （同键后到胜、异键并集），与 docstring "按注册顺序叠加" 一致。
    """
    from agent.hooks import HookRegistry, Hook, HookEvent, HookScriptConfig

    reg = HookRegistry()
    cfg = HookScriptConfig(handler_type="command", command=["echo"], timeout=1)
    reg._hooks[HookEvent.PRE_TOOL_USE].extend([
        Hook(name="h1", event=HookEvent.PRE_TOOL_USE, kind="declarative", script=cfg),
        Hook(name="h2", event=HookEvent.PRE_TOOL_USE, kind="declarative", script=cfg),
    ])

    mods = {"h1": {"b": 1}, "h2": {"c": 2}}

    def fake_invoke(self, hook, tool_name, args, session_id):
        # 并行语义：各自基于原始 args 计算
        return {"modify_args": {**args, **mods[hook.name]}}

    monkeypatch.setattr(HookRegistry, "_invoke_declarative_pre_tool", fake_invoke)

    deny, modified = reg.run_pre_tool_use("terminal", {"a": 0}, session_id="s")
    assert deny is None
    assert modified == {"a": 0, "b": 1, "c": 2}, \
        f"先到 hook 的修改被整体替换丢失: {modified!r}"


# ---------------------------------------------------------------------------
# C4（CCB 借鉴）：async hook + asyncRewake + statusMessage
# ---------------------------------------------------------------------------

def test_async_command_hook_nonblocking_with_rewake():
    """async hook 不阻塞（立即返回 None）；exit 2 + async_rewake → rewake 通知。

    rewake 通知带 hook 名 + status_message，由 agent 的 _drain_injected_messages
    消费为 ephemeral <task-notification>（模型下次轮看到可跟进）。
    """
    import sys as _sys
    import time as _time
    import agent.hook_exec as he
    from agent.hooks import HookScriptConfig, Hook, HookEvent

    he.drain_rewake_notifications()  # 清空

    cfg = HookScriptConfig(
        handler_type="command",
        command=[_sys.executable, "-c", "import sys; sys.exit(2)"],
        timeout=10, async_run=True, async_rewake=True,
        status_message="后台合规检查中",
    )
    hook = Hook(name="slow_async", event=HookEvent.POST_TOOL_USE,
                kind="declarative", script=cfg)

    t0 = _time.monotonic()
    out = he.dispatch_hook(hook, {"event": "post_tool_use"})
    elapsed = _time.monotonic() - t0

    assert out is None, "async hook 应立即返回 None（不阻塞）"
    assert elapsed < 0.5, f"async hook 不应等待子进程（耗时 {elapsed:.2f}s）"

    notes = _wait_rewake(he)
    assert notes, "exit 2 + async_rewake 应推 rewake 通知"
    note = notes[0]
    assert note["hook"] == "slow_async"
    assert note["status_message"] == "后台合规检查中"
    assert note.get("reason")


def test_async_hook_exit0_no_rewake():
    """async hook 正常退出（exit 0）不打扰模型。"""
    import sys as _sys
    import agent.hook_exec as he
    from agent.hooks import HookScriptConfig, Hook, HookEvent

    he.drain_rewake_notifications()
    cfg = HookScriptConfig(
        handler_type="command",
        command=[_sys.executable, "-c", "print('ok')"],
        timeout=10, async_run=True, async_rewake=True,
    )
    hook = Hook(name="quiet_async", event=HookEvent.POST_TOOL_USE,
                kind="declarative", script=cfg)
    assert he.dispatch_hook(hook, {"event": "post_tool_use"}) is None
    notes = _wait_rewake(he, expect=False)
    assert notes == [], "exit 0 不应推 rewake"


def _wait_rewake(he, expect=True, timeout=5.0):
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        notes = he.drain_rewake_notifications()
        if notes:
            return notes
        if not expect:
            _time.sleep(0.3)  # 给后台线程跑完的时间
            return he.drain_rewake_notifications()
        _time.sleep(0.05)
    return []


def test_drain_injected_messages_consumes_rewake(tmp_path):
    """agent 侧：rewake 通知进 injected dict，组装为 ephemeral task-notification。"""
    import json as _json
    import agent.hook_exec as he
    from agent import AIAgent

    agent = AIAgent(api_key="fake", model="test",
                    enabled_toolsets=[], omnimate_home=tmp_path)
    he.drain_rewake_notifications()
    he._push_rewake("hook_a", "发现问题 X", status_message="检查中")

    injected = agent._drain_injected_messages()
    assert injected.get("rewake_notifications"), "rewake 应被 drain 进 injected"
    assert injected["rewake_notifications"][0]["hook"] == "hook_a"

    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    hits = [m for m in msgs if "rewake_notification" in str(m.get("content", ""))]
    assert hits, "应注入 rewake 通知"
    assert hits[0].get("_ephemeral") is True
    assert "hook_a" in hits[0]["content"]
    assert injected["rewake_notifications"] == []  # 消费后清空
