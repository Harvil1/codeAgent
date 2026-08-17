"""Task N 测试：HookEvent 新增 6 事件 + HookRegistry 6 方法 + fail-open。

6 个新事件：
  FILE_CHANGED / CWD_CHANGED / INSTRUCTIONS_LOADED
  / SETUP / TEAMMATE_IDLE / ELICITATION_STARTED
"""
import pytest


# ---------------------------------------------------------------------------
# 1. HookEvent 新增 6 枚举值
# ---------------------------------------------------------------------------

def test_hook_event_has_6_new_values():
    """HookEvent 包含 Task N 新增的 6 个枚举。"""
    from agent.hooks import HookEvent
    assert HookEvent.FILE_CHANGED.value == "file_changed"
    assert HookEvent.CWD_CHANGED.value == "cwd_changed"
    assert HookEvent.INSTRUCTIONS_LOADED.value == "instructions_loaded"
    assert HookEvent.SETUP.value == "setup"
    assert HookEvent.TEAMMATE_IDLE.value == "teammate_idle"
    assert HookEvent.ELICITATION_STARTED.value == "elicitation_started"


def test_hook_event_total_count():
    """21 + 6 = 27 个枚举值。"""
    from agent.hooks import HookEvent
    assert len(list(HookEvent)) == 27


# ---------------------------------------------------------------------------
# 2. HookRegistry 6 个新 register/run 方法（参数化）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("event_name,register_fn,run_fn,payload", [
    ("file_changed", "register_file_changed", "run_file_changed",
     {"path": "/tmp/x.py", "op": "write"}),
    ("cwd_changed", "register_cwd_changed", "run_cwd_changed",
     {"old": "/a", "new": "/b"}),
    ("instructions_loaded", "register_instructions_loaded", "run_instructions_loaded",
     {"source": "OMNIMATE.md", "bytes": 1024}),
    ("setup", "register_setup", "run_setup",
     {"agent_home": "/home/.OmniMate"}),
    ("teammate_idle", "register_teammate_idle", "run_teammate_idle",
     {"member": "explorer"}),
    ("elicitation_started", "register_elicitation_started", "run_elicitation_started",
     {"prompt": "请输入你的名字"}),
])
def test_new_hook_events_register_and_run(event_name, register_fn, run_fn, payload):
    """6 个新事件都能 register 程序式 hook 并被 run 触发。"""
    from agent.hooks import HookRegistry, HookEvent
    reg = HookRegistry()
    assert HookEvent(event_name)  # 枚举存在
    captured = []
    getattr(reg, register_fn)(lambda p: captured.append(p), name="t")
    getattr(reg, run_fn)(payload)
    assert len(captured) == 1
    assert captured[0] is payload


# ---------------------------------------------------------------------------
# 3. fail-open：hook 异常不崩主流程
# ---------------------------------------------------------------------------

def test_new_hook_events_fail_open():
    """hook 抛异常不影响主流程（fail-open）。"""
    from agent.hooks import HookRegistry
    reg = HookRegistry()

    def bad(payload):
        raise RuntimeError("boom")
    reg.register_file_changed(bad, name="bad")
    # 不应抛
    reg.run_file_changed({"path": "/tmp/x", "op": "write"})


def test_cwd_changed_fail_open():
    from agent.hooks import HookRegistry
    reg = HookRegistry()
    reg.register_cwd_changed(lambda p: (_ for _ in ()).throw(ValueError("x")))
    reg.run_cwd_changed({"old": "/a", "new": "/b"})  # 不抛


def test_setup_fail_open():
    from agent.hooks import HookRegistry
    reg = HookRegistry()
    def boom(p): raise OSError("disk full")
    reg.register_setup(boom)
    reg.run_setup({"agent_home": "/x"})  # 不抛


# ---------------------------------------------------------------------------
# 4. 无 hook 注册时也不崩
# ---------------------------------------------------------------------------

def test_new_events_no_hook_passthrough():
    """未注册任何 hook 时，run_xxx 调用不崩。"""
    from agent.hooks import HookRegistry
    reg = HookRegistry()
    reg.run_file_changed({"path": "/x"})
    reg.run_cwd_changed({"old": "/a", "new": "/b"})
    reg.run_instructions_loaded({"source": "CLAUDE.md"})
    reg.run_setup({})
    reg.run_teammate_idle({"member": "x"})
    reg.run_elicitation_started({"prompt": "?"})


# ---------------------------------------------------------------------------
# 5. 声明式 hook 支持（file_changed）
# ---------------------------------------------------------------------------

def test_file_changed_supports_declarative():
    """FILE_CHANGED 也支持声明式 hook（command 类型）。"""
    import sys
    from agent.hooks import (
        HookRegistry, Hook, HookScriptConfig, HookEvent,
    )
    cmd = [sys.executable, "-c", "print('{\"ack\": true}')"]
    hook = Hook(
        name="watcher",
        event=HookEvent.FILE_CHANGED,
        kind="declarative",
        script=HookScriptConfig(command=cmd, timeout=5.0),
    )
    reg = HookRegistry()
    reg.register_declarative(hook)
    # 不应抛
    reg.run_file_changed({"path": "/tmp/x.py", "op": "write"})
