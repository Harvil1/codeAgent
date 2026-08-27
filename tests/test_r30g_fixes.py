# -*- coding: utf-8 -*-
"""权限与超时钳制回归测试。

  pre_tool_use hook 并行 + deny>ask>allow 聚合 + ask 档 fail-closed
  terminal 超时钳制 + 裸 sleep 拦截 + 超时引导
  复合命令段数上限（>50 升审批）
  cd+git 组合审批闸门（bare-repo RCE 防护）
  SessionEnd hook 超时钳制（timeout_cap 传导）
"""
import json
from types import SimpleNamespace


# ======================================================================
# ask 档 + 聚合
# ======================================================================

def test_pre_tool_ask_fails_closed_h5():
    from agent.hooks import HookEvent, HookRegistry

    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda name, args: {"ask": "需要人工确认"}, name="asker",
    )
    deny, _mod = reg.run_pre_tool_use("terminal", {"command": "ls"}, session_id="s")
    assert deny is not None
    assert "asker" in deny and "人工确认" in deny


def test_pre_tool_deny_beats_ask_h5():
    from agent.hooks import HookEvent, HookRegistry

    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda name, args: {"ask": "确认?"}, name="h_ask",
    )
    reg.register_pre_tool_use(
        lambda name, args: {"deny": "明确拒绝"}, name="h_deny",
    )
    deny, _mod = reg.run_pre_tool_use("terminal", {"command": "ls"}, session_id="s")
    # 聚合：deny > ask（deny 原因原样返回，旧格式兼容）
    assert deny is not None and "明确拒绝" in deny


def test_pre_tool_parallel_declarative_h5(monkeypatch):
    """多个声明式 hook 走线程池并行执行，结果聚合。"""
    from agent.hooks import Hook, HookEvent, HookRegistry, HookScriptConfig

    reg = HookRegistry()
    for name in ("d1", "d2"):
        reg._hooks[HookEvent.PRE_TOOL_USE].append(Hook(
            name=name, event=HookEvent.PRE_TOOL_USE, kind="declarative",
            script=HookScriptConfig(command=["whatever"], timeout=2.0),
        ))

    import agent.hook_exec as he
    calls = []

    def _fake_dispatch(hook, payload, **kw):
        calls.append(hook.name)
        return {"action": "allow"}
    monkeypatch.setattr(he, "dispatch_hook", _fake_dispatch)
    deny, _mod = reg.run_pre_tool_use("terminal", {}, session_id="s")
    assert deny is None
    assert sorted(calls) == ["d1", "d2"], "两个声明式 hook 都被执行"


# ======================================================================
# 权限闸门
# ======================================================================

def test_compound_segment_cap_m2(tmp_path):
    from agent.permission import PermissionChecker
    checker = PermissionChecker()
    wide = " && ".join(f"echo seg{i}" for i in range(51))  # 51 段
    result = checker.check(wide, cwd=str(tmp_path))
    assert result.allowed is False
    assert result.gate == "too_many_segments"
    # 50 段不触发（正常 readonly 通道放行）
    ok_cmd = " && ".join(f"echo seg{i}" for i in range(50))
    assert checker.check(ok_cmd, cwd=str(tmp_path)).allowed is True


def test_cd_git_combo_gate_m3(tmp_path):
    from agent.permission import PermissionChecker
    checker = PermissionChecker()
    blocked = checker.check("cd /tmp/evil-repo && git status", cwd=str(tmp_path))
    assert blocked.allowed is False
    assert blocked.gate == "cd_git"
    # xargs git 形态同拦
    blocked2 = checker.check("cd /tmp/x && xargs git status", cwd=str(tmp_path))
    assert blocked2.gate == "cd_git"
    # git 在 cd 之前（原目录跑）不构成攻击面，不触发本闸门
    ok = checker.check("git status && cd /tmp", cwd=str(tmp_path))
    assert ok.gate != "cd_git"


# ======================================================================
# terminal 超时钳制 + sleep 拦截
# ======================================================================

def test_terminal_timeout_clamped_h10(tmp_path, monkeypatch):
    from tools import terminal_tool as tt
    captured = {}

    def _fake_run(*a, **kw):
        captured["timeout"] = kw.get("timeout")
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(tt.subprocess, "run", _fake_run)
    tt._handle_terminal(
        {"command": "echo x", "cwd": str(tmp_path), "timeout": 100000},
        tool_call_id="c", omnimate_home=str(tmp_path), config={},
    )
    assert captured["timeout"] == 600, "LLM 传 1e9 也钳到 max_terminal_timeout"


def test_terminal_bare_sleep_blocked_h10(tmp_path):
    from tools.terminal_tool import _handle_terminal
    out = json.loads(_handle_terminal(
        {"command": "sleep 5", "cwd": str(tmp_path)}, config={},
    ))
    assert out.get("error_type") == "bare_sleep"
    assert "bg_task" in out["error"]
    # 短 sleep 不拦
    ok = json.loads(_handle_terminal(
        {"command": "sleep 1", "cwd": str(tmp_path)}, config={},
    ))
    assert ok.get("error_type") != "bare_sleep"


# ======================================================================
# SessionEnd 超时钳制传导
# ======================================================================

def test_run_script_hook_timeout_cap_m8(monkeypatch):
    import agent.hook_exec as he

    captured = {}

    def _fake_run(argv, **kw):
        captured["timeout"] = kw.get("timeout")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(he.subprocess, "run", _fake_run)
    hook = SimpleNamespace(
        name="slow", fail_closed=False, use_sandbox=False,
        script=SimpleNamespace(handler_type="command",
                               command=["x"], timeout=10.0, env=None),
    )
    he.run_script_hook(hook, {"event": "session_end"}, timeout_cap=1.5)
    assert captured["timeout"] == 1.5, "cap 更小则生效"

    he.run_script_hook(hook, {"event": "e"}, timeout_cap=None)
    assert captured["timeout"] == 10.0, "无 cap 用 hook 自配"
