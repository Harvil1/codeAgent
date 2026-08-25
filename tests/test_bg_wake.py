"""idle wake（后台唤醒）机制测试。

背景：主对话空闲时，后台任务/异步子代理完成 → 自动激活主循环跑一轮
处理结果。链路分三段，各自可测：
1. 生产端回调：BackgroundManager 完成通知入队后敲回调；
   DelegationCompletionQueue.push 后敲回调。
2. 预检：AIAgent.has_pending_wake_payload（CLI 哨兵分支靠它防空唤醒）。
3. 感知注入：_assemble_turn_messages 注入 <background_tasks_running>。

CLI 主循环的哨兵分支与 EOF 哨兵同款（闭包局部变量），不直接单测
（与现有 _EOF_SENTINEL 的处理一致），由上面三段 + test_bg_tool 覆盖。
"""

import json
import sys
import threading
import time
from pathlib import Path

# 触发工具注册
import tools.bg_task  # noqa: F401
from tools.registry import registry


def _quick_cmd():
    """跨平台的快速命令（跑完就退出）。"""
    if sys.platform == "win32":
        return [sys.executable, "-c", "print('done')"]
    return ["sh", "-c", "echo done"]


def _long_cmd():
    """跑 5 秒的命令（测试期间保持 running 状态用）。"""
    return [sys.executable, "-c", "import time; time.sleep(5)"]


def _wait_until(pred, timeout=10.0):
    """轮询等条件成立（deadline 语义，超时返回 False）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _make_agent(tmp_path):
    """构造一个关掉无关 drain 源的 AIAgent（模式同 test_delegation_drain.py）。"""
    from agent import AIAgent
    from tools.delegate_tool import DelegationCompletionQueue

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=Path(tmp_path),
        enabled_toolsets=[],
    )
    agent.bg_manager = None
    agent.cron_scheduler = None
    agent.team_bus = None
    agent.team_name = None
    agent._delegation_queue = DelegationCompletionQueue()
    return agent


# ---------------------------------------------------------------------------
# 1. 生产端回调
# ---------------------------------------------------------------------------

def test_bg_manager_completion_fires_wake_callback(tmp_path):
    """后台任务完成 → 通知入队 → 唤醒回调被敲。"""
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    fired = threading.Event()
    mgr.set_wake_callback(fired.set)
    mgr.start(_quick_cmd(), cwd=Path(tmp_path))
    try:
        assert _wait_until(fired.is_set), "任务完成后应触发唤醒回调"
    finally:
        mgr.shutdown()


def test_bg_manager_no_callback_when_none_registered(tmp_path):
    """没注册回调时照常跑完（fail-open，不炸）。"""
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    assert mgr._wake_callback is None
    tid = mgr.start(_quick_cmd(), cwd=Path(tmp_path))
    try:
        assert _wait_until(
            lambda: mgr.status(tid).status in ("completed", "failed"))
        assert mgr.has_notifications()
    finally:
        mgr.shutdown()


def test_delegation_queue_push_fires_wake_callback():
    """异步子代理结果入队 → 唤醒回调被敲。"""
    from tools.delegate_tool import DelegationCompletionQueue

    q = DelegationCompletionQueue()
    fired = threading.Event()
    q.set_wake_callback(fired.set)
    q.push({"delegation_id": "del_x", "success": True, "result": "ok"})
    assert fired.is_set


def test_delegation_queue_callback_fail_open():
    """回调抛异常不拖垮 push（结果必须照常进队）。"""
    from tools.delegate_tool import DelegationCompletionQueue

    q = DelegationCompletionQueue()

    def _boom():
        raise RuntimeError("callback boom")

    q.set_wake_callback(_boom)
    q.push({"delegation_id": "del_y", "success": True, "result": "ok"})
    assert q.has_pending()


# ---------------------------------------------------------------------------
# 2. 预检 has_pending_wake_payload
# ---------------------------------------------------------------------------

def test_has_pending_wake_payload_delegation_path(tmp_path):
    """委托信箱有待取结果 → True；drain 掉 → False。"""
    agent = _make_agent(tmp_path)

    assert agent.has_pending_wake_payload() is False
    agent._delegation_queue.push({
        "delegation_id": "del_a", "success": True, "result": "r",
    })
    assert agent.has_pending_wake_payload() is True
    drained = agent._drain_injected_messages()
    assert drained["delegation_results"], "drain 应取到委托结果"
    assert agent.has_pending_wake_payload() is False


def test_has_pending_wake_payload_bg_path(tmp_path):
    """bg 通知队列有完成通知 → True；drain 掉 → False。"""
    from agent.background import BackgroundManager

    agent = _make_agent(tmp_path)
    mgr = BackgroundManager()
    agent.bg_manager = mgr
    tid = mgr.start(_quick_cmd(), cwd=Path(tmp_path))
    try:
        assert _wait_until(
            lambda: mgr.status(tid).status in ("completed", "failed"))
        assert mgr.has_notifications()
        assert agent.has_pending_wake_payload() is True
        mgr.drain_notifications()
        assert agent.has_pending_wake_payload() is False
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# 3. 感知注入 <background_tasks_running>
# ---------------------------------------------------------------------------

def test_assemble_no_note_when_idle(tmp_path):
    """没有在跑的后台任务时不注入（不浪费 token）。"""
    agent = _make_agent(tmp_path)
    msgs = agent._assemble_turn_messages("sys", {})
    assert not any(
        "<background_tasks_running>" in str(m.get("content", ""))
        for m in msgs
    )


def test_assemble_injects_note_when_bg_running(tmp_path):
    """有 running 任务时注入 ephemeral 的 <background_tasks_running>。"""
    from agent.background import BackgroundManager

    agent = _make_agent(tmp_path)
    mgr = BackgroundManager()
    agent.bg_manager = mgr
    mgr.start(_long_cmd(), cwd=Path(tmp_path))
    try:
        msgs = agent._assemble_turn_messages("sys", {})
        notes = [
            m for m in msgs
            if "<background_tasks_running>" in str(m.get("content", ""))
        ]
        assert notes, "有 running 任务时应注入 background_tasks_running"
        note = notes[0]
        assert note.get("_ephemeral") is True, "状态注入必须 ephemeral（不进历史）"
        assert "无需轮询" in note["content"]
    finally:
        mgr.shutdown()


def test_bg_running_note_lists_async_subagents(monkeypatch, tmp_path):
    """在跑的异步子代理（花名册里线程还活着）也进清单。"""
    import tools.delegate_tool as dt

    agent = _make_agent(tmp_path)
    th = threading.Thread(target=lambda: time.sleep(5), daemon=True)
    th.start()
    monkeypatch.setattr(dt, "_async_tasks", {
        "del_fake": {
            "thread": th,
            "cancel_event": threading.Event(),
            "goal": "调查模块结构",
            "started_at": time.time(),
        },
    })
    note = agent._build_bg_running_note()
    assert note, "有存活异步子代理时应产出清单"
    assert "del_fake" in note
    assert "async 子代理" in note


# ---------------------------------------------------------------------------
# 4. 工具返回的预期文案 + 配置默认值
# ---------------------------------------------------------------------------

async def test_bg_start_hint_mentions_wake(tmp_path):
    """bg_start 返回要说明'完成会通知/自动唤醒，无需轮询'。"""
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    result_str = await registry.dispatch(
        "bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert "无需轮询" in parsed.get("hint", "")
    mgr.shutdown()


def test_config_idle_wake_default_on():
    """config 默认开（用户要的行为），键真实存在。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["bg_task"]["idle_wake"] is True


def test_wake_message_registered_in_cli_source():
    """源码扫描：cli 定义了哨兵/唤醒消息并注册了回调（闭包不可直测，扫锚点）。"""
    src = (Path(__file__).resolve().parent.parent / "cli.py").read_text(
        encoding="utf-8")
    assert "_BG_WAKE_SENTINEL" in src
    assert "_BG_WAKE_MESSAGE" in src
    assert "has_pending_wake_payload" in src
    assert "set_wake_callback(_on_bg_wake)" in src


def test_cli_wake_branch_uses_identity_check():
    """哨兵分支只认对象身份：普通字符串（哪怕相似文本）不走唤醒路径。

    分支在闭包里无法注入，用源码断言代替，确保哨兵判断用的是
    `is _BG_WAKE_SENTINEL` 而不是字符串比较。
    """
    src = (Path(__file__).resolve().parent.parent / "cli.py").read_text(
        encoding="utf-8")
    assert "user_input is _BG_WAKE_SENTINEL" in src, \
        "哨兵必须用 is 判对象身份（与 EOF 哨兵同款），不能按文本比较"
