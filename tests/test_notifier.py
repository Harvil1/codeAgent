"""CCAR11 Task 6 测试：notifier + 触发点接线。

覆盖：
1. notifier 本身：Windows toast / fail-open / 非 Windows / 节流 / config 关
2. 触发点接线：权限审批（真实 PermissionChecker 路径）+
   bg 完成（真实 _drain_injected_messages 路径，见文件末尾 CCAR13 用例）
"""
from unittest.mock import patch, MagicMock

import pytest

from agent.notifier import notify, _reset_throttle_for_test


# ============================================================================
# Part 1：notifier 本身（5 个测试，对齐 brief Step 1）
# ============================================================================

def test_notify_windows_toast():
    """Windows 平台 → 调 PowerShell 返回 True。"""
    _reset_throttle_for_test()
    with patch("agent.notifier.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert notify("标题", "内容") is True
        # PowerShell 被调
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "powershell"


def test_notify_failopen():
    """subprocess 抛 OSError → 不抛、返回 False。"""
    _reset_throttle_for_test()
    with patch("agent.notifier.subprocess.run", side_effect=OSError("no ps")):
        assert notify("t", "m") is False  # 不抛


def test_notify_non_windows(monkeypatch):
    """非 Windows → no-op 返回 False。"""
    monkeypatch.setattr("agent.notifier.sys.platform", "linux")
    assert notify("t", "m") is False


def test_notify_throttle():
    """同标题 30s 内只发一次；不同标题互不影响。"""
    _reset_throttle_for_test()
    with patch("agent.notifier.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert notify("同一标题", "a") is True
        assert notify("同一标题", "b") is False  # 30s 节流内
        assert notify("另一标题", "c") is True   # 不同标题不受影响


def test_notify_disabled_config():
    """config notifications.enabled=False → 不调 PowerShell，返回 False。"""
    _reset_throttle_for_test()
    with patch("agent.notifier.subprocess.run") as mock_run:
        with patch("agent.notifier._notifications_enabled", return_value=False):
            assert notify("t", "m") is False
            mock_run.assert_not_called()


# ============================================================================
# Part 2：触发点接线
# ============================================================================

def test_trigger_permission_approval_notifies(monkeypatch):
    """审批 callback 被调前触发 notify。

    触发路径：agent/permission.py check() 闸门 2 destructive 分支。
    用真实 PermissionChecker + mock subprocess 跑 destructive 命令审批。
    """
    _reset_throttle_for_test()
    called = []

    def fake_notify(title, message):
        called.append((title, message))
        return True

    from agent.permission import PermissionChecker
    from agent import notifier as notifier_mod

    checker = PermissionChecker(approval_callback=lambda cmd: True)

    # 触发 destructive 命令审批（rm 系列命中 destructive 模式）
    with patch.object(notifier_mod, "notify", fake_notify):
        result = checker.check("rm -rf /tmp/test_ccar11_notify")

    assert called, "审批前应触发 notify"
    assert called[0][0] == "需要审批"


# ============================================================================
# CCAR13 Task 2 B6：bg title 带 task_id（不同任务不互吞节流）
# ============================================================================

def test_bg_title_with_task_id_no_throttle_conflict():
    """两个不同 task_id 的 bg 通知 30s 内都发出（title 带区分）。"""
    _reset_throttle_for_test()
    with patch("agent.notifier.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert notify("后台任务:bg_00123456", "完成") is True
        assert notify("后台任务:bg_00999888", "完成") is True  # 不同任务不互吞


def test_bg_drain_title_contains_task_id(tmp_path):
    """真实接线：_drain_injected_messages 的 bg 通知 title 带 task_id 前 8 位。

    task_id 不足 8 位时用全量（切片语义自然覆盖）。
    """
    _reset_throttle_for_test()
    called = []

    def fake_notify(title, message):
        called.append((title, message))
        return True

    from agent import AIAgent

    fake_bg = MagicMock()
    fake_bg.drain_notifications.return_value = [
        {"task_id": "bg_001234567890", "status": "completed", "exit_code": 0},
        {"task_id": "short1", "status": "failed", "exit_code": 1},  # <8 位全量
        {"task_id": "bg-running", "status": "running"},  # 不触发
    ]
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[], omnimate_home=tmp_path,
        bg_manager=fake_bg,
    )
    with patch("agent.notifier.notify", fake_notify):
        agent._drain_injected_messages()

    assert len(called) == 2
    # title = "后台任务:<task_id 前 8 位>"
    assert called[0][0] == "后台任务:bg_00123"
    assert "bg_001234567890" in called[0][1] and "completed" in called[0][1]
    # 不足 8 位用全量
    assert called[1][0] == "后台任务:short1"
    assert "failed" in called[1][1]
