"""后台任务管理器测试。"""
import sys
import time
from pathlib import Path

import pytest

from agent.background import BackgroundManager, BackgroundTask


def _quick_cmd():
    """跨平台快命令：sleep 0.3 然后 print done。"""
    if sys.platform == "win32":
        return [sys.executable, "-c", "import time; time.sleep(0.3); print('done')"]
    return ["sh", "-c", "sleep 0.3; echo done"]


def test_start_returns_task_id_and_creates_running_task(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    assert task_id.startswith("bg_")
    task = mgr.status(task_id)
    assert task is not None
    assert task.status == "running"
    # 清理
    mgr.shutdown()


def test_start_increments_pid(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    task = mgr.status(task_id)
    assert task.pid is not None and task.pid > 0
    mgr.shutdown()


def test_concurrent_limit_returns_error(tmp_path):
    mgr = BackgroundManager(max_concurrent=2)
    id1 = mgr.start(_quick_cmd(), cwd=tmp_path)
    id2 = mgr.start(_quick_cmd(), cwd=tmp_path)
    with pytest.raises(Exception) as exc_info:
        mgr.start(_quick_cmd(), cwd=tmp_path)
    assert "max concurrent" in str(exc_info.value).lower() or "full" in str(exc_info.value).lower()
    mgr.shutdown()


def test_completed_task_has_status_and_exit_code(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    # 等完成（最多 5 秒）
    for _ in range(50):
        task = mgr.status(task_id)
        if task.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert task.status == "completed"
    assert task.exit_code == 0
    assert "done" in task.stdout
    mgr.shutdown()


def test_drain_notifications_returns_and_clears(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    # 等完成
    for _ in range(50):
        if mgr.status(task_id) and mgr.status(task_id).status in ("completed", "failed"):
            break
        time.sleep(0.1)
    # 给 daemon thread 时间 push
    time.sleep(0.2)
    notifications = mgr.drain_notifications()
    assert len(notifications) >= 1
    assert notifications[0]["task_id"] == task_id
    # 再 drain 应该空
    assert mgr.drain_notifications() == []
    mgr.shutdown()


def test_drain_notifications_empty_returns_empty_list():
    mgr = BackgroundManager()
    assert mgr.drain_notifications() == []


def test_status_unknown_task_returns_none():
    mgr = BackgroundManager()
    assert mgr.status("bg_nonexistent") is None


def test_result_unknown_task_returns_none():
    mgr = BackgroundManager()
    assert mgr.result("bg_nonexistent") is None


def test_stop_terminates_running_task(tmp_path):
    """stop 一个 running 任务 → terminated。"""
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    task_id = mgr.start(long_cmd, cwd=tmp_path)
    time.sleep(0.2)  # 确保子进程已起
    stopped = mgr.stop(task_id)
    assert stopped is True
    # 等一下让 daemon thread 看到
    for _ in range(20):
        if mgr.status(task_id).status in ("stopped", "failed"):
            break
        time.sleep(0.1)
    assert mgr.status(task_id).status in ("stopped", "failed")


def test_stop_already_completed_is_idempotent(tmp_path):
    """stop 已完成的任务返回 True 但不改状态。"""
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    for _ in range(50):
        if mgr.status(task_id).status == "completed":
            break
        time.sleep(0.1)
    result = mgr.stop(task_id)
    assert result is True
    assert mgr.status(task_id).status == "completed"


def test_stop_unknown_returns_false():
    mgr = BackgroundManager()
    assert mgr.stop("bg_nonexistent") is False


def test_command_not_found_marks_task_failed(tmp_path):
    mgr = BackgroundManager()
    task_id = mgr.start(["./nonexistent-cmd-xyz"], cwd=tmp_path)
    for _ in range(30):
        task = mgr.status(task_id)
        if task.status in ("failed", "completed"):
            break
        time.sleep(0.1)
    assert task.status == "failed"


def test_list_tasks_returns_all(tmp_path):
    mgr = BackgroundManager()
    mgr.start(_quick_cmd(), cwd=tmp_path)
    mgr.start(_quick_cmd(), cwd=tmp_path)
    tasks = mgr.list_tasks()
    assert len(tasks) == 2
    mgr.shutdown()


def test_shutdown_terminates_non_detach_tasks(tmp_path):
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    mgr.start(long_cmd, cwd=tmp_path)
    mgr.shutdown()
    # shutdown 后所有任务应被 terminate（status != running）
    for task in mgr.list_tasks():
        assert task.status != "running"


def test_detach_task_survives_shutdown(tmp_path):
    """detach=True 的任务在 shutdown 后不被终止（PID 仍存活）。

    用 subprocess 检查 PID 是否还存在。简单起见，detach 任务用长 sleep。
    """
    import subprocess
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    task_id = mgr.start(long_cmd, cwd=tmp_path, detach=True)
    time.sleep(0.3)  # 确保子进程已起
    task = mgr.status(task_id)
    pid = task.pid
    mgr.shutdown()
    # 检查 PID 仍存活（粗糙：kill 0 不抛即存活，POSIX only）
    if sys.platform != "win32":
        try:
            import os
            os.kill(pid, 0)
            still_alive = True
            # 清理
            os.kill(pid, 9)
        except OSError:
            still_alive = False
        assert still_alive, "detach 任务应存活"


def test_stop_watch_race_no_duplicate_notifications(tmp_path):
    """I-3 修复验证：stop() 与 _watch() 竞态不应产生重复通知。

    stop() 先将状态改为 "stopped"，_watch() 的 communicate() 返回后
    应检测到 status != "running" 并直接 return，不覆盖状态、不 push 第二个通知。
    """
    long_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    mgr = BackgroundManager()
    task_id = mgr.start(long_cmd, cwd=tmp_path)
    # 确认 running 后 stop
    for _ in range(50):
        task = mgr.status(task_id)
        if task and task.status == "running":
            break
        time.sleep(0.1)
    mgr.stop(task_id)
    # 等 daemon thread 的 communicate 返回（terminate 后会返回）
    for _ in range(50):
        task = mgr.status(task_id)
        if task and task.status != "running":
            break
        time.sleep(0.1)
    # 状态应为 stopped（不被 _watch 覆盖为 failed）
    assert task.status == "stopped"
    # drain 通知：只应有 1 条（stop 的），不应有 _watch 的重复
    notifications = mgr.drain_notifications()
    # 可能有 1 条（stop push），不应有 2 条
    stop_notifications = [n for n in notifications if n["status"] == "stopped"]
    assert len(stop_notifications) <= 1, f"不应有重复 stopped 通知: {stop_notifications}"


def test_status_returns_copy_not_live_reference(tmp_path):
    """I-4 修复验证：status() 返回浅拷贝，修改不影响内部状态。"""
    mgr = BackgroundManager()
    task_id = mgr.start(_quick_cmd(), cwd=tmp_path)
    task_copy = mgr.status(task_id)
    assert task_copy is not None
    # 修改拷贝的属性
    task_copy.status = "tampered"
    # 内部状态不受影响
    internal_task = mgr.status(task_id)
    assert internal_task.status != "tampered"
    mgr.shutdown()


def test_list_tasks_returns_copies(tmp_path):
    """I-4 修复验证：list_tasks() 返回浅拷贝列表。"""
    mgr = BackgroundManager()
    mgr.start(_quick_cmd(), cwd=tmp_path)
    tasks = mgr.list_tasks()
    assert len(tasks) == 1
    original_status = tasks[0].status
    tasks[0].status = "tampered"
    # 再取一次，内部状态不变
    tasks2 = mgr.list_tasks()
    assert tasks2[0].status == original_status
    mgr.shutdown()


# ============ P1-3: 停滞看门狗测试 ============

def _long_silent_cmd(seconds=5):
    """跨平台：睡 N 秒后才 print（模拟交互卡死）。"""
    if sys.platform == "win32":
        return [sys.executable, "-c",
                f"import time; time.sleep({seconds}); print('finally')"]
    return ["sh", "-c", f"sleep {seconds}; echo finally"]


def _steady_output_cmd(rounds=5, interval=0.3):
    """跨平台：每 interval 秒 print 一行，共 rounds 次（模拟持续输出）。"""
    if sys.platform == "win32":
        code = (
            f"import time\n"
            f"for i in range({rounds}):\n"
            f"    print(f'line {{i}}', flush=True)\n"
            f"    time.sleep({interval})\n"
        )
        return [sys.executable, "-c", code]
    return ["sh", "-c", f"for i in $(seq 1 {rounds}); do echo line$i; sleep {interval}; done"]


def test_stall_watchdog_triggers_on_silent_process(tmp_path):
    """stall_timeout=1.0s 时，沉睡 5 秒的进程应触发停滞通知。"""
    mgr = BackgroundManager(stall_timeout=1.0)
    task_id = mgr.start(_long_silent_cmd(seconds=5), cwd=tmp_path)

    # 等停滞通知（最多 5 秒）
    stall_notif = None
    for _ in range(50):
        notifs = mgr.drain_notifications()
        for n in notifs:
            if n.get("stall"):
                stall_notif = n
                break
        if stall_notif:
            break
        time.sleep(0.1)

    assert stall_notif is not None, "未触发停滞通知"
    assert stall_notif["task_id"] == task_id
    assert stall_notif.get("stall_seconds") == 1.0

    # task 仍 running（停滞不主动 kill）
    task = mgr.status(task_id)
    assert task.status == "running"

    mgr.stop(task_id)
    mgr.shutdown()


def test_stall_watchdog_no_trigger_on_steady_output(tmp_path):
    """stall_timeout=1.0s 时，每 0.3 秒输出一行的进程不应触发停滞。"""
    mgr = BackgroundManager(stall_timeout=1.0)
    # 5 轮 * 0.3 秒 = 1.5 秒，每 0.3 < stall_timeout=1.0
    task_id = mgr.start(_steady_output_cmd(rounds=5, interval=0.3), cwd=tmp_path)

    # 等任务完成
    for _ in range(100):
        task = mgr.status(task_id)
        if task.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    notifs = mgr.drain_notifications()
    stalls = [n for n in notifs if n.get("stall")]
    assert len(stalls) == 0, f"持续输出不应触发停滞：{stalls}"
    assert task.status == "completed"
    mgr.shutdown()


def test_stall_watchdog_enabled_by_default():
    """stall_timeout 默认 45.0（看门狗默认开）。

    用户仍可显式传 stall_timeout=0 禁用。
    """
    mgr = BackgroundManager()
    assert mgr._stall_timeout == 45.0, (
        f"默认 stall_timeout 应为 45.0（看门狗默认开），实际: {mgr._stall_timeout}"
    )


def test_stall_watchdog_default_constant_is_45():
    """DEFAULT_STALL_TIMEOUT = 45.0（learn-claude-code 风格）。"""
    from agent.background import DEFAULT_STALL_TIMEOUT
    assert DEFAULT_STALL_TIMEOUT == 45.0


def test_stall_timeout_wired_from_config(tmp_path):
    """cli.py 的 RuntimeContext 从 config['bg_task']['stall_timeout'] 接线到 BackgroundManager。

    端到端：DEFAULT_CONFIG['bg_task']['stall_timeout']=45.0 →
           RuntimeContext → BackgroundManager._stall_timeout=45.0
    """
    from config import DEFAULT_CONFIG
    # 1. DEFAULT_CONFIG 里有这个字段
    assert DEFAULT_CONFIG["bg_task"]["stall_timeout"] == 45.0

    # 2. 直接用 DEFAULT_CONFIG 的 bg_task 段构造 BackgroundManager
    bg_cfg = DEFAULT_CONFIG["bg_task"]
    mgr = BackgroundManager(
        max_concurrent=bg_cfg.get("max_concurrent", 5),
        notification_stdout_cap=bg_cfg.get("notification_stdout_cap", 500),
        result_stdout_cap=bg_cfg.get("result_stdout_cap", 5000),
        default_timeout=bg_cfg.get("default_timeout", 600),
        stall_timeout=bg_cfg.get("stall_timeout", 45.0),
    )
    assert mgr._stall_timeout == 45.0


def test_stall_timeout_can_be_disabled_via_config(tmp_path):
    """config['bg_task']['stall_timeout']=0 时禁用看门狗（向后兼容）。"""
    mgr = BackgroundManager(
        stall_timeout=0,
    )
    assert mgr._stall_timeout == 0


def test_stall_notification_does_not_kill_process(tmp_path):
    """停滞通知后进程仍能正常完成（不主动 kill）。"""
    mgr = BackgroundManager(stall_timeout=0.5)
    task_id = mgr.start(_long_silent_cmd(seconds=2), cwd=tmp_path)

    # 等停滞 + 完成
    seen_stall = False
    for _ in range(100):
        notifs = mgr.drain_notifications()
        for n in notifs:
            if n.get("stall"):
                seen_stall = True
        task = mgr.status(task_id)
        if task.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert seen_stall, "应触发停滞"
    task = mgr.status(task_id)
    # 最终正常完成（exit_code=0）
    assert task.status == "completed"
    assert task.exit_code == 0
    mgr.shutdown()
