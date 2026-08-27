# tests/test_win_job_object.py
import subprocess
import sys
import time

import pytest

from agent.win_job_object import WinJobObject, create_job_for_subprocess

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows Job Object 仅 Windows"
)


def test_job_create_and_assign():
    job = WinJobObject()
    p = subprocess.Popen([sys.executable, "-c", "print('hi')"])
    assert job.assign_process(p.pid) is True
    p.wait(timeout=10)
    job.close()


def test_kill_on_close_terminates_tree():
    """close() 句柄关闭 → KILL_ON_JOB_CLOSE 清理整个进程树。"""
    job = WinJobObject()
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    assert job.assign_process(p.pid) is True
    job.close()  # 触发 kill-on-close
    deadline = time.time() + 5
    while time.time() < deadline:
        if p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is not None, "close() 后子进程未被清理（kill-on-close 失效）"


def test_kill_terminates():
    job = WinJobObject()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    job.assign_process(p.pid)
    job.kill()
    deadline = time.time() + 5
    while time.time() < deadline:
        if p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is not None
    job.close()


def test_create_job_for_subprocess_failopen(monkeypatch):
    """ctypes 失败 → 返回 None（fail-open）。"""
    import agent.win_job_object as wjo
    monkeypatch.setattr(wjo, "_create_raw_job", lambda: None)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert create_job_for_subprocess(p) is None
    finally:
        p.wait(timeout=10)


# ---------------------------------------------------------------------------
# terminal 工具端到端（真实 job，无 mock）
# ---------------------------------------------------------------------------

def test_terminal_sandbox_on_real_job_end_to_end():
    """真实链路：sandbox on → terminal 命令正常执行 + job attach/close 全程无异常。"""
    import json
    import tools.terminal_tool as tt
    import agent.sandbox_runner as sr

    # 前置：Windows 上沙箱应可用且走 Job Object 模式
    sr.reset_availability_cache()
    assert sr.is_available() is True
    assert sr.uses_job_object() is True

    result = tt._handle_terminal(
        {"command": "echo hi"},
        sandbox_mode="on",
    )
    parsed = json.loads(result)
    assert "error" not in parsed, f"沙箱命令执行异常: {parsed}"
    assert "hi" in parsed.get("stdout", "")
    assert parsed.get("exit_code") == 0
