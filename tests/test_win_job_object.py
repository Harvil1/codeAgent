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
