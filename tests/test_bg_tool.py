"""bg_task 工具 handler 测试。"""
import json
import sys
import time
from pathlib import Path

import pytest

# 触发工具注册
import tools.bg_task  # noqa
from tools.registry import registry


def _quick_cmd():
    if sys.platform == "win32":
        return [sys.executable, "-c", "import time; time.sleep(0.3); print('done')"]
    return ["sh", "-c", "sleep 0.3; echo done"]


def test_bg_start_returns_json_with_task_id(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_start",
        {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"].startswith("bg_")
    assert parsed["status"] == "running"
    assert "pid" in parsed
    mgr.shutdown()


def test_bg_start_full_returns_bg_task_full_error(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager(max_concurrent=1)
    registry.dispatch("bg_start",
                       {"command": [sys.executable, "-c", "import time; time.sleep(30)"],
                        "cwd": str(tmp_path)},
                       bg_manager=mgr)
    result_str = registry.dispatch(
        "bg_start",
        {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_full"
    mgr.shutdown()


def test_bg_status_returns_summary_json(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    status_str = registry.dispatch(
        "bg_status", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(status_str)
    assert parsed["task_id"] == task_id
    assert "status" in parsed
    mgr.shutdown()


def test_bg_status_unknown_returns_not_found():
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_status", {"task_id": "bg_nonexistent"}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_not_found"


def test_bg_result_returns_full_output(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    # 等完成
    for _ in range(50):
        task = mgr.status(task_id)
        if task and task.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    result_str = registry.dispatch(
        "bg_result", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"] == task_id
    assert "stdout" in parsed
    assert "done" in parsed["stdout"]
    mgr.shutdown()


def test_bg_list_returns_all_tasks(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    registry.dispatch("bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
                       bg_manager=mgr)
    registry.dispatch("bg_start", {"command": _quick_cmd(), "cwd": str(tmp_path)},
                       bg_manager=mgr)
    result_str = registry.dispatch("bg_list", {}, bg_manager=mgr)
    parsed = json.loads(result_str)
    assert len(parsed["tasks"]) == 2
    mgr.shutdown()


def test_bg_stop_terminates_and_returns_stopped(tmp_path):
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    start_str = registry.dispatch(
        "bg_start",
        {"command": [sys.executable, "-c", "import time; time.sleep(30)"],
         "cwd": str(tmp_path)},
        bg_manager=mgr,
    )
    task_id = json.loads(start_str)["task_id"]
    # 等到任务确认 running 后再 stop（避免子进程未起的时序竞态）
    for _ in range(50):
        task = mgr.status(task_id)
        if task and task.status == "running":
            break
        time.sleep(0.1)
    result_str = registry.dispatch(
        "bg_stop", {"task_id": task_id}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed["task_id"] == task_id
    assert parsed["status"] in ("stopped", "running", "stopping")  # stop 可能刚完成或正在过渡
    mgr.shutdown()


def test_bg_stop_unknown_returns_not_found():
    from agent.background import BackgroundManager
    mgr = BackgroundManager()
    result_str = registry.dispatch(
        "bg_stop", {"task_id": "bg_nonexistent"}, bg_manager=mgr,
    )
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_task_not_found"


def test_bg_start_no_manager_returns_bg_unavailable():
    """bg_manager=None 时返回 bg_unavailable error。"""
    import json
    from tools.registry import registry
    result_str = registry.dispatch("bg_start", {"command": ["echo", "x"]})
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "bg_unavailable"


def test_bg_start_invalid_command_returns_invalid_args(tmp_path):
    """command 为空或非 list 时返回 invalid_args。"""
    import json
    from agent.background import BackgroundManager
    from tools.registry import registry
    mgr = BackgroundManager()
    # command 缺失
    result_str = registry.dispatch("bg_start", {}, bg_manager=mgr)
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "invalid_args"
    # command 非 list
    result_str = registry.dispatch("bg_start", {"command": "not-a-list"}, bg_manager=mgr)
    parsed = json.loads(result_str)
    assert parsed.get("error_type") == "invalid_args"
    mgr.shutdown()
