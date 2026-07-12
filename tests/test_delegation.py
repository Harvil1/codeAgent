"""委托系统测试。"""

import json
from unittest.mock import patch

import pytest

from tools.delegate_tool import (
    DelegationCompletionQueue, get_delegation_queue,
    _handle_delegate_task, _build_child_system_prompt,
    _delegate_sync, _delegate_batch, _run_child,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# DelegationCompletionQueue
# ---------------------------------------------------------------------------

def test_queue_push_drain():
    q = DelegationCompletionQueue()
    q.push({"id": 1, "result": "a"})
    q.push({"id": 2, "result": "b"})
    assert q.has_pending()

    drained = q.drain()
    assert len(drained) == 2
    assert drained[0]["id"] == 1
    assert not q.has_pending()


def test_queue_drain_empty():
    q = DelegationCompletionQueue()
    assert q.drain() == []
    assert q.has_pending() is False


# ---------------------------------------------------------------------------
# _build_child_system_prompt
# ---------------------------------------------------------------------------

def test_child_prompt_contains_goal():
    prompt = _build_child_system_prompt("搜索网页", "用户问 X", "leaf")
    assert "搜索网页" in prompt
    assert "用户问 X" in prompt
    assert "leaf" in prompt


def test_child_prompt_no_context():
    prompt = _build_child_system_prompt("任务", "", "orchestrator")
    assert "任务" in prompt
    assert "orchestrator" in prompt
    # 无 context 时不应该有"来自父代理的上下文"
    assert "来自父代理" not in prompt


def test_child_prompt_leaf_restriction():
    prompt = _build_child_system_prompt("g", "c", "leaf")
    assert "不能再派生" in prompt


# ---------------------------------------------------------------------------
# _handle_delegate_task 参数验证
# ---------------------------------------------------------------------------

def test_delegate_no_goal_no_tasks():
    """goal 和 tasks 都空时返回错误。"""
    result = registry.dispatch("delegate_task", {})
    data = json.loads(result)
    assert "error" in data


def test_delegate_orchestrator_depth_limit(monkeypatch):
    """orchestrator 达到深度上限时拒绝。"""
    monkeypatch.setenv("_SPAWN_DEPTH", "2")
    result = registry.dispatch(
        "delegate_task",
        {"goal": "test", "role": "orchestrator"},
        max_spawn_depth=2,
    )
    data = json.loads(result)
    assert "error" in data
    assert "嵌套深度" in data["error"]


def test_delegate_depth_limit_allows_within_range(monkeypatch):
    """orchestrator 在深度范围内允许。"""
    monkeypatch.setenv("_SPAWN_DEPTH", "1")
    # mock _run_child 避免真创建子代理
    with patch("tools.delegate_tool._run_child", return_value="子代理结果"):
        result = registry.dispatch(
            "delegate_task",
            {"goal": "test", "role": "orchestrator"},
            max_spawn_depth=2,
        )
    data = json.loads(result)
    # 不应该有深度错误
    assert "error" not in data or "嵌套深度" not in data.get("error", "")


# ---------------------------------------------------------------------------
# _delegate_sync
# ---------------------------------------------------------------------------

def test_delegate_sync_success():
    with patch("tools.delegate_tool._run_child", return_value="sync result"):
        result = _delegate_sync("goal", "ctx", "leaf")
    data = json.loads(result)
    assert data["success"] is True
    assert data["result"] == "sync result"
    assert data["mode"] == "sync"


def test_delegate_sync_failure():
    with patch("tools.delegate_tool._run_child", side_effect=RuntimeError("boom")):
        result = _delegate_sync("goal", "ctx", "leaf")
    data = json.loads(result)
    assert data["success"] is False
    assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# _delegate_async
# ---------------------------------------------------------------------------

def test_delegate_async_returns_immediately():
    """异步委托立即返回，结果进队列。"""
    queue = get_delegation_queue()
    # 清空队列
    queue.drain()

    with patch("tools.delegate_tool._run_child", return_value="async result"):
        result = registry.dispatch(
            "delegate_task",
            {"goal": "test", "background": True},
        )
        data = json.loads(result)
        assert data["mode"] == "async"
        assert "delegation_id" in data

        # 等后台线程完成（简化：轮询）
        import time
        for _ in range(50):
            if queue.has_pending():
                break
            time.sleep(0.05)

        drained = queue.drain()
        assert len(drained) == 1
        assert drained[0]["success"] is True
        assert drained[0]["result"] == "async result"


# ---------------------------------------------------------------------------
# _delegate_batch
# ---------------------------------------------------------------------------

def test_delegate_batch_parallel():
    """批量委托并行执行。"""
    with patch("tools.delegate_tool._run_child", return_value="batch result") as mock:
        result = _delegate_batch(
            [
                {"goal": "task1"},
                {"goal": "task2"},
                {"goal": "task3"},
            ],
            background=False,
        )
    data = json.loads(result)
    assert data["mode"] == "batch"
    assert len(data["results"]) == 3
    # 每个 task 都成功
    for r in data["results"]:
        assert r["success"] is True
    # _run_child 被调用 3 次
    assert mock.call_count == 3


def test_delegate_batch_handles_failure():
    """批量委托中单个失败不影响其他。"""
    def side_effect(goal, *args, **kwargs):
        if "fail" in goal:
            raise RuntimeError("intentional")
        return "ok"

    with patch("tools.delegate_tool._run_child", side_effect=side_effect):
        result = _delegate_batch(
            [
                {"goal": "ok-1"},
                {"goal": "will-fail"},
                {"goal": "ok-2"},
            ],
            background=False,
        )
    data = json.loads(result)
    successes = [r["success"] for r in data["results"]]
    assert successes.count(True) == 2
    assert successes.count(False) == 1


# ---------------------------------------------------------------------------
# _run_child（需要 mock AIAgent）
# ---------------------------------------------------------------------------

def test_run_child_missing_api_key(monkeypatch):
    """无 API key 时抛错。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # mock config 返回无 key 的配置
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key_env": "NO_SUCH_KEY", "base_url": None},
    }):
        with pytest.raises(RuntimeError, match="API key"):
            _run_child("goal", "", "leaf")
