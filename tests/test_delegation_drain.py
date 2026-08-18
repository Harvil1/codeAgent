"""async 子代理完成通知送达父代理的端到端测试。

Bug 背景：CCAR5 加了 _delegate_async → push 到 DelegationCompletionQueue，
但父代理主循环没 drain，async 结果永远无法送达父代理。
Fix：_drain_injected_messages drain delegation_queue + _assemble_turn_messages 注入。
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    DelegationCompletionQueue, get_delegation_queue,
)


# ---------------------------------------------------------------------------
# 1. DelegationCompletionQueue 基础（已有 test_delegation.py 覆盖，这里补 drain 返回新队列场景）
# ---------------------------------------------------------------------------

def test_queue_drain_resets_isolated():
    """drain 后 queue 应为空，新 push 走新一轮。"""
    q = DelegationCompletionQueue()
    q.push({"delegation_id": "del_1", "success": True, "result": "ok"})
    assert q.has_pending()
    drained = q.drain()
    assert len(drained) == 1
    assert not q.has_pending()
    # 第二次 drain 应为空
    assert q.drain() == []
    # 新 push 不受影响
    q.push({"delegation_id": "del_2", "success": True, "result": "again"})
    assert q.has_pending()


def test_queue_thread_safe_lock_present():
    """queue 内部有 _lock（防并发 push/drain 竞争）。"""
    q = DelegationCompletionQueue()
    assert hasattr(q, "_lock"), "DelegationCompletionQueue 应有 _lock 字段"


# ---------------------------------------------------------------------------
# 2. _drain_injected_messages 读 delegation_queue
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_queue(monkeypatch):
    """每个测试用独立的 queue 实例（避免全局单例污染）。"""
    fresh = DelegationCompletionQueue()
    monkeypatch.setattr(
        "tools.delegate_tool._delegation_queue", fresh,
    )
    return fresh


def test_drain_injected_messages_pulls_delegation(isolated_queue, tmp_path):
    """_drain_injected_messages 真读 delegation_queue。

    R30c-C1：drain 读 agent 实例队列（delegate push 侧定向到 agent_ref），
    测试直接把实例队列换成 isolated_queue。
    """
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=Path(tmp_path),
        enabled_toolsets=[],
    )
    # 关掉其他 drain 源
    agent.bg_manager = None
    agent.cron_scheduler = None
    agent.team_bus = None
    agent.team_name = None
    agent._delegation_queue = isolated_queue

    isolated_queue.push({
        "delegation_id": "del_a",
        "goal": "探索模块",
        "success": True,
        "result": "发现 3 个文件",
    })
    isolated_queue.push({
        "delegation_id": "del_b",
        "goal": "重构 A",
        "success": False,
        "error": "boom",
    })

    drained = agent._drain_injected_messages()
    assert "delegation_results" in drained
    assert len(drained["delegation_results"]) == 2
    assert drained["delegation_results"][0]["delegation_id"] == "del_a"
    assert drained["delegation_results"][1]["delegation_id"] == "del_b"
    # drain 后 queue 应空
    assert not isolated_queue.has_pending()


def test_drain_injected_messages_empty_queue_returns_empty_list(isolated_queue, tmp_path):
    """空 queue 时 delegation_results 是 []（不是 None / 缺 key）。"""
    from agent import AIAgent

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

    drained = agent._drain_injected_messages()
    assert drained["delegation_results"] == []


def test_drain_injected_messages_fail_open(isolated_queue, tmp_path, monkeypatch):
    """queue 抛异常时主流程不崩，delegation_results 为空。"""
    from agent import AIAgent

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

    # mock 实例队列抛异常（R30c-C1：drain 读 agent 实例队列）
    class _BoomQueue:
        def has_pending(self):
            raise RuntimeError("queue 挂了")
    agent._delegation_queue = _BoomQueue()

    drained = agent._drain_injected_messages()
    assert drained["delegation_results"] == []
    # 其他源仍应正常
    assert drained["bg_notifications"] == []


# ---------------------------------------------------------------------------
# 3. _assemble_turn_messages 注入 delegation_results
# ---------------------------------------------------------------------------

def _build_agent(tmp_path):
    from agent import AIAgent
    agent = AIAgent(
        api_key="fake",
        model="test",
        omnimate_home=Path(tmp_path),
        enabled_toolsets=[],
    )
    agent.conversation_history = []
    return agent


def test_assemble_injects_successful_delegation(tmp_path):
    """成功结果转一条 user 消息，含 [后台子代理完成] 标记。"""
    agent = _build_agent(tmp_path)
    injected = {
        "bg_notifications": [],
        "cron_messages": [],
        "team_messages_text": "",
        "delegation_results": [
            {
                "delegation_id": "del_x",
                "goal": "探索目录",
                "success": True,
                "result": "发现 5 个文件",
            },
        ],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    # 找含 [后台子代理完成] 的 user 消息
    completion_msgs = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理完成]" in m["content"]
    ]
    assert len(completion_msgs) == 1
    assert "del_x" in completion_msgs[0]["content"]
    assert "探索目录" in completion_msgs[0]["content"]
    assert "发现 5 个文件" in completion_msgs[0]["content"]
    assert "<delegation_completion>" in completion_msgs[0]["content"]
    # 消费后 injected 应清空
    assert injected["delegation_results"] == []


def test_assemble_injects_failed_delegation(tmp_path):
    """失败结果转一条 user 消息，含 [后台子代理失败] 标记。"""
    agent = _build_agent(tmp_path)
    injected = {
        "bg_notifications": [],
        "cron_messages": [],
        "team_messages_text": "",
        "delegation_results": [
            {
                "delegation_id": "del_y",
                "goal": "执行任务",
                "success": False,
                "error": "LLM 限流",
            },
        ],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    failure_msgs = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理失败]" in m["content"]
    ]
    assert len(failure_msgs) == 1
    assert "del_y" in failure_msgs[0]["content"]
    assert "LLM 限流" in failure_msgs[0]["content"]


def test_assemble_truncates_long_result(tmp_path):
    """result 超 2000 字截断（防 context 爆炸）。"""
    agent = _build_agent(tmp_path)
    long_result = "A" * 5000
    injected = {
        "bg_notifications": [],
        "cron_messages": [],
        "team_messages_text": "",
        "delegation_results": [
            {
                "delegation_id": "del_long",
                "goal": "大任务",
                "success": True,
                "result": long_result,
            },
        ],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    completion_msgs = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理完成]" in m["content"]
    ]
    assert len(completion_msgs) == 1
    content = completion_msgs[0]["content"]
    # 截断标记应存在
    assert "truncated" in content
    # 5000 字不应全在
    assert "AAAA" * 1000 not in content


def test_assemble_multi_delegations(tmp_path):
    """多个 delegation 结果转多条 user 消息。"""
    agent = _build_agent(tmp_path)
    injected = {
        "bg_notifications": [],
        "cron_messages": [],
        "team_messages_text": "",
        "delegation_results": [
            {"delegation_id": "d1", "goal": "g1", "success": True, "result": "r1"},
            {"delegation_id": "d2", "goal": "g2", "success": True, "result": "r2"},
            {"delegation_id": "d3", "goal": "g3", "success": False, "error": "e3"},
        ],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    completions = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理完成]" in m["content"]
    ]
    failures = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理失败]" in m["content"]
    ]
    assert len(completions) == 2
    assert len(failures) == 1


def test_assemble_no_delegations_no_injection(tmp_path):
    """空 delegation_results 时不注入任何 delegation 消息。"""
    agent = _build_agent(tmp_path)
    injected = {
        "bg_notifications": [],
        "cron_messages": [],
        "team_messages_text": "",
        "delegation_results": [],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    delegation_msgs = [
        m for m in msgs
        if m["role"] == "user" and "delegation_completion" in m.get("content", "")
    ]
    assert len(delegation_msgs) == 0


def test_assemble_interleaved_with_other_injected(tmp_path):
    """delegation 跟 bg/cron/team 一起注入（不互斥）。"""
    agent = _build_agent(tmp_path)
    injected = {
        "bg_notifications": [
            {"task_id": "t1", "status": "done", "exit_code": 0, "stdout": "ok"},
        ],
        "cron_messages": [
            {"job_id": "j1", "message": "tick"},
        ],
        "team_messages_text": "[from alice] hi",
        "delegation_results": [
            {"delegation_id": "del_z", "goal": "g", "success": True, "result": "done"},
        ],
    }
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    contents = [m.get("content", "") for m in msgs if m["role"] == "user"]
    joined = "\n".join(contents)
    assert "task_notification" in joined
    assert "scheduled_message" in joined
    assert "team_messages" in joined
    assert "delegation_completion" in joined


# ---------------------------------------------------------------------------
# 4. 端到端：drain_injected → assemble_turn
# ---------------------------------------------------------------------------

def test_end_to_end_drain_to_assemble(isolated_queue, tmp_path):
    """push 到 queue → _drain_injected_messages → _assemble_turn_messages → user 消息送达。"""
    from agent import AIAgent

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
    agent.conversation_history = []
    agent._delegation_queue = isolated_queue  # R30c-C1：drain 读实例队列

    # step 1：push 两个 async 子代理完成
    isolated_queue.push({
        "delegation_id": "del_e2e_1",
        "goal": "探索 A",
        "success": True,
        "result": "找到 3 处",
    })
    isolated_queue.push({
        "delegation_id": "del_e2e_2",
        "goal": "探索 B",
        "success": False,
        "error": "network",
    })

    # step 2：drain
    injected = agent._drain_injected_messages()
    assert len(injected["delegation_results"]) == 2
    assert not isolated_queue.has_pending()

    # step 3：assemble
    msgs = agent._assemble_turn_messages(system_prompt="sys", injected=injected)
    completion = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理完成]" in m["content"]
    ]
    failure = [
        m for m in msgs
        if m["role"] == "user" and "[后台子代理失败]" in m["content"]
    ]
    assert len(completion) == 1
    assert len(failure) == 1
    assert "del_e2e_1" in completion[0]["content"]
    assert "找到 3 处" in completion[0]["content"]
    assert "del_e2e_2" in failure[0]["content"]
    assert "network" in failure[0]["content"]


def test_second_drain_after_first_consumed(isolated_queue, tmp_path):
    """第一轮 drain+assemble 后，第二轮没有新 push 时不再注入。"""
    from agent import AIAgent

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
    agent.conversation_history = []
    agent._delegation_queue = isolated_queue  # R30c-C1：drain 读实例队列

    isolated_queue.push({
        "delegation_id": "del_once",
        "goal": "一次",
        "success": True,
        "result": "ok",
    })

    # round 1
    injected1 = agent._drain_injected_messages()
    msgs1 = agent._assemble_turn_messages("sys", injected1)
    assert any("[后台子代理完成]" in m.get("content", "")
               for m in msgs1 if m["role"] == "user")

    # round 2：没有新 push，应不再注入
    injected2 = agent._drain_injected_messages()
    assert injected2["delegation_results"] == []
    msgs2 = agent._assemble_turn_messages("sys", injected2)
    assert not any("delegation_completion" in m.get("content", "")
                   for m in msgs2 if m["role"] == "user")
