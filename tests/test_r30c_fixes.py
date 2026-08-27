# -*- coding: utf-8 -*-
"""并发/一致性修复回归测试。

覆盖：
  memory_injection ContextVar 隔离 + delegation_queue 实例定向
  team bus 锁超时 fail-closed（MessageBusLockTimeout）
  workflow 预算事前预留 + 事后结算（reserve/settle）
  DAG 依赖删除/缺失自动解链
  cron catch_up 单次语义（无行为变更，cron 既有测试覆盖）
  CLI 对象哨兵 + agent drain 跳过非 str 项
  journal save_meta 原子写
  GUI 判定 token 级（_is_gui_launch）
  deny 防御失效显式 ERROR 日志
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
# delegation_queue 实例定向
# ======================================================================

def test_aiagent_has_own_delegation_queue_c1(tmp_path):
    from agent import AIAgent
    from tools.delegate_tool import _delegation_queue

    a = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                enabled_toolsets=[])
    b = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                enabled_toolsets=[])
    assert a._delegation_queue is not None
    assert a._delegation_queue is not b._delegation_queue, "每个 agent 应有独立队列"
    assert a._delegation_queue is not _delegation_queue, "不应复用全局兜底队列"


def test_resolve_delegation_queue_directs_to_agent_ref_c1(tmp_path):
    from agent import AIAgent
    from tools.delegate_tool import _resolve_delegation_queue, _delegation_queue

    agent = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                    enabled_toolsets=[])
    # 有 agent_ref → 实例队列
    assert _resolve_delegation_queue({"agent_ref": agent}) is agent._delegation_queue
    # 无 agent_ref → 兜底全局
    assert _resolve_delegation_queue({}) is _delegation_queue


def test_memory_injection_cache_isolated_across_tasks_c1():
    """两个 asyncio task（各自 context 副本）不共享注入缓存。"""
    from agent.memory_injection import build_relevant_memories_message

    entry = SimpleNamespace(type="user", name="n", body="b")
    store = SimpleNamespace(
        full_index_text_with_age=lambda: "- general#a: n",
        get=lambda mid: entry,
    )
    mock_rr = AsyncMock(return_value=["general#a"])

    async def one_call():
        return await build_relevant_memories_message(
            query="q", memory_store=store, aux_llm_router=MagicMock())

    async def scenario():
        await asyncio.create_task(one_call())
        await asyncio.create_task(one_call())  # 新 task = 新 context 副本

    with patch("agent.memory_injection.retrieve_relevant", new=mock_rr):
        asyncio.run(scenario())
    assert mock_rr.await_count == 2, "跨 task 不应共享缓存（防并发 agent 串味）"


# ======================================================================
# bus 锁超时 fail-closed
# ======================================================================

def test_bus_lock_timeout_fail_closed_c2(tmp_path, monkeypatch):
    import agent.team.bus as bus_mod

    bus = bus_mod.MessageBus(team_dir=tmp_path / "team")

    def _timeout(fileobj, timeout=30.0):
        raise TimeoutError("locked")

    monkeypatch.setattr(bus_mod, "_acquire_lock", _timeout)
    with pytest.raises(bus_mod.MessageBusLockTimeout):
        bus.send(from_="a", to="b", type_="message", content="hi")
    with pytest.raises(bus_mod.MessageBusLockTimeout):
        bus.read_inbox("b")


def test_bus_normal_path_unaffected_c2(tmp_path):
    from agent.team.bus import MessageBus

    bus = MessageBus(team_dir=tmp_path / "team")
    bus.send(from_="a", to="b", type_="message", content="hi")
    msgs = bus.read_inbox("b")
    assert len(msgs) == 1 and msgs[0].content == "hi"
    assert bus.read_inbox("b") == []  # 消费式清空


# ======================================================================
# workflow 预算 reserve/settle
# ======================================================================

def test_workflow_budget_reserve_settle_c3():
    from agent.workflow_engine import WorkflowBudget, WorkflowBudgetExceeded

    b = WorkflowBudget(total=100)
    g1 = b.reserve(50)
    assert g1 == 50 and b.remaining == 50
    g2 = b.reserve(60)          # 只授予剩余的 50
    assert g2 == 50 and b.remaining == 0
    with pytest.raises(WorkflowBudgetExceeded):
        b.reserve(1)            # 耗尽在跑之前拒
    b.settle(g2, 0)             # 归还第二笔（未花）
    assert b.remaining == 50
    b.settle(g1, 40)            # 实花 40（reserved 清零）
    assert b.spent == 40 and b.remaining == 60
    with pytest.raises(WorkflowBudgetExceeded):
        b.settle(0, 100)        # 超支在结算时抛（与原 spend 语义一致）


@pytest.mark.asyncio
async def test_workflow_budget_exhausted_before_second_run_c3():
    from agent.workflow_engine import run_workflow

    calls = []

    async def runner(prompt: str) -> str:
        calls.append(prompt)
        return "x" * 400  # 估算 100 token

    script = "async def main():\n    await agent('a')\n    await agent('b')\n"
    result = await run_workflow(
        script, agent_runner=runner, budget_total=100,
    )
    # 第二次调用在跑之前被拒（runner 只跑了 1 次），脚本以预算错误收场
    assert len(calls) == 1
    assert result.get("ok") is False
    assert "预算" in (result.get("error") or "")


# ======================================================================
# DAG 依赖删除/缺失自动解链
# ======================================================================

def test_can_start_deleted_dependency_unlinks_c4(tmp_path):
    from agent.task_store import TaskStore

    store = TaskStore(omnimate_home=tmp_path)
    parent = store.create(subject="P", description="")
    child = store.create(subject="C", description="", blocked_by=[parent["id"]])
    assert store.can_start(child["id"]) is False  # parent 未完成

    store.delete(parent["id"])  # 软删除（status=deleted）
    assert store.can_start(child["id"]) is True, "依赖删除后应自动解链"
    assert any(t["id"] == child["id"] for t in store.find_ready())


def test_can_start_missing_dependency_file_unlinks_c4(tmp_path):
    from agent.task_store import TaskStore

    store = TaskStore(omnimate_home=tmp_path)
    parent = store.create(subject="P", description="")
    child = store.create(subject="C", description="", blocked_by=[parent["id"]])
    # 模拟文件被外部删掉
    (Path(store._dir) / f"{parent['id']}.json").unlink()
    assert store.can_start(child["id"]) is True, "依赖文件缺失应视为已满足"


# ======================================================================
# agent drain 跳过非 str 项（对象哨兵）
# ======================================================================

def test_drain_queued_input_skips_non_str_c6(tmp_path):
    import queue as queue_mod
    from agent import AIAgent

    agent = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                    enabled_toolsets=[])
    q = queue_mod.Queue()
    agent.set_input_queue(q)
    q.put(object())          # EOF/中断对象哨兵
    q.put("排队消息")
    agent._drain_queued_input()
    assert len(agent._pending_ephemeral_messages) == 1
    assert "排队消息" in agent._pending_ephemeral_messages[0]["content"]


# ======================================================================
# journal save_meta 原子写
# ======================================================================

def test_journal_save_meta_atomic_c7(tmp_path, monkeypatch):
    import agent.atomic_io as aio_mod
    import agent.workflow_journal as wj_mod

    calls = []
    orig = aio_mod.atomic_write_text

    def spy(path, content, **kw):
        calls.append(Path(path).name)
        return orig(path, content, **kw)

    monkeypatch.setattr(aio_mod, "atomic_write_text", spy)
    journal = wj_mod.WorkflowJournal(tmp_path / "run1")
    journal.save_meta({"status": "running"})
    journal.save_meta({"step": 2})
    assert calls == ["meta.json", "meta.json"], "save_meta 应走原子写"
    data = json.loads((tmp_path / "run1" / "meta.json").read_text(encoding="utf-8"))
    assert data["status"] == "running" and data["step"] == 2


# ======================================================================
# GUI 判定 token 级
# ======================================================================

def test_is_gui_launch_token_level_b5():
    from tools.terminal_tool import _is_gui_launch
    # 真启动
    assert _is_gui_launch("start chrome") is True
    assert _is_gui_launch("chrome.exe --headless") is True
    assert _is_gui_launch("C:/Tools/chrome.exe --flag") is True
    assert _is_gui_launch("build.sh && notepad.exe") is True  # 复合段首 token
    # 假命中（此前子串匹配误吞输出）
    assert _is_gui_launch('echo "chrome.exe"') is False
    assert _is_gui_launch("grep chrome.exe log.txt") is False
    assert _is_gui_launch("type notepad.exe.txt") is False
    # 普通命令
    assert _is_gui_launch("ls -la") is False


# ======================================================================
# deny 防御失效显式 ERROR
# ======================================================================

@pytest.mark.asyncio
async def test_dispatch_deny_failure_logs_error_b7(caplog, monkeypatch):
    import tools.skill_tools  # noqa: F401 注册 skills_list
    from tools.registry import registry

    def _boom(name):
        raise RuntimeError("rules broken")

    monkeypatch.setattr("agent.tool_permissions.is_tool_denied", _boom)
    with caplog.at_level("ERROR"):
        result = await registry.dispatch("skills_list", {})
    # fail-open 语义不变：工具正常执行
    data = json.loads(result)
    assert data.get("error_type") != "unknown_tool"
    assert any("deny 规则加载失败" in r.message for r in caplog.records), (
        "规则加载失败必须显式 ERROR（不再静默 pass）"
    )
