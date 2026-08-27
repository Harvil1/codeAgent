# -*- coding: utf-8 -*-
"""记忆检索与用量统计回归测试。

  记忆检索三件套：recentTools 反噪音 / alreadySurfaced 跨轮去重 /
     staleness 过期警示（memory_retriever + memory_injection + agent 接线）
  per-model token/成本追踪（usage_tracker + _record_llm_usage 接线）
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def _fake_llm(raw_ids: str, captured: dict):
    class _Client:
        async def chat_completions(self, messages, model=None, **kw):
            captured["prompt"] = messages[0]["content"]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=raw_ids),
            )])
    return _Client()


# ======================================================================
# 检索器反噪音 + 跨轮去重
# ======================================================================

def test_retriever_active_tools_rule_h9():
    from agent.memory_retriever import retrieve_relevant
    captured = {}
    client = _fake_llm('["general#a"]', captured)

    async def run():
        return await retrieve_relevant(
            query="spawn 子代理", index_text="- general#a: x",
            llm_client=client, model=None,
            active_tools=["subagent", "terminal"],
        )
    out = asyncio.run(run())
    assert out == ["general#a"]
    assert "subagent" in captured["prompt"]
    assert "不要" in captured["prompt"] and "用法" in captured["prompt"]


def test_retriever_exclude_ids_filtered_h9():
    from agent.memory_retriever import retrieve_relevant
    captured = {}
    # LLM 故意把已注入过的 b 也选回来
    client = _fake_llm('["general#a", "general#b", "general#c"]', captured)

    async def run():
        return await retrieve_relevant(
            query="q", index_text="- x",
            llm_client=client, model=None,
            exclude_ids={"general#b"},
        )
    out = asyncio.run(run())
    # 确定性后过滤：b 被滤掉，不依赖 LLM 听话
    assert out == ["general#a", "general#c"]
    assert "general#b" in captured["prompt"], "prompt 应提示已注入列表"


# ======================================================================
# 注入侧 surfaced 汇 + staleness 警示
# ======================================================================

def _entry(name, days_ago, eid):
    return SimpleNamespace(
        id=eid, type="user", name=name, body="b",
        updated_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


def test_injection_surfaced_and_staleness_h9():
    from agent.memory_injection import build_relevant_memories_message, reset_injection_cache

    reset_injection_cache()
    e_old = _entry("旧记忆", 30, "general#old")
    e_new = _entry("新记忆", 0, "general#new")
    store = SimpleNamespace(
        full_index_text_with_age=lambda: "- idx",
        get=lambda mid: {"general#old": e_old, "general#new": e_new}[mid],
    )
    captured = {}

    async def run():
        return await build_relevant_memories_message(
            query="q", memory_store=store,
            aux_llm_router=_fake_llm('["general#old", "general#new"]', captured),
            surfaced=surfaced,
        )

    surfaced = set()
    msg = asyncio.run(run())
    assert msg is not None
    assert surfaced == {"general#old", "general#new"}, "选中的 id 应收进 surfaced"
    assert "已过期" in msg["content"], ">1 天的记忆应带过期警示"
    # 两条都老/新混合：old 有标注即可（new 是 0 天，无标注）
    assert "30d" in msg["content"]


def test_agent_recent_active_tools_h9(tmp_path):
    from agent import AIAgent
    agent = AIAgent(api_key="fake", model="t", omnimate_home=tmp_path,
                    enabled_toolsets=[])
    agent.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "c3", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}},
        ]},
    ]
    tools = agent._recent_active_tools()
    assert tools == ["terminal", "read_file"], "按时间正序去重"


# ======================================================================
# UsageTracker
# ======================================================================

def test_usage_tracker_record_summary_persist_h8(tmp_path):
    from agent.usage_tracker import UsageTracker

    t = UsageTracker(tmp_path, "s1")
    t.record(model="deepseek-chat", prompt=1_000_000, completion=500_000,
             cache_read=200_000, cache_creation=100_000)
    t.record(model="some-unknown-model", prompt=100, completion=100)

    s = t.summary()
    assert s["models"]["deepseek-chat"]["calls"] == 1
    # 只统计 token，不算金额（价格计算已按用户裁决移除）
    assert "cost_usd" not in s["models"]["deepseek-chat"]
    assert "cost_usd" not in s["models"]["some-unknown-model"]
    assert "cost_usd" not in s["totals"]
    assert s["totals"]["calls"] == 2

    # 会话持久化：新实例从磁盘恢复
    t2 = UsageTracker(tmp_path, "s1")
    assert t2.summary()["models"]["deepseek-chat"]["calls"] == 1


def test_record_llm_usage_feeds_tracker_h8(tmp_path):
    from agent import AIAgent
    from agent.usage_tracker import UsageTracker

    agent = AIAgent(api_key="fake", model="m", omnimate_home=tmp_path,
                    enabled_toolsets=[])
    tracker = UsageTracker(tmp_path, "s2")
    agent.set_usage_tracker(tracker)

    resp = SimpleNamespace(
        model="deepseek-chat",
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=50,
            prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=20,
        ),
    )
    agent._record_llm_usage(resp, sent_message_count=5)
    row = tracker.summary()["models"]["deepseek-chat"]
    assert row["calls"] == 1
    assert row["prompt"] == 100 and row["completion"] == 50
    assert row["cache_read"] == 80 and row["cache_creation"] == 20
    # 权威锚点照常（tracker 不影响既有语义）
    assert agent._last_usage_anchor[0] == 5
