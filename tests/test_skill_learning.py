# -*- coding: utf-8 -*-
"""InstinctStore（instinct 行为记忆）单元测试。

CCAR15 Task 1：置信度累积存储层，对标 CCB instinctStore。
覆盖：合并累积 / scope 隔离 / cluster 归一化分组 / prune 过期 / JSON round-trip。
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.skill_learning.store import Instinct, InstinctStore


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _mk(trigger="run tests", action="pytest first", scope="global",
        confidence=0.3, evidence=None, updated_at=None):
    return Instinct(
        trigger=trigger,
        action=action,
        confidence=confidence,
        evidence=evidence if evidence is not None else ["obs-1"],
        scope=scope,
        updated_at=updated_at or _iso(datetime.now(timezone.utc)),
    )


class TestUpsertMerge:
    def test_same_key_second_upsert_confidence_rises(self, tmp_path):
        store = InstinctStore(tmp_path)
        r1 = store.upsert(_mk(confidence=0.3))
        assert r1.confidence == pytest.approx(0.3)
        r2 = store.upsert(_mk(confidence=0.3))
        # new = old * 0.9 + 0.25 = 0.3*0.9 + 0.25 = 0.52
        assert r2.confidence == pytest.approx(0.52)

    def test_confidence_monotonic_and_capped(self, tmp_path):
        store = InstinctStore(tmp_path)
        r = store.upsert(_mk(confidence=0.9))
        prev = r.confidence
        for _ in range(10):
            r = store.upsert(_mk(confidence=0.9))
            assert r.confidence >= prev  # 单调不降
            prev = r.confidence
        assert r.confidence <= 1.0
        assert r.confidence == pytest.approx(1.0)  # 封顶

    def test_evidence_dedup_and_cap_10(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(evidence=["a", "b"]))
        store.upsert(_mk(evidence=["b", "c"]))
        got = store.list_all(scope="global")
        assert len(got) == 1
        assert got[0].evidence == ["a", "b", "c"]

        ev = [f"e{i}" for i in range(15)]
        store.upsert(_mk(evidence=ev))
        got = store.list_all(scope="global")[0]
        assert len(got.evidence) == 10
        # 保留最新的（后进的）证据
        assert "e14" in got.evidence and "e0" not in got.evidence

    def test_trigger_normalization_merges_case_and_whitespace(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(trigger="Run  Tests"))
        store.upsert(_mk(trigger=" run tests "))
        assert len(store.list_all(scope="global")) == 1

    def test_different_action_no_merge(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(action="pytest first"))
        store.upsert(_mk(action="pytest all"))
        assert len(store.list_all(scope="global")) == 2


class TestScopeIsolation:
    def test_different_scope_not_merged(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(scope="global", confidence=0.4))
        store.upsert(_mk(scope="project:foo", confidence=0.4))
        g = store.list_all(scope="global")
        p = store.list_all(scope="project:foo")
        assert len(g) == 1 and len(p) == 1
        assert g[0].confidence == pytest.approx(0.4)  # 未被合并推高
        assert p[0].scope == "project:foo"

    def test_project_scope_key_with_windows_unsafe_chars(self, tmp_path):
        # scope 字符串带 ':'（Windows 目录名非法字符），落盘必须不炸且 round-trip 保真
        store = InstinctStore(tmp_path)
        store.upsert(_mk(scope="project:D--project-HermesAgent"))
        got = store.list_all(scope="project:D--project-HermesAgent")
        assert len(got) == 1
        assert got[0].scope == "project:D--project-HermesAgent"

    def test_list_all_no_filter_returns_all_scopes(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(scope="global"))
        store.upsert(_mk(scope="project:foo", action="x"))
        assert len(store.list_all()) == 2


class TestCluster:
    def test_cluster_groups_by_normalized_trigger(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(trigger="Run Tests", action="a"))
        store.upsert(_mk(trigger="run  tests", action="b"))
        store.upsert(_mk(trigger="deploy", action="c"))
        clusters = store.cluster("global")
        assert len(clusters) == 2
        norm_key = next(k for k in clusters if len(clusters[k]) == 2)
        assert {i.action for i in clusters[norm_key]} == {"a", "b"}
        # 归一化 key 是小写、压缩空白
        assert norm_key == "run tests"

    def test_cluster_scope_filtered(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(trigger="t", scope="global"))
        store.upsert(_mk(trigger="t", scope="project:foo"))
        assert len(store.cluster("global")) == 1
        assert len(store.cluster("project:foo")) == 1


class TestPrune:
    def test_prune_removes_stale_low_confidence(self, tmp_path):
        store = InstinctStore(tmp_path)
        old = _iso(datetime.now(timezone.utc) - timedelta(days=40))
        store.upsert(_mk(trigger="stale", confidence=0.2, updated_at=old))
        removed = store.prune(days=30)
        assert removed == 1
        assert store.list_all() == []

    def test_prune_keeps_old_high_confidence(self, tmp_path):
        store = InstinctStore(tmp_path)
        old = _iso(datetime.now(timezone.utc) - timedelta(days=40))
        store.upsert(_mk(trigger="golden", confidence=0.95, updated_at=old))
        assert store.prune(days=30) == 0
        assert len(store.list_all()) == 1

    def test_prune_keeps_recent_low_confidence(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(confidence=0.1))
        assert store.prune(days=30) == 0
        assert len(store.list_all()) == 1


class TestPersistence:
    def test_round_trip_across_instances(self, tmp_path):
        store1 = InstinctStore(tmp_path)
        inst = _mk(trigger="Round Trip Test", confidence=0.7,
                   evidence=["e1", "e2"], scope="project:bar")
        store1.upsert(inst)

        store2 = InstinctStore(tmp_path)
        got = store2.list_all(scope="project:bar")
        assert len(got) == 1
        r = got[0]
        assert r.trigger == "Round Trip Test"
        assert r.action == inst.action
        assert r.confidence == pytest.approx(0.7)
        assert r.evidence == ["e1", "e2"]
        assert r.scope == "project:bar"
        assert r.updated_at == inst.updated_at

    def test_file_layout_slug_and_dirs(self, tmp_path):
        store = InstinctStore(tmp_path)
        store.upsert(_mk(trigger="Run Tests! Now"))
        store.upsert(_mk(trigger="other", action="x", scope="project:foo"))

        root = Path(tmp_path) / "instincts"
        g_files = list((root / "global").glob("*.json"))
        p_files = list((root / "project-foo").glob("*.json"))
        assert len(g_files) == 1 and len(p_files) == 1
        # slug：非 [a-z0-9_-] 全替换为 '-'，截 60
        assert g_files[0].name == "run-tests--now.json"
        data = json.loads(g_files[0].read_text(encoding="utf-8"))
        assert data["trigger"] == "Run Tests! Now"

    def test_scope_dir_parent_not_confused_with_global(self, tmp_path):
        # "project:global" 的 scope 不能落到 global/ 目录里
        store = InstinctStore(tmp_path)
        store.upsert(_mk(scope="project:global"))
        assert store.list_all(scope="global") == []
        assert len(store.list_all(scope="project:global")) == 1

    def test_fingerprint_path_repeated_upsert_still_accumulates(self, tmp_path):
        # T1 Minor 补：同 trigger 不同 action 落指纹路径后，重复 upsert 该
        # action 仍要走置信度累积（指纹路径重读合并，而不是每次都当新条目）
        store = InstinctStore(tmp_path)
        store.upsert(_mk(action="a"))
        b1 = store.upsert(_mk(action="b", confidence=0.3))
        assert b1.confidence == pytest.approx(0.3)  # 首插原样落盘
        b2 = store.upsert(_mk(action="b", confidence=0.3))
        assert b2.confidence == pytest.approx(0.52)  # 指纹路径二次 upsert 累积
        assert len(store.list_all(scope="global")) == 2  # a / b 各一条


# ======================================================================
# Task 2：HeuristicObserver 四类启发信号
# ======================================================================
from unittest.mock import patch

from agent.skill_learning import observer


class _RecordingStore:
    """测试替身：记录 upsert 调用，不落盘。"""

    def __init__(self):
        self.calls = []

    def upsert(self, inst):
        self.calls.append(inst)
        return inst


def _obs(store, user_text="", tool_calls=None, tool_results=None, scope="global"):
    return observer.observe_turn(
        user_text=user_text,
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        store=store,
        scope=scope,
    )


class TestUserCorrection:
    def test_chinese_correction_creates_instinct(self):
        store = _RecordingStore()
        n = _obs(store, user_text="不要用grep，用ripgrep 来搜索")
        assert n == 1
        assert len(store.calls) == 1
        inst = store.calls[0]
        assert inst.trigger == "使用 grep"
        assert inst.action == "改用 ripgrep"
        assert inst.confidence == 0.5
        assert inst.scope == "global"
        assert inst.evidence and len(inst.evidence[0]) <= 80

    def test_english_correction_creates_instinct(self):
        store = _RecordingStore()
        n = _obs(store, user_text="don't use grep, use ripgrep here")
        assert n == 1
        inst = store.calls[0]
        assert inst.trigger == "使用 grep"
        assert inst.action == "改用 ripgrep"

    def test_plain_text_no_signal(self):
        store = _RecordingStore()
        assert _obs(store, user_text="今天天气不错") == 0
        assert store.calls == []


class TestFailureRecovery:
    def test_error_then_success_same_tool(self):
        store = _RecordingStore()
        calls = [
            {"name": "terminal", "arguments": {"command": "pytest tests/"}},
            {"name": "terminal", "arguments": {"command": "pytest tests/test_a.py"}},
        ]
        results = [
            {"name": "terminal", "error": "exit code 1: 3 failed", "content": ""},
            {"name": "terminal", "content": "3 passed"},
        ]
        assert _obs(store, tool_calls=calls, tool_results=results) == 1
        inst = store.calls[0]
        assert inst.trigger == "terminal 失败时"
        assert "重试" in inst.action
        assert "command" in inst.action  # 参数 diff 提到变化的参数名
        assert inst.confidence == 0.4
        assert inst.scope == "global"

    def test_error_without_recovery_no_signal(self):
        store = _RecordingStore()
        calls = [{"name": "terminal", "arguments": {"command": "x"}}]
        results = [{"name": "terminal", "error": "boom", "content": ""}]
        assert _obs(store, tool_calls=calls, tool_results=results) == 0
        assert store.calls == []


class TestRepeatedSequence:
    def test_same_triple_twice_creates_instinct(self):
        store = _RecordingStore()
        seq = ["read_file", "search", "edit_file"]
        calls = [{"name": n, "arguments": {}} for n in seq * 2]
        assert _obs(store, tool_calls=calls) == 1
        inst = store.calls[0]
        assert inst.action == "序列 read_file→search→edit_file"
        assert inst.trigger == "需要 edit_file"  # 目的词 = 序列终点工具
        assert inst.confidence == 0.3
        assert inst.scope == "global"

    def test_no_repeat_no_signal(self):
        store = _RecordingStore()
        calls = [{"name": n, "arguments": {}} for n in ["read_file", "search", "edit_file"]]
        assert _obs(store, tool_calls=calls) == 0
        assert store.calls == []


class TestProjectConvention:
    def test_convention_uses_internal_project_scope(self):
        store = _RecordingStore()
        with patch.object(observer, "get_project_memory_key", return_value="test-key"):
            n = _obs(store, user_text="提交信息必须用中文写并带模块前缀", scope="global")
        assert n == 1
        inst = store.calls[0]
        assert inst.trigger == "项目约定"
        assert "必须用中文写" in inst.action
        assert len(inst.action) <= 80
        assert inst.confidence == 0.45
        assert inst.scope == "project:test-key"

    def test_passed_project_key_used_for_scope(self):
        # 调用方传入项目 key 时，信号 4 直接用它（不再内部算）
        store = _RecordingStore()
        with patch.object(observer, "get_project_memory_key", return_value="wrong-key"):
            n = _obs(store, user_text="always run uv run pytest before commit", scope="my-proj")
        assert n == 1
        assert store.calls[0].scope == "project:my-proj"

    def test_short_convention_no_signal(self):
        store = _RecordingStore()
        # 关键词后不足 5 字符 → 不算约定
        assert _obs(store, user_text="必须") == 0
        assert store.calls == []


# ======================================================================
# Task 3：SkillEvolver（簇达标 → 生成 SKILL.md）+ 主循环接线
# ======================================================================
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent import AIAgent
from agent.memory_store import MemoryStore
from agent.skill_learning.evolver import maybe_evolve, _skill_md


def _seed(store, trigger, actions_conf, scope="global"):
    """往 store 里塞一个簇：同 trigger 多个 action（不同 action = 不同成员）。"""
    for action, conf in actions_conf:
        store.upsert(Instinct(
            trigger=trigger, action=action, confidence=conf,
            evidence=[f"ev-{action}"], scope=scope,
            updated_at=_iso(datetime.now(timezone.utc)),
        ))


class TestEvolveThreshold:
    def test_below_member_threshold_no_skill(self, tmp_path):
        # 簇成员只有 2（<3），即使平均 confidence 高也不生成
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [("a", 0.9), ("b", 0.9)])
        generated = maybe_evolve(store, "global", tmp_path / "skills")
        assert generated == []
        assert not (tmp_path / "skills").exists() or \
            not list((tmp_path / "skills").glob("learned-*"))

    def test_below_confidence_threshold_no_skill(self, tmp_path):
        # 成员 3 但平均 confidence 0.5 < 0.75，不生成
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [("a", 0.5), ("b", 0.5), ("c", 0.5)])
        assert maybe_evolve(store, "global", tmp_path / "skills") == []

    def test_empty_store_returns_empty(self, tmp_path):
        store = InstinctStore(tmp_path / "sl")
        assert maybe_evolve(store, "global", tmp_path / "skills") == []

    def test_scope_isolated_evolution(self, tmp_path):
        # project scope 的簇不参与 global 的演化判定
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "项目约定", [("a", 0.9), ("b", 0.9), ("c", 0.9)],
              scope="project:foo")
        assert maybe_evolve(store, "global", tmp_path / "skills") == []
        assert len(maybe_evolve(store, "project:foo", tmp_path / "skills")) == 1


class TestEvolveGenerates:
    def _mature_store(self, tmp_path):
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [
            ("先跑 pytest", 0.95), ("再报结果", 0.9), ("先看失败详情", 0.8),
        ])
        return store

    def test_generates_skill_md_file(self, tmp_path):
        store = self._mature_store(tmp_path)
        generated = maybe_evolve(store, "global", tmp_path / "skills")
        assert len(generated) == 1
        skill_md = Path(tmp_path) / "skills" / "learned-run-tests" / "SKILL.md"
        assert skill_md.exists()
        assert generated[0] == skill_md

    def test_skill_md_content_frontmatter_and_body(self, tmp_path):
        store = self._mature_store(tmp_path)
        maybe_evolve(store, "global", tmp_path / "skills")
        content = (Path(tmp_path) / "skills" / "learned-run-tests" / "SKILL.md") \
            .read_text(encoding="utf-8")
        # frontmatter
        assert content.startswith("---")
        assert "name: learned-run-tests" in content
        assert "description: " in content  # description = trigger 一句话
        # 正文：action + 证据
        assert "先跑 pytest" in content
        assert "ev-先跑 pytest" in content
        # 证据最多 3 条
        assert content.count("ev-") <= 3

    def test_idempotent_existing_skill_not_regenerated(self, tmp_path):
        store = self._mature_store(tmp_path)
        skills_dir = tmp_path / "skills"
        assert len(maybe_evolve(store, "global", skills_dir)) == 1
        # 第二次调用：技能已存在，不重复生成（后续维护交 curator）
        assert maybe_evolve(store, "global", skills_dir) == []
        assert len(list(skills_dir.glob("learned-*/SKILL.md"))) == 1


class TestSkillMdSlugFallback:
    def test_pure_chinese_trigger_fallback_slug(self, tmp_path):
        # 纯中文 trigger 提取不到 ascii 词 → habit-<成员数>-<hash> 保底
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "项目约定发布", [("a", 0.9), ("b", 0.9), ("c", 0.9)])
        generated = maybe_evolve(store, "global", tmp_path / "skills")
        assert len(generated) == 1
        # 目录名形如 learned-habit-3-xxxxxxxx
        name = generated[0].parent.name
        assert name.startswith("learned-habit-3-")
        assert len(name.split("learned-habit-3-")[1]) == 8  # 短 hash 8 位

    def test_mixed_chinese_ascii_trigger_uses_ascii_word(self, tmp_path):
        # "使用 grep" → 提取 ascii 词 "grep"（store 原始 slug 会变 "--grep"）
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "使用 grep", [("a", 0.9), ("b", 0.9), ("c", 0.9)])
        generated = maybe_evolve(store, "global", tmp_path / "skills")
        assert generated[0].parent.name == "learned-grep"

    def test_skill_md_direct_format(self):
        # _skill_md 纯函数：frontmatter + trigger/action/证据三段
        insts = [
            Instinct(trigger="run tests", action="先跑 pytest", confidence=0.9,
                     evidence=["用户说要跑测试"], scope="global",
                     updated_at="2026-01-01T00:00:00+00:00"),
            Instinct(trigger="run tests", action="再报结果", confidence=0.8,
                     evidence=["第二次观察"], scope="global",
                     updated_at="2026-01-01T00:00:00+00:00"),
        ]
        md = _skill_md("run tests", insts)
        assert "name: learned-run-tests" in md
        assert "run tests" in md  # trigger
        assert "先跑 pytest" in md and "再报结果" in md  # action
        assert "用户说要跑测试" in md  # 证据


class TestRunConversationWiring:
    """主循环接线：enabled 时轮末 observe；disabled / 子代理不跑；fail-open。"""

    def _make_agent(self, tmp_path, config=None, spawn_depth=0):
        agent = AIAgent(
            api_key="fake",
            model="test",
            enabled_toolsets=[],
            omnimate_home=tmp_path,
            memory_store=MemoryStore(omnimate_home=tmp_path),
            config=config if config is not None else {},
            spawn_depth=spawn_depth,
        )
        msg = SimpleNamespace(content="ok", tool_calls=None)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        agent.llm_client = SimpleNamespace(
            chat_completions=AsyncMock(return_value=resp))
        return agent

    async def test_observe_called_when_enabled(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        calls = []
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: calls.append(kw) or 0)
        monkeypatch.setattr(sl, "maybe_evolve", lambda *a, **k: [])

        agent = self._make_agent(
            tmp_path, config={"skill_learning": {"enabled": True}})
        resp = await agent.chat("不要用grep，用ripgrep")

        assert resp == "ok"
        assert len(calls) == 1
        assert calls[0]["user_text"] == "不要用grep，用ripgrep"
        assert calls[0]["scope"] == "global"
        assert calls[0]["store"] is not None

    async def test_not_called_when_disabled(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        calls = []
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: calls.append(kw) or 0)

        agent = self._make_agent(tmp_path, config={})  # 默认关
        await agent.chat("hello")

        assert calls == []

    async def test_not_called_for_subagent(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        calls = []
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: calls.append(kw) or 0)

        agent = self._make_agent(
            tmp_path, config={"skill_learning": {"enabled": True}},
            spawn_depth=1)
        await agent.chat("hello")

        assert calls == []  # 仅主代理 spawn_depth==0

    async def test_observer_raising_fail_open(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl

        def boom(**kw):
            raise RuntimeError("observer broken")

        monkeypatch.setattr(sl, "observe_turn", boom)
        agent = self._make_agent(
            tmp_path, config={"skill_learning": {"enabled": True}})
        resp = await agent.chat("hello")
        assert resp == "ok"  # 学习链路炸了不影响主对话

    async def test_no_memory_store_skipped(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        calls = []
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: calls.append(kw) or 0)
        agent = self._make_agent(
            tmp_path, config={"skill_learning": {"enabled": True}})
        agent.memory_store = None
        await agent.chat("hello")
        assert calls == []


class TestFailOpen:
    def test_upsert_raising_never_propagates(self):
        class BoomStore:
            def upsert(self, inst):
                raise RuntimeError("disk full")

        n = observer.observe_turn(
            user_text="不要用grep，用ripgrep",
            tool_calls=[],
            tool_results=[],
            store=BoomStore(),
            scope="global",
        )
        assert n == 0

    def test_malformed_inputs_return_zero(self):
        store = _RecordingStore()
        # tool_calls/tool_results 元素缺 name / arguments 也不炸
        assert _obs(store, tool_calls=[{"arguments": None}], tool_results=[{}]) == 0
        assert store.calls == []


# ======================================================================
# Task 4：LLM 观察后端 + config 白名单 + /skill-learning CLI
# ======================================================================
from agent.skill_learning import llm_observer
from agent.skill_learning.llm_observer import (
    observe_turn_llm,
    reset_llm_observer_state,
)


@pytest.fixture
def clean_llm_state():
    """LLM 观察器模块级状态（熔断/计数）隔离：每个用例前后重置。"""
    reset_llm_observer_state()
    yield
    reset_llm_observer_state()


def _router(content=None, side_effect=None):
    """假 aux_llm_router：chat_completions 返回 content（或抛 side_effect）。

    router 是"有 async chat_completions 的对象"（对齐 goal.py 的用法），
    不是 AsyncMock 本身——断言调用次数用 router.chat_completions.await_count。
    """
    msg = SimpleNamespace(content=content)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    if side_effect is not None:
        call = AsyncMock(side_effect=side_effect)
    else:
        call = AsyncMock(return_value=resp)
    return SimpleNamespace(chat_completions=call)


class TestObserveTurnLLMParse:
    async def test_parses_json_array_to_instincts(self, tmp_path, clean_llm_state):
        store = InstinctStore(tmp_path)
        router = _router(content=json.dumps([{
            "trigger": "用户要求跑测试", "action": "先跑 pytest 再汇报",
            "confidence": 0.8,
        }], ensure_ascii=False))
        got = await observe_turn_llm(
            user_text="跑一下测试", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 1
        assert got[0].trigger == "用户要求跑测试"
        assert got[0].action == "先跑 pytest 再汇报"
        assert got[0].confidence == pytest.approx(0.8)
        assert got[0].scope == "global"
        assert got[0].evidence  # 证据非空
        # 已入库
        assert len(store.list_all(scope="global")) == 1

    async def test_code_fence_tolerated(self, tmp_path, clean_llm_state):
        store = InstinctStore(tmp_path)
        router = _router(
            content='```json\n[{"trigger":"t","action":"a","confidence":0.5}]\n```')
        got = await observe_turn_llm(
            user_text="x", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 1 and got[0].trigger == "t"

    async def test_max_three_items(self, tmp_path, clean_llm_state):
        store = InstinctStore(tmp_path)
        items = [{"trigger": f"t{i}", "action": f"a{i}", "confidence": 0.5}
                 for i in range(5)]
        router = _router(content=json.dumps(items))
        got = await observe_turn_llm(
            user_text="x", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 3  # 上限 3 条（prompt 已声明）

    async def test_nan_and_inf_confidence_fall_to_default(self, tmp_path,
                                                          clean_llm_state):
        """NaN/Infinity 不能被 min(1.0, nan) 钳成 1.0（方向反了）。

        Python json.loads 接受 NaN/Infinity 字面量；clamp 对它们失效
        （nan < 1.0 为 False → min 返回 1.0）——必须落保守默认 0.4。
        """
        store = InstinctStore(tmp_path)
        # json.dumps 也接受这些字面量（allow_nan 默认 True）
        router = _router(content=json.dumps([
            {"trigger": "t", "action": "a", "confidence": float("nan")},
            {"trigger": "u", "action": "b", "confidence": float("inf")},
        ]))
        got = await observe_turn_llm(
            user_text="x", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 2
        assert got[0].confidence == pytest.approx(0.4)
        assert got[1].confidence == pytest.approx(0.4)

    async def test_empty_array_is_success_not_failure(self, tmp_path, clean_llm_state):
        """[] 是合法成功：返回空列表，且重置熔断计数（不算失败）。"""
        store = InstinctStore(tmp_path)
        router = _router(content="[]")
        got = await observe_turn_llm(
            user_text="闲聊", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert got == []
        assert llm_observer._consecutive_failures == 0

    async def test_garbage_falls_back_to_heuristic(self, tmp_path, clean_llm_state):
        """LLM 返回垃圾 → 回退启发式（返回启发式写入的条目）。"""
        store = InstinctStore(tmp_path)
        router = _router(content="这不是 JSON，抱歉")
        got = await observe_turn_llm(
            user_text="不要用grep，用ripgrep 搜索", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 1
        assert got[0].trigger == "使用 grep"  # 启发式信号 1 的产物
        assert llm_observer._consecutive_failures == 1

    async def test_router_none_falls_back(self, tmp_path, clean_llm_state):
        store = InstinctStore(tmp_path)
        got = await observe_turn_llm(
            user_text="不要用grep，用ripgrep 搜索", tool_calls=[], tool_results=[],
            aux_llm_router=None, store=store,
        )
        assert len(got) == 1 and got[0].trigger == "使用 grep"

    async def test_llm_exception_falls_back(self, tmp_path, clean_llm_state):
        store = InstinctStore(tmp_path)
        router = _router(side_effect=RuntimeError("api down"))
        got = await observe_turn_llm(
            user_text="不要用grep，用ripgrep 搜索", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert len(got) == 1  # 回退启发式，永不抛


class TestObserverCircuitBreaker:
    async def test_three_failures_open_circuit(self, tmp_path, clean_llm_state, monkeypatch):
        """连续 3 次失败开闸；第 4 次不再调 LLM 直接回退启发式。"""
        clock = {"t": 0.0}
        monkeypatch.setattr(llm_observer, "_now", lambda: clock["t"])
        store = InstinctStore(tmp_path)
        router = _router(content="garbage")
        for _ in range(3):
            await observe_turn_llm(
                user_text="不要用grep，用ripgrep", tool_calls=[], tool_results=[],
                aux_llm_router=router, store=store,
            )
        assert llm_observer._circuit_open is True
        assert router.chat_completions.await_count == 3

        # 第 4 次：熔断开闸 → 不调 LLM，直接启发式
        got = await observe_turn_llm(
            user_text="不要用grep，用ripgrep", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert router.chat_completions.await_count == 3  # 没有第 4 次调用
        assert len(got) == 1 and got[0].trigger == "使用 grep"  # 启发式兜底

    async def test_cooldown_recovery(self, tmp_path, clean_llm_state, monkeypatch):
        """开闸 30s 冷却期满 → 合闸重试（LLM 被再次调用）。"""
        clock = {"t": 0.0}
        monkeypatch.setattr(llm_observer, "_now", lambda: clock["t"])
        store = InstinctStore(tmp_path)
        router = _router(content="garbage")
        for _ in range(3):
            await observe_turn_llm(
                user_text="x", tool_calls=[], tool_results=[],
                aux_llm_router=router, store=store,
            )
        assert llm_observer._circuit_open is True

        clock["t"] = 31.0  # 冷却期满
        await observe_turn_llm(
            user_text="x", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert router.chat_completions.await_count == 4  # 冷却后恢复调用
        # 合闸时清零计数，本次失败计数=1（需再失败 2 次才重新开闸）
        assert llm_observer._consecutive_failures == 1

    async def test_session_cap_20(self, tmp_path, clean_llm_state):
        """每会话 LLM 调用上限 20 次；第 21 次直接回退启发式不再调。"""
        store = InstinctStore(tmp_path)
        router = _router(content="[]")
        for _ in range(20):
            await observe_turn_llm(
                user_text="x", tool_calls=[], tool_results=[],
                aux_llm_router=router, store=store,
            )
        assert router.chat_completions.await_count == 20

        got = await observe_turn_llm(
            user_text="不要用grep，用ripgrep", tool_calls=[], tool_results=[],
            aux_llm_router=router, store=store,
        )
        assert router.chat_completions.await_count == 20  # 上限后不再调 LLM
        assert len(got) == 1 and got[0].trigger == "使用 grep"  # 启发式兜底

    def test_reset_state(self):
        reset_llm_observer_state()
        llm_observer._consecutive_failures = 2
        llm_observer._circuit_open = True
        llm_observer._session_call_count = 10
        reset_llm_observer_state()
        assert llm_observer._consecutive_failures == 0
        assert llm_observer._circuit_open is False
        assert llm_observer._session_call_count == 0


class TestEvolveParams:
    def test_custom_lower_threshold_generates(self, tmp_path):
        """默认门槛 0.75 不达标，但传入更低 evolve_threshold 后达标。"""
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [("a", 0.6), ("b", 0.6), ("c", 0.6)])
        assert maybe_evolve(store, "global", tmp_path / "skills") == []
        assert len(maybe_evolve(
            store, "global", tmp_path / "skills", min_avg_confidence=0.5)) == 1

    def test_custom_min_members(self, tmp_path):
        """默认 3 成员不达标，传入 min_members=2 后达标。"""
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [("a", 0.9), ("b", 0.9)])
        assert maybe_evolve(store, "global", tmp_path / "skills") == []
        assert len(maybe_evolve(
            store, "global", tmp_path / "skills", min_members=2)) == 1

    def test_default_thresholds_unchanged(self, tmp_path):
        """不传参时默认行为与 T3 完全一致（0.75 / 3）。"""
        store = InstinctStore(tmp_path / "sl")
        _seed(store, "run tests", [("a", 0.9), ("b", 0.9), ("c", 0.9)])
        assert len(maybe_evolve(store, "global", tmp_path / "skills")) == 1


class TestWiringObserverConfig:
    """主循环接线：observer 配置选后端 + 演化门槛从 config 传入。"""

    def _make_agent(self, tmp_path, config=None):
        agent = AIAgent(
            api_key="fake",
            model="test",
            enabled_toolsets=[],
            omnimate_home=tmp_path,
            memory_store=MemoryStore(omnimate_home=tmp_path),
            config=config if config is not None else {},
        )
        msg = SimpleNamespace(content="ok", tool_calls=None)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        agent.llm_client = SimpleNamespace(
            chat_completions=AsyncMock(return_value=resp))
        return agent

    async def test_llm_backend_selected(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        llm_calls = []
        heur_calls = []
        monkeypatch.setattr(
            sl, "observe_turn_llm",
            AsyncMock(side_effect=lambda **kw: llm_calls.append(kw) or []))
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: heur_calls.append(kw) or 0)

        agent = self._make_agent(
            tmp_path, config={"skill_learning": {
                "enabled": True, "observer": "llm"}})
        agent.aux_llm_router = object()
        resp = await agent.chat("hello")

        assert resp == "ok"
        assert len(llm_calls) == 1
        assert llm_calls[0]["aux_llm_router"] is agent.aux_llm_router
        assert heur_calls == []  # llm 后端不直接走启发式入口

    async def test_heuristic_backend_default(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        heur_calls = []
        llm_calls = []
        monkeypatch.setattr(
            sl, "observe_turn",
            lambda **kw: heur_calls.append(kw) or 0)
        monkeypatch.setattr(
            sl, "observe_turn_llm",
            AsyncMock(side_effect=lambda **kw: llm_calls.append(kw) or []))

        agent = self._make_agent(
            tmp_path, config={"skill_learning": {"enabled": True}})
        await agent.chat("hello")
        assert len(heur_calls) == 1
        assert llm_calls == []  # 默认后端是启发式，不调 LLM 入口

    async def test_evolve_thresholds_passed_from_config(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl
        evolve_calls = []
        monkeypatch.setattr(sl, "observe_turn", lambda **kw: 0)
        monkeypatch.setattr(
            sl, "maybe_evolve",
            lambda *a, **kw: evolve_calls.append((a, kw)) or [])

        agent = self._make_agent(
            tmp_path, config={"skill_learning": {
                "enabled": True, "evolve_threshold": 0.5,
                "evolve_min_cluster": 2}})
        await agent.chat("hello")

        assert len(evolve_calls) == 1
        _args, kwargs = evolve_calls[0]
        assert kwargs["min_avg_confidence"] == 0.5
        assert kwargs["min_members"] == 2

    async def test_llm_backend_fail_open(self, tmp_path, monkeypatch):
        import agent.skill_learning as sl

        async def boom(**kw):
            raise RuntimeError("llm observer broken")

        monkeypatch.setattr(sl, "observe_turn_llm", boom)
        agent = self._make_agent(
            tmp_path, config={"skill_learning": {
                "enabled": True, "observer": "llm"}})
        resp = await agent.chat("hello")
        assert resp == "ok"  # LLM 后端炸了不影响主对话


# ---------------------------------------------------------------------------
# /skill-learning CLI
# ---------------------------------------------------------------------------

class _FakeSkillLearningRT:
    """/skill-learning 命令测试用最小 RT（对齐 test_cli_commands 的 FakeRT 模式）。"""

    def __init__(self, tmp_path):
        self.home = tmp_path
        self.config = {}
        self.agent = SimpleNamespace(config={})


class TestSkillLearningCLI:
    def test_status_shows_counts(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)
        # 种子：2 条同 trigger instinct（1 个 global 簇）+ 1 个 learned- 技能
        store = InstinctStore(tmp_path / ".skill-learning")
        store.upsert(_mk(trigger="run tests", action="a"))
        store.upsert(_mk(trigger="run tests", action="b"))
        skills = tmp_path / "skills" / "learned-foo"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text(
            "---\nname: learned-foo\n---\n", encoding="utf-8")

        assert _handle_command("/skill-learning status", rt) is True
        out = capsys.readouterr().out
        assert "instinct 总数" in out and "2" in out
        assert "global 簇数" in out and "1" in out
        assert "已进化技能" in out and "1" in out

    def test_start_stop_flips_runtime_config(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)

        assert _handle_command("/skill-learning start", rt) is True
        assert rt.config["skill_learning"]["enabled"] is True
        out = capsys.readouterr().out
        assert "config_set" in out  # 提示持久化通道

        assert _handle_command("/skill-learning stop", rt) is True
        assert rt.config["skill_learning"]["enabled"] is False

    def test_prune_removes_stale(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)
        store = InstinctStore(tmp_path / ".skill-learning")
        old = _iso(datetime.now(timezone.utc) - timedelta(days=40))
        store.upsert(_mk(trigger="stale", confidence=0.2, updated_at=old))

        assert _handle_command("/skill-learning prune", rt) is True
        out = capsys.readouterr().out
        assert "清理 1" in out
        assert store.list_all() == []

    def test_evolve_manual_trigger(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)
        store = InstinctStore(tmp_path / ".skill-learning")
        _seed(store, "run tests", [
            ("先跑 pytest", 0.95), ("再报结果", 0.9), ("先看失败详情", 0.8)])

        assert _handle_command("/skill-learning evolve", rt) is True
        out = capsys.readouterr().out
        assert "1 个技能" in out
        assert (tmp_path / "skills" / "learned-run-tests" / "SKILL.md").exists()

    def test_evolve_config_threshold_respected(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)
        rt.config["skill_learning"] = {
            "evolve_threshold": 0.5, "evolve_min_cluster": 2}
        store = InstinctStore(tmp_path / ".skill-learning")
        _seed(store, "run tests", [("a", 0.6), ("b", 0.6)])

        assert _handle_command("/skill-learning evolve", rt) is True
        out = capsys.readouterr().out
        assert "1 个技能" in out  # 放宽后的门槛达标

    def test_unknown_subcommand_shows_usage(self, tmp_path, capsys):
        from cli import _handle_command
        rt = _FakeSkillLearningRT(tmp_path)
        assert _handle_command("/skill-learning bogus", rt) is True
        assert "用法" in capsys.readouterr().out
