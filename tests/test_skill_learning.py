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
