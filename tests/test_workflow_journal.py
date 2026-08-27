"""journal 断点恢复测试。"""
from agent.workflow_journal import call_key

SRC = "async def main():\n    return 1\n"


class TestWorkflowJournal:
    def test_create_and_roundtrip(self, tmp_path):
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run1", SRC)
        assert (tmp_path / "run1" / "script.py").read_text(encoding="utf-8") == SRC
        assert j.lookup("k1") is None
        seq = j.append("k1", {"kind": "ok", "output": "v1"})
        assert seq == 1
        assert j.lookup("k1") == {"kind": "ok", "output": "v1"}
        assert j.lookup("missing") is None

    def test_reload_persists(self, tmp_path):
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run2", SRC)
        j.append("k1", {"kind": "ok", "output": "v1"})
        j2 = WorkflowJournal.load(tmp_path / "run2")
        assert j2.lookup("k1") == {"kind": "ok", "output": "v1"}

    def test_script_hash_mismatch_truncates(self, tmp_path):
        """脚本变了 → 整体截断（runWorkflow.ts:57 语义）。"""
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run3", SRC)
        j.append("k1", {"kind": "ok", "output": "v1"})
        (tmp_path / "run3" / "script.py").write_text(
            "async def main():\n    return 2\n", encoding="utf-8")
        j2 = WorkflowJournal.load(tmp_path / "run3")
        assert j2.lookup("k1") is None  # 截断

    def test_verify_script(self, tmp_path):
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run4", SRC)
        assert j.verify_script(SRC) is True
        assert j.verify_script("async def main():\n    return 9\n") is False

    def test_meta_roundtrip(self, tmp_path):
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run5", SRC)
        j.save_meta({"status": "running"})
        assert WorkflowJournal.load(tmp_path / "run5").load_meta()["status"] == "running"

    def test_call_key_stable_and_canonical(self):
        from agent.workflow_journal import call_key as ck
        assert ck("p", {"a": 1, "b": 2}) == ck("p", {"b": 2, "a": 1})
        assert ck("p", None) != ck("q", None)


class TestEngineJournalIntegration:
    async def test_cached_call_skips_runner(self, tmp_path):
        """journal 命中 → runner 不再被调（断点续跑核心语义）。"""
        from agent.workflow_engine import run_workflow
        from agent.workflow_journal import WorkflowJournal
        j = WorkflowJournal.create(tmp_path / "run", "async def main():\n    return await agent('x')\n")
        j.append(call_key("x", None), {"kind": "ok", "output": "cached!"})
        calls = []

        async def runner(p):
            calls.append(p)
            return "fresh"

        out = await run_workflow(
            "async def main():\n    return await agent('x')\n",
            agent_runner=runner, journal=j)
        assert out["return"] == "cached!"
        assert calls == []
        assert out["stats"]["cached"] == 1
