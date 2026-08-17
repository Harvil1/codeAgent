"""R28 W4：workflow 工具 + registry 测试。"""

SRC = (
    "async def main():\n"
    "    rs = await parallel([lambda: agent('a'), lambda: agent('b')])\n"
    "    return rs\n"
)

class TestWorkflowRegistry:
    def test_discover_and_override(self, tmp_path, monkeypatch):
        from agent import workflow_registry as WR
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        for d in (home / "workflows", proj / ".omnimate" / "workflows"):
            d.mkdir(parents=True)
        (home / "workflows" / "w1.py").write_text(SRC, encoding="utf-8")
        (proj / ".omnimate" / "workflows" / "w1.py").write_text(
            "async def main():\n    return 'proj'\n", encoding="utf-8")
        monkeypatch.setenv("OMNIMATE_HOME", str(home))
        monkeypatch.setattr("agent.workspace_context.get_workspace_cwd",
                            lambda: str(proj))
        WR._invalidate_cache()
        got = WR.load_workflow_scripts()
        assert "'proj'" in got["w1"]
        WR._invalidate_cache()  # 还原缓存，防污染其他测试

    def test_empty(self, tmp_path, monkeypatch):
        from agent import workflow_registry as WR
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "none"))
        monkeypatch.setattr("agent.workspace_context.get_workspace_cwd",
                            lambda: str(tmp_path))
        WR._invalidate_cache()
        assert WR.load_workflow_scripts() == {}
        WR._invalidate_cache()


class TestWorkflowTool:
    async def test_run_inline(self, monkeypatch, tmp_path):
        """run + script 内联 → 引擎跑通 → run 目录落盘。"""
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        from tools import workflow_tool as WT
        import agent.workflow_engine as WE

        async def fake_runner(p):
            return f"echo:{p}"
        monkeypatch.setattr(WE, "make_agent_runner", lambda kw: fake_runner)

        out = json.loads(await WT._handle_workflow(
            {"action": "run", "script": SRC}, config={}))
        assert out["ok"] is True
        assert sorted(out["return"]) == ["echo:a", "echo:b"]
        assert out["run_id"]
        # run 目录落盘
        from constants import get_omnimate_home
        run_dir = get_omnimate_home() / ".workflows" / out["run_id"]
        assert (run_dir / "script.py").exists()
        assert (run_dir / "journal.jsonl").exists()

    async def test_resume_uses_snapshot(self, monkeypatch, tmp_path):
        """resume 只信 run 目录快照（改 registry 不影响）。"""
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        from tools import workflow_tool as WT
        import agent.workflow_engine as WE

        async def fake_runner(p):
            return "echo:" + p
        monkeypatch.setattr(WE, "make_agent_runner", lambda kw: fake_runner)
        out1 = json.loads(await WT._handle_workflow(
            {"action": "run", "script": SRC}, config={}))
        run_id = out1["run_id"]
        from constants import get_omnimate_home
        snap = get_omnimate_home() / ".workflows" / run_id / "script.py"
        snap.write_text(
            "async def main():\n    return 'changed'\n", encoding="utf-8")
        out2 = json.loads(await WT._handle_workflow(
            {"action": "resume", "run_id": run_id}, config={}))
        # 快照变了 → hash 失配 → journal 截断重跑（用改后的快照）
        assert out2["ok"] is True and out2["return"] == "changed"

    async def test_status_list_kill(self, monkeypatch, tmp_path):
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        from tools import workflow_tool as WT
        import agent.workflow_engine as WE

        async def fake_runner(p):
            return "r"
        monkeypatch.setattr(WE, "make_agent_runner", lambda kw: fake_runner)
        out1 = json.loads(await WT._handle_workflow(
            {"action": "run", "script": "async def main():\n    return 1\n"},
            config={}))
        rid = out1["run_id"]
        st = json.loads(await WT._handle_workflow(
            {"action": "status", "run_id": rid}, config={}))
        assert st["status"] == "completed"
        ls = json.loads(await WT._handle_workflow({"action": "list"}, config={}))
        assert any(r["run_id"] == rid for r in ls["runs"])
        kill = json.loads(await WT._handle_workflow(
            {"action": "kill", "run_id": rid}, config={}))
        assert kill.get("killed") is False  # 已结束的 run 无活跃事件

    async def test_status_does_not_truncate_journal(self, monkeypatch, tmp_path):
        """status 是只读操作——hash 失配（手改快照）也不得删 journal。"""
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        from tools import workflow_tool as WT
        import agent.workflow_engine as WE

        async def fake_runner(p):
            return "r"
        monkeypatch.setattr(WE, "make_agent_runner", lambda kw: fake_runner)
        out1 = json.loads(await WT._handle_workflow(
            {"action": "run", "script": "async def main():\n    return await agent('x')\n"},
            config={}))
        rid = out1["run_id"]
        from constants import get_omnimate_home
        snap = get_omnimate_home() / ".workflows" / rid / "script.py"
        snap.write_text("async def main():\n    return 2\n", encoding="utf-8")  # 手改 → hash 失配
        st = json.loads(await WT._handle_workflow(
            {"action": "status", "run_id": rid}, config={}))
        assert st["status"] == "completed"
        assert st["journal_entries"] >= 1  # journal 还在（没被 status 截断）

    async def test_run_by_name_from_registry(self, monkeypatch, tmp_path):
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        (tmp_path / "home" / "workflows").mkdir(parents=True)
        (tmp_path / "home" / "workflows" / "demo.py").write_text(
            "async def main():\n    return 'named'\n", encoding="utf-8")
        from tools import workflow_tool as WT
        import agent.workflow_engine as WE

        monkeypatch.setattr(WE, "make_agent_runner", lambda kw: None)
        out = json.loads(await WT._handle_workflow(
            {"action": "run", "name": "demo"}, config={}))
        assert out["ok"] is True and out["return"] == "named"

    async def test_invalid_script_rejected(self, monkeypatch, tmp_path):
        import json
        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        from tools import workflow_tool as WT
        out = json.loads(await WT._handle_workflow(
            {"action": "run", "script": "import os\nasync def main():\n    return 1\n"},
            config={}))
        assert out["ok"] is False and out["error_type"] == "invalid_script"

    async def test_registered_and_blacklisted(self):
        from tools.registry import registry
        from toolsets import ASYNC_AGENT_DISALLOWED_TOOLS
        assert "workflow" in registry.list_all()
        assert "workflow" in ASYNC_AGENT_DISALLOWED_TOOLS
