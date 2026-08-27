"""LSP 工具测试（JSON-RPC 层可注入 fake，不起真 pylsp）。

实现约定：_rpc_request 是唯一的 JSON-RPC 接缝——fake 它即可让 handler
全路径跑通；_ensure_server 恒 False 使 didOpen 跳过，机器装了 pylsp
也保持 hermetic（不真起子进程）。
"""


class TestLspTool:
    def test_check_fn_false_without_pylsp(self, monkeypatch):
        """which 找不到 pylsp → check_fn False（工具自动隐藏）。"""
        import tools.lsp_tool as lt
        monkeypatch.setattr(lt.shutil, "which", lambda name: None)
        lt._reset_server()
        assert lt._lsp_available() is False

    def test_definitions_via_fake_rpc(self, monkeypatch, tmp_path):
        """fake JSON-RPC → definitions 返回跳转位置。"""
        import tools.lsp_tool as lt
        f = tmp_path / "a.py"
        f.write_text("import os\n", encoding="utf-8")

        def fake_rpc(method, params):
            if method == "initialize":
                return {"capabilities": {}}
            if method == "textDocument/definition":
                return [{"uri": "file:///b.py", "range": {"start": {"line": 3, "character": 0}}}]
            return None

        monkeypatch.setattr(lt, "_rpc_request", fake_rpc)
        monkeypatch.setattr(lt, "_lsp_available", lambda: True)
        # 不起真 pylsp：_ensure_server 恒 False → _ensure_ready False → didOpen 跳过
        monkeypatch.setattr(lt, "_ensure_server", lambda: False)
        out = lt._handle_lsp({"action": "definitions", "path": str(f), "line": 0, "character": 7}, agent_home=None)
        import json
        data = json.loads(out)
        assert data["results"][0]["uri"].endswith("b.py")

    def test_references_via_fake_rpc(self, monkeypatch, tmp_path):
        """fake JSON-RPC → references 返回引用点列表。"""
        import tools.lsp_tool as lt
        f = tmp_path / "a.py"
        f.write_text("def foo(): pass\n", encoding="utf-8")

        def fake_rpc(method, params):
            if method == "textDocument/references":
                return [{"uri": "file:///c.py", "range": {"start": {"line": 9, "character": 4}}}]
            return None

        monkeypatch.setattr(lt, "_rpc_request", fake_rpc)
        monkeypatch.setattr(lt, "_lsp_available", lambda: True)
        monkeypatch.setattr(lt, "_ensure_server", lambda: False)
        import json
        data = json.loads(lt._handle_lsp(
            {"action": "references", "path": str(f), "line": 0, "character": 4}, agent_home=None))
        assert len(data["results"]) == 1

    def test_unavailable_returns_error(self, monkeypatch):
        """pylsp 不可用 → 统一 error JSON（error_type=lsp_unavailable）。"""
        import tools.lsp_tool as lt
        import json
        monkeypatch.setattr(lt, "_lsp_available", lambda: False)
        out = lt._handle_lsp({"action": "definitions", "path": "x.py", "line": 0, "character": 0}, agent_home=None)
        assert "error" in json.loads(out)

    def test_registered_in_core(self):
        import tools.lsp_tool  # noqa: F401 —— 触发模块级 register（支持单测方法独立跑）
        from tools.registry import registry
        assert "lsp" in registry.list_all()
