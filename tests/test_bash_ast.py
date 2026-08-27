"""bashlex AST wrapper 测试（真 bashlex，非 mock）。"""


class TestParseInfo:
    def test_simple_command(self):
        from agent.bash_ast import parse_info
        info = parse_info("git status")
        assert info is not None
        assert info["segments"] == [["git", "status"]]
        assert info["has_redirect"] is False

    def test_compound_segments(self):
        from agent.bash_ast import parse_info
        info = parse_info("echo hi && git status | grep foo")
        assert info is not None
        # 三个简单命令段：echo hi / git status / grep foo（管道两侧各算一段）
        flat = [" ".join(s) for s in info["segments"]]
        assert "echo hi" in flat
        assert "git status" in flat
        assert "grep foo" in flat

    def test_quoted_ampersand_not_split(self):
        """引号内的 && 是 word 参数，不是命令分隔——AST 的核心价值。"""
        from agent.bash_ast import parse_info
        info = parse_info('echo "a && rm -rf /"')
        assert info is not None
        assert len(info["segments"]) == 1
        assert info["segments"][0][0] == "echo"
        assert "rm -rf" in " ".join(info["segments"][0])

    def test_env_assignment_segment(self):
        from agent.bash_ast import parse_info
        info = parse_info("FOO=1 rm -rf build")
        assert info is not None
        seg = " ".join(info["segments"][0])
        assert "rm" in seg  # assignment 前缀在段内，下游归一化处理

    def test_redirect_flag(self):
        from agent.bash_ast import parse_info
        info = parse_info("echo hi > out.txt")
        assert info is not None
        assert info["has_redirect"] is True

    def test_invalid_syntax_returns_none(self):
        from agent.bash_ast import parse_info
        assert parse_info("echo 'unclosed") is None

    def test_empty_returns_none(self):
        from agent.bash_ast import parse_info
        assert parse_info("") is None
        assert parse_info("   ") is None

    def test_missing_lib_returns_none(self, monkeypatch):
        """bashlex import 失败 → None（fail-open，不抛）。"""
        import builtins
        from agent import bash_ast
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "bashlex":
                raise ImportError("no bashlex")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert bash_ast.parse_info("echo hi") is None

    def test_substitution_flagged(self):
        from agent.bash_ast import parse_info
        info = parse_info("echo $(rm -rf /)")
        assert info is not None
        assert info["has_substitution"] is True

    def test_backtick_flagged(self):
        from agent.bash_ast import parse_info
        info = parse_info("echo `ls`")
        assert info is not None
        assert info["has_substitution"] is True

    def test_procsubstitution_flagged(self):
        """<() 进程替换实测 kind 是 'processsubstitution'——一并置标记。"""
        from agent.bash_ast import parse_info
        info = parse_info("cat <(ls)")
        assert info is not None
        assert info["has_substitution"] is True
