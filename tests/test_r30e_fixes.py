# -*- coding: utf-8 -*-
"""工具输出健壮性回归测试。

  terminal 大输出先无损落盘再截断兜底（stdout+stderr 都接 offload）
  read_file 同文件同 range mtime+size 去重（file_unchanged stub）+ 写后失效
  search_files offset 分页 + 上下文行 + 大小写开关
"""
import json
from pathlib import Path
from types import SimpleNamespace


# ======================================================================
# terminal 大输出无损落盘
# ======================================================================

def test_terminal_big_output_offload_full_h1(tmp_path, monkeypatch):
    from tools import terminal_tool as tt

    big_out = "A" * 30000 + "MIDDLE_MARKER" + "B" * 30000  # ~60K，中段有标记
    big_err = "E" * 51000

    def _fake_run(*a, **kw):
        return SimpleNamespace(stdout=big_out, stderr=big_err, returncode=0)

    monkeypatch.setattr(tt.subprocess, "run", _fake_run)
    out = tt._handle_terminal(
        {"command": "echo x", "cwd": str(tmp_path)},
        tool_call_id="call_h1", omnimate_home=str(tmp_path), config={},
    )
    data = json.loads(out)
    assert data["stdout_offloaded"] is True
    assert data["stderr_offloaded"] is True
    # stdout/stderr 字段是 offload JSON（preview + full_at 指针）
    stdout_obj = json.loads(data["stdout"])
    assert stdout_obj["truncated"] is True
    stdout_file = Path(stdout_obj["full_at"])
    assert stdout_file.exists()
    # 落盘的是**无损原文**——中段标记必须在（此前先截断再落盘，中段永久丢）
    assert "MIDDLE_MARKER" in stdout_file.read_text(encoding="utf-8")
    stderr_obj = json.loads(data["stderr"])
    assert Path(stderr_obj["full_at"]).exists()


def test_terminal_small_output_untouched_h1(tmp_path, monkeypatch):
    from tools import terminal_tool as tt

    def _fake_run(*a, **kw):
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(tt.subprocess, "run", _fake_run)
    out = tt._handle_terminal(
        {"command": "echo x", "cwd": str(tmp_path)},
        tool_call_id="call_h1b", omnimate_home=str(tmp_path), config={},
    )
    data = json.loads(out)
    assert data["stdout"] == "ok"
    assert data["stdout_offloaded"] is False


# ======================================================================
# read_file 去重
# ======================================================================

def test_read_dedup_and_invalidation_h2(tmp_path):
    from agent.workspace_context import workspace_cwd_context
    from tools import file_operations as fo

    with workspace_cwd_context(str(tmp_path)):
        f = tmp_path / "a.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        fo.reset_read_seen()

        r1 = json.loads(fo._handle_read_file({"path": str(f)}))
        assert r1.get("file_unchanged") is None
        assert "line1" in r1["content"]

        # 同 range 未变化 → stub（不重复送全文）
        r2 = json.loads(fo._handle_read_file({"path": str(f)}))
        assert r2.get("file_unchanged") is True
        assert "line1" not in r2["content"]
        assert r2["content_hash"] == r1["content_hash"]

        # 不同 range → 正常返回内容
        r3 = json.loads(fo._handle_read_file({"path": str(f), "offset": 0, "limit": 1}))
        assert r3.get("file_unchanged") is None

        # 写后失效 → 重新读回新内容
        fo._handle_write_file({"path": str(f), "content": "line1\nline2\nline3\n"})
        r4 = json.loads(fo._handle_read_file({"path": str(f)}))
        assert r4.get("file_unchanged") is None
        assert "line3" in r4["content"]


# ======================================================================
# search_files 分页/上下文/大小写
# ======================================================================

def test_search_pagination_context_case_h3(tmp_path):
    from tools import file_operations as fo

    d = tmp_path / "src"
    d.mkdir()
    (d / "f.txt").write_text(
        "\n".join(f"line-{i}-NEEDLE" for i in range(7)), encoding="utf-8",
    )

    # 第 1 页
    r1 = json.loads(fo._handle_search_files(
        {"pattern": "NEEDLE", "path": str(d), "max_matches": 3}))
    assert len(r1["matches"]) == 3
    assert r1["truncated"] is True
    assert "offset=3" in r1["pagination_hint"]

    # 第 2 页（offset 翻页，顺序稳定）
    r2 = json.loads(fo._handle_search_files(
        {"pattern": "NEEDLE", "path": str(d), "max_matches": 3, "offset": 3}))
    assert r2["matches"][0]["line"] == 4

    # 全量在一页内 → 无 truncated
    r3 = json.loads(fo._handle_search_files(
        {"pattern": "NEEDLE", "path": str(d), "max_matches": 10}))
    assert r3.get("truncated") is None
    assert r3["match_count"] == 7

    # 上下文 + 大小写不敏感
    (d / "g.txt").write_text("before\nFoo BAR\nafter\n", encoding="utf-8")
    r4 = json.loads(fo._handle_search_files(
        {"pattern": "foo bar", "path": str(d), "glob": "g.txt",
         "case_insensitive": True, "context": 1}))
    assert r4["match_count"] == 1
    m = r4["matches"][0]
    assert m["before"] == ["1: before"]
    assert m["after"] == ["3: after"]
