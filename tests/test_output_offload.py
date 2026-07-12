"""output_offload 模块测试。"""
import json
from pathlib import Path

import pytest

from agent.output_offload import maybe_offload


def test_under_threshold_returns_content_unchanged(tmp_path: Path):
    """小于阈值时原样返回（不是 JSON）。"""
    content = "short result"
    result = maybe_offload(
        content, tool_call_id="call_abc", agent_home=tmp_path,
        threshold=30000, preview_chars=2000,
    )
    assert result == content


def test_over_threshold_writes_file_and_returns_json(tmp_path: Path):
    """大于阈值时落盘并返回 JSON 指针。"""
    content = "x" * 50000
    result = maybe_offload(
        content, tool_call_id="call_abc", agent_home=tmp_path,
        threshold=30000, preview_chars=2000,
    )
    parsed = json.loads(result)
    assert parsed["truncated"] is True
    assert parsed["orig_chars"] == 50000
    assert len(parsed["preview"]) == 2000
    assert "full_at" in parsed
    assert "hint" in parsed

    offload_path = Path(parsed["full_at"])
    assert offload_path.exists()
    assert offload_path.read_text(encoding="utf-8") == content
    assert "call_abc" in offload_path.name


def test_offload_path_under_agent_home(tmp_path: Path):
    """落盘路径必须严格在 agent_home/.task_outputs/tool-results/ 下。"""
    content = "x" * 50000
    result = maybe_offload(content, tool_call_id="call_xyz", agent_home=tmp_path)
    parsed = json.loads(result)
    offload_path = Path(parsed["full_at"])
    assert offload_path.is_relative_to(tmp_path / ".task_outputs" / "tool-results")


def test_duplicate_tool_call_id_appends_counter(tmp_path: Path):
    """相同 tool_call_id 第二次落盘不覆盖，追加 _N。"""
    content1 = "x" * 50000
    maybe_offload(content1, tool_call_id="call_dup", agent_home=tmp_path)
    content2 = "y" * 50000
    result = maybe_offload(content2, tool_call_id="call_dup", agent_home=tmp_path)
    parsed = json.loads(result)
    assert Path(parsed["full_at"]).read_text(encoding="utf-8") == content2
    assert (tmp_path / ".task_outputs" / "tool-results" / "call_dup.txt").exists()


def test_disk_full_falls_back_to_truncated_content(tmp_path: Path, monkeypatch):
    """写入失败时不抛异常，降级为截断 + error 标注。"""
    def raise_oserror(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("agent.output_offload._write_atomically", raise_oserror)

    content = "x" * 50000
    result = maybe_offload(content, tool_call_id="call_err", agent_home=tmp_path)
    parsed = json.loads(result)
    assert parsed["error_type"] == "offload_io_error"
    assert "truncated_content" in parsed
    assert len(parsed["truncated_content"]) == 30000


def test_non_string_content_passthrough(tmp_path: Path):
    """非字符串 content（如 None / dict）原样返回，不尝试落盘。"""
    assert maybe_offload(None, tool_call_id="x", agent_home=tmp_path) is None
