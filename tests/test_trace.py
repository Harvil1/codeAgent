"""TraceSink 本地 trace sink 测试。

CCAR8 Task 4。验证：
- 每天 jsonl 文件
- fail-open 写盘
- query / summary 聚合
- agent_id 覆盖
"""
import json
import logging
from pathlib import Path

from agent.trace import TraceSink


def test_emit_writes_jsonl(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100, model="deepseek-chat")
    files = list((tmp_path / ".trace").glob("*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["event"] == "pre_llm_call"
    assert record["input_tokens"] == 100
    assert record["model"] == "deepseek-chat"
    assert record["agent_id"] == "main"
    assert "ts" in record


def test_emit_creates_daily_file(tmp_path: Path):
    """每天一个 jsonl 文件。"""
    import datetime
    sink = TraceSink(tmp_path)
    sink.emit("event_a")
    sink.emit("event_b")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    assert (tmp_path / ".trace" / f"{date_str}.jsonl").exists()
    lines = (tmp_path / ".trace" / f"{date_str}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_emit_with_agent_id_override(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("evt", agent_id="subagent_xyz")
    record = json.loads(
        list((tmp_path / ".trace").glob("*.jsonl"))[0].read_text(encoding="utf-8")
    )
    assert record["agent_id"] == "subagent_xyz"


def test_emit_failopen_on_disk_error(tmp_path: Path, caplog):
    """写盘失败不抛，只 log warning。"""
    sink = TraceSink(tmp_path)
    # 模拟写盘失败：把 _trace_dir 改成不可写的路径
    sink._trace_dir = "/nonexistent/path/that/does/not/exist"
    with caplog.at_level(logging.WARNING):
        sink.emit("event_x")  # 不抛
    assert any("trace emit fail-open" in r.message for r in caplog.records)


def test_query_filters(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100)
    sink.emit("post_tool_use", tool="read_file")
    sink.emit("post_tool_use", tool="write_file")
    results = sink.query(event="post_tool_use")
    assert len(results) == 2
    results = sink.query(event="pre_llm_call")
    assert len(results) == 1


def test_summary_aggregates(tmp_path: Path):
    sink = TraceSink(tmp_path)
    sink.emit("pre_llm_call", input_tokens=100)
    sink.emit("pre_llm_call", input_tokens=200)
    sink.emit("post_llm_call", output_tokens=50)
    sink.emit("tool_failed", tool="x", error="boom")
    summary = sink.summary()
    assert summary["total_events"] == 4
    assert summary["by_event"]["pre_llm_call"] == 2
    assert summary["by_event"]["post_llm_call"] == 1
    assert summary["by_event"]["tool_failed"] == 1
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 50
    assert summary["error_count"] == 1


def test_query_limit(tmp_path: Path):
    sink = TraceSink(tmp_path)
    for i in range(10):
        sink.emit("event_x", idx=i)
    results = sink.query(limit=3)
    assert len(results) == 3
