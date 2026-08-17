"""transcript 模块测试：压缩前快照存档。"""
import json
from pathlib import Path

from agent.transcript import snapshot_if_needed


SAMPLE_MESSAGES = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "tool_calls": [
        {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "result"},
]


def test_force_writes_jsonl(tmp_path: Path):
    """force=True 必落盘，返回路径。"""
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="sess_x",
        force=True,
    )
    assert path is not None
    assert path.exists()
    assert path.suffix == ".jsonl"
    # 每行是合法 JSON
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["role"] == "system"
    assert parsed[-1].get("_meta", {}).get("session_id") == "sess_x"
    assert parsed[-1]["_meta"]["orig_len"] == 4


def test_disabled_returns_none(tmp_path: Path):
    """enabled=False 直接返回 None。"""
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="sess_x",
        force=True, enabled=False,
    )
    assert path is None


def test_retention_prunes_oldest(tmp_path: Path):
    """超出 retention 时删最旧。"""
    for i in range(5):
        snapshot_if_needed(
            SAMPLE_MESSAGES, agent_home=tmp_path, session_id=f"sess_{i}",
            force=True, retention=3,
        )
    files = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(files) == 3  # 只保留最近 3 个


def test_filename_contains_timestamp_and_uuid(tmp_path: Path):
    """文件名格式：transcript_{YYYYMMDD_HHMMSS}_{shortuuid}.jsonl"""
    import re
    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s",
        force=True,
    )
    pattern = re.compile(r"transcript_\d{8}_\d{6}_[a-f0-9]{4}\.jsonl")
    assert pattern.match(path.name), f"文件名格式不对: {path.name}"


def test_creates_latest_pointer(tmp_path: Path):
    """同时维护 latest 指针（Windows 降级为文本文件）。"""
    snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s", force=True,
    )
    latest = tmp_path / ".transcripts" / "latest.txt"
    assert latest.exists()
    content = latest.read_text(encoding="utf-8")
    assert "transcript_" in content


def test_write_failure_returns_none_and_logs(tmp_path: Path, monkeypatch):
    """写入失败时不抛，返回 None。"""
    def raise_oserror(*args, **kwargs):
        raise OSError("permission denied")
    monkeypatch.setattr("agent.atomic_io.atomic_write_text_lite", raise_oserror)

    path = snapshot_if_needed(
        SAMPLE_MESSAGES, agent_home=tmp_path, session_id="s", force=True,
    )
    assert path is None
