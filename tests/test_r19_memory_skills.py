"""R19 记忆技能专项测试。

#22 curator 整理增强（delete_falsified / normalize_dates）
#24 秘密扫描扩展（公共 scanner + memory/curator/trace/handoff 链路）
#25 条件技能 paths
#28 skillify 内置技能
#21 每轮自动记忆提取
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.memory_curator import execute_action, parse_yaml_actions
from agent.secret_scanner import (
    SECRET_RULES_RE,
    find_secrets_in,
    redact_fields,
    scan_text,
)


# ---------------------------------------------------------------------------
# R19 #22：curator 整理增强
# ---------------------------------------------------------------------------

class _FakeStore:
    def __init__(self):
        self.deleted = []
        self.updated = {}
        self.entries = {}

    def delete(self, mid):
        self.deleted.append(mid)

    def get(self, mid):
        return self.entries.get(mid)

    def update(self, mid, body=None, **kw):
        self.updated[mid] = body


def test_parse_yaml_new_actions():
    raw = """```yaml
- action: delete_falsified
  archive: "proj#abc"
  evidence: 环境已迁移到 D 盘
- action: normalize_dates
  update_id: "user#xyz"
  new_body: |
    2026-08-10 完成了迁移
```"""
    actions = parse_yaml_actions(raw)
    assert [a["action"] for a in actions] == ["delete_falsified", "normalize_dates"]


def test_execute_delete_falsified(tmp_path):
    store = _FakeStore()
    result = execute_action(
        {"action": "delete_falsified", "archive": "proj#abc", "evidence": "被证伪"},
        store, tmp_path,
    )
    assert "delete_falsified" in result
    assert store.deleted == ["proj#abc"]


def test_execute_normalize_dates(tmp_path):
    store = _FakeStore()
    store.entries["user#xyz"] = SimpleNamespace(
        id="user#xyz", name="n", description="d",
        updated_at=__import__("datetime").datetime(2026, 8, 16),
        body="上周完成了部署",
    )
    result = execute_action(
        {"action": "normalize_dates", "update_id": "user#xyz",
         "new_body": "2026-08-10 完成了部署"},
        store, tmp_path,
    )
    assert "normalize_dates" in result
    assert store.updated["user#xyz"] == "2026-08-10 完成了部署"
    # 原文有备份（safe_rewrite_body 备份目录存在）
    assert list(tmp_path.glob("memory-rewrites-*"))


def test_execute_unknown_action_still_skipped(tmp_path):
    store = _FakeStore()
    result = execute_action({"action": "bogus"}, store, tmp_path)
    assert "skip" in result


# ---------------------------------------------------------------------------
# R19 #24：秘密扫描
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,rule", [
    ("sk-1234567890abcdefghijklmnop", "openai"),
    ("sk-ant-api03-1234567890abcdefghij", "anthropic"),
    ("ghp_" + "a" * 36, "github_pat"),
    ("AKIA" + "B" * 16, "aws_access_token"),
    ("AIza" + "c" * 35, "google_api_key"),
    ("xoxb-" + "1234567890abcd", "slack_token"),
    ("Bearer abcdefghijklmnopqrstuvwxyz123", "bearer"),
    ("api_key = \"abcdefghijklmnopqrst\"", "api_key"),
    ("-----BEGIN RSA PRIVATE KEY-----", "pem"),
    ("eyJ" + "a" * 24 + "." + "b" * 24, "jwt"),
])
def test_scan_text_rules(text, rule):
    hits = scan_text(text)
    assert hits and hits[0]["rule"] == rule, f"{text[:20]}... → {hits}"


def test_scan_text_clean():
    assert scan_text("普通文本，没有秘密。path/to/file.py") == []
    assert scan_text("") == []


def test_snippet_truncated():
    long_key = "sk-" + "x" * 100
    hits = scan_text(long_key)
    assert len(hits[0]["snippet"]) <= 50


def test_memory_store_rejects_secrets(tmp_path):
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    with pytest.raises(ValueError, match="密钥"):
        store.save(
            name="泄漏", description="含密钥",
            type="other", body="key: sk-" + "a" * 30,
        )
    # update 同样拒绝
    mid = store.save(name="正常", description="d", type="other", body="ok")
    with pytest.raises(ValueError, match="密钥"):
        store.update(mid, body="AKIA" + "B" * 16)


def test_curator_rewrite_rejected_on_secret(tmp_path):
    from agent.memory_curator import safe_rewrite_body
    store = _FakeStore()
    store.entries["t#1"] = SimpleNamespace(
        id="t#1", name="n", description="d",
        updated_at=__import__("datetime").datetime.now(),
        body="原文",
    )
    # new_body 含密钥 → 拒绝改写（返回 None，原文不动）
    result = safe_rewrite_body(store, "t#1", "sk-" + "a" * 30, tmp_path)
    assert result is None
    assert "t#1" not in store.updated


def test_trace_redact(tmp_path):
    from agent.trace import TraceSink
    sink = TraceSink(tmp_path)
    sink.emit("post_tool_use", command="export TOKEN=sk-" + "a" * 30, ok=True)
    # 写盘路径 <base_dir>/.trace/<date>.jsonl
    import datetime as _dt
    day = _dt.datetime.now().strftime("%Y-%m-%d")
    rec = json.loads(
        (tmp_path / ".trace" / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "[REDACTED:openai]" in rec["command"]
    assert "sk-aaaa" not in rec["command"]
    assert rec["ok"] is True  # 非字符串值不动


def test_handoff_scan_uses_common_scanner():
    from agent.handoff import _scan_for_secrets
    transcript = [
        {"role": "user", "content": "ghp_" + "a" * 36},
        {"role": "assistant", "content": "好的"},
    ]
    matches = _scan_for_secrets(transcript)
    assert len(matches) == 1
    assert matches[0]["pattern"] == "<github_pat>"
    assert matches[0]["message_index"] == 0


def test_redact_fields():
    out = redact_fields({
        "text": "token: abcdefghijklmnopqr",
        "count": 5,
    })
    assert out["count"] == 5
    assert "[REDACTED:token]" in out["text"]
