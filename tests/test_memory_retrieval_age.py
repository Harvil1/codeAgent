"""T4 记忆检索年龄衰减（核心机制对齐第 4 项）。

- 检索索引行附 [age: Nd] 年龄标注（存储格式不变，MEMORY.md 落盘不标注）
- retrieve prompt 加"同等相关优先新记忆、冲突时新记忆优先"规则
- 链接查不到对应条目 → 标 unknown，不炸（fail-open）
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_topic(home: Path, topic: str, rows: list) -> None:
    d = home / ".memory"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{topic}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_store(tmp_path):
    from agent.memory_store import MemoryStore
    return MemoryStore(omnimate_home=tmp_path)


# ---------------------------------------------------------------------------
# prompt 规则
# ---------------------------------------------------------------------------

def test_prompt_template_has_age_rule():
    """检索 prompt 必须含年龄标注说明 + 新记忆优先规则。"""
    from agent.memory_retriever import RETRIEVAL_PROMPT_TEMPLATE
    assert "[age:" in RETRIEVAL_PROMPT_TEMPLATE
    assert "优先" in RETRIEVAL_PROMPT_TEMPLATE
    # 规则要素：同等优先新 / 冲突新胜出 / 旧作背景
    assert "新记忆" in RETRIEVAL_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# 索引年龄标注
# ---------------------------------------------------------------------------

def test_index_annotated_with_age(tmp_path):
    """180 天前的记忆 → 索引行附 [age: 1xxd]；MEMORY.md 落盘不带标注。"""
    old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat(timespec="seconds")
    _write_topic(tmp_path, "general", [{
        "id": "aaa", "name": "框架版本", "description": "用户在用 React 16",
        "type": "user", "body": "React 16", "created_at": old, "updated_at": old,
    }])
    store = _make_store(tmp_path)
    text = store.full_index_text_with_age()
    assert "[age: 1" in text  # 179~181d
    # 存储格式不变：落盘 MEMORY.md 不含标注
    md = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "[age:" not in md


def test_fresh_entry_age_small(tmp_path):
    """刚写入的记忆 → [age: 0d]。"""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_topic(tmp_path, "general", [{
        "id": "bbb", "name": "新记忆", "description": "升到 React 19 了",
        "type": "user", "body": "", "created_at": now, "updated_at": now,
    }])
    store = _make_store(tmp_path)
    text = store.full_index_text_with_age()
    assert "[age: 0d]" in text


def test_annotate_unknown_link_not_crash():
    """纯函数：链接查不到 → 标 unknown；无链接行原样返回。"""
    from agent.memory_retriever import annotate_index_with_age
    index = (
        "## 全局记忆\n"
        "### 主题：general\n"
        "- [框架版本](.memory/general.jsonl#aaa) — 用户在用 React 16\n"
        "- [未知条目](.memory/general.jsonl#zzz) — 链接查不到\n"
    )
    out = annotate_index_with_age(index, {".memory/general.jsonl#aaa": 180})
    assert "[age: 180d]" in out
    assert "[age: unknown]" in out
    # 无链接的标题行不受影响
    assert "### 主题：general" in out


def test_annotate_naive_age_none_is_unknown():
    """age 映射值为 None（时间戳不可解析）→ unknown。"""
    from agent.memory_retriever import annotate_index_with_age
    index = "- [x](.memory/t.jsonl#1) — d\n"
    out = annotate_index_with_age(index, {".memory/t.jsonl#1": None})
    assert "[age: unknown]" in out


# ---------------------------------------------------------------------------
# 注入链路：annotated index 进 prompt
# ---------------------------------------------------------------------------

class _FakeClient:
    """记录 prompt 的假 LLM client。"""

    def __init__(self):
        self.captured = None

    async def chat_completions(self, messages, model=None, **kw):
        self.captured = messages[0]["content"]

        class _Msg:
            content = "[]"

        class _Resp:
            choices = [_Msg()]
        return _Resp()


@pytest.mark.asyncio
async def test_retrieve_prompt_contains_annotation(tmp_path):
    """retrieve_relevant 收到的 prompt 含年龄标注行。"""
    from agent.memory_retriever import retrieve_relevant

    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
    _write_topic(tmp_path, "general", [{
        "id": "ccc", "name": "偏好", "description": "深色主题",
        "type": "user", "body": "", "created_at": old, "updated_at": old,
    }])
    store = _make_store(tmp_path)
    client = _FakeClient()
    await retrieve_relevant(
        query="主题偏好", index_text=store.full_index_text_with_age(),
        llm_client=client, model="test",
    )
    assert client.captured is not None
    assert "[age:" in client.captured


def test_injection_and_recall_use_annotated_index():
    """两个调用点必须切换到 full_index_text_with_age（源码级检查防漏改）。"""
    import inspect
    import agent.memory_injection as mi
    import tools.memory_recall_tool as mrt

    assert "full_index_text_with_age" in inspect.getsource(mi)
    assert "full_index_text_with_age" in inspect.getsource(mrt)
