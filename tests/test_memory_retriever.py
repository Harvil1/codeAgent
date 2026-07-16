"""memory_retriever 测试。"""
import json
from unittest.mock import MagicMock

import pytest

from agent.memory_retriever import retrieve_relevant


def _mock_llm(returned_text: str):
    m = MagicMock()
    m.chat_completions.return_value.choices = [
        MagicMock(message=MagicMock(content=returned_text))
    ]
    return m


def test_retrieve_relevant_returns_ids():
    llm = _mock_llm('["id1", "id2"]')
    result = retrieve_relevant(
        query="如何配置 pytest",
        index_text="- [pytest](.memory/id1.md) — pytest 配置\n- [git](.memory/id2.md) — git 用法",
        llm_client=llm, model="test-model",
    )
    assert result == ["id1", "id2"]


def test_retrieve_relevant_handles_empty_index():
    llm = _mock_llm('[]')
    result = retrieve_relevant(
        query="x", index_text="",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_handles_llm_failure_returns_empty():
    """LLM 抛异常 → 返回空 list（fail-open）。"""
    llm = MagicMock()
    llm.chat_completions.side_effect = RuntimeError("API down")
    result = retrieve_relevant(
        query="x", index_text="some index",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_handles_malformed_json_returns_empty():
    llm = _mock_llm("not a json")
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_respects_max_results():
    llm = _mock_llm('["a", "b", "c", "d", "e", "f", "g"]')
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m", max_results=3,
    )
    assert len(result) == 3


def test_retrieve_relevant_handles_non_list_json():
    llm = _mock_llm('{"not": "a list"}')
    result = retrieve_relevant(
        query="x", index_text="some",
        llm_client=llm, model="m",
    )
    assert result == []


def test_retrieve_relevant_uses_correct_prompt():
    """验证 retriever 发给 LLM 的 prompt 含 query 和 index。"""
    from unittest.mock import MagicMock
    from agent.memory_retriever import retrieve_relevant

    captured_msgs = []
    def capture(msgs, **kw):
        captured_msgs.append(msgs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="[]"))]
        return resp

    llm = MagicMock()
    llm.chat_completions.side_effect = capture

    retrieve_relevant(
        query="如何配置 pytest",
        index_text="- [pytest](.memory/x.md) — pytest 配置",
        llm_client=llm, model="m",
    )
    assert len(captured_msgs) == 1
    prompt_text = captured_msgs[0][0]["content"]
    assert "pytest" in prompt_text  # query 关键词
    assert "<index>" in prompt_text or "pytest 配置" in prompt_text  # index 内容
