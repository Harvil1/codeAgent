"""Context（prompt_builder + context_compressor）测试。"""

import pytest
from pathlib import Path

from agent.prompt_builder import (
    build_system_prompt, MEMORY_GUIDANCE, SKILLS_GUIDANCE,
    SESSION_SEARCH_GUIDANCE, TOOL_USAGE_GUIDANCE, IDENTITY,
    _build_skill_index, _extract_description,
)
from agent.context_compressor import (
    maybe_compress, _fix_tool_call_pairs, _summarize_conversation,
    _rule_based_summary, estimate_message_tokens,
    MESSAGES_BEFORE_COMPRESS, KEEP_RECENT_MESSAGES,
)


# ---------------------------------------------------------------------------
# prompt_builder
# ---------------------------------------------------------------------------

def test_build_system_prompt_basic():
    """基础 system prompt 包含身份和指导。"""
    sp = build_system_prompt()
    assert "AI Agent" in sp
    assert MEMORY_GUIDANCE in sp
    assert SKILLS_GUIDANCE in sp


def test_build_system_prompt_no_guidance():
    """include_guidance=False 时不包含指导。"""
    sp = build_system_prompt(include_guidance=False)
    assert MEMORY_GUIDANCE not in sp


def test_build_system_prompt_with_memory(tmp_path):
    """包含记忆快照。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(tmp_path)
    store.add("memory", "测试记忆条目")

    sp = build_system_prompt(memory_store=store)
    assert "测试记忆条目" in sp


def test_build_system_prompt_with_skills(tmp_path):
    """包含技能索引。"""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "my-skill").mkdir()
    (skills / "my-skill" / "SKILL.md").write_text(
        '---\nname: my-skill\ndescription: "测试技能"\n---\n# Body',
        encoding="utf-8",
    )

    sp = build_system_prompt(skills_dir=skills)
    assert "my-skill" in sp
    assert "测试技能" in sp


def test_build_system_prompt_with_context_file(tmp_path):
    """包含上下文文件。"""
    cf = tmp_path / "AGENTS.md"
    cf.write_text("# 项目说明\n这是测试项目", encoding="utf-8")

    sp = build_system_prompt(context_files=[cf])
    assert "项目说明" in sp
    assert "测试项目" in sp


def test_build_system_prompt_byte_stable(tmp_path):
    """同一输入产生相同输出（byte-stable）。"""
    sp1 = build_system_prompt()
    sp2 = build_system_prompt()
    assert sp1 == sp2


# ---------------------------------------------------------------------------
# _build_skill_index
# ---------------------------------------------------------------------------

def test_skill_index_lists_skills(tmp_path):
    skills = tmp_path
    (skills / "a").mkdir()
    (skills / "a" / "SKILL.md").write_text(
        '---\ndescription: "技能A"\n---\n', encoding="utf-8",
    )
    (skills / "b").mkdir()
    (skills / "b" / "SKILL.md").write_text(
        '---\ndescription: "技能B"\n---\n', encoding="utf-8",
    )

    index = _build_skill_index(skills)
    assert "/a" in index
    assert "/b" in index
    assert "技能A" in index


def test_skill_index_skips_archived(tmp_path):
    """归档技能不出现在索引。"""
    import json
    skills = tmp_path
    (skills / "active").mkdir()
    (skills / "active" / "SKILL.md").write_text(
        '---\ndescription: "active"\n---\n', encoding="utf-8",
    )
    (skills / "archived").mkdir()
    (skills / "archived" / "SKILL.md").write_text(
        '---\ndescription: "archived"\n---\n', encoding="utf-8",
    )
    (skills / ".usage.json").write_text(
        json.dumps({"archived": {"state": "archived"}}),
        encoding="utf-8",
    )

    index = _build_skill_index(skills)
    assert "/active" in index
    assert "/archived" not in index


def test_skill_index_empty(tmp_path):
    """空目录返回空字符串。"""
    assert _build_skill_index(tmp_path) == ""


# ---------------------------------------------------------------------------
# _extract_description
# ---------------------------------------------------------------------------

def test_extract_description():
    content = '---\nname: x\ndescription: "hello"\n---\nbody'
    assert _extract_description(content) == "hello"


def test_extract_description_no_frontmatter():
    assert _extract_description("no frontmatter") == ""


def test_extract_description_single_quotes():
    content = "---\ndescription: 'hi'\n---\n"
    assert _extract_description(content) == "hi"


# ---------------------------------------------------------------------------
# maybe_compress
# ---------------------------------------------------------------------------

def _make_messages(count, include_system=True):
    """生成测试消息列表。"""
    msgs = []
    if include_system:
        msgs.append({"role": "system", "content": "system prompt"})
    for i in range(count):
        msgs.append({"role": "user", "content": f"消息 {i}"})
        msgs.append({"role": "assistant", "content": f"回复 {i}"})
    return msgs


def test_maybe_compress_not_enough_messages():
    """消息数不足时不压缩。"""
    msgs = _make_messages(5)
    result, compressed = maybe_compress(msgs, attempt_count=0)
    assert compressed is False
    assert result == msgs


def test_maybe_compress_max_attempts():
    """压缩次数用完时不压缩。"""
    msgs = _make_messages(30)
    result, compressed = maybe_compress(msgs, attempt_count=3)
    assert compressed is False


def test_maybe_compress_triggers_with_llm():
    """达到阈值 + 有 LLM 客户端时压缩。"""
    from types import SimpleNamespace

    # 构造足够的消息
    msgs = _make_messages(25)

    # SimpleNamespace 构造 mock 客户端
    def fake_create(**kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="这是总结")
            )]
        )

    client = SimpleNamespace(chat_completions=fake_create)

    result, compressed = maybe_compress(
        msgs, attempt_count=0, llm_client=client,
    )
    assert compressed is True
    assert len(result) < len(msgs)
    # system 保留
    assert result[0]["role"] == "system"
    # 包含总结消息
    assert any("总结" in m.get("content", "") for m in result)


def test_maybe_compress_fallback_rule_based():
    """无 LLM 客户端时降级到规则提取。"""
    msgs = _make_messages(25)
    result, compressed = maybe_compress(msgs, attempt_count=0, llm_client=None)
    assert compressed is True
    # 规则提取保留 user 消息
    assert any("规则提取" in m.get("content", "") for m in result)


# ---------------------------------------------------------------------------
# _fix_tool_call_pairs
# ---------------------------------------------------------------------------

def test_fix_pairs_no_issues():
    """完整配对的消息不变。"""
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "terminal"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    fixed = _fix_tool_call_pairs(msgs)
    assert len(fixed) == 2  # 无补充


def test_fix_pairs_missing_result():
    """有 tool_call 无结果时补充假结果。"""
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "terminal"}},
        ]},
    ]
    fixed = _fix_tool_call_pairs(msgs)
    assert len(fixed) == 2
    assert fixed[1]["role"] == "tool"
    assert fixed[1]["tool_call_id"] == "c1"


def test_fix_pairs_multiple_missing():
    """多个未配对的 tool_call 都被补充。"""
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "a"}},
            {"id": "c2", "function": {"name": "b"}},
        ]},
    ]
    fixed = _fix_tool_call_pairs(msgs)
    assert len(fixed) == 3  # 1 assistant + 2 补充


# ---------------------------------------------------------------------------
# _summarize_conversation / _rule_based_summary
# ---------------------------------------------------------------------------

def test_rule_based_summary():
    msgs = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]
    summary = _rule_based_summary(msgs)
    assert "第一问" in summary
    assert "第二问" in summary
    assert "规则提取" in summary


def test_summarize_no_client_uses_rules():
    """无客户端时走规则提取。"""
    msgs = [{"role": "user", "content": "hi"}]
    summary = _summarize_conversation(msgs, llm_client=None)
    assert "规则提取" in summary or "hi" in summary


def test_estimate_tokens():
    msgs = [{"role": "user", "content": "a" * 30}]
    tokens = estimate_message_tokens(msgs)
    assert tokens == 10  # 30 / 3
