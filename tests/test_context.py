"""Context（prompt_builder + context_compressor）测试。"""

import pytest
from pathlib import Path

from agent.prompt_builder import (
    build_system_prompt, MEMORY_GUIDANCE, SKILLS_GUIDANCE,
    SESSION_SEARCH_GUIDANCE, TOOL_USAGE_GUIDANCE,
    _build_skill_index, _extract_description,
)
from agent.context_compressor import (
    _fix_tool_call_pairs, _summarize_conversation,
    _rule_based_summary, estimate_message_tokens,
)


# ---------------------------------------------------------------------------
# prompt_builder
# ---------------------------------------------------------------------------

def test_build_system_prompt_basic():
    """基础 system prompt 包含指导段。"""
    sp = build_system_prompt()
    assert MEMORY_GUIDANCE in sp
    assert SKILLS_GUIDANCE in sp


def test_build_system_prompt_no_guidance():
    """include_guidance=False 时不包含指导。"""
    sp = build_system_prompt(include_guidance=False)
    assert MEMORY_GUIDANCE not in sp


def test_build_system_prompt_with_memory(tmp_path):
    """CCAR10 Task 2：snapshot 从 system prompt 退役——不再含记忆索引段。

    检索改走 ephemeral 注入（_pending_ephemeral_messages），
    system prompt 永不含记忆索引（保护 prompt cache）。
    memory_store 参数仍保留签名不变（向后兼容）。
    """
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    # 添加一些测试记忆（使用 project 类型）
    store.save(
        name="test-memory",
        description="测试记忆条目",
        type="project",
        body="这是测试记忆的正文"
    )

    sp = build_system_prompt(memory_store=store)
    # CCAR10: 记忆索引段已退役——不再注入 system prompt
    assert "## 记忆索引" not in sp
    # 记忆描述也不应通过索引进入 system prompt
    assert "测试记忆条目" not in sp


def test_system_prompt_no_memory_index_section(tmp_path):
    """集成验证：带 memory_store 构建的 system prompt 无记忆索引（CCAR10 Task 2）。"""
    from agent.memory_store import MemoryStore
    from agent.prompt_builder import build_system_prompt

    ms = MemoryStore(omnimate_home=tmp_path)
    ms.save(name="n", description="d", type="user")
    sp = build_system_prompt(memory_store=ms, include_guidance=False)
    assert "记忆索引" not in sp


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
# _fix_tool_call_pairs（pipeline 复用）
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


async def test_summarize_no_client_uses_rules():
    """无客户端时走规则提取。"""
    msgs = [{"role": "user", "content": "hi"}]
    summary = await _summarize_conversation(msgs, llm_client=None)
    assert "规则提取" in summary or "hi" in summary


def test_estimate_tokens():
    msgs = [{"role": "user", "content": "a" * 30}]
    tokens = estimate_message_tokens(msgs)
    assert tokens == 10  # 30 / 3


def test_system_prompt_includes_current_cwd(tmp_path):
    """system prompt 必须注入当前工作目录（log.log 案例）。

    恢复历史会话后 LLM 顺着旧项目路径模仿填 cwd，跑去探索别的项目。
    修复：context 层注入当前目录 + "以当前目录为准"提示。
    """
    from agent.prompt_builder import build_system_prompt_layers
    from agent.workspace_context import workspace_cwd_context

    fake_cwd = r"D:\project\deer-flow-main"
    with workspace_cwd_context(fake_cwd):
        layers = build_system_prompt_layers(include_guidance=False)
    assert "当前工作目录" in layers.context
    assert fake_cwd in layers.context
    # "以当前目录为准"提示（防历史会话路径误导）
    assert "以当前目录为准" in layers.context


def test_system_prompt_cwd_defaults_to_process_cwd():
    """未设 workspace context 时 fallback 到 os.getcwd()。"""
    import os
    from agent.prompt_builder import build_system_prompt_layers

    layers = build_system_prompt_layers(include_guidance=False)
    assert os.getcwd() in layers.context
