"""三层系统提示缓存测试（05）。

验证：
1. build_system_prompt_layers 三层独立生成
2. build_system_prompt 旧接口返回等价字符串（render_flat）
3. stable 多次构建内容一致（byte-stable）
4. context 在会话内一致
5. volatile 每轮可变
6. AIAgent 缓存 stable + context
7. 压缩后 invalidate 只重建 context（stable 保留）
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ----------------------------------------------------------------------------
# build_system_prompt_layers 基础
# ----------------------------------------------------------------------------

def test_layers_returns_three_strings():
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers()
    assert isinstance(layers.stable, str)
    assert isinstance(layers.context, str)
    assert isinstance(layers.volatile, str)
    assert layers.stable  # 非空
    # context/volatile 在无 memory/skills 时可能为空


def test_stable_contains_identity_and_guidance():
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers(language="zh")
    assert "AI Agent" in layers.stable or "自学习" in layers.stable
    assert "记忆系统" in layers.stable  # MEMORY_GUIDANCE
    assert "技能系统" in layers.stable  # SKILLS_GUIDANCE


def test_language_changes_stable():
    from agent.prompt_builder import build_system_prompt_layers
    zh = build_system_prompt_layers(language="zh")
    en = build_system_prompt_layers(language="en")
    assert zh.stable != en.stable
    assert "Respond in English" in en.stable
    assert "中文回复" in zh.stable


# ----------------------------------------------------------------------------
# render_flat / 旧接口兼容
# ----------------------------------------------------------------------------

def test_render_flat_combines_layers():
    from agent.prompt_builder import SystemPromptLayers
    layers = SystemPromptLayers(
        stable="S", context="C", volatile="V",
    )
    assert layers.render_flat() == "S\n\nC\n\nV"


def test_render_flat_skips_empty():
    from agent.prompt_builder import SystemPromptLayers
    layers = SystemPromptLayers(stable="S", context="", volatile="V")
    assert layers.render_flat() == "S\n\nV"


def test_legacy_build_system_prompt_matches_flat():
    """旧 build_system_prompt() 返回的字符串 == layers.render_flat()。"""
    from agent.prompt_builder import build_system_prompt, build_system_prompt_layers
    kwargs = dict(language="zh")
    old = build_system_prompt(**kwargs)
    layers = build_system_prompt_layers(**kwargs)
    assert old == layers.render_flat()


# ----------------------------------------------------------------------------
# byte-stable：多次构建 stable 一致
# ----------------------------------------------------------------------------

def test_stable_byte_stable():
    """同一组参数构建两次，stable 字节级一致。"""
    from agent.prompt_builder import build_system_prompt_layers
    a = build_system_prompt_layers(language="zh")
    b = build_system_prompt_layers(language="zh")
    assert a.stable == b.stable


# ----------------------------------------------------------------------------
# volatile 动态变化
# ----------------------------------------------------------------------------

def test_volatile_changes_with_todo():
    """volatile 受 todo_state 参数影响。"""
    from agent.prompt_builder import build_system_prompt_layers
    no_todo = build_system_prompt_layers()
    with_todo = build_system_prompt_layers(todo_state="step1,step2")
    assert no_todo.volatile != with_todo.volatile
    assert "step1" in with_todo.volatile
    assert "<todo_state>" in with_todo.volatile


def test_volatile_changes_with_reminder():
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers(reminder="<todo_reminder>3 轮未更新</todo_reminder>")
    assert "3 轮未更新" in layers.volatile


def test_stable_unchanged_when_volatile_changes():
    """volatile 变化不影响 stable（保证 prompt cache 命中）。"""
    from agent.prompt_builder import build_system_prompt_layers
    a = build_system_prompt_layers(todo_state="x")
    b = build_system_prompt_layers(todo_state="y", reminder="r")
    assert a.stable == b.stable


# ----------------------------------------------------------------------------
# context 层内容
# ----------------------------------------------------------------------------

def test_context_includes_memory_index(tmp_path: Path):
    """memory_store.snapshot_for_prompt() 返回的内容进 context 层。"""
    from agent.prompt_builder import build_system_prompt_layers
    fake_store = MagicMock()
    fake_store.snapshot_for_prompt.return_value = (
        "- [测试专用记忆](.memory/abc123.md) — 独一无二的内容XYZ789"
    )
    layers = build_system_prompt_layers(memory_store=fake_store)
    assert "独一无二的内容XYZ789" in layers.context
    assert "记忆索引" in layers.context
    # stable 不含这条独特内容（确保分层）
    assert "独一无二的内容XYZ789" not in layers.stable


def test_context_includes_skills(tmp_path: Path):
    """技能索引在 context 层（不在 stable）。"""
    from agent.prompt_builder import build_system_prompt_layers
    # 给一个空 skills_dir（无技能 → context 不含技能）
    empty = tmp_path / "skills"
    empty.mkdir()
    layers = build_system_prompt_layers(skills_dir=empty)
    # 空技能目录 → context 不含 "可用技能" 块
    assert "## 可用技能" not in layers.context


# ----------------------------------------------------------------------------
# AIAgent 集成
# ----------------------------------------------------------------------------

def _build_minimal_agent(**overrides):
    from agent import AIAgent
    return AIAgent(
        base_url="x", api_key="x", model="test",
        enabled_toolsets=[],
        system_prompt_override="test",  # 跳过真实构建
        **overrides,
    )


def test_agent_caches_stable_and_context():
    """AIAgent 缓存 stable + context；多次 _get_system_prompt 不重建。"""
    from agent import AIAgent
    agent = AIAgent(
        base_url="x", api_key="x", model="test",
        enabled_toolsets=[],
    )
    first = agent._get_system_prompt()
    # 抓 stable/context
    stable1 = agent._stable_prompt
    context1 = agent._context_prompt
    second = agent._get_system_prompt()
    # 不重建
    assert first == second
    assert agent._stable_prompt == stable1
    assert agent._context_prompt == context1


def test_invalidate_preserves_stable():
    """invalidate 后 stable 保留，context 重建。"""
    from agent import AIAgent
    agent = AIAgent(
        base_url="x", api_key="x", model="test",
        enabled_toolsets=[],
    )
    agent._get_system_prompt()
    stable_before = agent._stable_prompt
    # invalidate
    agent.invalidate_system_prompt()
    # stable 仍保留
    assert agent._stable_prompt == stable_before
    # context 已清空
    assert agent._context_prompt is None
    # 重新构建
    agent._get_system_prompt()
    assert agent._stable_prompt == stable_before  # 没变


def test_volatile_prompt_method_exists():
    """_get_volatile_prompt 方法可用。"""
    agent = _build_minimal_agent()
    # 无 todo_manager 或 should_remind=False 时返回空
    result = agent._get_volatile_prompt()
    assert isinstance(result, str)
