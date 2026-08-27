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
    layers = build_system_prompt_layers()
    assert "记忆系统" in layers.stable  # MEMORY_GUIDANCE
    assert "技能系统" in layers.stable  # SKILLS_GUIDANCE


def test_stable_includes_open_exploration_todo_hint():
    """TODO_GUIDANCE 必须提示开放式探索任务也要先列清单。

    背景：handoff 问题 2——agent 接到'学习这个项目'时完全不用 todo_write，
    因为现有示例只有具体步骤型（创建/删除文件 5 步）。
    强化后：开放式探索也要先列清单（如后端/前端/skills/部署）。
    """
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers()
    # 关键词至少一个出现：表明 prompt 提醒了"开放式任务也要 todo"
    keywords = ["开放式", "探索", "学习这个项目", "学习项目"]
    assert any(k in layers.stable for k in keywords), (
        f"TODO_GUIDANCE 缺开放式探索提示（所有关键词都不在 stable：{keywords}）"
    )


def test_stable_includes_delegate_guidance():
    """stable 层必须有 subagent 使用指南。

    背景：handoff 问题 3——agent 自己 read_file 几十轮不用 subagent，
    因为 prompt 里完全没有子代理触发条件。
    """
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers()
    # 子代理工具名 + 触发条件关键词（大项目/并行/子代理）
    assert "subagent" in layers.stable or "委托" in layers.stable, (
        "stable 缺 subagent 指南"
    )
    trigger_keywords = ["并行", "大项目", "子代理", "委托"]
    assert any(k in layers.stable for k in trigger_keywords), (
        f"DELEGATE_GUIDANCE 缺触发条件关键词：{trigger_keywords}"
    )


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
    old = build_system_prompt()
    layers = build_system_prompt_layers()
    assert old == layers.render_flat()


# ----------------------------------------------------------------------------
# byte-stable：多次构建 stable 一致
# ----------------------------------------------------------------------------

def test_stable_byte_stable():
    """同一组参数构建两次，stable 字节级一致。"""
    from agent.prompt_builder import build_system_prompt_layers
    a = build_system_prompt_layers()
    b = build_system_prompt_layers()
    assert a.stable == b.stable


# ----------------------------------------------------------------------------
# volatile 动态变化
# ----------------------------------------------------------------------------

def test_volatile_changes_with_task_state():
    """volatile 受 task_state 参数影响。"""
    from agent.prompt_builder import build_system_prompt_layers
    no_task = build_system_prompt_layers()
    with_task = build_system_prompt_layers(task_state="T1:pending,T2:in_progress")
    assert no_task.volatile != with_task.volatile
    assert "T1" in with_task.volatile
    assert "<current_tasks>" in with_task.volatile


def test_volatile_changes_with_reminder():
    from agent.prompt_builder import build_system_prompt_layers
    layers = build_system_prompt_layers(reminder="<todo_reminder>3 轮未更新</todo_reminder>")
    assert "3 轮未更新" in layers.volatile


def test_stable_unchanged_when_volatile_changes():
    """volatile 变化不影响 stable（保证 prompt cache 命中）。"""
    from agent.prompt_builder import build_system_prompt_layers
    a = build_system_prompt_layers(task_state="x")
    b = build_system_prompt_layers(task_state="y", reminder="r")
    assert a.stable == b.stable


# ----------------------------------------------------------------------------
# context 层内容
# ----------------------------------------------------------------------------

def test_context_no_longer_includes_memory_index(tmp_path: Path):
    """memory_store 的 snapshot 不进 context 层。

    snapshot 改走 ephemeral 注入（_pending_ephemeral_messages），
    system prompt（含 context 层）永不含记忆索引（保护 prompt cache）。
    memory_store 参数保留签名向后兼容，但不再注入任何内容。
    """
    from agent.prompt_builder import build_system_prompt_layers
    fake_store = MagicMock()
    fake_store.snapshot_for_prompt.return_value = (
        "- [测试专用记忆](.memory/abc123.md) — 独一无二的内容XYZ789"
    )
    layers = build_system_prompt_layers(memory_store=fake_store)
    # 记忆索引不注入 context 层
    assert "独一无二的内容XYZ789" not in layers.context
    assert "记忆索引" not in layers.context
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
