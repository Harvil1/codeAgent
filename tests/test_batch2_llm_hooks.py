"""batch2-T2: PRE_LLM_CALL / POST_LLM_CALL hooks 测试。"""
from unittest.mock import MagicMock

import pytest

from agent.hooks import HookEvent, HookRegistry


# ---------------------------------------------------------------------------
# PRE_LLM_CALL
# ---------------------------------------------------------------------------

def test_pre_llm_call_no_hooks_passthrough():
    reg = HookRegistry()
    msgs = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function"}]
    out_msgs, out_tools = reg.run_pre_llm_call(msgs, tools, session_id="s")
    assert out_msgs is msgs
    assert out_tools is tools


def test_pre_llm_call_single_hook_modifies_messages():
    reg = HookRegistry()
    original = [{"role": "user", "content": "hi"}]
    new_msgs = [{"role": "user", "content": "modified"}]
    reg.register_pre_llm_call(
        lambda msgs, tools: (new_msgs, tools), name="modifier"
    )
    out_msgs, out_tools = reg.run_pre_llm_call(original, None, session_id="s")
    assert out_msgs == new_msgs


def test_pre_llm_call_single_hook_modifies_tools():
    reg = HookRegistry()
    new_tools = [{"type": "function", "name": "new_tool"}]
    reg.register_pre_llm_call(
        lambda msgs, tools: (msgs, new_tools), name="tool_modifier"
    )
    out_msgs, out_tools = reg.run_pre_llm_call([], [], session_id="s")
    assert out_tools == new_tools


def test_pre_llm_call_none_passthrough():
    """hook 返回 None 时不修改。"""
    reg = HookRegistry()
    reg.register_pre_llm_call(lambda m, t: None, name="noop")
    msgs = [{"role": "user", "content": "hi"}]
    out_msgs, out_tools = reg.run_pre_llm_call(msgs, None, session_id="s")
    assert out_msgs is msgs
    assert out_tools is None


def test_pre_llm_call_chain_composition():
    """多个 hook 链式：每个看到前一个的输出。"""
    reg = HookRegistry()
    reg.register_pre_llm_call(
        lambda m, t: ([{"role": "user", "content": m[0]["content"] + "1"}], t),
        name="a",
    )
    reg.register_pre_llm_call(
        lambda m, t: ([{"role": "user", "content": m[0]["content"] + "2"}], t),
        name="b",
    )
    out_msgs, _ = reg.run_pre_llm_call(
        [{"role": "user", "content": "x"}], None, session_id="s"
    )
    assert out_msgs[0]["content"] == "x12"


def test_pre_llm_call_exception_isolated():
    """hook 抛异常时不影响链。"""
    reg = HookRegistry()

    def bad(m, t): raise ValueError("boom")
    reg.register_pre_llm_call(bad, name="bad")
    reg.register_pre_llm_call(
        lambda m, t: ([{"role": "user", "content": "ok"}], t), name="ok"
    )
    out_msgs, _ = reg.run_pre_llm_call(
        [{"role": "user", "content": "x"}], None, session_id="s"
    )
    assert out_msgs[0]["content"] == "ok"


def test_pre_llm_call_non_tuple_return_ignored():
    """hook 返回非元组时忽略。"""
    reg = HookRegistry()
    reg.register_pre_llm_call(
        lambda m, t: "not a tuple", name="bad_return"
    )
    msgs = [{"role": "user", "content": "hi"}]
    out_msgs, out_tools = reg.run_pre_llm_call(msgs, [], session_id="s")
    assert out_msgs is msgs  # 不变


# ---------------------------------------------------------------------------
# POST_LLM_CALL
# ---------------------------------------------------------------------------

def test_post_llm_call_no_hooks_passthrough():
    reg = HookRegistry()
    resp = MagicMock()
    out = reg.run_post_llm_call(resp, session_id="s")
    assert out is resp


def test_post_llm_call_single_hook_modifies():
    reg = HookRegistry()
    original = MagicMock()
    modified = MagicMock()
    reg.register_post_llm_call(lambda r: modified, name="modifier")
    out = reg.run_post_llm_call(original, session_id="s")
    assert out is modified


def test_post_llm_call_none_passthrough():
    reg = HookRegistry()
    reg.register_post_llm_call(lambda r: None, name="noop")
    resp = MagicMock()
    out = reg.run_post_llm_call(resp, session_id="s")
    assert out is resp


def test_post_llm_call_chain_composition():
    """多个 hook 链式。"""
    reg = HookRegistry()

    counter = [0]

    def add_tag_1(r):
        counter[0] += 1
        r.tag1 = True
        return r

    def add_tag_2(r):
        counter[0] += 10
        r.tag2 = True
        return r

    reg.register_post_llm_call(add_tag_1, name="tag1")
    reg.register_post_llm_call(add_tag_2, name="tag2")

    resp = MagicMock()
    out = reg.run_post_llm_call(resp, session_id="s")
    assert hasattr(out, "tag1")
    assert hasattr(out, "tag2")
    assert counter[0] == 11


def test_post_llm_call_exception_isolated():
    reg = HookRegistry()

    def bad(r): raise ValueError("boom")
    reg.register_post_llm_call(bad, name="bad")

    target = MagicMock()
    target.marked = True
    reg.register_post_llm_call(lambda r: target, name="ok")

    resp = MagicMock()
    out = reg.run_post_llm_call(resp, session_id="s")
    assert out is target


# ---------------------------------------------------------------------------
# clear 支持 LLM hooks
# ---------------------------------------------------------------------------

def test_clear_pre_llm_call():
    reg = HookRegistry()
    reg.register_pre_llm_call(lambda m, t: ([], t), name="x")
    assert len(reg._hooks[HookEvent.PRE_LLM_CALL]) == 1
    reg.clear(HookEvent.PRE_LLM_CALL)
    assert len(reg._hooks[HookEvent.PRE_LLM_CALL]) == 0


def test_clear_post_llm_call():
    reg = HookRegistry()
    reg.register_post_llm_call(lambda r: None, name="x")
    assert len(reg._hooks[HookEvent.POST_LLM_CALL]) == 1
    reg.clear(HookEvent.POST_LLM_CALL)
    assert len(reg._hooks[HookEvent.POST_LLM_CALL]) == 0


def test_clear_all_includes_llm_hooks():
    reg = HookRegistry()
    reg.register_pre_llm_call(lambda m, t: None, name="pre")
    reg.register_post_llm_call(lambda r: None, name="post")
    reg.clear()
    assert len(reg._hooks[HookEvent.PRE_LLM_CALL]) == 0
    assert len(reg._hooks[HookEvent.POST_LLM_CALL]) == 0
