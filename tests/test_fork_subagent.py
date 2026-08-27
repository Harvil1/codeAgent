"""fork 子代理路径测试。

测试 build_forked_messages + build_forked_system_prompt 的正确性：
- 父对话前缀的字节级 cache-identical 保证
- placeholder tool_result 配对
- max_parent_turns 截断
- fail-open fallback
- _run_child fork 分支端到端接入
"""

import hashlib
import json
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# build_forked_messages
# ---------------------------------------------------------------------------

from agent.fork_messages import (
    build_forked_messages,
    build_forked_system_prompt,
)


def test_fork_messages_basic_structure():
    """基本结构：assistant turn + directive（无 tool_calls 场景）。"""
    parent_messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    forked = build_forked_messages(
        parent_messages=parent_messages,
        parent_system_prompt="父 prompt",
        child_directive="请分析这段对话",
    )
    # 最后一条是 directive
    assert forked[-1]["role"] == "user"
    assert "请分析这段对话" in forked[-1]["content"]
    # assistant turn 被包含
    assistant_turns = [m for m in forked if m["role"] == "assistant"]
    assert len(assistant_turns) >= 1
    assert "你好，有什么可以帮你？" in assistant_turns[-1]["content"]


def test_fork_messages_placeholder_tool_result():
    """父 assistant turn 含 tool_calls 时，补 placeholder tool_result。"""
    parent_messages = [
        {"role": "user", "content": "读文件"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_abc", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "真实结果"},
        {"role": "assistant", "content": "文件内容是..."},
    ]
    forked = build_forked_messages(
        parent_messages=parent_messages,
        parent_system_prompt="父 prompt",
        child_directive="总结",
    )
    # 找到 call_abc 对应的 placeholder tool_result
    placeholder_results = [
        m for m in forked
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_abc"
    ]
    assert len(placeholder_results) == 1
    # placeholder 内容（不等于父真实结果）
    assert placeholder_results[0]["content"] != "真实结果"
    assert "fork" in placeholder_results[0]["content"].lower() or "placeholder" in placeholder_results[0]["content"].lower()


def test_fork_messages_max_parent_turns():
    """max_parent_turns 限制继承的 turn 数。"""
    parent_messages = []
    for i in range(10):
        parent_messages.append({"role": "user", "content": f"问题 {i}"})
        parent_messages.append({"role": "assistant", "content": f"回答 {i}"})

    forked = build_forked_messages(
        parent_messages=parent_messages,
        parent_system_prompt="父 prompt",
        child_directive="分析",
        max_parent_turns=3,
    )
    # assistant turn 数不超过 3（directive 是 user role，不算）
    assistant_count = sum(1 for m in forked if m["role"] == "assistant")
    assert assistant_count == 3


def test_fork_messages_directive_at_end():
    """directive 一定是最后一条 user 消息。"""
    parent_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    forked = build_forked_messages(
        parent_messages=parent_messages,
        parent_system_prompt="父 prompt",
        child_directive="去做任务 X",
    )
    last = forked[-1]
    assert last["role"] == "user"
    assert "去做任务 X" in last["content"]
    # directive 消息含 FORK 标记
    assert "FORK" in last["content"].upper()


def test_fork_messages_empty_parent():
    """父 messages 为空时，forked 只有 directive。"""
    forked = build_forked_messages(
        parent_messages=[],
        parent_system_prompt="父 prompt",
        child_directive="任务",
    )
    assert len(forked) == 1
    assert forked[0]["role"] == "user"
    assert "任务" in forked[0]["content"]


def test_fork_messages_parent_assistant_bytes_preserved():
    """父 assistant turn 字节被原封不动地包含（cache-identical 前提）。"""
    parent_assistant_content = "这是父的回答，含特殊字符：\n\t换行制表符。"
    parent_messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": parent_assistant_content},
    ]
    forked = build_forked_messages(
        parent_messages=parent_messages,
        parent_system_prompt="父 prompt",
        child_directive="分析",
    )
    # 父 assistant content 在 forked messages 中原样存在
    found = False
    for m in forked:
        if m["role"] == "assistant" and m.get("content") == parent_assistant_content:
            found = True
            break
    assert found, "父 assistant turn 字节未被原样保留"


# ---------------------------------------------------------------------------
# build_forked_system_prompt
# ---------------------------------------------------------------------------

def test_fork_system_prompt_prefix_identical_to_parent():
    """fork system prompt 的前缀必须与父 system prompt 字节完全一致。

    cache-identical 保证：用 hashlib.sha256 校验前缀字节。
    """
    parent_prompt = (
        "# 系统提示\n"
        "你是 OmniMate Agent。\n"
        "工具列表：read_file, terminal, ...\n"
    )
    fork_prompt = build_forked_system_prompt(parent_prompt, child_role="leaf")

    # fork_prompt 的前缀必须严格 startswith 父 prompt
    assert fork_prompt.startswith(parent_prompt), (
        "fork system prompt 前缀不等于父 prompt，cache 会 break"
    )

    # 字节级校验：fork_prompt 编码后前 len(parent_bytes) 字节必须等于 parent_bytes
    parent_bytes = parent_prompt.encode("utf-8")
    fork_full_bytes = fork_prompt.encode("utf-8")
    fork_prefix_bytes = fork_full_bytes[:len(parent_bytes)]
    assert hashlib.sha256(parent_bytes).digest() == hashlib.sha256(fork_prefix_bytes).digest()


def test_fork_system_prompt_has_fork_marker():
    """fork system prompt 末尾有 FORK MODE 标记。"""
    parent_prompt = "parent prompt"
    fork_prompt = build_forked_system_prompt(parent_prompt, child_role="leaf")

    assert "FORK" in fork_prompt.upper()
    assert "leaf" in fork_prompt


def test_fork_system_prompt_empty_parent():
    """父 prompt 为空时也不应崩溃。"""
    fork_prompt = build_forked_system_prompt("", child_role="leaf")
    assert "FORK" in fork_prompt.upper()


# ---------------------------------------------------------------------------
# _run_child fork 分支（mock AIAgent）
# ---------------------------------------------------------------------------

def test_run_child_fork_mode_passes_initial_messages(monkeypatch):
    """fork=True 时 _run_child 给 AIAgent 传 initial_messages。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "fork result"

    # 构造假父 agent
    fake_parent = MagicMock()
    fake_parent.conversation_history = [
        {"role": "user", "content": "父问题"},
        {"role": "assistant", "content": "父回答"},
    ]
    fake_parent._get_system_prompt = MagicMock(return_value="父 system prompt")
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        from tools.delegate_tool import _run_child
                        result = _run_child(
                            "fork task", "ctx", "leaf",
                            fork=True,
                            agent_ref=fake_parent,
                            subagent_type="general-purpose",
                        )

    # AIAgent 收到了 initial_messages
    assert "initial_messages" in captured
    initial = captured["initial_messages"]
    # 含父 assistant turn
    assistant_msgs = [m for m in initial if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    # 最后是 directive
    assert initial[-1]["role"] == "user"
    # system_prompt_override 前缀含父 system prompt
    assert captured["system_prompt_override"].startswith("父 system prompt")
    assert result == "fork result"


def test_run_child_fork_false_no_initial_messages(monkeypatch):
    """fork=False（默认）不传 initial_messages（回归保护）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "non-fork result"

    fake_parent = MagicMock()
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        from tools.delegate_tool import _run_child
                        result = _run_child(
                            "normal task", "ctx", "leaf",
                            fork=False,
                            agent_ref=fake_parent,
                            subagent_type="general-purpose",
                        )

    # 非 fork 模式不传 initial_messages
    assert "initial_messages" not in captured or captured.get("initial_messages") is None
    # system_prompt_override 不含 fork marker
    if captured.get("system_prompt_override"):
        assert "FORK" not in captured["system_prompt_override"].upper()
    assert result == "non-fork result"


def test_run_child_fork_fail_open_on_exception(monkeypatch):
    """fork 构造失败时 fallback 到非 fork 路径（fail-open）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "fallback result"

    # 构造异常父 agent（_get_system_prompt 抛错）
    fake_parent = MagicMock()
    fake_parent.conversation_history = "not a list"  # 会触发异常
    fake_parent._get_system_prompt = MagicMock(side_effect=RuntimeError("crash"))
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        from tools.delegate_tool import _run_child
                        result = _run_child(
                            "fork with broken parent", "ctx", "leaf",
                            fork=True,
                            agent_ref=fake_parent,
                            subagent_type="general-purpose",
                        )

    # 即使 fork 构造异常，也能 fallback 到非 fork 模式
    assert result == "fallback result"


def test_run_child_fork_config_disabled(monkeypatch):
    """fork_subagent_enabled=False 时，即使传 fork=True 也不走 fork 路径。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self.conversation_history = []

        async def chat(self, msg):
            return "non-fork result"

    fake_parent = MagicMock()
    fake_parent.conversation_history = [
        {"role": "user", "content": "父问题"},
    ]
    fake_parent._get_system_prompt = MagicMock(return_value="父 prompt")
    fake_parent.spawn_depth = 0
    fake_parent.effort_level = None
    fake_parent._children = []
    fake_parent.hooks_registry = None
    fake_parent.aux_llm_router = None
    fake_parent._stream_callback = None

    config_dict = {
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
        "delegation": {"fork_subagent_enabled": False},
    }
    with patch("config.load_config", return_value=config_dict):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        from tools.delegate_tool import _run_child
                        result = _run_child(
                            "task", "ctx", "leaf",
                            fork=True,
                            agent_ref=fake_parent,
                            subagent_type="general-purpose",
                            config=config_dict,
                        )

    # config 关闭 fork，不传 initial_messages
    assert "initial_messages" not in captured or not captured.get("initial_messages")


def test_subagent_schema_has_fork_param():
    """subagent schema 包含 fork 参数（T10 起支持 true | false | "full"）。"""
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    params = DELEGATE_TASK_SCHEMA["parameters"]
    props = params["properties"]
    assert "fork" in props
    assert "boolean" in props["fork"]["type"]
    assert "full" in props["fork"].get("enum", [])
    assert props["fork"].get("default", False) is False
