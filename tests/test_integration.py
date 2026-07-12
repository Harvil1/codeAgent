"""端到端集成测试。

验证各模块协同工作：
- 工具系统完整发现
- AIAgent 集成记忆/会话/技能
- CLI RuntimeContext 初始化
- Mock 一轮完整对话
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import AIAgent
from agent.memory_store import MemoryStore
from agent.memory_manager import MemoryManager
from agent.session_store import SessionStore
from agent.budget import IterationBudget
from model_tools import get_tool_definitions, handle_function_call
from tools.registry import registry


# ---------------------------------------------------------------------------
# 工具系统完整性
# ---------------------------------------------------------------------------

EXPECTED_CORE_TOOLS = {
    "terminal", "read_file", "write_file", "search_files",
    "memory", "skills_list", "skill_view", "skill_manage",
    "session_search", "delegate_task",
}


def test_all_core_tools_discovered():
    """所有 core 工具集的工具都被注册并能被发现。"""
    defs = get_tool_definitions(["core"])
    names = {d["function"]["name"] for d in defs}
    missing = EXPECTED_CORE_TOOLS - names
    assert not missing, f"缺少工具: {missing}"


def test_tool_dispatch_through_handle_function_call():
    """handle_function_call 能正确分发到 registry。"""
    result = handle_function_call(
        "read_file",
        {"path": "nonexistent_xyz"},
    )
    data = json.loads(result)
    assert "error" in data  # 文件不存在


# ---------------------------------------------------------------------------
# AIAgent 集成
# ---------------------------------------------------------------------------

def test_agent_accepts_all_components(tmp_path):
    """AIAgent 能接收 memory_store + memory_manager + session_store。"""
    memory_store = MemoryStore(tmp_path)
    memory_manager = MemoryManager(memory_store)
    session_store = SessionStore(tmp_path / "s.db")

    agent = AIAgent(
        base_url="https://example.com/v1",
        api_key="fake-key",
        model="test-model",
        memory_store=memory_store,
        memory_manager=memory_manager,
        session_store=session_store,
        harvil_home=tmp_path,
        enabled_toolsets=[],
    )

    assert agent.memory_store is memory_store
    assert agent.memory_manager is memory_manager
    assert agent.session_store is session_store
    assert agent.harvil_home == tmp_path


def test_agent_budget_initialized():
    agent = AIAgent(
        api_key="fake",
        model="test",
        max_iterations=42,
        enabled_toolsets=[],
    )
    assert isinstance(agent.iteration_budget, IterationBudget)
    assert agent.iteration_budget.total == 42
    assert agent.iteration_budget.remaining == 42


# ---------------------------------------------------------------------------
# Mock 完整对话
# ---------------------------------------------------------------------------

def _make_mock_llm_client(response_text="hello", tool_calls=None):
    """构造 mock LLMClient（有 chat_completions 方法，返回 OpenAI 兼容响应）。"""
    def fake_chat_completions(messages, *, tools=None, **kwargs):
        msg = SimpleNamespace(content=response_text, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return SimpleNamespace(chat_completions=fake_chat_completions)


# 向后兼容别名
def _make_mock_client(response_text="hello", tool_calls=None):
    return _make_mock_llm_client(response_text, tool_calls)


def test_mock_simple_conversation(tmp_path):
    """Mock 一轮简单对话（无工具调用）。"""
    memory_store = MemoryStore(tmp_path)
    memory_store.add("memory", "测试记忆")

    agent = AIAgent(
        api_key="fake",
        model="test",
        memory_store=memory_store,
        enabled_toolsets=[],
    )
    agent.llm_client = _make_mock_llm_client(response_text="你好，我是 agent")

    response = agent.chat("hi")

    assert response == "你好，我是 agent"
    # 消息历史包含 user + assistant
    roles = [m["role"] for m in agent.conversation_history]
    assert "user" in roles
    assert "assistant" in roles


def test_mock_conversation_with_tool_call(tmp_path):
    """Mock 一轮带工具调用的对话。"""
    # 第 1 次 API 调用返回 tool_call，第 2 次返回最终响应
    call_count = [0]

    def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # 返回工具调用
            tool_call = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(
                    name="read_file",
                    arguments=json.dumps({"path": str(tmp_path / "test.txt")}),
                ),
            )
            msg = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            # 返回最终响应
            msg = SimpleNamespace(content="文件不存在", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    # 准备测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello", encoding="utf-8")

    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=["core"],
        harvil_home=tmp_path,
    )
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    response = agent.chat("读这个文件")

    assert "文件不存在" not in response or "hello" in response or call_count[0] >= 2
    # 至少调用了 2 次 API
    assert call_count[0] >= 2


def test_mock_conversation_with_memory_injection(tmp_path):
    """记忆快照被注入到 system prompt。"""
    memory_store = MemoryStore(tmp_path)
    memory_store.add("memory", "特殊标记 XYZ")

    agent = AIAgent(
        api_key="fake",
        model="test",
        memory_store=memory_store,
        enabled_toolsets=[],
    )

    captured_messages = []

    def fake_chat_completions(messages, *, tools=None, **kwargs):
        captured_messages.append(messages)
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    agent.chat("test")

    # system prompt 包含记忆
    assert len(captured_messages) > 0
    system_msg = captured_messages[0][0]
    assert system_msg["role"] == "system"
    assert "特殊标记 XYZ" in system_msg["content"]


def test_interrupt_stops_conversation(tmp_path):
    """中断标志能停止对话循环。"""
    def fake_chat_completions(messages, *, tools=None, **kwargs):
        msg = SimpleNamespace(content="response", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=[],
    )
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    # 在循环前设置中断
    agent.interrupt()
    response = agent.chat("test")

    assert "中断" in response


# ---------------------------------------------------------------------------
# CLI RuntimeContext
# ---------------------------------------------------------------------------

def test_runtime_context_initializes(tmp_path, monkeypatch):
    """RuntimeContext 能完整初始化（不实际连接 API）。"""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")

    from cli import RuntimeContext
    rt = RuntimeContext()
    rt.initialize()

    assert rt.memory_store is not None
    assert rt.memory_manager is not None
    assert rt.session_store is not None
    assert rt.agent is not None
    assert rt.session_id is not None


def test_runtime_context_no_api_key(tmp_path, monkeypatch):
    """无 API key 时优雅退出。"""
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    # 显式写 api_key 为空的 settings.json，避免迁移/污染干扰
    import json
    (tmp_path / "settings.json").write_text(json.dumps({
        "models": {"deepseek": {
            "format": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
        }},
        "default_model": "deepseek",
        "enabled_toolsets": ["core"],
    }, ensure_ascii=True), encoding="utf-8")

    # 清除所有可能的 API key 环境变量
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "OPENROUTER_API_KEY", "GLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    from cli import RuntimeContext
    rt = RuntimeContext()
    with pytest.raises(SystemExit):
        rt.initialize()


# ---------------------------------------------------------------------------
# 系统级集成
# ---------------------------------------------------------------------------

def test_skill_injects_into_agent_context(tmp_path, monkeypatch):
    """技能通过 user 消息注入（不进入 system prompt）。"""
    from agent.skill_commands import execute_skill

    # 创建临时技能
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "test-skill").mkdir()
    (skills / "test-skill" / "SKILL.md").write_text(
        '---\nname: test-skill\ndescription: "测试"\n---\n# 测试技能\n指令内容',
        encoding="utf-8",
    )

    injected_msg = execute_skill(
        str(skills / "test-skill" / "SKILL.md"),
        "用户实际消息",
    )

    # 验证技能正文 + 用户消息都在
    assert "测试技能" in injected_msg
    assert "用户实际消息" in injected_msg
    assert "[技能已加载]" in injected_msg


def test_session_persistence_round_trip(tmp_path):
    """会话存储能往返持久化。"""
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session(model="test", provider="test")

    # 模拟一轮对话
    store.append_message(sid, "user", "hello")
    store.append_message(sid, "assistant", "hi there")

    # 读取验证
    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "hi there"

    # 搜索能找到
    results = store.search("hello")
    assert len(results) > 0


# ---------------------------------------------------------------------------
# 上下文管线开关分支
# ---------------------------------------------------------------------------

def test_compress_if_needed_signature_matches_integration():
    """compress_if_needed 签名匹配 AIAgent 集成层的调用约定。

    端到端验证在 Task 11 完成；此处只验证开关 True 时新管线能独立跑通。
    """
    from agent.context_pipeline import compress_if_needed, CompressionSessionState

    msgs = [{"role": "system", "content": "s"}]
    msgs += [{"role": "user", "content": f"u{i}"} for i in range(60)]
    state = CompressionSessionState()
    out, changed = compress_if_needed(
        msgs,
        attempt_count=0,
        llm_client=None,
        model=None,
        config={
            "snip_message_threshold": 50,
            "snip_keep_first": 3,
            "snip_keep_last": 47,
            "micro_keep_recent_results": 3,
            "llm_compact_token_threshold": 100000,
            "llm_compact_message_threshold": 100,
            "llm_compact_keep_recent": 10,
            "llm_compact_cooldown_turns": 5,
            "max_compress_attempts": 3,
            "transcript_enabled": False,
            "transcript_retention": 20,
        },
        session_state=state,
        agent_home=None,
        session_id="t",
    )
    # 60 条消息 > snip 阈值 50，L1 应触发
    assert changed is True
    # 输出含 snip_compact 占位消息
    assert any("snip_compact" in m.get("content", "") for m in out)


def test_aiagent_old_pipeline_explicit_false_no_crash(tmp_path):
    """use_new_pipeline=False（显式）时，AIAgent 走旧 maybe_compress 路径不抛。

    默认开关已切为 True（Commit 6），本测试显式设 False 验证旧路径仍然可用。
    """
    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=[],
        harvil_home=tmp_path,
        config={"context": {"use_new_pipeline": False}},
    )
    agent.llm_client = _make_mock_llm_client(response_text="好的")
    # 显式设 False
    assert agent.config.get("context", {}).get("use_new_pipeline") is False
    # compression_enabled 默认 True，主循环会调用 maybe_compress（旧路径）
    response = agent.chat("hi")
    assert response == "好的"


def test_aiagent_new_pipeline_flag_true(tmp_path):
    """use_new_pipeline=True 时走新管线（不抛即可）。

    端到端验证在 Task 11；此处只确认开关分支被正确命中。
    """
    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=[],
        harvil_home=tmp_path,
        config={"context": {"use_new_pipeline": True}},
    )
    agent.llm_client = _make_mock_llm_client(response_text="ok")
    assert agent.config.get("context", {}).get("use_new_pipeline") is True
    # 短对话不会触发任何压缩，开关分支只是被命中
    response = agent.chat("hi")
    assert response == "ok"


# ---------------------------------------------------------------------------
# 工具层 offload 集成（Task 10）
# ---------------------------------------------------------------------------

def test_prompt_builder_includes_offload_guidance():
    """system prompt 的 TOOL_USAGE_GUIDANCE 应含占位消息识别段。"""
    from agent.prompt_builder import TOOL_USAGE_GUIDANCE
    assert "snip_compact" in TOOL_USAGE_GUIDANCE
    assert ".transcripts" in TOOL_USAGE_GUIDANCE


def test_terminal_offload_not_triggered_when_flag_off(tmp_path):
    """use_new_pipeline=False 时，terminal 大输出不触发 offload。"""
    # 直接调 handler，模拟大 stdout
    long_stdout = "x" * 50000
    args = {"command": f"echo {long_stdout[:10]}"}
    result = _handle_terminal_direct(args, harvil_home=tmp_path, config={})
    # 开关 False，返回原样 JSON（不含 offload 标记）
    # 注意：echo 命令真实执行，但输出远小于 50000（shell 截断）
    # 这里主要验证 _finalize_output 不介入
    data = json.loads(result)
    assert "truncated" not in data  # 没有 offload 截断字段


def test_terminal_offload_triggered_when_flag_on(tmp_path):
    """use_new_pipeline=True 且 stdout 超阈值时，走 offload。"""
    from tools.terminal_tool import _finalize_output as terminal_finalize
    long_content = "x" * 50000
    config = {"context": {"use_new_pipeline": True, "output_offload_threshold": 30000}}
    result = terminal_finalize(long_content, "call_test_offload", tmp_path, config)
    parsed = json.loads(result)
    assert parsed.get("truncated") is True
    assert "full_at" in parsed
    assert "preview" in parsed


def test_terminal_offload_not_triggered_when_flag_off_explicit():
    """use_new_pipeline=False 时 _finalize_output 原样返回。"""
    from tools.terminal_tool import _finalize_output as terminal_finalize
    content = "x" * 50000
    result = terminal_finalize(content, "call_no_offload", None, {})
    assert result == content  # 原样返回


def test_file_read_offload_triggered_when_flag_on(tmp_path):
    """use_new_pipeline=True 且文件内容超阈值时，read_file 走 offload。"""
    from tools.file_operations import _finalize_output as file_finalize
    long_content = "y" * 50000
    config = {"context": {"use_new_pipeline": True, "output_offload_threshold": 30000}}
    result = file_finalize(long_content, "call_file_offload", tmp_path, config)
    parsed = json.loads(result)
    assert parsed.get("truncated") is True
    assert "full_at" in parsed


def test_file_read_offload_not_triggered_when_flag_off():
    """use_new_pipeline=False 时 file _finalize_output 原样返回。"""
    from tools.file_operations import _finalize_output as file_finalize
    content = "y" * 50000
    result = file_finalize(content, "call_no_file", None, {})
    assert result == content


def test_handle_function_call_passes_tool_call_id_and_config(tmp_path):
    """handle_function_call 透传 tool_call_id 和 config 给 handler。"""
    # 用一个能产生大输出的方式验证透传
    # 创建大文件，用 read_file 读
    big_file = tmp_path / "big.txt"
    big_file.write_text("A" * 50000, encoding="utf-8")

    result = handle_function_call(
        "read_file",
        {"path": str(big_file)},
        harvil_home=tmp_path,
        tool_call_id="call_integration_1",
        config={"context": {"use_new_pipeline": True, "output_offload_threshold": 30000}},
    )
    data = json.loads(result)
    # 开关开启 + 文件大 → 应该走 offload（content 字段被替换）
    assert data.get("content_offloaded") is True
    # offload 信息可解析
    offload_info = json.loads(data["content"])
    assert offload_info.get("truncated") is True


def _handle_terminal_direct(args, **kwargs):
    """直接调用 terminal handler（helper for test）。"""
    from tools.terminal_tool import _handle_terminal
    return _handle_terminal(args, **kwargs)


# ---------------------------------------------------------------------------
# 端到端：200 轮对话 + 新管线（Task 11）
# ---------------------------------------------------------------------------

def test_e2e_200_turn_conversation_with_pipeline(tmp_path):
    """端到端：200 轮工具调用对话，验证新管线稳定。

    - 构造 200 轮 user+assistant 对话历史
    - 每 10 轮注入一个 50KB 工具结果（触发 output_offload）
    - 跑 5 轮 compress_if_needed（模拟每轮 LLM 前调用）
    - 断言：offload 文件 > 0，transcript ≥ 1，L4 触发 ≥ 1，全程无异常
    - 最终 messages 长度应远小于起始（压缩生效）
    - reactive_compact 也能无异常调用（紧急通道不崩溃）
    """
    from agent.context_pipeline import (
        compress_if_needed, CompressionSessionState, reactive_compact,
    )
    from agent.output_offload import maybe_offload

    config = {
        "snip_message_threshold": 50,
        "snip_keep_first": 3,
        "snip_keep_last": 47,
        "micro_keep_recent_results": 3,
        # L4 阈值设低：L1 snip 后仍有 ~51 条 conv 消息，确保 L4 能触发
        "llm_compact_token_threshold": 100000,
        "llm_compact_message_threshold": 40,
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "transcript_enabled": True,
        "transcript_retention": 20,
    }

    class _FakeLLM:
        """假 LLM client，chat_completions 返回固定摘要。"""

        def __init__(self):
            self.call_count = 0

        def chat_completions(self, msgs, **kwargs):
            self.call_count += 1
            m = SimpleNamespace(
                content="这是对话摘要：用户进行了多轮工具调用。",
                tool_calls=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=m)])

    # 构造 200 轮对话历史
    messages = [{"role": "system", "content": "sys"}]
    for i in range(200):
        messages.append({"role": "user", "content": f"turn {i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
        # 每 10 轮加一个大 tool 结果（触发 offload）
        if i % 10 == 0:
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }],
            })
            big = "x" * 50000
            offloaded = maybe_offload(
                big, tool_call_id=f"call_{i}", agent_home=tmp_path,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "t",
                "content": offloaded,
            })

    initial_len = len(messages)
    assert initial_len > 400, f"初始历史应 >400 条，实际 {initial_len}"

    state = CompressionSessionState()
    llm = _FakeLLM()

    # 跑 5 轮压缩（模拟每轮 LLM 前调用）
    for turn in range(5):
        state.current_turn = turn
        messages, _ = compress_if_needed(
            messages,
            attempt_count=turn,
            llm_client=llm,
            model="x",
            config=config,
            session_state=state,
            agent_home=tmp_path,
            session_id="e2e",
        )

    # 断言：offload 文件 > 0
    offload_files = list(
        (tmp_path / ".task_outputs" / "tool-results").glob("*.txt")
    )
    assert len(offload_files) > 0, "应该有 offload 文件"

    # 断言：transcript ≥ 1（L4 触发时 force=True 落盘）
    transcripts = list((tmp_path / ".transcripts").glob("transcript_*.jsonl"))
    assert len(transcripts) >= 1, "应该至少有一个 transcript 快照"

    # 断言：L4 至少触发一次
    assert state.llm_compact_count >= 1, "L4 应至少触发一次"

    # 断言：最终 messages 长度应远小于起始（压缩生效）
    assert len(messages) < 100, (
        f"压缩后消息数应 <100，实际 {len(messages)}（起始 {initial_len}）"
    )

    # 断言：reactive_compact 也能无异常调用（紧急通道不崩溃）
    reactive_state = CompressionSessionState()
    reactive_out, reactive_changed = reactive_compact(
        messages, session_state=reactive_state, keep_recent=5,
    )
    assert reactive_changed is True
    assert len(reactive_out) <= 7  # system + placeholder + 5 recent
    assert reactive_state.reacted is True
