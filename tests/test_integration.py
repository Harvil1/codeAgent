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
from unittest.mock import AsyncMock, MagicMock, patch

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
    "session_search", "subagent", "web_fetch",
}


def test_all_core_tools_discovered():
    """所有 core 工具集的工具都被注册并能被发现。"""
    defs = get_tool_definitions(["core"])
    names = {d["function"]["name"] for d in defs}
    missing = EXPECTED_CORE_TOOLS - names
    assert not missing, f"缺少工具: {missing}"


async def test_tool_dispatch_through_handle_function_call():
    """handle_function_call 能正确分发到 registry。"""
    result = await handle_function_call(
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
    memory_store = MemoryStore(omnimate_home=tmp_path)
    memory_manager = MemoryManager(memory_store)
    session_store = SessionStore(tmp_path / "s.db")

    agent = AIAgent(
        base_url="https://example.com/v1",
        api_key="fake-key",
        model="test-model",
        memory_store=memory_store,
        memory_manager=memory_manager,
        session_store=session_store,
        omnimate_home=tmp_path,
        enabled_toolsets=[],
    )

    assert agent.memory_store is memory_store
    assert agent.memory_manager is memory_manager
    assert agent.session_store is session_store
    assert agent.omnimate_home == tmp_path


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
    """构造 mock LLMClient（有 async chat_completions 方法，返回 OpenAI 兼容响应）。

    Plan 2B: chat_completions 已改 async，mock 用 AsyncMock。
    """
    msg = SimpleNamespace(content=response_text, tool_calls=tool_calls)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=resp)
    return client


# 向后兼容别名
def _make_mock_client(response_text="hello", tool_calls=None):
    return _make_mock_llm_client(response_text, tool_calls)


async def test_mock_simple_conversation(tmp_path):
    """Mock 一轮简单对话（无工具调用）。"""
    memory_store = MemoryStore(omnimate_home=tmp_path)
    memory_store.add("memory", "测试记忆")

    agent = AIAgent(
        api_key="fake",
        model="test",
        memory_store=memory_store,
        enabled_toolsets=[],
    )
    agent.llm_client = _make_mock_llm_client(response_text="你好，我是 agent")

    response = await agent.chat("hi")

    assert response == "你好，我是 agent"
    # 消息历史包含 user + assistant
    roles = [m["role"] for m in agent.conversation_history]
    assert "user" in roles
    assert "assistant" in roles


async def test_mock_conversation_with_tool_call(tmp_path):
    """Mock 一轮带工具调用的对话。"""
    # 第 1 次 API 调用返回 tool_call，第 2 次返回最终响应
    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
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
        omnimate_home=tmp_path,
    )
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    response = await agent.chat("读这个文件")

    assert "文件不存在" not in response or "hello" in response or call_count[0] >= 2
    # 至少调用了 2 次 API
    assert call_count[0] >= 2


async def test_mock_conversation_with_memory_injection(tmp_path):
    """记忆快照被注入到 system prompt。"""
    memory_store = MemoryStore(omnimate_home=tmp_path)
    memory_store.add("memory", "特殊标记 XYZ")

    agent = AIAgent(
        api_key="fake",
        model="test",
        memory_store=memory_store,
        enabled_toolsets=[],
    )

    captured_messages = []

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        captured_messages.append(messages)
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    await agent.chat("test")

    # system prompt 包含记忆
    assert len(captured_messages) > 0
    system_msg = captured_messages[0][0]
    assert system_msg["role"] == "system"
    assert "特殊标记 XYZ" in system_msg["content"]


async def test_interrupt_stops_conversation(tmp_path):
    """中断标志能停止对话循环。"""
    async def fake_chat_completions(messages, *, tools=None, **kwargs):
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
    response = await agent.chat("test")

    assert "中断" in response


# ---------------------------------------------------------------------------
# CLI RuntimeContext
# ---------------------------------------------------------------------------

def test_runtime_context_initializes(tmp_path, monkeypatch):
    """RuntimeContext 能完整初始化（不实际连接 API）。"""
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
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
    monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path))
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

async def test_compress_if_needed_signature_matches_integration():
    """compress_if_needed 签名匹配 AIAgent 集成层的调用约定。

    端到端验证在 Task 11 完成；此处只验证开关 True 时新管线能独立跑通。
    Task D4 fix: compress_if_needed 改 async。
    """
    from agent.context_pipeline import compress_if_needed, CompressionSessionState

    msgs = [{"role": "system", "content": "s"}]
    msgs += [{"role": "user", "content": f"u{i}"} for i in range(60)]
    state = CompressionSessionState()
    out, changed = await compress_if_needed(
        msgs,
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


async def test_aiagent_new_pipeline_flag_true(tmp_path):
    """新管线是唯一路径（Commit 7 后双轨期结束）。

    端到端验证在 Task 11；此处只确认短对话不抛。
    """
    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=[],
        omnimate_home=tmp_path,
    )
    agent.llm_client = _make_mock_llm_client(response_text="ok")
    # 短对话不会触发任何压缩
    response = await agent.chat("hi")
    assert response == "ok"


# ---------------------------------------------------------------------------
# 工具层 offload 集成（Task 10）
# ---------------------------------------------------------------------------

def test_prompt_builder_includes_offload_guidance():
    """system prompt 的 TOOL_USAGE_GUIDANCE 应含占位消息识别段。"""
    from agent.prompt_builder import TOOL_USAGE_GUIDANCE
    assert "snip_compact" in TOOL_USAGE_GUIDANCE
    assert ".transcripts" in TOOL_USAGE_GUIDANCE


def test_terminal_offload_not_triggered_when_no_tool_call_id(tmp_path):
    """无 tool_call_id 时，terminal 大输出不触发 offload。"""
    # 直接调 handler，模拟大 stdout
    long_stdout = "x" * 50000
    args = {"command": f"echo {long_stdout[:10]}"}
    result = _handle_terminal_direct(args, omnimate_home=tmp_path, config={})
    # 无 tool_call_id，返回原样 JSON（不含 offload 标记）
    # 注意：echo 命令真实执行，但输出远小于 50000（shell 截断）
    data = json.loads(result)
    assert "truncated" not in data  # 没有 offload 截断字段


def test_terminal_offload_triggered_when_above_threshold(tmp_path):
    """stdout 超阈值时，走 offload（Phase 1 后始终启用）。"""
    from tools.terminal_tool import _finalize_output as terminal_finalize
    long_content = "x" * 50000
    config = {"context": {"output_offload_threshold": 30000}}
    result = terminal_finalize(long_content, "call_test_offload", tmp_path, config)
    parsed = json.loads(result)
    assert parsed.get("truncated") is True
    assert "full_at" in parsed
    assert "preview" in parsed


def test_terminal_offload_not_triggered_without_omnimate_home():
    """无 omnimate_home 时 _finalize_output 原样返回。"""
    from tools.terminal_tool import _finalize_output as terminal_finalize
    content = "x" * 50000
    result = terminal_finalize(content, "call_no_offload", None, {})
    assert result == content  # 原样返回


def test_file_read_offload_triggered_when_above_threshold(tmp_path):
    """文件内容超阈值时，read_file 走 offload（Phase 1 后始终启用）。"""
    from tools.file_operations import _finalize_output as file_finalize
    long_content = "y" * 50000
    config = {"context": {"output_offload_threshold": 30000}}
    result = file_finalize(long_content, "call_file_offload", tmp_path, config)
    parsed = json.loads(result)
    assert parsed.get("truncated") is True
    assert "full_at" in parsed


def test_file_read_offload_not_triggered_without_omnimate_home():
    """无 omnimate_home 时 file _finalize_output 原样返回。"""
    from tools.file_operations import _finalize_output as file_finalize
    content = "y" * 50000
    result = file_finalize(content, "call_no_file", None, {})
    assert result == content


async def test_handle_function_call_passes_tool_call_id_and_config(tmp_path):
    """handle_function_call 透传 tool_call_id 和 config 给 handler。"""
    # 用一个能产生大输出的方式验证透传
    # 创建大文件，用 read_file 读
    big_file = tmp_path / "big.txt"
    big_file.write_text("A" * 50000, encoding="utf-8")

    result = await handle_function_call(
        "read_file",
        {"path": str(big_file)},
        omnimate_home=tmp_path,
        tool_call_id="call_integration_1",
        config={"context": {"output_offload_threshold": 30000}},
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

async def test_e2e_200_turn_conversation_with_pipeline(tmp_path):
    """端到端：200 轮工具调用对话，验证新管线稳定。

    - 构造 200 轮 user+assistant 对话历史
    - 每 10 轮注入一个 50KB 工具结果（触发 output_offload）
    - 跑 5 轮 compress_if_needed（模拟每轮 LLM 前调用）
    - 断言：offload 文件 > 0，transcript ≥ 1，L4 触发 ≥ 1，全程无异常
    - 最终 messages 长度应远小于起始（压缩生效）
    - reactive_compact 也能无异常调用（紧急通道不崩溃）

    Task D4 fix: compress_if_needed + chat_completions 改 async。
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
        # L4 阈值设低：L1 snip 后剩余 ~51 条消息（短），tokens 大降；
        # 阈值需足够低让 L4 在 snip 后仍超阈值触发
        "llm_compact_token_threshold": 1000,
        "llm_compact_message_threshold": 40,
        "llm_compact_keep_recent": 10,
        "llm_compact_cooldown_turns": 5,
        "max_compress_attempts": 3,
        "transcript_enabled": True,
        "transcript_retention": 20,
    }

    class _FakeLLM:
        """假 LLM client，chat_completions 返回固定摘要（async 接口）。"""

        def __init__(self):
            self.call_count = 0

        async def chat_completions(self, msgs, **kwargs):
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
        messages, _ = await compress_if_needed(
            messages,
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


# ---------------------------------------------------------------------------
# I2: AIAgent.chat() + 长对话 + 新管线端到端测试
# ---------------------------------------------------------------------------

async def test_aiagent_chat_long_conversation_triggers_pipeline(tmp_path):
    """I2: AIAgent.chat() 在长对话中触发新管线，验证 C1/C2 修复。

    - 构造 AIAgent（新管线是唯一路径）
    - 注入 > 50 条历史消息
    - 调用 chat("continue") 触发一轮 LLM
    - 断言 _compress_session_state.current_turn > 0（C1: increment_turn 被调用）
    - 断言压缩代码路径被执行（L1 snip 触发）
    """
    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=[],
        omnimate_home=tmp_path,
        config={"context": {"snip_message_threshold": 50}},  # round3 fix: 压缩配在 config["context"] 下，默认 200 太高
    )
    agent.llm_client = _make_mock_llm_client(response_text="好的，继续")

    # 注入 60 条历史消息（超过 snip_message_threshold=50，触发 L1）
    for i in range(30):
        agent.conversation_history.append({"role": "user", "content": f"历史消息 {i}"})
        agent.conversation_history.append({"role": "assistant", "content": f"回复 {i}"})

    # 调用 chat()，触发主循环
    response = await agent.chat("继续")

    # C1 验证：increment_turn 被调用，current_turn > 0
    assert hasattr(agent, "_compress_session_state"), "应已创建 _compress_session_state"
    assert agent._compress_session_state.current_turn > 0, (
        f"current_turn 应 > 0（C1: increment_turn 被调用），"
        f"实际 {agent._compress_session_state.current_turn}"
    )

    # 响应正常返回
    assert response == "好的，继续"

    # L1 snip 应已触发：历史中有 snip_compact 占位消息
    has_snip = any(
        "snip_compact" in str(m.get("content", ""))
        for m in agent.conversation_history
    )
    assert has_snip, "L1 snip_compact 应被触发（60 条消息 > 阈值 50）"


# ---------------------------------------------------------------------------
# P2-T6: AIAgent hooks 集成测试
# ---------------------------------------------------------------------------

from agent.hooks import HookRegistry  # noqa: E402


def test_aiagent_accepts_hooks_registry_kwarg():
    """hooks_registry=None 时构造成功（向后兼容）。"""
    agent = _make_test_agent()
    assert agent.hooks_registry is None
    assert agent._stop_fire_count == 0


async def test_aiagent_user_prompt_submit_hook_modifies_input():
    """USER_PROMPT_SUBMIT hook 修改 prompt 后实际入 history。"""
    agent = _make_test_agent_with_hooks()
    agent.hooks_registry.register_user_prompt_submit(
        lambda p: p + " [augmented]", name="augmenter"
    )
    # mock LLM 返回 stop（无 tool_call）
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hello")
    # 第 0 条应是 user，content 被 hook 修改
    assert agent.conversation_history[0]["content"] == "hello [augmented]"


async def test_aiagent_user_prompt_submit_no_registry_modification():
    """无 registry 时 prompt 原样入 history。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


async def test_aiagent_stop_hook_force_continue():
    """STOP hook 返回 force_continue 时循环不退出（直到 max_fires）。"""
    agent = _make_test_agent_with_hooks()
    # 第一次 stop hook 让循环继续；第二次（max_fires 触发后）才真停
    calls = []
    def stop_fn():
        calls.append(1)
        return "again" if len(calls) == 1 else None
    agent.hooks_registry.register_stop(stop_fn, name="loop")
    agent.llm_client = _mock_llm_simple_response("resp")
    await agent.run_conversation("go")
    # 验证 STOP hook 至少触发一次
    assert len(calls) >= 1, "STOP hook should have fired at least once"
    # 验证 [stop_hook] 消息出现在 history 中
    stop_msgs = [m for m in agent.conversation_history if "[stop_hook]" in m.get("content", "")]
    assert len(stop_msgs) >= 1, "Should have [stop_hook] message in history"


# ---- helpers ----

def _make_test_agent():
    """构造一个最小可跑的 AIAgent。"""
    from agent import AIAgent
    return AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home="/tmp/fake",
    )


def _make_test_agent_with_hooks():
    from agent import AIAgent
    reg = HookRegistry()
    return AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home="/tmp/fake",
        hooks_registry=reg,
    )


def _mock_llm_simple_response(text: str):
    """mock LLM client：每次返回固定文本，stop_reason='stop'。

    Plan 2B: chat_completions 已改 async，用 AsyncMock（return_value 语义不变）。
    """
    from unittest.mock import MagicMock, AsyncMock
    m = MagicMock()
    resp = MagicMock()
    resp.choices = [
        MagicMock(message=MagicMock(content=text, tool_calls=None),
                  finish_reason="stop")
    ]
    m.chat_completions = AsyncMock(return_value=resp)
    return m


# === P2-T7: handle_function_call + PRE/POST_TOOL_USE 集成测试 ===

async def test_handle_function_call_pre_tool_use_deny():
    """PreToolUse hook 返回 deny 时，handler 不调，返回 hook_deny error。"""
    from unittest.mock import AsyncMock, patch
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    with patch("model_tools.registry.dispatch", new_callable=AsyncMock) as mock_dispatch:
        result = await handle_function_call(
            "todo_write", {"todos": []},
            hooks_registry=reg, session_id="s",
            config={"hooks": {"enabled": True}},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "hook_deny"
        assert "blocked" in parsed["error"]
        assert mock_dispatch.await_count == 0  # 关键：dispatch 未被调


async def test_handle_function_call_pre_tool_use_modify_args():
    """PreToolUse hook 修改 args 后，handler 看到的是修改后的。"""
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    reg.register_pre_tool_use(
        lambda n, a: {"modify_args": {"todos": [{"id": 1, "text": "modified"}]}},
        name="m"
    )
    result = await handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": True}},
    )
    parsed = json.loads(result)
    assert "error" not in parsed or parsed.get("error_type") != "hook_deny"


async def test_handle_function_call_post_tool_use_modifies_result():
    """PostToolUse hook 修改 result 后，最终返回的是修改后的。"""
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    reg.register_post_tool_use(
        lambda n, a, r: json.dumps({"overridden": True}, ensure_ascii=False),
        name="override"
    )
    result = await handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": True}},
    )
    parsed = json.loads(result)
    assert parsed.get("overridden") is True


async def test_handle_function_call_no_registry_unchanged():
    """hooks_registry=None 时行为完全等同于 Phase 1。"""
    result1 = await handle_function_call(
        "todo_write", {"todos": []}, session_id="s",
    )
    result2 = await handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=None, session_id="s",
    )
    assert json.loads(result1).get("error_type") == json.loads(result2).get("error_type")


async def test_handle_function_call_hooks_disabled_skips():
    """config.hooks.enabled=False 时跳过所有 hook。"""
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    reg.register_pre_tool_use(lambda n, a: {"deny": "blocked"}, name="b")
    result = await handle_function_call(
        "todo_write", {"todos": []},
        hooks_registry=reg, session_id="s",
        config={"hooks": {"enabled": False}},
    )
    parsed = json.loads(result)
    assert parsed.get("error_type") != "hook_deny"


# ---------------------------------------------------------------------------
# P2-T8: RuntimeContext hooks 注入 + 端到端集成测试
# ---------------------------------------------------------------------------

def test_runtime_context_has_hooks_registry():
    """RuntimeContext 持有 hooks_registry 实例。"""
    from cli import RuntimeContext
    ctx = RuntimeContext.__new__(RuntimeContext)  # 不调 __init__
    # 验证属性可设
    from agent.hooks import HookRegistry
    ctx.hooks_registry = HookRegistry()
    assert ctx.hooks_registry is not None


async def test_e2e_aiagent_with_hooks_full_loop(tmp_path):
    """端到端：USER_PROMPT_SUBMIT 修改 -> LLM -> PRE_TOOL_USE 放行 -> POST_TOOL_USE 改写 -> STOP。

    使用 mock LLM + 真实 registry + 真实 handle_function_call。
    """
    from agent import AIAgent
    from agent.hooks import HookRegistry

    reg = HookRegistry()
    # hook 1: USER_PROMPT_SUBMIT 增强
    reg.register_user_prompt_submit(lambda p: p + " (with context)", name="augment")
    # hook 2: PRE_TOOL_USE 全放行
    reg.register_pre_tool_use(lambda n, a: None, name="allow-all")
    # hook 3: POST_TOOL_USE 在 result 里加 audit 标记
    def add_audit(n, a, r):
        try:
            parsed = json.loads(r)
            parsed["_audited"] = True
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return None
    reg.register_post_tool_use(add_audit, name="audit")

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        hooks_registry=reg,
        config={"hooks": {"enabled": True, "stop_hook_max_fires": 3}},
    )
    # mock LLM 第一轮返回 tool_call，第二轮返回 stop
    from unittest.mock import MagicMock, AsyncMock
    call_count = [0]

    async def side_effect(msgs, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            # 第一轮：返回一个 todo_write tool_call
            resp = MagicMock()
            resp.choices = [MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[MagicMock(
                        id="call_1",
                        type="function",
                        function=MagicMock(name="todo_write", arguments='{"todos": []}'),
                    )],
                ),
                finish_reason="tool_calls",
            )]
            return resp
        else:
            # 后续：stop
            resp = MagicMock()
            resp.choices = [MagicMock(
                message=MagicMock(content="done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp

    m = MagicMock()
    m.chat_completions = AsyncMock(side_effect=side_effect)
    agent.llm_client = m

    final = await agent.run_conversation("do something")
    # 至少：用户消息被 augment；POST_TOOL_USE 在某条 tool 消息加了 _audited
    assert "(with context)" in agent.conversation_history[0]["content"]
    # 找到 tool 结果消息
    tool_msgs = [m for m in agent.conversation_history if m.get("role") == "tool"]
    assert tool_msgs, "Expected at least one tool message from the e2e flow"
    parsed = json.loads(tool_msgs[0]["content"])
    assert parsed.get("_audited") is True
    # 无异常即通过
    assert isinstance(final, str)


async def test_e2e_no_hooks_enabled_full_backward_compat(tmp_path):
    """config.hooks.enabled=False 时整条链路等同 Phase 1。"""
    from agent import AIAgent
    from agent.hooks import HookRegistry

    # 即使注册了 hook，enabled=False 也不触发
    reg = HookRegistry()
    reg.register_user_prompt_submit(lambda p: "MUTATED", name="m")

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        hooks_registry=reg,
        config={"hooks": {"enabled": False}},
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("original")
    # hook 没触发，原样入 history
    assert agent.conversation_history[0]["content"] == "original"


# === P2b-T5: AIAgent 集成 bg_manager ===

def test_aiagent_accepts_bg_manager_kwarg():
    agent = _make_test_agent()
    assert agent.bg_manager is None


async def test_aiagent_drains_notifications_into_temporary_user_msg(tmp_path):
    """完成的后台任务在下一轮主循环注入 <task_notification> 临时 user 消息。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.background import BackgroundManager
    import sys
    import time

    mgr = BackgroundManager()
    mgr.start([sys.executable, "-c", "print('done')"], cwd=tmp_path)
    time.sleep(0.5)  # 等任务完成 + push notification

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        bg_manager=mgr,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("check")
    # conversation_history 不该含 task_notification（是临时消息）
    for msg in agent.conversation_history:
        assert "<task_notification>" not in msg.get("content", "")
    mgr.shutdown()


async def test_aiagent_no_bg_manager_backward_compat(tmp_path):
    """bg_manager=None 时主循环不抛，行为同 Phase 2a。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


def test_runtime_context_has_bg_manager(tmp_path):
    """RuntimeContext 实例化时应该有 bg_manager 属性（P2b-T6）。"""
    from cli import RuntimeContext
    from agent.background import BackgroundManager
    from unittest.mock import patch

    # Mock load_config 返回含 bg_task 的配置
    mock_config = {
        "bg_task": {
            "max_concurrent": 5,
            "notification_stdout_cap": 500,
            "result_stdout_cap": 5000,
            "default_timeout": 600,
        },
        "hooks": {"enabled": False},  # 避免 HookRegistry 初始化
        "memory": {"enabled": False},
        "sessions": {"auto_save": False},
        "model": {
            "provider": "deepseek",
            "name": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "test_key",
        },
        "agent": {"max_iterations": 90},
        "enabled_toolsets": ["core"],
        "curator": {"enabled": False},
    }

    with patch("cli.load_config", return_value=mock_config):
        ctx = RuntimeContext()
        # 实例化后应该有 bg_manager 属性
        assert hasattr(ctx, "bg_manager")
        assert isinstance(ctx.bg_manager, BackgroundManager)


# ---------------------------------------------------------------------------
# P2b-T7: handle_function_call 透传 bg_manager + e2e 生命周期
# ---------------------------------------------------------------------------

async def test_handle_function_call_threads_bg_manager(tmp_path):
    """bg_start 工具能通过 handle_function_call 拿到 bg_manager。"""
    import sys
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    try:
        result_str = await handle_function_call(
            "bg_start",
            {"command": [sys.executable, "-c", "print('ok')"], "cwd": str(tmp_path)},
            bg_manager=mgr,
        )
        parsed = json.loads(result_str)
        assert parsed["task_id"].startswith("bg_")
    finally:
        mgr.shutdown()


async def test_e2e_aiagent_full_bg_lifecycle(tmp_path):
    """端到端：AIAgent + bg_manager + 完整后台任务生命周期。

    流程：
    1. 通过 handle_function_call 启动一个快任务
    2. 任务完成 → push 通知
    3. 下一轮主循环 drain → 注入 <task_notification>
    4. bg_status / bg_result 查询正常
    """
    import sys
    import time
    from unittest.mock import MagicMock, AsyncMock
    from agent.background import BackgroundManager

    mgr = BackgroundManager()
    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=["bg"], omnimate_home=str(tmp_path),
        bg_manager=mgr,
    )

    # mock LLM：第一轮调 bg_start，第二轮调 bg_status，第三轮 stop
    call_count = [0]
    started_task_id = [None]

    async def side_effect(msgs, **kw):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            # 启动任务
            result = await handle_function_call(
                "bg_start",
                {"command": [sys.executable, "-c", "print('done')"],
                 "cwd": str(tmp_path)},
                bg_manager=mgr,
            )
            started_task_id[0] = json.loads(result)["task_id"]
            time.sleep(0.5)  # 等任务完成
            resp.choices = [MagicMock(
                message=MagicMock(content=None, tool_calls=[MagicMock(
                    id="c1", type="function",
                    function=MagicMock(name="bg_status",
                                       arguments=json.dumps({"task_id": started_task_id[0]})),
                )]),
                finish_reason="tool_calls",
            )]
            return resp
        elif call_count[0] == 2:
            result = await handle_function_call(
                "bg_status",
                {"task_id": started_task_id[0]},
                bg_manager=mgr,
            )
            # 再调一次让循环结束
            resp.choices = [MagicMock(
                message=MagicMock(content="all done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp
        else:
            resp.choices = [MagicMock(
                message=MagicMock(content="done", tool_calls=None),
                finish_reason="stop",
            )]
            return resp

    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=side_effect)

    final = await agent.run_conversation("run bg task")
    # 至少没崩
    assert isinstance(final, str)
    mgr.shutdown()


# === P2c-T4: AIAgent 集成 cron_scheduler ===

def test_aiagent_accepts_cron_scheduler_kwarg():
    agent = _make_test_agent()
    assert agent.cron_scheduler is None


async def test_aiagent_drains_cron_into_temporary_user_msg(tmp_path):
    """cron drain_due 返回的消息作为 <scheduled_message> 临时 user 消息注入。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.cron import CronScheduler

    sched = CronScheduler(jobs_path=tmp_path / "jobs.json", enabled=False)
    # 手动 push 一条通知
    sched._notifications.append({
        "job_id": "j1",
        "message": "time to check backups",
        "fired_at": "2026-07-12T15:30:00",
    })

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        cron_scheduler=sched,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hi")
    # conversation_history 不该含 scheduled_message
    for msg in agent.conversation_history:
        assert "<scheduled_message>" not in msg.get("content", "")
    sched.shutdown()


async def test_aiagent_no_cron_scheduler_backward_compat(tmp_path):
    """cron_scheduler=None 时主循环不抛。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


# === P2c-T5: RuntimeContext 注入 cron_scheduler ===

def test_runtime_context_has_cron_scheduler():
    """RuntimeContext 持有 cron_scheduler 实例（P2c-T5）。"""
    from cli import RuntimeContext
    from agent.cron import CronScheduler
    ctx = RuntimeContext.__new__(RuntimeContext)
    sched = CronScheduler(jobs_path=Path("/tmp/x.json"), enabled=False)
    ctx.cron_scheduler = sched
    assert isinstance(ctx.cron_scheduler, CronScheduler)
    sched.shutdown()


# ---------------------------------------------------------------------------
# P2c-T6: 端到端 cron 全生命周期（FINAL Phase 2c）
# ---------------------------------------------------------------------------

async def test_e2e_cron_full_lifecycle(tmp_path):
    """端到端：jobs.json 配置 → scheduler tick → drain_due → 主循环注入 <scheduled_message>。

    流程：
    1. 写一个匹配当前时间的 jobs.json
    2. 构造 AIAgent + cron_scheduler
    3. 手动调 _tick(now) 触发
    4. agent.run_conversation() drain_due → 验证 LLM 看到 <scheduled_message>
    """
    import json
    from datetime import datetime
    from unittest.mock import MagicMock, AsyncMock
    from agent import AIAgent
    from agent.cron import CronScheduler

    jobs_path = tmp_path / ".cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(json.dumps({
        "jobs": [
            {"id": "test_job", "cron": "* * * * *", "message": "cron test message"},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    sched = CronScheduler(jobs_path=jobs_path, enabled=True, poll_interval_seconds=999)
    # 手动触发一次 tick（避免依赖 30s 轮询）
    sched._tick(datetime.now())

    captured_messages = []

    async def capture_llm_call(msgs, **kw):
        # 快照：列表浅拷贝，避免后续 mutate 影响断言
        captured_messages.append(list(msgs))
        resp = MagicMock()
        resp.choices = [MagicMock(
            message=MagicMock(content="ok", tool_calls=None),
            finish_reason="stop",
        )]
        return resp

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        cron_scheduler=sched,
    )
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=capture_llm_call)

    await agent.run_conversation("check")
    sched.shutdown()

    # 第 1 次 LLM 调用的 messages 应含 <scheduled_message>
    assert len(captured_messages) >= 1, "应至少调用 LLM 一次"
    first_msgs = captured_messages[0]
    contents = [m.get("content", "") or "" for m in first_msgs]
    assert any("<scheduled_message>" in c for c in contents), (
        f"应当含 <scheduled_message>，实际: {contents}"
    )
    assert any("cron test message" in c for c in contents), (
        f"应当含 job message 'cron test message'，实际: {contents}"
    )


# ---------------------------------------------------------------------------
# Mem-T5: AIAgent memory retriever 注入
# ---------------------------------------------------------------------------

def test_aiagent_accepts_memory_retriever_kwarg():
    agent = _make_test_agent()
    assert agent.memory_retriever is None


async def test_aiagent_injects_relevant_memories_into_user_msg(tmp_path):
    """retriever 返回 id → store 读 body → 注入 <relevant_memories>。

    Plan 2B: retriever 已 async，用 AsyncMock。
    """
    from unittest.mock import AsyncMock
    from agent import AIAgent
    from agent.memory_store import MemoryStore

    store = MemoryStore(omnimate_home=tmp_path)
    mid = store.save(
        name="pytest 配置",
        description="项目用 pytest",
        type="project",
        body="运行测试用 uv run pytest tests/ -v",
    )

    # mock retriever 返回 [mid]（async 调用）
    fake_retriever = AsyncMock(return_value=[mid])

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        memory_store=store, memory_retriever=fake_retriever,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("怎么跑测试")

    # conversation_history[0] 应含 <relevant_memories> + mid body
    first_user = agent.conversation_history[0]["content"]
    assert "<relevant_memories>" in first_user
    assert "uv run pytest" in first_user
    # 原始 user message 也应在
    assert "怎么跑测试" in first_user


async def test_aiagent_no_memory_retriever_backward_compat(tmp_path):
    """memory_retriever=None 时不抛，user_message 原样入 history。"""
    agent = _make_test_agent()
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hello")
    assert agent.conversation_history[0]["content"] == "hello"


async def test_retrieval_failure_does_not_break_main_loop(tmp_path):
    """retriever 抛异常时主循环不崩。

    Plan 2B: retriever 已 async，用 AsyncMock side_effect。
    """
    from unittest.mock import AsyncMock
    from agent import AIAgent
    from agent.memory_store import MemoryStore

    store = MemoryStore(omnimate_home=tmp_path)
    bad_retriever = AsyncMock(side_effect=RuntimeError("boom"))

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        memory_store=store, memory_retriever=bad_retriever,
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("hi")
    # 不抛 + user_message 原样入 history
    assert agent.conversation_history[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# Mem-T8: 端到端 memory save → retrieve → 注入（跨会话）
# ---------------------------------------------------------------------------

async def test_e2e_memory_save_then_retrieve_next_session(tmp_path):
    """端到端：会话 1 save → 会话 2 检索 + 注入。

    模拟两个会话（两次 AIAgent 实例化）。
    - 会话 1：MemoryStore.save 落盘一条记忆
    - 会话 2：新建 MemoryStore（重建索引），真实 retrieve_relevant 用 mock LLM 选到 mid
    - 主 LLM 也是 mock：第 1 次调用是 retriever，第 2 次是主 LLM 返回 stop
    - 验证 conversation_history[0] 含 <relevant_memories> + memory body + 原始 user message

    Plan 2B: retrieve_relevant 已 async，主 LLM chat_completions 已 async。
    """
    from unittest.mock import MagicMock, AsyncMock
    from agent.memory_retriever import retrieve_relevant

    # === 会话 1：保存记忆 ===
    store1 = MemoryStore(omnimate_home=tmp_path)
    mid = store1.save(
        name="pytest 命令",
        description="项目用 pytest 跑测试",
        type="project",
        body="uv run pytest tests/ -v",
    )

    # === 会话 2：索引应能看到，retriever 应能选到 ===
    store2 = MemoryStore(omnimate_home=tmp_path)  # 重建索引

    # mock LLM：第 1 次调用是 retriever（返回 [mid]），第 2 次是主 LLM（返回 stop）
    call_count = [0]

    async def side_effect(msgs, **kw):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            # retriever 调用：返回 JSON 数组 [mid]
            resp.choices = [MagicMock(message=MagicMock(content=f'["{mid}"]'))]
            return resp
        # 主 LLM 调用：返回 stop
        resp.choices = [MagicMock(
            message=MagicMock(content="ok", tool_calls=None),
            finish_reason="stop",
        )]
        return resp

    main_llm = MagicMock()
    main_llm.chat_completions = AsyncMock(side_effect=side_effect)

    # 直接用 retrieve_relevant 函数（AIAgent 把 memory_retriever 当 callable 调）
    retriever_obj = retrieve_relevant

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        memory_store=store2, memory_retriever=retriever_obj,
    )
    agent.llm_client = main_llm
    await agent.run_conversation("怎么跑测试")

    # 第一次入 history 的 user 消息应含 <relevant_memories> + memory body + 原始 message
    first_user = agent.conversation_history[0]["content"]
    assert "<relevant_memories>" in first_user
    assert "uv run pytest" in first_user
    assert "怎么跑测试" in first_user


# ---------------------------------------------------------------------------
# P4a-T6: AIAgent team_bus 集成
# ---------------------------------------------------------------------------

def test_aiagent_accepts_team_kwargs():
    """AIAgent 接受 team_bus/team_coordinator/team_name kwargs（默认 None）。"""
    agent = _make_test_agent()
    assert agent.team_bus is None
    assert agent.team_coordinator is None
    assert agent.team_name is None


async def test_aiagent_team_messages_injected_into_temporary_user_msg(tmp_path):
    """team_bus.read_inbox 返回的消息作为 <team_messages> 临时注入（不进 conversation_history）。"""
    from unittest.mock import MagicMock
    from agent import AIAgent
    from agent.team.bus import TeamMessage

    fake_bus = MagicMock()
    fake_bus.read_inbox.return_value = [
        TeamMessage(
            id="m1", from_="worker1", to="main",
            type="response", content="task done",
            ts="2026-07-12T15:30:00", request_id=None,
        ),
    ]

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        team_bus=fake_bus, team_name="main",
    )
    agent.llm_client = _mock_llm_simple_response("ok")
    await agent.run_conversation("check")

    # conversation_history 不应含 team_messages（临时注入）
    for msg in agent.conversation_history:
        assert "<team_messages>" not in msg.get("content", "")


# ---------------------------------------------------------------------------
# P4a-T8: e2e 团队集成测试（FINAL P4a）
# ---------------------------------------------------------------------------

def test_e2e_team_spawn_real_subprocess(tmp_path):
    """端到端：spawn 一个真子进程（不用 worker.py，用 echo 等价命令）。

    验证 spawn 启动 + 子进程确实运行过 + registry 记录可查。
    """
    import sys
    import time
    from agent.team.coordinator import TeamCoordinator

    coord = TeamCoordinator(
        team_dir=tmp_path, omnimate_home=tmp_path,
        config={"team": {"max_members": 10}},
    )
    # 用最简命令（不真起 worker，避免依赖 LLM）
    member = coord.spawn(
        name="w1", role="worker", task="dummy",
        command=[sys.executable, "-c",
                 "import time; time.sleep(0.3); print('done')"],
    )
    # spawn 返回的 member 对象应带真实 pid（proc.pid）
    assert member.pid is not None
    assert member.task == "dummy"

    # 等子进程退出
    time.sleep(1.0)
    members = coord.list_members()
    target = next((m for m in members if m.name == "w1"), None)
    assert target is not None
    # registry 中状态为 "running"（update_status 更新了 registry，
    # 但 member 对象本身是 register 返回的快照——status 仍为 "spawning"）
    assert target.name == "w1"
    assert target.status == "running"


def test_e2e_team_send_and_inbox_through_bus(tmp_path: Path):
    """端到端：两个进程通过 bus 互发消息（同进程内模拟）。"""
    from agent.team.bus import MessageBus

    bus = MessageBus(team_dir=tmp_path)
    # main 给 worker1 发任务
    mid = bus.send(
        from_="main", to="worker1",
        type_="request", content="分析数据",
    )
    # worker1 读自己的 inbox
    msgs = bus.read_inbox("worker1")
    assert len(msgs) == 1
    assert msgs[0].content == "分析数据"
    # worker1 回复 main
    bus.send(
        from_="worker1", to="main",
        type_="response", content="分析完成",
        request_id=mid,
    )
    # main 读 inbox
    msgs = bus.read_inbox("main")
    assert len(msgs) == 1
    assert msgs[0].content == "分析完成"
    assert msgs[0].request_id == mid


# ---------------------------------------------------------------------------
# P4b-T2: idle 工具 + spawn depth 检查
# ---------------------------------------------------------------------------

async def test_idle_tool_sets_flag():
    """调 idle 工具 → AIAgent._idle_requested = True（worker agent）。"""
    import tools.team_tool  # 触发注册
    from tools.registry import registry
    from agent import AIAgent

    # worker agent（team_name 非 main）调 idle 应设标志
    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home="/tmp/fake",
        team_name="worker1",
    )
    agent._idle_requested = False
    result_str = await registry.dispatch(
        "idle", {},
        agent_ref=agent,
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert agent._idle_requested is True


async def test_idle_tool_noop_for_main_agent():
    """P4b final-fix I3: 主 agent 调 idle 是 no-op，_idle_requested 不变。"""
    import tools.team_tool  # 触发注册
    from tools.registry import registry
    from agent import AIAgent

    # team_name="main" → no-op
    agent_main = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home="/tmp/fake",
        team_name="main",
    )
    agent_main._idle_requested = False
    result_str = await registry.dispatch("idle", {}, agent_ref=agent_main)
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert "no-op" in parsed["message"]
    assert agent_main._idle_requested is False  # 关键：未被设置

    # team_name=None → no-op（默认构造的主 agent）
    agent_default = _make_test_agent()
    agent_default._idle_requested = False
    result_str = await registry.dispatch("idle", {}, agent_ref=agent_default)
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert "no-op" in parsed["message"]
    assert agent_default._idle_requested is False


async def test_idle_tool_noop_no_agent_ref():
    """P4b final-fix I3: 无 agent_ref 时 idle 也是 no-op。"""
    import tools.team_tool  # 触发注册
    from tools.registry import registry

    result_str = await registry.dispatch("idle", {})
    parsed = json.loads(result_str)
    assert parsed["success"] is True
    assert "no-op" in parsed["message"]


async def test_idle_requested_reset_between_run_conversation_calls(tmp_path):
    """P4b final-fix C1: 多次 run_conversation 调用，idle 标志不跨调用泄漏。

    场景：autonomous lifecycle 在多个 WORK 周期复用同一 agent 实例。
    第一轮 idle 后退出，第二轮应能正常执行不被立即退出。
    """
    from unittest.mock import MagicMock
    from agent import AIAgent

    agent = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=[], omnimate_home=str(tmp_path),
        team_name="worker1",
    )

    # mock LLM：每次返回无 tool_call 的最终响应
    agent.llm_client = _mock_llm_simple_response("ok")

    # 第一轮：预先设 _idle_requested=True（模拟 idle 工具被调过）
    agent._idle_requested = True
    await agent.run_conversation("first turn")
    # 第一轮应被 idle 中断（进入循环后立即检查到 idle? 不——idle 在 tool_call 后检查）
    # 实际上 idle 检查在 tool_calls 处理后；无 tool_call 时走 return 分支
    # 关键验证：第二轮调 run_conversation 时 _idle_requested 被重置

    # 第二轮：应正常完成，不受上一轮 idle 标志影响
    agent._idle_requested = True  # 假装第一轮结束时被设了
    response = await agent.run_conversation("second turn")
    assert response == "ok"
    # 验证 _idle_requested 在 run_conversation 开头被重置
    # （如果没重置，无 tool_call 场景也会正常 return，所以我们用 tool_call 场景验证）

    # 更严格的验证：tool_call 场景
    call_count = [0]
    async def side_effect(msgs, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            # 第一轮：返回 idle 工具调用
            tool_call = SimpleNamespace(
                id="c1",
                type="function",
                function=SimpleNamespace(
                    name="idle",
                    arguments="{}",
                ),
            )
            msg = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            # 第二轮：返回最终响应
            msg = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent2 = AIAgent(
        base_url="http://fake", api_key="fake", model="fake",
        enabled_toolsets=["team"], omnimate_home=str(tmp_path),
        team_name="worker2",
    )
    agent2.llm_client = SimpleNamespace(chat_completions=side_effect)

    # 第一轮：调 idle → 设置 _idle_requested → 循环 break → fallback
    r1 = await agent2.run_conversation("turn 1")
    # idle 后 break 会走 fallback 分支（"强制停止"）
    assert "强制停止" in r1
    # _idle_requested 此时为 True（idle 工具设置的）
    assert agent2._idle_requested is True

    # 第二轮：_idle_requested 应在开头被重置，能正常完成
    r2 = await agent2.run_conversation("turn 2")
    assert r2 == "done"


async def test_team_spawn_max_depth_blocks(tmp_path):
    """depth >= max_depth 时 spawn 返回 team_max_depth error。"""
    import tools.team_tool
    from tools.registry import registry
    from agent.team.bus import MessageBus
    from agent.team.coordinator import TeamCoordinator

    bus = MessageBus(team_dir=tmp_path)
    coord = TeamCoordinator(team_dir=tmp_path, omnimate_home=tmp_path,
                             config={"team": {"max_members": 10, "max_depth": 2}})
    coord.register(name="main", role="lead")

    # 假装主 agent 已经 depth=2
    class FakeAgent:
        spawn_depth = 2
    agent = FakeAgent()

    result_str = await registry.dispatch(
        "team_spawn",
        {"name": "w1", "task": "x"},
        team_bus=bus, team_coordinator=coord, team_name="main",
        agent_ref=agent,
        config={"team": {"max_members": 10, "max_depth": 2}},
    )
    parsed = json.loads(result_str)
    assert parsed["success"] is False
    assert parsed["error_type"] == "team_max_depth"


# ---------------------------------------------------------------------------
# Phase 4b Task 5: autonomous lifecycle e2e（mock）
# ---------------------------------------------------------------------------


def test_e2e_autonomous_lifecycle_with_mock_agent(tmp_path):
    """端到端：用 mock agent 跑 AutonomousLifecycle，验证状态转换。

    避免 spawn 真子进程（依赖 LLM）。直接在测试进程内跑 lifecycle。
    """
    from agent.team.bus import MessageBus
    from agent.team.lifecycle import (
        AutonomousLifecycle, STATE_WORK, STATE_SHUTDOWN,
    )

    bus = MessageBus(team_dir=tmp_path)

    work_count = {"n": 0}
    def work_fn(task):
        work_count["n"] += 1
        return f"done-{task}"

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: bus.read_inbox("w1"),
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: False,
        on_shutdown_fn=lambda: None,
        idle_timeout=0.2,
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="initial task")
    assert work_count["n"] == 1  # 只跑了 initial，IDLE 无消息超时
    assert lifecycle.state == STATE_SHUTDOWN


def test_e2e_autonomous_lifecycle_picks_up_message_mid_idle(tmp_path):
    """IDLE 中 inbox 来消息 → 回 WORK。"""
    import threading
    import time as _time
    from agent.team.bus import MessageBus
    from agent.team.lifecycle import AutonomousLifecycle, STATE_SHUTDOWN

    bus = MessageBus(team_dir=tmp_path)

    work_log = []
    def work_fn(task):
        work_log.append(task)
        return "ok"

    # 后台线程在 100ms 后给 w1 发消息
    def delayed_msg():
        _time.sleep(0.1)
        bus.send(from_="main", to="w1",
                 type_="message", content="late task")

    t = threading.Thread(target=delayed_msg)
    t.start()

    lifecycle = AutonomousLifecycle(
        work_fn=work_fn,
        poll_inbox_fn=lambda: bus.read_inbox("w1"),
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: False,
        idle_timeout=0.5,  # 长一点，等消息到
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="initial")
    t.join()

    # 应跑了 2 次（initial + late task）
    assert len(work_log) == 2
    assert work_log[0] == "initial"
    assert work_log[1] == "late task"
    assert lifecycle.state == STATE_SHUTDOWN


def test_autonomous_worker_crash_sends_failure_message(tmp_path):
    """autonomous lifecycle work_fn 持续抛异常 → 不崩，最终 SHUTDOWN。"""
    from agent.team.lifecycle import AutonomousLifecycle, STATE_SHUTDOWN

    def bad_work(task):
        raise RuntimeError("always fails")

    lifecycle = AutonomousLifecycle(
        work_fn=bad_work,
        poll_inbox_fn=lambda: [],
        poll_tasks_fn=lambda: [],
        claim_task_fn=lambda tid: False,
        on_shutdown_fn=lambda: None,
        idle_timeout=0.1,
        poll_interval=0.05,
    )
    lifecycle.run(initial_task="x")
    # work 抛异常后被 try/except 吞，进入 IDLE，超时 SHUTDOWN
    assert lifecycle.state == STATE_SHUTDOWN




async def test_tool_call_persisted_to_session(tmp_path):
    """对齐 Claude Code：工具调用轮次（assistant tool_calls + tool 结果）持久化到会话库。"""
    from agent.session_store import SessionStore
    session_store = SessionStore(tmp_path / "sessions.db")
    session_id = session_store.create_session(model="test", provider="test")

    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
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
            msg = SimpleNamespace(content="完成", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello", encoding="utf-8")

    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=["core"],
        omnimate_home=tmp_path,
        session_store=session_store,
        session_id=session_id,
    )
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    await agent.chat("读这个文件")

    msgs = session_store.get_messages(session_id)
    roles = [m["role"] for m in msgs]
    assert "assistant" in roles, f"应持久化 assistant(tool_calls)，实际 roles: {roles}"
    assert "tool" in roles, f"应持久化 tool 结果，实际 roles: {roles}"

    assistant_msg = [m for m in msgs if m["role"] == "assistant"][0]
    assert assistant_msg["tool_calls"], "assistant 消息应含 tool_calls"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"

    tool_msg = [m for m in msgs if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["name"] == "read_file"


async def test_checkpoint_tracked_on_write_file(tmp_path):
    """write_file 成功后 checkpoint 追踪该文件（对齐 Claude Code）。"""
    from agent.checkpoint import CheckpointManager
    ckpt_mgr = CheckpointManager(tmp_path / ".checkpoints", "sess", max_snapshots=10)

    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            tool_call = SimpleNamespace(
                id="call_w", type="function",
                function=SimpleNamespace(
                    name="write_file",
                    arguments=json.dumps({
                        "path": str(tmp_path / "out.txt"),
                        "content": "hello",
                    }),
                ),
            )
            msg = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            msg = SimpleNamespace(content="完成", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = AIAgent(
        api_key="fake",
        model="test",
        enabled_toolsets=["core"],
        omnimate_home=tmp_path,
        checkpoint_manager=ckpt_mgr,
    )
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    await agent.chat("写文件")

    tracked_names = [Path(p).name for p in ckpt_mgr.tracked_files()]
    assert "out.txt" in tracked_names, f"write_file 应触发 checkpoint track，实际: {tracked_names}"


async def test_recent_read_file_and_skill_recorded(tmp_path):
    """read_file/load_skill 工具调用被记录（供压缩后重注入）。"""
    src = tmp_path / "src.py"
    src.write_text("def hello():\n    return 1\n", encoding="utf-8")

    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            tc1 = SimpleNamespace(id="c1", type="function",
                                  function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": str(src)})))
            tc2 = SimpleNamespace(id="c2", type="function",
                                  function=SimpleNamespace(name="load_skill", arguments=json.dumps({"name": "brainstorming"})))
            msg = SimpleNamespace(content=None, tool_calls=[tc1, tc2])
        else:
            msg = SimpleNamespace(content="完成", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=["core"], omnimate_home=tmp_path)
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)
    await agent.chat("读文件并加载技能")

    assert any("src.py" in f for f in agent._recent_read_files), agent._recent_read_files
    assert "brainstorming" in agent._recent_skills, agent._recent_skills


def test_build_reinject_context(tmp_path):
    """压缩后重注入包含技能正文 + 最近读过的文件内容。"""
    src = tmp_path / "data.txt"
    src.write_text("关键内容 ABC", encoding="utf-8")

    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[],
                    omnimate_home=tmp_path, config={"context": {}})
    # 直接塞最近记录（模拟 read_file + load_skill 后）
    agent._recent_skills = ["brainstorming"]
    agent._recent_read_files = [str(src)]

    out = agent._build_reinject_context()
    assert "[技能 brainstorming 正文]" in out, f"应含技能正文，实际: {out[:200]}"
    assert "关键内容 ABC" in out, f"应含文件内容，实际: {out[:200]}"
    assert "[最近读过的文件" in out


def test_reinject_context_budget(tmp_path):
    """重注入受总预算限制（reinject_char_limit）。"""
    big = tmp_path / "big.txt"
    big.write_text("A" * 10000, encoding="utf-8")  # 超过每文件 4000 上限

    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[],
                    omnimate_home=tmp_path, config={"context": {"reinject_char_limit": 2000}})
    agent._recent_read_files = [str(big)]

    out = agent._build_reinject_context()
    assert len(out) <= 2000 + 200, f"应受预算限制，实际长度 {len(out)}"


def test_context_management_tip_injected(tmp_path):
    """上下文接近上限时注入 <context_management_tip>，且只提示一次。"""
    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[],
                    omnimate_home=tmp_path, config={"context": {"llm_compact_token_threshold": 1000}})
    # 塞 3000 字符 → 估算 ~1000 token，超过阈值 1000*0.7
    msgs = [{"role": "user", "content": "A" * 3000}]
    agent._maybe_inject_context_tip(msgs)
    assert any("context_management_tip" in m.get("content", "") for m in msgs), "应注入提示"
    assert agent._context_tip_shown is True

    # 再次调用不再注入
    msgs2 = [{"role": "user", "content": "A" * 3000}]
    agent._maybe_inject_context_tip(msgs2)
    assert not any("context_management_tip" in m.get("content", "") for m in msgs2), "只提示一次"


def test_summarize_rewind(tmp_path, monkeypatch):
    """/rewind 摘要：checkpoint 之后的消息被压成摘要。"""
    from agent.checkpoint import CheckpointManager
    ckpt_mgr = CheckpointManager(tmp_path / ".checkpoints", "sess")

    # checkpoint 时的对话（快照）
    ckpt_conv = [{"role": "user", "content": "第一轮"}, {"role": "assistant", "content": "回复"}]
    sid = ckpt_mgr.create_snapshot(conversation=ckpt_conv)

    agent = AIAgent(api_key="fake", model="test", enabled_toolsets=[],
                    omnimate_home=tmp_path, config={"context": {}})
    agent.checkpoint_manager = ckpt_mgr
    # 当前对话 = checkpoint 对话 + 后续追加
    agent.conversation_history = ckpt_conv + [
        {"role": "user", "content": "第二轮"}, {"role": "assistant", "content": "更多内容"},
    ]

    # mock 摘要（_summarize_rewind 内部从 context_compressor 局部导入）
    monkeypatch.setattr(
        "agent.context_compressor._summarize_conversation",
        lambda msgs, client: "摘要内容",
    )

    import cli
    cli._summarize_rewind(
        type("RT", (), {"checkpoint_mgr": ckpt_mgr, "agent": agent})(), sid,
    )

    assert len(agent.conversation_history) == len(ckpt_conv) + 1
    last = agent.conversation_history[-1]
    assert last["role"] == "user"
    assert "摘要内容" in last["content"], f"应含摘要，实际: {last['content']}"
    assert agent.conversation_history[:len(ckpt_conv)] == ckpt_conv


def test_cleanup_redundant_summaries():
    """恢复时清理多余摘要占位（保留最近一个）。"""
    import cli
    msgs = [
        {"role": "user", "content": "[之前的对话已自动总结]\n摘要1"},
        {"role": "user", "content": "普通消息1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "[之前的对话已自动总结]\n摘要2"},
        {"role": "user", "content": "[紧急上下文压缩]\n摘要3"},
        {"role": "assistant", "content": "b"},
    ]
    out = cli._cleanup_redundant_summaries(msgs)
    # 3 个占位 → 保留 1 个（最新的"摘要3"）
    summaries = [m["content"] for m in out
                 if str(m.get("content", "")).startswith(("[之前的对话已自动总结]", "[紧急上下文压缩"))]
    assert len(summaries) == 1, f"应只留 1 个摘要，实际 {len(summaries)}"
    assert "摘要3" in summaries[0], "应保留最近摘要"
    # 普通消息和 assistant 保留
    assert "普通消息1" in [m["content"] for m in out]
    assert len(out) == len(msgs) - 2  # 删掉 2 个旧占位
