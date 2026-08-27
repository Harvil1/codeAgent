"""post-compact 主动恢复测试。

覆盖：
1. 单元测试：build_post_compact_brief 各场景
2. 预算控制：per-skill 5K + 总 25K + per-file preview 1K
3. fail-open：异常不崩
4. safe_path：受保护路径跳过
5. config 开关：post_compact_recovery_enabled=False 不注入
6. 端到端：通过 _run_context_compression 触发，验证 brief 真注入到 messages
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_async_tasks():
    """清空全局 _async_tasks（T2 起 recovery 会读它列 running 子代理，
    其他测试残留的 running 条目会污染空状态断言）。"""
    import tools.delegate_tool as _dt
    _dt._async_tasks.clear()
    yield
    _dt._async_tasks.clear()


# ============================================================================
# Helper：构造最小 AIAgent
# ============================================================================

def _make_minimal_agent(**overrides):
    """构造一个最小 mock 的 AIAgent（不连真 LLM）。"""
    from agent import AIAgent
    base = dict(
        base_url="http://localhost",
        api_key="test-key",
        model="test-model",
        enabled_toolsets=["core"],
    )
    base.update(overrides)
    with patch("agent.llm_client.create_llm_client") as mock:
        mock.return_value = MagicMock()
        return AIAgent(**base)


def _make_final_response(text="done"):
    """构造无 tool_calls 的最终响应。"""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = text
    mock_resp.choices[0].message.tool_calls = None
    return mock_resp


# ============================================================================
# 1. 单元测试：build_post_compact_brief 基础场景
# ============================================================================

def test_build_brief_empty_state(tmp_path):
    """空状态（无文件、无技能）→ brief 为空串。"""
    from agent.post_compact_recovery import build_post_compact_brief
    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = []

    result = build_post_compact_brief(agent)
    assert result == "", f"空状态应返回空串，实际: {result!r}"


def test_build_brief_only_files(tmp_path):
    """只有文件时，brief 含文件 preview。"""
    from agent.post_compact_recovery import build_post_compact_brief
    f1 = tmp_path / "a.txt"
    f1.write_text("content A", encoding="utf-8")
    f2 = tmp_path / "b.py"
    f2.write_text("print('hello')", encoding="utf-8")
    f3 = tmp_path / "c.md"
    f3.write_text("# Title", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1), str(f2), str(f3)]
    agent._recent_skills = []

    result = build_post_compact_brief(agent)
    assert "最近文件" in result or "最近读过的文件" in result, f"应含文件标题: {result[:200]}"
    assert "content A" in result, f"应含 f1 内容: {result[:200]}"
    assert "print('hello')" in result
    assert "# Title" in result


def test_build_brief_only_skills(tmp_path):
    """只有 invoked skills 时，brief 含 skill 正文。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = ["brainstorming"]

    result = build_post_compact_brief(agent)
    # brainstorming 是系统内置技能，应能加载到正文
    assert result, f"应非空: {result!r}"
    assert "brainstorming" in result.lower() or "技能" in result


def test_build_brief_mixed_files_and_skills(tmp_path):
    """混合：3 文件 + 2 skills → brief 含两段。"""
    from agent.post_compact_recovery import build_post_compact_brief

    f1 = tmp_path / "data.txt"
    f1.write_text("file content", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = ["brainstorming"]

    result = build_post_compact_brief(agent)
    assert "file content" in result
    # 技能段也在
    assert "brainstorming" in result.lower() or "技能" in result


# ============================================================================
# 2. 预算控制
# ============================================================================

def test_per_skill_budget_5k(tmp_path):
    """单个 skill 正文超 5K → 截到 5K + truncated 标记。"""
    from agent.post_compact_recovery import build_post_compact_brief, SKILL_PER_BUDGET_CHARS

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = ["giant-skill"]

    # mock _load_skill_body 返回 8K 内容
    agent._load_skill_body = lambda name: "X" * 8000

    result = build_post_compact_brief(agent)
    # 单 skill 不应超 5K + 标记
    assert "truncated" in result.lower(), f"应含 truncated 标记: {result[-200:]}"
    # body 部分（去掉标签）不应远超 5K
    assert result.count("X") <= SKILL_PER_BUDGET_CHARS + 100


def test_total_skill_budget_25k(tmp_path):
    """6 个 skill 各 5K = 30K → 截到 5 个（总 ≤25K）。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = [f"skill-{i}" for i in range(6)]

    # 每个 skill body 正好 5K（不超过 per-skill 上限，但总超 25K）
    agent._load_skill_body = lambda name: "Y" * 5000

    result = build_post_compact_brief(agent)
    # 总 Y 数不应超 25000（总 budget）
    total_y = result.count("Y")
    assert total_y <= 25500, f"总 budget 应 ≤25K，实际 Y 数: {total_y}"


def test_recent_file_preview_1k(tmp_path):
    """每个文件 preview ≤1K。"""
    from agent.post_compact_recovery import build_post_compact_brief, RECENT_FILE_PREVIEW_CHARS

    big = tmp_path / "big.txt"
    big.write_text("A" * 10000, encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(big)]
    agent._recent_skills = []

    result = build_post_compact_brief(agent)
    # 预览的 A 数应 ≤ RECENT_FILE_PREVIEW_CHARS
    assert result.count("A") <= RECENT_FILE_PREVIEW_CHARS + 100, (
        f"文件 preview 应 ≤{RECENT_FILE_PREVIEW_CHARS}，实际 A 数: {result.count('A')}"
    )


# ============================================================================
# 3. fail-open
# ============================================================================

def test_build_brief_fail_open_on_exception(tmp_path):
    """_load_skill_body 抛异常时，brief 跳过该 skill 不崩。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = ["bad-skill"]

    # mock 抛异常
    def fake_load(name):
        raise RuntimeError("boom")
    agent._load_skill_body = fake_load

    # 不应抛异常
    result = build_post_compact_brief(agent)
    # 结果是空串（因为 skill 加载失败、文件为空）
    assert result == "", f"fail-open 应返回空串: {result!r}"


def test_build_brief_fail_open_file_read_error(tmp_path):
    """文件读失败时，brief 跳过该文件不崩。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = ["/nonexistent/path/file.txt"]
    agent._recent_skills = []

    result = build_post_compact_brief(agent)
    # 不应崩，可能返回空串或含错误说明
    # 实际：文件不存在 → 跳过 → 空串
    assert isinstance(result, str)


# ============================================================================
# 4. safe_path 集成（受保护路径跳过）
# ============================================================================

def test_protected_path_skipped(tmp_path):
    """受保护路径（~/.ssh 等）的文件被 safe_path 拒绝 → 跳过。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    # /etc/passwd 是受保护路径
    agent._recent_read_files = ["/etc/passwd", "C:\\Windows\\System32\\drivers\\etc\\hosts"]
    agent._recent_skills = []

    result = build_post_compact_brief(agent)
    # 受保护路径应全被跳过 → 空串
    assert result == "", f"受保护路径应跳过: {result!r}"


# ============================================================================
# 5. config 开关
# ============================================================================

def test_config_disabled_no_inject(tmp_path):
    """post_compact_recovery_enabled=False → build_post_compact_brief 返回空。"""
    from agent.post_compact_recovery import build_post_compact_brief

    f1 = tmp_path / "data.txt"
    f1.write_text("content", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = ["brainstorming"]
    # 关闭 recovery
    agent.config = {"context": {"post_compact_recovery_enabled": False}}

    result = build_post_compact_brief(agent)
    assert result == "", f"config 关闭时不应注入: {result!r}"


def test_config_max_files_limit(tmp_path):
    """post_compact_recovery_max_files=2 → 只取最近 2 个文件。"""
    from agent.post_compact_recovery import build_post_compact_brief

    files = []
    for i in range(5):
        f = tmp_path / f"f{i}.txt"
        f.write_text(f"content-{i}", encoding="utf-8")
        files.append(str(f))

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = files
    agent._recent_skills = []
    agent.config = {"context": {"post_compact_recovery_max_files": 2}}

    result = build_post_compact_brief(agent)
    # 只含最近 2 个文件的内容
    assert "content-4" in result
    assert "content-3" in result
    assert "content-2" not in result, f"max_files=2 应排除第 3 个文件"
    assert "content-0" not in result


def test_config_max_skills_limit(tmp_path):
    """post_compact_recovery_max_skills=1 → 只取最近 1 个 skill。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = ["skill-a", "skill-b", "skill-c"]
    agent.config = {"context": {"post_compact_recovery_max_skills": 1}}
    agent._load_skill_body = lambda name: f"body of {name}"

    result = build_post_compact_brief(agent)
    assert "skill-c" in result, "最近 1 个 skill 应在"
    assert "skill-b" not in result, "max_skills=1 应排除 skill-b"
    assert "skill-a" not in result


# ============================================================================
# 6. 端到端测试：通过 _run_context_compression 触发
# ============================================================================

async def test_e2e_recovery_injected_via_run_context_compression(tmp_path):
    """端到端：通过 _run_context_compression 触发 compact，
    验证 post-compact recovery brief 真注入到 messages 末尾。"""
    f1 = tmp_path / "data.txt"
    f1.write_text("important content", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    # 模拟之前 read_file + load_skill 过
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []

    # mock compress_if_needed 返回 changed=True
    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    # 构造 messages（含 system + user）
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    system_prompt = "system prompt"

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress):
        out_msgs, out_sp, compressed = await agent._run_context_compression(
            messages, system_prompt,
        )

    assert compressed is True, "应触发压缩"

    # 验证 brief 在 messages 末尾
    brief_msgs = [
        m for m in out_msgs
        if "<post_compress_brief>" in (m.get("content") or "")
    ]
    assert brief_msgs, "messages 末尾应含 <post_compress_brief>"

    # 验证 brief 含 recovery 内容（文件内容）
    brief_content = brief_msgs[-1]["content"]
    assert "important content" in brief_content, (
        f"brief 应含最近文件内容: {brief_content[:300]}"
    )


async def test_e2e_recovery_not_injected_when_disabled(tmp_path):
    """端到端：config 关闭 recovery 时，brief 不含 recovery 内容。"""
    f1 = tmp_path / "data.txt"
    f1.write_text("important content", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []
    # 关闭 recovery
    agent.config = {"context": {"post_compact_recovery_enabled": False}}

    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress):
        out_msgs, _, compressed = await agent._run_context_compression(
            messages, "system prompt",
        )

    assert compressed is True
    # brief 应在（post_compress_brief 本身还在），但不含 recovery 段
    brief_msgs = [
        m for m in out_msgs
        if "<post_compress_brief>" in (m.get("content") or "")
    ]
    assert brief_msgs, "post_compress_brief 本身应在"
    brief_content = brief_msgs[-1]["content"]
    assert "important content" not in brief_content, (
        "recovery 关闭时不应含文件内容"
    )


async def test_e2e_recovery_not_injected_when_empty_state(tmp_path):
    """端到端：无最近文件、无 invoked skills 时，brief 不含 recovery 段。"""
    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = []

    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress):
        out_msgs, _, compressed = await agent._run_context_compression(
            messages, "system prompt",
        )

    assert compressed is True
    brief_msgs = [
        m for m in out_msgs
        if "<post_compress_brief>" in (m.get("content") or "")
    ]
    assert brief_msgs, "post_compress_brief 应在（基础 brief）"
    brief_content = brief_msgs[-1]["content"]
    # 空状态时不应有 recovery 段
    assert "最近加载的技能" not in brief_content, "空状态不应有 recovery 段"


# ============================================================================
# 7. 端到端：完整 run_conversation 触发链路
# ============================================================================

async def test_e2e_full_run_conversation_with_recovery(tmp_path):
    """完整端到端：run_conversation → _run_context_compression → recovery 注入。
    模拟之前 read_file 过一个文件，压缩后 recovery 段含该文件内容。"""
    f1 = tmp_path / "config.yml"
    f1.write_text("key: value", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    async def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("done")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        await agent.run_conversation("继续工作")

    assert captured_messages_list, "LLM 应被调用"
    last_msgs = captured_messages_list[0]

    # 找到 brief 消息
    brief_msg = next(
        (m for m in last_msgs if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    assert brief_msg is not None, "应注入 post_compress_brief"
    assert "key: value" in brief_msg["content"], (
        f"brief 应含最近文件内容: {brief_msg['content'][:300]}"
    )


# ============================================================================
# 8. 追踪测试：read_file / load_skill 真触发 _record_recent
# ============================================================================

async def test_read_file_triggers_recent_tracking(tmp_path):
    """read_file 工具调用后，路径记录到 _recent_read_files。"""
    import json as json_module
    from types import SimpleNamespace

    src = tmp_path / "src.py"
    src.write_text("def hello(): pass\n", encoding="utf-8")

    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            tc = SimpleNamespace(
                id="c1", type="function",
                function=SimpleNamespace(
                    name="read_file",
                    arguments=json_module.dumps({"path": str(src)}),
                ),
            )
            msg = SimpleNamespace(content=None, tool_calls=[tc])
        else:
            msg = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = _make_minimal_agent(omnimate_home=tmp_path, enabled_toolsets=["core"])
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    await agent.run_conversation("read file")

    assert any("src.py" in p for p in agent._recent_read_files), (
        f"read_file 应触发 _recent_read_files 追踪: {agent._recent_read_files}"
    )


async def test_load_skill_triggers_recent_tracking(tmp_path):
    """load_skill 工具调用后，技能名记录到 _recent_skills。"""
    import json as json_module
    from types import SimpleNamespace

    call_count = [0]

    async def fake_chat_completions(messages, *, tools=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            tc = SimpleNamespace(
                id="c1", type="function",
                function=SimpleNamespace(
                    name="load_skill",
                    arguments=json_module.dumps({"name": "brainstorming"}),
                ),
            )
            msg = SimpleNamespace(content=None, tool_calls=[tc])
        else:
            msg = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    agent = _make_minimal_agent(omnimate_home=tmp_path, enabled_toolsets=["core"])
    agent.llm_client = SimpleNamespace(chat_completions=fake_chat_completions)

    await agent.run_conversation("load skill")

    assert "brainstorming" in agent._recent_skills, (
        f"load_skill 应触发 _recent_skills 追踪: {agent._recent_skills}"
    )


# ============================================================================
# 9. 多轮测试：ephemeral 设计权衡（让设计 explicit）
# ============================================================================

async def test_recovery_content_visible_in_first_round_after_compact(tmp_path):
    """compact 后第 1 轮调 LLM，验证 LLM 收到的 messages 含 recovery brief 内容。

    构造：conversation_history 有历史 → run_conversation 触发 compress →
    压缩后 _run_context_compression 把 brief append 到 messages 末尾 →
    第 1 轮 LLM 能看到文件路径/内容。
    """
    f1 = tmp_path / "session.py"
    f1.write_text("SESSION_SECRET = 'compact_marker_42'", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []

    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    async def fake_call_with_retry(client, messages, **kwargs):
        captured_messages_list.append(list(messages))
        return _make_final_response("done")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        await agent.run_conversation("继续")

    assert captured_messages_list, "LLM 应被调用"
    first_round_msgs = captured_messages_list[0]

    # 第 1 轮的 messages 应含 recovery brief（在 <post_compress_brief> 里）
    brief_msg = next(
        (m for m in first_round_msgs
         if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    assert brief_msg is not None, "第 1 轮 LLM 调用应收到 post_compress_brief"
    assert "compact_marker_42" in brief_msg["content"], (
        f"第 1 轮 brief 应含最近文件内容: {brief_msg['content'][:300]}"
    )


async def test_recovery_content_lost_in_second_round_due_to_ephemeral_design(tmp_path):
    """compact 后第 2 轮组装的 messages 不含 recovery brief（ephemeral 设计）。

    这是 Task B 的已知设计权衡，让 reviewer 提出的 50/50 决策 explicit：

    **设计**：post_compress_brief 是 ephemeral（临时消息），_run_context_compression
    先把摘要后的 messages 同步到 conversation_history（line 1130），**之后**再 append
    brief 到局部变量 messages（line 1174）。brief 从不进 conversation_history。

    **后果**：第 1 轮 LLM 能看到 brief（在局部 messages 里），但第 2 轮通过
    _assemble_turn_messages 重建 messages 时，brief 没了（因为 history 不含它）。

    **为什么不修**：如果把 brief 写入 history，下次 compact 会把 brief 也压缩掉，
    导致摘要不要的内容污染；而且 brief 本质是"刚醒来的提醒"，不该长期驻留。
    （file attachments 式的结构化字段方案同理，都是 ephemeral。）

    本测试锁定这个设计行为：如果未来有人改成把 brief 写入 history，此测试会 FAIL
    提醒他重新评估设计权衡（而不是无意中改变行为）。
    """
    f1 = tmp_path / "lost.py"
    f1.write_text("UNIQUE_EPHEMERAL_MARKER", encoding="utf-8")

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []

    call_count = [0]
    captured_messages_list = []

    def fake_compress(messages, **kwargs):
        return messages, True, True  # 契约：(messages, changed, compacted)

    async def fake_call_with_retry(client, messages, **kwargs):
        call_count[0] += 1
        captured_messages_list.append(list(messages))
        # 第 1 轮返回带 tool_calls，让主循环跑第 2 轮
        if call_count[0] == 1:
            return _make_final_response("done")
        return _make_final_response("done2")

    with patch("agent.context_pipeline.compress_if_needed", side_effect=fake_compress), \
         patch("agent.llm_retry.call_with_retry", side_effect=fake_call_with_retry):
        # 调一次 run_conversation 触发 compact
        await agent.run_conversation("继续")

    # 手动模拟第 2 轮 _assemble_turn_messages（compact 已发生过，history 已重建）
    # 此时再组装一轮，应看不到 brief 内容
    second_round_msgs = agent._assemble_turn_messages(
        "system", {"bg_notifications": [], "cron_messages": [], "team_messages_text": ""},
    )

    second_round_brief = next(
        (m for m in second_round_msgs
         if "<post_compress_brief>" in (m.get("content") or "")),
        None,
    )
    # 设计权衡：第 2 轮组装的 messages 不应含 brief（ephemeral 不进 history）
    if second_round_brief is not None:
        assert "UNIQUE_EPHEMERAL_MARKER" not in second_round_brief["content"], (
            "ephemeral 设计：第 2 轮组装的 messages 不应含第 1 轮的 recovery brief 内容。"
            "如果此测试 FAIL，说明 brief 被写入 conversation_history 了——"
            "请重新评估设计权衡（brief 污染下次 compact 摘要的风险）。"
        )
    # 如果 second_round_brief 是 None（没有 brief 消息），也符合 ephemeral 设计
