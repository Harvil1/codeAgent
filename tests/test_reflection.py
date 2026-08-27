"""任务级反思引擎测试（CCALS-P0-2）。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.reflection import (
    extract_trajectory,
    run_reflection,
    apply_reflection,
    REFLECTION_PROMPT_TEMPLATE,
)


# ============ extract_trajectory ============

def test_extract_trajectory_empty_returns_empty():
    assert extract_trajectory([]) == ""


def test_extract_trajectory_keeps_recent_messages():
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好啊"},
        {"role": "user", "content": "帮我做事"},
    ]
    out = extract_trajectory(msgs)
    assert "你好" in out
    assert "帮我做事" in out
    assert "user:" in out
    assert "assistant:" in out


def test_extract_trajectory_truncates_long_content():
    """单条消息超 300 字符时被截断。"""
    long = "x" * 1000
    msgs = [{"role": "user", "content": long}]
    out = extract_trajectory(msgs)
    # 截断到 300 字符（不含 role 前缀）
    assert "x" * 300 in out
    assert "x" * 301 not in out


def test_extract_trajectory_marks_tool_calls():
    """工具调用消息特别标记。"""
    msgs = [
        {"role": "user", "content": "do search"},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
        },
    ]
    out = extract_trajectory(msgs)
    assert "调用工具: search" in out


def test_extract_trajectory_respects_max_chars():
    """总长超 max_chars 时只保留尾部最近的。"""
    msgs = [
        {"role": "user", "content": "old" * 100},
        {"role": "user", "content": "recent"},
    ]
    out = extract_trajectory(msgs, max_chars=50)
    assert "recent" in out
    # old 被截掉
    assert "old" * 100 not in out


# ============ run_reflection ============

def _mock_llm_with_insights(insights_list):
    """构造返回指定 insights JSON 的 mock LLM。

    生产代码 chat_completions 已改 async（Task D4），
    run_reflection 用 asyncio.run 驱动 → mock 必须用 AsyncMock。
    """
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(insights_list, ensure_ascii=False)),
        )],
    ))
    return client


def test_run_reflection_extracts_valid_insights():
    """LLM 返回规范 JSON 时提取出 insights。"""
    client = _mock_llm_with_insights([
        {
            "type": "user", "name": "偏好简洁",
            "description": "≤3 句话", "summary": "用户偏好简短回复",
            "body": "详细说明",
        },
        {
            "type": "feedback", "name": "delegate 并行",
            "description": "并行委托更快",
            "summary": "用 delegate_task 批量并行",
            "body": "",
        },
    ])
    result = run_reflection(
        messages=[{"role": "user", "content": "hi"}],
        llm_client=client,
    )
    assert len(result) == 2
    assert result[0]["type"] == "user"
    assert result[0]["name"] == "偏好简洁"
    assert result[1]["type"] == "feedback"


def test_run_reflection_filters_invalid_type():
    """非法 type（如 'other'）的条目被过滤。"""
    client = _mock_llm_with_insights([
        {"type": "user", "name": "ok", "description": "ok", "summary": "", "body": ""},
        {"type": "other", "name": "bad", "description": "bad"},  # 过滤
        {"type": "garbage", "name": "bad2", "description": "bad2"},  # 过滤
    ])
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert len(result) == 1
    assert result[0]["type"] == "user"


def test_run_reflection_filters_missing_required_fields():
    """缺 name 或 description 的条目被过滤。"""
    client = _mock_llm_with_insights([
        {"type": "user", "description": "no name"},  # 缺 name
        {"type": "user", "name": "no desc"},  # 缺 description
        {"type": "user", "name": "ok", "description": "ok"},  # OK
    ])
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert len(result) == 1


def test_run_reflection_handles_empty_list():
    """LLM 返回 [] 时（没值得记的）返回空 list。"""
    client = _mock_llm_with_insights([])
    result = run_reflection(
        messages=[{"role": "user", "content": "hi"}], llm_client=client,
    )
    assert result == []


def test_run_reflection_handles_llm_failure():
    """LLM 调用抛异常时 fail-open 返回空 list。"""
    client = MagicMock()
    client.chat_completions = AsyncMock(side_effect=RuntimeError("LLM down"))
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert result == []


def test_run_reflection_handles_malformed_json():
    """LLM 返回非合法 JSON 时尝试 [...] 提取，仍失败则 fail-open。"""
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="garbage text not json at all"),
        )],
    ))
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert result == []


def test_run_reflection_handles_json_with_surrounding_text():
    """LLM 在 JSON 前后有多余文本时仍能提取。"""
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    "好的，我来分析：\n"
                    '[{"type":"user","name":"ok","description":"ok","summary":"","body":""}]\n'
                    "以上就是分析结果"
                ),
            ),
        )],
    ))
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert len(result) == 1
    assert result[0]["name"] == "ok"


def test_run_reflection_truncates_overlong_fields():
    """LLM 输出超长字段被截断到安全长度。"""
    client = _mock_llm_with_insights([
        {
            "type": "user",
            "name": "x" * 200,  # 截到 60
            "description": "y" * 200,  # 截到 80
            "summary": "z" * 500,  # 截到 200
            "body": "",
        },
    ])
    result = run_reflection(
        messages=[{"role": "user", "content": "x"}], llm_client=client,
    )
    assert len(result[0]["name"]) <= 60
    assert len(result[0]["description"]) <= 80
    assert len(result[0]["summary"]) <= 200


def test_run_reflection_empty_trajectory_short_circuits():
    """空轨迹时不调 LLM。"""
    client = MagicMock()
    result = run_reflection(messages=[], llm_client=client)
    assert result == []
    client.chat_completions.assert_not_called()


def test_run_reflection_awaits_async_chat_completions():
    """回归 Task D4 follow-up: chat_completions 是 async coroutine，
    run_reflection 必须用 asyncio.run 驱动（而不是裸调用拿 coroutine 对象）。

    如果生产代码退回同步调用，AsyncMock 会拿到一个未 await 的 coroutine，
    `response.choices` → AttributeError on coroutine → fail-open 返回 []。
    本测试用 AsyncMock 校验 await 行为：
      - chat_completions 被 await 过（assert_awaited）
      - 拿到的是真实响应（不是 coroutine），所以能解析出 insights
    """
    client = MagicMock()
    client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps([
                {
                    "type": "user", "name": "async 回归",
                    "description": "验证 await", "summary": "", "body": "",
                },
            ], ensure_ascii=False)),
        )],
    ))
    result = run_reflection(
        messages=[{"role": "user", "content": "hi"}], llm_client=client,
    )
    # 关键断言：chat_completions 被 await 过（不是只调用没 await）
    client.chat_completions.assert_awaited_once()
    # 真实响应能被解析（不是 coroutine 导致 fail-open 返回 []）
    assert len(result) == 1
    assert result[0]["name"] == "async 回归"


# ============ apply_reflection ============

def test_apply_reflection_writes_new_insights(tmp_path):
    """新 insights 被写入 memory_store。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    client = _mock_llm_with_insights([
        {
            "type": "user", "name": "偏好简洁",
            "description": "简短回复",
            "summary": "用户偏好≤3 句话回复",
            "body": "详细说明",
        },
    ])
    count = apply_reflection(
        messages=[{"role": "user", "content": "hi"}],
        memory_store=store, llm_client=client,
    )
    assert count == 1
    entries = store.list_all()
    assert len(entries) == 1
    assert entries[0].name == "偏好简洁"
    assert entries[0].summary == "用户偏好≤3 句话回复"


def test_apply_reflection_dedupes_existing_insights(tmp_path):
    """同 type+name 已存在时跳过（避免重复堆积）。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    # 预置一条
    store.save(
        name="偏好简洁", description="旧版",
        type="user", body="...",
    )
    client = _mock_llm_with_insights([
        {
            "type": "user", "name": "偏好简洁",  # 同 type+name
            "description": "新版", "summary": "新", "body": "",
        },
        {
            "type": "feedback", "name": "新经验",
            "description": "新", "summary": "", "body": "",
        },
    ])
    count = apply_reflection(
        messages=[{"role": "user", "content": "x"}],
        memory_store=store, llm_client=client,
    )
    # 只写入 feedback 那条（user 被去重）
    assert count == 1
    entries = store.list_all()
    assert len(entries) == 2  # 原来的 + 新的 feedback


def test_apply_reflection_no_store_returns_zero():
    """memory_store=None 时 no-op。"""
    client = _mock_llm_with_insights([
        {"type": "user", "name": "x", "description": "y"},
    ])
    count = apply_reflection(
        messages=[{"role": "user", "content": "x"}],
        memory_store=None, llm_client=client,
    )
    assert count == 0


def test_apply_reflection_no_llm_returns_zero(tmp_path):
    """llm_client=None 时 no-op。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    count = apply_reflection(
        messages=[{"role": "user", "content": "x"}],
        memory_store=store, llm_client=None,
    )
    assert count == 0


def test_apply_reflection_no_insights_returns_zero(tmp_path):
    """LLM 没提炼出经验时不写。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    client = _mock_llm_with_insights([])
    count = apply_reflection(
        messages=[{"role": "user", "content": "x"}],
        memory_store=store, llm_client=client,
    )
    assert count == 0
    assert store.list_all() == []


def test_apply_reflection_dedupes_within_batch(tmp_path):
    """同一批反思里如果 LLM 重复输出同 name，只写一次。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    client = _mock_llm_with_insights([
        {"type": "user", "name": "同名", "description": "1", "summary": "", "body": ""},
        {"type": "user", "name": "同名", "description": "2", "summary": "", "body": ""},
    ])
    count = apply_reflection(
        messages=[{"role": "user", "content": "x"}],
        memory_store=store, llm_client=client,
    )
    assert count == 1  # 批内去重


# ============ Task A1: reference 型记忆 ============

def test_reflection_prompt_includes_reference_type():
    """REFLECTION_PROMPT_TEMPLATE 提到 reference 型（外部系统指针）。"""
    from agent.reflection import REFLECTION_PROMPT_TEMPLATE
    assert "reference" in REFLECTION_PROMPT_TEMPLATE.lower(), \
        "reflection prompt 应包含 reference 型说明"


def test_reflection_valid_types_includes_reference():
    """apply_reflection 接受 type=reference（不再只限 user/feedback/project）。"""
    import inspect
    from agent import reflection
    src = inspect.getsource(reflection)
    # valid_types 集合应含 reference
    assert '"reference"' in src or "'reference'" in src, \
        "reflection 源码 valid_types 应含 reference"


def test_run_reflection_accepts_reference_type():
    """run_reflection 不应过滤 type=reference 的条目。"""
    client = _mock_llm_with_insights([
        {
            "type": "reference",
            "name": "Linear 工单 PROJ-123",
            "description": "后端重构工单",
            "summary": "用户提到 Linear PROJ-123 作为后续上下文",
            "body": "",
        },
    ])
    result = run_reflection(
        messages=[{"role": "user", "content": "看下 PROJ-123"}],
        llm_client=client,
    )
    assert len(result) == 1
    assert result[0]["type"] == "reference"
    assert result[0]["name"] == "Linear 工单 PROJ-123"


# ============ 预注入已有记忆清单（防重复存储） ============


def test_reflection_prompt_includes_existing_manifest(tmp_path):
    """reflection prompt 预注入已有记忆清单（防 tanke 式重复存储）。

    场景：memory_store 已存一条 name="已有条目"，跑 apply_reflection 时
    传给 LLM 的 prompt 必须含"已有条目"字样 + "不要重复"字样。
    """
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    store.save(
        name="已有条目", description="这是一条已存在的记忆描述",
        type="user", body="...",
    )
    captured_prompt = []

    def _capture(messages, **kwargs):
        # messages 是 List[{"role","content"}]，捕第一条 content
        captured_prompt.append(messages[0]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="[]"),
            )],
        )

    client = MagicMock()
    client.chat_completions = AsyncMock(side_effect=_capture)

    apply_reflection(
        messages=[{"role": "user", "content": "hi"}],
        memory_store=store, llm_client=client,
    )
    assert captured_prompt, "应该捕到 prompt"
    p = captured_prompt[0]
    assert "已有条目" in p, "prompt 应含已有记忆的 name"
    assert "不要重复" in p, "prompt 应含明确的防重复字样"


def test_reflection_prompt_manifest_truncates_description(tmp_path):
    """清单中每条 description 截到前 60 字符（防 prompt 过长）。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    long_desc = "A" * 200  # 200 字符
    store.save(
        name="长描述条目", description=long_desc,
        type="project", body="...",
    )
    captured = []

    def _cap(msgs, **kw):
        captured.append(msgs[0]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )

    client = MagicMock()
    client.chat_completions = AsyncMock(side_effect=_cap)
    apply_reflection(
        messages=[{"role": "user", "content": "hi"}],
        memory_store=store, llm_client=client,
    )
    # description 不应完整出现（200 个 A），最多 60 个
    assert "A" * 61 not in captured[0], "description 应被截到 ≤60 字符"


def test_reflection_prompt_manifest_empty_store(tmp_path):
    """空 memory_store 时 prompt 不应含清单段标题（或含但不崩）。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    captured = []

    def _cap(msgs, **kw):
        captured.append(msgs[0]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )

    client = MagicMock()
    client.chat_completions = AsyncMock(side_effect=_cap)
    apply_reflection(
        messages=[{"role": "user", "content": "hi"}],
        memory_store=store, llm_client=client,
    )
    # 空清单：不应出现具体条目但也不应崩
    assert "已有记忆清单" not in captured[0], \
        "空 store 时不应拼出清单段标题（避免误导 LLM）"


def test_reflection_manifest_failopen_on_listall_error(tmp_path):
    """memory_store.list_all() 抛异常时清单为空，reflection 照常跑。"""
    from agent.memory_store import MemoryStore
    store = MemoryStore(omnimate_home=tmp_path)
    # 让 list_all 抛
    store.list_all = MagicMock(side_effect=RuntimeError("disk gone"))
    captured = []

    def _cap(msgs, **kw):
        captured.append(msgs[0]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
        )

    client = MagicMock()
    client.chat_completions = AsyncMock(side_effect=_cap)
    # 不应抛
    count = apply_reflection(
        messages=[{"role": "user", "content": "hi"}],
        memory_store=store, llm_client=client,
    )
    assert count == 0  # LLM 返 [] 自然 0
    # prompt 仍被调
    assert len(captured) == 1
    # 不含清单段标题（fail-open 后清单为空）
    assert "已有记忆清单" not in captured[0]
