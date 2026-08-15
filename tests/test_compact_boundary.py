"""T8（核心机制对齐第 8 项）：压缩边界标注保留段清单。

llm_compact 的 boundary 占位扩为三要素：
1. 压缩时间
2. 摘要覆盖范围（哪些消息被 LLM 转述）
3. 保留段范围（哪些消息原样保留，含工具结果原文）
+ "保留段内容是精确的，摘要内容是转述"提示，帮模型区分两类内容。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _mock_client():
    client = SimpleNamespace()
    client.chat_completions = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary here", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    ))
    return client


def _big_messages(n_pairs: int = 6):
    big = "x" * 50000
    msgs = [{"role": "system", "content": "sys"}]
    for _ in range(n_pairs):
        msgs.append({"role": "user", "content": big})
        msgs.append({"role": "assistant", "content": big})
    msgs.append({"role": "user", "content": "recent"})
    return msgs


async def test_full_mode_boundary_three_elements():
    """全量模式 boundary 含：时间 / 摘要覆盖 / 保留段 + 精确 vs 转述提示。"""
    from agent.context_pipeline import llm_compact

    messages = _big_messages()
    new_msgs, changed = await llm_compact(
        messages, llm_client=_mock_client(), model="m",
        keep_recent=4, token_threshold=100,
    )
    assert changed is True
    # 第一条 conv 消息是 boundary 占位
    boundary = new_msgs[1]["content"]
    assert "[compact_boundary]" in boundary
    assert "压缩时间" in boundary
    assert "摘要覆盖范围" in boundary
    assert "保留段范围" in boundary
    assert "4 条" in boundary  # keep_recent=4 原样保留
    # 精确 vs 转述区分提示
    assert "精确" in boundary and "转述" in boundary
    # 兼容：原提示语保留
    assert "之前的对话已自动总结" in boundary


async def test_partial_mode_boundary_three_elements():
    """partial 模式 boundary 同样含三要素（head + tail 都标注为原文保留）。"""
    from agent.context_pipeline import llm_compact

    messages = _big_messages()
    new_msgs, changed = await llm_compact(
        messages, llm_client=_mock_client(), model="m",
        keep_recent=4, token_threshold=100,
        from_idx=2, up_to_idx=8,
    )
    assert changed is True
    placeholders = [
        m for m in new_msgs if "[compact_boundary]" in str(m.get("content", ""))
    ]
    assert len(placeholders) == 1
    boundary = placeholders[0]["content"]
    assert "压缩时间" in boundary
    assert "摘要覆盖范围" in boundary
    assert "消息 2-8" in boundary  # partial 覆盖范围
    assert "保留段范围" in boundary
    assert "精确" in boundary and "转述" in boundary
