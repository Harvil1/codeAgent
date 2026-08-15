"""T10（核心机制对齐第 10 项）：fork 子代理继承全历史开关。

- subagent fork 字段升级 true | "full"（布尔=最近 N turn 现状；"full"=全部，
  截到 delegation.fork_full_history_max_turns 默认 50）
- fork=True 行为与现状逐字节一致（回归）
- 修既有 bug：schema 的 fork 字段此前从未接进 kwargs（LLM 传了也是死参数）
"""
import inspect


def _parent_stream(n_turns=6):
    """构造 n 轮父对话（user + assistant(tool_calls) + tool + assistant）。"""
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"用户消息{i}"})
        msgs.append({
            "role": "assistant",
            "content": f"助手回复{i}",
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"真实结果{i}"})
        msgs.append({"role": "assistant", "content": f"总结{i}"})
    return msgs


# ---------------------------------------------------------------------------
# build_forked_messages
# ---------------------------------------------------------------------------

def test_full_history_includes_user_messages():
    """full=True → 完整 user/assistant 流（user 消息在场，tool 结果为占位）。"""
    from agent.fork_messages import build_forked_messages

    parent = _parent_stream(4)
    out = build_forked_messages(
        parent, "sys", "directive", max_parent_turns=3, full_history=True,
    )
    user_msgs = [m for m in out if m.get("role") == "user"]
    assert any("用户消息" in str(m.get("content", "")) for m in user_msgs[:-1]), \
        "全量模式应含父 user 消息"
    # tool 结果替换为占位（fork 语义：真实结果不可见）
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs
    assert all("placeholder" in m["content"] or "占位" in m["content"] for m in tool_msgs)
    assert not any("真实结果" in str(m.get("content", "")) for m in tool_msgs)
    # 末尾是 directive
    assert out[-1]["role"] == "user"
    assert "FORK DIRECTIVE" in out[-1]["content"]


def test_full_history_capped_at_max_turns():
    """full=True 时 assistant turn 数截到 full_history_max_turns（默认 50，防失控）。"""
    from agent.fork_messages import build_forked_messages

    parent = _parent_stream(80)  # 160 个 assistant turn
    out = build_forked_messages(
        parent, "sys", "d", max_parent_turns=3,
        full_history=True, full_history_max_turns=50,
    )
    asst = [m for m in out if m.get("role") == "assistant"]
    assert len(asst) == 50
    # 截断从 user 边界开始（首条非 user 消息不应是 assistant 孤儿开头）
    assert out[0].get("role") == "user"


def test_fork_true_regression_byte_identical():
    """fork=True（full_history=False）→ 与现状逐字节一致：只有 assistant turn + 占位 + directive。"""
    from agent.fork_messages import build_forked_messages

    parent = _parent_stream(5)
    out = build_forked_messages(parent, "sys", "directive", max_parent_turns=3)
    # 无父 user 消息（现状语义）
    assert not any(
        "用户消息" in str(m.get("content", "")) for m in out[:-1]
    )
    # 最近 3 个 assistant turn（有 tool_calls 的算一个 turn；无 tool_calls 的总结也算）
    asst = [m for m in out if m.get("role") == "assistant"]
    assert len(asst) == 3
    # 占位 tool_result 跟随 tool_calls
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 1  # 最近 3 个 assistant 消息中只有 1 个含 tool_calls
    assert out[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# 接线
# ---------------------------------------------------------------------------

def test_delegate_wires_fork_arg():
    """_handle_delegate_task 必须把 args.fork 接进 kwargs（修死参数 bug）。"""
    import tools.delegate_tool as dt
    src = inspect.getsource(dt._handle_delegate_task)
    assert 'kwargs["fork"]' in src


def test_config_fork_full_history_max_turns():
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["delegation"]["fork_full_history_max_turns"] == 50
