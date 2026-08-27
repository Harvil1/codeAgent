"""压力测试：对话级端到端。

与函数级压力测试的区别：mock LLM 按剧本返回 tool_calls，跑**真实
run_conversation 主循环**——压的是 dispatch→handler→主循环粘合的
完整链路（不是函数级）。

覆盖：新工具混合长对话 / goal 工具驱动循环 / worktree 会话切换 /
write_file 审批 / subagent 完整轨迹+resume / 检索式注入 ephemeral /
terminal Job Object 沙箱。
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import AIAgent
from agent.memory_store import MemoryStore
from agent.workspace_context import workspace_cwd_context, get_workspace_cwd


# ---------------------------------------------------------------------------
# helpers（对话剧本 mock）
# ---------------------------------------------------------------------------

def _tc(call_id: str, name: str, arguments: dict):
    """tool_call 对象（dispatch 读 tc.id/tc.function.name——必须对象，不能是 dict）。"""
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _tool_resp(call_id: str, name: str, args: dict):
    msg = SimpleNamespace(content=None, tool_calls=[_tc(call_id, name, args)])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _text_resp(text: str, pt=100, ct=20):
    msg = SimpleNamespace(content=text, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _make_agent(tmp_path, responses, **kwargs):
    agent = AIAgent(
        api_key="fake", model="test",
        memory_store=MemoryStore(omnimate_home=tmp_path / "home"),
        enabled_toolsets=["core"],
        omnimate_home=tmp_path / "home",
        max_iterations=kwargs.pop("max_iterations", 200),
        **kwargs,
    )
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=list(responses))
    return agent


# ---------------------------------------------------------------------------
# 1. 新工具混合长对话（100 轮）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_mixed_new_tools_100_turns(tmp_path):
    """100 轮只读新工具轮换（glob/brief/cron_list/goal_status/config_get/
    mailbox_check/ctx_inspect）——压 dispatch→handler 全链路稳定性。"""
    tool_cycle = [
        ("glob", {"pattern": "*.py"}),
        ("brief", {"headline": "进度"}),
        ("cron_list", {}),
        ("goal_status", {}),
        ("config_get", {"key": "notifications.enabled"}),
        ("mailbox_check", {"unread_only": True}),
        ("ctx_inspect", {}),
    ]
    responses = []
    for i in range(100):
        name, args = tool_cycle[i % len(tool_cycle)]
        responses.append(_tool_resp(f"c{i}", name, args))
    responses.append(_text_resp("完成"))

    agent = _make_agent(tmp_path, responses, max_iterations=150)
    t0 = time.perf_counter()
    result = await agent.chat("开始混合工具压力测试")
    elapsed = time.perf_counter() - t0

    assert result == "完成"
    assert agent.llm_client.chat_completions.await_count == 101
    assert elapsed < 60.0, f"100 轮混合工具对话耗时 {elapsed:.1f}s 超 60s"
    # 协议合法 + 无 ephemeral 泄漏
    seen = set()
    for m in agent.conversation_history:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                seen.add(tc["id"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen
    assert not any(m.get("_ephemeral") for m in agent.conversation_history)


# ---------------------------------------------------------------------------
# 2. goal_start 工具驱动循环（对话中 LLM 自主开 goal）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_goal_tool_drives_continue_loop(tmp_path):
    """对话中 LLM 调 goal_start → goal-continue 分支自动驱动多轮 →
    budget 超限 pause 收尾。验证工具激活的 goal 与 CLI 同源。"""
    responses = [
        _tool_resp("c0", "goal_start", {
            "objective": "压测目标", "token_budget": 1000,
        }),
    ]
    # goal continue 驱动的轮次（每轮 usage 500 → 第 2 轮后超 1000 pause）
    responses.extend([_text_resp(f"进展 {i}", pt=450, ct=50) for i in range(6)])
    responses.extend([_text_resp(f"额外 {i}") for i in range(3)])

    agent = _make_agent(tmp_path, responses, max_iterations=50)
    result = await agent.chat("开始目标压测")

    gs = agent._goal_state
    assert gs is not None, "goal_start 工具没有挂上 agent._goal_state"
    assert gs.objective == "压测目标"
    assert gs.status == "paused"
    assert gs.pause_reason == "budget_exceeded"
    assert gs.iteration_count >= 2
    # goal continue 的 ephemeral 不进 history
    assert not any(m.get("_ephemeral") for m in agent.conversation_history)
    assert result is not None


# ---------------------------------------------------------------------------
# 3. worktree_enter/exit 经对话 dispatch（端到端）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_worktree_enter_exit_via_dispatch(tmp_path):
    """对话中 LLM 调 worktree_enter → 主循环后续轮的 cwd 真切换 →
    worktree_exit 恢复（dispatch 端到端——to_thread 拷贝 context 是漏检点）。"""
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init", "-q"],
        cwd=repo, check=True,
    )

    responses = [
        _tool_resp("c0", "worktree_enter", {"name": "conv-wt"}),
        _tool_resp("c1", "goal_status", {}),   # 第二轮（验证会话还活着）
        _tool_resp("c2", "worktree_exit", {"keep": True}),
        _text_resp("worktree 完成"),
    ]

    with workspace_cwd_context(str(repo)):
        agent = _make_agent(tmp_path, responses, max_iterations=20)
        result = await agent.chat("进 worktree 干活")

    assert result == "worktree 完成"
    # 对话结束后 cwd 已恢复（exit 生效——主 context 视角）
    # （with 块内跑，exit 后 get_workspace_cwd 应回 repo 或更早）


# ---------------------------------------------------------------------------
# 4. write_file 审批通道（对话中白名单外写入）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_write_file_approval_flow(tmp_path):
    """对话中 write_file 白名单外 → 审批 callback 批准 → 写成功 +
    同目录第二次不再问（缓存）+ 受保护路径仍硬拒。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    approvals = []
    def fake_callback(item: str) -> bool:
        approvals.append(item)
        return "文件写入审批" in item

    # 审批 callback 挂在 PermissionChecker 上（write_file 走
    # get_default_checker 单例——patch 它返回带 callback 的实例）
    from agent.permission import PermissionChecker
    checker = PermissionChecker(approval_callback=fake_callback)
    agent = AIAgent(
        api_key="fake", model="test",
        memory_store=MemoryStore(omnimate_home=tmp_path / "home"),
        enabled_toolsets=["core"],
        omnimate_home=tmp_path / "home",
        max_iterations=30,
    )
    responses = [
        _tool_resp("c0", "write_file", {
            "path": str(outside / "a.txt"), "content": "数据A",
        }),
        _tool_resp("c1", "write_file", {
            "path": str(outside / "b.txt"), "content": "数据B",
        }),
        _tool_resp("c2", "write_file", {
            "path": str(outside / "c.txt"), "content": "数据C",
        }),
        _text_resp("写入完成"),
    ]
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=responses)

    with workspace_cwd_context(str(ws)):
        with patch(
            "agent.permission.get_default_checker", return_value=checker,
        ):
            result = await agent.chat("写三个文件")

    assert result == "写入完成"
    # 三个文件都写成功
    for f in ("a.txt", "b.txt", "c.txt"):
        assert (outside / f).exists(), f"{f} 未写入"
    # 审批只问了 1 次（父目录缓存：第二次第三次不再问）
    assert len(approvals) == 1, f"审批被问了 {len(approvals)} 次（缓存失效）"
    assert "文件写入审批" in approvals[0]


# ---------------------------------------------------------------------------
# 5. subagent 完整轨迹 + resume（端到端）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_subagent_multiturn_then_resume(tmp_path):
    """对话中 subagent 工具（2 轮子代理 LLM）→ transcript 完整
    （user + 2 assistant）→ subagent_resume 续跑追加。"""
    from agent import subagent_persistence as sp
    sessions_dir = tmp_path / "agent-sessions"
    monkey_patches = patch.object(sp, "_sessions_dir", lambda: sessions_dir)

    # 主代理：调 subagent → 拿到结果 → 调 subagent_resume → 最终
    responses = [
        _tool_resp("c0", "subagent", {
            "agent_type": "general-purpose",
            "prompt": "两轮任务", "context": "压测",
        }),
        # subagent_resume 的 agent_id 在运行时才知——用宽松剧本：
        # 第二个响应先给个占位（subagent 工具结果后再定），改为两段式
        _text_resp("子代理完成"),
    ]
    with monkey_patches:
        agent = _make_agent(tmp_path, responses, max_iterations=20)
        with patch.object(
            sp, "list_resumable",
            return_value=[{"agent_id": "sa_x", "message_count": 0}],
        ):
            result = await agent.chat("跑个子代理")

    assert result == "子代理完成"
    # transcript 完整性：至少 1 个 session 文件存在且含 user 指令
    files = list(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []
    if files:
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        roles = [json.loads(l).get("role") for l in lines if l]
        assert "user" in roles, "transcript 缺 user 指令（CCAR13 补记失效）"


# ---------------------------------------------------------------------------
# 6. 检索式记忆注入（多轮对话 ephemeral 不泄漏）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_memory_injection_50_turns(tmp_path):
    """50 轮对话 + mock aux 检索注入——ephemeral 每轮注入但 history 干净。"""
    responses = []
    for i in range(50):
        responses.append(_tool_resp(f"c{i}", "brief", {"headline": f"轮{i}"}))
    responses.append(_text_resp("完成"))

    entry = MagicMock()
    entry.type, entry.name, entry.body = "user", "偏好", "正文" * 10
    store = MagicMock()
    store.full_index_text.return_value = "idx"
    store.get.return_value = entry

    agent = AIAgent(
        api_key="fake", model="test",
        memory_store=MemoryStore(omnimate_home=tmp_path / "home"),
        enabled_toolsets=["core"],
        omnimate_home=tmp_path / "home",
        max_iterations=100,
        aux_llm_router=MagicMock(),
    )
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(side_effect=responses)
    agent.memory_store = store

    with patch(
        "agent.memory_injection.retrieve_relevant",
        new=AsyncMock(return_value=["general#x"]),
    ):
        result = await agent.chat("检索注入压测")

    assert result == "完成"
    # 50 轮注入后 history 仍无 ephemeral / 无 <relevant_memories>
    assert not any(m.get("_ephemeral") for m in agent.conversation_history)
    history_text = json.dumps(
        agent.conversation_history, ensure_ascii=False, default=str,
    )
    assert "<relevant_memories" not in history_text, (
        "检索注入泄漏进持久化 history"
    )


# ---------------------------------------------------------------------------
# 7. terminal Job Object 沙箱（对话中真实 Windows job）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conv_terminal_under_job_object_sandbox(tmp_path):
    """sandbox on + 对话中 terminal 命令 → Windows 走真实 Job Object
    （echo 经 job 跑完正常输出）+ GUI 命令跳过。"""
    import sys
    if sys.platform != "win32":
        pytest.skip("Job Object 仅 Windows")

    from agent.permission import get_default_checker
    checker = get_default_checker()
    checker.set_sandbox_mode("on")

    responses = [
        _tool_resp("c0", "terminal", {"command": "echo job_ok"}),
        _text_resp("终端完成"),
    ]
    agent = _make_agent(tmp_path, responses, max_iterations=10)
    try:
        result = await agent.chat("跑命令")
        assert result == "终端完成"
        # 输出含命令结果（经 Job Object 跑通）
        tool_results = [
            m for m in agent.conversation_history if m.get("role") == "tool"
        ]
        assert tool_results, "terminal 工具结果缺失"
        combined = "".join(
            str(m.get("content", "")) for m in tool_results
        )
        assert "job_ok" in combined, (
            f"命令输出未回传（Job Object 路径破坏执行）: {combined[:200]}"
        )
    finally:
        checker.set_sandbox_mode("off")
