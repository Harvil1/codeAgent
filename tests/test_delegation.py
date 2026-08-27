"""委托系统测试。"""

import json
from unittest.mock import patch

import pytest

from tools.delegate_tool import (
    DelegationCompletionQueue, get_delegation_queue,
    _handle_delegate_task, _build_child_system_prompt,
    _delegate_sync, _delegate_batch, _run_child,
)
from tools.registry import registry


# ---------------------------------------------------------------------------
# DelegationCompletionQueue
# ---------------------------------------------------------------------------

def test_queue_push_drain():
    q = DelegationCompletionQueue()
    q.push({"id": 1, "result": "a"})
    q.push({"id": 2, "result": "b"})
    assert q.has_pending()

    drained = q.drain()
    assert len(drained) == 2
    assert drained[0]["id"] == 1
    assert not q.has_pending()


def test_queue_drain_empty():
    q = DelegationCompletionQueue()
    assert q.drain() == []
    assert q.has_pending() is False


# ---------------------------------------------------------------------------
# _build_child_system_prompt
# ---------------------------------------------------------------------------

def test_child_prompt_contains_goal():
    prompt = _build_child_system_prompt("搜索网页", "用户问 X", "leaf")
    assert "搜索网页" in prompt
    assert "用户问 X" in prompt
    assert "leaf" in prompt


def test_child_prompt_no_context():
    prompt = _build_child_system_prompt("任务", "", "orchestrator")
    assert "任务" in prompt
    assert "orchestrator" in prompt
    # 无 context 时不应该有"来自父代理的上下文"
    assert "来自父代理" not in prompt


def test_child_prompt_leaf_restriction():
    prompt = _build_child_system_prompt("g", "c", "leaf")
    assert "不能再派生" in prompt


# ---------------------------------------------------------------------------
# _handle_delegate_task 参数验证
# ---------------------------------------------------------------------------

async def test_delegate_no_goal_no_tasks():
    """goal 和 tasks 都空时返回错误。"""
    result = await registry.dispatch("delegate_task", {})
    data = json.loads(result)
    assert "error" in data


async def test_delegate_orchestrator_depth_limit(monkeypatch):
    """orchestrator 达到深度上限时拒绝。"""
    result = await registry.dispatch(
        "delegate_task",
        {"goal": "test", "role": "orchestrator"},
        max_spawn_depth=2,
        spawn_depth=2,
    )
    data = json.loads(result)
    assert "error" in data
    assert "嵌套深度" in data["error"]


async def test_delegate_depth_limit_allows_within_range(monkeypatch):
    """orchestrator 在深度范围内允许。"""
    # mock _run_child 避免真创建子代理
    with patch("tools.delegate_tool._run_child", return_value="子代理结果"):
        result = await registry.dispatch(
            "delegate_task",
            {"goal": "test", "role": "orchestrator"},
            max_spawn_depth=2,
            spawn_depth=1,
        )
    data = json.loads(result)
    # 不应该有深度错误
    assert "error" not in data or "嵌套深度" not in data.get("error", "")


# ---------------------------------------------------------------------------
# _delegate_sync
# ---------------------------------------------------------------------------

def test_delegate_sync_success():
    with patch("tools.delegate_tool._run_child", return_value="sync result"):
        result = _delegate_sync("goal", "ctx", "leaf")
    data = json.loads(result)
    assert data["success"] is True
    assert data["result"] == "sync result"
    assert data["mode"] == "sync"


def test_delegate_sync_failure():
    with patch("tools.delegate_tool._run_child", side_effect=RuntimeError("boom")):
        result = _delegate_sync("goal", "ctx", "leaf")
    data = json.loads(result)
    assert data["success"] is False
    assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# _delegate_async
# ---------------------------------------------------------------------------

async def test_delegate_async_returns_immediately():
    """异步委托立即返回，结果进队列。"""
    queue = get_delegation_queue()
    # 清空队列
    queue.drain()

    with patch("tools.delegate_tool._run_child", return_value="async result"):
        result = await registry.dispatch(
            "delegate_task",
            {"goal": "test", "background": True},
        )
        data = json.loads(result)
        assert data["mode"] == "async"
        assert "delegation_id" in data

        # 等后台线程完成（简化：轮询）
        import time
        for _ in range(50):
            if queue.has_pending():
                break
            time.sleep(0.05)

        drained = queue.drain()
        assert len(drained) == 1
        assert drained[0]["success"] is True
        assert drained[0]["result"] == "async result"


# ---------------------------------------------------------------------------
# _delegate_batch
# ---------------------------------------------------------------------------

def test_delegate_batch_parallel():
    """批量委托并行执行。"""
    with patch("tools.delegate_tool._run_child", return_value="batch result") as mock:
        result = _delegate_batch(
            [
                {"goal": "task1"},
                {"goal": "task2"},
                {"goal": "task3"},
            ],
            background=False,
        )
    data = json.loads(result)
    assert data["mode"] == "batch"
    assert len(data["results"]) == 3
    # 每个 task 都成功
    for r in data["results"]:
        assert r["success"] is True
    # _run_child 被调用 3 次
    assert mock.call_count == 3


def test_delegate_batch_handles_failure():
    """批量委托中单个失败不影响其他。"""
    def side_effect(goal, *args, **kwargs):
        if "fail" in goal:
            raise RuntimeError("intentional")
        return "ok"

    with patch("tools.delegate_tool._run_child", side_effect=side_effect):
        result = _delegate_batch(
            [
                {"goal": "ok-1"},
                {"goal": "will-fail"},
                {"goal": "ok-2"},
            ],
            background=False,
        )
    data = json.loads(result)
    successes = [r["success"] for r in data["results"]]
    assert successes.count(True) == 2
    assert successes.count(False) == 1


# ---------------------------------------------------------------------------
# _run_child（需要 mock AIAgent）
# ---------------------------------------------------------------------------

def test_run_child_missing_api_key(monkeypatch):
    """无 API key 时抛错。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # mock config 返回无 key 的配置
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key_env": "NO_SUCH_KEY", "base_url": None},
    }):
        with pytest.raises(RuntimeError, match="API key"):
            _run_child("goal", "", "leaf")


# ---------------------------------------------------------------------------
# E2 NEW: 自定义子代理 .md 定义集成
# ---------------------------------------------------------------------------

def test_build_child_system_prompt_override():
    """override 非空时 base 用 override，但仍追加约束/角色提示。"""
    override_text = "你是代码审查专家，只审 Java。"
    prompt = _build_child_system_prompt(
        "审查 PR", "PR #123", "leaf", override=override_text)
    # override 内容应作为 base
    assert override_text in prompt
    # 默认构建的"你是一个子代理"不应出现
    assert "你是一个子代理" not in prompt
    # 上下文仍要追加
    assert "PR #123" in prompt
    # 约束仍要追加
    assert "独立执行" in prompt
    # leaf 角色提示仍要追加
    assert "不能再派生" in prompt


def test_build_child_system_prompt_override_no_context():
    """override + 空 context 时不应出现上下文段。"""
    prompt = _build_child_system_prompt(
        "g", "", "leaf", override="自定义 system prompt")
    assert "自定义 system prompt" in prompt
    assert "来自父代理" not in prompt


def test_run_child_custom_def_not_found(monkeypatch):
    """自定义子代理名找不到定义时抛 RuntimeError。"""
    from agent.agent_defs import AgentDefinition

    # mock get_agent_def 返回 None
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: None)
    monkeypatch.setattr("agent.agent_defs.scan_agent_defs", lambda: {})

    # mock 掉 LLM 配置加载，让它通过到 _run_child 内部的自定义分支
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with pytest.raises(RuntimeError, match="未找到子代理定义"):
            _run_child("goal", "", "leaf", subagent_type="nonexistent-agent")


def test_run_child_custom_def_loads_config(monkeypatch):
    """自定义子代理名找到定义时，AIAgent 用定义的 model/perm/tools 配置。"""
    from agent.agent_defs import AgentDefinition

    custom_def = AgentDefinition(
        name="explorer",
        description="探索者",
        model="deepseek-chat-custom",
        tools=["core"],
        disallowed_tools=["subagent"],
        permission_mode="bypassPermissions",
        max_turns=42,
        system_prompt="你是探索者。",
    )

    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    # 捕获 AIAgent 构造参数
    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            return "ok"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
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
                        result = _run_child(
                            "goal", "ctx", "leaf",
                            subagent_type="explorer")

    # 验证用定义的 model
    assert captured["model"] == "deepseek-chat-custom"
    # 验证用定义的 permission_mode
    assert captured["permission_mode"] == "bypassPermissions"
    # 验证用定义的 max_iterations
    assert captured["max_iterations"] == 42
    # 验证用定义的 tools
    assert captured["enabled_toolsets"] == ["core"]
    # 验证 disabled_tools 走 config 透传
    assert captured["config"]["disabled_tools"] == ["subagent"]
    # 验证 system_prompt_override 包含 override 文本
    assert "你是探索者" in captured["system_prompt_override"]
    assert result == "ok"


def test_run_child_custom_def_isolation_worktree(monkeypatch):
    """自定义子代理 isolation=worktree 时透传 isolated_workspace=True。

    时序断言：worktree 创建（create_isolated_workspace）必须在 custom_def 加载后、
    AIAgent 构造前发生。修复前 bug：isolated 在 custom_def 写入 kwargs 之前就被读取，
    导致 create_isolated_workspace 永远不被调用。
    """
    from agent.agent_defs import AgentDefinition

    custom_def = AgentDefinition(
        name="isolated-worker",
        description="隔离工作",
        isolation="worktree",
    )

    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            return "done"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    ws_calls = []  # 记录 create_isolated_workspace 调用

    def fake_create_ws(name=None, **kw):
        ws_calls.append({"name": name, "kwargs": kw})
        return ("/tmp/fake_ws", lambda: None)

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
                        with patch(
                            "tools.worktree.create_isolated_workspace",
                            side_effect=fake_create_ws,
                        ):
                            _run_child(
                                "goal", "ctx", "leaf",
                                subagent_type="isolated-worker")

    # 断言 1：create_isolated_workspace 必须被调用一次（这是 bug 的核心症状）
    assert len(ws_calls) == 1, (
        f"expected create_isolated_workspace to be called once for "
        f"isolation=worktree custom def, got {len(ws_calls)} calls"
    )
    # 断言 2：workspace 名字带 goal 前缀（确认走的是 delegate worktree 分支）
    assert ws_calls[0]["name"] is not None
    assert "goal" in ws_calls[0]["name"].lower() or ws_calls[0]["name"] != "workspace"


def test_run_child_general_purpose_unchanged(monkeypatch):
    """general-purpose（默认）行为不受自定义分支影响。"""
    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            return "general result"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
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
                        result = _run_child("goal", "ctx", "leaf")

    # 默认 leaf 用 minimal，无 disabled_tools，无 config 透传
    assert captured["enabled_toolsets"] == ["minimal"]
    assert captured["permission_mode"] == "default"
    assert captured["config"] is None
    assert result == "general result"


# ---------------------------------------------------------------------------
# 回归：非 custom_def 路径下 disabled_tools 透传
# ---------------------------------------------------------------------------

def _make_fake_child(captured: dict):
    """构造 FakeChild，捕获 AIAgent 构造参数（复用模式）。"""
    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")
            self._children = []
            self.conversation_history = []
            self.spawn_depth = kwargs.get("spawn_depth", 0)
            self.effort_level = kwargs.get("effort_level")
            self._stream_callback = None
            self.aux_llm_router = None
            self.hooks_registry = None

        async def chat(self, msg):
            return "ok"
    return FakeChild


def _patch_run_child_env(monkeypatch):
    """打上 _run_child 跑通所需的最小 mock（config/ProgressReporter/hallucination）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    return patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    })


def test_run_child_passes_injected_disabled_tools_without_custom_def(monkeypatch):
    """非 custom_def 路径下，kwargs["config"]["disabled_tools"]
    必须透传到 AIAgent 的 config.disabled_tools。

    场景：_delegate_async 在 kwargs["config"]["disabled_tools"] 注入黑名单，
    但 stype=general-purpose（无 custom_def），若 child_config 恒为 None，
    注入的黑名单完全丢失。
    """
    captured = {}
    FakeChild = _make_fake_child(captured)

    with _patch_run_child_env(monkeypatch):
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
                        _run_child(
                            "goal", "ctx", "leaf",
                            config={"disabled_tools": ["bg_start", "team_spawn"]},
                        )

    # 期望：config 含 disabled_tools（不能为 None）
    assert captured["config"] is not None, (
        "非 custom_def 路径下 child_config 不应为 None（CCAR5 Important 1）"
    )
    disabled = captured["config"].get("disabled_tools") or []
    assert "bg_start" in disabled, "Task F 注入的 bg_start 应透传到 AIAgent"
    assert "team_spawn" in disabled, "Task F 注入的 team_spawn 应透传到 AIAgent"


def test_run_child_merges_custom_def_and_injected_disabled_tools(monkeypatch):
    """custom_def.disallowed_tools 和 kwargs 注入的 disabled_tools
    取并集（保序去重），不互相覆盖。
    """
    from agent.agent_defs import AgentDefinition
    custom_def = AgentDefinition(
        name="merger",
        description="合并测试",
        disallowed_tools=["subagent", "idle"],
    )
    monkeypatch.setattr("agent.agent_defs.get_agent_def", lambda n: custom_def)

    captured = {}
    FakeChild = _make_fake_child(captured)

    with _patch_run_child_env(monkeypatch):
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
                        _run_child(
                            "goal", "ctx", "leaf",
                            subagent_type="merger",
                            config={"disabled_tools": ["bg_start", "subagent"]},
                        )

    # 并集：custom_def（subagent/idle）+ injected（bg_start/subagent）= subagent/idle/bg_start
    disabled = captured["config"].get("disabled_tools") or []
    assert "subagent" in disabled, "custom_def 的 subagent 应保留"
    assert "idle" in disabled, "custom_def 的 idle 应保留"
    assert "bg_start" in disabled, "Task F 注入的 bg_start 应并入"
    # 去重：subagent 在两边都有，只应出现一次
    assert disabled.count("subagent") == 1, "并集去重：subagent 不应重复"


def test_run_child_no_disabled_tools_yields_none_child_config(monkeypatch):
    """回归保护：无任何 disabled_tools 时，child_config 仍为 None
    （不破坏 test_run_child_general_purpose_unchanged 的契约）。
    """
    captured = {}
    FakeChild = _make_fake_child(captured)

    with _patch_run_child_env(monkeypatch):
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
                        _run_child("goal", "ctx", "leaf")

    # 无任何 disabled 时 config 仍为 None（不构造空 dict）
    assert captured["config"] is None


def test_run_child_preserves_other_config_keys_with_disabled(monkeypatch):
    """构造 child_config 时其他 config 键不丢失。"""
    captured = {}
    FakeChild = _make_fake_child(captured)

    with _patch_run_child_env(monkeypatch):
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
                        _run_child(
                            "goal", "ctx", "leaf",
                            config={
                                "disabled_tools": ["bg_start"],
                                "delegation": {"max_concurrent_children": 3},
                                "other_key": "preserved",
                            },
                        )

    cfg = captured["config"]
    assert cfg is not None
    assert cfg.get("disabled_tools") == ["bg_start"], "disabled_tools 应注入"
    assert cfg.get("other_key") == "preserved", "其他键应保留"
    assert cfg.get("delegation", {}).get("max_concurrent_children") == 3


# ---------------------------------------------------------------------------
# 并发子代理 30s 进度摘要 ticker
# ---------------------------------------------------------------------------

class TestProgressTicker:
    """并发子代理运行期间每 interval 秒写一条进度摘要到 scratchpad progress.md。"""

    def test_ticker_writes_progress(self, tmp_path, monkeypatch):
        """≥2 children + ticker 跑一轮 → scratchpad progress.md 有 aux 摘要内容。"""
        from tools.delegate_tool import _start_progress_ticker
        from agent.scratchpad import scratchpad_dir
        import threading, time

        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        state = {"a": {"status": "running", "goal": "扫 agent/"},
                 "b": {"status": "done", "goal": "写方案"}}
        stop = threading.Event()

        class FakeAux:
            async def chat_completions(self, messages, **kw):
                class _Msg:
                    content = "a 在扫，b 已完成"
                class _Choice:
                    message = _Msg()
                class _Resp:
                    choices = [_Choice()]
                return _Resp()

        # 把 sleep 调成 0.05s 加速（ticker 参数 interval 可注入）
        t = _start_progress_ticker(
            state, stop, aux=FakeAux(), session_id="s-test", interval=0.05,
        )
        time.sleep(0.2)
        stop.set()
        t.join(timeout=2)
        p = scratchpad_dir("s-test") / "progress.md"
        assert p.exists()
        assert "a 在扫" in p.read_text(encoding="utf-8")

    def test_no_aux_mechanical_fallback(self, tmp_path, monkeypatch):
        """无 aux → 机械拼接状态行（不调 LLM）。"""
        from tools.delegate_tool import _start_progress_ticker
        from agent.scratchpad import scratchpad_dir
        import threading, time

        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        state = {"a": {"status": "running", "goal": "扫 agent/"},
                 "b": {"status": "done", "goal": "写方案"}}
        stop = threading.Event()

        t = _start_progress_ticker(
            state, stop, aux=None, session_id="s-test", interval=0.05,
        )
        time.sleep(0.2)
        stop.set()
        t.join(timeout=2)
        p = scratchpad_dir("s-test") / "progress.md"
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "a" in content
        assert "running" in content

    def test_batch_starts_and_stops_ticker(self, tmp_path, monkeypatch):
        """_delegate_batch ≥2 任务时启动 ticker；结束时停止 + 状态更新 done。"""
        import threading

        monkeypatch.setenv("OMNIMATE_HOME", str(tmp_path / "home"))
        started = {}

        def fake_ticker(children_state, stop_event, *, aux, session_id,
                        interval=30.0, **kw):
            started["state"] = children_state
            started["stop"] = stop_event
            started["aux"] = aux
            started["session_id"] = session_id
            return threading.Thread(target=lambda: None)

        fake_agent = type(
            "FakeAgent", (), {"session_id": "sess-x", "aux_llm_router": None})()

        with patch("tools.delegate_tool._start_progress_ticker",
                   side_effect=fake_ticker):
            with patch("tools.delegate_tool._run_child", return_value="ok"):
                _delegate_batch(
                    [{"goal": "t1"}, {"goal": "t2"}],
                    background=False, agent_ref=fake_agent,
                )

        # 启动参数：children ≥2、aux/session_id 来自 agent_ref
        assert len(started["state"]) == 2
        assert started["aux"] is None
        assert started["session_id"] == "sess-x"
        # finally 停止 ticker + child 完成时状态更新
        assert started["stop"].is_set()
        statuses = {v["status"] for v in started["state"].values()}
        assert statuses == {"done"}

    def test_batch_single_task_no_ticker(self, tmp_path, monkeypatch):
        """单任务（<2）不起 ticker——没有'并发干等'问题。"""
        with patch("tools.delegate_tool._start_progress_ticker") as mock_ticker:
            with patch("tools.delegate_tool._run_child", return_value="ok"):
                _delegate_batch([{"goal": "only-one"}], background=False)

        mock_ticker.assert_not_called()
