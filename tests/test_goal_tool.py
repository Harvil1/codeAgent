"""goal_tool 测试（CCAR12 Task 4）：LLM 可自主管理 goal + 共享函数抽取。

覆盖四层：
1. 共享函数 start_goal_agent（pause 旧 / 建新 / 持久化 / 挂 agent / 路径解析）
2. 工具 handler 行为（mock agent：start/status/pause/resume/clear 全路径 + 错误分支）
3. handler dispatch 契约（args, **kwargs，防 silent-dead-code）+ schema 键契约
4. registry 注册 + 并发分类 + core 可见性
"""
import inspect
import json
from pathlib import Path

import tools.goal_tool  # noqa: 触发注册
from agent.goal import GoalState, goal_persist_path, start_goal_agent
from tools.goal_tool import (
    GOAL_CLEAR_SCHEMA,
    GOAL_PAUSE_SCHEMA,
    GOAL_RESUME_SCHEMA,
    GOAL_START_SCHEMA,
    GOAL_STATUS_SCHEMA,
    _handle_goal_clear,
    _handle_goal_pause,
    _handle_goal_resume,
    _handle_goal_start,
    _handle_goal_status,
)
from tools.registry import registry


class FakeAgent:
    """最小 AIAgent mock：goal 工具读写它的 _goal_state + 持久化路径。"""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.omnimate_home = Path(home)
        self._goal_state = None
        self.conversation_history = []

    def _goal_state_path(self):
        return self.home / ".goal" / "current.json"

    def set_goal_state(self, gs):
        self._goal_state = gs


class BareAgent:
    """无 _goal_state_path / omnimate_home 的极简 mock（测 fallback）。"""

    def __init__(self):
        self._goal_state = None

    def set_goal_state(self, gs):
        self._goal_state = gs


# ---------------------------------------------------------------------------
# 1. 共享函数 start_goal_agent（CLI /goal 与 LLM goal_start 同源）
# ---------------------------------------------------------------------------

class TestStartGoalAgent:
    def test_creates_persists_and_attaches(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        gs = start_goal_agent(agent, "完成测试报告", token_budget=100_000)
        assert agent._goal_state is gs
        assert gs.objective == "完成测试报告"
        assert gs.status == "active"
        assert gs.token_budget_limit == 100_000
        # 持久化落盘
        path = tmp_path / ".goal" / "current.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["objective"] == "完成测试报告"

    def test_pauses_old_active_goal(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        old = start_goal_agent(agent, "旧目标")
        new = start_goal_agent(agent, "新目标")
        # 旧 goal paused + superseded 原因
        assert old.status == "paused"
        assert old.pause_reason == "superseded_by_new_goal"
        # agent 挂的是新 goal
        assert agent._goal_state is new
        assert new.objective == "新目标"
        # 落盘的是新 goal（单文件 current.json 被覆盖）
        data = json.loads(
            (tmp_path / ".goal" / "current.json").read_text(encoding="utf-8")
        )
        assert data["objective"] == "新目标"

    def test_old_paused_goal_not_repaused(self, tmp_path: Path):
        """已 paused 的旧 goal 不重复 pause（notes 不追加）。"""
        agent = FakeAgent(tmp_path)
        old = start_goal_agent(agent, "旧目标")
        old.pause(reason="manual")
        notes_before = len(old.notes)
        start_goal_agent(agent, "新目标")
        assert old.pause_reason == "manual"  # 原因不被覆盖
        assert len(old.notes) == notes_before

    def test_explicit_persist_path(self, tmp_path: Path):
        """CLI 路径：显式 persist_path（rt.home 下的路径）优先于自动解析。"""
        agent = FakeAgent(tmp_path / "agent_home")
        explicit = tmp_path / "cli_home" / ".goal" / "current.json"
        start_goal_agent(agent, "x", token_budget=1000, persist_path=explicit)
        assert explicit.exists()
        # 自动解析路径没被写
        assert not (tmp_path / "agent_home" / ".goal" / "current.json").exists()

    def test_does_not_touch_conversation_history(self, tmp_path: Path):
        """关键回归：共享函数绝不追加 user 消息到 conversation_history。

        工具路径在 assistant(tool_calls) 之后、tool result 回填之前追加
        user 消息会破坏消息历史严格交替（API 400）。CLI 的 [goal_start]
        注入留在 cli 层（会话循环外，安全）。
        """
        agent = FakeAgent(tmp_path)
        agent.conversation_history.append({"role": "user", "content": "hi"})
        len_before = len(agent.conversation_history)
        start_goal_agent(agent, "x")
        assert len(agent.conversation_history) == len_before

    def test_fallback_path_via_omnimate_home(self, tmp_path: Path):
        """agent 无 _goal_state_path 方法时退 omnimate_home 字段。"""

        class NoMethodAgent(FakeAgent):
            _goal_state_path = None  # 类型上不可调

        agent = NoMethodAgent(tmp_path)
        path = goal_persist_path(agent)
        assert path == tmp_path / ".goal" / "current.json"
        gs = start_goal_agent(agent, "x", persist_path=path)
        assert path.exists()
        assert agent._goal_state is gs

    def test_set_goal_state_missing_falls_back_to_attr(self, tmp_path: Path):
        """agent 无 set_goal_state 方法时直接赋 _goal_state 属性。"""
        agent = BareAgent()
        gs = start_goal_agent(
            agent, "x", persist_path=tmp_path / ".goal" / "current.json"
        )
        assert agent._goal_state is gs


# ---------------------------------------------------------------------------
# 2. handler 行为（mock agent）
# ---------------------------------------------------------------------------

class TestGoalStartHandler:
    def test_start_ok(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        result = _handle_goal_start(
            {"objective": "重构模块", "token_budget": 50000},
            agent_ref=agent,
        )
        data = json.loads(result)
        assert data["status"] == "active"
        assert data["objective"] == "重构模块"
        assert data["goal_id"]
        assert data["token_budget_limit"] == 50000
        # agent 已挂 + 落盘
        assert agent._goal_state.objective == "重构模块"
        assert (tmp_path / ".goal" / "current.json").exists()

    def test_start_default_budget(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        data = json.loads(_handle_goal_start({"objective": "x"}, agent_ref=agent))
        assert data["token_budget_limit"] == 200_000

    def test_start_supersedes_old(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        _handle_goal_start({"objective": "旧"}, agent_ref=agent)
        old = agent._goal_state
        _handle_goal_start({"objective": "新"}, agent_ref=agent)
        assert old.status == "paused"
        assert old.pause_reason == "superseded_by_new_goal"
        assert agent._goal_state.objective == "新"

    def test_start_missing_objective(self, tmp_path: Path):
        data = json.loads(
            _handle_goal_start({}, agent_ref=FakeAgent(tmp_path))
        )
        assert data["error_type"] == "invalid_args"

    def test_start_no_agent_ref(self):
        data = json.loads(_handle_goal_start({"objective": "x"}))
        assert data["error_type"] == "not_configured"


class TestGoalStatusHandler:
    def test_status_with_goal(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        gs = GoalState(objective="跑通测试", token_budget_limit=1000)
        gs.iteration_count = 3
        gs.token_budget = 400
        gs.pause(reason="manual")
        agent.set_goal_state(gs)
        data = json.loads(_handle_goal_status({}, agent_ref=agent))
        assert data["objective"] == "跑通测试"
        assert data["status"] == "paused"
        assert data["iteration_count"] == 3
        assert data["token_budget"] == 400
        assert data["pause_reason"] == "manual"
        assert data["token_budget_limit"] == 1000

    def test_status_no_goal(self, tmp_path: Path):
        data = json.loads(_handle_goal_status({}, agent_ref=FakeAgent(tmp_path)))
        assert data["status"] == "none"

    def test_status_no_agent_ref(self):
        data = json.loads(_handle_goal_status({}))
        assert data["error_type"] == "not_configured"


class TestGoalPauseResumeHandlers:
    def test_pause_ok(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        start_goal_agent(agent, "x", persist_path=agent._goal_state_path())
        data = json.loads(_handle_goal_pause({}, agent_ref=agent))
        assert data["status"] == "paused"
        assert data["pause_reason"] == "manual"
        # 落盘同步
        saved = json.loads(
            (tmp_path / ".goal" / "current.json").read_text(encoding="utf-8")
        )
        assert saved["status"] == "paused"

    def test_pause_custom_reason(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        start_goal_agent(agent, "x", persist_path=agent._goal_state_path())
        data = json.loads(
            _handle_goal_pause({"reason": "等用户输入"}, agent_ref=agent)
        )
        assert data["pause_reason"] == "等用户输入"

    def test_pause_no_goal(self, tmp_path: Path):
        data = json.loads(_handle_goal_pause({}, agent_ref=FakeAgent(tmp_path)))
        assert data["error_type"] == "no_active_goal"

    def test_resume_ok(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        gs = start_goal_agent(agent, "x", persist_path=agent._goal_state_path())
        gs.pause(reason="manual")
        data = json.loads(_handle_goal_resume({}, agent_ref=agent))
        assert data["status"] == "active"
        assert agent._goal_state.status == "active"
        assert agent._goal_state.pause_reason is None

    def test_resume_no_goal(self, tmp_path: Path):
        data = json.loads(_handle_goal_resume({}, agent_ref=FakeAgent(tmp_path)))
        assert data["error_type"] == "no_active_goal"


class TestGoalClearHandler:
    def test_clear_ok(self, tmp_path: Path):
        agent = FakeAgent(tmp_path)
        gs = start_goal_agent(agent, "x", persist_path=agent._goal_state_path())
        path = tmp_path / ".goal" / "current.json"
        assert path.exists()
        data = json.loads(_handle_goal_clear({}, agent_ref=agent))
        assert data["cleared"] is True
        assert gs.status == "cancelled"
        assert agent._goal_state is None
        # 持久化文件已删
        assert not path.exists()

    def test_clear_no_goal(self, tmp_path: Path):
        data = json.loads(_handle_goal_clear({}, agent_ref=FakeAgent(tmp_path)))
        assert data["error_type"] == "no_active_goal"


# ---------------------------------------------------------------------------
# 3. dispatch 契约 + schema 键契约（CCAR8/CCAR11 教训）
# ---------------------------------------------------------------------------

_ALL_HANDLERS = (
    _handle_goal_start,
    _handle_goal_status,
    _handle_goal_pause,
    _handle_goal_resume,
    _handle_goal_clear,
)


def test_handler_signature_matches_dispatch_contract():
    """dispatch 调 handler(args, **kwargs)，五个 handler 签名必须兼容。"""
    for handler in _ALL_HANDLERS:
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[0].name == "args"
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)


def test_schemas_use_openai_parameters_key():
    for schema in (
        GOAL_START_SCHEMA, GOAL_STATUS_SCHEMA, GOAL_PAUSE_SCHEMA,
        GOAL_RESUME_SCHEMA, GOAL_CLEAR_SCHEMA,
    ):
        assert "parameters" in schema, f"{schema.get('name')} 缺 parameters 键"
        assert "inputSchema" not in schema


def test_start_schema_required_fields():
    props = GOAL_START_SCHEMA["parameters"]["properties"]
    assert set(props) == {"objective", "token_budget"}
    assert GOAL_START_SCHEMA["parameters"]["required"] == ["objective"]
    assert props["token_budget"]["default"] == 200000


# ---------------------------------------------------------------------------
# 4. registry 注册 + 分类 + core 可见性
# ---------------------------------------------------------------------------

_GOAL_TOOLS = ("goal_start", "goal_status", "goal_pause", "goal_resume",
               "goal_clear")


def test_goal_tools_registered_in_core():
    for name in _GOAL_TOOLS:
        entry = registry.get(name)
        assert entry is not None, f"{name} 未注册"
        assert entry.toolset == "core"


def test_goal_tools_concurrency_classification():
    """goal_status 只读 SAFE；其余 4 个改状态 + 落盘，UNSAFE。"""
    assert registry.get("goal_status").isConcurrencySafe is True
    for name in ("goal_start", "goal_pause", "goal_resume", "goal_clear"):
        assert registry.get(name).isConcurrencySafe is False


def test_goal_tools_in_core_toolset_visible():
    """五工具必须进 _CORE_TOOLS 才对 LLM 可见（发现 ≠ 可见）。"""
    from toolsets import resolve_toolset
    core = resolve_toolset("core")
    for name in _GOAL_TOOLS:
        assert name in core, f"{name} 不在 core 工具集"


def test_classification_test_file_updated():
    """分类清单同步守护：goal_status 在 SAFE_TOOLS，4 个写操作在 UNSAFE_TOOLS。"""
    from tests.test_tool_concurrency_classification import SAFE_TOOLS, UNSAFE_TOOLS
    assert "goal_status" in SAFE_TOOLS
    for name in ("goal_start", "goal_pause", "goal_resume", "goal_clear"):
        assert name in UNSAFE_TOOLS
