"""statusline 测试。

每轮 AI 响应后尾部打一行紧凑状态：model / 会话 token / goal / 项目名。
"""
from unittest.mock import MagicMock


def _make_rt_agent(model="deepseek-chat", tokens=12300, goal=None, proj="D--project-HermesAgent"):
    """构造 mock rt/agent，简化测试 boilerplate。

    tokens 通过 _llm_usage_stats（与生产字段一致）传入；
    兼容旧版 session_total_tokens 字段。
    """
    # spec=[] 让 MagicMock 不自动生成属性（防止 getattr 自动命中 mock），
    # 这样 _llm_usage_stats 没显式设的话，生产代码里 getattr 返回 None 而不是 MagicMock。
    agent = MagicMock(spec=["model", "session_total_tokens", "_llm_usage_stats", "_goal_state"])
    agent.model = model
    agent.session_total_tokens = tokens
    agent._llm_usage_stats = {
        "total_calls": 1,
        "total_prompt_tokens": tokens // 2,
        "total_completion_tokens": tokens - tokens // 2,
        "total_cache_read_tokens": 0,
        "total_cache_creation_tokens": 0,
    }
    agent._goal_state = goal
    rt = MagicMock(spec=["config", "_statusline_project_key"])
    rt.config = {"statusline": {"enabled": True}}
    rt._statusline_project_key = proj
    return rt, agent


def test_renders_all_segments():
    """四段全显：model / token / goal:进行中#N / 项目名。"""
    from cli import _render_statusline
    goal = MagicMock()
    goal.status = "active"
    goal.iteration_count = 5
    rt, agent = _make_rt_agent(goal=goal)
    line = _render_statusline(rt, agent)
    assert "deepseek-chat" in line
    assert "12.3K" in line  # 12300 → 12.3K
    assert "goal:进行中#5" in line
    assert "HermesAgent" in line


def test_no_goal_segment_when_absent():
    """无 goal_state 时不显示 goal 段。"""
    from cli import _render_statusline
    rt, agent = _make_rt_agent(goal=None)
    line = _render_statusline(rt, agent)
    assert "goal" not in line


def test_paused_goal_shows_paused():
    """paused 状态显示"goal:已暂停"。"""
    from cli import _render_statusline
    goal = MagicMock()
    goal.status = "paused"
    rt, agent = _make_rt_agent(goal=goal)
    assert "goal:已暂停" in _render_statusline(rt, agent)


def test_completed_goal_shows_done():
    """completed 状态显示"goal:已完成"。"""
    from cli import _render_statusline
    goal = MagicMock()
    goal.status = "completed"
    goal.iteration_count = 8
    rt, agent = _make_rt_agent(goal=goal)
    assert "goal:已完成" in _render_statusline(rt, agent)


def test_cancelled_goal_hidden():
    """cancelled 状态不显示 goal 段（视为已废弃）。"""
    from cli import _render_statusline
    goal = MagicMock()
    goal.status = "cancelled"
    goal.iteration_count = 2
    rt, agent = _make_rt_agent(goal=goal)
    assert "goal" not in _render_statusline(rt, agent)


def test_disabled_returns_empty():
    """config.statusline.enabled=False → 空串。"""
    from cli import _render_statusline
    rt, agent = _make_rt_agent()
    rt.config = {"statusline": {"enabled": False}}
    assert _render_statusline(rt, agent) == ""


def test_no_statusline_config_defaults_enabled():
    """config 里没有 statusline 键时，默认 enabled=True（不崩）。"""
    from cli import _render_statusline
    rt, agent = _make_rt_agent()
    rt.config = {}  # 没有 statusline 键
    line = _render_statusline(rt, agent)
    assert "deepseek-chat" in line  # 仍然渲染


def test_token_formatting():
    """token 格式化：0 → '0'；1500 → '1.5K'；1_234_567 → '1.2M'。"""
    from cli import _format_tokens
    assert _format_tokens(0) == "0"
    assert _format_tokens(1500) == "1.5K"
    assert _format_tokens(1_234_567) == "1.2M"


def test_prefers_llm_usage_stats():
    """有 _llm_usage_stats 时优先用 prompt+completion 求和，
    而不是 session_total_tokens 字段（前者更准确，实时更新）。

    给两个字段不同的值，验证用的是 stats。
    """
    from cli import _render_statusline
    agent = MagicMock(spec=["model", "session_total_tokens", "_llm_usage_stats", "_goal_state"])
    agent.model = "opus"
    agent.session_total_tokens = 88888   # 故意大，模拟旧字段不准
    agent._llm_usage_stats = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
    }   # stats 求和=150 → 0.2K
    agent._goal_state = None
    rt = MagicMock(spec=["config", "_statusline_project_key"])
    rt.config = {"statusline": {"enabled": True}}
    rt._statusline_project_key = "D--project-Foo"
    line = _render_statusline(rt, agent)
    assert "150 tok" in line        # 取了 stats 求和 (100+50)
    assert "88.9K" not in line     # 没取 session_total_tokens
    assert "88.8K" not in line


def test_fallback_to_session_total_tokens():
    """没有 _llm_usage_stats 时退到 session_total_tokens 字段。"""
    from cli import _render_statusline
    agent = MagicMock()
    agent.model = "opus"
    agent.session_total_tokens = 5000
    # 没有 _llm_usage_stats
    del agent._llm_usage_stats
    rt = MagicMock()
    rt.config = {"statusline": {"enabled": True}}
    rt._statusline_project_key = "D--project-Foo"
    line = _render_statusline(rt, agent)
    assert "5.0K" in line


def test_fail_open_on_exception():
    """任何异常都返回空串（statusline 永不影响主流程）。"""
    from cli import _render_statusline
    rt = MagicMock()
    # 故意让 rt.config 抛
    rt.config = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    agent = MagicMock()
    agent.model = "x"
    # config 访问会抛 → fail-open 返回 ""
    # 注意 property 在 MagicMock 上不一定会抛，所以直接测 _render_statusline 的 try 覆盖
    # 这里用一个真会抛的对象：
    class Boom:
        @property
        def config(self):
            raise RuntimeError("boom")
    rt2 = Boom()
    assert _render_statusline(rt2, agent) == ""


def test_no_project_key_omits_project_segment():
    """无 _statusline_project_key 时不显示项目段。"""
    from cli import _render_statusline
    rt, agent = _make_rt_agent()
    rt._statusline_project_key = ""
    line = _render_statusline(rt, agent)
    assert "项目" not in line


def test_no_model_omits_model_segment():
    """无 model 时不显示 model 段。"""
    from cli import _render_statusline
    rt, agent = _make_rt_agent(model="")
    line = _render_statusline(rt, agent)
    assert "⚡" not in line


def test_failed_goal_shows_failed():
    """goal failed 状态显示。"""
    from cli import _render_statusline
    goal = MagicMock()
    goal.status = "failed"
    rt, agent = _make_rt_agent(goal=goal)
    assert "goal:失败" in _render_statusline(rt, agent)
