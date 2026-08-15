"""T2（核心机制对齐第 2 项）：post-compact 恢复统一 token 预算 + plan/async 状态。

- config context.post_compact_recovery_budget（默认 40000 tokens）总预算统筹
- 优先级：plan/async 状态 > 最近文件 > 技能正文（超预算按优先级截断）
- 新恢复源：plan_mode / _last_approved_plan + _async_tasks running 列表
- 无 plan 无 async 时与现状等价（回归）
"""
import time

import pytest

from tests.test_post_compact_recovery import _make_minimal_agent


@pytest.fixture(autouse=True)
def _clear_async_tasks():
    """清空全局 _async_tasks（防其他测试残留 running 条目污染断言）。"""
    import tools.delegate_tool as _dt
    _dt._async_tasks.clear()
    yield
    _dt._async_tasks.clear()


# ---------------------------------------------------------------------------
# plan / async 状态恢复源
# ---------------------------------------------------------------------------

def test_plan_mode_active_in_brief(tmp_path):
    """plan_mode 激活 → brief 含 plan 调研状态段。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent.plan_mode = True
    agent._recent_read_files = []
    agent._recent_skills = []
    result = build_post_compact_brief(agent)
    assert "plan" in result.lower()
    assert "调研" in result or "计划" in result


def test_last_approved_plan_in_brief(tmp_path):
    """已批准计划（执行中）→ brief 含计划全文段。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent.plan_mode = False
    agent._last_approved_plan = "1. 改 agent/foo.py\n2. 加测试\n3. 跑回归"
    agent._recent_read_files = []
    agent._recent_skills = []
    result = build_post_compact_brief(agent)
    assert "执行" in result or "计划" in result
    assert "agent/foo.py" in result


def test_async_running_tasks_in_brief(tmp_path):
    """有 running async 子代理 → brief 列出（id/指令摘要/已运行时长）。"""
    from agent.post_compact_recovery import build_post_compact_brief
    import tools.delegate_tool as dt

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent._recent_read_files = []
    agent._recent_skills = []
    try:
        dt._async_tasks["del_test123"] = {
            "thread": None,
            "cancel_event": None,
            "goal": "扫描所有工具注册点",
            "started_at": time.time() - 30,
        }
        result = build_post_compact_brief(agent)
        assert "del_test123" in result
        assert "扫描所有工具注册点" in result
        assert "async" in result.lower() or "子代理" in result
    finally:
        dt._async_tasks.pop("del_test123", None)


def test_async_registry_records_goal_and_started_at():
    """_delegate_async 注册 _async_tasks 时记录 goal/started_at（源码级防漏改）。"""
    import inspect
    import tools.delegate_tool as dt
    src = inspect.getsource(dt._delegate_async)
    assert '"goal": goal' in src
    assert '"started_at"' in src


# ---------------------------------------------------------------------------
# 统一预算
# ---------------------------------------------------------------------------

def test_budget_small_plan_survives_files_skills_truncated(tmp_path):
    """预算小时文件/技能被截断，plan 段保留（优先级最高）。"""
    from agent.post_compact_recovery import build_post_compact_brief

    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent.plan_mode = False
    agent._last_approved_plan = "关键计划步骤：" + "步骤X" * 50
    big_file = tmp_path / "big.txt"
    big_file.write_text("F" * 40000, encoding="utf-8")
    agent._recent_read_files = [str(big_file)]
    agent._recent_skills = []

    agent.config = {"context": {"post_compact_recovery_budget": 3000}}  # 3000 tokens
    result = build_post_compact_brief(agent)
    assert "关键计划步骤" in result  # plan 段完整保留
    # 文件被预算截断（40000 字符的文件不可能全在 3000 token 预算内）
    assert "F" * 4000 not in result


def test_budget_default_config_key():
    """DEFAULT_CONFIG 含 post_compact_recovery_budget=40000（有真实读取点）。"""
    from config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["context"]["post_compact_recovery_budget"] == 40000


def test_no_plan_no_async_equivalent_output(tmp_path):
    """无 plan 无 async → 只含文件/技能段（与现状等价，回归保护）。"""
    from agent.post_compact_recovery import build_post_compact_brief

    f1 = tmp_path / "a.txt"
    f1.write_text("plain content", encoding="utf-8")
    agent = _make_minimal_agent(omnimate_home=tmp_path)
    agent.plan_mode = False
    agent._last_approved_plan = ""
    agent._recent_read_files = [str(f1)]
    agent._recent_skills = []
    result = build_post_compact_brief(agent)
    assert "plain content" in result
    assert "计划" not in result and "子代理" not in result
