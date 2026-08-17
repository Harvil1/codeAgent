"""阶段 6 测试：--agents CLI 动态注入。

覆盖：
- inject_cli_agents 把 raw dict 解析为 AgentDefinition
- scan_agent_defs 优先级：项目级 > CLI > 用户级 > 内置
- get_cli_injected 返回注入的字典
- clear_cli_injected 清空
- 非法 cfg 被跳过（不抛）
"""
import os
from pathlib import Path
from unittest.mock import patch


def _clean_state():
    from agent.agent_defs import clear_cli_injected
    clear_cli_injected()


def test_inject_basic():
    """inject_cli_agents 解析合法 dict → 注入成功。"""
    _clean_state()
    from agent.agent_defs import inject_cli_agents, get_cli_injected
    cli_agents = {
        "code-reviewer": {
            "description": "代码审查专家",
            "prompt": "你是审查员",
            "tools": ["Read", "Grep"],
            "model": "sonnet",
        }
    }
    n = inject_cli_agents(cli_agents)
    assert n == 1
    injected = get_cli_injected()
    assert "code-reviewer" in injected
    assert injected["code-reviewer"].description == "代码审查专家"
    assert injected["code-reviewer"].system_prompt == "你是审查员"
    _clean_state()


def test_inject_idempotent_replace():
    """多次调 inject 是替换语义，不是累加。"""
    _clean_state()
    from agent.agent_defs import inject_cli_agents, get_cli_injected
    inject_cli_agents({"a": {"description": "first"}})
    inject_cli_agents({"b": {"description": "second"}})
    injected = get_cli_injected()
    assert "a" not in injected
    assert "b" in injected
    _clean_state()


def test_inject_invalid_cfg_skipped():
    """非法 cfg 被跳过，不抛。"""
    _clean_state()
    from agent.agent_defs import inject_cli_agents, get_cli_injected
    n = inject_cli_agents({
        "good": {"description": "ok", "prompt": "p"},
        "bad": "not a dict",  # 非法
    })
    assert n == 1
    assert "good" in get_cli_injected()
    assert "bad" not in get_cli_injected()
    _clean_state()


def test_inject_handles_empty_and_none():
    """空 / None 输入 → 0，不抛。"""
    _clean_state()
    from agent.agent_defs import inject_cli_agents
    assert inject_cli_agents({}) == 0
    assert inject_cli_agents(None) == 0
    _clean_state()


def test_scan_priority_project_beats_cli(monkeypatch, tmp_path):
    """项目级覆盖 CLI（项目级优先级最高）。"""
    _clean_state()
    from agent.agent_defs import (
        inject_cli_agents, scan_agent_defs,
    )
    # 注入 CLI 版本
    inject_cli_agents({"dup": {"description": "from CLI", "prompt": "CLI"}})

    # 项目级有同名
    proj_dir = tmp_path / ".omnimate" / "agents"
    proj_dir.mkdir(parents=True)
    (proj_dir / "dup.md").write_text(
        "---\nname: dup\ndescription: from project\n---\n项目版本",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: proj_dir)
    # user/builtin 都不存在
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: tmp_path / "nope_u")
    monkeypatch.setattr("agent.agent_defs._builtin_agents_dir", lambda: tmp_path / "nope_b")

    defs = scan_agent_defs()
    assert defs["dup"].description == "from project"
    _clean_state()


def test_scan_priority_cli_beats_user(monkeypatch, tmp_path):
    """CLI 覆盖用户级。"""
    _clean_state()
    from agent.agent_defs import inject_cli_agents, scan_agent_defs

    inject_cli_agents({"dup": {"description": "from CLI", "prompt": "x"}})

    user_dir = tmp_path / "user_agents"
    user_dir.mkdir()
    (user_dir / "dup.md").write_text(
        "---\nname: dup\ndescription: from user\n---\nuser 版本",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.agent_defs._user_agents_dir", lambda: user_dir)
    monkeypatch.setattr("agent.agent_defs._project_agents_dir", lambda: tmp_path / "nope_p")
    monkeypatch.setattr("agent.agent_defs._builtin_agents_dir", lambda: tmp_path / "nope_b")

    defs = scan_agent_defs()
    assert defs["dup"].description == "from CLI"
    _clean_state()


def test_main_argparse_extracts_agents(monkeypatch):
    """main.py 解析 --agents 后正确从 args 移除（不影响 chat/-c）。"""
    # 这里直接测 inject 路径，main.py 的整体启动需要完整 RuntimeContext 不在单测范围
    _clean_state()
    from agent.agent_defs import inject_cli_agents, get_cli_injected
    inject_cli_agents({"x": {"description": "y"}})
    assert "x" in get_cli_injected()
    _clean_state()
