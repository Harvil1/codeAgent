"""启动期校验/容错测试：Bug #1/#2/#3 修复验证。"""
import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Bug #2: model_cfg 字段校验
# ---------------------------------------------------------------------------

def test_validate_model_config_passes_with_full_config():
    """完整 config 不抛。"""
    from cli import _validate_model_config
    cfg = {"model": {"name": "deepseek-chat", "provider": "deepseek"}}
    # 不抛即通过
    _validate_model_config(cfg)


def test_validate_model_config_missing_name_raises_systemexit(capsys):
    """缺 model.name → SystemExit(2) + 友好提示含 'model.name'。"""
    from cli import _validate_model_config
    cfg = {"model": {"provider": "deepseek"}}  # 没 name
    with pytest.raises(SystemExit) as exc_info:
        _validate_model_config(cfg)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "model.name" in combined or "model" in combined.lower(), (
        f"提示应含 model.name 字段名，实际: {combined}"
    )


def test_validate_model_config_missing_provider_raises_systemexit(capsys):
    """缺 model.provider → SystemExit(2) + 提示含 'provider'。"""
    from cli import _validate_model_config
    cfg = {"model": {"name": "deepseek-chat"}}  # 没 provider
    with pytest.raises(SystemExit) as exc_info:
        _validate_model_config(cfg)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "provider" in combined.lower(), (
        f"提示应含 provider 字段名，实际: {combined}"
    )


def test_validate_model_config_missing_model_section_raises_systemexit(capsys):
    """整个 model 段缺失 → SystemExit(2) + 提示含 'model'。"""
    from cli import _validate_model_config
    cfg = {}  # 整段 model 都没
    with pytest.raises(SystemExit) as exc_info:
        _validate_model_config(cfg)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "model" in combined.lower()


def test_validate_model_config_empty_values_rejected(capsys):
    """name/provider 是空字符串也算缺失（避免静默失败）。"""
    from cli import _validate_model_config
    cfg = {"model": {"name": "", "provider": "deepseek"}}
    with pytest.raises(SystemExit):
        _validate_model_config(cfg)


# ---------------------------------------------------------------------------
# Bug #3: curator 状态安全保存
# ---------------------------------------------------------------------------

def test_run_memory_curator_saves_state_on_partial_failure(tmp_path, monkeypatch):
    """Bug #3 fix: curator 中途失败时 try/finally 仍 save_state。

    验证：apply_automatic_transitions 抛异常时，state 仍被写盘。
    """
    import cli

    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()

    # mock 模块级函数（apply_automatic_transitions 抛异常模拟中途崩溃）
    def exploding_apply(d):
        raise RuntimeError("模拟中途崩溃")

    saved_state = {"calls": []}

    def fake_save_state(d, state):
        saved_state["calls"].append(state.copy())

    def fake_load_state(d):
        return {"last_run_at": None, "last_run_summary": None}

    monkeypatch.setattr("agent.memory_curator.apply_automatic_transitions", exploding_apply)
    monkeypatch.setattr("agent.memory_curator.load_memory_curator_state", fake_load_state)
    monkeypatch.setattr("agent.memory_curator.save_memory_curator_state", fake_save_state)
    monkeypatch.setattr("agent.memory_curator.should_run_now_memory", lambda d, config: True)

    # 构造一个最小 rt，调内部 curator runner
    # 直接调 _run_memory_curator_once（提取出的函数）
    assert hasattr(cli, "_run_memory_curator_once"), "应提取 _run_memory_curator_once 函数"
    cli._run_memory_curator_once(memory_dir, config={})

    # 即使中途崩溃，state 也被保存
    assert len(saved_state["calls"]) >= 1, "中途崩溃也应保存 state"
    last = saved_state["calls"][-1]
    assert "last_run_at" in last
    # 失败信息写入 summary
    assert "失败" in last.get("last_run_summary", "") or "error" in last.get("last_run_summary", "").lower()


# ---------------------------------------------------------------------------
# Bug #1: MCP 失败可见性（main.py 改动小，间接测：函数可独立调用）
# ---------------------------------------------------------------------------

def test_main_module_initialization_failure_is_visible(capsys, monkeypatch):
    """Bug #1 fix: MCP 初始化失败时用户终端可见（不只是 debug log）。

    验证：mock initialize_mcp 抛异常，跑 main.py 顶层 MCP 初始化包裹函数
    _init_mcp_safely()，stderr/stdout 含 'MCP' 字样。
    """
    import main as main_module

    def exploding_init():
        raise RuntimeError("mock MCP 失败")

    monkeypatch.setattr("tools.mcp_tool.initialize_mcp", exploding_init)

    # 调提取出的包裹函数（不应抛）
    assert hasattr(main_module, "_init_mcp_safely"), "应提取 _init_mcp_safely 函数"
    main_module._init_mcp_safely()

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "MCP" in combined or "mcp" in combined.lower(), (
        f"MCP 失败应在终端可见，实际: {combined}"
    )
