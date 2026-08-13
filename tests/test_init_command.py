"""CCAR9 Task 4: /init 命令生成 OMNIMATE.md 的测试。

对标 Claude Code 的 /init 命令：收集项目信息 → 主 LLM 生成四段式
（项目本质/常用命令/架构/约定）→ 写 <cwd>/OMNIMATE.md。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_rt(tmp_path, llm_text="# OMNIMATE\n内容"):
    """构造 mock RuntimeContext + agent（带 async llm_client）。"""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=llm_text, tool_calls=None),
        )],
    )
    agent = MagicMock()
    agent.llm_client = MagicMock()
    agent.llm_client.chat_completions = AsyncMock(return_value=resp)
    rt = MagicMock()
    rt.agent = agent
    rt.home = tmp_path
    return rt


def _run_init(rt, args="", cwd=None):
    """跑 /init，patch get_workspace_cwd 让目标 cwd 指向给定目录。

    get_workspace_cwd 在 cli._handle_init_command 内部局部 import，
    所以 patch 源头 agent.workspace_context.get_workspace_cwd。
    """
    from cli import _handle_init_command
    with patch(
        "agent.workspace_context.get_workspace_cwd",
        return_value=str(cwd),
    ):
        return _handle_init_command(rt, args)


def test_init_generates_file(tmp_path, monkeypatch):
    """正常流程：空目录 + README → 生成 OMNIMATE.md 含 LLM 输出。"""
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "README.md").write_text("# 测试项目\n技术栈说明", encoding="utf-8")
    rt = _make_rt(tmp_path)
    _run_init(rt, cwd=proj)
    out = proj / "OMNIMATE.md"
    assert out.exists()
    assert "内容" in out.read_text(encoding="utf-8")


def test_init_existing_without_force(tmp_path, monkeypatch):
    """已存在 OMNIMATE.md 且不带 --force：不覆盖，返回提示。"""
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "OMNIMATE.md").write_text("旧内容", encoding="utf-8")
    rt = _make_rt(tmp_path)
    result = _run_init(rt, cwd=proj)
    # 不覆盖
    assert "旧内容" in (proj / "OMNIMATE.md").read_text(encoding="utf-8")
    # 返回 truthy 表示命令已处理（CLI 惯例：console 提示 + return True）
    assert result


def test_init_force_overwrites(tmp_path, monkeypatch):
    """--force 覆盖重新生成。"""
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "OMNIMATE.md").write_text("旧内容", encoding="utf-8")
    rt = _make_rt(tmp_path, llm_text="全新生成")
    _run_init(rt, args="--force", cwd=proj)
    assert "全新生成" in (proj / "OMNIMATE.md").read_text(encoding="utf-8")


def test_init_collection_failopen(tmp_path, monkeypatch):
    """项目信息收集失败不崩（空目录也能生成）。"""
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "empty_proj"
    proj.mkdir()
    rt = _make_rt(tmp_path, llm_text="空项目")
    _run_init(rt, cwd=proj)  # 不抛
    assert (proj / "OMNIMATE.md").exists()
