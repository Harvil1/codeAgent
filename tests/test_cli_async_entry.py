"""验证 cli 主入口 + async run_conversation 适配（Task E1）。

核心契约（来自 plan Task E1，根据实际代码调整）：
1. cli.py 必须有一个 main 入口函数（之前没有，本 task 新增）
2. main.py 必须通过 cli.main 进入（不再内联参数解析）
3. 所有 run_conversation 调用必须被 asyncio.run 或 await 驱动
4. cli.py 必须 import asyncio

设计决策（implementer 注）：
- 保持 run_interactive / run_one_shot 同步签名，内部用 asyncio.run 驱动 async run_conversation
  原因：run_skill_in_fork 内部已用 asyncio.run(child.chat(...))，
       若 run_interactive 外层再套 asyncio.run 会嵌套报错
       "asyncio.run() cannot be called from a running event loop"。
       全量 async 改造留到 Plan 2B（届时 skill_fork 也改 async，可统一 await）。
- 新增 cli.main 同步入口：参数解析 + 分发到 run_interactive / run_one_shot。
  cli.main 不外层套 asyncio.run（避免与 run_interactive 内部的 asyncio.run 嵌套）。
"""
import inspect

import pytest


# ----------------------------------------------------------------------------
# 契约 1: cli 模块必须有 main 入口函数
# ----------------------------------------------------------------------------

def test_cli_has_main_entry():
    """cli 模块应提供 main 入口函数（供 main.py 调用）。"""
    import cli
    assert hasattr(cli, "main"), (
        "cli 模块应有 main 入口函数（之前 main.py 内联参数解析，本 task 抽出为 cli.main）"
    )
    assert callable(cli.main), "cli.main 应该是可调用的"


# ----------------------------------------------------------------------------
# 契约 2: cli.main 是同步函数（不外层套 asyncio.run，避免与内部 asyncio.run 嵌套）
# ----------------------------------------------------------------------------

def test_main_is_sync_function():
    """cli.main 应是同步函数。

    理由：run_interactive / run_one_shot 内部已用 asyncio.run 驱动 async run_conversation，
    若 cli.main 外层再 asyncio.run 包装，会报 "asyncio.run() cannot be called from a
    running event loop"。Plan 2B 全量 async 化后才能把 cli.main 改 asyncio.run 包装。
    """
    import cli
    assert not inspect.iscoroutinefunction(cli.main), (
        "cli.main 应是同步函数（run_interactive/run_one_shot 内部已经 asyncio.run，"
        "外层再套会嵌套报错）"
    )


# ----------------------------------------------------------------------------
# 契约 3: cli 模块 import asyncio
# ----------------------------------------------------------------------------

def test_cli_imports_asyncio():
    """cli 模块应 import asyncio（asyncio.run 的前提）。"""
    import cli
    source = inspect.getsource(cli)
    # 处理两种合法写法：import asyncio 或 from asyncio import run
    assert ("import asyncio" in source) or ("from asyncio import" in source), (
        "cli.py 应该 import asyncio（用于 asyncio.run 包装 async run_conversation）"
    )


# ----------------------------------------------------------------------------
# 契约 4: run_conversation 调用必须被 asyncio.run 驱动（同步包装器模式）
# ----------------------------------------------------------------------------

def test_run_conversation_calls_driven_by_asyncio():
    """cli.py 中所有 run_conversation 调用必须由 asyncio.run 驱动（同步包装器模式）。

    扫描 cli.py 源码：每个 rt.agent.run_conversation(...) 调用要么
    - 包在 asyncio.run(...) 里（本 task 的设计），要么
    - 前面有 await（如果以后 run_interactive 改 async）
    """
    import cli
    source = inspect.getsource(cli)

    # 找到所有含 run_conversation 的行
    lines = source.splitlines()
    rv_lines = [
        (i + 1, line)
        for i, line in enumerate(lines)
        if "run_conversation" in line and "def " not in line
    ]

    assert rv_lines, "预期 cli.py 含至少一处 run_conversation 调用"

    for lineno, line in rv_lines:
        # 注释行不算调用
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # 必须是 asyncio.run(...run_conversation...) 或 await ... run_conversation
        # 检查本行或前 2 行（防止跨行调用）
        context_lines = lines[max(0, lineno - 3):lineno + 1]
        context = "\n".join(context_lines)
        assert (
            ("asyncio.run" in context and "run_conversation" in context)
            or "await" in line
        ), (
            f"line {lineno} 的 run_conversation 调用未由 asyncio.run/await 驱动:\n"
            f"context:\n{context}"
        )


def test_run_one_shot_drives_async_run_conversation():
    """run_one_shot 应使用 asyncio.run 驱动 async run_conversation。"""
    import cli
    source = inspect.getsource(cli.run_one_shot)
    assert "asyncio.run" in source, (
        f"run_one_shot 应用 asyncio.run 驱动 async run_conversation，实际:\n{source}"
    )
    assert "run_conversation" in source


def test_run_interactive_drives_async_run_conversation():
    """run_interactive 应使用 asyncio.run 驱动 async run_conversation。"""
    import cli
    source = inspect.getsource(cli.run_interactive)
    assert "asyncio.run" in source, (
        f"run_interactive 应用 asyncio.run 驱动 async run_conversation，实际:\n{source}"
    )
    assert "run_conversation" in source


# ----------------------------------------------------------------------------
# 契约 5: run_interactive / run_one_shot 保持同步签名（避免破坏调用链）
# ----------------------------------------------------------------------------

def test_run_interactive_is_sync_function():
    """run_interactive 应保持同步签名（内部用 asyncio.run 驱动 async）。

    设计理由：run_skill_in_fork 内部用 asyncio.run(child.chat(...))，
    若 run_interactive 改 async，调用链会嵌套 asyncio.run 报 RuntimeError。
    Plan 2B 全量 async 化时才能改 async。
    """
    import cli
    assert not inspect.iscoroutinefunction(cli.run_interactive), (
        "run_interactive 应保持同步签名（设计决策：避免破坏 run_skill_in_fork）"
    )


def test_run_one_shot_is_sync_function():
    """run_one_shot 应保持同步签名。"""
    import cli
    assert not inspect.iscoroutinefunction(cli.run_one_shot), (
        "run_one_shot 应保持同步签名"
    )


# ----------------------------------------------------------------------------
# 契约 6: main.py 调用 cli.main（不是直接调 run_interactive）
# ----------------------------------------------------------------------------

def test_main_py_uses_cli_main():
    """main.py 应通过 cli.main() 进入（不是直接调 run_interactive/run_one_shot）。"""
    import pathlib
    main_py = pathlib.Path("main.py").read_text(encoding="utf-8")
    assert "cli.main" in main_py or "from cli import main" in main_py, (
        "main.py 应通过 cli.main() 进入主入口（本 task 抽出 cli.main 函数）"
    )


# ----------------------------------------------------------------------------
# 契约 7: cli.main 参数解析 + 分发逻辑（行为测试）
# ----------------------------------------------------------------------------

def test_cli_main_dispatches_chat_to_run_one_shot(monkeypatch):
    """cli.main 应识别 chat 子命令并调 run_one_shot。"""
    import cli

    called = {"one_shot": None, "interactive": None}

    def fake_one_shot(msg):
        called["one_shot"] = msg

    def fake_interactive(resume_last=False, cli_agents=None):
        called["interactive"] = (resume_last, cli_agents)

    monkeypatch.setattr(cli, "run_one_shot", fake_one_shot)
    monkeypatch.setattr(cli, "run_interactive", fake_interactive)

    # 模拟 python main.py chat "你好"
    cli.main(["main.py", "chat", "你好", "世界"])

    assert called["one_shot"] == "你好 世界"
    assert called["interactive"] is None


def test_cli_main_dispatches_interactive_when_no_chat(monkeypatch):
    """cli.main 无 chat 子命令时应走交互模式。"""
    import cli

    called = {"one_shot": None, "interactive": None}

    def fake_one_shot(msg):
        called["one_shot"] = msg

    def fake_interactive(resume_last=False, cli_agents=None):
        called["interactive"] = (resume_last, cli_agents)

    monkeypatch.setattr(cli, "run_one_shot", fake_one_shot)
    monkeypatch.setattr(cli, "run_interactive", fake_interactive)

    cli.main(["main.py"])

    assert called["one_shot"] is None
    assert called["interactive"] == (False, None)


def test_cli_main_extracts_continue_flag(monkeypatch):
    """cli.main 应识别 -c / --continue 标志。"""
    import cli

    called = {"interactive": None}

    def fake_interactive(resume_last=False, cli_agents=None):
        called["interactive"] = (resume_last, cli_agents)

    monkeypatch.setattr(cli, "run_interactive", fake_interactive)
    monkeypatch.setattr(cli, "run_one_shot", lambda msg: None)

    cli.main(["main.py", "-c"])
    assert called["interactive"][0] is True

    cli.main(["main.py", "--continue"])
    assert called["interactive"][0] is True
