"""验证 agent.team.worker 适配 async run_conversation（Task E2）。

核心契约（来自 plan Task E2 + 对齐 E1 模式）：
1. worker.py 必须 import asyncio（asyncio.run 用于驱动 async run_conversation）
2. 所有 agent.run_conversation 调用必须由 asyncio.run 或 await 驱动（不能裸调）
3. worker.main 保持同步签名（CLI 入口，子进程启动，避免嵌套 asyncio.run）
4. AutonomousLifecycle 内的 work_fn 包装（_run_work）内部用 asyncio.run 驱动
5. 一次性模式分支也用 asyncio.run 驱动 async run_conversation

设计决策（implementer 注）：
- worker 是 CLI 子进程入口（python -m agent.team.worker ...），不是 threading target，
  所以 main() 保持同步签名，内部用 asyncio.run 驱动 async run_conversation（同 E1 模式）。
- AutonomousLifecycle.run() 是同步状态机，它的 work_fn 回调签名是 sync Callable[[str], str]，
  所以 _run_work 必须是同步函数，内部用 asyncio.run 驱动 async run_conversation。
"""
import inspect
import re

import pytest


WORKER_PATH = "agent/team/worker.py"
WORKER_SOURCE = None


def _load_worker_source() -> str:
    """读取 worker.py 源代码（缓存）。"""
    global WORKER_SOURCE
    if WORKER_SOURCE is None:
        with open(WORKER_PATH, encoding="utf-8") as f:
            WORKER_SOURCE = f.read()
    return WORKER_SOURCE


# ----------------------------------------------------------------------------
# 契约 1: worker.py 必须 import asyncio
# ----------------------------------------------------------------------------

def test_worker_imports_asyncio():
    """worker.py 必须导入 asyncio（驱动 async run_conversation）。"""
    src = _load_worker_source()
    assert "import asyncio" in src, (
        "worker.py 应导入 asyncio（asyncio.run 用于驱动 async run_conversation）"
    )


# ----------------------------------------------------------------------------
# 契约 2: worker.main 保持同步签名（CLI 入口，子进程启动）
# ----------------------------------------------------------------------------

def test_worker_main_is_sync():
    """worker.main 应保持同步签名（CLI 入口，由 __main__ 直接调）。

    设计原因：
    - worker 由 coordinator.spawn 以子进程启动（sys.executable -m agent.team.worker）
    - main() 是 argparse 入口，子进程的 __main__ 直接调 main()
    - 不需要把 main 改 async（async def main 配 asyncio.run(main()) 也可以，
      但当前架构更简洁的是 main 保持同步，内部在调用点用 asyncio.run 驱动）
    """
    import agent.team.worker as worker_mod
    assert hasattr(worker_mod, "main"), "worker 模块应有 main 函数"
    # main 应是同步函数（不是 coroutine function）
    assert not inspect.iscoroutinefunction(worker_mod.main), (
        "worker.main 应保持同步签名（CLI 入口；asyncio.run 包装在调用点）"
    )


# ----------------------------------------------------------------------------
# 契约 3: 所有 run_conversation 调用由 asyncio.run 或 await 驱动
# ----------------------------------------------------------------------------

def test_no_bare_run_conversation_calls():
    """worker.py 中所有 run_conversation 调用必须被 asyncio.run 或 await 驱动。

    裸调 async run_conversation 会返回 coroutine，后续操作（json 序列化、
    bus.send content=response）会 TypeError: the JSON object must be str...
    not coroutine。
    """
    src = _load_worker_source()
    # 找所有 run_conversation 调用点
    lines = src.splitlines()
    bare_call_pattern = re.compile(r"run_conversation\s*\(")
    issues = []
    for i, line in enumerate(lines, start=1):
        if bare_call_pattern.search(line):
            # 排除注释行（含 # 的行）
            stripped = line.split("#", 1)[0]
            if not bare_call_pattern.search(stripped):
                continue  # 在注释里
            # 检查行内或前一行是否有 asyncio.run 或 await 驱动
            # 允许的模式：
            #   asyncio.run(...run_conversation(...))
            #   response = asyncio.run(... run_conversation(...))
            #   await ... run_conversation(...)
            #   return asyncio.run(... run_conversation(...))
            # 也可能是多行调用，往前看几行找 asyncio.run(
            ctx = "\n".join(lines[max(0, i-5):i+2])
            has_driver = (
                "asyncio.run" in ctx or
                "await" in stripped or
                "await" in ctx and "run_conversation" in ctx
            )
            if not has_driver:
                issues.append(f"行 {i}: {line.strip()}")
    assert not issues, (
        f"以下 run_conversation 调用未由 asyncio.run 或 await 驱动（裸调 async 会返回 coroutine）:\n"
        + "\n".join(issues)
    )


# ----------------------------------------------------------------------------
# 契约 4: 一次性模式分支用 asyncio.run 驱动
# ----------------------------------------------------------------------------

def test_one_shot_mode_uses_asyncio_run():
    """一次性模式分支（else 分支，agent.run_conversation(args.task)）应包装 asyncio.run。

    该分支末尾 bus.send(content=response or "") 需要真实字符串，
    若裸调 async run_conversation 会拿到 coroutine 导致 TypeError。
    """
    src = _load_worker_source()
    # 验证：一次性分支里的 run_conversation 调用周围有 asyncio.run
    assert re.search(
        r"asyncio\.run\s*\([^)]*run_conversation",
        src,
        re.DOTALL
    ), (
        "worker.py 应有 asyncio.run 包装 run_conversation 调用（一次性模式分支）"
    )


# ----------------------------------------------------------------------------
# 契约 5: autonomous 模式的 _run_work 也由 asyncio.run 驱动
# ----------------------------------------------------------------------------

def test_autonomous_run_work_uses_asyncio_run():
    """autonomous 模式的 _run_work 闭包应内部用 asyncio.run 驱动 async run_conversation。

    AutonomousLifecycle.work_fn 签名是 sync Callable[[str], str]，
    所以 _run_work 必须同步，内部用 asyncio.run 驱动 async run_conversation。
    """
    src = _load_worker_source()
    # 找到 _run_work 函数定义，验证它内部有 asyncio.run
    match = re.search(
        r"def\s+_run_work\s*\([^)]*\)[^:]*:.*?(?=\n    [a-z]|\n    [A-Z]|\nclass|\Z)",
        src,
        re.DOTALL
    )
    assert match is not None, "worker.py 应有 _run_work 闭包（autonomous 模式）"
    body = match.group(0)
    assert "asyncio.run" in body, (
        "_run_work 内部应用 asyncio.run 驱动 async run_conversation"
    )


# ----------------------------------------------------------------------------
# 契约 6: worker.py 不引入 threading（保持子进程入口语义）
# ----------------------------------------------------------------------------

def test_worker_does_not_use_threading():
    """worker.py 不应引入 threading（worker 是子进程入口，不是线程目标）。

    plan 提到 "如果是 threading.Thread 调用，需要在 Thread target 里用 asyncio.run 包装"。
    实际 worker 是 subprocess 入口（coordinator.spawn），所以无需 threading。
    若未来引入 threading 则此契约需调整。
    """
    src = _load_worker_source()
    assert "import threading" not in src, (
        "worker.py 不应使用 threading（保持子进程入口语义）"
    )
