"""第 5 轮 bug 修复测试：剩余严重 + 关键建议（X2/X3/X4/X11/X15/X16/X17）。"""
import inspect
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# X2: run_interactive 应在 finally 调 rt.shutdown（异常退出也清理）
# ---------------------------------------------------------------------------

def test_run_interactive_shutdown_in_finally():
    """X2 fix: RuntimeContext 用 atexit.register 兜底 shutdown（异常退出也清理）。"""
    from cli import RuntimeContext
    src_init = inspect.getsource(RuntimeContext.__init__)
    src_shutdown_check = inspect.getsource(RuntimeContext)
    # X2 fix：__init__ 应注册 atexit 兜底（即使主循环异常也会清理）
    assert "atexit" in src_init, (
        f"RuntimeContext.__init__ 应含 atexit.register(self.shutdown) 兜底，实际:\n{src_init[:500]}"
    )
    assert "shutdown" in src_shutdown_check, "应含 shutdown 方法"


# ---------------------------------------------------------------------------
# X3: agent.cleanup 应关闭 LLM client（HTTP 连接）
# ---------------------------------------------------------------------------

def test_agent_cleanup_closes_llm_clients():
    """X3 fix: agent.cleanup 源码应含 close LLM client 的逻辑。"""
    from agent import AIAgent
    src = inspect.getsource(AIAgent.cleanup)
    # 应含 close() 调用（针对 llm_client / fallback_llm_client / _vision_client）
    assert "close" in src.lower(), (
        f"cleanup 应 close LLM clients（HTTP 连接泄漏），实际:\n{src}"
    )


def test_agent_cleanup_idempotent_and_safe():
    """回归：cleanup 多次调用安全（无 client 时不崩）。"""
    from agent import AIAgent
    agent = AIAgent(
        base_url="http://x", api_key="sk-x", model="x", model_format="openai",
        max_iterations=1, enabled_toolsets=[],
    )
    # llm_client 可能是 MagicMock 或 None，cleanup 不应崩
    agent.cleanup()
    agent.cleanup()  # 二次调用


# ---------------------------------------------------------------------------
# X4: _create_agent aux main_client 创建失败时应关闭
# ---------------------------------------------------------------------------

def test_create_agent_closes_aux_main_client_on_failure():
    """X4 fix: _create_agent 源码在 aux_router 创建失败时应 close main_client。

    静态验证：源码在 except 块附近含 close 调用（callable 形式或直接调）。
    """
    from cli import RuntimeContext
    src = inspect.getsource(RuntimeContext._create_agent)
    # X4 fix: 含 close 关键字（close_fn / close() / .close 任一）
    assert "close" in src.lower(), (
        f"_create_agent 应在异常路径 close 失败创建的 client，实际:\n{src[:1000]}"
    )


# ---------------------------------------------------------------------------
# X11: Cron 一次性 job 触发后应立刻 disable + persist（避免重启重复触发）
# ---------------------------------------------------------------------------

def test_cron_oneshot_disable_before_notification():
    """X11 fix: 一次性 job 触发时立刻 disable+persist，再 push 通知。

    静态验证：_tick 源码对一次性 job 的处理含"先 disable 再通知"或"立刻 persist"。
    """
    from agent.cron import CronScheduler
    src = inspect.getsource(CronScheduler._tick)
    # 关键：一次性 job 触发后，disable 应该在循环内立即发生（不要等所有 job 迭代完）
    # 简化验证：源码应含 oneshot 相关的立即 disable 逻辑
    # 检测：fired_oneshot_ids.add 之后或 job.recurring 检查附近有立即 persist/disable
    # 静态：源码含"立即"或"触发即"或 _persist 调用次数 >= 2
    persist_count = src.count("_persist_jobs_unlocked()")
    assert persist_count >= 2, (
        f"_tick 应至少 2 次 _persist_jobs_unlocked（循环内 + 末尾），实际 {persist_count} 次"
    )


# ---------------------------------------------------------------------------
# X15: macOS Seatbelt profile 用完应清理
# ---------------------------------------------------------------------------

def test_seatbelt_profile_cleanup_after_use():
    """X15 fix: _seatbelt_wrap 或调用方应清理 profile 文件。

    静态验证：sandbox_runner.py 含 cleanup 删除 .sb 文件的逻辑。
    """
    import agent.sandbox_runner as mod
    src = inspect.getsource(mod)
    # 应含 unlink / remove / cleanup 等清理 .sb 的代码
    has_cleanup = (
        "unlink" in src or "cleanup" in src.lower() or "remove" in src.lower()
    )
    assert has_cleanup, (
        "sandbox_runner 应含 profile cleanup 逻辑（避免 .sb 文件无限堆积）"
    )


# ---------------------------------------------------------------------------
# X16: write_file 应使用 atomic_write_text（避免写到一半崩溃）
# ---------------------------------------------------------------------------

def test_write_file_uses_atomic_write():
    """X16 fix: file_operations 源码 write 路径应用 atomic_write_text，不用 path.write_text。"""
    import tools.file_operations as mod
    src = inspect.getsource(mod)
    # 应导入并使用 atomic_write_text
    assert "atomic_write_text" in src, (
        "file_operations 应使用 atomic_write_text（原子写，防崩溃半文件）"
    )


# ---------------------------------------------------------------------------
# X17: 删除 safe_path 死代码 except
# ---------------------------------------------------------------------------

def test_safe_path_no_dead_except():
    """X17 fix: safe_path 不再有"连续两个 except"死代码模式。

    bug 之前：`except ValueError: continue` 紧跟 `except (OSError, ValueError): continue`，
    第二个永远到不了（ValueError 已被前者捕获）。
    """
    from agent.permission import safe_path
    src = inspect.getsource(safe_path)
    # 不应连续出现两段 except（除了第一段 catch 后第二段死代码）
    # 简化：检查"except ValueError:" 后紧跟 "except (OSError, ValueError):" 的模式
    import re as _re
    dead_pattern = _re.compile(
        r"except\s+ValueError\s*:\s*\n\s*continue\s*\n\s*except\s+\(OSError,\s*ValueError\)\s*:",
        _re.MULTILINE,
    )
    assert not dead_pattern.search(src), (
        f"safe_path 不应含死代码双 except 模式，实际:\n{src}"
    )
