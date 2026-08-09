"""registry.dispatch async 行为测试。

TDD RED-GREEN 驱动 Task C1 改造：
- dispatch 必须是 async def（coroutine function）
- 同步 handler 用 anyio.to_thread.run_sync 包装
- async handler 直接 await
- ToolEntry 加 isConcurrencySafe 字段（默认 False）
- register 可显式传 isConcurrencySafe=True

注意：本项目的 register 签名是 `register(name, toolset, schema, handler, ...)`，
不是 plan 伪代码里的 `register(name, handler, schema=...)`——以实际代码为准。
"""
import inspect

import pytest

from tools.registry import ToolRegistry, ToolEntry


async def test_dispatch_is_coroutine_function():
    """registry.dispatch 必须是 async def。"""
    reg = ToolRegistry()
    assert inspect.iscoroutinefunction(reg.dispatch), \
        "ToolRegistry.dispatch 必须是 async def（coroutine function）"


async def test_dispatch_calls_sync_handler_via_to_thread():
    """同步 handler 应通过 anyio.to_thread.run_sync 包装（不阻塞事件循环）。"""
    reg = ToolRegistry()
    call_log = []

    def sync_handler(args, **kwargs):
        call_log.append(("sync", args, kwargs))
        return '{"result": "ok"}'

    reg.register("test_sync_tool", "core", {}, sync_handler)
    result = await reg.dispatch("test_sync_tool", {"x": 1})
    assert result == '{"result": "ok"}'
    assert call_log == [("sync", {"x": 1}, {})]


async def test_dispatch_awaits_async_handler_directly():
    """async handler 应直接 await（不走 to_thread）。"""
    reg = ToolRegistry()
    call_log = []

    async def async_handler(args, **kwargs):
        call_log.append(("async", args, kwargs))
        return '{"result": "async ok"}'

    reg.register(
        "test_async_tool", "core", {}, async_handler,
        is_async=True,
    )
    result = await reg.dispatch("test_async_tool", {"x": 2})
    assert result == '{"result": "async ok"}'
    assert call_log == [("async", {"x": 2}, {})]


async def test_concurrency_safe_field_defaults_false():
    """register 默认 isConcurrencySafe=False。"""
    reg = ToolRegistry()

    def handler(args, **kwargs):
        return '{}'

    reg.register("t", "core", {}, handler)
    entry = reg.get("t")
    assert entry is not None, "reg.get('t') 应返回 ToolEntry"
    assert entry.isConcurrencySafe is False, \
        "默认 isConcurrencySafe 应为 False（安全默认 > 事后补救）"


async def test_concurrency_safe_field_can_be_set():
    """register 可以显式传 isConcurrencySafe=True。"""
    reg = ToolRegistry()

    def handler(args, **kwargs):
        return '{}'

    reg.register(
        "t_safe", "core", {}, handler,
        isConcurrencySafe=True,
    )
    entry = reg.get("t_safe")
    assert entry is not None
    assert entry.isConcurrencySafe is True, \
        "显式传 isConcurrencySafe=True 应落到 entry 上"
