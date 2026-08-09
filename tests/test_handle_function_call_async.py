"""handle_function_call async 契约测试。

Task C2：把 handle_function_call 从 sync 改 async。
契约：
- 必须是 coroutine function
- 仍返回 JSON 字符串（不变）
"""
import inspect

import pytest

from model_tools import handle_function_call


def test_handle_function_call_is_coroutine():
    """handle_function_call 必须是 async def。"""
    assert inspect.iscoroutinefunction(handle_function_call), \
        "handle_function_call 必须是 async def（Task C2 改造）"


async def test_handle_function_call_returns_json_string():
    """仍返回 JSON 字符串（不变）。"""
    from tools.registry import ToolRegistry

    reg = ToolRegistry()

    def fake_read(args, **kwargs):
        return '{"content": "hello"}'

    reg.register("test_tool", "core", schema={}, handler=fake_read)

    # mock ensure_tools_discovered + 替换全局 registry
    import model_tools

    orig_ensure = model_tools.ensure_tools_discovered
    orig_registry = model_tools.registry
    model_tools.ensure_tools_discovered = lambda: None
    model_tools.registry = reg
    try:
        result = await handle_function_call(
            "test_tool", {"path": "/tmp/x"},
            config={},
        )
    finally:
        model_tools.ensure_tools_discovered = orig_ensure
        model_tools.registry = orig_registry

    assert isinstance(result, str)
    assert "content" in result
