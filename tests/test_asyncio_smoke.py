"""验证 pytest-asyncio mode=auto 生效。"""
import asyncio

import anyio


async def test_async_test_runs():
    """async def test 自动被 pytest-asyncio 当 async 测试跑。"""
    result = await asyncio.sleep(0, result="ok")
    assert result == "ok"


async def test_anyio_importable():
    """anyio 已装且可 import。"""
    assert hasattr(anyio, "to_thread")
    assert hasattr(anyio.to_thread, "run_sync")
