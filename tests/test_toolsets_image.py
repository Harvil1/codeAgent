"""回归测试：image_analyze / image_ocr 必须在 core toolset 中可见。

历史 bug：B1 任务只 registry.register(toolset="core")，但 toolsets.py:_CORE_TOOLS
静态列表没追加，导致 resolve_toolset("core") 看不到，LLM 无法调用。
"""

from toolsets import resolve_toolset


def test_image_tools_in_core():
    """image_analyze 和 image_ocr 都应在 core toolset 中。"""
    core = resolve_toolset("core")
    assert "image_analyze" in core, (
        "image_analyze 未在 core toolset 中暴露给 LLM — "
        "检查 toolsets.py:_CORE_TOOLS 是否包含"
    )
    assert "image_ocr" in core


def test_core_toolset_dedup_still_works():
    """core toolset 去重逻辑正常（不因新增工具破坏）。"""
    core = resolve_toolset("core")
    assert len(core) == len(set(core)), "core toolset 有重复项"


def test_all_core_registered_tools_are_visible():
    """任何声明 toolset="core" 的工具都必须出现在 resolve_toolset("core") 中。

    防止历史 bug：registry.register(toolset="core") 但 toolsets.py:_CORE_TOOLS
    没追加 → LLM 看不到。不写死数量（core 随新增工具增长）。
    """
    from model_tools import ensure_tools_discovered
    from tools.registry import registry

    ensure_tools_discovered()
    core = set(resolve_toolset("core"))
    missing = [
        name for name, entry in registry._tools.items()
        if entry.toolset == "core" and name not in core
    ]
    assert not missing, f"声明 core 但未在 _CORE_TOOLS 中暴露: {missing}"
