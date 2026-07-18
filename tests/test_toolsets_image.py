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


def test_image_tools_count_matches():
    """core 应有 19 个工具（原 17 + image_analyze + image_ocr）。

    防止后续误删或重复添加。
    """
    core = resolve_toolset("core")
    assert len(core) == 19, f"core 工具数应为 19，实际 {len(core)}: {core}"
