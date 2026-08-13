import json
from tools.brief_tool import _handle_brief, BRIEF_SCHEMA


def test_brief_schema_required_fields():
    """schema 必须有 headline（必填）+ steps/risks/audience（可选）。"""
    assert BRIEF_SCHEMA["name"] == "brief"
    props = BRIEF_SCHEMA["inputSchema"]["properties"]
    assert "headline" in props
    assert "steps" in props
    assert "risks" in props
    assert "audience" in props
    assert "headline" in BRIEF_SCHEMA["inputSchema"]["required"]


def test_brief_minimal_headline_only():
    result = _handle_brief({}, {"headline": "测试标题"}, None)
    data = json.loads(result)
    assert data["headline"] == "测试标题"
    assert data["steps"] == []
    assert data["risks"] == []
    assert data["audience"] == "user"


def test_brief_full_format():
    result = _handle_brief(
        {},
        {
            "headline": "重写认证模块",
            "steps": ["step1", "step2"],
            "risks": ["风险1"],
            "audience": "approval",
        },
        None,
    )
    data = json.loads(result)
    assert data["headline"] == "重写认证模块"
    assert data["steps"] == ["step1", "step2"]
    assert data["risks"] == ["风险1"]
    assert data["audience"] == "approval"


def test_brief_registered_in_registry():
    """模块 import 后 brief 应该已注册到 registry。"""
    import tools.brief_tool  # noqa: 触发注册
    from tools.registry import registry
    entry = registry.get("brief")
    assert entry is not None
    assert entry.toolset == "core"
