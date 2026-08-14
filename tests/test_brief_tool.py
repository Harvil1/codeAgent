import inspect
import json
from tools.brief_tool import _handle_brief, BRIEF_SCHEMA


def test_brief_schema_required_fields():
    """schema 必须有 headline（必填）+ steps/risks/audience（可选）。"""
    assert BRIEF_SCHEMA["name"] == "brief"
    props = BRIEF_SCHEMA["parameters"]["properties"]
    assert "headline" in props
    assert "steps" in props
    assert "risks" in props
    assert "audience" in props
    assert "headline" in BRIEF_SCHEMA["parameters"]["required"]


def test_brief_minimal_headline_only():
    # 模拟 dispatch 真实调用：handler(args, **dispatch_kwargs)
    result = _handle_brief({"headline": "测试标题"}, memory_store=None, agent_ref=None)
    data = json.loads(result)
    assert data["headline"] == "测试标题"
    assert data["steps"] == []
    assert data["risks"] == []
    assert data["audience"] == "user"


def test_brief_full_format():
    # 模拟 dispatch 真实调用：工具参数全在 args
    result = _handle_brief(
        {
            "headline": "重写认证模块",
            "steps": ["step1", "step2"],
            "risks": ["风险1"],
            "audience": "approval",
        },
        memory_store=None,
        agent_ref=None,
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


def test_handler_signature_matches_dispatch_contract():
    """dispatch 调 handler(args, **kwargs)，签名必须兼容（防 silent-dead-code）。

    历史教训：曾写成 (args, kwargs, ctx) 三位置参数，单元测试直调三参数
    漏检，生产 dispatch 调用 100% TypeError（silent-dead-code）。
    """
    sig = inspect.signature(_handle_brief)
    params = list(sig.parameters.values())
    # 第一个参数是位置参数（args）
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[0].name == "args"
    # 必须有 **kwargs 接收 dispatch 上下文
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
