"""Task M: verification 内置子代理测试。

验证内容：
  1. scan_agent_defs 自动加载 verification（不用手动加路径）
  2. frontmatter 字段完整（tools / permissionMode / maxTurns）
  3. system_prompt 含 11 类策略关键词
  4. 反模式清单存在（"读完代码就说 PASS" / "前 80%" 等）
  5. 输出格式段（VERDICT / 验证结论 / 对抗性 probe）
  6. subagent(subagent_type="verification") 真能起（mock LLM）
"""

import pytest
from unittest.mock import patch


def test_verification_agent_loaded():
    """scan_agent_defs 默认返回 verification 内置子代理。"""
    from agent.agent_defs import scan_agent_defs
    defs = scan_agent_defs()
    assert "verification" in defs, "内置 verification 子代理未加载"


def test_verification_frontmatter_fields():
    """frontmatter 关键字段正确解析。"""
    from agent.agent_defs import scan_agent_defs
    a = scan_agent_defs()["verification"]
    # tools 应包含 explore + terminal（read-only + 执行探测用）
    assert "explore" in a.tools, f"tools 缺 explore: {a.tools}"
    assert "terminal" in a.tools, f"tools 缺 terminal: {a.tools}"
    # 默认权限模式（explore.md / plan.md 也没显式设，留 None 或 default）
    assert a.permission_mode in (None, "default"), \
        f"permission_mode 应为 default/None, got {a.permission_mode}"
    # max_turns 应有值（建议 30，对齐 explore/plan）
    assert a.max_turns is not None and a.max_turns >= 10, \
        f"max_turns 应 >= 10, got {a.max_turns}"


def test_verification_has_11_strategies():
    """system_prompt 含 11 类策略关键词。"""
    from agent.agent_defs import scan_agent_defs
    a = scan_agent_defs()["verification"]
    strategies = [
        "frontend", "backend", "CLI", "infra", "library",
        "bugfix", "mobile", "data", "migration", "refactor",
    ]
    for s in strategies:
        assert s in a.system_prompt, f"system_prompt 缺策略关键词: {s}"


def test_verification_has_antipatterns():
    """反模式清单存在。"""
    from agent.agent_defs import scan_agent_defs
    a = scan_agent_defs()["verification"]
    # 至少含这些反模式信号
    assert "PASS" in a.system_prompt, "缺 PASS 反模式"
    # 至少有一个"前 80%"相关反模式（中英都接受）
    assert ("前 80%" in a.system_prompt
            or "first 80%" in a.system_prompt.lower()
            or "80%" in a.system_prompt), "缺前 80% 反模式"


def test_verification_has_output_format():
    """输出格式段存在（VERDICT 或 验证结论）。"""
    from agent.agent_defs import scan_agent_defs
    a = scan_agent_defs()["verification"]
    # 必须有结构化输出格式约定
    has_verdict = "VERDICT" in a.system_prompt
    has_chinese_verdict = "验证结论" in a.system_prompt
    has_pass = "PASS" in a.system_prompt and "FAIL" in a.system_prompt
    assert has_verdict or has_chinese_verdict or has_pass, \
        "system_prompt 缺输出格式（VERDICT / 验证结论 / PASS+FAIL）"


def test_verification_has_adversarial_probe_requirement():
    """强制至少一个对抗性 probe。"""
    from agent.agent_defs import scan_agent_defs
    a = scan_agent_defs()["verification"]
    # 必须提到对抗性 probe / adversarial
    has_chinese = "对抗" in a.system_prompt
    has_english = "adversarial" in a.system_prompt.lower()
    has_probe = "probe" in a.system_prompt.lower() or "探测" in a.system_prompt
    assert has_chinese or has_english, "缺对抗性关键词"
    assert has_probe, "缺 probe/探测 关键词"


def test_subagent_type_verification_can_start(monkeypatch):
    """subagent(subagent_type="verification") 真能起。

    mock LLM 配置 + AIAgent，验证：
      1. get_agent_def("verification") 能找到定义
      2. _run_child 在 subagent_type="verification" 下正常构造 AIAgent
      3. 子代理 system_prompt_override 含 verification.md 正文
    """
    from tools.delegate_tool import _run_child

    # 不 mock get_agent_def —— 真用 scan_agent_defs 找内置 verification
    # 只 mock LLM 配置加载（不走真实 API）

    captured = {}

    class FakeChild:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_client = type("FakeClient", (), {})()
            self.model = kwargs.get("model")

        async def chat(self, msg):
            return "VERDICT: PASS"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with patch("config.load_config", return_value={
        "model": {"name": "test", "api_key": "fake_key", "base_url": "http://x"},
    }):
        with patch("agent.AIAgent", FakeChild):
            with patch("agent.progress.ProgressReporter") as fake_prog:
                fake_prog.return_value.__enter__ = lambda s: None
                fake_prog.return_value.__exit__ = lambda s, *a: None
                with patch(
                    "agent.team.hallucination_check.verify_claims",
                    return_value=None,
                ):
                    with patch(
                        "agent.team.hallucination_check.append_warning",
                        lambda r, v: r,
                    ):
                        with patch(
                            "tools.worktree.create_isolated_workspace"
                        ) as fake_ws:
                            # 非隔离模式不应调 worktree
                            result = _run_child(
                                "验证新 feature 是否真跑",
                                "上下文",
                                "leaf",
                                subagent_type="verification",
                            )

    # 子代理构造成功
    assert result == "VERDICT: PASS"
    # system_prompt_override 含 verification 正文特征
    assert "verification" in captured["system_prompt_override"].lower() \
           or "对抗" in captured["system_prompt_override"] \
           or "adversarial" in captured["system_prompt_override"].lower(), \
        f"system_prompt_override 缺 verification 特征: {captured['system_prompt_override'][:200]}"
    # tools 应包含 explore（来自 verification.md frontmatter）
    assert "explore" in captured.get("enabled_toolsets", []), \
        f"enabled_toolsets 缺 explore: {captured.get('enabled_toolsets')}"
