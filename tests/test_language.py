"""多语言 system prompt 测试。"""

import pytest

from agent.prompt_builder import (
    build_system_prompt,
    IDENTITY_ZH,
    IDENTITY_EN,
    OUTPUT_CONVENTION_ZH,
    OUTPUT_CONVENTION_EN,
    IDENTITY,  # 向后兼容别名
)


def test_default_language_is_zh():
    """不传 language 参数时默认中文。"""
    prompt = build_system_prompt(include_guidance=False)
    assert IDENTITY_ZH in prompt
    assert OUTPUT_CONVENTION_ZH in prompt
    assert "使用中文回复" in prompt


def test_explicit_zh_language():
    prompt = build_system_prompt(language="zh", include_guidance=False)
    assert IDENTITY_ZH in prompt
    assert OUTPUT_CONVENTION_ZH in prompt


def test_en_language():
    """language=en 应该用英文身份和输出约定。"""
    prompt = build_system_prompt(language="en", include_guidance=False)
    assert IDENTITY_EN in prompt
    assert "Respond in English" in prompt
    # 不应包含中文输出约定
    assert "使用中文回复" not in prompt


def test_identity_backward_compat_alias():
    """IDENTITY 别名应等于 IDENTITY_ZH（向后兼容）。"""
    assert IDENTITY == IDENTITY_ZH


def test_unknown_language_falls_back_to_zh():
    """未知 language 值应回退到中文（默认）。"""
    prompt = build_system_prompt(language="fr", include_guidance=False)
    assert IDENTITY_ZH in prompt


def test_language_does_not_affect_guidance():
    """language 只影响身份/输出约定；GUIDANCE 段保持中文（暂未多语言化）。"""
    prompt_en = build_system_prompt(language="en", include_guidance=True)
    # GUIDANCE 段仍是中文
    assert "记忆系统使用指南" in prompt_en
    assert "技能系统指南" in prompt_en


def test_config_default_has_language():
    """DEFAULT_CONFIG 应含 language 字段。"""
    from config import DEFAULT_CONFIG
    assert "language" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["language"] == "zh"
