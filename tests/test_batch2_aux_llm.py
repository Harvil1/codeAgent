"""batch2-T3: AuxLLMRouter 统一辅助 LLM 路由器测试。"""
from unittest.mock import MagicMock, patch

import pytest

from agent.aux_llm import AuxLLMRouter


# ---------------------------------------------------------------------------
# 无 aux 配置时 fallback 到主 client
# ---------------------------------------------------------------------------

def test_aux_router_no_aux_config_uses_main():
    """aux_config=None 时直接用主 client。"""
    main = MagicMock()
    main.chat_completions.return_value = "main_response"

    router = AuxLLMRouter(main_client=main, main_model="main-model")
    assert not router.is_aux_configured

    result = router.chat_completions([{"role": "user", "content": "hi"}])
    main.chat_completions.assert_called_once()
    assert result == "main_response"


def test_aux_router_empty_aux_config_uses_main():
    """aux_config={} 时直接用主 client。"""
    main = MagicMock()
    main.chat_completions.return_value = "main_response"

    router = AuxLLMRouter(main_client=main, main_model="m", aux_config={})
    assert not router.is_aux_configured


def test_aux_router_none_model_in_config_uses_main():
    """aux_config 有键但 model 为空 → 不启用 aux。"""
    main = MagicMock()
    router = AuxLLMRouter(
        main_client=main, main_model="m",
        aux_config={"base_url": "http://x", "api_key": "k", "model": None},
    )
    assert not router.is_aux_configured


# ---------------------------------------------------------------------------
# 有 aux 配置时优先用 aux
# ---------------------------------------------------------------------------

def test_aux_router_uses_aux_when_configured():
    """配置了 aux → 优先用 aux client。"""
    main = MagicMock()
    aux = MagicMock()
    aux.chat_completions.return_value = "aux_response"

    with patch("agent.llm_client.create_llm_client", return_value=aux):
        router = AuxLLMRouter(
            main_client=main, main_model="main-model",
            aux_config={
                "format": "openai",
                "base_url": "http://aux.api/v1",
                "api_key": "aux-key",
                "model": "cheap-model",
            },
        )

    assert router.is_aux_configured
    result = router.chat_completions([{"role": "user", "content": "hi"}])
    aux.chat_completions.assert_called_once()
    main.chat_completions.assert_not_called()
    assert result == "aux_response"


# ---------------------------------------------------------------------------
# aux 失败时 fallback 到主 client
# ---------------------------------------------------------------------------

def test_aux_router_fallback_on_aux_failure():
    """aux 抛异常 → 自动 fallback 到主 client。"""
    main = MagicMock()
    main.chat_completions.return_value = "main_response"
    aux = MagicMock()
    aux.chat_completions.side_effect = RuntimeError("aux API down")

    with patch("agent.llm_client.create_llm_client", return_value=aux):
        router = AuxLLMRouter(
            main_client=main, main_model="main",
            aux_config={
                "format": "openai",
                "base_url": "http://aux.api/v1",
                "api_key": "aux-key",
                "model": "cheap-model",
            },
        )

    assert router.is_aux_configured
    result = router.chat_completions([{"role": "user", "content": "hi"}])
    # aux 被调了但失败
    aux.chat_completions.assert_called_once()
    # main 被 fallback 调了
    main.chat_completions.assert_called_once()
    assert result == "main_response"


# ---------------------------------------------------------------------------
# aux client 创建失败时 fallback
# ---------------------------------------------------------------------------

def test_aux_router_aux_create_failure_uses_main():
    """aux client 创建失败 → 直接用主 client。"""
    main = MagicMock()

    with patch("agent.llm_client.create_llm_client",
               side_effect=ConnectionError("can't connect")):
        router = AuxLLMRouter(
            main_client=main, main_model="main",
            aux_config={
                "format": "openai",
                "base_url": "http://aux.api/v1",
                "api_key": "aux-key",
                "model": "cheap-model",
            },
        )

    assert not router.is_aux_configured
    main.chat_completions.return_value = "main_response"
    result = router.chat_completions([])
    main.chat_completions.assert_called_once()
    assert result == "main_response"


# ---------------------------------------------------------------------------
# kwargs 透传
# ---------------------------------------------------------------------------

def test_aux_router_passes_kwargs():
    """kwargs 应该透传给底层 client。"""
    main = MagicMock()
    main.chat_completions.return_value = "ok"

    router = AuxLLMRouter(main_client=main, main_model="m")
    router.chat_completions(
        [{"role": "user", "content": "x"}],
        model="custom-model",
        temperature=0.1,
    )
    _, kwargs = main.chat_completions.call_args
    assert kwargs["model"] == "custom-model"
    assert kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# 配置测试
# ---------------------------------------------------------------------------

def test_default_config_has_aux_model():
    """config.py DEFAULT_CONFIG 应含 aux_model 键。"""
    from config import DEFAULT_CONFIG
    assert "aux_model" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["aux_model"] is None


# ---------------------------------------------------------------------------
# AIAgent 集成
# ---------------------------------------------------------------------------

def test_agent_accepts_aux_llm_router():
    """AIAgent 能接收 aux_llm_router 参数。"""
    from agent import AIAgent

    router = MagicMock()
    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[],
        aux_llm_router=router,
    )
    assert agent.aux_llm_router is router


def test_agent_aux_llm_router_defaults_none():
    """AIAgent aux_llm_router 默认 None。"""
    from agent import AIAgent

    agent = AIAgent(
        api_key="fake", model="test",
        enabled_toolsets=[],
    )
    assert agent.aux_llm_router is None
