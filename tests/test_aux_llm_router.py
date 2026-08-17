"""AuxLLMRouter 多 endpoint 熔断链测试（07）。

验证：
1. 主 endpoint 失败后自动切到第二个
2. 连续 3 次失败触发熔断
3. 熔断期间跳过 endpoint
4. 所有 endpoint 失败时降级到主 client
5. api_key 未配置的 endpoint 被跳过
6. 老的 aux_config 兼容
7. endpoints 按优先级排序
8. 成功后失败计数重置

测试策略：mock agent.llm_client.create_llm_client，根据 model 名返回
对应的 _FakeClient。__init__ eager 创建时 patch 生效；chat_completions
时 patch 已退，但 client 已缓存。
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.aux_llm import AuxLLMRouter, LLMEndpoint


class _FakeClient:
    """假 LLM client（async chat_completions，对齐 LLMClient async 接口）。"""

    def __init__(self, name: str, *, succeed=True, response=None):
        self.name = name
        self._succeed = succeed
        self._response = response or f"resp from {name}"
        self.call_count = 0

    async def chat_completions(self, messages, **kwargs):
        self.call_count += 1
        if not self._succeed:
            raise RuntimeError(f"{self.name} failed")
        return SimpleNamespace(content=self._response)


def _build_router(main_client, fakes_by_model, endpoint_specs):
    """构造 router，patch create_llm_client 按 model 返回 fake。"""
    def fake_create(cfg):
        return fakes_by_model.get(cfg["model"])

    with patch("agent.llm_client.create_llm_client", side_effect=fake_create):
        return AuxLLMRouter(
            main_client=main_client,
            endpoints=endpoint_specs,
        )


def _ep(name, model, priority=100, **kw):
    """便捷构造 endpoint（默认带 api_key_default 让创建成功）。"""
    return LLMEndpoint(
        name=name, base_url=kw.get("base_url", "http://x"),
        model=model, priority=priority,
        api_key_env=kw.get("api_key_env", ""),
        api_key_default=kw.get("api_key_default", "test-key"),
        enabled=kw.get("enabled", True),
    )


# ----------------------------------------------------------------------------
# 多 endpoint fallback
# ----------------------------------------------------------------------------

async def test_first_endpoint_success():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1")
    router = _build_router(
        main, {"ep1-model": ep1},
        [_ep("ep1", "ep1-model")],
    )
    resp = await router.chat_completions([])
    assert ep1.call_count == 1
    assert main.call_count == 0


async def test_first_fails_second_succeeds():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1", succeed=False)
    ep2 = _FakeClient("ep2")
    router = _build_router(
        main,
        {"ep1-model": ep1, "ep2-model": ep2},
        [_ep("ep1", "ep1-model", priority=1),
         _ep("ep2", "ep2-model", priority=2)],
    )
    resp = await router.chat_completions([])
    assert ep1.call_count == 1
    assert ep2.call_count == 1
    assert main.call_count == 0
    assert "ep2" in resp.content


async def test_all_endpoints_fail_falls_back_to_main():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1", succeed=False)
    ep2 = _FakeClient("ep2", succeed=False)
    router = _build_router(
        main,
        {"ep1-model": ep1, "ep2-model": ep2},
        [_ep("ep1", "ep1-model"), _ep("ep2", "ep2-model")],
    )
    resp = await router.chat_completions([])
    assert "main" in resp.content
    assert main.call_count == 1


# ----------------------------------------------------------------------------
# 熔断
# ----------------------------------------------------------------------------

async def test_circuit_opens_after_threshold_failures():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1", succeed=False)
    router = _build_router(main, {"m": ep1}, [_ep("ep1", "m")])

    await router.chat_completions([])
    await router.chat_completions([])
    status = router.get_circuit_status()
    assert status["ep1"]["failures"] == 2
    assert status["ep1"]["circuit_open"] is False

    # 第 3 次失败 → 熔断
    await router.chat_completions([])
    status = router.get_circuit_status()
    assert status["ep1"]["circuit_open"] is True
    assert status["ep1"]["circuit_remaining_seconds"] > 0


async def test_circuit_skips_endpoint_when_open():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1", succeed=False)
    ep2 = _FakeClient("ep2", succeed=True)
    router = _build_router(
        main,
        {"ep1-model": ep1, "ep2-model": ep2},
        [_ep("ep1", "ep1-model", priority=1),
         _ep("ep2", "ep2-model", priority=2)],
    )
    # ep1 失败 3 次，触发熔断
    for _ in range(3):
        await router.chat_completions([])
    assert router.get_circuit_status()["ep1"]["circuit_open"] is True

    ep1_calls_before = ep1.call_count
    resp = await router.chat_completions([])
    # ep1 没被调用（熔断中）
    assert ep1.call_count == ep1_calls_before
    # ep2 被调用
    assert ep2.call_count >= 1
    assert "ep2" in resp.content


async def test_success_resets_failure_count():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1", succeed=True)
    router = _build_router(main, {"m": ep1}, [_ep("ep1", "m")])
    # 人为制造 2 次失败计数
    router._failure_counts["ep1"] = 2
    await router.chat_completions([])
    assert router._failure_counts["ep1"] == 0


def test_reset_circuit():
    main = _FakeClient("main")
    ep1 = _FakeClient("ep1")
    router = _build_router(main, {"m": ep1}, [_ep("ep1", "m")])
    router._failure_counts["ep1"] = 5
    router._circuit_open_until["ep1"] = 99999999999
    assert router.get_circuit_status()["ep1"]["circuit_open"] is True

    router.reset_circuit("ep1")
    assert router.get_circuit_status()["ep1"]["circuit_open"] is False
    assert router.get_circuit_status()["ep1"]["failures"] == 0


# ----------------------------------------------------------------------------
# api_key 未配置
# ----------------------------------------------------------------------------

async def test_endpoint_without_api_key_skipped(monkeypatch):
    """api_key_env 没设 + 没 default → endpoint 被跳过。"""
    monkeypatch.delenv("FAKE_KEY_FOR_TEST", raising=False)
    main = _FakeClient("main")
    # 不 patch create_llm_client：因为 _create_client_for 在 api_key="" 时直接返回 None
    router = AuxLLMRouter(
        main_client=main,
        endpoints=[
            LLMEndpoint(
                name="ep1", base_url="http://x", model="m",
                api_key_env="FAKE_KEY_FOR_TEST",
                api_key_default="",
            ),
        ],
    )
    # ep1 没创建 client，所以 _endpoints 为空
    assert "ep1" not in router._client_cache
    assert router.is_aux_configured is False
    # 直接走 main
    resp = await router.chat_completions([])
    assert main.call_count == 1


# ----------------------------------------------------------------------------
# 老配置兼容
# ----------------------------------------------------------------------------

async def test_legacy_aux_config_compatible():
    main = _FakeClient("main")
    aux = _FakeClient("legacy")
    with patch("agent.llm_client.create_llm_client", return_value=aux):
        router = AuxLLMRouter(
            main_client=main,
            aux_config={
                "format": "openai",
                "base_url": "http://aux/v1",
                "api_key": "aux-key",
                "model": "cheap-model",
            },
        )
    assert router.is_aux_configured is True
    assert len(router._endpoints) == 1
    assert router._endpoints[0].name == "legacy_aux"
    # 调用走 aux
    await router.chat_completions([])
    assert aux.call_count == 1
    assert main.call_count == 0


async def test_no_config_returns_main_only():
    main = _FakeClient("main")
    router = AuxLLMRouter(main_client=main)
    assert router.is_aux_configured is False
    await router.chat_completions([])
    assert main.call_count == 1


# ----------------------------------------------------------------------------
# 优先级排序
# ----------------------------------------------------------------------------

def test_endpoints_sorted_by_priority():
    main = _FakeClient("main")
    router = _build_router(
        main,
        {
            "high-model": _FakeClient("high"),
            "mid-model": _FakeClient("mid"),
            "low-model": _FakeClient("low"),
        },
        [
            _ep("low", "low-model", priority=99),
            _ep("high", "high-model", priority=1),
            _ep("mid", "mid-model", priority=50),
        ],
    )
    names = [e.name for e in router._endpoints]
    assert names == ["high", "mid", "low"]


def test_disabled_endpoint_filtered():
    main = _FakeClient("main")
    router = AuxLLMRouter(
        main_client=main,
        endpoints=[
            _ep("on", "on-model", enabled=True),
            _ep("off", "off-model", enabled=False),
        ],
    )
    names = [e.name for e in router._endpoints]
    assert "off" not in names
    assert "on" in names


# ----------------------------------------------------------------------------
# 集成：从 endpoints 配置构造
# ----------------------------------------------------------------------------

async def test_build_router_from_endpoints_config():
    """模拟 cli.py 从 config 构造 router。"""
    main = _FakeClient("main")
    fake_deepseek = _FakeClient("deepseek")
    fake_openrouter = _FakeClient("openrouter")

    def fake_create(cfg):
        return {"deepseek-chat": fake_deepseek,
                "claude-3-haiku": fake_openrouter}.get(cfg["model"])

    endpoints_cfg = [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
         "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat", "priority": 1},
        {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1",
         "api_key_env": "OPENROUTER_API_KEY", "model": "claude-3-haiku",
         "priority": 2},
    ]
    endpoints = [
        LLMEndpoint(
            name=ep["name"], base_url=ep["base_url"],
            api_key_env=ep.get("api_key_env", ""),
            api_key_default=ep.get("api_key_default", "test-key"),
            model=ep["model"],
            priority=ep.get("priority", 100),
            enabled=ep.get("enabled", True),
        )
        for ep in endpoints_cfg
    ]
    # patch env 让 api_key 解析成功
    import os
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k1", "OPENROUTER_API_KEY": "k2"}):
        with patch("agent.llm_client.create_llm_client", side_effect=fake_create):
            router = AuxLLMRouter(main_client=main, endpoints=endpoints)

    assert router.is_aux_configured is True
    assert [e.name for e in router._endpoints] == ["deepseek", "openrouter"]
    await router.chat_completions([])
    assert fake_deepseek.call_count == 1
