"""R26 终审 follow-up：ThreadedLLMClient 跨事件循环安全（per-call 独立 client）。"""


class TestThreadedLLMClient:
    def test_fresh_client_per_call_and_closed(self, monkeypatch):
        """两次调用（各自独立 asyncio.run 循环）→ 两个独立 client，都关闭。"""
        import asyncio
        from agent import llm_client as LC

        created = []

        class FakeInner:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        class FakeClient:
            def __init__(self, cfg):
                self.cfg = cfg
                self.client = FakeInner()
                created.append(self)

            async def chat_completions(self, messages, *, tools=None, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(choices=[
                    SimpleNamespace(message=SimpleNamespace(content="ok"))
                ])

        monkeypatch.setattr(LC, "create_llm_client", FakeClient)
        tlc = LC.ThreadedLLMClient({"format": "openai", "model": "m", "api_key": "k"})
        r1 = asyncio.run(tlc.chat_completions([{"role": "user", "content": "x"}]))
        r2 = asyncio.run(tlc.chat_completions([{"role": "user", "content": "y"}]))
        assert r1.choices[0].message.content == "ok"
        assert len(created) == 2  # per-call 独立
        assert all(c.client.closed for c in created)  # 用完即关

    def test_aclose_tolerates_sync_close_and_missing(self):
        import asyncio
        from agent.llm_client import aclose_llm_client

        class SyncClose:
            def __init__(self):
                self.client = type("C", (), {"close": lambda self: None})()

        class NoAttr:
            pass

        asyncio.run(aclose_llm_client(SyncClose()))  # 不抛
        asyncio.run(aclose_llm_client(NoAttr()))     # 不抛

    def test_thread_helper_returns_threaded_client(self, monkeypatch):
        """cli._thread_llm_client：正常 config → ThreadedLLMClient（不建真连接）。"""
        from cli import _thread_llm_client
        from agent.llm_client import ThreadedLLMClient
        cfg = {"model": {"format": "openai", "base_url": "https://x", "name": "m",
                         "api_key": "sk-test"}}
        c = _thread_llm_client(cfg)
        assert isinstance(c, ThreadedLLMClient)
        assert c._model_config["model"] == "m"

    def test_thread_helper_env_fallback(self, monkeypatch):
        """settings 无 key 时按 base_url 域名兜底环境变量（与主推导链一致）。"""
        from cli import _thread_llm_client
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        cfg = {"model": {"format": "openai",
                         "base_url": "https://api.deepseek.com/v1",
                         "name": "deepseek-chat"}}
        c = _thread_llm_client(cfg)
        assert c._model_config["api_key"] == "sk-env"

    def test_thread_helper_auth_token_preferred(self):
        """auth_token 场景：auth_token 键独立透传（anthropic 分支 auth_token 优先 Bearer）。"""
        from cli import _thread_llm_client
        cfg = {"model": {"format": "anthropic", "name": "m",
                         "base_url": "https://x", "auth_token": "tok-1"}}
        c = _thread_llm_client(cfg)
        assert c._model_config["auth_token"] == "tok-1"
        # 现场实际：_derive_api_key 末尾 `api_key or auth_token` 折叠保持不动，
        # api_key 字段仍为 "tok-1"；anthropic 分支 auth_token 优先，不受影响。
        assert c._model_config["api_key"] == "tok-1"

    def test_thread_helper_auth_token_passthrough(self):
        """auth_token 独立透传（anthropic + Bearer 端点不被折叠丢失）。"""
        from cli import _thread_llm_client
        cfg = {"model": {"format": "anthropic", "base_url": "https://x",
                          "name": "m", "auth_token": "tok-abc"}}
        c = _thread_llm_client(cfg)
        assert c._model_config["auth_token"] == "tok-abc"
        # 现场实际修正：_derive_api_key 保持折叠（api_key == "tok-abc"），
        # 修复核心是 auth_token 键存在且被 anthropic 分支优先消费（Bearer）。
        assert c._model_config["api_key"] == "tok-abc"
