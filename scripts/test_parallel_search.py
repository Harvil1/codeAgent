"""默认搜索回归：只替换 HTTP 边界，保留设置、握手、注册、分发和结果消费链。

运行：uv run pytest scripts/test_parallel_search.py
不调用真实模型，也不需要搜索或模型密钥。
"""

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from agent import AIAgent
from agent import mcp_client, tool_permissions
from agent.settings import load_settings, save_settings
from config import load_config
from model_tools import get_tool_definitions, handle_function_call
from tools import mcp_tool
from tools.registry import registry


ENDPOINT = "https://search.parallel.ai/mcp"
SEARCH = "mcp__parallel-search__web_search"
FETCH = "mcp__parallel-search__web_fetch"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """隔离用户数据、工作目录、共享 manager 和凭据，捕获真实 HTTP 请求。"""
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEAGENT_HOME", str(home))
    for key in list(__import__("os").environ):
        if any(part in key.upper() for part in ("API_KEY", "AUTH_TOKEN")):
            monkeypatch.delenv(key, raising=False)
    manager = mcp_client.MCPManager()
    monkeypatch.setattr(mcp_client, "_mcp_manager", manager)
    tool_permissions.reset_rules_cache()
    state = SimpleNamespace(home=home, workspace=workspace, manager=manager,
                            requests=[], failure=None, entered=threading.Event(),
                            release=threading.Event())

    def respond(request):
        state.requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(200, json={})
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [
                {"name": "web_search", "description": "Search current documentation",
                 "inputSchema": {"type": "object", "properties": {
                     "objective": {"type": "string"},
                     "search_queries": {"type": "array", "items": {"type": "string"}}},
                     "required": ["objective", "search_queries"]}},
                {"name": "web_fetch", "description": "Read page excerpts",
                 "inputSchema": {"type": "object", "properties": {
                     "urls": {"type": "array", "items": {"type": "string"}}},
                     "required": ["urls"]}},
            ]}
        elif method == "tools/call":
            if state.failure == "timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            if state.failure == "http":
                return httpx.Response(503, text="temporarily unavailable")
            if state.failure == "rpc":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"],
                                                "error": {"code": -32602, "message": "invalid arguments"}})
            if state.failure == "blocking":
                state.entered.set()
                assert state.release.wait(3)
            if state.failure == "tool":
                result = {"isError": True, "content": [{"type": "text", "text": "service rejected query"}]}
            else:
                result = {"content": [{"type": "text", "text": json.dumps({
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "excerpt": "asyncio provides APIs to run coroutines concurrently."})}]}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    class FixtureClient(httpx.Client):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "transport": httpx.MockTransport(respond)})

    monkeypatch.setattr(httpx, "Client", FixtureClient)
    yield state
    state.release.set()
    manager.close_all()
    for name in registry.list_all():
        if name.startswith("mcp__"):
            registry.unregister(name)
    tool_permissions.reset_rules_cache()


def test_default_loader_dispatch_and_history(runtime):
    """从全新 profile 启动，搜索和抓取都经过生产分发器并进入对话历史。"""
    settings = load_settings()
    assert settings["enabled_toolsets"] == ["core", "mcp"]
    assert mcp_tool.initialize_mcp() == 4
    config = load_config()
    names = [d["function"]["name"] for d in get_tool_definitions(config["enabled_toolsets"])]
    assert SEARCH in names and FETCH in names

    async def dispatch():
        schemas = json.loads(await handle_function_call("tool_search", {"query": "web_search"}))
        assert schemas["results"][0]["parameters"]["required"] == ["objective", "search_queries"]
        agent = AIAgent(api_key="fixture", model="fixture", config=config,
                        enabled_toolsets=config["enabled_toolsets"], session_id="local-conversation")
        for name, args in [(SEARCH, {"objective": "Python asyncio documentation", "search_queries": ["Python asyncio docs"]}),
                           (FETCH, {"urls": ["https://docs.python.org/3/library/asyncio.html"]})]:
            tc = SimpleNamespace(id=name, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            content = await agent._run_unsafe_tool_call(tc, handle_function_call)
            agent._merge_results_in_order([tc], [], [(tc, content)])
            result = json.loads(agent.conversation_history[-1]["content"])
            text = json.loads(result["content"][0]["text"])
            assert text["url"] == "https://docs.python.org/3/library/asyncio.html"
            assert "coroutines" in text["excerpt"]
        assert agent.conversation_history[-1]["role"] == "tool"

    asyncio.run(dispatch())
    assert runtime.requests
    for request in runtime.requests:
        assert str(request.url) == ENDPOINT
        assert request.headers["user-agent"] == "CodeAgent/0.1.0"
        assert "authorization" not in request.headers
        assert "text/event-stream" in request.headers["accept"]
    calls = [json.loads(r.content) for r in runtime.requests if r.method == "POST"]
    assert {p["method"] for p in calls} >= {"initialize", "tools/list", "tools/call"}
    # 配置型接入不创建进程级 session_id，也不把本地会话标识发给服务。
    for call in calls:
        if call["method"] == "tools/call":
            assert "session_id" not in call["params"]["arguments"]


@pytest.mark.parametrize("patch", [
    {"mcp_builtin_search": False},
    {"enabled_toolsets": ["core"]},
    {"enabled_toolsets": []},
    {"permissions": {"deny": ["mcp__parallel-search"]}},
    {"disabled_tools": [SEARCH, FETCH]},
])
def test_disabled_never_connects(runtime, patch):
    save_settings({**load_settings(), **patch})
    assert mcp_tool.initialize_mcp() == 0
    assert runtime.requests == []
    assert "parallel-search" not in runtime.manager.servers


@pytest.mark.parametrize("location", ["settings", "legacy", "project", "inline", "plugin"])
def test_explicit_server_no_fallback(runtime, monkeypatch, location):
    cfg = {"chosen": {"url": "https://chosen.example/mcp", "transport": "http",
                      "headers": {"Authorization": "Bearer intentionally-configured"}}}
    if location == "settings":
        save_settings({**load_settings(), "mcpServers": cfg})
    elif location == "legacy":
        (runtime.home / ".mcp.json").write_text(json.dumps({"mcpServers": cfg}), encoding="utf-8")
    elif location == "project":
        (runtime.workspace / ".mcp.json").write_text(json.dumps({"mcpServers": cfg}), encoding="utf-8")
    elif location == "inline":
        monkeypatch.setattr("agent.agent_defs.project_inline_mcp_servers", lambda: cfg)
    elif location == "plugin":
        monkeypatch.setattr(mcp_tool, "collect_plugin_mcp_servers", lambda **kw: {"chosen": {"cfg": cfg["chosen"]}})
    attempts = []

    def unavailable(name, config, **kwargs):
        attempts.append((name, config))
        raise RuntimeError("configured server unavailable")

    monkeypatch.setattr(runtime.manager, "connect_one", unavailable)
    mcp_tool.initialize_mcp()
    assert not any(name == "parallel-search" for name, _ in attempts)
    if location in ("settings", "legacy", "plugin"):
        assert attempts == [("chosen", cfg["chosen"])]
    else:
        assert attempts == []  # 未批准的项目/内联配置保持关闭。
    assert runtime.requests == []


@pytest.mark.parametrize("location", ["legacy", "project"])
def test_explicit_empty_config_no_fallback(runtime, location):
    directory = runtime.home if location == "legacy" else runtime.workspace
    (directory / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


@pytest.mark.parametrize("content", ["{broken", "[]"])
def test_invalid_settings_no_default_connection(runtime, content):
    (runtime.home / "settings.json").write_text(content, encoding="utf-8")
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


@pytest.mark.parametrize("patch", [
    {"mcpServers": value} for value in (None, False, [], 0, "")
] + [
    {"enabled_toolsets": "mcp"},
    {"disabled_tools": None},
    {"permissions": None},
    {"permissions": {"deny": "mcp__parallel-search"}},
    {"permissions": {"deny": [None]}},
    {"mcp_builtin_search": "true"},
])
def test_malformed_setting_types_do_not_authorize_default(runtime, patch):
    (runtime.home / "settings.json").write_text(json.dumps({
        "enabled_toolsets": ["core", "mcp"], **patch,
    }), encoding="utf-8")
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


@pytest.mark.parametrize("content", ['{"mcpServers":{"chosen":', '{"mcpServers": []}',
                                     '{"mcpServers": {"chosen": null}}', '{"mcpServers": {}}'])
def test_invalid_plugin_config_no_default_connection(runtime, content):
    plugin = runtime.home / "plugins" / "chosen"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text('{"enabled": true}', encoding="utf-8")
    (plugin / ".mcp.json").write_text(content, encoding="utf-8")
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


def test_bad_plugin_does_not_discard_other_servers(runtime, monkeypatch):
    root = runtime.home / "plugins"
    for name, content in [("bad", '{"mcpServers":'),
                          ("good", '{"mcpServers":{"chosen":{"url":"https://chosen.example/mcp"}}}')]:
        plugin = root / name
        plugin.mkdir(parents=True)
        (plugin / "plugin.json").write_text('{"enabled":true}', encoding="utf-8")
        (plugin / ".mcp.json").write_text(content, encoding="utf-8")
    attempts = []

    def unavailable(name, config, **kwargs):
        attempts.append((name, config))
        raise RuntimeError("configured server unavailable")

    monkeypatch.setattr(runtime.manager, "connect_one", unavailable)
    mcp_tool.initialize_mcp()
    assert attempts == [("chosen", {"url": "https://chosen.example/mcp"})]
    assert runtime.requests == []


@pytest.mark.parametrize("location", ["inline", "plugin"])
def test_discovery_failure_no_default_connection(runtime, monkeypatch, location):
    def unavailable(**kwargs):
        raise OSError("fixture configuration discovery failed")

    if location == "inline":
        monkeypatch.setattr("agent.agent_defs.project_inline_mcp_servers", unavailable)
    else:
        monkeypatch.setattr(mcp_tool, "collect_plugin_mcp_servers", unavailable)
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


@pytest.mark.parametrize("location", ["user", "project", "plugin"])
def test_custom_agent_files_suppress_default_even_if_unparseable(runtime, location):
    if location == "user":
        directory = runtime.home / "agents"
    elif location == "project":
        directory = runtime.workspace / ".codeAgent" / "agents"
    else:
        directory = runtime.home / "plugins" / "chosen" / "agents"
        directory.parent.mkdir(parents=True)
        (directory.parent / "plugin.json").write_text('{"enabled":true}', encoding="utf-8")
    directory.mkdir(parents=True)
    (directory / "chosen.md").write_text('---\nname: chosen\ninlineMcpServers: [broken\n---\n', encoding="utf-8")
    mcp_tool.initialize_mcp()
    assert runtime.requests == []


@pytest.mark.parametrize("failure, expected", [("timeout", "fixture timeout"), ("http", "503"),
                                                ("rpc", "invalid arguments"), ("tool", "service rejected query")])
def test_dispatch_errors_are_visible(runtime, failure, expected):
    mcp_tool.initialize_mcp()
    runtime.failure = failure
    result = asyncio.run(handle_function_call(SEARCH, {"objective": "docs", "search_queries": ["Python docs"]}))
    assert expected in result
    assert len([r for r in runtime.requests if r.method == "POST"
                and json.loads(r.content)["method"] == "tools/call"]) == 1


def test_permission_and_disabled_tools_enforced_at_dispatch(runtime):
    mcp_tool.initialize_mcp()
    before = len(runtime.requests)
    settings = load_settings()
    settings["permissions"]["deny"] = [SEARCH]
    save_settings(settings)
    tool_permissions.reset_rules_cache()
    assert SEARCH not in [d["function"]["name"] for d in get_tool_definitions(["core", "mcp"])]
    denied = asyncio.run(handle_function_call(SEARCH, {}))
    disabled = asyncio.run(handle_function_call(FETCH, {}, config={"disabled_tools": [FETCH]}))
    assert json.loads(denied)["error_type"] == "permission_denied"
    assert json.loads(disabled)["error_type"] == "permission_denied"
    assert len(runtime.requests) == before


def test_cancellation_keeps_native_dispatch_semantics(runtime):
    """取消等待立即生效；原有同步 HTTP 线程结束前不会偷偷重试请求。"""
    mcp_tool.initialize_mcp()
    runtime.failure = "blocking"

    async def cancel():
        task = asyncio.create_task(handle_function_call(SEARCH, {"objective": "docs", "search_queries": ["Python docs"]}))
        assert await asyncio.to_thread(runtime.entered.wait, 2)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            runtime.release.set()

    asyncio.run(cancel())
    assert len([r for r in runtime.requests if r.method == "POST"
                and json.loads(r.content)["method"] == "tools/call"]) == 1
