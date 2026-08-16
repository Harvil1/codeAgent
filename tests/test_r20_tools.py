"""R20 工具面专项测试。

#33 空结果保护
#34 Read 双上限错误化
#31 WebFetch aux 提炼
#35 NotebookEdit
"""

import json
from types import SimpleNamespace

import pytest

from model_tools import ensure_tools_discovered
from tools.registry import registry

ensure_tools_discovered()


# ---------------------------------------------------------------------------
# R20 #33：空结果保护
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_result_protection():
    """handler 返回空 → 注入显式 no output 标记（防误判回合边界）。"""
    from model_tools import handle_function_call

    async def empty_handler(args, **kw):
        return ""  # 空串

    async def blank_handler(args, **kw):
        return "   "  # 纯空白

    async def empty_dict_handler(args, **kw):
        return "{}"  # 空 dict

    async def normal_handler(args, **kw):
        return json.dumps({"success": True})

    # 注册临时工具（测试后注销）
    for hname, h in [
        ("_t_empty_20", empty_handler),
        ("_t_blank_20", blank_handler),
        ("_t_emptydict_20", empty_dict_handler),
        ("_t_normal_20", normal_handler),
    ]:
        registry.register(
            name=hname, toolset="core",
            schema={"name": hname, "parameters": {"type": "object", "properties": {}}},
            handler=h, emoji="t", isConcurrencySafe=True,
        )
    try:
        r1 = json.loads(await handle_function_call("_t_empty_20", {}))
        assert r1["empty_output"] is True
        assert "completed with no output" in r1["content"]

        r2 = json.loads(await handle_function_call("_t_blank_20", {}))
        assert r2["empty_output"] is True

        r3 = json.loads(await handle_function_call("_t_emptydict_20", {}))
        assert r3["empty_output"] is True

        r4 = json.loads(await handle_function_call("_t_normal_20", {}))
        assert r4 == {"success": True}  # 非空不动
    finally:
        for name in ("_t_empty_20", "_t_blank_20", "_t_emptydict_20", "_t_normal_20"):
            try:
                registry.unregister(name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# R20 #34：Read 双上限错误化
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_file_size_precheck(tmp_path):
    """文件 > 256KB → file_too_large 错误（不读盘）。"""
    big = tmp_path / "big.log"
    big.write_text("x" * (256 * 1024 + 100), encoding="utf-8")
    r = json.loads(await registry.dispatch("read_file", {"path": str(big)}))
    assert "error" in r
    assert r["error_type"] == "file_too_large"
    assert "offset/limit" in r["error"]


@pytest.mark.asyncio
async def test_read_output_token_check(tmp_path):
    """分段后输出 > 25K tokens（约 75K 字符）→ output_too_large 错误。"""
    f = tmp_path / "wide.txt"
    # 每行 100 字符 × 1000 行 = 100K 字符（< 256KB 但 > 75K）
    f.write_text("\n".join("y" * 99 for _ in range(1000)), encoding="utf-8")
    r = json.loads(await registry.dispatch("read_file", {"path": str(f)}))
    assert "error" in r
    assert r["error_type"] == "output_too_large"
    assert r["total_lines"] == 1000

    # limit 分段后正常返回
    r2 = json.loads(await registry.dispatch(
        "read_file", {"path": str(f), "limit": 100},
    ))
    assert "content" in r2 and r2["shown_lines"] == "1-100"


@pytest.mark.asyncio
async def test_read_normal_file_still_works(tmp_path):
    """正常小文件读取不受影响（回归保护）。"""
    f = tmp_path / "ok.py"
    f.write_text("print('hi')\n", encoding="utf-8")
    r = json.loads(await registry.dispatch("read_file", {"path": str(f)}))
    assert "print" in r["content"]


# ---------------------------------------------------------------------------
# R20 #31：WebFetch aux 提炼
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_fetch_refines_with_aux(monkeypatch):
    """有 prompt + aux 可用 → 提炼结果（refined=true）。"""
    import tools.web_fetch_tool as wf

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "text/plain"}
        content = b"v2.3.1 released on 2026-08-01 with bug fixes."
        def raise_for_status(self):
            pass

    def fake_get(url, **kw):
        return _FakeResp()

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    class _Aux:
        async def chat_completions(self, msgs, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="版本 2.3.1，2026-08-01 发布，修复若干 bug。",
            ))])

    agent_ref = SimpleNamespace(aux_llm_router=_Aux())
    r = json.loads(await wf._handle_web_fetch(
        {"url": "https://example.com/changelog", "prompt": "最新版本"},
        agent_ref=agent_ref,
    ))
    assert r["refined"] is True
    assert "2.3.1" in r["content"]


@pytest.mark.asyncio
async def test_web_fetch_aux_failure_falls_back(monkeypatch):
    """aux 失败 → 降级全文（refined=false，fail-open）。"""
    import tools.web_fetch_tool as wf
    import httpx

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "text/plain"}
        content = b"full content here"
        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResp())

    class _BrokenAux:
        async def chat_completions(self, msgs, **kw):
            raise RuntimeError("aux down")

    agent_ref = SimpleNamespace(aux_llm_router=_BrokenAux())
    r = json.loads(await wf._handle_web_fetch(
        {"url": "https://example.com/x", "prompt": "要点"},
        agent_ref=agent_ref,
    ))
    assert r["refined"] is False
    assert r["content"] == "full content here"


@pytest.mark.asyncio
async def test_web_fetch_no_prompt_no_aux(monkeypatch):
    """无 prompt → 不走提炼（全文），无 agent_ref 也不炸。"""
    import tools.web_fetch_tool as wf
    import httpx

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "text/plain"}
        content = b"raw body"
        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResp())
    r = json.loads(await wf._handle_web_fetch({"url": "https://example.com/y"}))
    assert r["refined"] is False
    assert r["content"] == "raw body"
