"""工具面专项测试。

空结果保护
Read 双上限错误化
WebFetch aux 提炼
NotebookEdit
"""

import json
from types import SimpleNamespace

import pytest

from model_tools import ensure_tools_discovered
from tools.registry import registry

ensure_tools_discovered()


# ---------------------------------------------------------------------------
# 空结果保护
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
# Read 双上限错误化
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
# WebFetch aux 提炼
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


# ---------------------------------------------------------------------------
# NotebookEdit
# ---------------------------------------------------------------------------

def _mk_nb(tmp_path):
    nb = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": "print(1)",
             "outputs": [], "execution_count": None, "id": "c1"},
            {"cell_type": "markdown", "metadata": {}, "source": "# 标题", "id": "c2"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


class _AllowChecker:
    """测试用放行 checker（tmp_path 不在生产白名单）。"""
    def check_path(self, path, write=False, mode_override=None, **kw):
        return SimpleNamespace(allowed=True, reason="ok", gate="ok")


@pytest.mark.asyncio
async def test_notebook_replace(tmp_path):
    p = _mk_nb(tmp_path)
    r = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "cell_id": "c1",
        "new_source": "print(42)", "edit_mode": "replace",
    }))
    assert "action" in r and r["total_cells"] == 2
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["cells"][0]["source"] == "print(42)"


@pytest.mark.asyncio
async def test_notebook_replace_by_index(tmp_path):
    """纯数字 cell_id 按索引匹配（旧文件无 id 的兼容路径）。"""
    p = _mk_nb(tmp_path)
    r = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "cell_id": "1",
        "new_source": "# 新标题", "edit_mode": "replace",
    }))
    assert "action" in r
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["cells"][1]["source"] == "# 新标题"


@pytest.mark.asyncio
async def test_notebook_insert_and_delete(tmp_path):
    p = _mk_nb(tmp_path)
    # 末尾插入 markdown
    r = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "edit_mode": "insert",
        "cell_type": "markdown", "new_source": "## 尾注",
    }))
    assert r["total_cells"] == 3
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["cells"][2]["cell_type"] == "markdown"

    # 在 c1 前插入 code
    r2 = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "edit_mode": "insert", "cell_id": "c1",
        "cell_type": "code", "new_source": "import os",
    }))
    assert r2["total_cells"] == 4
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["cells"][0]["source"] == "import os"

    # 删除
    r3 = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "cell_id": "c1", "edit_mode": "delete",
    }))
    assert r3["total_cells"] == 3
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert all(c.get("id") != "c1" for c in nb["cells"])


@pytest.mark.asyncio
async def test_notebook_errors(tmp_path):
    """缺 cell_id / 找不到 / 非 ipynb / 索引越界。"""
    p = _mk_nb(tmp_path)
    r1 = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "edit_mode": "delete",
    }))
    assert "error" in r1  # 缺 cell_id

    r2 = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "cell_id": "nope", "edit_mode": "delete",
    }))
    assert r2.get("error_type") == "cell_not_found"

    txt = tmp_path / "plain.txt"
    txt.write_text("hi", encoding="utf-8")
    r3 = json.loads(await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(txt), "cell_id": "0", "edit_mode": "delete",
    }))
    assert r3.get("error_type") == "invalid_args"


@pytest.mark.asyncio
async def test_notebook_backfills_ids(tmp_path):
    """无 id 的旧 notebook 补齐 cell_<n> id（nbformat 4.5+ 标准）。"""
    nb = {"cells": [
        {"cell_type": "code", "metadata": {}, "source": "x", "outputs": [], "execution_count": None},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
    p = tmp_path / "old.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    await registry.dispatch("notebook_edit", permission_checker=_AllowChecker(), args={
        "notebook_path": str(p), "cell_id": "0",
        "new_source": "y", "edit_mode": "replace",
    })
    nb2 = json.loads(p.read_text(encoding="utf-8"))
    assert nb2["cells"][0].get("id") == "cell_0"
    assert nb2["cells"][0]["source"] == "y"
