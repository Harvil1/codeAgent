"""WebFetch 工具测试（对齐 Claude Code WebFetch）。"""

import json

import pytest

from tools.web_fetch_tool import _handle_web_fetch, WEB_FETCH_SCHEMA


@pytest.mark.asyncio
async def test_schema_url_required():
    """url 是必填。"""
    assert "url" in WEB_FETCH_SCHEMA["parameters"]["required"]
    assert "url" in WEB_FETCH_SCHEMA["parameters"]["properties"]


@pytest.mark.asyncio
async def test_url_missing():
    """url 为空 → error。"""
    res = json.loads(await _handle_web_fetch({}))
    assert res["error_type"] == "invalid_args"


@pytest.mark.asyncio
async def test_invalid_url_scheme():
    """非 http/https URL 拒绝。"""
    res = json.loads(await _handle_web_fetch({"url": "file:///etc/passwd"}))
    assert res["error_type"] == "invalid_url"


@pytest.mark.asyncio
async def test_fetch_html_to_text(monkeypatch):
    """HTML 被抓取并转纯文本。"""
    class FakeResp:
        headers = {"content-type": "text/html; charset=utf-8"}
        content = (
            "<html><head><style>.x{}</style></head><body>"
            "<h1>标题</h1><p>正文内容 ABC</p>"
            "<script>var x=1;</script></body></html>"
        ).encode("utf-8")
        def raise_for_status(self):
            pass

    monkeypatch.setattr("httpx.get", lambda url, **kw: FakeResp())
    res = json.loads(await _handle_web_fetch({"url": "https://example.com"}))
    assert res["url"] == "https://example.com"
    assert "正文内容 ABC" in res["content"]
    assert "标题" in res["content"]
    assert res["truncated"] is False


@pytest.mark.asyncio
async def test_fetch_timeout(monkeypatch):
    """超时 → error timeout。"""
    def _raise(url, **kw):
        import httpx
        raise httpx.TimeoutException("timeout")
    monkeypatch.setattr("httpx.get", _raise)
    res = json.loads(await _handle_web_fetch({"url": "https://example.com"}))
    assert res["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_fetch_http_error(monkeypatch):
    """HTTP 错误 → error http_error。"""
    def _raise(url, **kw):
        import httpx
        raise httpx.HTTPStatusError("404", request=None, response=None)
    monkeypatch.setattr("httpx.get", _raise)
    res = json.loads(await _handle_web_fetch({"url": "https://example.com/404"}))
    assert res["error_type"] == "http_error"


@pytest.mark.asyncio
async def test_binary_content_rejected(monkeypatch):
    """二进制内容拒绝。"""
    class FakeResp:
        headers = {"content-type": "application/octet-stream"}
        content = b"\x00\x01\x02"
        def raise_for_status(self):
            pass
    monkeypatch.setattr("httpx.get", lambda url, **kw: FakeResp())
    res = json.loads(await _handle_web_fetch({"url": "https://example.com/a.bin"}))
    assert res["error_type"] == "binary_content"


@pytest.mark.asyncio
async def test_prompt_passthrough(monkeypatch):
    """prompt 参数透传到返回。"""
    class FakeResp:
        headers = {"content-type": "text/plain"}
        content = b"hello"
        def raise_for_status(self):
            pass
    monkeypatch.setattr("httpx.get", lambda url, **kw: FakeResp())
    res = json.loads(await _handle_web_fetch({"url": "https://example.com", "prompt": "总结"}))
    assert res["prompt"] == "总结"
    assert res["content"] == "hello"
