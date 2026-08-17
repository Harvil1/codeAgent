import json
import os
from unittest.mock import patch, MagicMock


def test_web_search_no_api_key_hidden():
    """无 TAVILY_API_KEY 时 check_fn 返回 False（工具隐藏）。"""
    from tools.web_search_tool import _check_tavily_configured
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TAVILY_API_KEY", None)
        assert _check_tavily_configured() is False


def test_web_search_with_api_key_visible():
    """有 TAVILY_API_KEY 时 check_fn 返回 True。"""
    from tools.web_search_tool import _check_tavily_configured
    with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
        assert _check_tavily_configured() is True


def test_web_search_returns_results():
    """调用 handler 返回结构化结果。"""
    from tools.web_search_tool import _handle_web_search
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"title": "Example", "url": "https://example.com", "content": "abc"},
            {"title": "Two", "url": "https://two.com", "content": "def"},
        ]
    }
    with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
        with patch("tools.web_search_tool.requests.post", return_value=mock_resp):
            out = json.loads(_handle_web_search({"query": "test"}))
    assert out["query"] == "test"
    assert len(out["results"]) == 2
    assert out["results"][0]["url"] == "https://example.com"


def test_web_search_api_error_returns_error_json():
    """API 报错时返回 error JSON（不抛）。"""
    from tools.web_search_tool import _handle_web_search
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "unauthorized"
    with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-bad"}):
        with patch("tools.web_search_tool.requests.post", return_value=mock_resp):
            out = json.loads(_handle_web_search({"query": "x"}))
    assert "error" in out
    assert out.get("error_type") == "tavily_api_error"
