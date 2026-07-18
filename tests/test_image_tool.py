"""image_analyze + image_ocr 测试。"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 最小 PNG（8x8 透明）—— 用于写真实文件
# ---------------------------------------------------------------------------

# PNG magic + IHDR + IDAT + IEND（~67 bytes）
MINIMAL_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000080000000808060000007ed7dca"
    "00000001649444154789c636000000000020001e221bc330000000049454e44ae"
    "426082"
)


@pytest.fixture
def png_file(tmp_path):
    """写一个最小 PNG 文件用于测试。"""
    p = tmp_path / "test.png"
    p.write_bytes(MINIMAL_PNG_BYTES)
    return p


class FakeAgent:
    """最小 AIAgent mock。"""
    def __init__(self, vision_client=None, llm_client=None):
        self._vision_client = vision_client
        self.llm_client = llm_client
        self.config = {"model": {"name": "test-model"}}


def make_fake_vision_client(response_text: str = "fake description"):
    """构造 mock vision client（OpenAI 兼容响应）。"""
    client = MagicMock()
    msg = MagicMock()
    msg.message.content = response_text
    resp = MagicMock()
    resp.choices = [msg]
    client.chat.completions.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_image_analyze_happy_path(png_file):
    """mock client 返回 description，验证 base64 + messages 结构。"""
    from tools.image_tool import _handle_image_analyze

    fake_client = make_fake_vision_client("a white circle")
    agent = FakeAgent(vision_client=fake_client)

    result = _handle_image_analyze(
        {"image_path": str(png_file), "query": "what is this?"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["description"] == "a white circle"
    assert data["query"] == "what is this?"

    # 验证 client 被调用，messages 含 image_url
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    msg = call_kwargs["messages"][0]
    assert msg["role"] == "user"
    content = msg["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "what is this?"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_ocr_uses_ocr_prompt(png_file):
    """OCR 工具自动拼 '提取图中所有可见文字' 提示。"""
    from tools.image_tool import _handle_image_ocr

    fake_client = make_fake_vision_client("extracted text")
    agent = FakeAgent(vision_client=fake_client)

    result = _handle_image_ocr(
        {"image_path": str(png_file)},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["text"] == "extracted text"

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    actual_query = call_kwargs["messages"][0]["content"][0]["text"]
    assert "提取图中所有可见文字" in actual_query


def test_image_ocr_with_hint(png_file):
    """带 hint 时拼到 prompt。"""
    from tools.image_tool import _handle_image_ocr

    fake_client = make_fake_vision_client("code here")
    agent = FakeAgent(vision_client=fake_client)

    result = _handle_image_ocr(
        {"image_path": str(png_file), "hint": "Python 代码"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["success"] is True

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    actual_query = call_kwargs["messages"][0]["content"][0]["text"]
    assert "Python 代码" in actual_query


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------

def test_unsupported_format_bmp(tmp_path):
    """.bmp 后缀拒绝。"""
    from tools.image_tool import _handle_image_analyze

    # 写个假 bmp（内容不重要，因为后缀检查在文件读取之前）
    bmp = tmp_path / "fake.bmp"
    bmp.write_bytes(b"BM\x00\x00")

    agent = FakeAgent(vision_client=make_fake_vision_client())
    result = _handle_image_analyze(
        {"image_path": str(bmp), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "unsupported_format"
    assert "bmp" in data["error"].lower() or "bmp" in str(data)


def test_unsupported_format_tiff(tmp_path):
    from tools.image_tool import _handle_image_analyze
    tiff = tmp_path / "x.tiff"
    tiff.write_bytes(b"II*\x00")
    agent = FakeAgent(vision_client=make_fake_vision_client())
    result = _handle_image_analyze(
        {"image_path": str(tiff), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "unsupported_format"


def test_file_not_found(tmp_path):
    from tools.image_tool import _handle_image_analyze
    agent = FakeAgent(vision_client=make_fake_vision_client())
    result = _handle_image_analyze(
        {"image_path": str(tmp_path / "missing.png"), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "file_not_found"


def test_protected_path_rejected(tmp_path, monkeypatch):
    """~/.ssh/foo.png 被safe_path 拒绝。"""
    from tools.image_tool import _handle_image_analyze
    from agent.permission import PermissionResult

    # mock safe_path 返回拒绝
    def fake_safe_path(path, *, write=False, allowed_roots=None):
        return PermissionResult(False, "受保护路径: ~/.ssh", "protected")

    monkeypatch.setattr("tools.image_tool.safe_path", fake_safe_path)

    # 文件不存在没关系，safe_path 在文件存在检查之前
    agent = FakeAgent(vision_client=make_fake_vision_client())
    result = _handle_image_analyze(
        {"image_path": "/fake/path/.ssh/foo.png", "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "permission_denied"


def test_file_too_large(tmp_path, monkeypatch):
    """文件 > 20MB 拒绝。"""
    from tools.image_tool import _handle_image_analyze

    big = tmp_path / "big.png"
    big.write_bytes(MINIMAL_PNG_BYTES)

    # mock stat 返回 25MB
    class FakeStat:
        st_size = 25 * 1024 * 1024
    monkeypatch.setattr(Path, "stat", lambda self: FakeStat())

    agent = FakeAgent(vision_client=make_fake_vision_client())
    result = _handle_image_analyze(
        {"image_path": str(big), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "file_too_large"


def test_vision_client_fallback_to_main_llm(png_file):
    """_vision_client=None 时回退到 llm_client。"""
    from tools.image_tool import _handle_image_analyze

    fake_main = make_fake_vision_client("from main client")
    agent = FakeAgent(vision_client=None, llm_client=fake_main)

    result = _handle_image_analyze(
        {"image_path": str(png_file), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["success"] is True
    assert data["description"] == "from main client"
    fake_main.chat.completions.create.assert_called_once()


def test_vision_client_unavailable_all_none(png_file):
    """vision_client 和 llm_client 都 None → vision_unavailable。"""
    from tools.image_tool import _handle_image_analyze

    agent = FakeAgent(vision_client=None, llm_client=None)
    result = _handle_image_analyze(
        {"image_path": str(png_file), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "vision_unavailable"


def test_vision_api_failure_returns_vision_error(png_file):
    """LLM 调用抛异常 → vision_error。"""
    from tools.image_tool import _handle_image_analyze

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("API timeout")
    agent = FakeAgent(vision_client=fake_client)

    result = _handle_image_analyze(
        {"image_path": str(png_file), "query": "x"},
        agent_ref=agent,
    )
    data = json.loads(result)
    assert data["error_type"] == "vision_error"
    assert "API timeout" in data["error"]


def test_mime_inference_all_formats():
    """所有支持的后缀都映射到正确 MIME。"""
    from tools.image_tool import _infer_mime, SUPPORTED_FORMATS

    assert _infer_mime("foo.png") == "image/png"
    assert _infer_mime("foo.PNG") == "image/png"  # 大小写不敏感
    assert _infer_mime("foo.jpeg") == "image/jpeg"
    assert _infer_mime("foo.jpg") == "image/jpeg"
    assert _infer_mime("foo.webp") == "image/webp"
    assert _infer_mime("foo.gif") == "image/gif"
    # 不支持的
    assert _infer_mime("foo.bmp") is None
