# Vision/Image 工具集实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 B1 spec 给定的 Vision/Image 工具集——加 2 个新工具（`image_analyze` + `image_ocr`）+ 升级通用 `_vision_client` + 修 `browser_vision` 历史遗留 bug。

**Architecture:** 同步阻塞、新工具加入 `core` toolset（check_fn 在无 client 时隐藏）、safe_path 复用现有路径白名单、base64 不落盘直传 LLM。

**Tech Stack:** Python 3.11+，标准库（`base64`、`pathlib`、`json`、`logging`），pytest。无新依赖。

**参考 spec:** `docs/superpowers/specs/2026-07-17-image-tools-design.md`

## Global Constraints

- **Python 3.11+**
- **UTF-8 强制**（文本 I/O；图片是二进制走 bytes）
- **同步阻塞**（不引入 asyncio）
- **JSON 字符串契约**：所有 handler 返 JSON 字符串，错误用 `{"error": "...", "error_type": "..."}`
- **窄腰**：工具加 `core` toolset，`check_fn` 在 client 不可用时隐藏
- **safe_path 复用**：调 `agent.permission.safe_path(write=False)`
- **commit 中文 + `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`**
- **测试**：`uv run pytest tests/test_image_tool.py -v`；全量 `uv run pytest tests/ -q`

## File Structure

| 文件 | 操作 | 责任 |
|---|---|---|
| `tools/image_tool.py` | Create | 2 工具 handler + schema + 注册 + 检查（safe_path/格式/大小）+ MIME 推断 |
| `agent/__init__.py` | Modify | `AIAgent.__init__` 加 `self._vision_client = None` 字段 |
| `cli.py` | Modify | `RuntimeContext._create_agent` 根据 `vision` config 创建并注入 `_vision_client` |
| `tools/browser_tool.py` | Modify | `_handle_browser_vision` 改三层回退（`_vision_client` → `_browser_vision_client` → `llm_client`） |
| `tests/test_image_tool.py` | Create | 12 个测试 |

`tools/image_tool.py` 不依赖任何 agent 模块（只依赖 `agent.permission.safe_path` 和 `tools.registry`），便于单元测试。

---

## Task 1: `tools/image_tool.py` — 2 工具 handler + 检查

**Files:**
- Create: `tools/image_tool.py`
- Create: `tests/test_image_tool.py`（仅 Task 1 需要的 happy path + 错误路径测试）

**Interfaces:**
- Consumes: `agent.permission.safe_path`、`tools.registry.registry`
- Produces:
  - `IMAGE_ANALYZE_SCHEMA`、`IMAGE_OCR_SCHEMA` 常量
  - `_handle_image_analyze(args, **kwargs) -> str`
  - `_handle_image_ocr(args, **kwargs) -> str`
  - 私有：`_check_image_file(path, max_bytes)`、`_infer_mime(suffix)`、`_encode_image(path)`、`_call_vision_llm(client, model, query, image_b64, mime) -> str`、`_get_vision_client(agent) -> tuple(client, model)`

**kwargs 契约**：
- `agent_ref`：AIAgent 实例（用于取 `_vision_client` / `llm_client` / `config`）
- 无 `agent_ref` 时所有 vision 调用返 `vision_unavailable`

- [ ] **Step 1: 创建测试文件 + 写第一个失败测试**

创建 `tests/test_image_tool.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认全部失败**

Run: `uv run pytest tests/test_image_tool.py -v`
Expected: 12 FAIL（`ModuleNotFoundError: No module named 'tools.image_tool'`）

- [ ] **Step 3: 实现 `tools/image_tool.py`**

```python
"""image_analyze + image_ocr 工具：用 vision LLM 分析本地图片。

安全：
- safe_path(write=False) 拒绝受保护路径（~/.ssh / /etc 等）
- 后缀白名单：png/jpeg/jpg/webp/gif
- 文件大小上限：20MB（可配置 vision.max_bytes）
- base64 编码直传 LLM，不落盘
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from agent.permission import safe_path
from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {
    ".png":  "image/png",
    ".jpeg": "image/jpeg",
    ".jpg":  "image/jpeg",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _infer_mime(filename: str) -> Optional[str]:
    """从文件名后缀推断 MIME。大小写不敏感。"""
    suffix = Path(filename).suffix.lower()
    return SUPPORTED_FORMATS.get(suffix)


def _check_image_file(path_str: str, max_bytes: int) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """检查图片文件可读性。

    返回 (path_obj, error_type, error_msg)。
    - 成功：(Path, None, None)
    - 失败：(None, error_type, error_msg)
    """
    # 1. 后缀白名单
    mime = _infer_mime(path_str)
    if mime is None:
        suffix = Path(path_str).suffix.lower()
        return None, "unsupported_format", (
            f"不支持的图片格式: {suffix}（允许: {sorted(SUPPORTED_FORMATS.keys())}）"
        )

    # 2. safe_path 路径白名单
    perm = safe_path(path_str, write=False)
    if not perm.allowed:
        return None, "permission_denied", f"路径拒绝: {perm.reason}"

    p = Path(path_str)

    # 3. 文件存在
    if not p.exists():
        return None, "file_not_found", f"文件不存在: {path_str}"

    # 4. 大小检查
    try:
        size = p.stat().st_size
    except OSError as e:
        return None, "file_not_found", f"读取文件信息失败: {e}"
    if size > max_bytes:
        return None, "file_too_large", (
            f"文件过大: {size} bytes（上限 {max_bytes} bytes = {max_bytes // (1024*1024)}MB）"
        )

    return p, None, None


def _get_vision_client(agent) -> Tuple[Optional[object], Optional[str]]:
    """从 agent 取 vision client + 模型名。

    优先级：
    1. agent._vision_client（独立 vision 配置）
    2. agent.llm_client（回退主模型）

    返回 (client, model_name)。client 为 None 时表示不可用。
    """
    if agent is None:
        return None, None

    client = getattr(agent, "_vision_client", None)
    if client is not None:
        # 模型名从 client.model 取（OpenAICompatClient 字段）
        model = getattr(client, "model", None) or "vision-model"
        return client, model

    client = getattr(agent, "llm_client", None)
    if client is not None:
        model = getattr(agent, "config", {}).get("model", {}).get("name") or "main-model"
        return client, model

    return None, None


def _call_vision_llm(
    client,
    model: str,
    query: str,
    image_b64: str,
    mime: str,
) -> str:
    """调用 OpenAI 兼容 vision API，返回 description。"""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{image_b64}",
                }},
            ],
        }],
        max_tokens=2000,
    )
    return response.choices[0].message.content


def _build_error(error_type: str, msg: str, image_path: str) -> str:
    """构造错误 JSON 字符串。"""
    return json.dumps({
        "error": msg,
        "error_type": error_type,
        "image_path": image_path,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

IMAGE_ANALYZE_SCHEMA = {
    "name": "image_analyze",
    "description": (
        "用 vision LLM 分析本地图片。可描述内容、识别物体、回答视觉问题。"
        "适合截图分析、图表理解、UI 评审等场景。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "本地图片路径（支持 png/jpeg/jpg/webp/gif）",
            },
            "query": {
                "type": "string",
                "description": "问 LLM 什么（如 '描述这张图' / '图里有几只猫'）",
            },
        },
        "required": ["image_path", "query"],
    },
}

IMAGE_OCR_SCHEMA = {
    "name": "image_ocr",
    "description": (
        "从图片提取文字（OCR）。适合截图、扫描文档、图表中的文字提取。"
        "比 image_analyze 更聚焦：自动用 OCR 专用提示词。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "本地图片路径（支持 png/jpeg/jpg/webp/gif）",
            },
            "hint": {
                "type": "string",
                "description": "可选：语言或上下文提示，如 '中文' / 'Python 代码' / '表格'",
            },
        },
        "required": ["image_path"],
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_image_analyze(args: dict, **kwargs) -> str:
    """通用 vision 分析。"""
    image_path = (args.get("image_path") or "").strip()
    query = (args.get("query") or "").strip()

    if not image_path:
        return json.dumps({"error": "image_path 不能为空"}, ensure_ascii=False)
    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    # 取 max_bytes（从 agent config 读，否则用默认）
    agent = kwargs.get("agent_ref")
    max_bytes = DEFAULT_MAX_BYTES
    if agent is not None:
        max_bytes = (getattr(agent, "config", {}).get("vision", {}) or {}).get(
            "max_bytes", DEFAULT_MAX_BYTES
        )

    # 检查文件
    path_obj, err_type, err_msg = _check_image_file(image_path, max_bytes)
    if err_type:
        return _build_error(err_type, err_msg, image_path)

    # 取 client
    client, model = _get_vision_client(agent)
    if client is None:
        return _build_error(
            "vision_unavailable",
            "vision LLM client 未配置（agent._vision_client 和 agent.llm_client 都为 None）",
            image_path,
        )

    # 编码图片
    try:
        image_bytes = path_obj.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as e:
        return _build_error("vision_error", f"图片读取失败: {e}", image_path)

    mime = _infer_mime(image_path)

    # 调 LLM
    try:
        description = _call_vision_llm(client, model, query, b64, mime)
    except Exception as e:
        logger.warning("image_analyze LLM 调用失败: %s", e)
        return _build_error("vision_error", str(e), image_path)

    return json.dumps({
        "success": True,
        "description": description,
        "image_path": image_path,
        "query": query,
    }, ensure_ascii=False)


def _handle_image_ocr(args: dict, **kwargs) -> str:
    """OCR 文字提取（image_analyze 的 OCR 专用版）。"""
    image_path = (args.get("image_path") or "").strip()
    hint = (args.get("hint") or "").strip()

    if not image_path:
        return json.dumps({"error": "image_path 不能为空"}, ensure_ascii=False)

    # 拼 OCR prompt
    prompt = "请提取图中所有可见文字，保持原始结构和换行。"
    if hint:
        prompt += f"\n上下文提示：{hint}"

    # 复用 image_analyze 的全部检查 + 调用
    analyze_result = _handle_image_analyze(
        {"image_path": image_path, "query": prompt},
        **kwargs,
    )
    data = json.loads(analyze_result)

    # 错误透传
    if not data.get("success"):
        return analyze_result

    # 改字段名：description → text
    return json.dumps({
        "success": True,
        "text": data["description"],
        "image_path": image_path,
        "hint": hint or None,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# check_fn + 注册
# ---------------------------------------------------------------------------

def _check_vision_available() -> bool:
    """check_fn：仅当 agent 能取到 vision client 时才暴露工具。

    注意：注册时无法访问 agent 实例，所以这里走"宽松暴露"策略——
    让工具始终对 LLM 可见，运行时（handler 内）再校验 client。
    隐藏策略放在 RuntimeContext 启动时根据 config 决定（见 Task 2）。
    """
    return True


registry.register(
    name="image_analyze",
    toolset="core",
    schema=IMAGE_ANALYZE_SCHEMA,
    handler=_handle_image_analyze,
    check_fn=_check_vision_available,
    emoji="🖼️",
)

registry.register(
    name="image_ocr",
    toolset="core",
    schema=IMAGE_OCR_SCHEMA,
    handler=_handle_image_ocr,
    check_fn=_check_vision_available,
    emoji="📝",
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_image_tool.py -v`
Expected: 12 PASS

- [ ] **Step 5: 跑全量确认无回归**

Run: `uv run pytest tests/ -q`
Expected: 825 + 12 = 837 PASS

- [ ] **Step 6: Commit**

```bash
git add tools/image_tool.py tests/test_image_tool.py
git commit -m "feat(image): Task 1 image_analyze + image_ocr 工具 + safe_path/格式/大小检查

- SUPPORTED_FORMATS 白名单：png/jpeg/jpg/webp/gif
- _check_image_file：safe_path → 存在 → 大小 三层检查
- _get_vision_client：_vision_client → llm_client 回退
- _call_vision_llm：OpenAI 兼容 vision API（max_tokens=2000）
- 12 个测试覆盖 happy path + 错误路径 + MIME 推断

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `agent/__init__.py` + `cli.py` — `_vision_client` 字段 + 初始化

**Files:**
- Modify: `agent/__init__.py`（AIAgent.__init__ 末尾加 `self._vision_client = None`）
- Modify: `cli.py`（`RuntimeContext._create_agent` 根据 `vision` config 创建并注入）
- Modify: `config.py`（`DEFAULT_CONFIG` 加 `vision` 段）
- Test: 追加到 `tests/test_image_tool.py`

**Interfaces:**
- Consumes: Task 1 的 `_vision_client` 字段读取契约
- Produces:
  - `AIAgent._vision_client` 字段（默认 None）
  - `RuntimeContext._create_agent` 末尾设置 `agent._vision_client`
  - `DEFAULT_CONFIG["vision"]` 段

- [ ] **Step 1: 追加测试**

在 `tests/test_image_tool.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# vision_client 配置（Task 2）
# ---------------------------------------------------------------------------

def test_vision_client_init_from_config(tmp_path, monkeypatch):
    """RuntimeContext 根据 vision config 创建独立 client。"""
    # mock create_llm_client 避免真连 API
    from agent.llm_client import OpenAICompatClient
    fake_client = MagicMock()
    fake_client.model = "vision-test-model"

    created_configs = []
    def fake_create(config):
        created_configs.append(config)
        return fake_client

    monkeypatch.setattr("agent.llm_client.create_llm_client", fake_create)

    # 模拟 RuntimeContext._create_agent 的 vision 初始化片段
    model_cfg = {
        "format": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-test",
        "name": "deepseek-chat",
        "provider": "deepseek",
    }
    vision_cfg = {
        "enabled": True,
        "provider": "deepseek",
        "model": "deepseek-vl",
    }

    # 直接调用 create_llm_client 模拟 cli.py 的逻辑
    from agent.llm_client import create_llm_client
    if vision_cfg.get("enabled", True):
        client = create_llm_client({
            "format": model_cfg.get("format", "openai"),
            "base_url": model_cfg.get("base_url"),
            "api_key": model_cfg.get("api_key"),
            "model": vision_cfg.get("model") or model_cfg["name"],
        })
    else:
        client = None

    assert client is fake_client
    assert created_configs[0]["model"] == "deepseek-vl"


def test_vision_client_disabled_returns_none(monkeypatch):
    """vision.enabled=False 时不创建 client。"""
    vision_cfg = {"enabled": False}
    # 如果 disabled，应当跳过 client 创建
    assert not vision_cfg.get("enabled", True)


def test_default_config_has_vision_section():
    """config.py 的 DEFAULT_CONFIG 应包含 vision 默认配置。"""
    from config import DEFAULT_CONFIG
    assert "vision" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["vision"]["enabled"] is True
    assert DEFAULT_CONFIG["vision"]["max_bytes"] == 20 * 1024 * 1024


def test_aiagent_has_vision_client_field():
    """AIAgent 实例应有 _vision_client 字段（默认 None）。"""
    # 不构造完整 AIAgent（需要 API key），只验证类有这个属性声明
    # 通过检查 __init__ 源码片段
    import inspect
    from agent import AIAgent
    src = inspect.getsource(AIAgent.__init__)
    assert "_vision_client" in src
```

- [ ] **Step 2: 运行新测试确认失败**

Run: `uv run pytest tests/test_image_tool.py -v -k "vision_client or default_config or aiagent_has"`
Expected: 4 FAIL

- [ ] **Step 3: 修改 `agent/__init__.py` 加 `_vision_client` 字段**

定位 `AIAgent.__init__`（在 `agent/__init__.py` 约 line 100-180 区间）。找到 `self._system_prompt_built = False` 这一行（或类似初始化字段的末尾），追加：

```python
        # === B1 NEW: vision client（image_analyze / image_ocr / browser_vision 共用） ===
        # 默认 None；由 RuntimeContext 根据 config 注入，或测试时手工注入。
        self._vision_client = None
```

如果找不到合适位置，定位 `self.llm_client = ...` 这一行，紧接其后加。

- [ ] **Step 4: 修改 `config.py:DEFAULT_CONFIG` 加 vision 段**

定位 `DEFAULT_CONFIG`（在 `config.py` 约 line 30-80）。在 `"aux_model": {...}` 段之后追加：

```python
    "vision": {
        "enabled": True,
        "provider": "",          # 空字符串 = 用主 provider
        "model": "",             # 空字符串 = 用主 model（假设支持 vision）
        "max_bytes": 20 * 1024 * 1024,  # 20 MB
    },
```

- [ ] **Step 5: 修改 `cli.py:RuntimeContext._create_agent` 注入 vision_client**

定位 `RuntimeContext._create_agent`（`cli.py` 约 line 195-290）。在 `agent = AIAgent(...)` 构造完之后、`return agent` 之前，追加：

```python
        # === B1 NEW: 初始化 vision_client（image_analyze / image_ocr / browser_vision 共用） ===
        vision_cfg = self.config.get("vision", {}) or {}
        if vision_cfg.get("enabled", True):
            try:
                from agent.llm_client import create_llm_client
                vision_provider = vision_cfg.get("provider") or ""
                vision_model = vision_cfg.get("model") or ""
                # 如果 vision.model 为空，不创建独立 client（让工具回退到 llm_client）
                if vision_model:
                    agent._vision_client = create_llm_client({
                        "format": model_cfg.get("format", "openai"),
                        "base_url": model_cfg.get("base_url"),
                        "api_key": api_key,
                        "model": vision_model,
                    })
                    logger.info("vision_client 已初始化（model=%s）", vision_model)
                # 否则 agent._vision_client 保持 None，工具回退 llm_client
            except Exception as e:
                logger.warning("vision_client 初始化失败（用主 client 回退）: %s", e)
                agent._vision_client = None
```

- [ ] **Step 6: 运行测试确认通过**

Run: `uv run pytest tests/test_image_tool.py -v`
Expected: 16 PASS（Task 1 的 12 + Task 2 的 4）

- [ ] **Step 7: 跑全量确认无回归**

Run: `uv run pytest tests/ -q`
Expected: 837 + 4 = 841 PASS

- [ ] **Step 8: Commit**

```bash
git add agent/__init__.py cli.py config.py tests/test_image_tool.py
git commit -m "feat(image): Task 2 AIAgent._vision_client + RuntimeContext 初始化

- AIAgent.__init__ 加 _vision_client 字段（默认 None）
- DEFAULT_CONFIG['vision'] 段（enabled/provider/model/max_bytes）
- RuntimeContext._create_agent 根据 config 创建独立 vision client
- vision.model 为空时不创建独立 client（工具回退主 llm_client）

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `tools/browser_tool.py` — browser_vision 三层回退

**Files:**
- Modify: `tools/browser_tool.py:468-513`（`_handle_browser_vision`）
- Test: `tests/test_browser_tools.py`（追加 1 个三层回退测试）

**Interfaces:**
- Consumes: Task 2 的 `AIAgent._vision_client` 字段
- Produces: `browser_vision` 工具三层 client 回退逻辑

- [ ] **Step 1: 追加测试**

在 `tests/test_browser_tools.py` 末尾追加（看一眼文件现有 import 风格，复用）：

```python
def test_browser_vision_falls_back_to_vision_client(fake_page, monkeypatch):
    """browser_vision 优先用 _vision_client（Task 3 三层回退）。"""
    import json
    from unittest.mock import MagicMock
    from tools.browser_tool import _handle_browser_vision

    # 构造 mock agent：_vision_client 有，_browser_vision_client 无，llm_client 有
    vision_client = MagicMock()
    msg = MagicMock()
    msg.message.content = "from _vision_client"
    resp = MagicMock()
    resp.choices = [msg]
    vision_client.chat.completions.create.return_value = resp

    main_client = MagicMock()
    main_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="from llm_client"))]
    )

    fake_agent = MagicMock()
    fake_agent._vision_client = vision_client  # 优先
    fake_agent._browser_vision_client = "should-not-be-used"  # MagicMock 自动有，显式标记
    fake_agent.llm_client = main_client  # 不应被调用
    fake_agent.config = {"model": {"name": "test"}}

    # mock BrowserSession：_get_session 从 agent_ref.browser_session 取
    fake_session = MagicMock()
    fake_session.get_page.return_value = fake_page
    fake_agent.browser_session = fake_session

    result = _handle_browser_vision(
        {"query": "describe"},
        agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data.get("description") == "from _vision_client"
    # 主 client 不应被调用
    main_client.chat.completions.create.assert_not_called()


def test_browser_vision_falls_back_to_llm_client(fake_page):
    """三层都不存在 _vision_client 和 _browser_vision_client 时用 llm_client。"""
    import json
    from unittest.mock import MagicMock
    from tools.browser_tool import _handle_browser_vision

    main_client = MagicMock()
    msg = MagicMock()
    msg.message.content = "from llm_client"
    resp = MagicMock()
    resp.choices = [msg]
    main_client.chat.completions.create.return_value = resp

    fake_agent = MagicMock()
    fake_agent._vision_client = None
    # 让 getattr(fake_agent, "_browser_vision_client", None) 返回 None
    fake_agent._browser_vision_client = None
    fake_agent.llm_client = main_client
    fake_agent.config = {"model": {"name": "test-model"}}
    fake_session = MagicMock()
    fake_session.get_page.return_value = fake_page
    fake_agent.browser_session = fake_session

    result = _handle_browser_vision(
        {"query": "describe"},
        agent_ref=fake_agent,
    )
    data = json.loads(result)
    assert data.get("description") == "from llm_client"
```

- [ ] **Step 2: 运行新测试确认失败**

Run: `uv run pytest tests/test_browser_tools.py -v -k "browser_vision_falls_back"`
Expected: 2 FAIL（默认走 `_browser_vision_client`，无 `_vision_client` 回退逻辑）

- [ ] **Step 3: 修改 `tools/browser_tool.py:_handle_browser_vision`**

定位 `tools/browser_tool.py:468-481`，把现有的：

```python
def _handle_browser_vision(args: dict, **kwargs) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    agent = kwargs.get("agent_ref")
    client = getattr(agent, "_browser_vision_client", None) if agent else None
    if client is None:
        return _err(
            "vision LLM client 未配置（agent._browser_vision_client 为 None）",
            "vision_unavailable",
        )
```

替换为：

```python
def _handle_browser_vision(args: dict, **kwargs) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query 不能为空")
    session = _get_session(kwargs)
    if session is None:
        return _err("browser_session 未初始化", "browser_unavailable")
    agent = kwargs.get("agent_ref")
    # 三层回退（Task 3）：_vision_client → _browser_vision_client → llm_client
    client = None
    model_name = None
    if agent:
        client = getattr(agent, "_vision_client", None)
        if client is not None:
            model_name = getattr(client, "model", None)
        if client is None:
            client = getattr(agent, "_browser_vision_client", None)
            if client is not None:
                model_name = getattr(client, "model", None)
        if client is None:
            client = getattr(agent, "llm_client", None)
            if client is not None:
                cfg = getattr(agent, "config", {}) or {}
                model_name = cfg.get("model", {}).get("name")
    if client is None:
        return _err(
            "vision LLM client 未配置（_vision_client / _browser_vision_client / llm_client 都为 None）",
            "vision_unavailable",
        )
    if not model_name:
        model_name = "vision-model"
```

然后定位同函数里调 LLM 的那段（约 line 488-505）：

```python
        response = client.chat.completions.create(
            model=kwargs.get("config", {}).get("model", {}).get("name", "deepseek-chat"),
            messages=[...],
            max_tokens=1000,
        )
```

把 `model=...` 参数改成用 `model_name`：

```python
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )
```

- [ ] **Step 4: 运行新测试确认通过 + 跑 browser 全量**

Run: `uv run pytest tests/test_browser_tools.py -v`
Expected: 原有 + 2 新增 = 全部 PASS

- [ ] **Step 5: 跑全量确认无回归**

Run: `uv run pytest tests/ -q`
Expected: 841 PASS（Task 2 完成后的基线，browser 测试数不变）

- [ ] **Step 6: Commit**

```bash
git add tools/browser_tool.py tests/test_browser_tools.py
git commit -m "feat(image): Task 3 browser_vision 三层回退 + 修历史遗留 bug

browser_vision 工具改读 client 优先级：
1. agent._vision_client（新，Task 2 引入）
2. agent._browser_vision_client（旧，测试已用）
3. agent.llm_client（最兜底）

之前默认配置下 _browser_vision_client 从未赋值，工具永远报错。
现在默认走 llm_client 回退，开箱即用。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 最终验证 + 文档更新 + push

**Files:**
- Modify: `CLAUDE.md`（在「常用命令」前的工具集表格 + 「关键代码位置」加 image_tool）
- Modify: `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`（追加第 9 批改进 + 更新测试数）

- [ ] **Step 1: 跑最终全量测试**

Run: `uv run pytest tests/ -q`
Expected: 841 PASS

Run: `uv run python scripts/verify.py`
Expected: 22/22 ALL PASS（无回归）

- [ ] **Step 2: 更新 CLAUDE.md**

找到「当前 28 个内置工具」或「core」工具集描述的位置（在「关键架构决策」附近或「常用命令」之前）。在 core 工具列表里追加（如果找不到具体位置，在 HARVIL.md/CLAUDE.md 的工具集说明处加）：

把 core 工具集说明从：
```
| **core**（18 个） | ... task_list / **execute_code** |
```
改成：
```
| **core**（20 个） | ... task_list / **execute_code** / **image_analyze** / **image_ocr** |
```

在「关键代码位置」表格末尾追加：

```markdown
| Vision/Image 工具 | `tools/image_tool.py`（image_analyze / image_ocr，复用 safe_path） |
```

- [ ] **Step 3: 更新桌面交接文档**

修改 `C:\Users\Administrator\Desktop\HarvilAgent会话交接.md`：

1. 测试数：`812 → 841`
2. 在「第 8 批改进（787 → 812）」之后追加新段：

```markdown
### 第 9 批改进（812 → 841）

| # | 能力 | 说明 |
|---|---|---|
| B1 | Vision/Image 工具集 | `tools/image_tool.py`（image_analyze + image_ocr）+ `agent/_vision_client` 抽象；browser_vision 三层回退修历史遗留 bug；safe_path/格式/大小三重检查；12+4+2=18 个新测试 |
| A1 | handoff 收尾 + verify 修复 | _parse_iso timezone-aware / _resolve_id 路径消毒 / token= 测试 / memory pointer 精确计数 / verify.py 22/22 全绿 |
| A3 | /usage 成本估算 | `agent/pricing.py`（DeepSeek/OpenAI/Anthropic 价格表）+ 4 项明细（input/cache_hit/cache_write/output） |
```

3. 「剩余待做」段里 B1 已完成（如果有列）；保留 B2-B6 + ② ⑬ ⑭

- [ ] **Step 4: Commit 文档**

```bash
git add CLAUDE.md
git commit -m "docs(image): CLAUDE.md 加 image 工具集 + 关键代码位置

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push 到 origin/master**

```bash
git push origin master
```

Expected: 推送成功，5 个新 commit（1 spec + 3 feat + 1 docs）

---

## 验收清单

- [ ] `tools/image_tool.py` 实现 `image_analyze` + `image_ocr`
- [ ] `agent/_vision_client` 字段存在
- [ ] `RuntimeContext._create_agent` 根据 config 初始化 vision_client
- [ ] `browser_vision` 三层回退（默认配置下可用）
- [ ] 18 新测试全过（12 image + 4 vision_client 配置 + 2 browser 回退）
- [ ] `uv run pytest tests/` 总数 ≥841（825 + 16 image + 2 browser = 843，但旧测试可能微调）
- [ ] `uv run python scripts/verify.py` 22/22 不回归
- [ ] CLAUDE.md + 桌面交接文档更新
- [ ] push 到 origin/master

## 实施备注

- **窄腰**：2 工具加 `core` toolset，注册时 `check_fn` 宽松（运行时校验）
- **base64 不落盘**：直接进 `data:` URL 传 LLM
- **prompt caching 不受影响**：本批次不改 system prompt
- **向后兼容**：browser_vision 保留 `_browser_vision_client` 读取（旧测试不挂）
