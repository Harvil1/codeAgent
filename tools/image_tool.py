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
# 注册
# ---------------------------------------------------------------------------

registry.register(
    name="image_analyze",
    toolset="core",
    schema=IMAGE_ANALYZE_SCHEMA,
    handler=_handle_image_analyze,
    emoji="🖼️",
    isConcurrencySafe=False,  # 外部调用：调 vision API（消耗配额 + 耗时），串行更稳
)

registry.register(
    name="image_ocr",
    toolset="core",
    schema=IMAGE_OCR_SCHEMA,
    handler=_handle_image_ocr,
    emoji="📝",
    isConcurrencySafe=False,  # 外部调用：调 OCR API（消耗配额 + 耗时），串行更稳
)
