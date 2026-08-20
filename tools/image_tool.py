"""图片理解工具（两个）：image_analyze 让"看得懂图的大模型"（vision LLM）
描述/解答本地图片；image_ocr 专门从图片里抠文字（OCR）。

安全规则（碰文件的工具都得过安检）：
- 路径要过 safe_path 的只读检查——受保护目录（~/.ssh、/etc 等）直接拒绝
- 文件后缀只认白名单：png/jpeg/jpg/webp/gif
- 文件大小最多 20MB（想改可配置 vision.max_bytes）
- 图片转成 base64 编码直接塞进 API 请求里，不在磁盘上留临时副本
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

# 支持的图片后缀 → 对应的网络传输格式名（MIME 类型）

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
    """看文件名后缀，查出它对应的 MIME 类型（如 .png → "image/png"）。

    背景：调 vision API 时要在请求里声明图片是什么格式，格式名就从后缀查表来。
    后缀统一转小写再查，所以 ".PNG" 和 ".png" 等价。

    参数：
        filename：文件名或完整路径。

    返回：MIME 类型字符串；后缀不在白名单里返回 None（表示不认识这个格式）。
    """
    suffix = Path(filename).suffix.lower()
    return SUPPORTED_FORMATS.get(suffix)


def _check_image_file(path_str: str, max_bytes: int) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """图片文件的四步预检：格式对不对、路径让不让读、文件在不在、是不是太大。

    背景：真正调 API 之前先把"明显不行"的情况挡掉，每一步失败都能给出
    具体原因，省得模型瞎猜为什么错。

    参数：
        path_str：图片路径字符串。
        max_bytes：允许的最大文件字节数。

    返回：三元组 (path对象, 错误类型, 错误信息)。
    - 成功：(Path, None, None)
    - 失败：(None, error_type, error_msg)——后两项填具体失败原因
    """
    # 1. 后缀要在白名单里
    mime = _infer_mime(path_str)
    if mime is None:
        suffix = Path(path_str).suffix.lower()
        return None, "unsupported_format", (
            f"不支持的图片格式: {suffix}（允许: {sorted(SUPPORTED_FORMATS.keys())}）"
        )

    # 2. 路径过 safe_path 安检（只读）：受保护目录（~/.ssh 等）拒绝
    perm = safe_path(path_str, write=False)
    if not perm.allowed:
        return None, "permission_denied", f"路径拒绝: {perm.reason}"

    p = Path(path_str)

    # 3. 文件得真的存在
    if not p.exists():
        return None, "file_not_found", f"文件不存在: {path_str}"

    # 4. 文件不能超过大小上限
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
    """从主对话对象（agent）身上找到能看图的模型客户端和它的模型名。

    背景：可以给视觉任务单独配一个模型（省得用昂贵的主模型），没配时
    就退回用主模型顶上。按优先级找：
    1. agent._vision_client——专门配的视觉模型客户端
    2. agent.llm_client——没有专配就借用主模型客户端

    参数：
        agent：主对话对象（AIAgent 实例）；传 None 表示没有（直接判不可用）。

    返回：(客户端, 模型名) 二元组。客户端为 None 表示两个都没找到、用不了。
    """
    if agent is None:
        return None, None

    client = getattr(agent, "_vision_client", None)
    if client is not None:
        # 模型名从客户端的 model 字段读（OpenAICompatClient 上的真实字段）
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
    """把"问题 + 图片"打包发给看得懂图的模型，拿回它对图片的描述/回答。

    背景：图片不能直接当文字发，要按 OpenAI 兼容接口的格式编码成
    data:image/png;base64,xxx 这样的内嵌地址放进消息里。

    参数：
        client：模型客户端（有 chat.completions.create 方法）。
        model：模型名。
        query：想问图片什么（如"描述这张图"）。
        image_b64：图片内容的 base64 编码字符串。
        mime：图片的 MIME 类型（如 image/png）。

    返回：模型回答的文本（description）。
    """
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
    """按项目统一格式拼一个错误 JSON 字符串。

    参数：
        error_type：错误类别代号（如 file_not_found、permission_denied）。
        msg：给人看的错误说明。
        image_path：出问题的图片路径（方便模型知道是哪张图错了）。

    返回：JSON 字符串，形如 {"error": ..., "error_type": ..., "image_path": ...}。
    """
    return json.dumps({
        "error": msg,
        "error_type": error_type,
        "image_path": image_path,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 工具的"说明书"（schema）——发给模型看的参数定义
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
# 处理函数（handlers）——真正干活的地方
# ---------------------------------------------------------------------------

def _handle_image_analyze(args: dict, **kwargs) -> str:
    """通用图片分析：把本地图片和问题一起交给看得懂图的模型，拿回回答。

    流程：校验参数 → 读大小上限配置 → 四步预检 → 找模型客户端 →
    图片编码成 base64 → 调模型 → 打包结果。

    参数：
        args：工具参数字典，来自模型——image_path（本地图片路径）、
            query（想问图片什么，如"图里有几只猫"）。
        **kwargs：分发器注入的运行上下文——重点用 agent_ref（主对话对象），
            从它身上取视觉模型客户端和 vision.max_bytes 配置。

    返回：JSON 字符串，成功含 description（模型的回答）；失败是
        {"error": ..., "error_type": ...} 格式。
    """
    image_path = (args.get("image_path") or "").strip()
    query = (args.get("query") or "").strip()

    if not image_path:
        return json.dumps({"error": "image_path 不能为空"}, ensure_ascii=False)
    if not query:
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    # 大小上限优先从 agent 的 config 里读（vision.max_bytes），没配就用默认 20MB
    agent = kwargs.get("agent_ref")
    max_bytes = DEFAULT_MAX_BYTES
    if agent is not None:
        max_bytes = (getattr(agent, "config", {}).get("vision", {}) or {}).get(
            "max_bytes", DEFAULT_MAX_BYTES
        )

    # 四步预检（格式/路径/存在/大小）
    path_obj, err_type, err_msg = _check_image_file(image_path, max_bytes)
    if err_type:
        return _build_error(err_type, err_msg, image_path)

    # 找能看图的模型客户端（专配的视觉模型，没有就退回主模型）
    client, model = _get_vision_client(agent)
    if client is None:
        return _build_error(
            "vision_unavailable",
            "vision LLM client 未配置（agent._vision_client 和 agent.llm_client 都为 None）",
            image_path,
        )

    # 把图片读出来编码成 base64 文字（API 只认这种形式）
    try:
        image_bytes = path_obj.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as e:
        return _build_error("vision_error", f"图片读取失败: {e}", image_path)

    mime = _infer_mime(image_path)

    # 发给模型拿回答
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
    """从图片里抠文字（OCR）——image_analyze 的"提取文字特化版"。

    背景：提取文字是个高频需求，单独做个工具免得模型每次自己想提示词；
    内部就是拼一条 OCR 专用提示词，然后复用 image_analyze 的整套
    检查和调用流程，不重复造轮子。

    参数：
        args：工具参数字典，来自模型——image_path（本地图片路径）、
            hint（可选提示，如"中文"/"Python 代码"/"表格"，帮模型提得更准）。
        **kwargs：原样透传给 image_analyze 的运行上下文。

    返回：JSON 字符串，成功时文字放在 text 字段（注意不叫 description）；
        失败时把 image_analyze 的错误原样转出来。
    """
    image_path = (args.get("image_path") or "").strip()
    hint = (args.get("hint") or "").strip()

    if not image_path:
        return json.dumps({"error": "image_path 不能为空"}, ensure_ascii=False)

    # 拼 OCR 专用提示词（让模型照着原图的排版和换行抄文字）
    prompt = "请提取图中所有可见文字，保持原始结构和换行。"
    if hint:
        prompt += f"\n上下文提示：{hint}"

    # 直接复用 image_analyze 的全部检查 + 调用
    analyze_result = _handle_image_analyze(
        {"image_path": image_path, "query": prompt},
        **kwargs,
    )
    data = json.loads(analyze_result)

    # 失败就原样把错误转出去
    if not data.get("success"):
        return analyze_result

    # 成功时换字段名：description 改叫 text（OCR 语境下更贴切）
    return json.dumps({
        "success": True,
        "text": data["description"],
        "image_path": image_path,
        "hint": hint or None,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册（这个文件一被 import 就自动登记进中央注册表）
# ---------------------------------------------------------------------------

registry.register(
    name="image_analyze",
    toolset="core",
    schema=IMAGE_ANALYZE_SCHEMA,
    handler=_handle_image_analyze,
    emoji="🖼️",
    isConcurrencySafe=False,  # 要调外部视觉模型接口（费配额、耗时长），串行更稳
)

registry.register(
    name="image_ocr",
    toolset="core",
    schema=IMAGE_OCR_SCHEMA,
    handler=_handle_image_ocr,
    emoji="📝",
    isConcurrencySafe=False,  # 要调外部模型接口（费配额、耗时长），串行更稳
)
