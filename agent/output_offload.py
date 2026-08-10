"""大输出落盘：工具结果超过阈值时写到磁盘，messages 里只留预览。

这是分层压缩管线的旁路 L3（见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3）。

设计目标：
- 信息无损：完整内容在磁盘上，LLM 可通过 read_file 工具读回
- 文件名用 tool_call_id 保证唯一
- 失败降级：磁盘满时截断 content 并标注，不抛异常
"""
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 50000
DEFAULT_PREVIEW_CHARS = 2000


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """清理 tool_call_id，只保留文件名安全字符（防路径穿越）。"""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', tool_call_id)


def maybe_offload(
    content: str,
    *,
    tool_call_id: str,
    agent_home: Path,
    threshold: int = DEFAULT_THRESHOLD,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """工具 handler 调用。返回值直接作为 tool 消息 content 用。

    - content 字符数 <= threshold：原样返回
    - content 字符数 > threshold：写入 agent_home/.task_outputs/tool-results/{tool_call_id}.txt，
      返回 JSON 字符串（含 preview + full_at 指针）
    - 写入失败（如磁盘满）：返回 JSON，含 error_type=offload_io_error + truncated_content

    参数：
        content: 工具原始输出（非字符串时原样返回）
        tool_call_id: OpenAI 兼容协议的工具调用 ID（每次唯一）
        agent_home: agent 根目录（如 ~/.OmniMate）
        threshold: 触发落盘的字符数阈值
        preview_chars: 落盘后保留在 messages 里的预览长度
    """
    if not isinstance(content, str) or len(content) <= threshold:
        return content

    offload_dir = agent_home / ".task_outputs" / "tool-results"
    safe_id = _sanitize_tool_call_id(tool_call_id)

    target_path = None
    try:
        target_path = _resolve_unique_path(offload_dir, safe_id)
        # I3: 写入前过 safe_path 权限检查
        from agent.permission import safe_path
        perm = safe_path(target_path, write=True, allowed_roots=[offload_dir.resolve()])
        if not perm.allowed:
            raise OSError(f"safe_path 拒绝: {perm.reason}")
        from agent.atomic_io import atomic_write_text_lite
        atomic_write_text_lite(target_path, content)
    except OSError as e:
        target_desc = str(target_path) if target_path else str(offload_dir / safe_id)
        logger.warning("offload 写入失败 (%s)，降级为截断: %s", target_desc, e)
        return json.dumps({
            "error": f"offload failed: {e}",
            "error_type": "offload_io_error",
            "truncated_content": content[:threshold],
        }, ensure_ascii=False)

    return json.dumps({
        "truncated": True,
        "orig_chars": len(content),
        "preview": content[:preview_chars],
        "full_at": str(target_path),
        "hint": "完整结果已落盘，需要时调 read_file 读取 full_at",
    }, ensure_ascii=False)


def _resolve_unique_path(offload_dir: Path, tool_call_id: str) -> Path:
    """生成不冲突的落盘路径。tool_call_id 文件已存在时追加 _N。

    上限 1000 次尝试（防病理情况 O(n²) 循环）——超出时 raise OSError。
    正常场景下决策冻结机制保证同 tool_call_id 不重复 offload，这个分支几乎到不了。
    """
    offload_dir.mkdir(parents=True, exist_ok=True)
    base = offload_dir / f"{tool_call_id}.txt"
    if not base.exists():
        return base
    for counter in range(1, 1001):
        candidate = offload_dir / f"{tool_call_id}_{counter}.txt"
        if not candidate.exists():
            return candidate
    raise OSError(
        f"offload 路径冲突：{tool_call_id} 已有 1000+ 同名文件"
    )


def finalize_tool_output(
    result_content: str,
    tool_call_id,
    agent_home,
    config: dict | None = None,
) -> str:
    """工具 handler 通用收尾：超阈值内容走 offload（落盘 + 预览）。

    - 若 ``tool_call_id`` 或 ``agent_home`` 缺失：原样返回（不 offload）
    - 否则委托给 :func:`maybe_offload`，阈值/预览长度从 ``config["context"]`` 读取

    所有工具 handler 在返回前调这个函数即可，避免每个 tool 模块各写一份。
    """
    if not tool_call_id or not agent_home:
        return result_content
    cfg = (config or {}).get("context", {})
    return maybe_offload(
        result_content,
        tool_call_id=tool_call_id,
        agent_home=Path(agent_home),
        threshold=cfg.get("output_offload_threshold", DEFAULT_THRESHOLD),
        preview_chars=cfg.get("output_offload_preview", DEFAULT_PREVIEW_CHARS),
    )
