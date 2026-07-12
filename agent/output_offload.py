"""大输出落盘：工具结果超过阈值时写到磁盘，messages 里只留预览。

这是分层压缩管线的旁路 L3（见 docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3）。

设计目标：
- 信息无损：完整内容在磁盘上，LLM 可通过 read_file 工具读回
- 文件名用 tool_call_id 保证唯一
- 失败降级：磁盘满时截断 content 并标注，不抛异常
"""
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 30000
DEFAULT_PREVIEW_CHARS = 2000


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
        agent_home: agent 根目录（如 ~/.agent）
        threshold: 触发落盘的字符数阈值
        preview_chars: 落盘后保留在 messages 里的预览长度
    """
    if not isinstance(content, str) or len(content) <= threshold:
        return content

    offload_dir = agent_home / ".task_outputs" / "tool-results"
    target_path = _resolve_unique_path(offload_dir, tool_call_id)

    try:
        _write_atomically(target_path, content)
    except OSError as e:
        logger.warning("offload 写入失败 (%s)，降级为截断: %s", target_path, e)
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
    """生成不冲突的落盘路径。tool_call_id 文件已存在时追加 _N。"""
    offload_dir.mkdir(parents=True, exist_ok=True)
    base = offload_dir / f"{tool_call_id}.txt"
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = offload_dir / f"{tool_call_id}_{counter}.txt"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_atomically(path: Path, content: str) -> None:
    """原子写入：先写临时文件再 replace，防半写状态。"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)  # 原子 rename
