"""工具大输出的"搬仓库"模块：输出超过阈值就整份写到磁盘，对话里只留预览。

背景：有的工具一口气吐几万字符（比如列出一整个目录），全塞进对话历史
会把上下文撑爆。这里是分层压缩管线的旁路 L3（设计文档见
docs/superpowers/specs/2026-07-12-claude-code-improvements-design.md §3）。

设计目标（三条底线）：
- 信息不丢：完整内容躺在磁盘上，模型需要时可用 read_file 工具读回来
- 文件名用 tool_call_id（每次工具调用的唯一编号），保证不重名
- 失败降级：磁盘写不进（比如满了）就截断内容并标注，绝不抛异常
"""
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 50000
DEFAULT_PREVIEW_CHARS = 2000


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """把 tool_call_id 清洗成能安全当文件名的字符串（除字母数字下划线
    连字符外全换成下划线，防止别有用心的人塞 ../ 之类的路径穿越）。

    参数：
        tool_call_id: 原始调用 ID

    返回：清洗后的安全文件名片段。
    """
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', tool_call_id)


def maybe_offload(
    content: str,
    *,
    tool_call_id: str,
    agent_home: Path,
    threshold: int = DEFAULT_THRESHOLD,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """工具 handler 调的收口：超过阈值就落盘，返回值直接当工具消息用。

    三种结果：
    - 内容不超过阈值：原样返回，啥也不做
    - 超过阈值：整份写入 agent_home/.task_outputs/tool-results/{tool_call_id}.txt，
      返回一段 JSON（含开头预览 + 指向完整文件的路径 full_at）
    - 写入失败（如磁盘满）：返回带 error_type=offload_io_error 和截断内容
      的 JSON——宁可降级也不能炸

    参数：
        content: 工具的原始输出（不是字符串就直接原样返回）
        tool_call_id: OpenAI 协议的工具调用 ID（每次唯一，兼做文件名）
        agent_home: agent 的根目录（如 ~/.OmniMate）
        threshold: 触发落盘的字符数门槛
        preview_chars: 落盘后留在对话里的预览长度

    返回：直接可用的工具消息 content（原文或 JSON 字符串）。
    """
    if not isinstance(content, str) or len(content) <= threshold:
        return content

    offload_dir = agent_home / ".task_outputs" / "tool-results"
    safe_id = _sanitize_tool_call_id(tool_call_id)

    target_path = None
    try:
        target_path = _resolve_unique_path(offload_dir, safe_id)
        # I3 修复留下的规矩：写文件前必须过 safe_path 路径安全检查
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
    """挑一个没被占用的落盘文件路径；同名文件已存在就加 _1、_2 编号。

    背景：正常情况下同一个调用 ID 不会落盘两次（有决策冻结机制兜着），
    这个分支几乎走不到；万一撞名就编号重试。上限试 1000 次——防止病态
    情况下无限循环，超了抛 OSError。

    参数：
        offload_dir: 落盘目录（不存在会自动建）
        tool_call_id: 清洗过的调用 ID（做文件名用）

    返回：可用的文件路径。
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
    """所有工具 handler 共用的收尾步骤：超阈值的大输出自动走落盘。

    背景：如果每个工具模块都自己写一遍"判断大小→写盘→留预览"就太啰嗦了，
    统一在返回前调这一个函数。

    规则：
    - tool_call_id 或 agent_home 缺了：原样返回（不落盘，比如测试场景）
    - 否则交给 maybe_offload 处理，阈值和预览长度从 config["context"] 读

    参数：
        result_content: 工具产出的结果文本
        tool_call_id: 本次调用的 ID
        agent_home: agent 根目录
        config: 配置字典（读落盘阈值用；空则用默认值）

    返回：可直接作为工具消息 content 的文本。
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
