"""工具大输出的"搬仓库"模块：输出超过阈值就整份写到磁盘，对话里只留预览。

有的工具一口气吐几万字符（比如列出一整个目录），全塞进对话历史会把
上下文撑爆。本模块是分层压缩管线的旁路 L3。

设计目标（三条底线）：
- 信息不丢：完整内容躺在磁盘上，模型需要时可用 read_file 工具读回来
- 文件名用 tool_call_id（每次工具调用的唯一编号），保证不重名
- 失败降级：磁盘写不进（比如满了）就截断内容并标注，绝不抛异常
"""
import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 50000
DEFAULT_PREVIEW_CHARS = 2000
DEFAULT_TAIL_CHARS = 1000
DEFAULT_TOOL_OUTPUT_RETENTION_DAYS = 14


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
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> str:
    """工具 handler 调的收口：超过阈值就落盘，返回值直接当工具消息用。

    三种结果：
    - 内容不超过阈值：原样返回，啥也不做
    - 超过阈值：整份写入 agent_home/.task_outputs/tool-results/{tool_call_id}.txt，
      返回一段 JSON（含开头预览 + 结尾预览 + 指向完整文件的路径 full_at）
    - 写入失败（如磁盘满）：返回带 error_type=offload_io_error 和截断内容
      的 JSON——宁可降级也不能炸

    参数：
        content: 工具的原始输出（不是字符串就直接原样返回）
        tool_call_id: OpenAI 协议的工具调用 ID（每次唯一，兼做文件名）
        agent_home: agent 的根目录（如 ~/.OmniMate）
        threshold: 触发落盘的字符数门槛
        preview_chars: 落盘后留在对话里的开头预览长度
        tail_chars: 结尾预览长度（日志/测试输出关键信息常在结尾——
                    头尾都给，模型多数场景就不用读回了；0 = 不带 tail）

    返回：直接可用的工具消息 content（原文或 JSON 字符串）。
    """
    if not isinstance(content, str) or len(content) <= threshold:
        return content

    offload_dir = agent_home / ".task_outputs" / "tool-results"
    safe_id = _sanitize_tool_call_id(tool_call_id)

    target_path = None
    try:
        target_path = _resolve_unique_path(offload_dir, safe_id)
        # 写文件前必须过 safe_path 路径安全检查
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

    payload = {
        "truncated": True,
        "orig_chars": len(content),
        "preview": content[:preview_chars],
        "full_at": str(target_path),
        "hint": (
            "完整结果已落盘；preview 是开头、tail 是结尾，"
            "需要中间内容时调 read_file 读取 full_at（大文件用 offset/limit 分段）"
        ),
    }
    # 结尾预览：内容短到和开头预览重叠就不带（重复送没意义）
    if tail_chars > 0 and len(content) > preview_chars + tail_chars:
        payload["tail"] = content[-tail_chars:]
    return json.dumps(payload, ensure_ascii=False)


def _resolve_unique_path(offload_dir: Path, tool_call_id: str) -> Path:
    """挑一个没被占用的落盘文件路径；同名文件已存在就加 _1、_2 编号。

    同一个调用 ID 正常不会落盘两次（有决策冻结机制兜着）；万一撞名就
    编号重试。上限试 1000 次——防病态情况下无限循环，超了抛 OSError。

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

    统一收口，免得每个工具模块都自己写一遍"判断大小→写盘→留预览"。

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
        tail_chars=cfg.get("output_offload_tail", DEFAULT_TAIL_CHARS),
    )


def cleanup_old_tool_outputs(
    agent_home=None,
    retention_days: int = DEFAULT_TOOL_OUTPUT_RETENTION_DAYS,
) -> int:
    """清理「超过 retention_days 天没动过」的落盘工具结果文件。返回清了几个。

    .task_outputs/tool-results/ 只进不出（大结果落盘 + delegation
    全文回流），长期使用磁盘无限涨——按 mtime 清理（占位引用的
    preview 还在会话库里，删的是磁盘底稿；真要细节模型可以重跑工具）。
    0 = 关闭。fail-open 全吞。

    参数：
        agent_home：agent 根目录，不传用默认 ~/.OmniMate
        retention_days：保留天数（默认 14）
    返回：删掉的文件数；任何异常返回 0。
    """
    try:
        if retention_days <= 0:
            return 0
        if agent_home is None:
            try:
                from constants import get_omnimate_home
                agent_home = get_omnimate_home()
            except Exception:
                return 0
        root = Path(agent_home) / ".task_outputs" / "tool-results"
        if not root.exists():
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for f in root.glob("*.txt"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except Exception:
                continue
        if removed:
            logger.info("tool-results 清理 %d 个过期落盘文件", removed)
        return removed
    except Exception:
        return 0
