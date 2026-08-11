"""prompt cache 检测系统（借鉴 claude-code-main promptCacheBreakDetection）。

工作流程：
1. pre-call：record_prompt_state 快照 prompt 维度（12 维 + per-tool hash）
2. post-call：check_cache_break 比较 cacheReadTokens + 找根因
3. break 时 log + 记入 _break_history + 写 diff 文件（供 /cache-stats 展示）

12 维度（对齐 claude-code-main promptCacheBreakDetection.ts）：
  1. system_hash       2. tools_hash         3. model
  4. cache_strategy    5. betas_hash         6. max_tokens
  7. temperature       8. stream_mode        9. tool_choice
  10. user_content_prefix  11. messages_count 12. system_boundary

扩展（CCAR4 Task A）：
  - per-tool hash（ToolHashEntry 列表，指出具体哪个工具变了）
  - diff 文件落盘（~/.OmniMate/.cache-breaks/cache-break-*.diff）
  - TTL 时长分析（5min / 1h 阈值）

不打断主流程（fail-open）。
"""
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolHashEntry:
    """单个工具的 hash 条目（per-tool hash 用）。

    name: 工具名（function.name）
    schema_hash: input_schema 的 hash（检测 schema 变化）
    """
    name: str = ""
    schema_hash: int = 0


@dataclass
class PromptState:
    """API 调用前的 prompt 状态快照（12 维度全实现 + per-tool hash）。

    对齐 claude-code-main promptCacheBreakDetection.ts。
    """
    # 5 核心 维度（原实现）
    system_hash: int = 0
    tools_hash: int = 0
    model: str = ""
    cache_strategy: str = ""
    betas_hash: int = 0
    # 新增 7 维度（CCAR4 Task A）
    max_tokens: int = 0
    temperature: Optional[float] = None
    stream_mode: bool = False
    tool_choice: Optional[str] = None  # "auto" / "none" / 指定工具名
    user_content_prefix: int = 0  # user 首条消息前 N 字符的 hash
    messages_count: int = 0
    system_boundary: str = ""  # "single" / "multi-block"
    system_len: int = 0  # system prompt 长度（只存长度不存原文，用于 delta 显示）
    # per-tool hash（工具列表每个工具单独 hash，精准定位哪个工具变了）
    tool_hashes: List[ToolHashEntry] = field(default_factory=list)


# 模块级状态（对齐 brief：单进程单会话，session 切换时 reset）
_last_state: Optional[PromptState] = None
_last_cache_read: Optional[int] = None
_pending_compaction: bool = False  # compact 后预期 cache 下降
_break_history: List[dict] = []  # break 事件累计（给 /cache-stats 展示）
_BREAK_HISTORY_LIMIT: int = 100  # 防长会话内存膨胀
_last_baseline_at: Optional[float] = None  # 上次 baseline 设置时间（TTL 分析用）
_diff_counter: int = 0  # diff 文件名递增计数器（防同秒覆盖）


def _hash_content(content: Any) -> int:
    """计算内容哈希（int，用于快速比较）。fail-open：异常返回 0。"""
    if isinstance(content, str):
        return hash(content)
    try:
        return hash(str(content))
    except Exception:
        return 0


def record_prompt_state(
    *,
    system_prompt: Any,
    tools: list,
    model: str,
    max_tokens: int = 0,
    temperature: Optional[float] = None,
    stream_mode: bool = False,
    tool_choice: Optional[str] = None,
    user_content_prefix: str = "",
    betas: Optional[dict] = None,
    **kwargs,
) -> PromptState:
    """API 调用前快照 prompt 状态（12 维度 + per-tool hash）。

    参数：
        system_prompt: system prompt（str 或 list of blocks）
        tools: 工具 schema 列表（OpenAI 格式）
        model: 模型名
        max_tokens: 最大输出 tokens
        temperature: 采样温度
        stream_mode: 是否流式调用
        tool_choice: tool_choice 参数
        user_content_prefix: user 消息前缀（捕捉 user 消息变化）
        betas: beta header 字典（如 anthropic_beta）
        **kwargs: 兼容扩展（cache_strategy / messages_count 等）

    fail-open：异常返回默认 PromptState（不抛）。
    """
    try:
        # per-tool hash 列表
        tool_hashes: List[ToolHashEntry] = []
        for t in (tools or []):
            fn = t.get("function", {}) if isinstance(t, dict) else getattr(t, "function", None)
            if not fn:
                continue
            if isinstance(fn, dict):
                name = fn.get("name", "")
                schema = fn.get("input_schema", {}) or fn.get("parameters", {})
            else:
                name = getattr(fn, "name", "")
                schema = getattr(fn, "input_schema", {}) or getattr(fn, "parameters", {})
            tool_hashes.append(ToolHashEntry(
                name=name or "",
                schema_hash=_hash_content(schema),
            ))

        # user_content_prefix: 前 500 字 hash
        ucp_str = (user_content_prefix or "")[:500]

        # system_len：只存长度（不存原文），用于 delta 显示
        if isinstance(system_prompt, list):
            _sys_len = sum(
                len(b.get("text", "")) if isinstance(b, dict) else len(str(b))
                for b in system_prompt
            )
        else:
            _sys_len = len(system_prompt or "")

        state = PromptState(
            system_hash=_hash_content(system_prompt),
            tools_hash=_hash_content([(t.name, t.schema_hash) for t in tool_hashes]),
            model=model or "",
            cache_strategy=kwargs.get("cache_strategy", ""),
            betas_hash=_hash_content(betas or {}),
            max_tokens=max_tokens or 0,
            temperature=temperature,
            stream_mode=stream_mode,
            tool_choice=tool_choice,
            user_content_prefix=_hash_content(ucp_str),
            messages_count=kwargs.get("messages_count", 0),
            system_boundary="multi-block" if isinstance(system_prompt, list) else "single",
            system_len=_sys_len,
            tool_hashes=tool_hashes,
        )
        return state
    except Exception as e:
        logger.debug("record_prompt_state fail-open: %s", e)
        return PromptState()


def check_cache_break(
    *,
    current_state: PromptState,
    cache_read_tokens: int,
    query_source: str = "",
) -> Optional[str]:
    """API 调用后检查 cache 是否 break（12 维根因 + per-tool diff + TTL 分析）。

    判定条件：cache read 下降 > 5% **且** > 2000 tokens。
    返回 None（没 break）或 break 根因描述字符串。
    fail-open：任何异常返回 None（绝不影响主流程）。
    """
    global _last_state, _last_cache_read, _pending_compaction, _last_baseline_at
    try:
        # 防御：cache_read_tokens 异常值
        if cache_read_tokens is None or cache_read_tokens < 0:
            cache_read_tokens = 0

        # compact 后预期下降，不算 break
        if _pending_compaction:
            _pending_compaction = False
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 首次调用（无 baseline）
        if _last_state is None or _last_cache_read is None:
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 判定 break：cache read 下降 > 5% 且 > 2000 tokens
        token_drop = _last_cache_read - cache_read_tokens
        if cache_read_tokens >= _last_cache_read * 0.95 or token_drop < 2000:
            # 不算 break，更新 baseline
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 真 break：12 维根因 + per-tool diff + TTL 分析
        reasons = _diagnose_break(current_state, _last_state, token_drop)

        # 写 diff 文件（如果 system 或 tools 变了）——独立 try/except，fail-open
        diff_path = None
        try:
            diff_path = _write_break_diff(_last_state, current_state, reasons)
        except Exception as de:
            logger.debug("_write_break_diff fail-open: %s", de)

        # 记录到历史（限长，防长会话内存膨胀）
        _break_history.append({
            "from": _last_cache_read,
            "to": cache_read_tokens,
            "drop": token_drop,
            "root_cause": "; ".join(reasons) if reasons else "未知原因",
            "query_source": query_source,
            "diff_path": diff_path,
        })
        if len(_break_history) > _BREAK_HISTORY_LIMIT:
            del _break_history[0:len(_break_history) - _BREAK_HISTORY_LIMIT]

        # LRU 清理 diff 文件（只在真写了 diff 时才清理，避免无谓 IO）
        if diff_path:
            _enforce_diff_lru_limit()

        # 更新 baseline
        prev_read = _last_cache_read
        _last_state = current_state
        _last_cache_read = cache_read_tokens
        _last_baseline_at = time.time()

        root_cause = "; ".join(reasons) if reasons else "未知原因"
        logger.warning(
            "prompt cache break! cache read %d -> %d (drop %d tokens). cause: %s%s",
            prev_read, cache_read_tokens, token_drop, root_cause,
            f" diff: {diff_path}" if diff_path else "",
        )
        return root_cause
    except Exception as e:
        logger.debug("check_cache_break fail-open: %s", e)
        return None


def _diagnose_break(current: PromptState, prev: PromptState, token_drop: int) -> list:
    """对比 12 维度 + per-tool hash + TTL 时长，返回根因列表。"""
    reasons = []

    # 12 维度对比
    if current.system_hash != prev.system_hash:
        delta = current.system_len - prev.system_len
        reasons.append(
            f"system prompt 变了 ({'+' if delta >= 0 else ''}{delta} chars)"
        )
    if current.tools_hash != prev.tools_hash:
        # 用 per-tool hash 找具体哪个工具变了
        tool_diff = _diff_tool_hashes(current.tool_hashes, prev.tool_hashes)
        reasons.append(f"工具 schema 变了 ({tool_diff})")
    if current.model != prev.model:
        reasons.append(f"model 变了 ({prev.model} → {current.model})")
    if current.max_tokens != prev.max_tokens:
        reasons.append(f"max_tokens 变了 ({prev.max_tokens} → {current.max_tokens})")
    if current.temperature != prev.temperature:
        reasons.append(f"temperature 变了 ({prev.temperature} → {current.temperature})")
    if current.stream_mode != prev.stream_mode:
        reasons.append(f"stream 模式变 ({prev.stream_mode} → {current.stream_mode})")
    if current.tool_choice != prev.tool_choice:
        reasons.append(f"tool_choice 变了 ({prev.tool_choice} → {current.tool_choice})")
    if current.user_content_prefix != prev.user_content_prefix:
        reasons.append("user content prefix 变了")
    if current.messages_count != prev.messages_count:
        reasons.append(
            f"messages count 变了 ({prev.messages_count} → {current.messages_count})"
        )
    if current.system_boundary != prev.system_boundary:
        reasons.append(
            f"system 边界变 ({prev.system_boundary} → {current.system_boundary})"
        )
    if current.betas_hash != prev.betas_hash:
        reasons.append("betas 变了")
    if current.cache_strategy != prev.cache_strategy:
        reasons.append(
            f"cache_strategy 变了 ({prev.cache_strategy} → {current.cache_strategy})"
        )

    # TTL 时长分析（5min / 1h 过期判定）
    global _last_baseline_at
    if not reasons:
        elapsed = time.time() - (_last_baseline_at or time.time())
        if elapsed > 3600:
            reasons.append(
                f"无字段变化但 break（>1h，可能 1h TTL 过期，elapsed={int(elapsed)}s）"
            )
        elif elapsed > 300:
            reasons.append(
                f"无字段变化但 break（>5min，可能 5min TTL 过期，elapsed={int(elapsed)}s）"
            )
        else:
            reasons.append(
                f"无字段变化（server-side 或未知，elapsed={int(elapsed)}s）"
            )

    return reasons


def _diff_tool_hashes(current: List[ToolHashEntry], prev: List[ToolHashEntry]) -> str:
    """对比 per-tool hash，返回 ±N 描述。"""
    cur_names = {t.name: t.schema_hash for t in (current or [])}
    prev_names = {t.name: t.schema_hash for t in (prev or [])}
    added = set(cur_names) - set(prev_names)
    removed = set(prev_names) - set(cur_names)
    changed = [n for n in cur_names if n in prev_names and cur_names[n] != prev_names[n]]
    parts = []
    if added:
        parts.append(f"+{len(added)} ({','.join(sorted(added)[:3])})")
    if removed:
        parts.append(f"-{len(removed)} ({','.join(sorted(removed)[:3])})")
    if changed:
        parts.append(f"~{len(changed)} ({','.join(sorted(changed)[:3])})")
    return " ".join(parts) if parts else "schema 全等但 tools_hash 变了"


def _write_break_diff(
    prev: PromptState, cur: PromptState, reasons: List[str],
) -> Optional[str]:
    """break 时写 unified diff 文件到 ~/.OmniMate/.cache-breaks/。fail-open。

    只在 system 或 tools 变了才写 diff（其他维度变化对 debug 帮助小）。
    """
    global _diff_counter
    try:
        if not reasons:
            return None
        # 只在 system 或 tools 变了才写 diff
        if not any("system" in r or "工具" in r for r in reasons):
            return None

        from constants import get_omnimate_home
        diff_dir = get_omnimate_home() / ".cache-breaks"
        diff_dir.mkdir(parents=True, exist_ok=True)

        _diff_counter += 1
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        diff_path = diff_dir / f"cache-break-{ts}-{_diff_counter:04d}.diff"

        # 简化版 diff（per-tool + system hash 对比）
        lines = [
            f"# Cache break at {ts}",
            f"# Reasons: {'; '.join(reasons)}",
            f"# Token drop context: see /cache-stats",
            "",
        ]
        if prev.system_hash != cur.system_hash:
            lines.append("## system prompt")
            lines.append(f"OLD hash: {prev.system_hash}")
            lines.append(f"NEW hash: {cur.system_hash}")
            lines.append(
                f"boundary: {prev.system_boundary} → {cur.system_boundary}"
            )
            lines.append("")
        if prev.tools_hash != cur.tools_hash:
            lines.append("## tools schema")
            lines.append(
                f"OLD tools: {[t.name for t in (prev.tool_hashes or [])]}"
            )
            lines.append(
                f"NEW tools: {[t.name for t in (cur.tool_hashes or [])]}"
            )
            # per-tool hash diff 明细
            tool_diff = _diff_tool_hashes(cur.tool_hashes, prev.tool_hashes)
            lines.append(f"diff: {tool_diff}")
            lines.append("")

        diff_path.write_text("\n".join(lines), encoding="utf-8")
        return str(diff_path)
    except Exception as e:
        logger.debug("write_break_diff fail-open: %s", e)
        return None


def _enforce_diff_lru_limit() -> None:
    """diff 文件 LRU 上限（默认 100 个，超过删最旧）。fail-open。"""
    try:
        from constants import get_omnimate_home
        # 读 config 的 max_cache_break_diff_files（默认 100）
        limit = 100  # 兜底；config 通过 _read_diff_limit() 读
        try:
            limit = _read_diff_limit()
        except Exception:
            pass

        diff_dir = get_omnimate_home() / ".cache-breaks"
        if not diff_dir.exists():
            return
        diff_files = list(diff_dir.glob("cache-break-*.diff"))
        if len(diff_files) <= limit:
            return
        # 按 mtime 排序，删最旧的
        diff_files.sort(key=lambda p: p.stat().st_mtime)
        to_delete = diff_files[:len(diff_files) - limit]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass
    except Exception as e:
        logger.debug("enforce_diff_lru_limit fail-open: %s", e)


# 模块级 diff 上限（可被 AIAgent.__init__ 覆盖）
# 放在 _read_diff_limit 之前（定义先于使用）
_diff_limit: int = 100


def _read_diff_limit() -> int:
    """从 config 读 max_cache_break_diff_files（默认 100）。"""
    try:
        # 不直接 import config（避免循环依赖），从环境查
        # 约定：DEFAULT_CONFIG["context"]["max_cache_break_diff_files"] = 100
        # 上层 AIAgent 会把 config 传下来，但 cache_monitor 不持有 config 实例
        # 所以用一个模块级变量 _diff_limit，由 AIAgent.__init__ 设置
        return _diff_limit
    except Exception:
        return 100


def set_diff_limit(limit: int) -> None:
    """设置 diff 文件 LRU 上限（由 AIAgent.__init__ 从 config 读后调）。"""
    global _diff_limit
    try:
        _diff_limit = max(1, int(limit))
    except Exception:
        pass


def notify_compaction() -> None:
    """标记：下次调用预期 cache 下降（compact / cache_edits 后调）。

    压缩会修改 messages，下次 cache read 必然下降——这不算 break，
    调本函数让 check_cache_break 跳过一次判定。
    """
    global _pending_compaction
    _pending_compaction = True


def get_stats() -> dict:
    """返回 cache 监控统计（给 /cache-stats 命令用）。"""
    return {
        "total_breaks": len(_break_history),
        "last_break": _break_history[-1] if _break_history else None,
        "last_cache_read": _last_cache_read,
    }


def reset_cache_monitor() -> None:
    """会话开始时重置（避免跨会话污染）。

    在 AIAgent.__init__ 末尾调用。
    """
    global _last_state, _last_cache_read, _pending_compaction, _break_history
    global _last_baseline_at
    _last_state = None
    _last_cache_read = None
    _pending_compaction = False
    _break_history = []
    _last_baseline_at = None
